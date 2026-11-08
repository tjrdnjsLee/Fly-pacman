"""진짜 엔진의 단계 2–5. 고정 장면에서 이동과 사건을 검사한다."""

import numpy as np
import pytest

from flypac.core import directions as d
from flypac.core.config import Config, load_config
from flypac.core.rng import make_rng
from flypac.engine.game import Game
from helpers import MAP, cell
from test_engine_real import SMOKE


def new_game(**engine):
    cfg = load_config(SMOKE).to_dict()
    # 연기 시험 전용 상한 대신 각 단위 시험에서 준 속도를 사용한다.
    cfg['engine'].pop('pacman_normal_speed_cap', None)
    cfg['engine'].pop('fly_normal_speed_cap', None)
    cfg['engine'].update(engine)
    # 이동만 보는 시험은 점 감속을 끈다. 감속 시험은 별도 값을 명시한다.
    cfg['engine']['pacman_dot_speed'] = engine.get('pacman_dot_speed', cfg['engine']['pacman_speed'])
    game = Game(MAP, Config(cfg), make_rng(9, 0, 'engine'))
    game.reset_game(make_rng(9, 1, 'engine'))
    # 다른 곳에 점을 남겨 이동 시험 도중 클리어를 막는다.
    game.dots[:] = False
    game.energizers[:] = False
    game.dots[cell(29, 1)] = True
    game.pac_cell = cell(1, 1)
    game.pac_dir = d.EAST
    return game


@pytest.mark.parametrize('speed', [0.0, 0.4, 0.8, 1.0])
def test_pacman_average_speed_through_step(speed):
    game = new_game(pacman_speed=speed)
    moves = 0
    for _ in range(1000):
        before = game.pac_cell
        direction = d.EAST if before == cell(1, 1) else d.WEST
        game.step_pacman(direction)
        moves += game.pac_cell != before
    assert abs(moves - 1000 * speed) <= 1


def test_pacman_can_reverse_and_wrap_tunnel():
    game = new_game(pacman_speed=1)
    game.pac_cell, game.pac_dir = cell(14, 0), d.EAST
    game.step_pacman(d.WEST)
    assert game.pac_cell == cell(14, 27) and game.pac_dir == d.WEST
    game.step_pacman(d.EAST)
    assert game.pac_cell == cell(14, 0)


def test_buffered_turn_is_applied_when_open():
    game = new_game(pacman_speed=1)
    game.pac_cell = cell(1, 5)
    game.step_pacman(d.SOUTH)  # 여기서는 막혔지만 다음 갈림길에서는 열린다.
    assert game.pac_cell == cell(1, 6) and game.pac_dir == d.EAST
    game.step_pacman(-1)  # 새 입력 없이 이전 입력 유지
    assert game.pac_cell == cell(2, 6) and game.pac_dir == d.SOUTH


def test_wall_and_house_door_block_pacman():
    game = new_game(pacman_speed=1)
    game.pac_dir = d.NORTH
    game.step_pacman(d.NORTH)
    assert game.pac_cell == cell(1, 1)
    game.pac_cell, game.pac_dir = cell(11, 13), d.SOUTH
    game.step_pacman(d.SOUTH)
    assert game.pac_cell == cell(11, 13)


def test_frozen_pacman_keeps_position_direction_and_accumulator():
    game = new_game(pacman_speed=1)
    game.frozen = True
    before = (game.pac_cell, game.pac_dir, game.pac_acc)
    assert game.step_pacman(d.SOUTH) == []
    assert (game.pac_cell, game.pac_dir, game.pac_acc) == before


def test_dot_event_score_and_single_consumption():
    from flypac.core.events import Kind
    game = new_game(pacman_speed=1, pacman_dot_speed=1)
    target = cell(1, 2)
    game.dots[target] = True
    game.index, game.tick, game.learning_active = 2, 7, True
    events = game.step_pacman(d.EAST)
    assert [event.kind for event in events] == [Kind.DOT]
    assert (events[0].game, events[0].tick, events[0].active) == (2, 7, True)
    assert game.score == 10 and not game.dots[target]
    game.step_pacman(d.WEST)
    assert game.step_pacman(d.EAST) == [] and game.score == 10


def test_energizer_score_without_three_frame_pause():
    from flypac.core.events import Kind
    game = new_game(pacman_speed=1, pacman_power_speed=1)
    game.energizers[cell(1, 2)] = True
    events = game.step_pacman(d.EAST)
    assert [event.kind for event in events] == [Kind.ENERGIZER, Kind.POWER_START] + [Kind.Z_CHANGE] * 4
    assert game.score == 50 and not game.energizers[cell(1, 2)]
    assert game.pac_acc == 0
    game.step_pacman(d.EAST)
    assert game.pac_cell == cell(1, 3)


def test_pellet_slowdown_through_actual_movement():
    game = new_game(pacman_speed=0.8, pacman_dot_speed=0.71)
    moves = 0
    for _ in range(10000):
        before = game.pac_cell
        target = cell(1, 2) if before == cell(1, 1) else cell(1, 1)
        game.dots[target] = True
        game.step_pacman(d.EAST if before == cell(1, 1) else d.WEST)
        moves += game.pac_cell != before
    assert abs(moves / 10000 - 0.71) < 0.0002
    assert game.score == moves * 10


def test_no_pickup_when_accumulator_is_not_ready():
    game = new_game(pacman_speed=0.4, pacman_dot_speed=0.4)
    game.dots[cell(1, 2)] = True
    assert game.step_pacman(d.EAST) == []
    assert game.score == 0 and game.dots[cell(1, 2)]


@pytest.mark.parametrize('lives', [1, 2, 3])
def test_catch_freezes_without_automatic_reset(lives):
    from flypac.core.events import Kind, Winner
    game = new_game(pacman_speed=1)
    game.lives = lives
    game.learning_active = True
    game.fly_cell[0] = cell(1, 2)
    events = game.step_pacman(d.EAST)
    expected = [Kind.CATCH, Kind.LIFE_LOST]
    if lives == 1:
        expected.append(Kind.GAME_OVER)
        assert events[-1].winner is Winner.FLIES
    assert [event.kind for event in events] == expected
    assert all(event.active for event in events)
    assert game.lives == lives - 1 and game.frozen and game.learning_active
    assert game.pac_cell == cell(1, 2)  # 포획 위치를 실행기가 기록한 뒤 초기화한다.
    before = game.fly_cell.copy()
    ev, moved, committed, _ = game.step_flies(np.zeros(4, dtype=int))
    assert ev == [] and not moved.any() and not committed.any()
    assert np.array_equal(game.fly_cell, before)
    assert game.step_pacman(d.WEST) == []  # 종료 보상을 중복 생성하지 않는다.


@pytest.mark.parametrize('energizer', [False, True])
def test_last_pickup_wins_before_collision(energizer):
    from flypac.core.events import Kind, Winner
    game = new_game(pacman_speed=1)
    game.dots[:] = False
    target = cell(1, 2)
    (game.energizers if energizer else game.dots)[target] = True
    game.fly_cell[0] = target
    events = game.step_pacman(d.EAST)
    pickups = [Kind.ENERGIZER, Kind.POWER_START] + [Kind.Z_CHANGE] * 4 if energizer else [Kind.DOT]
    assert [event.kind for event in events] == pickups + [Kind.GAME_OVER]
    assert events[-1].winner is Winner.PACMAN and game.lives == 3 and game.frozen


def test_swap_is_caught_in_pacman_phase():
    from flypac.core.events import Kind
    game = new_game(pacman_speed=1)
    game.fly_cell[0], game.fly_dir[0] = cell(1, 2), d.WEST
    events = game.step_pacman(d.EAST)
    assert events[0].kind is Kind.CATCH
    game.step_flies(np.full(4, d.FORWARD))
    assert game.fly_cell[0] == cell(1, 2)  # 서로 마주 오는 두 캐릭터가 통과하지 않는다.


def test_life_reset_preserves_progress_rng_and_game_tick():
    game = new_game(pacman_speed=1)
    game.tick = 9
    game.score = 123
    game.fly_cell[0] = cell(1, 2)
    game.step_pacman(d.EAST)
    dots = game.dots.copy()
    rng = game.rng
    state = rng.bit_generator.state
    game.reset_life()
    assert game.rng is rng and game.rng.bit_generator.state == state
    assert game.tick == 9 and game.score == 123 and game.lives == 2
    assert np.array_equal(game.dots, dots) and not game.frozen


def test_frozen_tick_is_counted_without_advancing_life_timer():
    game = new_game()
    game.tick, game.life_tick, game.frozen = 9, 4, True
    assert game.tick_timers() == []
    assert game.tick == 10 and game.life_tick == 4


def test_watchdog_ends_game_once_with_censored_event():
    from flypac.core.events import Kind, Winner
    game = new_game(watchdog_ticks=2)
    assert game.tick_timers() == []
    events = game.tick_timers()
    assert len(events) == 1 and events[0].kind is Kind.GAME_OVER
    assert events[0].winner is Winner.CENSORED and events[0].tick == 1
    assert game.tick == 2 and game.lives == 3 and game.frozen
    assert game.tick_timers() == [] and game.tick == 3
