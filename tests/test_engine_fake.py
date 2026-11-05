import numpy as np
import pytest
from helpers import MAP, cell, make_cfg, place

from flypac.core import directions as d
from flypac.core.events import Kind, Winner
from flypac.engine.fake import FakeGame
from flypac.engine.game import VecGame
from flypac.core.rng import make_rng


def new_game(**over):
    g = FakeGame(MAP, make_cfg(**over), make_rng(0, 0, "engine"))
    return g


def kinds(events):
    return [e.kind for e in events]


def test_speed_accumulator_halves_movement():
    g = new_game(engine={"pacman_speed": 0.5})
    start = g.pac_cell
    moved = 0
    for _ in range(10):
        before = g.pac_cell
        g.step_pacman(d.WEST)
        moved += g.pac_cell != before
    assert moved == 5 and g.pac_cell != start


def test_pacman_eats_dot_and_score():
    g = new_game()
    place(g, pac=cell(5, 3))
    g.dots[:] = False
    g.dots[cell(5, 4)] = True
    g.dots[cell(5, 20)] = True
    ev = g.step_pacman(d.EAST)
    assert kinds(ev) == [Kind.DOT] and g.score == 10 and not g.dots[cell(5, 4)]


def test_pacman_blocked_by_wall_keeps_direction_or_stays():
    g = new_game()
    place(g, pac=cell(5, 10), pac_dir=d.EAST)
    g.step_pacman(d.NORTH)  # 북쪽은 벽: 가던 동쪽으로
    assert g.pac_cell == cell(5, 11)


def test_pacman_never_enters_house():
    g = new_game()
    door_above = int(MAP.nbr[MAP.door_cells[0], d.NORTH])
    place(g, pac=door_above, pac_dir=d.SOUTH)  # 가던 방향도 집 쪽이라 막힌다
    g.step_pacman(d.SOUTH)
    assert g.pac_cell == door_above  # 막혀서 제자리


def test_release_order_and_all_released_once():
    g = new_game(engine={"pacman_speed": 0.0, "release_ticks": [0, 2, 4, 6]})
    allev = []
    for _ in range(8):
        ev1, ex, ac = g.step_flies(np.full(4, -1))
        ev = ev1 + g.tick_timers()
        allev += ev
        if g.frozen:
            break
    rel = [e for e in allev if e.kind is Kind.ALL_RELEASED]
    assert len(rel) == 1 and rel[0].fly == 3 and rel[0].active is False
    assert g.learning_active and g.released.all()
    assert g.fly_cell[3] == MAP.fly_starts[0] or not g.view().fly_in_house[3]


def test_events_before_all_released_are_inactive_and_after_active():
    g = new_game(engine={"release_ticks": [0, 0, 0, 0]})
    place(g, pac=cell(11, 12), released=[True, True, True, False], flies={0: cell(11, 13)})
    ev = g.step_pacman(d.EAST)  # 0번을 잡음, 아직 4번이 안 나옴
    assert [e.active for e in ev] == [False, False]
    g2 = new_game()
    place(g2, pac=cell(11, 12), released=True, active=True, flies={0: cell(11, 13)})
    assert all(e.active for e in g2.step_pacman(d.EAST))


def test_catch_in_pacman_phase_freezes_and_flies_do_not_move():
    g = new_game()
    place(g, pac=cell(11, 12), released=True, active=True, flies={0: cell(11, 13)})
    ev = g.step_pacman(d.EAST)
    assert kinds(ev) == [Kind.CATCH, Kind.LIFE_LOST] and ev[0].fly == 0 and g.frozen
    before = g.fly_cell.copy()
    ev2, executed, active = g.step_flies(np.full(4, -1))
    assert ev2 == [] and not executed.any() and (g.fly_cell == before).all()
    assert g.tick_timers() == []


def test_second_fly_catch_stops_remaining_flies():
    g = new_game()
    place(g, pac=cell(5, 10), released=True, active=True,
          flies={0: cell(5, 1), 1: cell(5, 9), 2: cell(5, 20), 3: cell(5, 22)},
          fly_dirs={0: d.EAST, 1: d.EAST, 2: d.WEST, 3: d.WEST})
    g.pac_acc = -10.0  # 이번 틱 팩맨 정지
    ev, executed, active = g.step_flies(np.array([-1, d.FORWARD, -1, -1]))
    assert kinds(ev) == [Kind.CATCH, Kind.LIFE_LOST] and ev[0].fly == 1
    assert executed.tolist() == [True, True, False, False]
    assert g.fly_cell[2] == cell(5, 20) and g.fly_cell[3] == cell(5, 22)


def test_last_life_events_and_dots_win_events():
    g = new_game()
    g.lives = 1
    place(g, pac=cell(11, 12), released=True, active=True, flies={0: cell(11, 13)})
    ev = g.step_pacman(d.EAST)
    assert kinds(ev) == [Kind.CATCH, Kind.LIFE_LOST, Kind.GAME_OVER]
    assert ev[2].winner is Winner.FLIES
    g = new_game()
    place(g, pac=cell(5, 3))
    g.dots[:] = False
    g.energizers[:] = False
    g.dots[cell(5, 4)] = True
    ev = g.step_pacman(d.EAST)
    assert kinds(ev) == [Kind.DOT, Kind.GAME_OVER] and ev[1].winner is Winner.PACMAN  # LIFE_LOST 없음


def test_last_dot_and_collision_same_tick_pacman_wins():
    g = new_game()
    place(g, pac=cell(5, 3), released=True, active=True, flies={0: cell(5, 4)})
    g.dots[:] = False
    g.energizers[:] = False
    g.dots[cell(5, 4)] = True
    ev = g.step_pacman(d.EAST)
    assert kinds(ev) == [Kind.DOT, Kind.GAME_OVER] and g.lives == 3


def test_watchdog_censors():
    g = new_game(engine={"watchdog_ticks": 3})
    out = []
    for _ in range(3):
        out += g.tick_timers()
    assert kinds(out) == [Kind.GAME_OVER] and out[0].winner is Winner.CENSORED and g.frozen


def test_decision_info_corridor_vs_junction():
    g = new_game()
    place(g, released=True, flies={0: cell(5, 10), 1: cell(5, 6)}, fly_dirs={0: d.EAST, 1: d.EAST})
    legal, decide = g.decision_info()
    assert not decide[0] and legal[0].tolist() == [False, True, False, False]  # 통로: 직진만
    assert decide[1] and legal[1][d.LEFT] and legal[1][d.RIGHT] and legal[1][d.FORWARD]
    assert not legal[1][d.BACK]
    assert not legal[2].any() and not decide[2]  # 아직 집 안


def test_illegal_action_raises():
    g = new_game()
    place(g, released=True, flies={0: cell(5, 10)}, fly_dirs={0: d.EAST})
    with pytest.raises(ValueError):
        g.step_flies(np.array([d.LEFT, -1, -1, -1]))  # 복도 북쪽은 벽


def test_reset_is_not_automatic_and_keeps_dots_across_lives():
    g = new_game()
    place(g, pac=cell(5, 3))
    g.step_pacman(d.EAST)
    left = g.dots.sum()
    g.reset_life()
    assert g.dots.sum() == left and g.pac_cell == MAP.pacman_start and not g.learning_active
    g.reset_game()
    assert g.dots.sum() == MAP.dots.sum() and g.lives == 3 and g.score == 0


def test_tunnel_wraps_for_pacman():
    g = new_game()
    place(g, pac=cell(14, 0), pac_dir=d.WEST)
    g.step_pacman(d.WEST)
    assert g.pac_cell == cell(14, 27)


def test_event_game_index_from_vecgame():
    class V(VecGame):
        game_cls = FakeGame

    v = V(2, MAP, make_cfg(), [make_rng(0, i, "engine") for i in range(2)])
    place(v.games[1], pac=cell(11, 12), released=True, active=True, flies={0: cell(11, 13)})
    ev = v.step_pacman(np.array([d.EAST, d.EAST]))
    assert ev[0] == [] or all(e.game == 0 for e in ev[0])
    assert all(e.game == 1 for e in ev[1]) and ev[1]
