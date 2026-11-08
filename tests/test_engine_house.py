"""유령 집 카운터와 출구 통과 사건 (TRD 4.5, 7.3)."""

import numpy as np

from flypac.core import directions as d
from flypac.core.events import Kind
from helpers import cell
from test_engine_movement import new_game


def house_game(**overrides):
    return new_game(**({
        'house_personal_dots': [0, 0, 30, 60],
        'house_global_dots': [7, 17, 32],
        'house_no_dot_s': 4,
        'fly_speed': 1,
        'pacman_speed': 1,
    } | overrides))


def fly_step(game):
    return game.step_flies(np.full(4, -1))


def eat_dots(game, count):
    for _ in range(count):
        game.pac_cell = cell(1, 1)
        game.dots[cell(1, 2)] = True
        assert [event.kind for event in game.step_pacman(d.EAST)] == [Kind.DOT]


def test_initial_exit_traverses_door_without_commit():
    game = house_game()
    for target in [(13, 13), (12, 13), (11, 13)]:
        assert not game.decision_info()[1][1]
        events, moved, committed, active = fly_step(game)
        assert game.fly_cell[1] == cell(*target)
        assert moved[1] and not committed[1] and not active[1]
        assert not events
        assert bool(game.view().fly_in_house[1]) == (target != (11, 13))
    assert game.released.tolist() == [True, True, False, False]


def test_personal_counters_release_in_order_and_activate_at_last_crossing():
    game = house_game()
    for _ in range(3):
        fly_step(game)
    eat_dots(game, 29)
    assert not fly_step(game)[1][2]
    eat_dots(game, 1)
    for _ in range(5):
        assert fly_step(game)[1][2]
    assert game.released.tolist() == [True, True, True, False]

    eat_dots(game, 59)
    assert not fly_step(game)[1][3]
    eat_dots(game, 1)
    for _ in range(4):
        events, moved, committed, active = fly_step(game)
        assert moved[3] and not committed[3] and not active[3]
        assert not events and not game.learning_active
    events, moved, committed, active = fly_step(game)
    assert moved[3] and not committed[3] and not active[3]
    assert [(e.kind, e.fly, e.active) for e in events] == [(Kind.ALL_RELEASED, 3, False)]
    assert game.learning_active and not game.view().fly_in_house.any()
    assert all(e.kind is not Kind.ALL_RELEASED for e in fly_step(game)[0])


def test_life_loss_uses_global_count_from_zero():
    game = house_game()
    eat_dots(game, 10)
    game.lives -= 1
    game.reset_life()
    assert not game.learning_active and not fly_step(game)[1][1]
    for index, additional_dots in [(1, 7), (2, 10), (3, 15)]:
        eat_dots(game, additional_dots - 1)
        assert not fly_step(game)[1][index]
        eat_dots(game, 1)
        assert fly_step(game)[1][index]
        for _ in range(4):
            fly_step(game)
        assert game.released[index]


def test_no_dot_timer_restarts_after_pickup_and_forces_one_exit():
    game = house_game(house_personal_dots=[0, 100, 100, 100])
    for _ in range(37):
        game.tick_timers()
    assert not fly_step(game)[1][1:].any()
    eat_dots(game, 1)
    game.tick_timers()  # 먹은 틱은 무점 시간에 포함하지 않는다.
    for _ in range(37):
        game.tick_timers()
    assert not fly_step(game)[1][1:].any()
    game.tick_timers()
    assert fly_step(game)[1][1:].tolist() == [True, False, False]
    for _ in range(3):
        fly_step(game)
    assert game.released.tolist() == [True, True, False, False]


def test_last_release_activates_collision_on_exit_tile():
    game = house_game(house_personal_dots=[0, 0, 0, 0])
    game.fly_cell[3] = cell(12, 13)
    game.fly_cell[:3] = cell(1, 1)
    game.released[:3] = True
    game.pac_cell = cell(11, 13)
    events, moved, committed, active = fly_step(game)
    assert moved[3] and not committed[3] and not active[3]
    assert [(e.kind, e.active) for e in events] == [
        (Kind.ALL_RELEASED, False), (Kind.CATCH, True), (Kind.LIFE_LOST, True),
    ]


def test_dot_on_first_tick_counts_for_inky_before_pinky_crosses():
    game = house_game()
    eat_dots(game, 1)
    for _ in range(3):
        fly_step(game)
    assert game.released[1]
    eat_dots(game, 28)
    assert not fly_step(game)[1][2]
    eat_dots(game, 1)
    assert fly_step(game)[1][2]
