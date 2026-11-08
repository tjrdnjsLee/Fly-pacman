"""파워 쿠키, 먹힘, 부활과 PPO 보상 순번 (TRD 4.5, 7.3)."""

import numpy as np

from flypac.core import directions as d
from flypac.core.events import Kind, ZCause
from helpers import cell
from test_engine_movement import new_game


def power_game(**overrides):
    return new_game(**({
        'pacman_power_speed': 1,
        'pacman_power_dot_speed': 1,
        'fly_frightened_speed': .5,
        'score_ghosts': [200, 400, 800, 1600],
    } | overrides))


def energize(game):
    game.pac_cell = cell(1, 1)
    game.energizers[cell(1, 2)] = True
    return game.step_pacman(d.EAST)


def test_energizer_changes_house_flies_and_turns_expire_after_57_ticks():
    game = power_game()
    events = energize(game)
    assert [e.kind for e in events[:2]] == [Kind.ENERGIZER, Kind.POWER_START]
    assert [(e.fly, e.cause) for e in events[2:]] == [
        (k, ZCause.POWER_START) for k in range(4)
    ]
    assert game.view().fly_z.all() and game.view().fly_in_house[1:].all()
    assert game.view().power_ticks == 57
    for _ in range(56):
        assert not game.tick_timers()
    assert game.view().power and game.view().power_ticks == 1
    events = game.tick_timers()
    assert [e.kind for e in events] == [Kind.POWER_END] + [Kind.Z_CHANGE] * 4
    assert all(e.cause == ZCause.POWER_END for e in events[1:])
    assert not game.view().power and not game.view().fly_z.any()


def test_pickup_before_collision_eats_two_flies_and_keeps_streak_at_power_end():
    game = power_game()
    game.fly_cell[:2] = cell(1, 2)
    game.released[1] = True
    events = energize(game)
    eaten = [e.fly for e in events if e.kind == Kind.EATEN]
    revived = [e.fly for e in events if e.cause == ZCause.REVIVE]
    view = game.view()
    assert eaten == revived == [0, 1]
    assert not any(e.kind == Kind.CATCH for e in events)
    assert view.lives == 3 and view.score == 50 + 200 + 400 and view.ghost_streak == 2
    assert np.array_equal(view.fly_cell[:2], game.map.fly_starts[:2])
    assert view.fly_z.tolist() == [False, False, True, True]
    for _ in range(57):
        ended = game.tick_timers()
    assert [e.fly for e in ended if e.kind == Kind.Z_CHANGE] == [2, 3]
    assert game.view().ghost_streak == 2


def test_repeated_power_resets_timer_and_score_order():
    game = power_game()
    game.fly_cell[0] = cell(1, 2)
    energize(game)
    assert game.view().ghost_streak == 1 and game.view().score == 250
    for _ in range(10):
        game.tick_timers()
    game.fly_cell[0] = cell(1, 2)
    events = energize(game)
    assert [e.fly for e in events if e.cause == ZCause.POWER_START] == [0]
    assert game.view().power_ticks == 57
    assert game.view().ghost_streak == 1 and game.view().score == 500


def test_revived_fly_can_catch_while_pacman_is_powered():
    game = power_game()
    game.fly_cell[0] = cell(1, 2)
    energize(game)
    game.fly_cell[0] = cell(1, 3)
    events = game.step_pacman(d.EAST)
    assert [e.kind for e in events] == [Kind.CATCH, Kind.LIFE_LOST]
    assert game.view().power and game.view().lives == 2
    game.reset_life()
    assert not game.view().power and not game.view().fly_z.any()


def test_frightened_speed_agrees_between_decide_and_move():
    game = power_game()
    energize(game)
    game.fly_cell[0], game.fly_dir[0] = cell(1, 6), d.EAST
    assert not game.decision_info()[1][0]
    assert not game.step_flies(np.full(4, -1))[1][0]
    assert game.decision_info()[1][0]
    _, moved, committed, _ = game.step_flies(np.array([d.RIGHT, -1, -1, -1]))
    assert moved[0] and committed[0] and game.fly_cell[0] == cell(2, 6)


def test_powered_pacman_speed_and_dot_debt():
    game = power_game(pacman_power_speed=.9, pacman_power_dot_speed=.79)
    energize(game)
    game.dots[cell(1, 3)] = True
    assert not game.step_pacman(d.EAST)
    assert [e.kind for e in game.step_pacman(d.EAST)] == [Kind.DOT]
    assert np.isclose(game.pac_acc, .8 - (.9 / .79 - 1))
