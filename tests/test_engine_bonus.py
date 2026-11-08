"""과일, 엘로이, 터널 감속 (PRD 3.6, 3.12, 3.13)."""

import numpy as np
import pytest

from flypac.core import directions as d
from flypac.core.events import Kind
from helpers import cell
from test_engine_house import eat_dots, fly_step
from test_engine_movement import new_game


def test_fruit_appears_at_both_thresholds_and_expires():
    game = new_game()
    eat_dots(game, 69)
    assert not game.view().fruit
    eat_dots(game, 1)
    assert game.view().fruit
    for _ in range(89):
        game.tick_timers()
    assert game.view().fruit
    game.tick_timers()
    assert not game.view().fruit

    eat_dots(game, 99)
    assert not game.view().fruit
    eat_dots(game, 1)
    assert game.view().fruit
    game.pac_cell = cell(17, 12)
    score = game.score
    assert [e.kind for e in game.step_pacman(d.EAST)] == [Kind.FRUIT]
    assert game.score == score + 100 and not game.view().fruit


def test_fruit_clears_on_life_loss_without_replaying_a_threshold():
    game = new_game()
    eat_dots(game, 70)
    assert game.view().fruit
    game.lives -= 1
    game.reset_life()
    assert not game.view().fruit
    eat_dots(game, 99)
    assert not game.view().fruit
    eat_dots(game, 1)
    assert game.view().fruit


def set_remaining(game, count):
    game.dots[:] = False
    game.dots[np.flatnonzero(game.map.dots)[:count]] = True


def test_elroy_thresholds_and_speed_affect_only_fly_zero():
    game = new_game(fly_speed=.75)
    for remaining, expected in [(21, 0), (20, 1), (11, 1), (10, 2)]:
        set_remaining(game, remaining)
        game.tick_timers()
        assert game.view().elroy_stage == expected
    game.fly_cell[:2], game.fly_dir[:2] = cell(1, 6), d.EAST
    game.released[:2] = True
    game.fly_acc[:2] = .20
    assert game.decision_info()[1][:2].tolist() == [True, False]
    _, moved, committed, _ = game.step_flies(np.array([d.FORWARD, -1, -1, -1]))
    assert moved[:2].tolist() == committed[:2].tolist() == [True, False]
    assert game.fly_acc[0] == pytest.approx(.05)


def test_elroy_pauses_after_death_until_clyde_heads_to_exit():
    game = new_game(fly_speed=.75)
    set_remaining(game, 10)
    game.tick_timers()
    assert game.view().elroy_stage == 2
    game.lives -= 1
    game.reset_life()
    game.tick_timers()
    assert game.view().elroy_stage == 0
    eat_dots(game, 32)
    fly_step(game)
    game.tick_timers()
    assert game.view().elroy_stage == 0
    for _ in range(38):
        game.tick_timers()
    assert game.view().elroy_stage == 2


def test_tunnel_slows_frightened_fly_but_not_pacman():
    game = new_game(fly_speed=.75)
    game.fly_cell[0], game.fly_dir[0] = cell(14, 0), d.WEST
    game.fly_z = np.array([True, False, False, False])
    for _ in range(2):
        assert not fly_step(game)[1][0]
    assert fly_step(game)[1][0]
    assert game.fly_cell[0] == cell(14, 27)
    assert game.fly_acc[0] == pytest.approx(.2)
    game.pac_cell = cell(14, 0)
    game.step_pacman(d.EAST)
    assert game.pac_cell == cell(14, 1)


def test_real_normal_speed_caps_leave_a_slower_explicit_speed_in_effect():
    game = new_game(
        pacman_speed=1, fly_speed=.6,
        pacman_normal_speed_cap=.8, fly_normal_speed_cap=.75,
        pacman_dot_speed=.71,
    )
    assert game.step_pacman(d.EAST) == [] and game.pac_cell == cell(1, 1)
    assert game.pac_acc == pytest.approx(.8)
    assert not fly_step(game)[1][0]
    assert game.fly_acc[0] == pytest.approx(.6)
