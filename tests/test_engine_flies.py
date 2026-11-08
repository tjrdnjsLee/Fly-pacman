"""초파리 이동과 갈림길: DECISIONS 4.1의 1·2·3·5·6행 (4행은 유령 집 단계)."""

import numpy as np
import pytest
from helpers import MAP, cell
from test_engine_movement import new_game

from flypac.core import directions as d
from flypac.core.events import Kind


@pytest.mark.parametrize('rc,heading,options,decide', [
    ((1, 1), d.WEST, {d.SOUTH}, False),
    ((1, 2), d.EAST, {d.EAST}, False),
    ((1, 6), d.EAST, {d.EAST, d.SOUTH}, True),
    ((5, 6), d.EAST, {d.NORTH, d.EAST, d.SOUTH}, True),
    ((5, 12), d.EAST, {d.NORTH, d.EAST}, True),
    ((14, 0), d.WEST, {d.WEST}, False),
    ((11, 13), d.EAST, {d.EAST}, False),
])
def test_original_maze_decision_samples(rc, heading, options, decide):
    game = new_game(fly_speed=1)
    game.fly_cell[0], game.fly_dir[0] = cell(*rc), heading
    legal, decisions = game.decision_info()
    got = {int(d.ego_to_abs(heading, a)) for a in np.flatnonzero(legal[0])}
    assert got == options
    assert bool(decisions[0]) == decide
    assert not legal[1:].any() and not decisions[1:].any()


@pytest.mark.parametrize('action,target', [(d.FORWARD, (1, 7)), (d.RIGHT, (2, 6))])
def test_junction_commits_chosen_direction(action, target):
    game = new_game(fly_speed=1)
    game.fly_cell[0], game.fly_dir[0] = cell(1, 6), d.EAST
    assert game.decision_info()[1][0]
    events, moved, committed, active = game.step_flies(np.array([action, -1, -1, -1]))
    assert events == [] and moved.tolist() == [True, False, False, False]
    assert committed.tolist() == [True, False, False, False]
    assert not active.any() and game.fly_cell[0] == cell(*target)


def test_decision_waits_for_speed_and_does_not_change_heading_early():
    game = new_game(fly_speed=.25)
    game.fly_cell[0], game.fly_dir[0] = cell(1, 6), d.EAST
    for expected in (.25, .5, .75):
        assert not game.decision_info()[1][0]
        _, moved, committed, _ = game.step_flies(np.array([d.RIGHT, -1, -1, -1]))
        assert not moved.any() and not committed.any()
        assert game.fly_cell[0] == cell(1, 6) and game.fly_dir[0] == d.EAST
        assert game.fly_acc[0] == expected
    assert game.decision_info()[1][0]
    _, moved, committed, _ = game.step_flies(np.array([d.FORWARD, -1, -1, -1]))
    assert moved[0] and committed[0] and game.fly_cell[0] == cell(1, 7)


@pytest.mark.parametrize('start,heading,target', [
    ((1, 2), d.EAST, (1, 3)),
    ((1, 1), d.WEST, (2, 1)),
    ((14, 0), d.WEST, (14, 27)),
])
def test_unique_route_moves_without_commit(start, heading, target):
    game = new_game(fly_speed=1)
    game.fly_cell[0], game.fly_dir[0] = cell(*start), heading
    _, moved, committed, _ = game.step_flies(np.full(4, -1))
    assert moved[0] and not committed.any() and game.fly_cell[0] == cell(*target)


def test_nondecision_positive_action_moves_without_commit():
    game = new_game(fly_speed=1)
    game.fly_cell[0], game.fly_dir[0] = cell(1, 2), d.EAST
    assert not game.decision_info()[1][0]
    _, moved, committed, _ = game.step_flies(np.array([d.FORWARD, -1, -1, -1]))
    assert moved[0] and not committed.any()


def test_catch_commits_catching_fly_and_discards_later_proposals():
    game = new_game(fly_speed=1)
    game.pac_cell = cell(1, 7)
    game.fly_cell[:2], game.fly_dir[:2] = cell(1, 6), d.EAST
    game.released[:2] = True
    game.learning_active = True
    before = game.fly_cell.copy()
    assert game.decision_info()[1][:2].all()
    events, moved, committed, active = game.step_flies(np.full(4, d.FORWARD))
    assert [e.kind for e in events] == [Kind.CATCH, Kind.LIFE_LOST]
    assert events[0].fly == 0 and all(e.active for e in events)
    assert moved.tolist() == committed.tolist() == [True, False, False, False]
    assert active[0] and np.array_equal(game.fly_cell[1:], before[1:])


def test_frozen_game_does_not_decide_or_move():
    game = new_game(fly_speed=1)
    game.fly_cell[0], game.fly_dir[0], game.frozen = cell(1, 6), d.EAST, True
    assert not game.decision_info()[1].any()
    before = game.fly_cell.copy()
    _, moved, committed, _ = game.step_flies(np.full(4, d.FORWARD))
    assert not moved.any() and not committed.any()
    assert np.array_equal(game.fly_cell, before)


def test_reverse_is_not_legal_at_normal_junction():
    game = new_game(fly_speed=1)
    game.fly_cell[0], game.fly_dir[0] = cell(1, 6), d.EAST
    assert not game.decision_info()[0][0, d.BACK]
    with pytest.raises(ValueError, match='불가능한 행동'):
        game.step_flies(np.array([d.BACK, -1, -1, -1]))


def test_returned_arrays_survive_later_calls():
    game = new_game(fly_speed=1)
    game.fly_cell[0], game.fly_dir[0] = cell(1, 6), d.EAST
    legal, decide = game.decision_info()
    expected_legal, expected_decide = legal.copy(), decide.copy()
    _, moved, committed, active = game.step_flies(np.array([d.FORWARD, -1, -1, -1]))
    expected = [x.copy() for x in (moved, committed, active)]
    game.frozen = True
    game.decision_info()
    game.step_flies(np.full(4, -1))
    assert np.array_equal(legal, expected_legal) and np.array_equal(decide, expected_decide)
    assert all(np.array_equal(x, y) for x, y in zip((moved, committed, active), expected))


def test_life_reset_discards_reserved_absolute_direction():
    game = new_game()
    game.pending_fly_dir[0] = d.SOUTH
    game.reset_life()
    assert np.all(game.pending_fly_dir == -1)
    assert game.released.tolist() == [True, False, False, False]
