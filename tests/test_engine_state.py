"""진짜 엔진의 칸별 난수, 배열 소유권, 동결 틱 (DECISIONS 3, 8, 11절)."""

import numpy as np
import pytest

from flypac.core import directions as d
from flypac.core.config import load_config
from flypac.core.events import Kind
from flypac.core.rng import game_rngs, make_rng
from flypac.engine.game import VecGame
from helpers import MAP, cell
from test_engine_real import SMOKE


def batch():
    slots = game_rngs([9, 4], 'engine')
    vec = VecGame(2, MAP, load_config(SMOKE), [slots[0], slots[1]])
    vec.reset_game(np.array([True, True]), slots)
    return vec, slots


@pytest.mark.parametrize('short_mask', [False, True])
def test_invalid_batch_reset_does_not_partially_replace_rng(short_mask):
    vec, slots = batch()
    fresh = make_rng(9, 2, 'engine')
    mask = np.array([True]) if short_mask else np.array([True, True])
    rngs = [fresh, slots[1]] if short_mask else [fresh]
    with pytest.raises(ValueError):
        vec.reset_game(mask, rngs)
    assert all(g.rng is slots[i] for i, g in enumerate(vec.games))


def test_masked_reset_rejects_generator_from_another_live_slot():
    vec, slots = batch()
    with pytest.raises(ValueError):
        vec.reset_game(np.array([True, False]), [slots[1], slots[1]])
    assert all(g.rng is slots[i] for i, g in enumerate(vec.games))


def test_reset_life_continues_rng_and_masked_new_game_replaces_only_one():
    vec, slots = batch()
    expect = make_rng(9, 1, 'engine').integers(10000, size=2)
    g0, g1 = vec.games
    assert g0.rng.integers(10000) == expect[0]
    vec.reset_life(np.array([True, True]))
    assert g0.rng.integers(10000) == expect[1]
    slots.renew(np.array([False, True]), [1, 2])
    vec.reset_game(np.array([False, True]), slots)
    assert g1.rng.integers(10000) == make_rng(4, 2, 'engine').integers(10000)
    assert g0.rng is slots[0] and g1.rng is slots[1]


def test_views_survive_capture_timer_and_single_slot_reset():
    vec, slots = batch()
    g0, g1 = vec.games
    g0.pac_cell = cell(1, 1)
    g0.pac_acc = .5
    g0.fly_cell[0] = cell(1, 2)
    g0.tick = g1.tick = 12
    before = vec.views()
    caught = vec.step_pacman(np.array([d.EAST, d.WEST]))
    assert [e.kind for e in caught[0]][-2:] == [Kind.CATCH, Kind.LIFE_LOST]
    assert all(e.tick == 12 for e in caught[0])
    vec.step_flies(np.full((2, 4), -1))
    vec.tick_timers()
    assert [v.tick for v in vec.views()] == [13, 13]
    assert before[0].tick == 12 and before[0].lives == 3 and before[0].dots[cell(1, 2)]
    ended = vec.views()[0]
    vec.reset_life(np.array([True, False]))
    assert ended.frozen and ended.fly_cell[0] == cell(1, 2)
    assert vec.views()[0].tick == 13 and vec.views()[0].lives == 2
    slots.renew(np.array([True, False]), [2, 1])
    vec.reset_game(np.array([True, False]), slots)
    assert [v.tick for v in vec.views()] == [0, 13]
