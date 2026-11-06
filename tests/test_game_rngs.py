"""게임별 · 용도별 난수 생성기 (TRD 13.2, 점검 F05)."""

from __future__ import annotations

import numpy as np
import pytest
from helpers import make_cfg

from flypac.core.rng import GameRngs, game_rngs, make_rng
from flypac.core.runner import build_runner


def test_slot_stream_depends_only_on_seed_game_and_purpose():
    a = game_rngs([3, 7], "ghost")
    b = game_rngs([7], "ghost")
    assert np.array_equal(a[1].random(5), b[0].random(5))
    assert np.array_equal(game_rngs([7], "ghost")[0].random(5), make_rng(7, 1, "ghost").random(5))
    assert not np.array_equal(game_rngs([7], "ghost")[0].random(5), game_rngs([7], "brain")[0].random(5))


def test_renew_switches_to_new_game_number_only_for_mask():
    r = game_rngs([1, 2], "ghost")
    keep = r[0].bit_generator.state
    r.renew(np.array([False, True]), np.array([1, 2]))
    assert r[0].bit_generator.state == keep
    assert list(r.game_no) == [1, 2]
    assert np.array_equal(r[1].random(3), make_rng(2, 2, "ghost").random(3))


def test_state_dict_roundtrip_and_bad_purpose():
    r = game_rngs([4, 5], "brain")
    st = r.state_dict()
    x = r[1].random(4)
    r.load_state_dict(st)
    assert np.array_equal(r[1].random(4), x)
    with pytest.raises(KeyError):
        GameRngs([1], "nope")


def _action_streams(seeds, n_games=2):
    """(시드, 판 번호)별 무작위 유령의 행동열."""
    cfg = make_cfg(seeds=list(seeds), n_games=n_games)
    r = build_runner(cfg)
    streams: dict[tuple[int, int], list] = {}
    original = r.ghosts.propose

    def spy(obs, alive, views=None):
        out = original(obs, alive, views)
        for b, s in enumerate(seeds):
            if not r.finished[b]:
                streams.setdefault((int(s), int(r.game_no[b])), []).append(out[b].copy())
        return out

    r.ghosts.propose = spy
    r.run()
    return {k: np.stack(v) for k, v in streams.items()}


def test_same_seed_game_same_actions_regardless_of_batch_size_and_order():
    alone = _action_streams([7])
    pair = _action_streams([3, 7])
    swapped = _action_streams([7, 3])
    for g in (1, 2):
        key = (7, g)
        assert (alone[key] >= 0).any()                       # 결정이 실제로 있었다
        assert np.array_equal(alone[key], pair[key])
        assert np.array_equal(alone[key], swapped[key])
    assert np.array_equal(pair[(3, 1)], swapped[(3, 1)])
    assert not np.array_equal(alone[(7, 1)][:50], alone[(7, 2)][:50])   # 판마다 다른 난수열
