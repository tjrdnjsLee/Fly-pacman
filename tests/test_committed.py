"""moved와 committed의 분리 (TRD 4.7, 10.4, DECISIONS 4절의 표, 점검 F04)."""

from __future__ import annotations

import numpy as np
from helpers import MAP, cell, make_cfg, place

from flypac.core import directions as d
from flypac.core.rng import make_rng
from flypac.engine.fake import FakeGame


def new_game(**over):
    over.setdefault("engine", {})
    over["engine"] = {"pacman_speed": 0.0, **over["engine"]}
    return FakeGame(MAP, make_cfg(**over), make_rng(0, 0, "engine"))


def _find(g, want_decide: bool):
    """초파리 0을 둘 (칸, 방향): decide가 want_decide이고 반전 외 선택지가 있는 곳."""
    for c in range(MAP.nbr.shape[0]):
        for h in range(4):
            g.fly_cell[0], g.fly_dir[0] = c, h
            legal, decide = g.decision_info()
            if legal[0].any() and bool(decide[0]) == want_decide and not g.view().fly_in_house[0]:
                return c, h
    raise AssertionError("자리가 없다")


def _game_with_fly0(want_decide):
    g = new_game()
    place(g, pac=cell(23, 1), released=True, active=True, flies={1: cell(5, 1), 2: cell(5, 20), 3: cell(5, 22)},
          fly_dirs={1: d.EAST, 2: d.WEST, 3: d.WEST})
    g.fly_acc[:] = 0.99  # 이번 틱에 모두 움직인다
    c, h = _find(g, want_decide)
    g.fly_cell[0], g.fly_dir[0] = c, h
    return g


def test_corridor_move_is_moved_but_not_committed():
    g = _game_with_fly0(want_decide=False)
    before = int(g.fly_cell[0])
    ev, moved, committed, active = g.step_flies(np.full(4, -1))
    assert moved[0] and int(g.fly_cell[0]) != before
    assert not committed[0]


def test_decision_move_is_moved_and_committed():
    g = _game_with_fly0(want_decide=True)
    legal, decide = g.decision_info()
    act = np.full(4, -1)
    act[0] = int(np.flatnonzero(legal[0])[0])
    ev, moved, committed, active = g.step_flies(act)
    assert moved[0] and committed[0]


def test_house_exit_is_moved_but_not_committed():
    g = new_game(engine={"release_ticks": [0, 0, 0, 0]})
    ev, moved, committed, active = g.step_flies(np.full(4, -1))
    assert moved.all() and not committed.any()


def test_too_slow_to_move_is_neither():
    g = _game_with_fly0(want_decide=True)
    g.fly_acc[0] = -5.0  # 이번 틱에 움직이지 못한다
    legal, decide = g.decision_info()
    assert not decide[0]
    ev, moved, committed, active = g.step_flies(np.full(4, -1))
    assert not moved[0] and not committed[0]


def test_frozen_game_is_neither():
    g = _game_with_fly0(want_decide=True)
    g.frozen = True
    ev, moved, committed, active = g.step_flies(np.zeros(4, dtype=np.int64))
    assert not moved.any() and not committed.any()
