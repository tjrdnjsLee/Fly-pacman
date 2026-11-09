"""시험용 도우미. docs/ 파일에 의존하지 않는다."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from flypac.core.config import Config, load_config
from flypac.ghosts.random_ghost import RandomGhost
from flypac.maps.original import load_original
from flypac.pacman.base import PacmanController

MAP = load_original()

BASE = {
    "experiment": "test",
    "condition": "regime1",
    "seeds": [0],
    "n_games": 2,
    "engine": load_config(Path(__file__).resolve().parents[1] / "configs" / "m0_smoke.yaml").engine.to_dict(),
    "time": {"v_max_tiles_per_s": 9.47},
    "odor": {"D": 1.0, "lam": 0.05, "s": 1.0, "c0": 0.01, "reset_on_life_loss": True},
    "phero": {"D": 1.0, "lam": 0.1},
    "reward": {"p": 0.25, "q": 1.0},
    "pacman": {"kind": "rule_based"},
    "rule_based": {"danger_dist": 6, "chase_dist": 8},
    "ghosts": {"kind": "random"},
    "run": {"max_loop_ticks": 50000},
}


def make_cfg(**over) -> Config:
    """BASE에 최상위 절 단위로 덮어쓴다 (절은 사전 병합)."""
    d = copy.deepcopy(BASE)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            d[k].update(v)
        else:
            d[k] = v
    return Config(d)


def cell(r: int, c: int) -> int:
    i = int(MAP.cell_id[r, c])
    assert i >= 0, f"({r},{c})는 벽이다"
    return i


def place(game, pac=None, pac_dir=None, flies=None, fly_dirs=None, released=None, active=None):
    """FakeGame의 상태를 직접 지정한다. flies: {번호: 칸}."""
    if pac is not None:
        game.pac_cell = pac
    if pac_dir is not None:
        game.pac_dir = pac_dir
    for k, c in (flies or {}).items():
        game.fly_cell[k] = c
    for k, h in (fly_dirs or {}).items():
        game.fly_dir[k] = h
    if released is not None:
        game.released[:] = released
    if active is not None:
        game.learning_active = active


class Scripted(PacmanController):
    """정해진 방향을 계속 내는 팩맨."""

    def __init__(self, direction):
        self.direction = direction
        self.resets = 0

    def act(self, views):
        return np.full(len(views), self.direction, dtype=np.int64)

    def reset(self, mask):
        self.resets += int(np.sum(mask))


class SpyGhost(RandomGhost):
    uses_views = False

    def __init__(self, cfg, B, rng):
        super().__init__(cfg, B, rng)
        self.log = []
        self.probe = None  # flush 시점에 부를 함수 (예: 게임의 learning_active 읽기)

    def propose(self, obs, alive, views=None):
        self.log.append(("propose", alive.copy(), views))
        return super().propose(obs, alive, views)

    def commit(self, committed, active):
        self.log.append(("commit", committed.copy(), active.copy()))
        super().commit(committed, active)

    def on_release(self, mask):
        self.log.append(("on_release", mask.copy()))
        super().on_release(mask)

    def flush_learning(self, mask):
        self.log.append(("flush", mask.copy(), self.probe() if self.probe else None))
        super().flush_learning(mask)

    def reset_life(self, mask):
        self.log.append(("reset_life", mask.copy()))
        super().reset_life(mask)


class SpyRecorder:
    def __init__(self):
        self.calls = []

    def on_events(self, game, events, view):
        self.calls.append(("events", game, list(events), view))

    def on_life_end(self, game, view, events):
        self.calls.append(("life_end", game, view, list(events)))

    def on_game_end(self, game, view, events):
        self.calls.append(("game_end", game, view, list(events)))

    def replay_frame(self, game, views, obs, odor, phero, brain):
        self.calls.append(("frame", game, odor, phero))

    def replay_end(self, game, keep):
        self.calls.append(("replay_end", game, keep))

    def of(self, name):
        return [c for c in self.calls if c[0] == name]


def share_journal(runner, recorder):
    """조종기와 기록기가 같은 호출 기록을 쓰게 한다. 호출 순서를 한 목록에서 본다.

    게임 초기화 호출도 같은 기록에 남긴다 (마스크가 하나라도 참일 때만).
    """
    journal = runner.ghosts.log
    recorder.calls = journal
    for name in ("retire", "reset_life", "reset_game"):
        original = getattr(runner.game, name)

        def wrapped(mask, *args, _name=name, _original=original):
            if np.any(mask):
                journal.append((f"game_{_name}", np.array(mask).copy()))
            return _original(mask, *args)

        setattr(runner.game, name, wrapped)
    return journal
