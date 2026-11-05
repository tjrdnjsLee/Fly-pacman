"""시험용 도우미. docs/ 파일에 의존하지 않는다."""

from __future__ import annotations

import copy

import numpy as np

from flypac.core.config import Config
from flypac.ghosts.random_ghost import RandomGhost
from flypac.maps.original import load_original
from flypac.pacman.base import PacmanController

MAP = load_original()

BASE = {
    "experiment": "test",
    "condition": "regime1",
    "seeds": [0],
    "n_games": 2,
    "engine": {"kind": "fake", "lives": 3, "pacman_speed": 1.0, "fly_speed": 1.0,
               "release_ticks": [0, 10, 20, 30], "score_dot": 10, "score_energizer": 50,
               "watchdog_ticks": 20000},
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

    def propose(self, obs, alive, views=None):
        self.log.append(("propose", alive.copy(), views))
        return super().propose(obs, alive, views)

    def commit(self, executed, active):
        self.log.append(("commit", executed.copy(), active.copy()))
        super().commit(executed, active)

    def on_release(self, mask):
        self.log.append(("on_release", mask.copy()))
        super().on_release(mask)

    def flush_learning(self, mask):
        self.log.append(("flush", mask.copy()))
        super().flush_learning(mask)


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
        self.calls.append(("frame", game))

    def replay_end(self, game, keep):
        self.calls.append(("replay_end", game, keep))

    def of(self, name):
        return [c for c in self.calls if c[0] == name]
