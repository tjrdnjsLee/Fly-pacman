"""디버그용 무작위 초파리 (TRD 3.1, 5.5). propose는 가능한 행동 중 무작위, commit은 기록만. views를 받지 않는다."""

from __future__ import annotations

import numpy as np

from flypac.brain.base import GhostController
from flypac.core.types import FlyObs, GameView


class RandomGhost(GhostController):
    def __init__(self, cfg, B: int, rng: np.random.Generator):
        self.B, self.F = B, 4
        self.rng = rng
        self.proposed = np.full((B, self.F), -1, dtype=np.int64)
        self.n_committed = np.zeros((B, self.F), dtype=np.int64)
        self.n_committed_active = np.zeros((B, self.F), dtype=np.int64)
        self.reward_total = np.zeros((B, self.F))
        self.n_release = np.zeros(B, dtype=np.int64)
        self.n_eaten = np.zeros((B, self.F), dtype=np.int64)
        self.n_flush = np.zeros(B, dtype=np.int64)
        self.n_reset_life = np.zeros(B, dtype=np.int64)

    def propose(self, obs: FlyObs, alive: np.ndarray, views: list[GameView] | None = None) -> np.ndarray:
        out = np.full((self.B, self.F), -1, dtype=np.int64)
        for b in range(self.B):
            if not alive[b]:
                continue
            for k in range(self.F):
                if obs.decide[b, k]:
                    out[b, k] = self.rng.choice(np.flatnonzero(obs.legal[b, k]))
        self.proposed = out
        return out

    def commit(self, executed: np.ndarray, active: np.ndarray) -> None:
        self.n_committed += executed
        self.n_committed_active += executed & active

    def give_reward(self, R: np.ndarray) -> None:
        self.reward_total += R

    def on_release(self, mask: np.ndarray) -> None:
        self.n_release += mask

    def on_eaten(self, mask: np.ndarray) -> None:
        self.n_eaten += mask

    def flush_learning(self, mask: np.ndarray) -> None:
        self.n_flush += mask

    def reset_life(self, mask: np.ndarray) -> None:
        self.n_reset_life += mask
        self.proposed[mask] = -1

    def state_dict(self) -> dict:
        return {"rng": self.rng.bit_generator.state, "reward_total": self.reward_total.copy()}

    def load_state_dict(self, state: dict) -> None:
        self.rng.bit_generator.state = state["rng"]
        self.reward_total = np.asarray(state["reward_total"]).copy()
