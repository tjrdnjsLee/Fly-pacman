"""디버그용 무작위 초파리 (TRD 3.1, 5.5). propose는 가능한 행동 중 무작위, commit은 기록만. views를 받지 않는다."""

from __future__ import annotations

import numpy as np

from flypac.brain.base import GhostController
from flypac.core.rng import GameRngs
from flypac.core.types import FlyObs, GameView


class RandomGhost(GhostController):
    def __init__(self, cfg, B: int, rngs: GameRngs):
        """rngs: 칸별 생성기 묶음 (core.rng.game_rngs(seeds, "ghost")). 칸 b는 rngs[b]만 쓴다."""
        if not isinstance(rngs, GameRngs) or len(rngs) != B:
            raise TypeError("RandomGhost에는 길이 B의 GameRngs(core.rng.game_rngs)를 넘긴다")
        self.B, self.F = B, 4
        self.rngs = rngs
        self.proposed = np.full((B, self.F), -1, dtype=np.int64)
        self.last_pi = np.zeros((B, self.F, 4))  # 재생 로그용: 고른 행동 확률 (가능한 행동에 균등)
        self.n_committed = np.zeros((B, self.F), dtype=np.int64)
        self.n_committed_active = np.zeros((B, self.F), dtype=np.int64)
        self.reward_total = np.zeros((B, self.F))
        self.n_release = np.zeros(B, dtype=np.int64)
        self.n_eaten = np.zeros((B, self.F), dtype=np.int64)
        self.n_flush = np.zeros(B, dtype=np.int64)
        self.n_reset_life = np.zeros(B, dtype=np.int64)

    def propose(self, obs: FlyObs, alive: np.ndarray, views: list[GameView] | None = None) -> np.ndarray:
        out = np.full((self.B, self.F), -1, dtype=np.int64)
        self.last_pi = np.zeros((self.B, self.F, 4))
        for b in range(self.B):
            if not alive[b]:
                continue
            for k in range(self.F):
                if obs.decide[b, k]:
                    legal = np.flatnonzero(obs.legal[b, k])
                    out[b, k] = self.rngs[b].choice(legal)
                    self.last_pi[b, k, legal] = 1.0 / len(legal)
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
        # 제자리에서 고치지 않는다: propose가 돌려준 배열은 실행기 · 기록기가 아직 들고 있을 수 있다
        self.proposed = np.where(np.asarray(mask, dtype=bool)[:, None], -1, self.proposed)

    def state_dict(self) -> dict:
        return {"rngs": self.rngs.state_dict(), "reward_total": self.reward_total.copy()}

    def load_state_dict(self, state: dict) -> None:
        self.rngs.load_state_dict(state["rngs"])
        self.reward_total = np.asarray(state["reward_total"]).copy()
