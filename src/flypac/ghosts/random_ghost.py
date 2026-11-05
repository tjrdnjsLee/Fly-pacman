"""디버그용 무작위 초파리 (TRD 3.1, 5.5). 자리 파일. propose는 가능한 행동 중 무작위, commit은 기록만."""

from __future__ import annotations

import numpy as np

from flypac.brain.base import GhostController
from flypac.core.types import FlyObs, GameView


class RandomGhost(GhostController):
    def __init__(self, cfg, B: int, rng: np.random.Generator):
        raise NotImplementedError

    def propose(self, obs: FlyObs, alive: np.ndarray, views: list[GameView] | None = None) -> np.ndarray:
        raise NotImplementedError

    def commit(self, executed: np.ndarray, active: np.ndarray) -> None:
        raise NotImplementedError

    def give_reward(self, R: np.ndarray) -> None:
        raise NotImplementedError

    def on_release(self, mask: np.ndarray) -> None:
        raise NotImplementedError

    def on_eaten(self, mask: np.ndarray) -> None:
        raise NotImplementedError

    def flush_learning(self, mask: np.ndarray) -> None:
        raise NotImplementedError

    def reset_life(self, mask: np.ndarray) -> None:
        raise NotImplementedError

    def state_dict(self) -> dict:
        raise NotImplementedError

    def load_state_dict(self, state: dict) -> None:
        raise NotImplementedError
