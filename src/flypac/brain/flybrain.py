"""초파리 뇌 (TRD 5.5, 10절). 자리 파일: 구현은 뇌 담당. B0, B0'는 같은 코드에서 설정으로 만든다."""

from __future__ import annotations

import numpy as np

from flypac.brain.base import GhostController
from flypac.core.types import FlyObs, GameView


class FlyBrain(GhostController):
    """views를 받지 않는다 (TRD 원칙 2)."""

    def __init__(self, cfg, B: int, rng: np.random.Generator):
        raise NotImplementedError

    def propose(self, obs: FlyObs, alive: np.ndarray, views: list[GameView] | None = None) -> np.ndarray:
        raise NotImplementedError

    def commit(self, committed: np.ndarray, active: np.ndarray) -> None:
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
