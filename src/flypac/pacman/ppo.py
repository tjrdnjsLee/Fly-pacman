"""PPO 팩맨 (TRD 5.7, 12.2). 자리 파일: 실험 머신용. torch는 pacman 선택 묶음이라 여기서 가져오지 않는다."""

from __future__ import annotations

import numpy as np

from flypac.core.types import GameView
from flypac.pacman.base import PacmanController


class PPOPacman(PacmanController):
    def __init__(self, checkpoint: str, device: str, eps: float):
        raise NotImplementedError

    def act(self, views: list[GameView]) -> np.ndarray:
        raise NotImplementedError

    def reset(self, mask: np.ndarray) -> None:
        raise NotImplementedError
