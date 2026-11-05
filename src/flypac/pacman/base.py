"""팩맨 인터페이스 (TRD 5.7). 자리 파일."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from flypac.core.types import GameView


class PacmanController(ABC):
    @abstractmethod
    def act(self, views: list[GameView]) -> np.ndarray:
        """절대 방향(시계방향 번호) (B,)."""
        raise NotImplementedError

    @abstractmethod
    def reset(self, mask: np.ndarray) -> None:
        """프레임 이력 초기화 (목숨 · 판 종료 시). mask (B,) bool."""
        raise NotImplementedError
