"""보상 체제 공통 틀 (TRD 5.6, 11절). 자리 파일. 활성이 거짓인 사건은 무시한다 (TRD 4.5)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from flypac.core.events import Event


class Regime(ABC):
    def __init__(self, cfg):
        raise NotImplementedError

    @abstractmethod
    def rewards(self, events: list[Event]) -> np.ndarray:
        """한 판의 이번 틱 사건 → 초파리별 R (F,). 활성이 거짓인 사건은 무시."""
        raise NotImplementedError

    @abstractmethod
    def emission(self, obs_one_game) -> np.ndarray:
        """페로몬 방출 세기 (F,). 체제 1 · 2는 0."""
        raise NotImplementedError
