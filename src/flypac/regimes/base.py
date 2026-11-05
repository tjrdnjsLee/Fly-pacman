"""보상 체제 공통 틀 (TRD 5.6, 11절). 자리 파일. 활성이 거짓인 사건은 무시한다 (TRD 4.5)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from flypac.core.events import Event


def fly_index(ev: Event, n_flies: int = 4) -> int:
    """사건의 초파리 번호를 배열 인덱스로 쓰기 전에 검사한다.

    fly = -1이 numpy 인덱스로 들어가면 오류 없이 마지막 초파리를 가리키므로,
    체제 구현은 R[ev.fly] 대신 R[fly_index(ev, F)]를 쓴다.
    """
    if not 0 <= ev.fly < n_flies:
        raise ValueError(f"{ev.kind} 사건의 초파리 번호 {ev.fly}가 0 <= fly < {n_flies}를 벗어났다")
    return ev.fly


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
