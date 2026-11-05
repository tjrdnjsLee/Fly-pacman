"""냄새 장 (TRD 5.3, 8절). 자리 파일. 팩맨 냄새와 페로몬은 같은 클래스를 다른 파라미터로 두 번 만든다."""

from __future__ import annotations

import numpy as np

from flypac.maps.original import MapSpec


class Field:
    """lam > 0 검사, P와 Q를 미리 계산 (TRD 8절). 단위는 격자 단위(칸, 틱)."""

    def __init__(self, map: MapSpec, D: float, lam: float):
        raise NotImplementedError

    @property
    def c(self) -> np.ndarray:
        """농도 (B,N)."""
        raise NotImplementedError

    def reset(self, mask: np.ndarray) -> None:
        """mask (B,) bool: 지정한 게임의 c = 0. 첫 호출의 mask 길이가 B를 정한다."""
        raise NotImplementedError

    def step(self, S: np.ndarray, mask: np.ndarray) -> None:
        """원천 S (B,N): c ← c Pᵀ + S Qᵀ. mask (B,) bool이 거짓인(동결된) 게임은 건너뜀."""
        raise NotImplementedError

    def total(self) -> np.ndarray:
        """총량 (B,). 질량 보존 테스트용."""
        raise NotImplementedError
