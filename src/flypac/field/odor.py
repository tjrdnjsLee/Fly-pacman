"""냄새 장 (TRD 5.3, 8절). 팩맨 냄새와 페로몬은 같은 클래스를 다른 파라미터로 두 번 만든다."""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm, solve


class Field:
    """lam > 0 검사, P와 Q를 미리 계산 (TRD 8절). 단위는 격자 단위(칸, 틱).

    게임 수 B는 첫 `reset(mask)`의 mask 길이로 정해진다. reset 전의 step, 또는 B와 다른
    길이의 mask는 오류다. map은 `nbr` (N,4) 속성(이웃표, 벽 -1)만 쓴다.
    """

    def __init__(self, map, D: float, lam: float):
        if not lam > 0:
            raise ValueError(f"lam > 0 이어야 한다 (TRD 8.1): {lam}")
        nbr = np.asarray(map.nbr)
        n = nbr.shape[0]
        adj = np.zeros((n, n))
        rows, dirs = np.nonzero(nbr >= 0)
        adj[rows, nbr[rows, dirs]] = 1.0
        adj = np.maximum(adj, adj.T)
        L = np.diag(adj.sum(axis=1)) - adj
        A = D * L + lam * np.eye(n)
        self.N = n
        self.P = expm(-A)
        self.Q = solve(A, np.eye(n) - self.P, assume_a="pos")
        self._PT = self.P.T.copy()
        self._QT = self.Q.T.copy()
        self._c: np.ndarray | None = None

    @property
    def B(self) -> int | None:
        return None if self._c is None else self._c.shape[0]

    @property
    def c(self) -> np.ndarray:
        """농도 (B,N). reset 전에는 오류."""
        if self._c is None:
            raise RuntimeError("Field.reset(mask)를 먼저 불러야 한다 (B가 아직 정해지지 않음)")
        return self._c

    def _check_mask(self, mask) -> np.ndarray:
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim != 1:
            raise ValueError(f"mask는 (B,) 이어야 한다: 모양 {mask.shape}")
        if self._c is not None and mask.shape[0] != self._c.shape[0]:
            raise ValueError(f"mask 길이 {mask.shape[0]}가 처음 정한 B={self._c.shape[0]}와 다르다")
        return mask

    def reset(self, mask: np.ndarray) -> None:
        """mask (B,) bool: 지정한 게임의 c = 0. 첫 호출의 mask 길이가 B를 정한다."""
        mask = self._check_mask(mask)
        if self._c is None:
            self._c = np.zeros((mask.shape[0], self.N))
        self._c[mask] = 0.0

    def step(self, S: np.ndarray, mask: np.ndarray) -> None:
        """원천 S (B,N): c ← c Pᵀ + S Qᵀ. mask (B,) bool이 거짓인(동결된) 게임은 건너뜀."""
        c = self.c  # reset 전이면 여기서 오류
        mask = self._check_mask(mask)
        S = np.asarray(S, dtype=float)
        if S.shape != c.shape:
            raise ValueError(f"S의 모양은 {c.shape}이어야 한다: {S.shape}")
        c[mask] = c[mask] @ self._PT + S[mask] @ self._QT

    def total(self) -> np.ndarray:
        """총량 (B,). 질량 보존 테스트용."""
        return self.c.sum(axis=1)
