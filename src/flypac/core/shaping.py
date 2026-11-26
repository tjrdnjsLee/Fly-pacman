"""냄새 퍼텐셜 보상 (DECISIONS 42.5). 행동이 아니라 목표에 냄새를 넣는다.

    R ← R + λ [Φ(s_t) − Φ(s_{t−1})],   Φ = σ(z) ln(1 + c(초파리 칸)/c0),   σ(0) = +1, σ(1) = −1

- 틱마다 더한다. 결정 사이의 합은 망원으로 Φ(결정 n) − Φ(결정 n−1)이 된다 (퍼텐셜 기반 보상 설계, Ng 외 1999).
- 평소엔 다가가면 +, 겁먹으면 멀어지면 +. 한자리에 머물면 0이라 "꽁무니에 머물기"로는 쌓이지 않는다.
- 기준점 Φ_prev를 다시 잡는 때 (그 틱의 항 0): 목숨 · 판 시작(reset), 먹혀 집으로 순간이동(eaten), z가 바뀐 틱,
  진행 중이 아닌 칸. 초파리가 고를 수 없는 뜀(장 초기화, 순간이동, 부호 전환)을 보상에 넣지 않는다.
  이 셋 때문에 엄밀한 퍼텐셜 형태에서 벗어난다 (DECISIONS 42.5에 적었다).
- λ = 0이면 TickCore가 이 객체를 만들지 않는다 (v1과 같다).
"""

from __future__ import annotations

import numpy as np


class OdorShaping:
    def __init__(self, B: int, F: int, lam: float, c0: float):
        if not lam > 0:
            raise ValueError(f"λ > 0 이어야 한다: {lam}")
        if not c0 > 0:
            raise ValueError(f"c0 > 0 이어야 한다: {c0}")
        self.B, self.F, self.lam, self.c0 = int(B), int(F), float(lam), float(c0)
        self.prev = np.zeros((B, F))
        self.zprev = np.zeros((B, F), dtype=np.int64)
        self.valid = np.zeros((B, F), dtype=bool)

    def reset(self, mask: np.ndarray) -> None:
        """칸 mask(B,)의 네 마리 기준점을 버린다 (목숨 · 판 시작)."""
        self.valid[np.asarray(mask, dtype=bool)] = False

    def eaten(self, mask: np.ndarray) -> None:
        """먹힌 초파리 mask(B,F)의 기준점을 버린다 (집으로 순간이동)."""
        self.valid[np.asarray(mask, dtype=bool)] = False

    def phi(self, c: np.ndarray, views) -> tuple[np.ndarray, np.ndarray]:
        B, F = self.B, self.F
        out = np.zeros((B, F))
        z = np.zeros((B, F), dtype=np.int64)
        for b in range(B):
            cells = np.asarray(views[b].fly_cell, dtype=np.int64)
            z[b] = np.asarray(views[b].fly_z, dtype=np.int64)
            out[b] = np.where(z[b] == 1, -1.0, 1.0) * np.log1p(c[b, cells] / self.c0)
        return out, z

    def step(self, c: np.ndarray, views, running: np.ndarray) -> np.ndarray:
        """이번 틱의 냄새 항 λ ΔΦ (B,F). c: 이번 틱 확산 뒤의 냄새 장 (B,N). views: 그 시점의 views."""
        phi, z = self.phi(c, views)
        ok = self.valid & (z == self.zprev) & np.asarray(running, dtype=bool)[:, None]
        d = np.where(ok, phi - self.prev, 0.0)
        self.prev, self.zprev = phi, z
        self.valid = np.ones((self.B, self.F), dtype=bool)
        return self.lam * d
