"""게임 상태 사본과 관측 묶음 (TRD 4.6, 4.7)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GameView:
    """게임 한 판의 읽기 전용 사본 (TRD 4.6). 만든 순간의 상태다.

    팩맨 AI, 기준선 B1, 감각, 기록이 쓴다. 초파리 뇌는 받지 않는다.
    """

    # 팩맨
    pac_cell: int
    pac_dir: int
    lives: int
    power: bool
    power_ticks: int
    ghost_streak: int
    # 초파리 F마리: 모두 (F,)
    fly_cell: np.ndarray
    fly_dir: np.ndarray
    fly_z: np.ndarray
    fly_in_house: np.ndarray
    # 맵 상태: 칸 번호로 색인한 (N,) bool
    dots: np.ndarray
    energizers: np.ndarray
    fruit: bool
    # 진행
    tick: int
    score: int
    learning_active: bool
    elroy_stage: int
    frozen: bool
    # 속도 누적값 (재생 로그의 보간용, TRD 19.6). 기본값이 있어 TRD 4.6의 필드 구성을 깨지 않는다.
    pac_acc: float = 0.0
    fly_acc: np.ndarray | None = None  # (F,)


@dataclass(frozen=True)
class FlyObs:
    """초파리 관측 묶음 (TRD 4.7). B: 게임 수, F: 초파리 수.

    odor, phero: (B,F,4) 자기중심 순서(좌·앞·우·뒤), ln(1+c/c0) 적용 후, 벽은 0.
    z: (B,F) 내부 상태. legal: (B,F,4) bool. decide: (B,F) bool.
    """

    odor: np.ndarray
    phero: np.ndarray
    z: np.ndarray
    legal: np.ndarray
    decide: np.ndarray

    def validate(self) -> None:
        b, f = self.z.shape
        for name in ("odor", "phero", "legal"):
            if getattr(self, name).shape != (b, f, 4):
                raise ValueError(f"{name}의 모양은 {(b, f, 4)}이어야 한다")
        if self.decide.shape != (b, f):
            raise ValueError(f"decide의 모양은 {(b, f)}이어야 한다")
