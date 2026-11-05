"""감각 (TRD 5.4, 9절). 자리 파일."""

from __future__ import annotations

import numpy as np

from flypac.core.types import FlyObs, GameView


def sense(
    odor: np.ndarray,
    phero: np.ndarray,
    views: list[GameView],
    info: tuple[np.ndarray, np.ndarray],
    c0: float,
) -> FlyObs:
    """이웃표로 자기중심 순서의 칸 농도를 읽고 ln(1 + c/c0). 결정 정보 결합 (TRD 5.4).

    odor, phero: (B,N) 농도. info: Game.decision_info()의 (legal, decide).
    """
    raise NotImplementedError
