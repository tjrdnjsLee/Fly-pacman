"""감각 (TRD 5.4, 9절). 뇌에는 FlyObs만 넘긴다 (원칙 2)."""

from __future__ import annotations

import numpy as np

from flypac.core import directions as d
from flypac.core.types import FlyObs, GameView


def _read(c: np.ndarray, cells: np.ndarray, headings: np.ndarray, nbr: np.ndarray, c0: float) -> np.ndarray:
    """c (B,N) 농도 → (B,F,4) 자기중심 순서의 이웃 칸 ln(1 + c/c0). 벽은 0."""
    ego = np.arange(4)
    absd = d.ego_to_abs(headings[..., None], ego)  # (B,F,4)
    neighbor = nbr[cells[..., None], absd]  # (B,F,4), 벽은 -1
    b = np.arange(c.shape[0])[:, None, None]
    val = np.log1p(c[b, np.clip(neighbor, 0, None)] / c0)
    return np.where(neighbor >= 0, val, 0.0)


def sense(
    odor: np.ndarray,
    phero: np.ndarray,
    views: list[GameView],
    info: tuple[np.ndarray, np.ndarray],
    c0: float,
    *,
    nbr: np.ndarray,
) -> FlyObs:
    """이웃표로 자기중심 순서의 칸 농도를 읽고 ln(1 + c/c0). 결정 정보 결합 (TRD 5.4).

    odor, phero: (B,N) 농도. info: Game.decision_info()의 (legal, decide).
    nbr: 맵의 이웃표 (N,4). TRD 5.4 표가 이웃표의 출처를 정하지 않아 키워드 인자로 받는다 (DECISIONS 참고).
    """
    if not c0 > 0:
        raise ValueError(f"c0 > 0 이어야 한다: {c0}")
    cells = np.stack([v.fly_cell for v in views]).astype(np.int64)  # (B,F)
    headings = np.stack([v.fly_dir for v in views]).astype(np.int64)
    z = np.stack([v.fly_z for v in views])
    legal, decide = info
    obs = FlyObs(
        odor=_read(odor, cells, headings, nbr, c0),
        phero=_read(phero, cells, headings, nbr, c0),
        z=z,
        legal=np.asarray(legal, dtype=bool),
        decide=np.asarray(decide, dtype=bool),
    )
    obs.validate()
    return obs
