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


def see(cells: np.ndarray, headings: np.ndarray, pac: np.ndarray, nbr: np.ndarray, R: int, s: float) -> np.ndarray:
    """시각 (DECISIONS 42.4) → (B,F,4) 자기중심 순서 (좌 · 앞 · 우 · 뒤).

    겹눈: 네 방향을 모두 본다. 방향 a로 이웃표를 따라 곧게 1 ~ R칸을 가며, 벽(-1)을 만나면 멈춘다
    (터널은 이웃표가 잇는 대로 이어진다, 냄새 읽기와 같다). 그 안에 팩맨 칸이 있으면 s, 없으면 0. 같은 칸(거리 0)은 0.
    행동 규칙이 아니다: 무엇이 보이는지만 알려 준다. 어떻게 쓸지는 버섯체가 배운다.
    """
    B, F = cells.shape
    out = np.zeros((B, F, 4))
    if R <= 0:
        return out
    absd = d.ego_to_abs(headings[..., None], np.arange(4))   # (B,F,4)
    for b in range(B):
        for f in range(F):
            for e in range(4):
                c = int(cells[b, f])
                a = int(absd[b, f, e])
                for _ in range(int(R)):
                    c = int(nbr[c, a])
                    if c < 0:
                        break
                    if c == int(pac[b]):
                        out[b, f, e] = s
                        break
    return out


def sense(
    odor: np.ndarray,
    phero: np.ndarray,
    views: list[GameView],
    info: tuple[np.ndarray, np.ndarray],
    c0: float,
    *,
    nbr: np.ndarray,
    vision_R: int = 0,
    vision_s: float = 0.0,
) -> FlyObs:
    """이웃표로 자기중심 순서의 칸 농도를 읽고 ln(1 + c/c0). 결정 정보 결합 (TRD 5.4).

    odor, phero: (B,N) 농도. info: Game.decision_info()의 (legal, decide).
    nbr: 맵의 이웃표 (N,4). TRD 5.4 표가 이웃표의 출처를 정하지 않아 키워드 인자로 받는다 (DECISIONS 참고).
    vision_R > 0이면 시각(see)을 계산해 obs.vision에 넣는다. 0이면 vision = None (v1과 같다).
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
        vision=(see(cells, headings, np.array([v.pac_cell for v in views], dtype=np.int64), nbr,
                    int(vision_R), float(vision_s)) if vision_R > 0 else None),
    )
    obs.validate()
    return obs
