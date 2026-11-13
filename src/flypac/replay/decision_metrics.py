"""결정 단위 지표: 상태별 최강 방향 회피 (DECISIONS 17.6, 31.1, 33.4). 순수 함수만 둔다.

- `strongest_set(odor, legal)`: 합법 방향 중 냄새 관측이 가장 센 방향의 집합. 기록기가 재생 로그의 `strongest`
  필드를 만들 때 쓰는 유일한 정의다 (형식 5). 뇌가 받은 float64 관측에서 정확한 같음으로 비교한다.
- `strongest_avoidance(replay)`: 재생 로그 하나(형식 5)에서 상태 z별 회피 비율과 우연 기준.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

MIN_VERSION_FIELDS = ("legal", "strongest", "committed", "action", "fly_z")  # 형식 5 이상에만 다 있다


def strongest_set(odor: np.ndarray, legal: np.ndarray) -> np.ndarray:
    """`legal & (odor == max_{a∈legal} odor_a)` (마지막 축이 방향). 동률은 전부 참, 합법 방향이 없으면 모두 거짓.

    odor는 형을 바꾸지 않고 그대로 비교한다 (float16 등으로 줄이면 가까운 두 값의 순서가 사라진다, DECISIONS 33.4)."""
    odor = np.asarray(odor)
    legal = np.asarray(legal, dtype=bool)
    if odor.shape != legal.shape:
        raise ValueError(f"odor 모양 {odor.shape} ≠ legal 모양 {legal.shape}")
    masked = np.where(legal, odor, -np.inf)
    best = masked.max(axis=-1, keepdims=True)
    return legal & (masked == best)


def strongest_avoidance(replay: Mapping) -> dict[int, dict]:
    """재생 로그 하나의 배열 사전(`read_replay`의 둘째 값)에서 {z: {"n", "avoid", "chance"}} (z = 0, 1).

    분모는 확정된 결정(`committed`)이다 (DECISIONS 33.4). 표본마다 L = legal, M = strongest, a = action,
    z = fly_z (그 틱의 상태)로 회피 = 1[a ∉ M], 우연 기준 = (|L| − |M|)/|L| (17.6). avoid · chance는 z별 표본 평균이고
    n = 0이면 None. 형식 5보다 옛 로그(legal · strongest 없음)는 정확히 계산할 수 없어 ValueError."""
    missing = [k for k in MIN_VERSION_FIELDS if k not in replay]
    if missing:
        raise ValueError(f"재생 로그에 {missing}이 없다: 형식 5 이상이어야 한다 (DECISIONS 33.4)")
    committed = np.asarray(replay["committed"], dtype=bool)
    action = np.asarray(replay["action"]).astype(np.int64)
    legal = np.asarray(replay["legal"], dtype=bool)
    strongest = np.asarray(replay["strongest"], dtype=bool)
    z = np.asarray(replay["fly_z"], dtype=bool)
    t_idx, f_idx = np.nonzero(committed)
    a = action[t_idx, f_idx]
    if (a < 0).any() or (a >= legal.shape[-1]).any():
        raise ValueError("확정된 결정의 행동이 0 ~ 3이 아니다")
    L, M = legal[t_idx, f_idx], strongest[t_idx, f_idx]
    if not L[np.arange(len(a)), a].all():
        raise ValueError("확정된 결정의 행동이 합법 방향이 아니다")
    if (M & ~L).any():
        raise ValueError("strongest가 합법 방향 밖에 있다")
    nL, nM = L.sum(axis=1), M.sum(axis=1)
    avoid = ~M[np.arange(len(a)), a]
    chance = (nL - nM) / nL
    zz = z[t_idx, f_idx]
    out = {}
    for state in (0, 1):
        sel = zz == bool(state)
        n = int(sel.sum())
        out[state] = {"n": n, "avoid": float(avoid[sel].mean()) if n else None,
                      "chance": float(chance[sel].mean()) if n else None}
    return out
