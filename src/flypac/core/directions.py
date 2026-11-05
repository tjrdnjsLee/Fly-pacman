"""방향 규격 (TRD 4.3). 방향 변환은 이 파일에서만 한다.

절대 방향은 시계방향 번호: 북 0, 동 1, 남 2, 서 3.
자기중심 번호 e: 좌 0, 직진 1, 우 2, 반전 3. 행동과 냄새 샘플이 같은 순서를 쓴다.
"""

from __future__ import annotations

import numpy as np

NORTH, EAST, SOUTH, WEST = 0, 1, 2, 3
ABS_NAMES = ("N", "E", "S", "W")

LEFT, FORWARD, RIGHT, BACK = 0, 1, 2, 3
EGO_NAMES = ("left", "forward", "right", "back")

# 회전 delta_e: abs_dir = (heading + DELTA[e]) % 4
DELTA = (3, 0, 1, 2)
# 역변환: ego = _DELTA_INV[(abs_dir - heading) % 4]
_DELTA_INV = (1, 2, 3, 0)

# 절대 방향별 (행, 열) 변화량. 북은 행이 줄어든다.
STEP_RC = ((-1, 0), (0, 1), (1, 0), (0, -1))

# 시계방향 번호 -> pacman-ai 번호(상 0, 하 1, 좌 2, 우 3)
_TO_PACMAN_AI = (0, 3, 1, 2)
# pacman-ai 번호 -> 시계방향 번호
_FROM_PACMAN_AI = (0, 2, 3, 1)

_DELTA_ARR = np.asarray(DELTA, dtype=np.int64)
_DELTA_INV_ARR = np.asarray(_DELTA_INV, dtype=np.int64)
_TO_PAC_ARR = np.asarray(_TO_PACMAN_AI, dtype=np.int64)
_FROM_PAC_ARR = np.asarray(_FROM_PACMAN_AI, dtype=np.int64)


def ego_to_abs(heading, ego):
    """자기중심 행동 e를 절대 방향으로. 배열도 받는다 (넘파이 방송)."""
    return (np.asarray(heading) + _DELTA_ARR[np.asarray(ego)]) % 4


def abs_to_ego(heading, abs_dir):
    """절대 방향을 자기중심 번호로 (ego_to_abs의 역)."""
    return _DELTA_INV_ARR[(np.asarray(abs_dir) - np.asarray(heading)) % 4]


def opposite(abs_dir):
    return (np.asarray(abs_dir) + 2) % 4


def to_pacman_ai(abs_dir):
    """시계방향 번호 -> pacman-ai 방향 번호."""
    return _TO_PAC_ARR[np.asarray(abs_dir)]


def from_pacman_ai(pac_dir):
    """pacman-ai 방향 번호 -> 시계방향 번호."""
    return _FROM_PAC_ARR[np.asarray(pac_dir)]
