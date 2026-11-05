"""원작 맵 (TRD 5.1). 자리 파일: 본문은 3단계에서 채운다. 이름·인자·반환 모양은 바꾸지 않는다."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MapSpec:
    """TRD 5.1의 MapSpec 필드. N: 걸을 수 있는 칸 수(유령 집 포함), 원작 기준 320."""

    grid: np.ndarray  # (31, 28) 격자
    cell_id: np.ndarray  # (31, 28) 칸 번호표, 벽은 -1 (TRD 4.2)
    cell_rc: np.ndarray  # (N, 2) 칸 번호 -> (행, 열)
    nbr: np.ndarray  # (N, 4) 이웃표, 벽은 -1, 터널은 연결로 포함 (TRD 4.4)
    move_mask: dict  # 캐릭터별 이동 마스크 {"pacman": (N,4) bool, "fly": (N,4) bool}
    tunnel_cells: np.ndarray  # 터널 칸 번호
    house_cells: np.ndarray  # 유령 집 칸 번호
    door_cells: np.ndarray  # 유령 집 문 칸 번호
    pacman_start: int
    fly_starts: np.ndarray  # (4,) 초파리 시작 칸
    fruit_cell: int
    dots: np.ndarray  # (N,) bool 점 배치
    energizers: np.ndarray  # (N,) bool 파워 쿠키 배치


def load_original() -> MapSpec:
    """원작 1레벨 맵 (TRD 5.1)."""
    raise NotImplementedError
