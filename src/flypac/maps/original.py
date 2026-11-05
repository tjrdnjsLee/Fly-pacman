"""원작 1레벨 맵 (TRD 5.1). 미로 데이터는 pacman-ai의 maze_data.py에서 가져왔다 (NOTICE, _maze_data.py)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flypac.core.directions import STEP_RC
from flypac.maps import _maze_data as md


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


# 칸 종류 (_maze_data.py의 부호)
WALL, EMPTY, PELLET, POWER_PELLET, HOUSE, DOOR, TUNNEL = range(7)


def load_original() -> MapSpec:
    """원작 1레벨 맵 (TRD 5.1). 걸을 수 있는 칸은 320개 (유령 집 포함)."""
    grid = np.array(md.CLASSIC_MAZE, dtype=np.int8)
    rows, cols = grid.shape
    walk = grid != WALL
    cell_id = np.full(grid.shape, -1, dtype=np.int32)
    cell_rc = np.argwhere(walk)  # 행 우선 순서
    cell_id[walk] = np.arange(len(cell_rc))
    n = len(cell_rc)

    nbr = np.full((n, 4), -1, dtype=np.int32)
    for d, (dr, dc) in enumerate(STEP_RC):
        for i, (r, c) in enumerate(cell_rc):
            nr, nc = r + dr, (c + dc) % cols  # 열은 감싸서 터널 연결
            if 0 <= nr < rows:
                nbr[i, d] = cell_id[nr, nc]

    tile = grid[walk]
    in_house = (tile == HOUSE) | (tile == DOOR)
    pac_mask = (nbr >= 0) & ~in_house[np.clip(nbr, 0, None)]  # 팩맨은 유령 집과 문에 못 들어간다
    fly_mask = nbr >= 0

    def cid(rc):
        return int(cell_id[rc])

    return MapSpec(
        grid=grid,
        cell_id=cell_id,
        cell_rc=cell_rc,
        nbr=nbr,
        move_mask={"pacman": pac_mask, "fly": fly_mask},
        tunnel_cells=np.flatnonzero(tile == TUNNEL),
        house_cells=np.flatnonzero(tile == HOUSE),
        door_cells=np.flatnonzero(tile == DOOR),
        pacman_start=cid(md.PACMAN_START),
        # 1번 초파리는 집 밖에서 시작 (엘로이 대상), 나머지는 집 안
        fly_starts=np.array([cid(md.GHOST_START_POSITIONS[k]) for k in range(4)], dtype=np.int32),
        fruit_cell=cid(md.FRUIT_POSITION),
        dots=tile == PELLET,
        energizers=tile == POWER_PELLET,
    )
