"""PPO 팩맨 관측 (TRD 12.2 관측 계약). 학습과 실험이 이 파일 하나를 같이 쓴다.

원 저장소(pacman-ai `env/vec_env.py`) 기준: 격자 채널 8개 × 프레임 4개 + 스칼라 5개.

| 채널 | 내용 |
|---|---|
| 0 | 벽 |
| 1 | 팩맨 위치 |
| 2 | 점 |
| 3 | 파워 쿠키 |
| 4 | 위험한 유령: 집 밖, z = 0 |
| 5 | 먹을 수 있는 유령: 집 밖, z = 1 |
| 6 | 유령 집과 문 |
| 7 | 과일 |

스칼라: 파워 남은 비율, 목숨 비율, 이번 파워에서 먹은 유령 수 / 4, 먹은 점 비율, 팩맨 방향 / 3.
방향 스칼라는 원 저장소 번호(상 0, 하 1, 좌 2, 우 3)로 바꿔 넣는다 (core/directions.py).

프레임 이력: 목숨 또는 판이 끝나면 지우고 첫 프레임으로 4칸을 모두 채운다 (실행기가 목숨 끝에
PacmanController.reset을 부르므로 학습 환경도 같은 시점에 지운다).

위험한 유령 판정(`dangerous_flies`)은 이 파일 하나에만 있다: 관측 채널 4 · 5와 학습 보상의 근접 항이 같은 함수를
쓴다 (DECISIONS 13.1절). 파워 분모와 목숨 분모는 `engine_constants`가 엔진 설정에서 가져온다.
"""

from __future__ import annotations

import numpy as np

from flypac.core import directions as d
from flypac.core.timing import config_ticks
from flypac.core.types import GameView
from flypac.maps.original import WALL, MapSpec

# 관측 규격을 바꾸면 올린다. 체크포인트에 저장하고, 불러올 때 다르면 멈춘다.
OBS_VERSION = 1
NUM_CHANNELS = 8
NUM_SCALARS = 5


def engine_constants(cfg) -> dict:
    """관측 정규화에 쓰는 엔진 숫자. 엔진 설정에서만 가져온다: 파워 분모 = 겁먹는 시간(틱), 목숨 분모 = 초기 목숨 수.

    체크포인트에 기록하고, 실험 시작 때 실험 설정의 값과 대조한다 (pacman/ppo.py, trainer.py).
    """
    try:
        power = config_ticks(cfg, cfg.engine.power_s)
        lives = int(cfg.engine.lives)
    except (AttributeError, KeyError) as e:
        raise ValueError(f"설정에 {e} 가 없다: engine.power_s(초), engine.lives, time.v_max_tiles_per_s가 필요하다") from None
    return {"power_ticks_max": max(power, 1), "lives": max(lives, 1)}


def dangerous_flies(view: GameView) -> np.ndarray:
    """(F,) bool: 집 밖이고 z = 0인 초파리 = 위험한 유령. 관측과 보상이 같이 쓰는 유일한 판정."""
    return ~np.asarray(view.fly_in_house, dtype=bool) & ~np.asarray(view.fly_z, dtype=bool)


def edible_flies(view: GameView) -> np.ndarray:
    """(F,) bool: 집 밖이고 z = 1인 초파리 = 먹을 수 있는 유령."""
    return ~np.asarray(view.fly_in_house, dtype=bool) & np.asarray(view.fly_z, dtype=bool)


class ObsEncoder:
    """GameView 묶음 → (격자 (B,8,H,W) float32, 스칼라 (B,5) float32). 프레임 이력은 갖지 않는다."""

    def __init__(self, map: MapSpec, power_ticks_max: int, lives_max: int):
        self.map = map
        self.H, self.W = map.grid.shape
        self.power_ticks_max = max(int(power_ticks_max), 1)
        self.lives_max = max(int(lives_max), 1)
        self.rows = map.cell_rc[:, 0]
        self.cols = map.cell_rc[:, 1]
        self.n_pellets = max(int(map.dots.sum() + map.energizers.sum()), 1)
        static = np.zeros((NUM_CHANNELS, self.H, self.W), dtype=np.float32)
        static[0] = map.grid == WALL
        house = np.concatenate([map.house_cells, map.door_cells])
        static[6, self.rows[house], self.cols[house]] = 1.0
        self.static = static

    def encode(self, views: list[GameView]) -> tuple[np.ndarray, np.ndarray]:
        B = len(views)
        grid = np.zeros((B, NUM_CHANNELS, self.H, self.W), dtype=np.float32)
        grid[:, 0] = self.static[0]
        grid[:, 6] = self.static[6]
        scalars = np.zeros((B, NUM_SCALARS), dtype=np.float32)
        r, c = self.rows, self.cols
        for b, v in enumerate(views):
            g = grid[b]
            g[1, r[v.pac_cell], c[v.pac_cell]] = 1.0
            g[2, r, c] = v.dots
            g[3, r, c] = v.energizers
            cells = np.asarray(v.fly_cell)
            danger, edible = cells[dangerous_flies(v)], cells[edible_flies(v)]
            g[4, r[danger], c[danger]] = 1.0
            g[5, r[edible], c[edible]] = 1.0
            if v.fruit:
                g[7, r[self.map.fruit_cell], c[self.map.fruit_cell]] = 1.0
            left = int(np.count_nonzero(v.dots) + np.count_nonzero(v.energizers))
            scalars[b] = (
                min(v.power_ticks / self.power_ticks_max, 1.0) if v.power else 0.0,
                v.lives / self.lives_max,
                v.ghost_streak / 4.0,
                (self.n_pellets - left) / self.n_pellets,
                int(d.to_pacman_ai(v.pac_dir)) / 3.0,
            )
        return grid, scalars


class FrameStack:
    """게임 칸별 최근 k 프레임. 반환 격자는 (B, k*8, H, W) uint8이고 오래된 프레임이 앞이다 (원 저장소 순서).

    격자 값은 0 · 1뿐이라 uint8로 담아도 정확히 같다. 신경망에 넣을 때 실수로 바꾼다 (ppo_agent.grid_tensor).
    """

    def __init__(self, B: int, k: int, shape: tuple[int, int, int]):
        self.k = int(k)
        self.buf = np.zeros((B, self.k, *shape), dtype=np.uint8)
        self.fresh = np.ones(B, dtype=bool)  # 다음 push에서 k칸을 모두 그 프레임으로 채운다

    def reset(self, mask: np.ndarray) -> None:
        self.fresh |= np.asarray(mask, dtype=bool)

    def push(self, frame: np.ndarray) -> np.ndarray:
        frame = frame.astype(np.uint8)
        self.buf[:, :-1] = self.buf[:, 1:]
        self.buf[:, -1] = frame
        if self.fresh.any():
            self.buf[self.fresh] = frame[self.fresh][:, None]
            self.fresh[:] = False
        B = self.buf.shape[0]
        return self.buf.reshape(B, -1, *self.buf.shape[3:]).copy()


def legal_mask(map: MapSpec, views: list[GameView], mask_reverse: bool) -> np.ndarray:
    """(B,4) bool, 절대 방향(시계방향) 순서. 벽 쪽은 막는다.

    mask_reverse: 원 저장소처럼 반전을 막는다 (다른 길이 있을 때만). 엔진 규칙이 아니라 학습용 행동 마스크다.
    """
    pm = map.move_mask["pacman"]
    out = np.zeros((len(views), 4), dtype=bool)
    for b, v in enumerate(views):
        m = pm[v.pac_cell].copy()
        if mask_reverse:
            back = int(d.opposite(v.pac_dir))
            if m[back] and m.sum() > 1:
                m[back] = False
        if not m.any():
            m[:] = True  # 갈 곳이 없으면 아무 행동이나 (엔진이 무시한다)
        out[b] = m
    return out


class PacmanObserver:
    """프레임 이력을 가진 관측기. 학습 환경과 PPOPacman이 같은 코드로 관측 · 프레임 이력 · 행동 마스크를 만든다.

    프레임 이력은 `reset(mask)`가 표시한 칸만 다음 `observe`에서 첫 프레임으로 다시 채운다.
    B를 주지 않으면 첫 `observe`의 views 수로 정한다.
    """

    def __init__(self, map: MapSpec, frame_stack: int, mask_reverse: bool, constants: dict, B: int | None = None):
        self.map = map
        self.k = int(frame_stack)
        self.mask_reverse = bool(mask_reverse)
        self.encoder = ObsEncoder(map, constants["power_ticks_max"], constants["lives"])
        self.frames: FrameStack | None = None
        if B is not None:
            self._make_frames(int(B))

    def _make_frames(self, B: int) -> None:
        self.frames = FrameStack(B, self.k, (NUM_CHANNELS, *self.map.grid.shape))

    def observe(self, views: list[GameView]) -> dict:
        if self.frames is None or self.frames.buf.shape[0] != len(views):
            self._make_frames(len(views))
        grid, scalars = self.encoder.encode(views)
        return {"grid": self.frames.push(grid), "scalars": scalars}

    def reset(self, mask: np.ndarray) -> None:
        if self.frames is not None:  # 아직 없으면 첫 observe가 모두 새로 채운다
            self.frames.reset(mask)

    def legal(self, views: list[GameView]) -> np.ndarray:
        """학습 · 탐욕 행동용 마스크 (mask_reverse면 반전 금지)."""
        return legal_mask(self.map, views, self.mask_reverse)

    def engine_legal(self, views: list[GameView]) -> np.ndarray:
        """엔진이 허용하는 방향 전체 (반전 포함). ε 무작위 행동이 여기서 고른다 (DECISIONS 13.2절)."""
        return legal_mask(self.map, views, False)

    def current(self) -> dict:
        """push 없이 지금 프레임 이력의 묶음 격자 (B, k*8, H, W) (체크포인트 복원용)."""
        buf = self.frames.buf
        return {"grid": buf.reshape(buf.shape[0], -1, *buf.shape[3:]).copy()}
