"""엔진 인터페이스 (TRD 5.2). 자리 파일: 진짜 엔진은 M1(팩맨 담당)이 채운다.

엔진은 스스로 초기화하지 않는다. reset_life, reset_game은 실행기만 부른다.
"""

from __future__ import annotations

import numpy as np

from flypac.core.events import Event
from flypac.core.types import GameView
from flypac.maps.original import MapSpec


class Game:
    """한 판 (TRD 5.2)."""

    def __init__(self, map: MapSpec, cfg, rng: np.random.Generator):
        raise NotImplementedError

    def step_pacman(self, action: int) -> list[Event]:
        """팩맨 이동, 먹기, 충돌. 목숨이 끝나면 그 틱 동안 동결. action: 절대 방향."""
        raise NotImplementedError

    def step_flies(self, actions: np.ndarray) -> tuple[list[Event], np.ndarray, np.ndarray]:
        """동결이 아니면 초파리 1→4번 이동, 매번 충돌. actions (F,) 자기중심 행동, -1은 유일한 경로.

        반환: (events, executed (F,) bool, active (F,) bool). 도중 종료 시 나머지는 실행하지 않음.
        """
        raise NotImplementedError

    def tick_timers(self) -> list[Event]:
        """동결이 아니면 파워 · 과일 · 유령 집 · 엘로이."""
        raise NotImplementedError

    def view(self) -> GameView:
        """현재 상태의 읽기 전용 사본."""
        raise NotImplementedError

    def decision_info(self) -> tuple[np.ndarray, np.ndarray]:
        """(legal (F,4) bool, decide (F,) bool)."""
        raise NotImplementedError

    def reset_life(self) -> None:
        raise NotImplementedError

    def reset_game(self) -> None:
        raise NotImplementedError


class VecGame:
    """Game B개 묶음. 자동 초기화 없음 (TRD 5.2). 각 메서드는 Game의 같은 이름 메서드를 B개에 적용한다."""

    def __init__(self, B: int, map: MapSpec, cfg, rngs: list[np.random.Generator]):
        raise NotImplementedError

    def step_pacman(self, actions: np.ndarray) -> list[list[Event]]:
        """actions (B,) 절대 방향."""
        raise NotImplementedError

    def step_flies(self, actions: np.ndarray) -> tuple[list[list[Event]], np.ndarray, np.ndarray]:
        """actions (B,F). 반환: (events 게임별 목록, executed (B,F), active (B,F))."""
        raise NotImplementedError

    def tick_timers(self) -> list[list[Event]]:
        raise NotImplementedError

    def views(self) -> list[GameView]:
        raise NotImplementedError

    def decision_info(self) -> tuple[np.ndarray, np.ndarray]:
        """(legal (B,F,4) bool, decide (B,F) bool)."""
        raise NotImplementedError

    def reset_life(self, mask: np.ndarray) -> None:
        """mask (B,) bool: 초기화할 게임."""
        raise NotImplementedError

    def reset_game(self, mask: np.ndarray) -> None:
        raise NotImplementedError
