"""유령 조종기 인터페이스 (TRD 5.5). 초파리 뇌, B0, B1, 무작위 초파리가 같은 모양을 가진다.

구현은 brain/flybrain.py(뇌 담당)와 ghosts/ 에 둔다. B: 게임 수, F: 초파리 수.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from flypac.core.types import FlyObs, GameView


class GhostController(ABC):
    @abstractmethod
    def propose(self, obs: FlyObs, alive: np.ndarray, views: list[GameView] | None = None) -> np.ndarray:
        """제안: 자기중심 행동과 그 확률 π를 계산해 보관. 상태는 바꾸지 않음.

        alive (B,) bool. 동결된 게임(alive=거짓)은 -1. 반환 (B,F) int.
        """
        raise NotImplementedError

    @abstractmethod
    def commit(self, executed: np.ndarray, active: np.ndarray) -> None:
        """확정: 실제로 실행된 결정만 학습 갱신과 흔적에 반영 (TRD 10.4). executed, active: (B,F) bool."""
        raise NotImplementedError

    @abstractmethod
    def give_reward(self, R: np.ndarray) -> None:
        """R (B,F) 누적 → 다음 결정 때 학습."""
        raise NotImplementedError

    @abstractmethod
    def on_release(self, mask: np.ndarray) -> None:
        """활성 시작: e = 0, R_n = 0. x, h는 유지. mask (B,) bool: 게임 단위 (ALL_RELEASED가 난 게임, TRD 4.5).

        학습 활성은 게임 전체에 대해 시작한다. ALL_RELEASED의 fly는 기록용이며 초기화 대상이 아니다.
        """
        raise NotImplementedError

    @abstractmethod
    def on_eaten(self, mask: np.ndarray) -> None:
        """먹힌 초파리의 x = 0, h = 0. e는 유지. mask (B,F) bool."""
        raise NotImplementedError

    @abstractmethod
    def flush_learning(self, mask: np.ndarray) -> None:
        """종료 직전 남은 보상을 기존 흔적에 한 번 반영. 새 결정은 만들지 않음 (TRD 6.2). mask (B,) bool."""
        raise NotImplementedError

    @abstractmethod
    def reset_life(self, mask: np.ndarray) -> None:
        """증거 · 흔적 · 누적 보상 초기화. mask (B,) bool."""
        raise NotImplementedError

    @abstractmethod
    def state_dict(self) -> dict:
        """가중치 · R̄ · 난수 상태."""
        raise NotImplementedError

    @abstractmethod
    def load_state_dict(self, state: dict) -> None:
        raise NotImplementedError
