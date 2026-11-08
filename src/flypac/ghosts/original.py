"""기준선 B1: 원작 유령 규칙 (TRD 5.5). 자리 파일: 팩맨 담당. views를 사용한다."""

from __future__ import annotations

import numpy as np

from flypac.brain.base import GhostController
from flypac.core.rng import GameRngs
from flypac.core.types import FlyObs, GameView


class OriginalGhosts(GhostController):
    """원작 규칙의 절대 방향을 자기중심으로 변환해 돌려준다."""

    uses_views = True

    def __init__(self, cfg, B: int, rngs: GameRngs):
        """rngs: 칸별 생성기 묶음 (core.rng.game_rngs(seeds, "ghost")). 칸 b는 rngs[b]만 쓰고, 이 묶음을 `self.rngs`로
        둔다: 판 시작 때 TickCore가 (시드, 판 번호, 용도)로 다시 만든다 (DECISIONS 11절)."""
        raise NotImplementedError

    def propose(self, obs: FlyObs, alive: np.ndarray, views: list[GameView] | None = None) -> np.ndarray:
        raise NotImplementedError

    def commit(self, committed: np.ndarray, active: np.ndarray) -> None:
        raise NotImplementedError

    def give_reward(self, R: np.ndarray) -> None:
        raise NotImplementedError

    def on_release(self, mask: np.ndarray) -> None:
        raise NotImplementedError

    def on_eaten(self, mask: np.ndarray) -> None:
        raise NotImplementedError

    def flush_learning(self, mask: np.ndarray) -> None:
        raise NotImplementedError

    def reset_life(self, mask: np.ndarray) -> None:
        raise NotImplementedError

    def state_dict(self) -> dict:
        raise NotImplementedError

    def load_state_dict(self, state: dict) -> None:
        raise NotImplementedError
