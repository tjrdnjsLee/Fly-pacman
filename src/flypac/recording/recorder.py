"""기록기 (TRD 5.8, 13.3, 19.5). 자리 파일."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from flypac.core.events import Event
from flypac.core.types import FlyObs, GameView


class Recorder:
    def __init__(self, run_dir: Path, cfg):
        """결과 폴더 생성, 설정 · 커밋 번호 저장."""
        raise NotImplementedError

    def on_events(self, game: int, events: list[Event], view: GameView) -> None:
        """사건 누적."""
        raise NotImplementedError

    def on_life_end(self, game: int, view: GameView, events: list[Event]) -> None:
        """초기화 전의 종료 상태로 lives.csv 한 줄 기록."""
        raise NotImplementedError

    def on_game_end(self, game: int, view: GameView, events: list[Event]) -> None:
        """초기화 전의 종료 상태로 games.csv, flies.csv 기록."""
        raise NotImplementedError

    def replay_frame(
        self,
        game: int,
        views: list[GameView],
        obs: FlyObs,
        odor: np.ndarray,
        phero: np.ndarray,
        brain: dict | None,
    ) -> None:
        """매 틱 현재 판의 재생 정보를 메모리에 쌓음 (TRD 19.4, 19.5)."""
        raise NotImplementedError

    def replay_end(self, game: int, keep: bool) -> None:
        """판이 끝나면 남길 판(keep)만 파일로 저장."""
        raise NotImplementedError
