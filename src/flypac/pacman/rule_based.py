"""규칙 기반 팩맨 (TRD 5.7, 12.1). 자리 파일: 팀원 노트북용."""

from __future__ import annotations

import numpy as np

from flypac.core.types import GameView
from flypac.maps.original import MapSpec
from flypac.pacman.base import PacmanController


class RuleBasedPacman(PacmanController):
    def __init__(self, map: MapSpec, cfg, rng: np.random.Generator):
        raise NotImplementedError

    def act(self, views: list[GameView]) -> np.ndarray:
        raise NotImplementedError

    def reset(self, mask: np.ndarray) -> None:
        raise NotImplementedError
