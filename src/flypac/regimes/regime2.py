"""보상 체제 2 (TRD 5.6, 7.1, 11절). 자리 파일: 체제 2 담당이 자기 파일의 두 함수만 쓴다."""

from __future__ import annotations

import numpy as np

from flypac.core.events import Event
from flypac.regimes.base import Regime


class Regime2(Regime):
    def __init__(self, cfg):
        raise NotImplementedError

    def rewards(self, events: list[Event]) -> np.ndarray:
        raise NotImplementedError

    def emission(self, obs_one_game) -> np.ndarray:
        raise NotImplementedError
