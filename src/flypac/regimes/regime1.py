"""보상 체제 1: 독점 (TRD 5.6, PRD 7.1). 체제 1 담당이 자기 파일의 두 함수만 쓴다."""

from __future__ import annotations

import numpy as np

from flypac.core.events import Event, Kind
from flypac.regimes.base import REWARD_DEFAULTS, Regime, fly_index


class Regime1(Regime):
    _reward_defaults = REWARD_DEFAULTS

    def rewards(self, events: list[Event]) -> np.ndarray:
        R = np.zeros(self.F)
        for ev in events:
            if not ev.active:  # 발생 시점 기준
                continue
            if ev.kind is Kind.CATCH:
                fly = fly_index(ev, self.F)
                R[:fly] -= self.p
                R[fly] += 1.0
                R[fly + 1:] -= self.p
            elif ev.kind is Kind.EATEN:
                R[fly_index(ev, self.F)] -= self.q
        return R

    def emission(self, obs_one_game) -> np.ndarray:
        return np.zeros(self.F)
