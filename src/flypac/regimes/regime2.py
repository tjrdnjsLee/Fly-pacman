"""보상 체제 2: 공유 (TRD 5.6, PRD 7.1). 체제 2 담당이 자기 파일의 두 함수만 쓴다."""

from __future__ import annotations

import numpy as np

from flypac.core.events import Event, Kind
from flypac.regimes.base import Regime


class Regime2(Regime):
    def rewards(self, events: list[Event]) -> np.ndarray:
        R = np.zeros(self.F)
        for ev in events:
            if not ev.active:  # 발생 시점 기준
                continue
            if ev.kind is Kind.CATCH:
                R[:] += 1 / self.F  # 잡은 놈도 나머지도 +1/F = +1/4 (팀 합 1)
            elif ev.kind is Kind.EATEN:
                R[:] -= self.q / self.F  # 처벌도 나눈다 (q/F = q/4)
        return R

    def emission(self, obs_one_game) -> np.ndarray:
        return np.zeros(self.F)
