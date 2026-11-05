"""보상 체제 3: 공유+소통 (TRD 5.6, PRD 7.1). 체제 3 담당이 자기 파일의 두 함수만 쓴다."""

from __future__ import annotations

import numpy as np

from flypac.core.events import Event, Kind
from flypac.regimes.base import Regime


class Regime3(Regime):
    def rewards(self, events: list[Event]) -> np.ndarray:
        R = np.zeros(self.F)
        for ev in events:
            if not ev.active:  # 발생 시점 기준
                continue
            if ev.kind is Kind.CATCH:
                R[:] += 0.25  # 잡은 놈도 나머지도 +1/4 (팀 합 1)
            elif ev.kind is Kind.EATEN:
                R[:] -= self.q / 4  # 처벌도 나눈다
        return R

    def emission(self, obs_one_game) -> np.ndarray:
        # TODO(미결 5): 체제3 페로몬 방출 규칙은 11/13(M2)까지 확정. 그 전에는 방출 없음.
        return np.zeros(self.F)
