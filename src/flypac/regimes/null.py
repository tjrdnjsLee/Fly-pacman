"""보상 없는 체제. 기준선 b0, b1이 쓴다 (DECISIONS 7절).

기준선은 학습하지 않으므로 보상이 없다. 페로몬도 방출하지 않는다.
"""

from __future__ import annotations

import numpy as np

from flypac.core.events import Event
from flypac.regimes.base import N_FLIES, Regime


class NullRegime(Regime):
    def __init__(self, cfg):
        # 기준선 설정에는 reward.p, reward.q가 없을 수 있어 기반 클래스의 초기화를 거치지 않는다
        self.cfg = cfg
        self.F = N_FLIES

    def rewards(self, events: list[Event]) -> np.ndarray:
        return np.zeros(self.F)

    def emission(self, obs_one_game) -> np.ndarray:
        return np.zeros(self.F)
