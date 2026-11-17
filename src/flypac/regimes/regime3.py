"""보상 체제 3: 공유+소통 (TRD 5.6, PRD 7.1, 7.4). 방출 규칙은 DECISIONS 29.

방출 선택 `phero.emission` (DECISIONS 41.3): `relay`(기본, 키가 없을 때도. 29.2) 또는 `marker`(39.3의 위치 표지:
네 마리 모두 매 틱 세기 1, 집 안 포함, 냄새 · z와 무관). `marker`이면 `phero.gain`은 읽지 않는다."""

from __future__ import annotations

import numpy as np

from flypac.core.events import Event, Kind
from flypac.regimes.base import Regime


def _required(cfg, section: str, key: str) -> float:
    """설정의 section.key를 읽는다. 없으면 키 이름을 담은 오류 (조용한 기본값을 두지 않는다)."""
    block = getattr(cfg, section, None)
    if block is None or key not in block:
        raise ValueError(f"체제 3에는 설정 {section}.{key}가 필요하다 (DECISIONS 29)")
    return float(block[key])


EMISSIONS = ("relay", "marker")


class Regime3(Regime):
    def __init__(self, cfg):
        super().__init__(cfg)
        block = getattr(cfg, "phero", None)
        self.emission_kind = str(block["emission"]) if block is not None and "emission" in block else "relay"
        if self.emission_kind not in EMISSIONS:
            raise ValueError(f"phero.emission은 {EMISSIONS} 중 하나다: {self.emission_kind!r} (DECISIONS 41.3)")
        if self.emission_kind == "relay":
            self.gain = _required(cfg, "phero", "gain")  # κ (29.2)
        self.c0 = _required(cfg, "odor", "c0")       # 실행기가 sense에 넘기는 값과 같은 키

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
        """E_k = κ ĉ_k 1[z_k = 0], ĉ_k = c0 (exp(max_a o_{k,a}) − 1) (DECISIONS 29.2).

        냄새 관측과 z만 본다. 페로몬 관측과 난수는 쓰지 않는다.
        `marker`이면 관측과 무관하게 ones(F) (DECISIONS 39.3 · 41.3).
        """
        if self.emission_kind == "marker":
            return np.ones(np.asarray(obs_one_game.z).shape[0])
        c_hat = self.c0 * np.expm1(np.max(np.asarray(obs_one_game.odor, dtype=float), axis=1))
        return self.gain * c_hat * (np.asarray(obs_one_game.z) == 0)
