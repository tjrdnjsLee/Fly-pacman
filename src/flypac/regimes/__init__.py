from flypac.regimes.null import NullRegime
from flypac.regimes.regime1 import Regime1
from flypac.regimes.regime2 import Regime2
from flypac.regimes.regime3 import Regime3

_REGIMES = {"regime1": Regime1, "regime2": Regime2, "regime3": Regime3}
_BASELINES = ("b0", "b1")


def make_regime(cfg):
    """설정의 condition에서 보상 체제를 만든다. 기준선 b0, b1은 학습하지 않아 NullRegime(보상 0, 방출 0).

    b0p는 없앴다 (DECISIONS 10.4: 기준선 빼기로 B0와 같은 정책, 본 실험 5조건). 실수로 6조건이 되지 않게 오류다.
    """
    condition = str(cfg.condition)
    if condition == "b0p":
        raise ValueError("조건 b0p는 없앴다 (DECISIONS 10.4, 본 실험 5조건: regime1 · 2 · 3, b0, b1)")
    if condition in _BASELINES:
        return NullRegime(cfg)
    if condition in _REGIMES:
        return _REGIMES[condition](cfg)
    raise ValueError(f"알 수 없는 condition: {condition!r}")
