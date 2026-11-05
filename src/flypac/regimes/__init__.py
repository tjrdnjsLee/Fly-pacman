from flypac.regimes.null import NullRegime
from flypac.regimes.regime1 import Regime1
from flypac.regimes.regime2 import Regime2
from flypac.regimes.regime3 import Regime3

_REGIMES = {"regime1": Regime1, "regime2": Regime2, "regime3": Regime3}
_BASELINES = ("b0", "b0p", "b1")


def make_regime(cfg):
    """설정의 condition에서 보상 체제를 만든다. 기준선 b0, b0p, b1은 학습하지 않아 NullRegime(보상 0, 방출 0)."""
    condition = str(cfg.condition)
    if condition in _BASELINES:
        return NullRegime(cfg)
    if condition in _REGIMES:
        return _REGIMES[condition](cfg)
    raise ValueError(f"알 수 없는 condition: {condition!r}")
