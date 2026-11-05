from flypac.regimes.regime1 import Regime1
from flypac.regimes.regime2 import Regime2
from flypac.regimes.regime3 import Regime3


def make_regime(cfg):
    """설정의 condition에서 보상 체제를 만든다. b0, b0p, b1은 학습하지 않아 체제가 쓰이지 않는다.

    TODO: 기준선 조건의 처리. 지금은 Regime1을 대신 둔다.
    """
    return {"regime2": Regime2, "regime3": Regime3}.get(cfg.condition, Regime1)(cfg)
