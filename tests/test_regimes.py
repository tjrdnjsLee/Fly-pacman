import numpy as np
import pytest
from helpers import BASE, make_cfg

from flypac.core.config import Config
from flypac.core.events import Event, Kind
from flypac.regimes import make_regime
from flypac.regimes.null import NullRegime
from flypac.regimes.regime1 import Regime1
from flypac.regimes.regime2 import Regime2
from flypac.regimes.regime3 import Regime3

CFG = make_cfg()
P, Q = CFG.reward.p, CFG.reward.q


def catch(fly, active=True):
    return Event(Kind.CATCH, 0, 5, fly, active)


def eaten(fly, active=True):
    return Event(Kind.EATEN, 0, 5, fly, active)


def test_regime1_catch_and_team_sum():
    R = Regime1(CFG).rewards([catch(2)])
    assert R[2] == 1.0 and np.allclose(np.delete(R, 2), -P)
    assert np.isclose(R.sum(), 1 - 3 * P)


def test_regime1_eaten_only_victim():
    R = Regime1(CFG).rewards([eaten(1)])
    assert R.tolist() == [0.0, -Q, 0.0, 0.0]


@pytest.mark.parametrize("cls", [Regime2, Regime3])
def test_shared_regimes_sums(cls):
    r = cls(CFG)
    assert np.allclose(r.rewards([catch(0)]), 0.25) and np.isclose(r.rewards([catch(3)]).sum(), 1.0)
    assert np.allclose(r.rewards([eaten(2)]), -Q / 4)


@pytest.mark.parametrize("cls", [Regime1, Regime2, Regime3])
def test_inactive_events_ignored_and_emission_zero(cls):
    r = cls(CFG)
    assert not r.rewards([catch(0, False), eaten(1, False)]).any()
    assert not r.emission(None).any() and r.emission(None).shape == (4,)


def test_non_reward_events_do_nothing():
    evs = [Event(Kind.DOT, 0, 1, -1, True), Event(Kind.LIFE_LOST, 0, 1, -1, True)]
    for cls in (Regime1, Regime2, Regime3):
        assert not cls(CFG).rewards(evs).any()


def test_null_regime_has_no_reward_and_no_emission():
    r = NullRegime(CFG)
    assert not r.rewards([catch(0), eaten(1), catch(2, active=False)]).any()
    assert r.rewards([]).shape == (4,) and r.emission(None).shape == (4,) and not r.emission(None).any()


def test_null_regime_needs_no_reward_settings():
    data = {k: v for k, v in BASE.items() if k != "reward"}
    data["condition"] = "b0"
    assert isinstance(make_regime(Config(data)), NullRegime)


@pytest.mark.parametrize("condition", ["b0", "b0p", "b1"])
def test_baselines_use_null_regime(condition):
    assert isinstance(make_regime(make_cfg(condition=condition)), NullRegime)


@pytest.mark.parametrize("condition,cls", [("regime1", Regime1), ("regime2", Regime2), ("regime3", Regime3)])
def test_regime_conditions_map_to_their_regime(condition, cls):
    assert type(make_regime(make_cfg(condition=condition))) is cls


def test_unknown_condition_is_an_error():
    with pytest.raises(ValueError):
        make_regime(make_cfg(condition="regime9"))
