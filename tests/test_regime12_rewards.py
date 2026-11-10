import numpy as np
import pytest

from flypac.core.config import Config
from flypac.core.events import Event, Kind
from flypac.regimes import make_regime


@pytest.fixture(params=["regime1", "regime2"])
def condition(request):
    return request.param


def event(kind, fly, active=True):
    return Event(kind, game=0, tick=5, fly=fly, active=active)


@pytest.mark.parametrize(
    "condition, catch_reward, eaten_reward, catch_total",
    [
        ("regime1", [-1 / 3, -1 / 3, 1.0, -1 / 3], [0.0, -1.0, 0.0, 0.0], 0.0),
        ("regime2", [0.25] * 4, [-0.25] * 4, 1.0),
    ],
)
def test_default_reward_table_and_team_sums(condition, catch_reward, eaten_reward, catch_total):
    regime = make_regime(Config({"condition": condition}))

    catch = regime.rewards([event(Kind.CATCH, 2)])
    eaten = regime.rewards([event(Kind.EATEN, 1)])

    np.testing.assert_array_equal(catch, catch_reward)
    np.testing.assert_array_equal(eaten, eaten_reward)
    assert catch.sum() == pytest.approx(catch_total)
    assert eaten.sum() == pytest.approx(-1.0)


def test_same_tick_rewards_are_additive(condition):
    regime = make_regime(Config({"condition": condition}))
    catch = event(Kind.CATCH, 2)
    eaten = event(Kind.EATEN, 1)
    catch_reward = regime.rewards([catch])
    eaten_reward = regime.rewards([eaten])

    np.testing.assert_array_equal(regime.rewards([catch, eaten]), catch_reward + eaten_reward)
    np.testing.assert_array_equal(regime.rewards([eaten, catch]), catch_reward + eaten_reward)
    np.testing.assert_array_equal(regime.rewards([catch, catch]), 2 * catch_reward)


def test_inactive_events_and_emission_are_zero(condition):
    regime = make_regime(Config({"condition": condition}))
    inactive = [event(Kind.CATCH, 2, active=False), event(Kind.EATEN, 1, active=False)]

    np.testing.assert_array_equal(regime.rewards(inactive), np.zeros(4))
    np.testing.assert_array_equal(regime.emission(np.ones((4, 4))), np.zeros(4))


@pytest.mark.parametrize(
    "condition, catch_reward, eaten_reward",
    [
        ("regime1", [-0.4, -0.4, 1.0, -0.4], [0.0, -0.6, 0.0, 0.0]),
        ("regime2", [0.25] * 4, [-0.15] * 4),
    ],
)
def test_configured_reward_parameters(condition, catch_reward, eaten_reward):
    regime = make_regime(Config({"condition": condition, "reward": {"p": 0.4, "q": 0.6}}))

    np.testing.assert_array_equal(regime.rewards([event(Kind.CATCH, 2)]), catch_reward)
    np.testing.assert_array_equal(regime.rewards([event(Kind.EATEN, 1)]), eaten_reward)


@pytest.mark.parametrize("reward, p, q", [({"p": 0.4}, 0.4, 1.0), ({"q": 0.6}, 1 / 3, 0.6)])
def test_missing_reward_parameters_default_individually(condition, reward, p, q):
    regime = make_regime(Config({"condition": condition, "reward": reward}))

    assert (regime.p, regime.q) == (p, q)
