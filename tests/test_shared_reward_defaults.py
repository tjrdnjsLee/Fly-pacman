from pathlib import Path

import numpy as np
import pytest

from flypac.core.config import Config, load_config
from flypac.core.events import Event, Kind
from flypac.core.runner import build_runner
from flypac.recording.recorder import Recorder
from flypac.regimes import make_regime
from flypac.regimes.regime3 import Regime3

SMOKE_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "m0_smoke.yaml"
# 체제 3이 요구하는 키 (DECISIONS 29). 보상에는 쓰이지 않는다
EMISSION = {"odor": {"c0": 0.01}, "phero": {"gain": 0.5922}}


def test_regime3_builds_from_smoke_config_with_only_condition_changed():
    data = load_config(SMOKE_CONFIG).to_dict()
    data["condition"] = "regime3"
    data["phero"]["gain"] = EMISSION["phero"]["gain"]  # m0_smoke의 phero는 자리표시라 gain이 없다 (DECISIONS 29.5)

    runner = build_runner(Config(data))

    assert isinstance(runner.regime, Regime3)
    assert runner.regime.p == 1 / 3


@pytest.mark.parametrize("reward, expected_q", [({}, 1.0), ({"q": 0.625}, 0.625)])
def test_regimes_use_same_q_for_same_reward_settings(reward, expected_q):
    for condition in ("regime1", "regime2", "regime3"):
        regime = make_regime(Config({"condition": condition, "reward": reward, **EMISSION}))
        assert regime.q == expected_q, condition


@pytest.mark.parametrize("reward, expected_q", [({}, 1.0), ({"q": 0.625}, 0.625)])
def test_regime3_records_effective_rewards_in_result_config(tmp_path, reward, expected_q):
    cfg = Config({"condition": "regime3", "seeds": [0], "reward": reward, **EMISSION})
    run_dir = tmp_path / "run"
    recorder = Recorder(run_dir, cfg)
    try:
        saved = load_config(run_dir / "config.yaml")
        assert saved.reward.p == recorder.regime.p == 1 / 3
        assert saved.reward.q == recorder.regime.q == expected_q
    finally:
        recorder.close()


@pytest.mark.parametrize("reward", [{}, {"q": 0.625}], ids=["default-q", "configured-q"])
@pytest.mark.parametrize(
    "event_specs",
    [
        pytest.param([(Kind.CATCH, True)], id="active-catch"),
        pytest.param([(Kind.CATCH, False)], id="inactive-catch"),
        pytest.param([(Kind.EATEN, True)], id="active-eaten"),
        pytest.param([(Kind.EATEN, False)], id="inactive-eaten"),
        pytest.param(
            [(Kind.EATEN, True), (Kind.CATCH, False), (Kind.CATCH, True), (Kind.EATEN, False)],
            id="mixed",
        ),
    ],
)
def test_regime2_and_regime3_rewards_match(reward, event_specs):
    regime2 = make_regime(Config({"condition": "regime2", "reward": reward}))
    regime3 = make_regime(Config({"condition": "regime3", "reward": reward, **EMISSION}))
    events = [
        Event(kind, game=0, tick=5, fly=fly, active=active)
        for fly, (kind, active) in enumerate(event_specs)
    ]

    np.testing.assert_array_equal(regime3.rewards(events), regime2.rewards(events))
