from pathlib import Path

import pytest

from flypac.core.config import Config, load_config
from flypac.core.runner import build_runner
from flypac.recording.recorder import Recorder
from flypac.regimes import make_regime
from flypac.regimes.regime3 import Regime3

SMOKE_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "m0_smoke.yaml"


def test_regime3_builds_from_smoke_config_with_only_condition_changed():
    data = load_config(SMOKE_CONFIG).to_dict()
    data["condition"] = "regime3"

    runner = build_runner(Config(data))

    assert isinstance(runner.regime, Regime3)
    assert runner.regime.p == 1 / 3


@pytest.mark.parametrize("reward, expected_q", [({}, 1.0), ({"q": 0.625}, 0.625)])
def test_regimes_use_same_q_for_same_reward_settings(reward, expected_q):
    for condition in ("regime1", "regime2", "regime3"):
        regime = make_regime(Config({"condition": condition, "reward": reward}))
        assert regime.q == expected_q, condition


@pytest.mark.parametrize("reward, expected_q", [({}, 1.0), ({"q": 0.625}, 0.625)])
def test_regime3_records_effective_q_in_result_config(tmp_path, reward, expected_q):
    cfg = Config({"condition": "regime3", "seeds": [0], "reward": reward})
    run_dir = tmp_path / "run"
    recorder = Recorder(run_dir, cfg)
    try:
        saved = load_config(run_dir / "config.yaml")
        assert saved.reward.q == recorder.regime.q == expected_q
    finally:
        recorder.close()
