"""본 실험 · M3 곡선 설정을 시험으로 고정한다 (DECISIONS 28.6, 28.1, 28.2, 28.3, 28.5)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from flypac.core.config import Config
from flypac.core.runner import check_loop_limit

ROOT = Path(__file__).resolve().parents[1]
CKPT_SHA = "4670fad239ee6324464b231cbe944afe8993167fbe20e1909fc23648b61e07f8"   # DECISIONS 14.6


def load(name):
    return yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text(encoding="utf-8"))


INTEG, MAIN, CURVE = load("integ_v1"), load("main_v1"), load("m3_curve_v1")
SHARED = ("engine", "time", "odor", "phero")


@pytest.mark.parametrize("cfg", [MAIN, CURVE], ids=["main_v1", "m3_curve_v1"])
def test_shared_blocks_match_integ(cfg):
    for k in SHARED:
        assert cfg[k] == INTEG[k], k
    assert cfg["pacman"]["checkpoint_sha256"] == CKPT_SHA
    assert {k: v for k, v in cfg["pacman"].items() if k != "checkpoint_sha256"} == INTEG["pacman"]
    assert cfg["ghosts"] == INTEG["ghosts"]
    assert cfg["weights"] == {"every_games": 100} and cfg["parallel"] == {"split": 1}
    check_loop_limit(Config(cfg))


def test_main_values_follow_decisions_27_and_28():
    assert MAIN["experiment"] == "main_v1" and MAIN["seeds"] == list(range(8000, 8010)) and MAIN["n_games"] == 3000
    assert MAIN["run"]["max_loop_ticks"] == 60_000_000 and MAIN["replay"]["brain"] is True


def test_main_replay_schedule_is_the_one_recorded_in_28_8():
    # 28.8: 추정 용량이 20 GB 이하라 판 1과 100의 배수 판 31개
    assert MAIN["replay"]["keep_games"] == [1, *range(100, 3001, 100)] and len(MAIN["replay"]["keep_games"]) == 31
    text = (ROOT / "docs" / "DECISIONS.md").read_text(encoding="utf-8")
    assert "판 1과 100의 배수 판 31개" in text.split("### 28.8")[1]


def test_curve_values_follow_decisions_28_1():
    assert CURVE["experiment"] == "m3_curve_v1" and CURVE["seeds"] == list(range(7200, 7210))
    assert CURVE["n_games"] == 300 and CURVE["run"]["max_loop_ticks"] == 6_000_000
    assert CURVE["replay"] == {"keep_games": [1, 100, 200, 300], "brain": True}
    assert not set(CURVE["seeds"]) & set(MAIN["seeds"])


def test_curve_differs_from_main_only_in_allowed_keys():
    allowed = {"experiment", "seeds", "n_games", "run", "replay"}
    diff = {k for k in set(MAIN) | set(CURVE) if MAIN.get(k) != CURVE.get(k)}
    assert diff == allowed
    assert MAIN["run"].keys() == CURVE["run"].keys()


def test_final_eval_shares_blocks_with_main():
    fe = load("final_eval_v1")
    for k in (*SHARED, "pacman"):
        assert fe[k] == MAIN[k], k
    assert fe["final_eval"]["train_seeds"] == MAIN["seeds"] and fe["final_eval"]["train_n_games"] == MAIN["n_games"]
    assert fe["run"]["max_loop_ticks"] == 4_000_000 and fe["replay"]["keep_games"] == [1]
    check_loop_limit(Config(fe))
