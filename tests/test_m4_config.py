"""M4 최종 통합 검증의 두 설정을 시험으로 고정한다 (DECISIONS 34.2)."""

from __future__ import annotations

from pathlib import Path

import yaml

from flypac.core.config import Config
from flypac.core.runner import check_loop_limit

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text(encoding="utf-8"))


MAIN, FINAL, CURVE = load("main_v1"), load("final_eval_v1"), load("m3_curve_v1")
CHECK, EVAL = load("m4_check_v1"), load("m4_eval_v1")


def diff_keys(a: dict, b: dict) -> set:
    return {k for k in set(a) | set(b) if a.get(k) != b.get(k)}


def test_m4_check_differs_from_main_only_in_allowed_keys():
    assert diff_keys(MAIN, CHECK) == {"experiment", "seeds", "n_games", "run", "replay"}
    assert MAIN["run"].keys() == CHECK["run"].keys() and MAIN["replay"].keys() == CHECK["replay"].keys()


def test_m4_check_values_follow_decisions_34_2():
    assert CHECK["experiment"] == "m4_check_v1" and CHECK["seeds"] == list(range(7300, 7310)) and CHECK["n_games"] == 20
    assert CHECK["run"] == {"max_loop_ticks": 400_020}                       # 20 × (20000 + 1), 33.3
    assert CHECK["replay"] == {"keep_games": [1, 20], "brain": True}
    assert CHECK["weights"] == {"every_games": 100} and CHECK["parallel"] == {"split": 1}   # main_v1 그대로 (26.1, 28.6)
    check_loop_limit(Config(CHECK))


def test_m4_eval_differs_from_final_eval_only_in_allowed_keys():
    assert diff_keys(FINAL, EVAL) == {"experiment", "seeds", "n_games", "run", "final_eval"}
    assert diff_keys(FINAL["final_eval"], EVAL["final_eval"]) == {"train_seeds", "train_n_games"}
    assert FINAL["run"].keys() == EVAL["run"].keys()


def test_m4_eval_values_follow_decisions_34_2_and_pair_with_m4_check():
    assert EVAL["experiment"] == "m4_eval_v1" and EVAL["seeds"] == list(range(7400, 7410)) and EVAL["n_games"] == 5
    assert EVAL["run"] == {"max_loop_ticks": 100_005}                        # 5 × (20000 + 1), 33.3
    assert EVAL["replay"] == FINAL["replay"] == {"keep_games": [1], "brain": True}
    assert EVAL["final_eval"]["train_seeds"] == CHECK["seeds"] and EVAL["final_eval"]["train_n_games"] == CHECK["n_games"]
    for k in ("engine", "time", "odor", "phero", "pacman"):                  # 평가가 학습 폴더와 대조하는 블록 (28.4)
        assert EVAL[k] == CHECK[k], k
    check_loop_limit(Config(EVAL))


def test_m4_seeds_are_used_nowhere_else():
    m4 = set(CHECK["seeds"]) | set(EVAL["seeds"])
    assert len(m4) == 20                                                     # 학습 · 평가 시드끼리도 겹치지 않는다
    others = set(MAIN["seeds"]) | set(FINAL["seeds"]) | set(FINAL["final_eval"]["train_seeds"]) | set(CURVE["seeds"])
    assert not m4 & others
    for path in sorted((ROOT / "configs").glob("*.yaml")):                   # 그 밖의 어느 설정도 M4 시드를 쓰지 않는다
        if path.stem in ("m4_check_v1", "m4_eval_v1"):
            continue
        seeds = yaml.safe_load(path.read_text(encoding="utf-8")).get("seeds") or []
        assert not m4 & {s for s in seeds if isinstance(s, int)}, path.name
