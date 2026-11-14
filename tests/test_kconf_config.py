"""본능 세기의 학습, 확인 실행의 설정 (DECISIONS 38.5 · 38.9).

- kconf_v1은 kdev_v1에서 experiment · seeds만, kconf_eval_v1은 kdev_eval_v1에서 experiment · seeds ·
  final_eval.train_seeds · final_eval.conditions만 다르다.
- 확인 시드(학습 8100 ~ 8119, 평가 9400 ~ 9419)는 다른 설정에 없다.
- 개발 설정과 뇌 설정의 Config.hash()는 이 PR 전(main 573e5d7)과 같다.
- 두 설정이 무한 루프 한도 검사(check_loop_limit, DECISIONS 33.3)를 통과한다.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from flypac.core.config import load_config
from flypac.core.runner import check_loop_limit

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
TRAIN_SEEDS = list(range(8100, 8120))
EVAL_SEEDS = list(range(9400, 9420))
NEW = ("kconf_v1", "kconf_eval_v1")

# 이 PR 전의 main(573e5d7)에서 잰 Config.hash()
OLD_CONFIG_HASH = {"brain_k_v1": "0d1ed15c6b6b5a28", "kdev_v1": "f9a866e1a853773b", "kdev_eval_v1": "85c51a96f1947ce3"}


def load(name: str) -> dict:
    return yaml.safe_load((CONFIGS / f"{name}.yaml").read_text(encoding="utf-8"))


def diff_keys(a: dict, b: dict) -> set:
    return {k for k in set(a) | set(b) if a.get(k) != b.get(k)}


def test_kconf_differs_from_kdev_only_in_allowed_keys():
    dev, conf = load("kdev_v1"), load("kconf_v1")
    assert diff_keys(dev, conf) == {"experiment", "seeds"}
    assert conf["experiment"] == "kconf_v1" and conf["seeds"] == TRAIN_SEEDS
    assert conf["n_games"] == 3000 and conf["ghosts"] == {"kind": "flybrain", "config": "configs/brain_k_v1.yaml"}


def test_kconf_eval_differs_from_kdev_eval_only_in_allowed_keys():
    dev, conf = load("kdev_eval_v1"), load("kconf_eval_v1")
    assert diff_keys(dev, conf) == {"experiment", "seeds", "final_eval"}
    assert diff_keys(dev["final_eval"], conf["final_eval"]) == {"train_seeds", "conditions"}
    assert conf["experiment"] == "kconf_eval_v1" and conf["seeds"] == EVAL_SEEDS and conf["n_games"] == 200
    fe = conf["final_eval"]
    assert fe["train_seeds"] == TRAIN_SEEDS and fe["conditions"] == ["regime1", "b0"]


def test_kconf_eval_matches_kconf_training():
    train, ev = load("kconf_v1"), load("kconf_eval_v1")
    fe = ev["final_eval"]
    assert fe["train_seeds"] == train["seeds"] and fe["train_n_games"] == train["n_games"]
    assert len(fe["train_seeds"]) == len(ev["seeds"]) == 20                # 위치로 짝 (26.4), 자유도 19 (38.9)
    for k in ("engine", "time", "odor", "phero", "pacman"):               # 학습과 같아야 하는 블록 (28.4)
        assert ev[k] == train[k], k


def test_confirm_seeds_are_used_nowhere_else():
    confirm = set(TRAIN_SEEDS) | set(EVAL_SEEDS)
    for path in sorted(CONFIGS.glob("*.yaml")):
        if path.stem in NEW:
            continue
        d = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        used = list(d.get("seeds") or []) + list((d.get("final_eval") or {}).get("train_seeds") or [])
        assert not confirm & {s for s in used if isinstance(s, int)}, path.name


def test_dev_and_brain_config_hashes_unchanged():
    for name, want in OLD_CONFIG_HASH.items():
        assert load_config(CONFIGS / f"{name}.yaml").hash() == want, name


def test_kconf_configs_pass_loop_limit():
    for name in NEW:
        check_loop_limit(load_config(CONFIGS / f"{name}.yaml"))
