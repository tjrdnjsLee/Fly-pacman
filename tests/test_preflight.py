"""설정 둘(연기 시험용 · 본 학습용)과 시작 검사 (torch 없이 돈다)."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from flypac.core.config import Config, load_config
from flypac.core.rng import game_rngs
from flypac.pacman.preflight import PreflightError, check_problems, curriculum_phases, eval_opponent, require_ready
from flypac.pacman.train_env import GHOST_FACTORIES
from test_train_env_equiv import ViewsGhost

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def mod(cfg: Config, fn) -> Config:
    d = copy.deepcopy(cfg.to_dict())
    fn(d)
    return Config(d)


def smoke() -> Config:
    return load_config(CONFIGS / "pacman_train_smoke.yaml")


def test_smoke_config_is_fake_engine_random_ghosts_random_eval():
    cfg = smoke()
    assert cfg.engine.kind == "fake"
    phases = curriculum_phases(cfg.pacman_train)
    assert {g for _, g, _ in phases} == {"random"} and {c for _, _, c in phases} == {"regime1"}
    assert eval_opponent(cfg.pacman_train) == ("random", "regime1")
    assert check_problems(cfg) == []
    require_ready(cfg)


def test_production_config_uses_real_engine_and_original_ghosts():
    cfg = load_config(CONFIGS / "pacman_train_v1.yaml")
    assert cfg.engine.kind == "real" and cfg.engine.power_s == 6
    assert curriculum_phases(cfg.pacman_train) == [(0, "random", "regime1"), (800, "original", "b1")]
    assert eval_opponent(cfg.pacman_train) == ("original", "b1")
    assert "power_ticks_max" not in cfg.pacman_train.obs  # 파워 분모는 엔진 설정에서 온다


def test_missing_condition_is_an_error_not_inferred():
    cfg = mod(smoke(), lambda d: d["pacman_train"]["curriculum"][0].pop("condition"))
    with pytest.raises(ValueError, match="condition"):
        curriculum_phases(cfg.pacman_train)
    assert any("condition" in m for m in check_problems(cfg))
    cfg = mod(smoke(), lambda d: d["pacman_train"]["training"].pop("eval_opponent"))
    assert any("eval_opponent" in m for m in check_problems(cfg))


def test_all_missing_parts_are_reported_together_before_training():
    def bad(d):
        d["engine"]["kind"] = "bogus"
        d["pacman_train"]["curriculum"].append({"from_update": 15, "ghosts": "bogus", "condition": "regime1"})
        del d["engine"]["power_s"]

    cfg = mod(smoke(), bad)
    with pytest.raises(PreflightError) as e:
        require_ready(cfg)
    msg = str(e.value)
    assert "bogus" in msg and "엔진" in msg and "power_s" in msg and "단계 2" in msg
    assert msg.count("\n - ") >= 3  # 한꺼번에 모아서 알린다


def test_part_that_cannot_be_built_is_named(monkeypatch):
    def boom(cfg, B, seeds):
        raise NotImplementedError("B1 자리 파일")

    monkeypatch.setitem(GHOST_FACTORIES, "original", boom)
    cfg = mod(smoke(), lambda d: d["pacman_train"]["training"].update(eval_opponent={"ghosts": "original", "condition": "b1"}))
    problems = check_problems(cfg)
    assert any("평가 상대" in m and "original" in m and "NotImplementedError" in m for m in problems)


def test_condition_mismatch_is_reported(monkeypatch):
    monkeypatch.setitem(GHOST_FACTORIES, "original",
                        lambda cfg, B, seeds: ViewsGhost(cfg, B, game_rngs(seeds, "ghost")))
    ok = mod(smoke(), lambda d: d["pacman_train"]["curriculum"].append(
        {"from_update": 15, "ghosts": "original", "condition": "b1"}))
    assert check_problems(ok) == []  # 부품이 있고 조건이 맞으면 한 스텝까지 돌려 본다
    wrong = mod(smoke(), lambda d: d["pacman_train"]["curriculum"].append(
        {"from_update": 15, "ghosts": "original", "condition": "regime1"}))
    assert any("b1" in m and "단계 2" in m for m in check_problems(wrong))
