"""학습 가중치 저장과 평가 모드 (DECISIONS 26.1 · 26.3). 가짜 엔진 · 규칙 팩맨 · 작은 판 수로만 돈다."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from helpers import make_cfg

from flypac.brain.flybrain import FlyBrain, load_brain_config
from flypac.core.config import Config
from flypac.core.runner import build_runner
from flypac.recording.recorder import Recorder
from flypac.recording.weights import parse_weight_filename, read_weight_file, weight_filename

BRAIN = "configs/brain_v1.yaml"
GHOSTS = {"kind": "flybrain", "config": BRAIN}


def cfg_for(condition="regime1", seeds=(0, 1), n_games=5, weights=None, **over):
    extra = dict(over)
    if weights is not None:
        extra["weights"] = weights
    return make_cfg(condition=condition, seeds=list(seeds), n_games=n_games, ghosts=GHOSTS,
                    replay={"keep_games": [1], "brain": True}, **extra)


def run(tmp_path, name, cfg):
    rec = Recorder(tmp_path / name, cfg)
    runner = build_runner(cfg, rec)
    runner.run()
    return tmp_path / name, runner


def files(d: Path):
    return sorted(p.name for p in (d / "weights").glob("*.npz")) if (d / "weights").exists() else []


def test_interval_every_2_of_5_games_writes_2_4_5(tmp_path):
    d, _ = run(tmp_path, "r", cfg_for(weights={"every_games": 2}))
    want = sorted(weight_filename(s, g) for s in (0, 1) for g in (2, 4, 5))
    assert files(d) == want
    assert not list((d / "weights").glob("*.tmp"))


def test_last_game_file_equals_dw_right_after_flush_learning(tmp_path):
    cfg = cfg_for(weights={"every_games": 100})
    rec = Recorder(tmp_path / "r", cfg)
    runner = build_runner(cfg, rec)
    ghosts, snap = runner.ghosts, {}
    orig = ghosts.flush_learning

    def flush(mask):
        orig(mask)
        for b in np.flatnonzero(mask):
            snap[int(b)] = (ghosts.dw[b].copy(), ghosts.rbar[b].copy())

    ghosts.flush_learning = flush
    runner.run()
    for b, seed in enumerate((0, 1)):
        f = read_weight_file(tmp_path / "r" / "weights" / weight_filename(seed, 5))
        dw, rbar = snap[b]  # 마지막 판의 마지막 flush_learning 직후 (판의 마지막 보상이 들어 있다)
        assert np.any(dw != 0), "학습이 일어나지 않았다: 시험이 아무것도 확인하지 못한다"
        assert np.array_equal(f["dw_masked"], dw[:, ghosts.circuit.M]) and np.array_equal(f["rbar"], rbar)
        assert (f["seed"], f["game_no"], f["condition"], f["format"]) == (seed, 5, "regime1", 1)
        assert f["circuit_sha256"] == ghosts.circuit.source_hash and f["brain_config_hash"] == ghosts.config_hash
        assert f["run_config_hash"] == rec.config_hash
        assert f["dw_masked"].dtype == np.float64 and f["dw_masked"].shape == (4, int(ghosts.circuit.M.sum()))


def test_roundtrip_exact_and_zero_outside_mask(tmp_path):
    d, runner = run(tmp_path, "r", cfg_for(weights={"every_games": 100}))
    g = runner.ghosts
    fresh = FlyBrain(load_brain_config(BRAIN), 2, [0, 1], condition="regime1")
    fresh.load_slot_state(1, read_weight_file(d / "weights" / weight_filename(1, 5)))
    assert np.array_equal(fresh.dw[1], g.dw[1]) and np.array_equal(fresh.rbar[1], g.rbar[1])
    assert not fresh.dw[0].any() and not fresh.dw[..., ~g.circuit.M].any()  # 다른 칸은 그대로, M 밖은 0


@pytest.mark.parametrize("bad", ["format", "condition", "circuit_sha256", "brain_config_hash"])
def test_load_slot_state_rejects_mismatch(tmp_path, bad):
    d, _ = run(tmp_path, "r", cfg_for(weights={"every_games": 100}, n_games=1))
    st = read_weight_file(d / "weights" / weight_filename(0, 1))
    st[bad] = 2 if bad == "format" else "other"
    with pytest.raises(ValueError, match=bad):
        FlyBrain(load_brain_config(BRAIN), 2, [0, 1], condition="regime1").load_slot_state(0, st)


def test_outputs_identical_with_and_without_weights(tmp_path):
    a, _ = run(tmp_path, "off", cfg_for())
    b, _ = run(tmp_path, "on", cfg_for(weights={"every_games": 2}))
    assert files(a) == [] and files(b)
    for name in ("games.csv", "lives.csv", "flies.csv"):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
    ra, rb = sorted((a / "replay").glob("*")), sorted((b / "replay").glob("*"))
    assert ra and [p.name for p in ra] == [p.name for p in rb]
    for x, y in zip(ra, rb):
        assert x.read_bytes() == y.read_bytes(), x.name


def test_no_files_for_b0_and_b1(tmp_path):
    d, runner = run(tmp_path, "b0", cfg_for(condition="b0", n_games=2, weights={"every_games": 1}))
    assert runner.weight_writer is None and files(d) == []
    from test_original_ghosts import original_cfg

    cfg = original_cfg()
    cfg = Config({**cfg.to_dict(), "seeds": [0], "n_games": 1, "ghosts": {"kind": "original"},
                  "pacman": {"kind": "rule_based"}, "weights": {"every_games": 1}})
    d, runner = run(tmp_path, "b1", cfg)
    assert runner.weight_writer is None and files(d) == []


def test_weights_without_run_dir_is_an_error():
    with pytest.raises(ValueError, match="결과 폴더"):
        build_runner(cfg_for(weights={"every_games": 1}))


def test_filename_roundtrip():
    assert weight_filename(8003, 1000) == "s8003_g01000.npz"
    assert parse_weight_filename("s8003_g01000.npz") == (8003, 1000) and parse_weight_filename("x.npz") is None


# ---------------------------------------------------------------- 평가 모드

def test_evaluate_mode_keeps_dw_fixed_and_eta_zero(tmp_path):
    d, _ = run(tmp_path, "train", cfg_for(weights={"every_games": 100}, n_games=3))
    cfg = cfg_for(n_games=2, evaluate=True)
    runner = build_runner(cfg, Recorder(tmp_path / "ev", cfg))
    g = runner.ghosts
    assert g.evaluate and g.eta == 0.0
    for b, seed in enumerate((0, 1)):
        g.load_slot_state(b, read_weight_file(d / "weights" / weight_filename(seed, 3)))
    before = g.dw.copy()
    assert before.any()
    runner.run()
    assert np.array_equal(g.dw, before)  # 판을 돌려도 dw가 변하지 않는다


def test_evaluate_flag_makes_eta_zero_for_learning_condition():
    brain = load_brain_config(BRAIN)
    assert FlyBrain(brain, 1, [0], condition="regime1").eta == float(brain.learn.eta)
    assert FlyBrain(brain, 1, [0], condition="regime1", evaluate=True).eta == 0.0


def test_slot_i_receives_training_seed_s_i_file(tmp_path):
    """시드를 섞은 가짜 파일: 칸마다 서로 다른 dw를 넣으면 칸 i가 자기 파일만 갖는다."""
    brain = load_brain_config(BRAIN)
    g = FlyBrain(brain, 3, [10, 11, 12], condition="regime1", evaluate=True)
    n = int(g.circuit.M.sum())
    for b, val in ((2, 0.3), (0, 0.1), (1, 0.2)):  # 읽는 순서를 섞는다
        st = g.slot_state(b)
        st["dw_masked"] = np.full((4, n), val)
        st["rbar"] = np.full(4, val)
        g.load_slot_state(b, st)
    for b, val in enumerate((0.1, 0.2, 0.3)):
        assert (g.dw[b][:, g.circuit.M] == val).all() and (g.rbar[b] == val).all()


def test_zero_dw_regime1_eval_plays_same_games_as_b0(tmp_path):
    def play(condition, name):
        cfg = cfg_for(condition=condition, seeds=(0, 1, 2), n_games=3, evaluate=True)
        rec = Recorder(tmp_path / name, cfg)
        runner = build_runner(cfg, rec)
        if condition != "b0":
            for b in range(3):
                runner.ghosts.load_slot_state(b, FlyBrain(brain, 3, [0, 1, 2], condition=condition).slot_state(b))
        runner.run()
        import csv

        with open(tmp_path / name / "games.csv", encoding="utf-8-sig", newline="") as fh:
            return [[v for k, v in r.items() if k != "조건"] for r in csv.DictReader(fh)]

    brain = load_brain_config(BRAIN)
    a, b = play("regime1", "r1"), play("b0", "b0")
    assert a and a == b
