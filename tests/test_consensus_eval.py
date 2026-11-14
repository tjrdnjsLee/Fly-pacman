"""합의 가중치 평가 시험 (DECISIONS 37). torch · 체크포인트는 쓰지 않는다.

진짜 엔진 + 규칙 팩맨, 학습 시드 0 · 1, 칸마다 2판으로 학습 묶음을 임시 폴더에 만들고, 평가 시드 20 · 21로
합의 묶음(`consensus_eval`의 preflight → execute, 체크포인트 · 장치 확인만 뺀다)과 개별 묶음(`final_eval.run`과 같은 순서)을
만든 뒤 report · compare가 끝까지 도는지 본다.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from helpers import BASE as HELPER_BASE

from flypac.core.config import Config
from flypac.core.runner import build_runner
from flypac.recording.recorder import Recorder
from flypac.recording.weights import read_weight_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ce = _load("consensus_eval")
fe = ce.fe                      # consensus_eval이 부르는 바로 그 final_eval (예외 클래스가 같다)
rp = _load("run_parallel")


def load(name):
    return yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text(encoding="utf-8"))


FINAL, CONS, INDIV = load("final_eval_v1"), load("consensus_eval_v1"), load("indiv_eval_v1")
CONDS = ["regime1", "regime2", "regime3", "b0"]
TRAIN_SEEDS, EVAL_SEEDS, N_GAMES = [0, 1], [20, 21], 2
FAKE_CKPT_SHA = "ab" * 32


# ---------------------------------------------------------------- 설정 (37.3, 1.1)

def diff_keys(a: dict, b: dict) -> set:
    return {k for k in set(a) | set(b) if a.get(k) != b.get(k)}


@pytest.mark.parametrize("name,cfg", [("consensus_eval_v1", CONS), ("indiv_eval_v1", INDIV)])
def test_configs_differ_from_final_eval_only_in_allowed_keys(name, cfg):
    assert diff_keys(FINAL, cfg) == {"experiment", "seeds", "final_eval"}
    assert diff_keys(FINAL["final_eval"], cfg["final_eval"]) == {"conditions"}
    assert cfg["experiment"] == name and cfg["seeds"] == list(range(9100, 9110))
    assert cfg["final_eval"]["conditions"] == CONDS
    assert cfg["final_eval"]["train_seeds"] == list(range(8000, 8010)) and cfg["final_eval"]["train_n_games"] == 3000


def test_seeds_9100_9109_are_used_nowhere_else():
    ours = set(range(9100, 9110))
    for path in sorted((ROOT / "configs").glob("*.yaml")):
        if path.stem in ("consensus_eval_v1", "indiv_eval_v1"):
            continue
        d = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        used = list(d.get("seeds") or []) + list((d.get("final_eval") or {}).get("train_seeds") or [])
        assert not ours & {s for s in used if isinstance(s, int)}, path.name


# ---------------------------------------------------------------- 합의 가중치 (순수)

def state(seed, dw, rbar, **over):
    st = {"dw_masked": np.asarray(dw, dtype=np.float64), "rbar": np.asarray(rbar, dtype=np.float64), "seed": seed,
          "game_no": 3000, "condition": "regime1", "format": 1, "circuit_sha256": "c" * 64,
          "brain_config_hash": "h" * 16, "run_config_hash": "r" * 16}
    st.update(over)
    return st


def test_consensus_is_the_seed_mean_per_fly(tmp_path):
    a = state(8000, [[1.0, -0.5, 0.0], [2.0, 4.0, 6.0]], [0.2, 0.4])          # 초파리 2마리 × 연결 3
    b = state(8001, [[3.0, 0.5, 1.0], [0.0, 0.0, 0.0]], [0.6, 0.0])
    for i, st in enumerate((a, b)):                                            # 손으로 만든 파일 둘
        np.savez(tmp_path / f"s{8000 + i}_g03000.npz", **{k: (np.str_(v) if isinstance(v, str) else v)
                                                          for k, v in st.items()})
    got = ce.consensus_state([read_weight_file(tmp_path / f"s{s}_g03000.npz") for s in (8000, 8001)], "regime1")
    np.testing.assert_array_equal(got["dw_masked"], [[2.0, 0.0, 0.5], [1.0, 2.0, 3.0]])
    np.testing.assert_allclose(got["rbar"], [0.4, 0.2])
    assert {k: got[k] for k in ce.META_KEYS} == {k: a[k] for k in ce.META_KEYS}
    h = ce.state_sha256(got)
    assert set(h) == {"dw_masked", "rbar"} and h != ce.state_sha256({**got, "rbar": got["rbar"] + 1})


@pytest.mark.parametrize("key,value", [("circuit_sha256", "d" * 64), ("brain_config_hash", "x" * 16),
                                       ("format", 2), ("condition", "regime2")])
def test_consensus_stops_when_source_meta_differs(key, value):
    a, b = state(8000, [[1.0]], [0.0]), state(8001, [[1.0]], [0.0], **{key: value})
    with pytest.raises(fe.EvalSetupError, match=key):
        ce.consensus_state([a, b], "regime1")


def test_consensus_stops_on_shape_or_condition_mismatch():
    with pytest.raises(fe.EvalSetupError, match="모양"):
        ce.consensus_state([state(8000, [[1.0]], [0.0]), state(8001, [[1.0, 2.0]], [0.0])], "regime1")
    with pytest.raises(fe.EvalSetupError, match="조건"):
        ce.consensus_state([state(8000, [[1.0]], [0.0])], "regime2")


# ---------------------------------------------------------------- 진짜 엔진 작은 묶음

def _small(cfg: dict, seeds: list[int]) -> dict:
    d = json.loads(json.dumps(cfg))
    d["seeds"], d["n_games"] = list(seeds), N_GAMES
    d["pacman"] = {"kind": "rule_based", "checkpoint_sha256": FAKE_CKPT_SHA}
    d["rule_based"] = dict(HELPER_BASE["rule_based"])
    d["run"] = {"max_loop_ticks": N_GAMES * (d["engine"]["watchdog_ticks"] + 1)}
    d["replay"] = {"keep_games": [1], "brain": True}
    return d


def _eval_small(cfg: dict) -> dict:
    d = _small(cfg, EVAL_SEEDS)
    d["final_eval"]["train_seeds"], d["final_eval"]["train_n_games"] = list(TRAIN_SEEDS), N_GAMES
    return d


def _indiv(base: dict, train_dirs: list[Path], runs_root: Path) -> Path:
    """개별 묶음: final_eval.run과 같은 순서 (체크포인트 · 장치 확인만 뺀다)."""
    fe_blk = base["final_eval"]
    weights = fe.collect_weights(train_dirs, CONDS, TRAIN_SEEDS, train_n_games=N_GAMES)
    fe.check_train_dirs(base, train_dirs)
    out = runs_root / base["experiment"]
    out.mkdir(parents=True)
    runs = []
    for cond in fe_blk["conditions"]:
        cfg = fe.eval_config(base, cond)
        run_dir = runs_root / f"x_{cfg.experiment}"
        runner = build_runner(cfg, Recorder(run_dir, cfg))
        if cond in fe.LEARNING:
            for i, s in enumerate(TRAIN_SEEDS):
                runner.ghosts.load_slot_state(i, read_weight_file(weights[(cond, s)]["path"]))
        runner.run()
        runs.append({"condition": cond, "run_dir": run_dir.name})
    (out / "index.json").write_text(json.dumps({"runs": runs}), encoding="utf-8")
    return out


@pytest.fixture(scope="module")
def bundles(tmp_path_factory):
    root = tmp_path_factory.mktemp("consensus")
    train = _small(load("main_v1"), TRAIN_SEEDS)
    T = root / "T"
    for job in rp.plan(train, CONDS, 1):
        cfg = Config(job["config"])
        build_runner(cfg, Recorder(T / job["name"], cfg)).run()
    train_dirs = [T / f"{c}_p1" for c in CONDS]
    R = root / "runs"
    mp = pytest.MonkeyPatch()
    for v in fe.THREAD_VARS:
        mp.setenv(v, "1")
    try:
        cons_base = _eval_small(CONS)
        weights = ce.preflight(cons_base, train_dirs, R)
        C = ce.execute(cons_base, weights, R)
        assert fe.report(C) == 0
        with pytest.raises(fe.EvalSetupError, match="이미 있다"):            # 묶음이 있으면 다시 돌지 않는다
            ce.preflight(cons_base, train_dirs, R)
    finally:
        mp.undo()
    indiv_base = _eval_small(INDIV)
    I = _indiv(indiv_base, train_dirs, R)
    assert fe.report(I) == 0
    return root, T, C, I, weights


def test_consensus_run_loads_the_mean_into_every_slot(bundles):
    root, T, C, I, weights = bundles
    index = json.loads((C / "index.json").read_text(encoding="utf-8"))
    assert [r["condition"] for r in index["runs"]] == CONDS
    for r in index["runs"]:
        d = C.parent / r["run_dir"]
        man = json.loads((d / "eval_manifest.json").read_text(encoding="utf-8"))
        if r["condition"] == "b0":
            assert man["sources"] == [] and man["consensus"] is None
            continue
        cond = r["condition"]
        assert [s["train_seed"] for s in man["sources"]] == TRAIN_SEEDS
        assert all(s["sha256"] == fe._sha256(weights[(cond, s["train_seed"])]["path"]) for s in man["sources"])
        assert [f["eval_seed"] for f in man["files"]] == EVAL_SEEDS
        with np.load(d / man["consensus"]["file"]) as z:
            saved = {"dw_masked": z["dw_masked"], "rbar": z["rbar"]}
            assert list(z["train_seeds"]) == TRAIN_SEEDS and int(z["game_no"]) == N_GAMES
        src = [read_weight_file(weights[(cond, s)]["path"]) for s in TRAIN_SEEDS]
        np.testing.assert_array_equal(saved["dw_masked"], (src[0]["dw_masked"] + src[1]["dw_masked"]) / 2)
        np.testing.assert_array_equal(saved["rbar"], (src[0]["rbar"] + src[1]["rbar"]) / 2)
        assert man["consensus"]["sha256"] == ce.state_sha256(saved)
        assert (saved["dw_masked"] != 0).any()                                # 학습이 실제로 가중치를 바꿨다
    assert (C / "report.md").exists() and (C / "results.json").exists()


def test_compare_runs_end_to_end(bundles, capsys):
    root, T, C, I, _ = bundles
    assert ce.main(["consensus_eval.py", "compare", str(C), str(I)]) == 0
    res = json.loads((C / "compare.json").read_text(encoding="utf-8"))
    assert res["b0_identical"] and res["eval_seeds"] == EVAL_SEEDS
    assert [r["condition"] for r in res["comparisons"]] == ["regime1", "regime2", "regime3"]
    for r in res["comparisons"]:
        mc, mi = res["metrics"]["consensus"][r["condition"]], res["metrics"]["indiv"][r["condition"]]
        assert r["score"]["deltas"] == pytest.approx([a - b for a, b in zip(mc["score"], mi["score"])])
    assert "합의 regime1 − 개별 regime1" in (C / "compare.md").read_text(encoding="utf-8")
    assert "다중 비교 보정은 하지 않았다" in capsys.readouterr().out


@pytest.fixture
def copy(bundles, tmp_path):
    root, *_ = bundles
    dst = tmp_path / "c"
    shutil.copytree(root, dst)
    return dst / "runs" / "consensus_eval_v1", dst / "runs" / "indiv_eval_v1"


def _b0_csv(bundle: Path) -> Path:
    index = json.loads((bundle / "index.json").read_text(encoding="utf-8"))
    (r,) = [r for r in index["runs"] if r["condition"] == "b0"]
    return bundle.parent / r["run_dir"] / "games.csv"


def test_b0_check_ignores_row_order_but_stops_on_a_changed_row(copy, capsys):
    C, I = copy
    p = _b0_csv(I)
    lines = p.read_bytes().splitlines(keepends=True)
    p.write_bytes(lines[0] + b"".join(reversed(lines[1:])))                  # 행 순서만 바꾼다 → 통과
    assert ce.compare(C, I) == 0
    lines = p.read_bytes().splitlines(keepends=True)
    f = lines[1].decode("utf-8").split(",")
    f[3] = str(int(f[3]) + 10)                                                 # 점수 한 칸
    p.write_bytes(lines[0] + ",".join(f).encode("utf-8") + b"".join(lines[2:]))
    assert ce.main(["consensus_eval.py", "compare", str(C), str(I)]) == 1
    assert "b0 games.csv가 다르다" in capsys.readouterr().err


def test_compare_stops_when_seed_lists_differ(copy):
    C, I = copy
    index = json.loads((I / "index.json").read_text(encoding="utf-8"))
    for r in index["runs"]:                                                    # 개별 묶음의 시드 목록 순서를 바꾼다
        p = I.parent / r["run_dir"] / "config.yaml"
        d = yaml.safe_load(p.read_text(encoding="utf-8"))
        d["seeds"] = list(reversed(d["seeds"]))
        p.write_text(yaml.safe_dump(d, allow_unicode=True), encoding="utf-8")
    with pytest.raises(fe.EvalSetupError, match="seeds"):
        ce.build_compare(C, I)
