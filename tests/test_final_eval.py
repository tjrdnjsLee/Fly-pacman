"""최종 평가 도구 시험 (DECISIONS 26.3 · 26.4). torch · 체크포인트는 쓰지 않는다."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
_spec = importlib.util.spec_from_file_location("final_eval", ROOT / "scripts" / "final_eval.py")
fe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fe)

BASE = yaml.safe_load((ROOT / "configs" / "final_eval_v1.yaml").read_text(encoding="utf-8"))
INTEG = yaml.safe_load((ROOT / "configs" / "integ_v1.yaml").read_text(encoding="utf-8"))
SEEDS = list(range(8000, 8010))


def train_dir(root, name, condition, seeds, games=(100,), n_games=100):
    d = root / name
    (d / "weights").mkdir(parents=True)
    (d / "config.yaml").write_text(yaml.safe_dump({"condition": condition, "n_games": n_games}), encoding="utf-8")
    for s in seeds:
        for g in games:
            (d / "weights" / f"s{s}_g{g:05d}.npz").write_bytes(b"x")
    return d


def test_config_blocks_match_integ_and_values():
    for k in ("engine", "time", "odor"):
        assert BASE[k] == INTEG[k], k
    assert BASE["phero"] == {"D": 4.0, "lam": 0.05, "gain": 0.5922}   # DECISIONS 29.5 (integ_v1은 자리표시)
    for name in ("main_v1", "m3_curve_v1"):
        other = yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text(encoding="utf-8"))
        assert other["phero"] == BASE["phero"], name
    # pacman은 checkpoint_sha256 키 하나만 더 있다 (DECISIONS 28.3). run은 판 수 × watchdog_ticks 이상 (28.2)
    assert {k: v for k, v in BASE["pacman"].items() if k != "checkpoint_sha256"} == INTEG["pacman"]
    assert len(BASE["pacman"]["checkpoint_sha256"]) == 64
    assert BASE["run"]["max_loop_ticks"] >= BASE["n_games"] * BASE["engine"]["watchdog_ticks"]
    assert BASE["final_eval"]["train_n_games"] == 3000 and BASE["final_eval"]["censored_warn_fraction"] == 0.05
    assert BASE["seeds"] == list(range(9000, 9010)) and BASE["n_games"] == 200
    assert BASE["final_eval"]["train_seeds"] == SEEDS and BASE["final_eval"]["ci_level"] == 0.95
    assert BASE["final_eval"]["conditions"] == ["regime1", "regime2", "regime3", "b0", "b1"]


def test_eval_config_fills_per_condition_fields():
    c = fe.eval_config(BASE, "b1")
    assert c.condition == "b1" and c.ghosts.kind == "original" and c.evaluate and c.experiment == "final_eval_v1_b1"
    c = fe.eval_config(BASE, "regime2")
    assert c.ghosts.kind == "flybrain" and c.evaluate and c.experiment == "final_eval_v1_regime2"


def test_collect_picks_last_game_file_across_folders(tmp_path):
    a = train_dir(tmp_path, "a", "regime1", SEEDS[:5], games=(50, 100))   # 한 조건이 시드를 나눠 여러 폴더로
    b = train_dir(tmp_path, "b", "regime1", SEEDS[5:], games=(50, 100))
    got = fe.collect_weights([a, b], ["regime1", "b0", "b1"], SEEDS)
    assert len(got) == 10 and all(v["game_no"] == 100 for v in got.values())
    assert got[("regime1", 8007)]["path"].parent.parent.name == "b"


def test_collect_stops_on_missing_duplicate_or_unfinished(tmp_path):
    a = train_dir(tmp_path, "a", "regime1", SEEDS[:9])
    with pytest.raises(fe.EvalSetupError, match="8009"):
        fe.collect_weights([a], ["regime1"], SEEDS)                      # 빠짐
    b = train_dir(tmp_path, "b", "regime1", SEEDS[8:])
    with pytest.raises(fe.EvalSetupError, match="8008"):
        fe.collect_weights([a, b], ["regime1"], SEEDS)                   # 겹침
    c = train_dir(tmp_path, "c", "regime2", SEEDS, games=(50,), n_games=100)
    with pytest.raises(fe.EvalSetupError, match="끝까지"):
        fe.collect_weights([c], ["regime2"], SEEDS)                      # 마지막 판이 아님
    fe.collect_weights([c], ["b0", "b1"], SEEDS)                         # B0 · B1은 파일이 필요 없다


def test_run_stops_before_anything_when_seed_files_missing(tmp_path, capsys, monkeypatch):
    for v in fe.THREAD_VARS:
        monkeypatch.setenv(v, "1")
    a = train_dir(tmp_path, "a", "regime1", SEEDS[:9])
    cfg = tmp_path / "final_eval.yaml"
    cfg.write_text(yaml.safe_dump(BASE, allow_unicode=True), encoding="utf-8")
    assert fe.run(cfg, [a]) == 1
    assert "멈춤" in capsys.readouterr().err


def test_paired_ci_hand_computed():
    x, y = [3.0, 6.0, 9.0, 12.0], [1.0, 2.0, 3.0, 4.0]    # Δ = 2, 4, 6, 8
    ci = fe.paired_ci(x, y, 0.95)
    se = (20 / 3) ** 0.5 / 2
    assert ci["deltas"] == [2, 4, 6, 8] and ci["df"] == 3 and ci["mean"] == 5.0
    assert ci["se"] == pytest.approx(se) and ci["t"] == pytest.approx(3.182446305)
    assert ci["lo"] == pytest.approx(5 - 3.182446305 * se) and ci["hi"] == pytest.approx(5 + 3.182446305 * se)
    assert ci["excludes_zero"]
    assert not fe.paired_ci([1.0, -1.0, 2.0, -2.0], [0.0] * 4, 0.95)["excludes_zero"]


def games(path, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["조건", "시드", "판", "점수", "승자", "틱 수"])
        w.writerows(rows)


def test_per_seed_metrics_and_report(tmp_path):
    seeds = [9000, 9001]
    bundle = tmp_path / "final_eval_v1"
    bundle.mkdir()
    cfg = {"seeds": seeds, "final_eval": {"ci_level": 0.95, "censored_warn_fraction": 0.05}}
    data = {  # 조건: 시드별 (점수, 승자) 두 판씩
        "regime1": {9000: [(100, "FLIES"), (300, "PACMAN")], 9001: [(200, "FLIES"), (200, "FLIES")]},
        "b0": {9000: [(500, "PACMAN"), (700, "PACMAN")], 9001: [(300, "FLIES"), (500, "PACMAN")]},
    }
    runs = []
    for cond, per in data.items():
        d = tmp_path / f"x_{cond}"
        d.mkdir()
        (d / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
        games(d / "games.csv", [[cond, s, i + 1, sc, w, 400] for s, lst in per.items() for i, (sc, w) in enumerate(lst)])
        runs.append({"condition": cond, "run_dir": d.name})
    (bundle / "index.json").write_text(json.dumps({"runs": runs}), encoding="utf-8")

    m = fe.per_seed_metrics(fe.read_games(tmp_path / "x_regime1" / "games.csv"), seeds)
    assert m["score"] == [200.0, 200.0] and m["W"] == [0.5, 1.0] and m["games"] == [2, 2]
    assert fe.report(bundle) == 0
    res = json.loads((bundle / "results.json").read_text(encoding="utf-8"))
    assert res["metrics"]["b0"]["score"] == [600.0, 400.0] and res["metrics"]["b0"]["W"] == [0.0, 0.5]
    (c,) = [r for r in res["comparisons"] if (r["a"], r["b"]) == ("regime1", "b0")]
    assert c["score"]["deltas"] == [-400.0, -200.0] and c["score"]["mean"] == -300.0 and c["kind"] == "학습 효과"
    assert c["W"]["deltas"] == [0.5, 0.5] and c["W"]["sd"] == 0.0
    md = (bundle / "report.md").read_text(encoding="utf-8")
    assert "다중 비교 보정은 하지 않았다" in md and "regime1 − b0" in md


# ---------------------------------------------------------------- 28.4 시작 전 확인

def test_threads_must_all_be_one_and_message_gives_command():
    ok = {v: "1" for v in fe.THREAD_VARS}
    fe.check_threads(ok)
    for v in fe.THREAD_VARS:
        bad = dict(ok, **{v: "4"})
        with pytest.raises(fe.EvalSetupError, match=v) as e:
            fe.check_threads(bad)
        assert "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run" in str(e.value)
    with pytest.raises(fe.EvalSetupError):
        fe.check_threads({})                                      # 설정되지 않은 것도 멈춘다


def test_run_stops_when_threads_not_set(tmp_path, capsys, monkeypatch):
    for v in fe.THREAD_VARS:
        monkeypatch.delenv(v, raising=False)
    cfg = tmp_path / "final_eval.yaml"
    cfg.write_text(yaml.safe_dump(BASE, allow_unicode=True), encoding="utf-8")
    assert fe.run(cfg, [tmp_path]) == 1
    assert "OMP_NUM_THREADS=1" in capsys.readouterr().err


def real_train_dir(root, name, cfg_over=None, seed_in_file=None, file_hash=None):
    """config.yaml은 평가 설정의 공통 블록 + 학습 판 수. 가중치 파일은 진짜 npz (시드 · 실행 설정 해시만 든다)."""
    import numpy as np

    cfg = {k: BASE[k] for k in ("engine", "time", "odor", "phero", "pacman")}
    cfg.update(condition="regime1", n_games=3000, seeds=SEEDS[:2], weights={"every_games": 100},
               reward={"p": 1 / 3, "q": 1.0})
    cfg.update(cfg_over or {})
    d = root / name
    (d / "weights").mkdir(parents=True)
    (d / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    h = file_hash or fe.recorder_config_hash(yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8")))
    for s in SEEDS[:2]:
        np.savez(d / "weights" / f"s{s}_g03000.npz", dw_masked=np.zeros(3), rbar=np.zeros(1),
                 seed=np.int64(seed_in_file if seed_in_file is not None else s), run_config_hash=np.str_(h))
    return d


def test_check_train_dirs_passes_when_everything_matches(tmp_path):
    fe.check_train_dirs(BASE, [real_train_dir(tmp_path, "ok")])


def test_check_train_dirs_stops_on_n_games_block_seed_and_hash(tmp_path):
    with pytest.raises(fe.EvalSetupError, match="학습 판 수"):
        fe.check_train_dirs(BASE, [real_train_dir(tmp_path, "a", {"n_games": 300})])
    other = dict(BASE["odor"], D=2.0)
    with pytest.raises(fe.EvalSetupError, match="odor"):
        fe.check_train_dirs(BASE, [real_train_dir(tmp_path, "b", {"odor": other})])
    with pytest.raises(fe.EvalSetupError, match="시드"):
        fe.check_train_dirs(BASE, [real_train_dir(tmp_path, "c", seed_in_file=1)])
    with pytest.raises(fe.EvalSetupError, match="재현되지 않는다"):
        fe.check_train_dirs(BASE, [real_train_dir(tmp_path, "d", file_hash="0" * 16)])


def test_check_train_dirs_stops_when_files_have_different_hashes(tmp_path):
    import numpy as np

    d = real_train_dir(tmp_path, "e")
    np.savez(d / "weights" / "s8001_g03000.npz", dw_masked=np.zeros(3), rbar=np.zeros(1), seed=np.int64(8001),
             run_config_hash=np.str_("f" * 16))
    with pytest.raises(fe.EvalSetupError, match="하나가 아니다"):
        fe.check_train_dirs(BASE, [d])


def test_recorder_hash_ignores_weights_block():
    cfg = {"a": 1, "b": {"c": [1, 2]}}
    assert fe.recorder_config_hash(cfg) == fe.recorder_config_hash({**cfg, "weights": {"every_games": 100}})
    assert fe.recorder_config_hash(cfg) != fe.recorder_config_hash({**cfg, "a": 2})


# ---------------------------------------------------------------- 28.4 중단 판 제외

def test_censored_games_are_excluded_from_score_and_w(tmp_path):
    rows = [{"시드": "9000", "승자": w, "점수": str(s), "틱 수": "400"} for w, s in
            [("FLIES", 100), ("PACMAN", 300), ("CENSORED", 9000)]]
    rows += [{"시드": "9001", "승자": "CENSORED", "점수": "500", "틱 수": "400"},
             {"시드": "9001", "승자": "FLIES", "점수": "200", "틱 수": "400"}]
    m = fe.per_seed_metrics(rows, [9000, 9001])
    assert m["score"] == [200.0, 200.0]            # 중단 판의 9000 · 500점은 들어가지 않는다
    assert m["W"] == [0.5, 1.0] and m["censored"] == [1, 1] and m["games"] == [3, 2]
    only_censored = fe.per_seed_metrics([{"시드": "9000", "승자": "CENSORED", "점수": "1", "틱 수": "1"}], [9000])
    assert only_censored["score"] == [None] and only_censored["W"] == [None]


def make_bundle(tmp_path, censored_per_seed):
    seeds = [9000, 9001]
    bundle = tmp_path / "final_eval_v1"
    bundle.mkdir(parents=True)
    cfg = {"seeds": seeds, "final_eval": {"ci_level": 0.95, "censored_warn_fraction": 0.05}}
    runs = []
    for cond in ("regime1", "b0"):
        d = tmp_path / f"x_{cond}"
        d.mkdir()
        (d / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
        rows = []
        for s in seeds:
            n_c = censored_per_seed if cond == "regime1" else 0
            rows += [[cond, s, i + 1, 100, "CENSORED" if i < n_c else "FLIES", 400] for i in range(20)]
        games(d / "games.csv", rows)
        runs.append({"condition": cond, "run_dir": d.name})
    (bundle / "index.json").write_text(json.dumps({"runs": runs}), encoding="utf-8")
    return bundle


def test_report_warns_only_when_censored_fraction_exceeds_limit(tmp_path):
    # 칸당 20판 중 1판 = 0.05: 넘지 않는다 (경계는 경고 아님). 2판 = 0.10: 경고
    b1 = make_bundle(tmp_path / "one", 1)
    assert fe.report(b1) == 0
    md = (b1 / "report.md").read_text(encoding="utf-8")
    assert "해석 주의" not in md and "## 시드별 중단 수" in md
    b2 = make_bundle(tmp_path / "two", 2)
    assert fe.report(b2) == 0
    md = (b2 / "report.md").read_text(encoding="utf-8")
    assert md.splitlines()[2].startswith("> **해석 주의: 중단 비율.**") and "regime1 (0.100)" in md and "b0 (" not in md.split("\n\n")[1]
    res = json.loads((b2 / "results.json").read_text(encoding="utf-8"))
    assert res["metrics"]["regime1"]["censored"] == [2, 2] and res["summary"]["regime1"]["censored"] == 4
