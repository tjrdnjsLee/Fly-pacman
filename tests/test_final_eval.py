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
    for k in ("engine", "time", "odor", "phero", "pacman", "run"):
        assert BASE[k] == INTEG[k], k
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


def test_run_stops_before_anything_when_seed_files_missing(tmp_path, capsys):
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
    cfg = {"seeds": seeds, "final_eval": {"ci_level": 0.95}}
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
