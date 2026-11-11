"""ε 선정 도구 시험 (DECISIONS 24): 판정 함수, report 표, 설정 일치. torch · 체크포인트는 쓰지 않는다."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("select_eps", ROOT / "scripts" / "select_eps.py")
se = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(se)

SEL = yaml.safe_load((ROOT / "configs" / "eps_select_v1.yaml").read_text(encoding="utf-8"))["eps_select"]


def st(W, catch=2.0, cens=0.0):
    return {"W": W, "active_catches_per_game": catch, "censored_fraction": cens}


def dec(a, b, c):
    return se.decide({0.0: a, 0.1: b, 0.2: c}, SEL)


def test_smallest_in_band_wins():
    r = dec(st(0.3), st(0.2), st(0.4))
    assert (r["action"], r["eps"], r["branch"]) == ("select", 0.0, "24.3")
    r = dec(st(0.7), st(0.3), st(0.2))
    assert r["eps"] == 0.1


def test_all_above_ceiling():
    r = dec(st(0.9), st(0.8), st(0.6))
    assert r["eps"] == 0.0 and "천장 위험" in r["flags"]


def test_all_below_floor():
    r = dec(st(0.01), st(0.02), st(0.05))
    assert r["eps"] == 0.2 and any("관리자 보고" in f for f in r["flags"])


def test_mixed_stops():
    r = dec(st(0.9), st(0.05), st(0.05))
    assert r["action"] == "stop" and r["eps"] is None


def test_catch_failure_only_stops():
    r = dec(st(0.3, catch=0.5), st(0.3, catch=0.9), st(0.3, catch=0.2))
    assert r["action"] == "stop"


def test_censored_over_limit_stops_even_if_selectable():
    r = dec(st(0.3), st(0.3, cens=0.06), st(0.3))
    assert r["action"] == "stop" and "중단" in r["branch"]
    assert dec(st(0.3, cens=0.05), st(0.3), st(0.3))["action"] == "select"  # 0.05는 통과


def test_boundaries_inclusive():
    assert dec(st(0.10), st(0.9), st(0.9))["eps"] == 0.0
    assert dec(st(0.50), st(0.9), st(0.9))["eps"] == 0.0
    assert dec(st(0.3, catch=1.0), st(0.9), st(0.9))["eps"] == 0.0
    assert dec(st(0.3, catch=0.999), st(0.9), st(0.9))["action"] == "stop"


def test_undefined_w_stops():
    assert dec(st(None), st(0.3), st(0.3))["action"] == "stop"


def test_run_config_fills_per_run_fields():
    base = yaml.safe_load((ROOT / "configs" / "eps_select_v1.yaml").read_text(encoding="utf-8"))
    cfg = se.run_config(base, "b1", 0.1)
    assert cfg.condition == "b1" and cfg.ghosts.kind == "original" and cfg.pacman.eps == 0.1
    assert cfg.experiment == "eps_select_v1_b1_eps0.1"
    assert base["pacman"]["eps"] == 0.0  # 원본은 그대로


COLS = ["조건", "시드", "판", "점수", "승자", "틱 수", "누적 틱", "누적 결정 수", "남은 점",
        "포획 수", "활성 후 포획 수", "먹힘 수", "활성 후 먹힘 수"]


def write_games(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLS)
        for seed, winner, score, ticks, caught, eaten in rows:
            w.writerow(["b0", seed, 1, score, winner, ticks, 0, 0, 0, caught, caught, eaten, eaten])


def test_summarize_hand_made():
    rows = [
        {"시드": "1", "승자": "FLIES", "점수": "100", "틱 수": "400", "활성 후 포획 수": "2", "활성 후 먹힘 수": "1"},
        {"시드": "1", "승자": "PACMAN", "점수": "300", "틱 수": "500", "활성 후 포획 수": "0", "활성 후 먹힘 수": "3"},
        {"시드": "2", "승자": "PACMAN", "점수": "200", "틱 수": "600", "활성 후 포획 수": "1", "활성 후 먹힘 수": "2"},
        {"시드": "2", "승자": "CENSORED", "점수": "0", "틱 수": "900", "활성 후 포획 수": "1", "활성 후 먹힘 수": "0"},
    ]
    s = se.summarize(rows)
    assert (s["games"], s["fly_wins"], s["pacman_wins"], s["censored"]) == (4, 1, 2, 1)
    assert s["censored_fraction"] == 0.25 and s["W"] == pytest.approx(1 / 3)
    assert (s["seed_W_min"], s["seed_W_max"]) == (0.0, 0.5)  # 시드 1: 1/2, 시드 2: 0/1
    assert s["active_catches_per_game"] == 1.0 and s["active_eaten_per_game"] == 1.5
    assert s["pacman_mean_score"] == 150.0 and s["game_length_median"] == 550.0


def test_report_reads_games_csv(tmp_path):
    bundle = tmp_path / "eps_select_v1"
    bundle.mkdir()
    cfg = yaml.safe_load((ROOT / "configs" / "eps_select_v1.yaml").read_text(encoding="utf-8"))
    runs = []
    for cond in ("b0", "b1"):
        for eps in (0.0, 0.1, 0.2):
            name = f"x_{cond}_eps{eps}"
            d = tmp_path / name
            d.mkdir()
            (d / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
            (d / "environment.json").write_text(json.dumps({"pacman_device": "cuda"}), encoding="utf-8")
            (d / "git_commit.txt").write_text("abc123\n", encoding="utf-8")
            # b0: W = 3/4 (ε=0), 2/4 (ε=0.1), 1/4 (0.2). 판당 포획 2
            n_fly = {0.0: 3, 0.1: 2, 0.2: 1}[eps]
            rows = [(7000, "FLIES" if i < n_fly else "PACMAN", 100, 400, 2, 1) for i in range(4)]
            write_games(d / "games.csv", rows)
            runs.append({"condition": cond, "eps": eps, "run_dir": name})
    (bundle / "index.json").write_text(json.dumps({"runs": runs}), encoding="utf-8")
    assert se.report(bundle) == 0
    res = json.loads((bundle / "results.json").read_text(encoding="utf-8"))
    assert res["selection"]["eps"] == 0.1 and res["selection"]["branch"] == "24.3"  # W=0.5는 경계 포함
    b0 = [r for r in res["runs"] if r["condition"] == "b0"]
    assert [r["stats"]["W"] for r in b0] == [0.75, 0.5, 0.25]
    md = (bundle / "report.md").read_text(encoding="utf-8")
    assert "ε = 0.1" in md and "abc123" in md and "24.3" in md


def test_engine_time_odor_blocks_match_integ():
    a = yaml.safe_load((ROOT / "configs" / "eps_select_v1.yaml").read_text(encoding="utf-8"))
    b = yaml.safe_load((ROOT / "configs" / "integ_v1.yaml").read_text(encoding="utf-8"))
    for k in ("engine", "time", "odor", "phero", "run"):
        assert a[k] == b[k], k
    assert a["pacman"]["device"] == b["pacman"]["device"] == "cuda"
