"""동시 도착 측정 도구 시험 (DECISIONS 25.3 · 25.4). torch · 체크포인트는 쓰지 않는다."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from helpers import make_cfg

from flypac.core.events import Event, Kind
from flypac.core.runner import build_runner
from flypac.maps.original import load_original
from flypac.recording.recorder import Recorder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
_spec = importlib.util.spec_from_file_location("measure_ties", ROOT / "scripts" / "measure_ties.py")
mt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mt)

MAP = load_original()
TIES = yaml.safe_load((ROOT / "configs" / "ties_v1.yaml").read_text(encoding="utf-8"))["ties"]


def cell(r, c):
    return int(MAP.cell_id[r, c])


def view(pac, flies, z=(0, 0, 0, 0), house=(0, 0, 0, 0)):
    return SimpleNamespace(pac_cell=pac, fly_cell=np.array(flies), fly_z=np.array(z, dtype=bool),
                           fly_in_house=np.array(house, dtype=bool))


def ev(kind, fly=-1, active=True):
    return Event(kind, 0, 1, fly, active, None)


P = cell(1, 2)
FAR = cell(29, 1)


def test_classify_categories_and_house_exclusion():
    v = view(P, [P, P, cell(1, 3), FAR], z=(0, 1, 0, 0))
    got = mt.classify(v, 0, MAP.nbr)
    assert got == {"T1": [], "T2": [1], "T3": [2]}
    v = view(P, [P, P, P, P], z=(0, 0, 1, 0), house=(0, 0, 0, 1))
    assert mt.classify(v, 0, MAP.nbr) == {"T1": [1], "T2": [2], "T3": []}  # 집 안 3번 제외, j 제외
    v = view(P, [P, cell(1, 3), cell(1, 3), FAR], z=(0, 1, 0, 0))
    assert mt.classify(v, 0, MAP.nbr)["T3"] == [2]  # 겁먹은 이웃은 T3가 아니다


def test_count_life_active_and_inactive_and_non_catch():
    c = mt.new_counts()
    mt.count_life(c, view(P, [P, P, FAR, FAR]), [ev(Kind.CATCH, 0), ev(Kind.LIFE_LOST)], MAP.nbr)
    mt.count_life(c, view(P, [FAR, P, FAR, P], z=(0, 0, 0, 1)),
                  [ev(Kind.EATEN, 3), ev(Kind.CATCH, 1, active=False), ev(Kind.LIFE_LOST)], MAP.nbr)
    mt.count_life(c, view(P, [P, P, P, P]), [ev(Kind.LIFE_LOST)], MAP.nbr)  # CATCH 없음: 세지 않는다
    assert (c["catches"], c["active_catches"]) == (2, 1)
    assert c["events"]["T1"] == {"all": 1, "active": 1} and c["events"]["T2"] == {"all": 1, "active": 0}
    assert c["order_effect_active"] == 1 and c["caught_fly_active"] == [1, 0, 0, 0]
    assert c["pairs"]["T1"] == {"0,1": 1} and c["pairs_active"]["T2"] == {}
    assert (c["eaten_same_tick"], c["catches_with_eaten_same_tick"]) == (1, 1)


def counts_with(active, effect):
    c = mt.new_counts()
    c["active_catches"], c["order_effect_active"] = active, effect
    return c


def test_decide_ties_threshold_inclusive_and_stops():
    assert mt.decide_ties(counts_with(100, 2), TIES, True)["verdict"] == "가"   # 0.02 포함
    assert mt.decide_ties(counts_with(100, 3), TIES, True)["verdict"] == "나"
    assert mt.decide_ties(counts_with(100, 0), TIES, True)["verdict"] == "가"
    assert mt.decide_ties(counts_with(100, 0), TIES, False)["verdict"] == "stop"
    assert mt.decide_ties(counts_with(0, 0), TIES, True)["verdict"] == "stop"
    assert mt.decide_ties(counts_with(100, 3), {"order_effect_max_fraction": 0.03}, True)["verdict"] == "가"


def test_counting_recorder_does_not_change_outputs(tmp_path):
    cfg = make_cfg(n_games=2, seeds=[0, 1])
    build_runner(cfg, Recorder(tmp_path / "plain", cfg)).run()
    rec = mt.CountingRecorder(Recorder(tmp_path / "wrapped", cfg), MAP.nbr)
    build_runner(cfg, rec).run()
    for name in mt.CSV_FILES:
        assert (tmp_path / "plain" / name).read_bytes() == (tmp_path / "wrapped" / name).read_bytes(), name
    assert rec.counts["catches"] > 0  # 목숨이 끝난 포획을 실제로 셌다
    assert mt.compare_csvs(tmp_path / "plain", tmp_path / "wrapped")["lives.csv"] == {"same": True}


def test_compare_csvs_reports_first_diff_line(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    for n in mt.CSV_FILES:
        (a / n).write_bytes("h\nx\ny\n".encode("utf-8-sig"))
        (b / n).write_bytes("h\nx\ny\n".encode("utf-8-sig"))
    (b / "lives.csv").write_bytes("h\nx\nz\n".encode("utf-8-sig"))
    r = mt.compare_csvs(a, b)
    assert r["games.csv"]["same"] and r["lives.csv"] == {"same": False, "first_diff_line": 3}


def test_report_from_hand_made_bundle(tmp_path):
    bundle = tmp_path / "ties_v1"
    bundle.mkdir()
    cfg = yaml.safe_load((ROOT / "configs" / "eps_select_v1.yaml").read_text(encoding="utf-8"))
    runs, reg = [], {}
    for cond, effect in (("b0", 1), ("b1", 0)):
        name = f"x_{cond}"
        d = tmp_path / name
        d.mkdir()
        (d / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
        (d / "environment.json").write_text(json.dumps({"pacman_device": "cuda"}), encoding="utf-8")
        (d / "git_commit.txt").write_text("abc\n", encoding="utf-8")
        (d / "lives.csv").write_text("종료 원인,잡은 초파리,출발 전 포획 여부\n" + "포획,0,0\n" * 60 + "포획,2,0\n" * 40
                                     + "포획,1,1\n", encoding="utf-8-sig")
        c = counts_with(100, effect)
        c["caught_fly_active"] = [60, 0, 40, 0]
        (bundle / f"counts_{cond}.json").write_text(json.dumps(c), encoding="utf-8")
        runs.append({"condition": cond, "eps": 0.0, "run_dir": name})
        reg[cond] = {n: {"same": True} for n in mt.CSV_FILES}
    (bundle / "index.json").write_text(json.dumps({"runs": runs, "regression": reg}), encoding="utf-8")
    assert mt.report(bundle) == 0
    res = json.loads((bundle / "results.json").read_text(encoding="utf-8"))
    assert res["decision"]["verdict"] == "가" and res["decision"]["fraction"] == 0.01
    assert res["lives_match"] == {"b0": True, "b1": True}
    assert "안: **가**" in (bundle / "report.md").read_text(encoding="utf-8")
    reg["b1"]["lives.csv"] = {"same": False, "first_diff_line": 2}  # 회귀 대조 실패면 멈춘다
    (bundle / "index.json").write_text(json.dumps({"runs": runs, "regression": reg}), encoding="utf-8")
    mt.report(bundle)
    assert json.loads((bundle / "results.json").read_text(encoding="utf-8"))["decision"]["verdict"] == "stop"
