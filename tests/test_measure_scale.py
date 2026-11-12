"""규모 측정 도구 시험 (DECISIONS 27). 가짜 엔진 · 규칙 팩맨 · 무작위 유령으로 짧게 돈다. torch · 체크포인트는 쓰지 않는다."""

from __future__ import annotations

import copy
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rp = _load("run_parallel")
ms = _load("measure_scale")
SCALE = yaml.safe_load((ROOT / "configs" / "scale_v1.yaml").read_text(encoding="utf-8"))
INTEG = yaml.safe_load((ROOT / "configs" / "integ_v1.yaml").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- 설정

def test_scale_config_values_and_copied_blocks():
    for k in ("engine", "time", "odor", "phero", "pacman", "run"):
        assert SCALE[k] == INTEG[k], k
    assert SCALE["experiment"] == "scale_v1" and SCALE["seeds"] == list(range(7100, 7110))
    assert SCALE["n_games"] == 30 and SCALE["replay"] == {"keep_games": [], "brain": False}
    assert SCALE["weights"] == {"every_games": 100}
    sc = SCALE["scale"]
    assert sc["conditions"] == ["regime1", "regime2", "regime3", "b0", "b1"]
    assert [(b["name"], b["split"], b["sequential"]) for b in sc["batches"]] == [
        ("S", 1, True), ("A", 1, False), ("B", 2, False)]
    assert (sc["budget_hours"], sc["safety_factor"], sc["eval_games"]) == (8, 1.5, 200)
    assert sc["n_games_candidates"] == [1000, 2000, 3000] and sc["main_seeds"] == list(range(8000, 8010))


# ---------------------------------------------------------------- 정렬 해시

COLS_L = ["조건", "시드", "판", "목숨 번호", "x"]


def write_csv(path, cols, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        w.writerows(rows)


ROWS = [["b0", 7, 1, 1, "a"], ["b0", 7, 1, 2, "b"], ["b0", 7, 2, 1, "c"], ["b0", 8, 1, 1, "d"], ["b0", 8, 10, 1, "e"],
        ["b0", 8, 2, 1, "f"]]


def h(tmp, name, rows, cols=COLS_L):
    write_csv(tmp / name / "lives.csv", cols, rows)
    return tmp / name / "lives.csv"


KEY = ("시드", "판", "목숨 번호")


def test_hash_ignores_row_order_but_sees_a_cell(tmp_path):
    a = ms.sorted_rows_hash([h(tmp_path, "a", ROWS)], KEY)
    b = ms.sorted_rows_hash([h(tmp_path, "b", list(reversed(ROWS)))], KEY)
    assert a == b
    changed = [r[:] for r in ROWS]
    changed[3][4] = "D"
    assert ms.sorted_rows_hash([h(tmp_path, "c", changed)], KEY) != a


def test_hash_sorts_numerically_not_as_text(tmp_path):
    # 판 10은 판 2 뒤에 온다: 글자순이면 같은 행 집합이어도 정렬이 달라질 수 있다. 숫자순으로 고정되는지 본다
    a = ms.sorted_rows_hash([h(tmp_path, "a", ROWS)], KEY)
    b = ms.sorted_rows_hash([h(tmp_path, "b", [ROWS[4], ROWS[5], ROWS[0], ROWS[3], ROWS[2], ROWS[1]])], KEY)
    assert a == b


def test_hash_of_split_folders_equals_unsplit(tmp_path):
    whole = ms.sorted_rows_hash([h(tmp_path, "whole", ROWS)], KEY)
    p1 = h(tmp_path, "p1", [ROWS[0], ROWS[1], ROWS[2]])
    p2 = h(tmp_path, "p2", [ROWS[5], ROWS[3], ROWS[4]])  # 칸이 끝나는 순서가 달라도
    assert ms.sorted_rows_hash([p1, p2], KEY) == whole
    assert ms.sorted_rows_hash([p2, p1], KEY) == whole


def test_weights_hash_uses_array_bytes_only(tmp_path):
    def wdir(name, dw, extra):
        d = tmp_path / name / "weights"
        d.mkdir(parents=True)
        np.savez(d / "s7_g00030.npz", dw_masked=dw, rbar=np.array([0.5, 1.5]), run_config_hash=np.str_(extra))
        return tmp_path / name

    dw = np.arange(6, dtype=np.float64)
    assert ms.weights_hash([wdir("a", dw, "x")]) == ms.weights_hash([wdir("b", dw, "다른 실행 해시")])
    assert ms.weights_hash([wdir("c", dw + 1e-12, "x")]) != ms.weights_hash([tmp_path / "a"])
    assert ms.weights_hash([tmp_path / "none"]) is None


# ---------------------------------------------------------------- 규칙

HS = {c: f"h_{c}" for c in SCALE["scale"]["conditions"]}
HOUR = 3600.0


def decide(a_h, b_h, ta, tb, s=None):
    hashes = {"S": HS, "A": a_h if a_h is not None else HS, "B": b_h if b_h is not None else HS}
    return ms.decide_scale(hashes, {"S": 1.0, "A": ta, "B": tb}, s or SCALE["scale"])


def test_several_in_budget_picks_largest():
    # T̂(n) = T (1.5n + 200)/30. T = 360초: n=1000 → 20400초(5.67h), 2000 → 38400초(10.67h) → 1000
    # T = 100초: 1000 → 5666초, 2000 → 10666초, 3000 → 15666초 (4.35h) → 모두 안 → 3000
    d = decide(None, None, 100.0, 100.0)
    assert d["action"] == "decide" and d["n_games"] == 3000 and d["nights"] == 1
    assert [r["within_budget"] for r in d["table"]] == [True, True, True]


def test_one_in_budget():
    # T = 600초: 1000 → 34000초(9.4h) 밖. T = 400초: 1000 → 22666초(6.3h), 2000 → 42666초(11.8h) → 1000
    d = decide(None, None, 400.0, 400.0)
    assert d["n_games"] == 1000 and [r["within_budget"] for r in d["table"]] == [True, False, False]
    assert d["nights"] == 1


def test_none_in_budget_gives_1000_over_two_nights():
    d = decide(None, None, 1000.0, 1000.0)
    assert d["n_games"] == 1000 and d["nights"] == 2 and not any(r["within_budget"] for r in d["table"])


def test_exactly_eight_hours_is_included():
    # T × 1700/30 = 28800초 → T = 508.2352...: 분수로 맞춘다. n=1000에서 정확히 8시간이 되는 T
    from fractions import Fraction
    T = Fraction(28800 * 30, 1700)
    s = copy.deepcopy(SCALE["scale"])
    d = ms.decide_scale({"S": HS, "A": HS, "B": HS}, {"S": 1.0, "A": T, "B": T}, s)
    assert d["table"][0]["within_budget"] and d["n_games"] == 1000 and d["nights"] == 1
    d2 = ms.decide_scale({"S": HS, "A": HS, "B": HS}, {"S": 1.0, "A": T + Fraction(1, 10**6), "B": T + 1}, s)
    assert not d2["table"][0]["within_budget"] and d2["nights"] == 2


def test_b_used_only_when_equal_and_faster():
    assert decide(None, None, 200.0, 150.0)["batch"] == "B"
    assert decide(None, None, 200.0, 200.0)["batch"] == "A"      # 같은 시간이면 A
    assert decide(None, None, 150.0, 200.0)["batch"] == "A"      # 더 느리면 A
    other = dict(HS, b0="다름")
    d = decide(None, other, 200.0, 50.0)                          # B가 A와 다르면 더 빨라도 A, 멈추지는 않는다
    assert d["action"] == "decide" and d["batch"] == "A" and d["ab_diff"] == ["b0"] and not d["ab_equal"]
    assert d["t_seconds"] == 200.0


def test_stops_when_a_differs_from_s():
    other = dict(HS, regime2="다름")
    d = decide(other, None, 200.0, 100.0)
    assert d["action"] == "stop" and d["sa_diff"] == ["regime2"] and "n_games" not in d and "table" not in d


def test_rule_reads_numbers_from_scale_block():
    s = copy.deepcopy(SCALE["scale"])
    s["n_games_candidates"], s["budget_hours"], s["safety_factor"], s["eval_games"] = [10, 20], 1, 2, 0
    # T = 30초: n=10 → 30×20/30 = 20초, n=20 → 40초. 예산 1시간 안
    d = decide(None, None, 30.0, 30.0, s)
    assert d["n_games"] == 20 and [r["t_hat_hours"] for r in d["table"]] == pytest.approx([20 / 3600, 40 / 3600])


# ---------------------------------------------------------------- 실행기와 지우기 (가짜 엔진)

def fake_base(n_games=2, seeds=(1, 2, 3, 4)):
    from helpers import BASE
    d = copy.deepcopy(BASE)
    d.update(experiment="scale_test", seeds=list(seeds), n_games=n_games, condition="b0",
             ghosts={"kind": "random"}, replay={"keep_games": [1], "brain": False})
    d["run"] = {"max_loop_ticks": 20000}
    d["scale"] = {"conditions": ["b0"], "batches": [{"name": "S", "split": 1, "sequential": True},
                                                    {"name": "A", "split": 1, "sequential": False},
                                                    {"name": "B", "split": 2, "sequential": False}]}
    return d


def digests(index_dir, index, learning=False):
    out = {}
    for cond in sorted({r["condition"] for r in index["runs"]}):
        dirs = [index_dir / r["run_dir"] for r in index["runs"] if r["condition"] == cond]
        out[cond] = ms.condition_digest(dirs, learning)
    return out


def test_split_seeds_front_loaded_and_validated():
    assert rp.split_seeds([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 2) == [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
    assert rp.split_seeds([1, 2, 3, 4, 5], 2) == [[1, 2, 3], [4, 5]]
    with pytest.raises(ValueError):
        rp.split_seeds([1, 2], 3)


def test_condition_config_fills_condition_and_ghosts():
    base = fake_base()
    b1 = rp.condition_config(base, "b1", [1], 1)
    b0 = rp.condition_config(base, "regime1", [1], 2)
    assert b1["ghosts"] == {"kind": "original"} and b1["condition"] == "b1"
    assert b0["ghosts"] == {"kind": "random"} and b0["condition"] == "regime1" and b0["seeds"] == [1]


def test_split_2_gives_same_sorted_hash_as_split_1(tmp_path):
    base = fake_base()
    one = rp.run_batch(base, ["b0"], 1, False, tmp_path / "one")
    two = rp.run_batch(base, ["b0"], 2, False, tmp_path / "two")
    assert not one["failed"] and not two["failed"] and len(two["runs"]) == 2
    d1, d2 = digests(tmp_path / "one", one), digests(tmp_path / "two", two)
    assert d1["b0"]["combined"] == d2["b0"]["combined"] and d1["b0"]["game_ticks"] == d2["b0"]["game_ticks"] > 0
    for r in two["runs"]:
        assert {"condition", "seeds", "run_dir", "start", "end", "wall_seconds", "exit_code"} <= set(r)
    assert json.loads((tmp_path / "two" / "index.json").read_text(encoding="utf-8"))["runs"][0]["exit_code"] == 0


def test_sequential_equals_parallel(tmp_path):
    base = fake_base()
    s = rp.run_batch(base, ["b0"], 1, True, tmp_path / "s")
    a = rp.run_batch(base, ["b0"], 1, False, tmp_path / "a")
    assert digests(tmp_path / "s", s)["b0"]["combined"] == digests(tmp_path / "a", a)["b0"]["combined"]


def test_failure_waits_reports_and_exits_nonzero(tmp_path):
    base = fake_base()
    base["pacman"] = {"kind": "없는 종류"}
    idx = rp.run_batch(base, ["b0"], 2, False, tmp_path / "bad")
    assert len(idx["failed"]) == 2 and all(r["exit_code"] != 0 for r in idx["runs"])
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(yaml.safe_dump(base, allow_unicode=True), encoding="utf-8")
    assert rp.main(["run_parallel.py", str(cfg), "--conditions", "b0", "--split", "1",
                    "--out", str(tmp_path / "bad2")]) == 1


def test_child_env_sets_single_thread():
    env = rp.child_env()
    assert all(env[k] == "1" for k in rp.THREAD_VARS)


def test_measure_erases_results_and_prints_none(tmp_path, capsys):
    base = fake_base()
    done = ms.measure_all(base, tmp_path / "scale")
    assert [m["name"] for m in done] == ["S", "A", "B"] and all(not m["failed"] for m in done)
    # 세 배치의 정렬 해시는 같다 (가짜 엔진은 결정론적)
    assert len({m["conditions"]["b0"]["combined"] for m in done}) == 1
    files = {p.relative_to(tmp_path / "scale").as_posix() for p in (tmp_path / "scale").rglob("*") if p.is_file()}
    for f in files:
        name = Path(f).name
        assert name not in ("games.csv", "lives.csv", "flies.csv") and not name.endswith(".npz"), f
        assert "/replay/" not in f and "/weights/" not in f, f
    kept = {Path(f).name for f in files}
    assert {"config.yaml", "environment.json", "git_commit.txt", "measure.json", "index.json"} <= kept
    # 측정 결과 파일에는 해시 · 게임-틱 합 · 시간만 있다
    meas = json.loads((tmp_path / "scale" / "A" / "measure.json").read_text(encoding="utf-8"))
    assert set(meas["conditions"]["b0"]) == {"games", "lives", "flies", "weights", "combined", "game_ticks",
                                              "wall_seconds", "run_dirs"}
    # 화면에는 승패 · 점수 같은 말이 없다
    out = capsys.readouterr().out
    for word in ("승", "점수", "보상", "포획", "FLIES", "PACMAN", "CENSORED", "winner", "score", "reward"):
        assert word not in out, word


def test_report_has_no_learning_results(tmp_path):
    base = fake_base()
    base["scale"].update(budget_hours=8, safety_factor=1.5, n_games_candidates=[1000, 2000, 3000], eval_games=200)
    ms.measure_all(base, tmp_path / "scale")
    assert ms.report(tmp_path / "scale") == 0
    md = (tmp_path / "scale" / "report.md").read_text(encoding="utf-8")
    res = json.loads((tmp_path / "scale" / "results.json").read_text(encoding="utf-8"))
    for text in (md, json.dumps(res, ensure_ascii=False)):
        for word in ("점수", "보상", "포획", "승률", "FLIES", "PACMAN", "CENSORED", "winner", "score"):
            assert word not in text, word
    assert res["decision"]["action"] == "decide" and "게임-틱 합" in md


def test_measure_erases_even_when_a_process_fails(tmp_path):
    base = fake_base()
    # 두 번째 조건 이름이 잘못이면 그 프로세스만 실패하고 첫 조건의 결과는 지워진다
    base["scale"]["conditions"] = ["b0", "없는조건"]
    done = ms.measure_all(base, tmp_path / "scale")
    assert done[0]["failed"] and len(done) == 1
    left = [p.name for p in (tmp_path / "scale").rglob("*.csv")]
    assert left == []


def test_learning_condition_without_weights_is_an_error(tmp_path):
    d = tmp_path / "r"
    for fname, cols in (("games.csv", ["조건", "시드", "판", "틱 수"]), ("lives.csv", COLS_L),
                        ("flies.csv", ["조건", "시드", "판", "초파리"])):
        write_csv(d / fname, cols, [])
    with pytest.raises(ValueError):
        ms.condition_digest([d], True)
    assert ms.condition_digest([d], False)["weights"] is None


def test_erase_results_removes_csv_replay_and_weights_keeps_meta(tmp_path):
    d = tmp_path / "r"
    for f in ("games.csv", "lives.csv", "flies.csv", "config.yaml", "environment.json", "git_commit.txt"):
        (d / f).parent.mkdir(parents=True, exist_ok=True)
        (d / f).write_text("x", encoding="utf-8")
    for sub, name in (("replay", "g.npz"), ("weights", "s1_g00030.npz")):
        (d / sub).mkdir()
        (d / sub / name).write_bytes(b"x")
    ms.erase_results(d)
    assert sorted(p.name for p in d.rglob("*")) == ["config.yaml", "environment.json", "git_commit.txt"]
