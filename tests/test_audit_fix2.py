"""Codex 독립 점검의 처리 시험 (DECISIONS 33): 평가 입력 검증(33.1), 가중치의 판 번호(33.2), 감시 장치와 한도(33.3),
재생 형식 5의 합법 · 최강 방향(33.4). torch · 체크포인트 · runs/는 쓰지 않는다. 진짜 엔진 실행은 규칙 팩맨 · 시드 0 · 2판."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from helpers import make_cfg

from flypac.core.runner import build_runner
from flypac.recording.recorder import Recorder
from flypac.replay import schema as s
from flypac.replay.decision_metrics import strongest_avoidance, strongest_set
from flypac.replay.reader import read_replay

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
_spec = importlib.util.spec_from_file_location("final_eval", ROOT / "scripts" / "final_eval.py")
fe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fe)

CONDS = ["regime1", "regime2", "regime3", "b0", "b1"]
EVAL_SEEDS = list(range(10))   # 합성 입력의 라벨일 뿐이다 (게임을 돌리지 않는다)


# ================================================================ 33.1 평가 보고의 입력 검증

def make_bundle(root: Path, *, seeds_of=None, games_of=None, cfg_over=None, conds=CONDS, n_games=2,
                score=lambda cond, s, g: 10 * s) -> Path:
    """조건마다 config.yaml · games.csv를 만든 평가 묶음. seeds_of[cond]로 그 조건의 시드 순서를,
    games_of[cond]로 (시드, 판) 목록을, cfg_over[cond]로 설정 덮어쓰기를 바꾼다."""
    bundle = root / "final_eval_v1"
    bundle.mkdir(parents=True)
    runs = []
    for cond in conds:
        seeds = (seeds_of or {}).get(cond, EVAL_SEEDS)
        cfg = {"condition": cond, "seeds": seeds, "n_games": n_games,
               "final_eval": {"conditions": CONDS, "ci_level": 0.95, "censored_warn_fraction": 0.05}}
        for k, v in ((cfg_over or {}).get(cond) or {}).items():
            if k == "final_eval":
                cfg["final_eval"].update(v)
            else:
                cfg[k] = v
        d = root / f"x_{cond}"
        d.mkdir()
        (d / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
        pairs = (games_of or {}).get(cond, [(sd, g) for sd in seeds for g in range(1, n_games + 1)])
        with open(d / "games.csv", "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["조건", "시드", "판", "점수", "승자", "틱 수"])
            w.writerows([cond, sd, g, score(cond, sd, g), "FLIES" if g % 2 else "PACMAN", 400] for sd, g in pairs)
        runs.append({"condition": cond, "run_dir": d.name})
    (bundle / "index.json").write_text(json.dumps({"runs": runs}), encoding="utf-8")
    return bundle


def test_complete_bundle_passes_and_pairs_are_by_seed(tmp_path):
    res = fe.build_results(make_bundle(tmp_path))
    assert set(res["metrics"]) == set(CONDS) and len(res["comparisons"]) == 3 + 3 + 4
    for c in res["comparisons"]:
        assert c["seeds"] == EVAL_SEEDS and c["score"]["deltas"] == [0.0] * 10 and c["score"]["n"] == 10


def test_codex_reordered_seeds_are_stopped(tmp_path):
    # Codex 8.3: B0의 config만 시드 순서를 뒤집으면 위치 짝은 Δ = −90 … +90을 냈다. 이제 시작에서 멈춘다
    b = make_bundle(tmp_path, seeds_of={"b0": EVAL_SEEDS[::-1]})
    with pytest.raises(fe.EvalSetupError, match="b0.*seeds"):
        fe.build_results(b)


def test_compare_matches_pairs_by_seed_value_not_position():
    # 순서 검사를 우회한 입력이라도 compare는 시드 값으로 짝짓는다 (이중 방어). 같은 시드의 점수가 같으므로 Δ = 0
    seeds = EVAL_SEEDS
    a = {"seeds": seeds, "score": [10.0 * s for s in seeds], "W": [0.5] * 10}
    b0 = {"seeds": seeds[::-1], "score": [10.0 * s for s in seeds[::-1]], "W": [0.5] * 10}
    (row,) = fe.compare({"regime1": a, "b0": b0}, 0.95)
    assert row["score"]["deltas"] == [0.0] * 10 and row["seeds"] == seeds
    # 위치로 짝지었다면 나왔을 값과 다르다
    assert [x - y for x, y in zip(a["score"], b0["score"])] != [0.0] * 10
    with pytest.raises(fe.EvalSetupError, match="시드 집합"):
        fe.compare({"regime1": a, "b0": {**b0, "seeds": [s + 1 for s in seeds]}}, 0.95)


@pytest.mark.parametrize("conds,match", [
    (CONDS[:1], "빠진 것"),                       # Codex: 조건 하나만으로 정상 제목의 보고서가 나왔다
    (CONDS[:4], r"빠진 것 \['b1'\]"),
    (CONDS + ["regime4"], r"남는 것 \['regime4'\]"),
])
def test_condition_set_must_equal_final_eval_conditions(tmp_path, conds, match):
    with pytest.raises(fe.EvalSetupError, match=match):
        fe.build_results(make_bundle(tmp_path, conds=conds))


def test_duplicate_condition_in_index_is_stopped(tmp_path):
    b = make_bundle(tmp_path)
    idx = json.loads((b / "index.json").read_text(encoding="utf-8"))
    idx["runs"].append(idx["runs"][0])
    (b / "index.json").write_text(json.dumps(idx), encoding="utf-8")
    with pytest.raises(fe.EvalSetupError, match="두 번"):
        fe.build_results(b)


@pytest.mark.parametrize("over,key", [
    ({"final_eval": {"ci_level": 0.9}}, "ci_level"),
    ({"final_eval": {"censored_warn_fraction": 0.1}}, "censored_warn_fraction"),
    ({"n_games": 3}, "n_games"),
    ({"seeds": EVAL_SEEDS[:9]}, "seeds"),
])
def test_common_settings_must_match_across_conditions(tmp_path, over, key):
    with pytest.raises(fe.EvalSetupError, match=f"regime3: {key}"):
        fe.build_results(make_bundle(tmp_path, cfg_over={"regime3": over}))


def test_condition_in_config_must_match_index(tmp_path):
    with pytest.raises(fe.EvalSetupError, match="condition"):
        fe.build_results(make_bundle(tmp_path, cfg_over={"b1": {"condition": "b0"}}))


def test_codex_one_game_per_seed_is_stopped(tmp_path):
    # Codex 8.3: 시드당 1판(10행)만으로 10쌍 · df 9를 냈다. 판 수가 200이면 판 2 ~ 200이 빠졌다고 멈춘다
    one = [(sd, 1) for sd in EVAL_SEEDS]
    b = make_bundle(tmp_path, n_games=200, games_of={c: one for c in CONDS})
    with pytest.raises(fe.EvalSetupError, match=r"regime1: 시드 0의 판이 1 ~ 200와 다르다 \(빠진 판 \[2, 3"):
        fe.build_results(b)


def test_duplicate_missing_or_foreign_games_name_condition_seed_and_game(tmp_path):
    full = [(sd, g) for sd in EVAL_SEEDS for g in (1, 2)]
    cases = [
        (full + [(3, 2)], "b0: 시드 3의 판 2가 두 번"),
        ([p for p in full if p != (7, 1)], r"b0: 시드 7의 판이 1 ~ 2와 다르다 \(빠진 판 \[1\]"),
        (full + [(5, 3)], r"b0: 시드 5의 판이 1 ~ 2와 다르다 \(빠진 판 \[\], 범위 밖 판 \[3\]"),
        (full + [(42, 1)], "b0: 시드 42"),
    ]
    for i, (pairs, match) in enumerate(cases):
        with pytest.raises(fe.EvalSetupError, match=match):
            fe.build_results(make_bundle(tmp_path / str(i), games_of={"b0": pairs}))


def test_report_command_stops_without_writing(tmp_path, capsys):
    b = make_bundle(tmp_path, seeds_of={"b0": EVAL_SEEDS[::-1]})
    assert fe.main(["final_eval.py", "report", str(b)]) == 1
    assert "멈춤" in capsys.readouterr().err
    assert not (b / "report.md").exists() and not (b / "results.json").exists()


# ================================================================ 33.2 가중치의 판 번호

def _real_slot_file(d: Path, n_games: int) -> Path:
    """진짜 FlyBrain 초기 슬롯 상태(게임을 돌리지 않음, 시드 0)를 WeightWriter로 판 1에 쓴다."""
    from flypac.brain.flybrain import FlyBrain, load_brain_config
    from flypac.recording.weights import WeightWriter

    brain = FlyBrain(load_brain_config(ROOT / "configs" / "brain_v1.yaml"), 1, [0], condition="regime1")
    w = WeightWriter(d / "weights", seeds=[0], every_games=1, n_games=n_games, run_config_hash="0" * 16)
    w.on_game_end(0, 1, brain)
    return w.files[0]


def test_codex_renamed_weight_file_is_stopped(tmp_path):
    # Codex 8.3: 파일 안 판 1인 파일을 s0_g00003.npz로 이름만 바꾸면 통과했다
    d = tmp_path / "train"
    d.mkdir()
    (d / "config.yaml").write_text(yaml.safe_dump({"condition": "regime1", "n_games": 3}), encoding="utf-8")
    p = _real_slot_file(d, 3)
    p.rename(p.with_name("s0_g00003.npz"))
    with pytest.raises(fe.EvalSetupError, match="파일 안의 판 번호 1.*파일 이름의 판 번호 3"):
        fe.collect_weights([d], ["regime1"], [0], train_n_games=3)


def test_collect_passes_and_returns_internal_game_no(tmp_path):
    d = tmp_path / "train"
    d.mkdir()
    (d / "config.yaml").write_text(yaml.safe_dump({"condition": "regime1", "n_games": 1}), encoding="utf-8")
    _real_slot_file(d, 1)
    got = fe.collect_weights([d], ["regime1"], [0], train_n_games=1)
    assert got[("regime1", 0)]["game_no"] == 1 and isinstance(got[("regime1", 0)]["game_no"], int)


def _meta_only(d: Path, cond: str, n_games: int, name_game: int, inner_game, seed=0, inner_seed=None):
    d.mkdir(parents=True)
    (d / "config.yaml").write_text(yaml.safe_dump({"condition": cond, "n_games": n_games}), encoding="utf-8")
    (d / "weights").mkdir()
    extra = {} if inner_game is None else {"game_no": np.int64(inner_game)}
    np.savez(d / "weights" / f"s{seed}_g{name_game:05d}.npz", dw_masked=np.zeros(3), rbar=np.zeros(1),
             seed=np.int64(seed if inner_seed is None else inner_seed), run_config_hash=np.str_("0" * 16), **extra)
    return d


def test_train_n_games_missing_game_no_and_inner_seed_are_checked(tmp_path):
    d = _meta_only(tmp_path / "a", "regime1", 300, 300, 300)
    with pytest.raises(fe.EvalSetupError, match="train_n_games = 3000"):
        fe.collect_weights([d], ["regime1"], [0], train_n_games=3000)   # 이름 = 파일 안 = 학습 판 수, 평가 설정만 다름
    fe.collect_weights([d], ["regime1"], [0], train_n_games=300)
    with pytest.raises(fe.EvalSetupError, match="game_no"):
        fe.collect_weights([_meta_only(tmp_path / "b", "regime1", 5, 5, None)], ["regime1"], [0], train_n_games=5)
    with pytest.raises(fe.EvalSetupError, match="파일 안의 시드 7"):
        fe.collect_weights([_meta_only(tmp_path / "c", "regime1", 5, 5, 5, inner_seed=7)], ["regime1"], [0],
                           train_n_games=5)


# ================================================================ 33.3 감시 장치와 한도

def _watchdog_cfg(limit, **over):
    # Codex 8절의 재현: 진짜 엔진 · 규칙 팩맨 · 시드 0 · watchdog 2 · 속도 0 · 2판
    return make_cfg(n_games=2, seeds=[0], engine={"kind": "real", "watchdog_ticks": 2, "pacman_speed": 0.0,
                                                  "pacman_dot_speed": 0.0, "fly_speed": 0.0},
                    run={"max_loop_ticks": limit}, **over)


def test_codex_limit_4_is_rejected_at_start():
    r = build_runner(_watchdog_cfg(4))
    with pytest.raises(ValueError, match="33.3"):
        r.run()
    assert r.total_ticks == 0 and r.results == []    # 한 틱도 돌지 않았다 (예전에는 첫 판만 마치고 감시 오류)


def test_codex_limit_6_finishes_both_games_each_censored_at_watchdog_plus_one(tmp_path):
    cfg = _watchdog_cfg(6)
    rec = Recorder(tmp_path / "run", cfg)
    out = build_runner(cfg, rec).run()
    assert out["ticks"] == 6
    assert [(g["game_no"], g["winner"]) for g in out["games"]] == [(1, "CENSORED"), (2, "CENSORED")]
    with open(tmp_path / "run" / "games.csv", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [int(r["틱 수"]) for r in rows] == [3, 3]           # 한 판 = watchdog_ticks + 1 = 3틱
    assert [r["승자"] for r in rows] == ["CENSORED", "CENSORED"]


# ================================================================ 33.4 재생 형식 5: legal · strongest

def test_strongest_codex_counterexample_float64_vs_float16():
    odor = np.array([1.0001, 1.0002, 0.0, 0.0])
    legal = np.array([True, True, False, False])
    assert strongest_set(odor, legal).tolist() == [False, True, False, False]
    assert np.float16(odor[0]) == np.float16(odor[1])            # float16으로는 구분되지 않는 값
    assert strongest_set(odor.astype(np.float16), legal).tolist() == [True, True, False, False]


def test_strongest_ties_and_illegal_directions():
    assert strongest_set(np.array([0.5, 2.0, 2.0, 0.1]), np.array([1, 1, 1, 1], bool)).tolist() == \
        [False, True, True, False]                               # 동률은 둘 다
    assert strongest_set(np.array([9.0, 1.0, 1.0, 0.5]), np.array([0, 1, 1, 1], bool)).tolist() == \
        [False, True, True, False]                               # 합법이 아니면 값이 커도 거짓
    assert not strongest_set(np.array([1.0, 2.0, 3.0, 4.0]), np.zeros(4, bool)).any()   # 합법 없음 → 모두 거짓
    many = strongest_set(np.zeros((3, 4, 4)), np.ones((3, 4, 4), bool))
    assert many.shape == (3, 4, 4) and many.all()


def test_strongest_avoidance_definition_on_hand_made_log():
    T, F = 3, 4
    legal = np.zeros((T, F, 4), bool)
    strongest = np.zeros((T, F, 4), bool)
    action = np.full((T, F), -1, np.int8)
    committed = np.zeros((T, F), bool)
    z = np.zeros((T, F), bool)
    # (t0, f0) z=0: 합법 {0,1,2}, 최강 {1}, 고름 0 → 회피, 우연 2/3
    legal[0, 0, :3], strongest[0, 0, 1], action[0, 0], committed[0, 0] = True, True, 0, True
    # (t1, f1) z=0: 합법 {0,1}, 최강 {0,1} (동률), 고름 1 → 회피 아님, 우연 0
    legal[1, 1, :2], strongest[1, 1, :2], action[1, 1], committed[1, 1] = True, True, 1, True
    # (t2, f2) z=1: 합법 {1,3}, 최강 {3}, 고름 1 → 회피, 우연 1/2
    legal[2, 2, [1, 3]], strongest[2, 2, 3], action[2, 2], committed[2, 2], z[2, 2] = True, True, 1, True, True
    # 확정되지 않은 제안은 세지 않는다
    legal[2, 3, :2], strongest[2, 3, 0], action[2, 3] = True, True, 1
    got = strongest_avoidance({"legal": legal, "strongest": strongest, "action": action, "committed": committed,
                               "fly_z": z})
    assert got[0] == {"n": 2, "avoid": 0.5, "chance": pytest.approx((2 / 3 + 0) / 2)}
    assert got[1] == {"n": 1, "avoid": 1.0, "chance": 0.5}
    with pytest.raises(ValueError, match="형식 5"):
        strongest_avoidance({"action": action, "committed": committed, "fly_z": z})


def _brain_run(tmp_path, name, replay: bool):
    over = {"n_games": 2, "seeds": [0], "engine": {"kind": "real"},
            "ghosts": {"kind": "flybrain", "config": "configs/brain_v1.yaml"}}
    if replay:
        over["replay"] = {"keep_games": [1, 2], "brain": True}
    cfg = make_cfg(**over)
    rec = Recorder(tmp_path / name, cfg)
    runner = build_runner(cfg, rec)
    out = runner.run()
    return rec, runner, out


@pytest.fixture(scope="module")
def brain_runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("fmt5")
    on = _brain_run(tmp, "on", True)
    off = _brain_run(tmp, "off", False)
    return tmp, on, off


def test_real_engine_brain_log_v5_reads_back_and_metric_runs(brain_runs):
    _, (rec, _, _), _ = brain_runs
    assert len(rec.replay_files) == 2
    for path in rec.replay_files:
        meta, a = read_replay(path)
        assert meta["format_version"] == 5
        T = len(a["pac_cell"])
        assert a["legal"].shape == a["strongest"].shape == (T, 4, 4) and a["legal"].dtype == bool
        assert not (a["strongest"] & ~a["legal"]).any()
        assert (a["strongest"].any(axis=2) == a["legal"].any(axis=2)).all()   # 합법이 있으면 최강도 있다
        c = a["committed"]
        assert c.any()
        t, f = np.nonzero(c)
        assert a["legal"][t, f, a["action"][t, f]].all()                     # 확정된 행동은 합법이다
        got = strongest_avoidance(a)
        assert set(got) == {0, 1} and got[0]["n"] + got[1]["n"] == int(c.sum())
        for v in got.values():
            if v["n"]:
                assert 0.0 <= v["avoid"] <= 1.0 and 0.0 <= v["chance"] < 1.0


def test_output_invariance_csv_bytes_and_learning_state(brain_runs):
    tmp, (rec_on, run_on, out_on), (rec_off, run_off, out_off) = brain_runs
    assert rec_on.replay_files and not rec_off.replay_files
    for name in ("games.csv", "lives.csv", "flies.csv"):
        assert (tmp / "on" / name).read_bytes() == (tmp / "off" / name).read_bytes(), name
    assert out_on == out_off
    # 학습 경로: 끝난 뒤 칸의 가중치 변화와 R̄가 바이트까지 같다
    a, b = run_on.ghosts.slot_state(0), run_off.ghosts.slot_state(0)
    assert a["dw_masked"].tobytes() == b["dw_masked"].tobytes() and a["rbar"].tobytes() == b["rbar"].tobytes()
    assert np.abs(a["dw_masked"]).sum() > 0                                  # 학습이 실제로 일어났다


def test_format4_file_made_here_still_reads(brain_runs, tmp_path):
    _, (rec, _, _), _ = brain_runs
    meta, a = read_replay(rec.replay_files[0])
    old = {k: v for k, v in a.items() if k not in ("legal", "strongest")}
    meta4 = dict(meta, format_version=4)
    old[s.META_KEY] = np.array(json.dumps(meta4, ensure_ascii=False))
    path = tmp_path / "v4.npz"
    np.savez_compressed(path, **old)
    meta_r, got = read_replay(path)
    assert meta_r["format_version"] == 4 and "legal" not in got and "brain_on" in got
    for name in got:
        assert np.array_equal(got[name], a[name]), name
    with pytest.raises(ValueError, match="형식 5"):
        strongest_avoidance(got)                     # 옛 로그로는 정확히 계산하지 않는다 (33.4)
    bad = dict(old)
    bad["legal"] = a["legal"]
    with pytest.raises(s.SchemaError, match="legal"):
        s.validate(bad)                              # 형식 4 파일에는 legal이 없어야 한다
    with pytest.raises(s.SchemaError, match="strongest"):
        s.validate({k: v for k, v in {**a, s.META_KEY: np.array(json.dumps(meta))}.items() if k != "strongest"})
