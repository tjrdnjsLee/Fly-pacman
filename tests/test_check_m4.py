"""M4 확인 스크립트 시험 (DECISIONS 34.3의 C1 ~ C9). torch · 체크포인트는 쓰지 않는다.

진짜 엔진 + 규칙 팩맨, 시드 0 · 1 (평가 시드 10 · 11), 칸마다 2판으로 학습 묶음과 평가 묶음을 임시 폴더에 한 번 만들고
(실제 경로와 같은 `run_parallel.plan` · `Recorder` · `final_eval`의 함수들로), 항목마다 통과하는 경우와 그 묶음을
복사해 한 곳을 망가뜨린 실패하는 경우를 본다.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import re
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
from flypac.recording.weights import read_weight_file, weight_filename

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fe = _load("final_eval")
rp = _load("run_parallel")
cm = _load("check_m4")

CONDS = ["regime1", "regime2", "regime3", "b0", "b1"]
TRAIN_SEEDS, EVAL_SEEDS, N_GAMES = [0, 1], [10, 11], 2
FAKE_CKPT_SHA = "ab" * 32      # 규칙 팩맨에는 체크포인트가 없다. 대조만 보려고 설정과 environment.json에 같은 글을 둔다
FAKE_COMMIT = "c" * 40
SENTINEL = "987654321"          # 성능 값 자리에 심는 표시 (출력에 나오면 안 된다)


def _small(name: str, seeds: list[int]) -> dict:
    """M4 설정을 작게: 규칙 팩맨, 시드 · 판 수만 줄인다. 그 밖(엔진 · 냄새 · 페로몬 · 뇌 · 가중치)은 M4 설정 그대로."""
    d = yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text(encoding="utf-8"))
    d["seeds"], d["n_games"] = list(seeds), N_GAMES
    d["pacman"] = {"kind": "rule_based", "checkpoint_sha256": FAKE_CKPT_SHA}
    d["rule_based"] = dict(HELPER_BASE["rule_based"])
    d["run"] = {"max_loop_ticks": N_GAMES * (d["engine"]["watchdog_ticks"] + 1)}
    return d


def _train_cfg() -> dict:
    d = _small("m4_check_v1", TRAIN_SEEDS)
    d["replay"] = {"keep_games": [1, N_GAMES], "brain": True}
    return d


def _eval_cfg() -> dict:
    d = _small("m4_eval_v1", EVAL_SEEDS)
    d["final_eval"]["train_seeds"], d["final_eval"]["train_n_games"] = list(TRAIN_SEEDS), N_GAMES
    return d


def _stamp(run_dir: Path) -> None:
    """실행 뒤의 사실 두 가지를 고정한다: 커밋(시험 중 작업 폴더는 더러울 수 있다)과 체크포인트 해시(규칙 팩맨)."""
    (run_dir / "git_commit.txt").write_text(FAKE_COMMIT + "\n", encoding="utf-8")
    env = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
    env["pacman_checkpoint_sha256"] = FAKE_CKPT_SHA
    (run_dir / "environment.json").write_text(json.dumps(env, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    """(뿌리 폴더, 사양). 뿌리 아래 T(학습 묶음), m4_eval(평가 묶음), x_m4_eval_<조건>(평가 결과 폴더)."""
    root = tmp_path_factory.mktemp("m4")
    train, evalc = _train_cfg(), _eval_cfg()
    T = root / "T"
    runs = []
    for job in rp.plan(train, CONDS, 1):          # run_parallel과 같은 조건 설정 · 폴더 이름 (<조건>_p1)
        cfg = Config(job["config"])
        build_runner(cfg, Recorder(T / job["name"], cfg)).run()
        _stamp(T / job["name"])
        runs.append({"condition": job["condition"], "seeds": job["seeds"], "run_dir": job["name"], "exit_code": 0})
    (T / "index.json").write_text(json.dumps({"conditions": CONDS, "split": 1, "runs": runs, "failed": []}),
                                  encoding="utf-8")
    # 평가: final_eval.run과 같은 순서 (체크포인트 · 장치 확인만 뺀다)
    train_dirs = [T / f"{c}_p1" for c in CONDS]
    weights = fe.collect_weights(train_dirs, CONDS, TRAIN_SEEDS, train_n_games=N_GAMES)
    fe.check_train_dirs(evalc, train_dirs)
    E = root / "m4_eval"
    E.mkdir()
    index = []
    for cond in CONDS:
        cfg = fe.eval_config(evalc, cond)
        run_dir = root / f"x_{cfg.experiment}"
        runner = build_runner(cfg, Recorder(run_dir, cfg))
        manifest = []
        if cond in fe.LEARNING:
            for i, (e, s) in enumerate(zip(EVAL_SEEDS, TRAIN_SEEDS)):
                hit = weights[(cond, s)]
                runner.ghosts.load_slot_state(i, read_weight_file(hit["path"]))
                manifest.append({"slot": i, "eval_seed": e, "train_seed": s, "file": str(hit["path"]),
                                 "game_no": hit["game_no"], "sha256": fe._sha256(hit["path"])})
        (run_dir / "eval_manifest.json").write_text(json.dumps({"condition": cond, "files": manifest}), encoding="utf-8")
        runner.run()
        _stamp(run_dir)
        index.append({"condition": cond, "run_dir": run_dir.name})
    (E / "index.json").write_text(json.dumps({"runs": index}), encoding="utf-8")
    res = fe.build_results(E)
    (E / "results.json").write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    (E / "report.md").write_text(fe.render_report(res), encoding="utf-8")
    return root, cm.make_spec(train, evalc)


@pytest.fixture
def copy(bundle, tmp_path):
    """묶음의 복사본 (T, E, 사양): 망가뜨려도 다른 시험에 닿지 않는다."""
    root, spec = bundle
    dst = tmp_path / "m4"
    shutil.copytree(root, dst)
    return dst / "T", dst / "m4_eval", spec


def item(T, E, spec, cid):
    (it,) = [i for i in cm.run_checks(T, E, spec)["items"] if i["id"] == cid]
    return it


def edit_json(path: Path, fn) -> None:
    d = json.loads(path.read_text(encoding="utf-8"))
    fn(d)
    path.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")


def edit_csv(path: Path, fn) -> None:
    """fn(열 이름 목록, 행 사전 목록) -> (열 이름 목록, 행 사전 목록)."""
    with open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        rows, cols = list(reader), list(reader.fieldnames)
    cols, rows = fn(cols, rows)
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def edit_npz(path: Path, fn) -> None:
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    fn(arrays)
    np.savez(path, **arrays)


def replay(run_dir: Path, cond: str, seed: int, game: int) -> Path:
    return run_dir / "replay" / f"{cond}_s{seed}_g{game}.npz"


def eval_dir(E: Path, cond: str) -> Path:
    return E.parent / f"x_m4_eval_v1_{cond}"


# ================================================================ 사양 · 통과

def test_spec_reads_the_m4_configs():
    spec = cm.make_spec(yaml.safe_load(cm.TRAIN_CONFIG.read_text(encoding="utf-8")),
                        yaml.safe_load(cm.EVAL_CONFIG.read_text(encoding="utf-8")))
    assert spec.conditions == tuple(CONDS) and spec.brain == ("regime1", "regime2", "regime3", "b0")
    assert spec.learning == ("regime1", "regime2", "regime3") and spec.no_learning == ("b0", "b1")
    assert spec.train_seeds == tuple(range(7300, 7310)) and spec.eval_seeds == tuple(range(7400, 7410))
    assert spec.pair_train_seeds == spec.train_seeds and spec.train_keep == (1, 20) and spec.eval_keep == (1,)
    assert (spec.train_n_games, spec.eval_n_games, spec.manifest_game_no, spec.censored_max) == (20, 5, 20, 0.05)


def test_real_small_bundle_passes_every_item(bundle):
    root, spec = bundle
    res = cm.run_checks(root / "T", root / "m4_eval", spec)
    assert [i["id"] for i in res["items"]] == [f"C{k}" for k in range(1, 10)]
    assert res["passed"], [(i["id"], i["reasons"]) for i in res["items"] if not i["passed"]]
    by = {i["id"]: i["counts"] for i in res["items"]}
    assert by["C1"]["games.csv가 완결인 폴더 수"] == 10 and by["C1"]["학습 exit_code 0인 조건 수"] == 5
    assert by["C2"] == {"기대한 재생 파일 수": 20, "형식이 맞는 파일 수": 20}
    assert by["C3"]["학습 재생: 오류 없이 계산된 파일 수"] == 16 and by["C3"]["평가 재생: 오류 없이 계산된 파일 수"] == 8
    assert by["C4"]["조건별 반전 기회의 합"]["b1"] == 0
    assert by["C5"]["체제 3: 페로몬 장이 있고 최댓값 > 0인 파일 수"] == 4 and by["C5"]["다른 조건: 페로몬 장이 없는 파일 수"] == 16
    assert by["C6"]["판 번호가 맞는 파일 수"] == 6 and by["C6"]["학습 없는 조건의 가중치 파일 수"] == {"b0": 0, "b1": 0}
    assert by["C7"]["짝이 맞는 칸 수"] == 6 and by["C7"]["매니페스트가 빈 조건 수 (학습 없는 조건)"] == 2
    # C8: B1의 회로 해시가 null이면 통과다 (B1은 회로를 읽지 않는다)
    for d in (root / "T" / "b1_p1", eval_dir(root / "m4_eval", "b1")):
        assert json.loads((d / "environment.json").read_text(encoding="utf-8"))["circuit_sha256"] is None
    assert by["C8"] == {"체크포인트 sha256이 맞는 폴더 수": 10, "회로 sha256이 맞는 폴더 수 (뇌 조건)": 8,
                        "회로 sha256이 빈 폴더 수 (B1)": 2}
    assert by["C9"]["커밋"] == [FAKE_COMMIT] and by["C9"]["읽은 폴더 수"] == 10


# ================================================================ 항목별 실패

def test_c1_fails_on_failed_run_exit_code_or_missing_eval_condition(copy):
    T, E, spec = copy
    edit_json(T / "index.json", lambda d: (d["failed"].append("b0_p1"), d["runs"][3].update(exit_code=1)))
    edit_json(E / "index.json", lambda d: d["runs"].pop())
    it = item(T, E, spec, "C1")
    assert not it["passed"] and it["counts"]["학습 exit_code 0인 조건 수"] == 4
    text = " ".join(it["reasons"])
    assert "failed" in text and "exit_code" in text


def test_c1_fails_on_incomplete_games_csv(copy):
    T, E, spec = copy
    edit_csv(T / "regime2_p1" / "games.csv", lambda cols, rows: (cols, rows[:-1]))
    it = item(T, E, spec, "C1")
    assert not it["passed"] and it["counts"]["games.csv가 완결인 폴더 수"] == 9
    assert "학습 regime2" in it["reasons"][0] and "완결" in it["reasons"][0]


def test_c1_fails_when_censored_fraction_is_over_the_limit(copy):
    T, E, spec = copy

    def censor(cols, rows):
        rows[0]["승자"] = "CENSORED"      # 4판 중 1판 = 0.25 > 0.05
        return cols, rows

    edit_csv(eval_dir(E, "b1") / "games.csv", censor)
    it = item(T, E, spec, "C1")
    assert not it["passed"] and it["counts"]["폴더별 중단 판 수"]["평가 b1"] == 1
    assert len(it["reasons"]) == 1 and "평가 b1" in it["reasons"][0]


def test_c2_fails_on_missing_file_or_old_format(copy):
    T, E, spec = copy
    replay(T / "b1_p1", "b1", 0, 1).unlink()

    def to_format4(a):
        meta = json.loads(str(a["meta"][()]))
        meta["format_version"] = 4
        a["meta"] = np.array(json.dumps(meta, ensure_ascii=False))
        del a["legal"], a["strongest"]

    edit_npz(replay(T / "regime1_p1", "regime1", 1, 2), to_format4)
    it = item(T, E, spec, "C2")
    assert not it["passed"] and it["counts"] == {"기대한 재생 파일 수": 20, "형식이 맞는 파일 수": 18}
    text = " ".join(it["reasons"])
    assert "b1_s0_g1.npz" in text and "형식 4" in text and "legal" in text


def test_c3_catches_avoidance_metric_error(copy):
    T, E, spec = copy

    def break_legal(a):   # 확정된 결정의 행동을 합법이 아니게 한다 → strongest_avoidance가 멈춘다
        a["legal"][:] = False
        a["strongest"][:] = False

    edit_npz(replay(eval_dir(E, "b0"), "b0", 10, 1), break_legal)
    it = item(T, E, spec, "C3")
    assert not it["passed"] and it["counts"]["평가 재생: 오류 없이 계산된 파일 수"] == 7
    assert it["counts"]["학습 재생: 오류 없이 계산된 파일 수"] == 16
    assert "b0_s10_g1.npz" in it["reasons"][0] and "합법 방향이 아니다" in it["reasons"][0]


def test_c3_fails_on_missing_replay_and_on_zero_samples(copy, monkeypatch):
    T, E, spec = copy
    replay(T / "regime3_p1", "regime3", 0, 2).unlink()
    assert not item(T, E, spec, "C3")["passed"]
    monkeypatch.setattr(cm, "strongest_avoidance",
                        lambda a: {0: {"n": 3, "avoid": 0.5, "chance": 0.5}, 1: {"n": 0, "avoid": None, "chance": None}})
    it = item(*copy[:2], spec, "C3")
    assert it["counts"]["표본 수 n (z = 1)"] == 0 and any("z = 1" in x for x in it["reasons"])


def _set_chances(value):
    def fn(cols, rows):
        for r in rows:
            r[cm.CHANCE_COL] = value
        return cols, rows
    return fn


def test_c4_fails_when_b1_has_chances_or_a_brain_condition_has_none(copy):
    T, E, spec = copy
    edit_csv(T / "b1_p1" / "flies.csv", _set_chances(1))
    it = item(T, E, spec, "C4")
    assert not it["passed"] and len(it["reasons"]) == 1 and it["reasons"][0].startswith("b1")
    edit_csv(T / "b1_p1" / "flies.csv", _set_chances(0))
    assert item(T, E, spec, "C4")["passed"]
    edit_csv(T / "b0_p1" / "flies.csv", _set_chances(0))
    it = item(T, E, spec, "C4")
    assert not it["passed"] and it["counts"]["조건별 반전 기회의 합"]["b0"] == 0 and it["reasons"][0].startswith("b0")


def test_c4_fails_without_cause_columns(copy):
    T, E, spec = copy
    edit_csv(T / "regime1_p1" / "flies.csv", lambda cols, rows: (cols[:-4], rows))
    it = item(T, E, spec, "C4")
    assert not it["passed"] and it["counts"]["원인별 네 열이 있는 폴더 수"] == 4 and "파워 시작 반전 기회" in it["reasons"][0]


def test_c5_fails_on_missing_or_zero_field_in_regime3_and_on_field_elsewhere(copy):
    T, E, spec = copy
    edit_npz(replay(T / "regime3_p1", "regime3", 0, 1), lambda a: a.pop("phero_field"))
    edit_npz(replay(T / "regime3_p1", "regime3", 1, 1), lambda a: a["phero_field"].fill(0))
    edit_npz(replay(T / "regime1_p1", "regime1", 0, 1), lambda a: a.update(phero_field=a["odor_field"].copy()))
    it = item(T, E, spec, "C5")
    assert not it["passed"] and len(it["reasons"]) == 3
    assert it["counts"]["체제 3: 페로몬 장이 있고 최댓값 > 0인 파일 수"] == 2
    assert it["counts"]["다른 조건: 페로몬 장이 없는 파일 수"] == 15


def test_c6_fails_on_missing_file_wrong_inner_game_or_stray_file(copy):
    T, E, spec = copy
    name = weight_filename(0, N_GAMES)
    shutil.copy(T / "regime1_p1" / "weights" / name, T / "regime1_p1" / "stray.npz")
    (T / "regime1_p1" / "weights" / name).unlink()
    it = item(T, E, spec, "C6")
    assert not it["passed"] and it["counts"]["판 번호가 맞는 파일 수"] == 5 and name in it["reasons"][0]
    shutil.copy(T / "regime1_p1" / "stray.npz", T / "regime1_p1" / "weights" / name)
    assert item(T, E, spec, "C6")["passed"]
    edit_npz(T / "regime2_p1" / "weights" / name, lambda a: a.update(game_no=np.int64(1)))
    it = item(T, E, spec, "C6")
    assert not it["passed"] and "파일 안 판 번호가 1" in it["reasons"][0]
    (T / "b0_p1" / "weights").mkdir()
    shutil.copy(T / "regime1_p1" / "stray.npz", T / "b0_p1" / "weights" / name)
    it = item(T, E, spec, "C6")
    assert it["counts"]["학습 없는 조건의 가중치 파일 수"] == {"b0": 1, "b1": 0} and len(it["reasons"]) == 2


def test_c7_fails_on_missing_report_bad_manifest_or_input_check_error(copy):
    T, E, spec = copy
    (E / "report.md").unlink()
    it = item(T, E, spec, "C7")
    assert not it["passed"] and it["reasons"] == ["평가 묶음에 report.md이 없다"]
    (E / "report.md").write_text("x", encoding="utf-8")
    assert item(T, E, spec, "C7")["passed"]

    edit_json(eval_dir(E, "regime1") / "eval_manifest.json", lambda d: d["files"][0].update(game_no=1))

    def swap(d):
        d["files"][0]["train_seed"], d["files"][1]["train_seed"] = d["files"][1]["train_seed"], d["files"][0]["train_seed"]

    edit_json(eval_dir(E, "regime2") / "eval_manifest.json", swap)
    edit_json(eval_dir(E, "regime3") / "eval_manifest.json", lambda d: d["files"].pop())
    edit_json(eval_dir(E, "b0") / "eval_manifest.json", lambda d: d["files"].append({"slot": 0}))
    it = item(T, E, spec, "C7")
    c = it["counts"]
    assert not it["passed"] and c["판 번호가 맞는 칸 수"] == 4 and c["짝이 맞는 칸 수"] == 3
    assert c["학습 조건별 매니페스트 칸 수"] == {"regime1": 2, "regime2": 2, "regime3": 1}
    assert c["매니페스트가 빈 조건 수 (학습 없는 조건)"] == 1 and c["입력 검사 (build_results)"] == "오류 없음"


def test_c7_fails_when_report_input_check_stops(copy):
    T, E, spec = copy
    edit_csv(eval_dir(E, "regime3") / "games.csv", lambda cols, rows: (cols, rows + [rows[0]]))   # (시드, 판) 겹침
    it = item(T, E, spec, "C7")
    assert not it["passed"] and it["counts"]["입력 검사 (build_results)"] == "오류"
    assert "EvalSetupError" in it["reasons"][0]


def test_c8_null_circuit_fails_for_brain_condition_and_wrong_hashes_fail(copy):
    T, E, spec = copy
    edit_json(T / "b0_p1" / "environment.json", lambda d: d.update(circuit_sha256=None))
    it = item(T, E, spec, "C8")
    assert not it["passed"] and it["counts"]["회로 sha256이 맞는 폴더 수 (뇌 조건)"] == 7 and "학습 b0" in it["reasons"][0]
    edit_json(eval_dir(E, "b1") / "environment.json", lambda d: d.update(circuit_sha256="0" * 64))
    edit_json(eval_dir(E, "regime1") / "environment.json", lambda d: d.update(pacman_checkpoint_sha256="1" * 64))
    edit_json(T / "regime2_p1" / "environment.json", lambda d: d.update(pacman_checkpoint_sha256=None))
    it = item(T, E, spec, "C8")
    assert it["counts"] == {"체크포인트 sha256이 맞는 폴더 수": 8, "회로 sha256이 맞는 폴더 수 (뇌 조건)": 7,
                            "회로 sha256이 빈 폴더 수 (B1)": 1}


def test_c8_compares_circuit_with_the_file_given(copy, tmp_path):
    T, E, spec = copy
    other = tmp_path / "other.npz"
    other.write_bytes(b"not the circuit")
    it = [i for i in cm.run_checks(T, E, spec, circuit=other)["items"] if i["id"] == "C8"][0]
    assert not it["passed"] and it["counts"]["회로 sha256이 맞는 폴더 수 (뇌 조건)"] == 0
    assert it["counts"]["회로 sha256이 빈 폴더 수 (B1)"] == 2


def test_c9_fails_on_dirty_or_different_commit(copy):
    T, E, spec = copy
    (T / "regime1_p1" / "git_commit.txt").write_text(FAKE_COMMIT + "-dirty\n", encoding="utf-8")
    it = item(T, E, spec, "C9")
    assert not it["passed"] and it["counts"]["-dirty인 폴더 수"] == 1
    (T / "regime1_p1" / "git_commit.txt").write_text("d" * 40 + "\n", encoding="utf-8")
    it = item(T, E, spec, "C9")
    assert not it["passed"] and it["counts"]["서로 다른 커밋 수"] == 2 and it["counts"]["-dirty인 폴더 수"] == 0


# ================================================================ 예외 격리 · 출력

def test_exception_in_one_item_does_not_stop_the_others(copy):
    T, E, spec = copy
    (T / "index.json").unlink()
    res = cm.run_checks(T, E, spec)
    by = {i["id"]: i for i in res["items"]}
    assert not res["passed"] and not by["C1"]["passed"] and "FileNotFoundError" in by["C1"]["reasons"][0]
    assert all(by[f"C{k}"]["passed"] for k in range(2, 10))


def _no_performance_values(obj, path="") -> None:
    """m4_check.json 안에는 정수 개수 · 글 · 참거짓만 있다 (소수가 없다 = 비율 · 평균 · 구간이 없다)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            _no_performance_values(v, f"{path}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _no_performance_values(v, f"{path}[{i}]")
    else:
        assert isinstance(obj, (bool, int, str)), f"{path}: {obj!r}"


def _plant_sentinels(T: Path, E: Path) -> None:
    def games(cols, rows):
        for r in rows:
            r["점수"] = SENTINEL
        return cols, rows

    def flies(cols, rows):
        for r in rows:
            r["합계 R"] = SENTINEL
        return cols, rows

    for d in [*T.glob("*_p1"), *E.parent.glob("x_m4_eval_v1_*")]:
        edit_csv(d / "games.csv", games)
        edit_csv(d / "flies.csv", flies)


def _write_cfgs(tmp_path):
    t, e = tmp_path / "train.yaml", tmp_path / "eval.yaml"
    t.write_text(yaml.safe_dump(_train_cfg(), allow_unicode=True), encoding="utf-8")
    e.write_text(yaml.safe_dump(_eval_cfg(), allow_unicode=True), encoding="utf-8")
    return ["--train-config", str(t), "--eval-config", str(e)]


def test_main_writes_one_file_and_no_performance_values(copy, tmp_path, capsys):
    T, E, _ = copy
    _plant_sentinels(T, E)
    before = {p for p in T.parent.rglob("*")}
    capsys.readouterr()
    assert cm.main(["check_m4.py", str(T), str(E), *_write_cfgs(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert {p for p in T.parent.rglob("*")} - before == {T / "m4_check.json"}       # 쓰는 파일은 하나다
    text = (T / "m4_check.json").read_text(encoding="utf-8")
    saved = json.loads(text)
    assert saved["passed"] and saved["train_bundle"] == "T" and saved["eval_bundle"] == "m4_eval"
    _no_performance_values(saved)
    for blob in (out, text):
        assert SENTINEL not in blob
        for word in ("점수", "score", "avoid", "chance", "fly_wins", "pacman_wins", "W_mean", "합계 R", "신뢰구간", "deltas"):
            assert word not in blob, word
    assert out.count("[통과]") == 9 and "[실패]" not in out
    # 화면의 근거 줄(네 칸 들여쓴 줄)에도 소수가 없다 (비율 · 평균이 없다)
    evidence = [line for line in out.splitlines() if line.startswith("    ")]
    assert len(evidence) > 9 and not any(re.search(r"\d\.\d", line) for line in evidence)


def test_main_returns_1_on_failure_and_still_hides_values(copy, tmp_path, capsys):
    T, E, _ = copy
    _plant_sentinels(T, E)
    edit_csv(eval_dir(E, "regime1") / "games.csv", lambda cols, rows: (cols, rows[1:]))   # C1 · C7 실패
    capsys.readouterr()
    assert cm.main(["check_m4.py", str(T), str(E), *_write_cfgs(tmp_path)]) == 1
    out = capsys.readouterr().out
    saved = json.loads((T / "m4_check.json").read_text(encoding="utf-8"))
    failed = [i["id"] for i in saved["items"] if not i["passed"]]
    assert failed == ["C1", "C7"] and not saved["passed"] and out.count("[실패]") == 2
    _no_performance_values(saved)
    assert SENTINEL not in out and SENTINEL not in json.dumps(saved, ensure_ascii=False)


def test_main_stops_without_train_bundle(tmp_path, capsys):
    assert cm.main(["check_m4.py", str(tmp_path / "none"), str(tmp_path / "e")]) == 2
    assert "멈춤" in capsys.readouterr().err
