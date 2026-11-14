"""합의 가중치 평가 (DECISIONS 37).

  consensus_eval.py run <configs/consensus_eval_v1.yaml> <학습 결과 폴더> ...   합의 묶음 (체크포인트 필요)
  consensus_eval.py compare <합의 묶음> <개별 묶음>                            합의 − 개별의 짝 비교

run은 `final_eval.run`과 같은 순서로 돌고, 다른 것은 가중치뿐이다: 학습 조건마다 초파리 번호 f별로 학습 시드
(`final_eval.train_seeds`)의 마지막 판 `dw_masked[f]` · `rbar[f]`를 산술 평균한 **합의 가중치** 하나를 만들어 칸 모두에
넣는다. 시작 전 확인 · 조건 설정 · 묶음 index.json · 보고서는 `final_eval`의 함수를 그대로 부른다
(그래서 `final_eval.py report <합의 묶음>`도 그대로 돈다).

compare는 두 묶음이 `final_eval.build_results`의 입력 검사를 통과하고 평가 시드 · 판 수 · 조건이 같으며 B0의
games.csv가 (시드, 판)으로 정렬한 행까지 바이트 단위로 같을 때만, 체제마다 합의 − 개별의 짝 차이를 낸다.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import final_eval as fe  # noqa: E402
import select_eps  # noqa: E402

from flypac.core.config import load_config  # noqa: E402

META_KEYS = ("format", "condition", "circuit_sha256", "brain_config_hash")   # 원본 파일끼리 같아야 하는 값 (37, 1.2-2)
CONSENSUS_DIR = "consensus_weights"


# ---------------------------------------------------------------- 합의 가중치 (순수)

def consensus_state(states: list[dict], condition: str) -> dict:
    """원본 가중치 사전(`read_weight_file`의 모양) 여럿을 평균한 합의 상태. `load_slot_state`에 그대로 넘길 수 있다.

    형식 · 조건 · 회로 해시 · 뇌 설정 해시가 모든 원본에서 같고 조건이 condition이어야 한다. 모양도 같아야 한다.
    다르면 fe.EvalSetupError. dw_masked와 rbar는 원소별 산술 평균 (float64)이다: 초파리 번호 f의 값은
    시드들의 f번 값만의 평균이다."""
    if not states:
        raise fe.EvalSetupError(f"{condition}: 합의에 쓸 가중치 파일이 없다")
    ref = states[0]
    for i, st in enumerate(states):
        for k in META_KEYS:
            if st[k] != ref[k]:
                raise fe.EvalSetupError(f"{condition}: 원본 가중치의 {k}가 다르다 (파일 0: {ref[k]!r}, 파일 {i}: {st[k]!r})")
        for k in ("dw_masked", "rbar"):
            if np.shape(st[k]) != np.shape(ref[k]):
                raise fe.EvalSetupError(f"{condition}: 원본 가중치의 {k} 모양이 다르다 "
                                        f"({np.shape(ref[k])} / {np.shape(st[k])})")
    if ref["condition"] != condition:
        raise fe.EvalSetupError(f"{condition}: 원본 가중치의 조건이 {ref['condition']}이다")
    dw = np.mean(np.stack([np.asarray(s["dw_masked"], dtype=np.float64) for s in states]), axis=0)
    rbar = np.mean(np.stack([np.asarray(s["rbar"], dtype=np.float64) for s in states]), axis=0)
    return {**{k: ref[k] for k in META_KEYS}, "dw_masked": dw, "rbar": rbar}


def state_sha256(state: dict) -> dict:
    """합의 가중치의 sha256: dw_masked · rbar 각각의 바이트 (float64, C 순서)."""
    return {k: hashlib.sha256(np.ascontiguousarray(state[k], dtype=np.float64).tobytes()).hexdigest()
            for k in ("dw_masked", "rbar")}


def save_consensus(path: Path, state: dict, train_seeds: list[int], game_no: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        np.savez(fh, dw_masked=state["dw_masked"], rbar=state["rbar"], format=np.int64(state["format"]),
                 condition=np.str_(state["condition"]), circuit_sha256=np.str_(state["circuit_sha256"]),
                 brain_config_hash=np.str_(state["brain_config_hash"]),
                 train_seeds=np.asarray(train_seeds, dtype=np.int64), game_no=np.int64(game_no))


# ---------------------------------------------------------------- run

def preflight(base: dict, train_dirs: list[Path], runs_root: Path) -> dict:
    """`final_eval.run`의 시작 전 확인 중 체크포인트 · 장치를 뺀 것. 가중치 목록을 돌려준다. 실패하면 EvalSetupError."""
    fe.check_threads()
    fe_blk = base["final_eval"]
    train_seeds, eval_seeds = [int(s) for s in fe_blk["train_seeds"]], [int(s) for s in base["seeds"]]
    if len(train_seeds) != len(eval_seeds):
        raise fe.EvalSetupError("train_seeds와 seeds의 길이가 다르다 (칸 수)")
    weights = fe.collect_weights(train_dirs, list(fe_blk["conditions"]), train_seeds,
                                 train_n_games=int(fe_blk["train_n_games"]))
    fe.check_train_dirs(base, train_dirs)
    out_dir = Path(runs_root) / str(base["experiment"])
    if (out_dir / "index.json").exists():
        raise fe.EvalSetupError(f"{out_dir / 'index.json'}가 이미 있다")
    return weights


def check_hardware(base: dict) -> None:
    """체크포인트 sha256과 장치 (final_eval.run과 같은 확인)."""
    ckpt = Path(base["pacman"]["checkpoint"])
    if not ckpt.exists() or select_eps._sha256(ckpt) != select_eps.CHECKPOINT_SHA256:
        raise fe.EvalSetupError(f"체크포인트가 없거나 sha256이 DECISIONS 14.6의 값과 다르다: {ckpt}")
    from flypac.pacman.trainer import resolve_device  # torch는 여기서 처음 필요하다

    if resolve_device(str(base["pacman"]["device"])).type != "cuda":
        raise fe.EvalSetupError("실제 장치가 cuda가 아니다")


def execute(base: dict, weights: dict, runs_root: Path) -> Path:
    """확인을 통과한 뒤의 실행. 조건마다 결과 폴더 하나, 묶음 index.json을 쓰고 묶음 폴더를 돌려준다."""
    from flypac.core.runner import build_runner
    from flypac.recording.recorder import Recorder, make_run_dir
    from flypac.recording.weights import read_weight_file

    fe_blk = base["final_eval"]
    conditions, train_seeds = list(fe_blk["conditions"]), [int(s) for s in fe_blk["train_seeds"]]
    eval_seeds = [int(s) for s in base["seeds"]]
    out_dir = Path(runs_root) / str(base["experiment"])
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for cond in conditions:
        cfg = fe.eval_config(base, cond)
        run_dir = make_run_dir(cfg, Path(runs_root))
        print(f"[{cond}] 시작: {run_dir}", flush=True)
        runner = build_runner(cfg, Recorder(run_dir, cfg))
        manifest = {"condition": cond, "kind": "consensus", "files": [], "sources": [], "consensus": None}
        if cond in fe.LEARNING:
            hits = [weights[(cond, s)] for s in train_seeds]
            state = consensus_state([read_weight_file(h["path"]) for h in hits], cond)
            for i in range(len(eval_seeds)):            # 칸 모두에 같은 합의 가중치 (37.3)
                runner.ghosts.load_slot_state(i, state)
            rel = Path(CONSENSUS_DIR) / f"{cond}.npz"
            save_consensus(run_dir / rel, state, train_seeds, hits[0]["game_no"])
            manifest["sources"] = [{"train_seed": s, "file": str(h["path"]), "game_no": h["game_no"],
                                    "sha256": fe._sha256(h["path"])} for s, h in zip(train_seeds, hits)]
            manifest["consensus"] = {"file": str(rel), "sha256": state_sha256(state),
                                     "file_sha256": fe._sha256(run_dir / rel)}
            manifest["files"] = [{"slot": i, "eval_seed": e, "consensus": str(rel)} for i, e in enumerate(eval_seeds)]
        (run_dir / "eval_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                                    encoding="utf-8")
        summary = runner.run()
        wins = [g["winner"] for g in summary["games"]]
        print(f"[{cond}] 끝: {summary['ticks']}틱, {len(wins)}판 (초파리 {wins.count('FLIES')}, "
              f"팩맨 {wins.count('PACMAN')}, 중단 {wins.count('CENSORED')})", flush=True)
        runs.append({"condition": cond, "run_dir": run_dir.name})
        (out_dir / "index.json").write_text(json.dumps({"runs": runs}, ensure_ascii=False, indent=2) + "\n",
                                            encoding="utf-8")
    return out_dir


def run(config_path: Path, train_dirs: list[Path], runs_root: Path = Path("runs")) -> int:
    base = load_config(config_path).to_dict()
    try:
        weights = preflight(base, train_dirs, runs_root)
        check_hardware(base)
    except fe.EvalSetupError as e:
        print(f"멈춤: {e}", file=sys.stderr)
        return 1
    return fe.report(execute(base, weights, runs_root))


# ---------------------------------------------------------------- compare

def bundle_common(bundle: Path) -> dict:
    """묶음의 공통 값 (조건 · 시드 · 판 수 …, fe.check_bundle)과 조건별 결과 폴더."""
    index = json.loads((bundle / "index.json").read_text(encoding="utf-8"))
    cfgs = {str(r["condition"]): yaml.safe_load((bundle.parent / r["run_dir"] / "config.yaml").read_text(encoding="utf-8"))
            for r in index["runs"]}
    return {**fe.check_bundle(index["runs"], cfgs),
            "dirs": {str(r["condition"]): bundle.parent / r["run_dir"] for r in index["runs"]}}


def sorted_rows(path: Path) -> list[bytes]:
    """games.csv의 머리줄과, (시드, 판)으로 정렬한 행들 (바이트, BOM · 줄끝 그대로)."""
    lines = Path(path).read_bytes().splitlines(keepends=True)
    head, rows = lines[0], lines[1:]
    cols = next(csv.reader([head.decode("utf-8-sig")]))
    i_s, i_g = cols.index("시드"), cols.index("판")

    def key(line: bytes):
        f = next(csv.reader([line.decode("utf-8")]))
        return int(f[i_s]), int(f[i_g])

    return [head] + sorted(rows, key=key)


def check_b0_identical(a: Path, b: Path) -> None:
    ra, rb = sorted_rows(a), sorted_rows(b)
    if ra != rb:
        n = next((i for i, (x, y) in enumerate(zip(ra, rb)) if x != y), min(len(ra), len(rb)))
        raise fe.EvalSetupError(f"두 묶음의 b0 games.csv가 다르다 (정렬한 {n}번째 줄, 행 수 {len(ra)} / {len(rb)}): "
                                f"{a} / {b}")


def build_compare(cons: Path, indiv: Path) -> dict:
    res_c, res_i = fe.build_results(cons), fe.build_results(indiv)       # 입력 검사 (33.1)
    com_c, com_i = bundle_common(cons), bundle_common(indiv)
    for k in ("seeds", "n_games", "conditions", "ci_level"):
        if com_c[k] != com_i[k]:
            raise fe.EvalSetupError(f"두 묶음의 {k}가 다르다: {com_c[k]} / {com_i[k]}")
    if "b0" not in com_c["dirs"]:
        raise fe.EvalSetupError("묶음에 b0가 없다 (B0 동일성 확인)")
    check_b0_identical(com_c["dirs"]["b0"] / "games.csv", com_i["dirs"]["b0"] / "games.csv")
    level = com_c["ci_level"]
    rows = []
    for cond in [c for c in fe.LEARNING if c in com_c["conditions"]]:
        mc, mi = res_c["metrics"][cond], res_i["metrics"][cond]
        order = [int(s) for s in mc["seeds"]]
        row = {"condition": cond, "a": f"합의 {cond}", "b": f"개별 {cond}", "seeds": order}
        for m in ("score", "W"):
            xa, xb = dict(zip(mc["seeds"], mc[m])), dict(zip(mi["seeds"], mi[m]))    # 평가 시드 값으로 맞춘다
            xs, ys = [xa[s] for s in order], [xb[s] for s in order]
            row[m] = None if None in xs or None in ys else fe.paired_ci(xs, ys, level)
        rows.append(row)
    return {"consensus_bundle": str(cons), "indiv_bundle": str(indiv), "eval_seeds": com_c["seeds"],
            "n_games": com_c["n_games"], "ci_level": level, "b0_identical": True, "comparisons": rows,
            "metrics": {"consensus": res_c["metrics"], "indiv": res_i["metrics"]}}


def render_compare(res: dict) -> str:
    lvl = int(round(res["ci_level"] * 100))
    L = ["# 합의 − 개별 (DECISIONS 37.4 부 질문)", "",
         f"- 합의 묶음 `{res['consensus_bundle']}`, 개별 묶음 `{res['indiv_bundle']}`. 평가 시드 {res['eval_seeds']}, "
         f"칸마다 {res['n_games']}판.",
         f"- 짝 차이 Δ_s = Y_s(합의) − Y_s(개별), 같은 평가 시드. 평균과 {lvl}% 신뢰구간 (t 분포, 자유도 n − 1). "
         "중단 판은 뺀다 (28.4).",
         "- B0 동일성: 두 묶음의 b0 games.csv가 (시드, 판)으로 정렬한 행까지 바이트 단위로 같다.",
         "- **다중 비교 보정은 하지 않았다.**"]
    for m, title, nd in (("score", "팩맨 평균 점수 (대표)", 1), ("W", "초파리 승률 W (보조)", 4)):
        L += ["", f"## {title}", "",
              f"| 앞 − 뒤 | 평균 Δ | {lvl}% 신뢰구간 | 0 제외 | 앞이 더 낮은 쌍 수 |", "|---|---|---|---|---|"]
        for r in res["comparisons"]:
            ci = r[m]
            if ci is None:
                L.append(f"| {r['a']} − {r['b']} | - | - | - | - |")
                continue
            lower = sum(d < 0 for d in ci["deltas"])
            L.append(f"| {r['a']} − {r['b']} | {fe._f(ci['mean'], nd)} | [{fe._f(ci['lo'], nd)}, {fe._f(ci['hi'], nd)}] | "
                     f"{'예' if ci['excludes_zero'] else '아니오'} | {lower}/{ci['n']} |")
    L += ["", "## 짝 차이 (Δ_s, 평가 시드 순서)", ""]
    for r in res["comparisons"]:
        for m, nd in (("score", 1), ("W", 4)):
            if r[m]:
                L.append(f"- {r['a']} − {r['b']} [{m}]: " + ", ".join(fe._f(d, nd) for d in r[m]["deltas"]))
    return "\n".join(L) + "\n"


def compare(cons: Path, indiv: Path) -> int:
    res = build_compare(Path(cons), Path(indiv))
    (Path(cons) / "compare.json").write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    text = render_compare(res)
    (Path(cons) / "compare.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


def main(argv: list[str]) -> int:
    if len(argv) >= 4 and argv[1] == "run":
        return run(Path(argv[2]), [Path(p) for p in argv[3:]])
    if len(argv) == 4 and argv[1] == "compare":
        try:
            return compare(Path(argv[2]), Path(argv[3]))
        except fe.EvalSetupError as e:   # 입력 · B0 확인 실패: 결과를 쓰지 않는다
            print(f"멈춤: {e}", file=sys.stderr)
            return 1
    print("사용법: consensus_eval.py run <설정.yaml> <학습 결과 폴더> ... | "
          "consensus_eval.py compare <합의 묶음> <개별 묶음>", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
