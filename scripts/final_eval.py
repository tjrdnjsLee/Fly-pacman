"""최종 평가 (DECISIONS 26.3 · 26.4).

  final_eval.py run <평가 설정.yaml> <학습 결과 폴더> ...   조건마다 평가 실행 하나 (체크포인트 필요)
  final_eval.py report <평가 묶음 폴더>                     games.csv만 읽어 26.4의 표 (예: runs/final_eval_v1)

칸 i는 평가 시드 e_i로 게임을 하고, 학습 조건이면 학습 시드 s_i의 마지막 가중치를 쓴다 (짝은 위치 순서).
계산(`per_seed_metrics`, `paired_ci`, `compare`)은 순수 함수이고 t 분포 임계값은 scipy.stats.t.ppf로 구한다.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import statistics
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import select_eps  # noqa: E402

from flypac.core.config import Config, load_config  # noqa: E402
from flypac.recording.weights import parse_weight_filename  # noqa: E402

LEARNING = ("regime1", "regime2", "regime3")
SHARED_BLOCKS = ("engine", "time", "odor", "phero", "pacman")  # 학습 실행과 같아야 하는 블록 (DECISIONS 28.4)
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
CSV_ENCODING = "utf-8-sig"


class EvalSetupError(ValueError):
    """시작 전 확인 실패: 멈춘다."""


# ---------------------------------------------------------------- 가중치 파일 모으기

def collect_weights(train_dirs: list[Path], conditions: list[str], train_seeds: list[int]) -> dict:
    """{(조건, 학습 시드): {"path", "game_no"}}. 학습 조건마다 학습 시드 하나에 마지막 판 파일이 정확히 하나가
    아니면 EvalSetupError. 마지막 판은 그 학습 실행의 n_games여야 한다 (덜 돈 실행을 평가하지 않는다)."""
    found: dict[tuple[str, int], list[dict]] = {}
    for d in map(Path, train_dirs):
        cfg = yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8"))
        cond, n_games = str(cfg["condition"]), int(cfg["n_games"])
        per_seed: dict[int, list[tuple[int, Path]]] = {}
        for p in sorted((d / "weights").glob("*.npz")):
            parsed = parse_weight_filename(p.name)
            if parsed:
                per_seed.setdefault(parsed[0], []).append((parsed[1], p))
        for seed, items in per_seed.items():
            g, p = max(items)
            found.setdefault((cond, seed), []).append({"path": p, "game_no": g, "n_games": n_games})
    out = {}
    for cond in conditions:
        if cond not in LEARNING:
            continue
        for s in train_seeds:
            hits = found.get((cond, int(s)), [])
            if len(hits) != 1:
                raise EvalSetupError(f"{cond} 학습 시드 {s}의 가중치 파일이 {len(hits)}개다 (정확히 1개여야 한다)")
            if hits[0]["game_no"] != hits[0]["n_games"]:
                raise EvalSetupError(f"{cond} 학습 시드 {s}: 마지막 파일이 판 {hits[0]['game_no']}이고 학습은 "
                                     f"{hits[0]['n_games']}판이다 (끝까지 돈 실행이 아니다)")
            out[(cond, int(s))] = hits[0]
    return out


# ---------------------------------------------------------------- 시작 전 확인 (DECISIONS 28.4)

def check_threads(env=None) -> None:
    """수치 라이브러리 스레드가 모두 "1"이 아니면 EvalSetupError (붙여야 할 명령을 메시지에 담는다)."""
    env = os.environ if env is None else env
    bad = [v for v in THREAD_VARS if env.get(v) != "1"]
    if bad:
        prefix = " ".join(f"{v}=1" for v in THREAD_VARS)
        raise EvalSetupError(f"스레드 환경변수가 1이 아니다: {', '.join(bad)}. 다음처럼 붙여 다시 실행한다: "
                             f"{prefix} uv run --extra pacman python scripts/final_eval.py run ...")


def recorder_config_hash(cfg: dict) -> str:
    """기록기와 같은 방식의 실행 설정 해시: weights 블록을 뺀 사전의 Config 해시 (Recorder.config_hash)."""
    return Config({k: v for k, v in cfg.items() if k != "weights"}).hash()


def read_weight_meta(path: Path) -> dict:
    """가중치 파일에서 시드와 실행 설정 해시만 읽는다 (큰 배열은 읽지 않는다)."""
    with np.load(Path(path), allow_pickle=False) as z:
        return {"seed": int(z["seed"]), "run_config_hash": str(z["run_config_hash"])}


def check_train_dirs(base: dict, train_dirs: list[Path]) -> None:
    """학습 폴더마다: 판 수, 공통 블록, 가중치 파일의 시드 · 실행 설정 해시를 평가 설정과 대조한다. 다르면 EvalSetupError."""
    expected_n = int(base["final_eval"]["train_n_games"])
    for d in map(Path, train_dirs):
        cfg = yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8"))
        if int(cfg["n_games"]) != expected_n:
            raise EvalSetupError(f"{d}: 학습 판 수가 {cfg['n_games']}이다 (평가 설정의 train_n_games = {expected_n})")
        for k in SHARED_BLOCKS:
            if cfg.get(k) != base.get(k):
                raise EvalSetupError(f"{d}: 설정의 {k} 블록이 평가 설정과 다르다")
        want = recorder_config_hash(cfg)
        hashes = set()
        for p in sorted((d / "weights").glob("*.npz")):
            parsed = parse_weight_filename(p.name)
            if not parsed:
                continue
            meta = read_weight_meta(p)
            if meta["seed"] != parsed[0]:
                raise EvalSetupError(f"{p}: 파일 안의 시드 {meta['seed']}가 파일 이름의 시드 {parsed[0]}와 다르다")
            hashes.add(meta["run_config_hash"])
        if len(hashes) > 1:
            raise EvalSetupError(f"{d}: 가중치 파일들의 run_config_hash가 하나가 아니다 ({sorted(hashes)})")
        if hashes and hashes != {want}:
            raise EvalSetupError(f"{d}: 파일의 run_config_hash {sorted(hashes)[0]}가 config.yaml로 다시 계산한 {want}와 다르다 "
                                 "(계산이 재현되지 않는다)")


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- 계산 (순수)

def per_seed_metrics(rows: list[dict], seeds: list[int]) -> dict:
    """games.csv 행들에서 시드(평가 칸)별 팩맨 평균 점수 Y_s와 초파리 승률 W_s. seeds 순서를 따른다.
    중단(CENSORED) 판은 점수와 W 모두에서 뺀다 (DECISIONS 28.4). 시드별 중단 수는 `censored`에 낸다."""
    score, W, games, censored = [], [], [], []
    for s in seeds:
        mine = [r for r in rows if int(r["시드"]) == int(s)]
        done = [r for r in mine if r["승자"] != "CENSORED"]
        fly = sum(r["승자"] == "FLIES" for r in done)
        pac = sum(r["승자"] == "PACMAN" for r in done)
        games.append(len(mine))
        censored.append(len(mine) - len(done))
        score.append(statistics.fmean(int(r["점수"]) for r in done) if done else None)
        W.append(fly / (fly + pac) if fly + pac else None)
    return {"score": score, "W": W, "games": games, "censored": censored}


def paired_ci(x: list[float], y: list[float], level: float) -> dict:
    """짝 차이 Δ_s = x_s - y_s의 평균과 level 신뢰구간 (t 분포, 자유도 n - 1)."""
    from scipy import stats

    if len(x) != len(y) or len(x) < 2:
        raise ValueError("짝 차이에는 같은 길이 2 이상의 목록이 필요하다")
    deltas = [a - b for a, b in zip(x, y)]
    n = len(deltas)
    mean, sd = statistics.fmean(deltas), statistics.stdev(deltas)
    se = sd / n ** 0.5
    tcrit = float(stats.t.ppf(1 - (1 - level) / 2, n - 1))
    lo, hi = mean - tcrit * se, mean + tcrit * se
    return {"deltas": deltas, "n": n, "df": n - 1, "mean": mean, "sd": sd, "se": se, "t": tcrit,
            "lo": lo, "hi": hi, "excludes_zero": bool(lo > 0 or hi < 0)}


def compare(metrics: dict[str, dict], level: float) -> list[dict]:
    """조건 쌍의 짝 차이. 체제 i vs B0 (학습 효과 판단), 체제끼리, 참고로 B1과의 차이."""
    regimes = [c for c in LEARNING if c in metrics]
    pairs = [(r, "b0", "학습 효과") for r in regimes if "b0" in metrics]
    pairs += [(a, b, "체제 비교") for a, b in combinations(regimes, 2)]
    pairs += [(c, "b1", "참고") for c in (*regimes, "b0") if c in metrics and "b1" in metrics]
    out = []
    for a, b, kind in pairs:
        row = {"a": a, "b": b, "kind": kind}
        for m in ("score", "W"):
            xs, ys = metrics[a][m], metrics[b][m]
            row[m] = None if None in xs or None in ys else paired_ci(xs, ys, level)
        out.append(row)
    return out


# ---------------------------------------------------------------- 보고

def read_games(path: Path) -> list[dict]:
    with open(path, encoding=CSV_ENCODING, newline="") as fh:
        return list(csv.DictReader(fh))


def _f(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def build_results(bundle: Path) -> dict:
    index = json.loads((bundle / "index.json").read_text(encoding="utf-8"))
    root = bundle.parent
    metrics, summary, level, seeds, warn = {}, {}, None, None, None
    for r in index["runs"]:
        d = root / r["run_dir"]
        cfg = yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8"))
        seeds = [int(s) for s in cfg["seeds"]]
        level = float(cfg["final_eval"]["ci_level"])
        warn = float(cfg["final_eval"]["censored_warn_fraction"])
        rows = read_games(d / "games.csv")
        metrics[r["condition"]] = per_seed_metrics(rows, seeds)
        fly, pac = sum(x["승자"] == "FLIES" for x in rows), sum(x["승자"] == "PACMAN" for x in rows)
        summary[r["condition"]] = {
            "games": len(rows), "fly_wins": fly, "pacman_wins": pac,
            "censored": sum(x["승자"] == "CENSORED" for x in rows),
            "censored_fraction": (sum(x["승자"] == "CENSORED" for x in rows) / len(rows)) if rows else None,
            "tick_median": statistics.median(int(x["틱 수"]) for x in rows) if rows else None,
            "score_mean_of_seed_means": (statistics.fmean(v for v in metrics[r["condition"]]["score"] if v is not None)
                                         if any(v is not None for v in metrics[r["condition"]]["score"]) else None),
            "W_mean_of_seed_means": (statistics.fmean(metrics[r["condition"]]["W"])
                                     if None not in metrics[r["condition"]]["W"] else None)}
    return {"eval_seeds": seeds, "ci_level": level, "censored_warn_fraction": warn, "metrics": metrics, "summary": summary,
            "comparisons": compare(metrics, level), "runs": index["runs"]}


def render_report(res: dict) -> str:
    lvl = int(round(res["ci_level"] * 100))
    warn = res["censored_warn_fraction"]
    flagged = [c for c, s in res["summary"].items() if s["censored_fraction"] is not None and s["censored_fraction"] > warn]
    L = ["# 최종 평가 (DECISIONS 26.4)", ""]
    if flagged:
        L += [f"> **해석 주의: 중단 비율.** 중단 비율이 {warn}를 넘는 조건: "
              + ", ".join(f"{c} ({res['summary'][c]['censored_fraction']:.3f})" for c in flagged), ""]
    L += [
         f"- 독립 반복 단위는 학습 시드다. 시드별 평균 Y_s를 먼저 내고 짝 차이 Δ_s = Y_s(앞) − Y_s(뒤)의 평균과 {lvl}% 신뢰구간"
         f"(t 분포, 자유도 n − 1)을 낸다. 평가 시드: {res['eval_seeds']}",
         "- 대표 지표는 팩맨 평균 점수(낮을수록 초파리 우수), 보조 지표는 초파리 승률 W다. 중단(CENSORED) 판은 둘 다에서 뺀다 (28.4).",
         "- **다중 비교 보정은 하지 않았다.**", "", "## 조건별", "",
         "| 조건 | 판 수 | 초파리 승 | 팩맨 승 | 중단 | 판 길이(틱) 중앙값 | 팩맨 점수 (시드 평균의 평균) | W (시드 평균의 평균) |",
         "|---|---|---|---|---|---|---|---|"]
    for c, s in res["summary"].items():
        L.append(f"| {c} | {s['games']} | {s['fly_wins']} | {s['pacman_wins']} | {s['censored']} | "
                 f"{_f(s['tick_median'], 1)} | {_f(s['score_mean_of_seed_means'], 1)} | {_f(s['W_mean_of_seed_means'])} |")
    for m, title, nd in (("score", "팩맨 평균 점수 (대표)", 1), ("W", "초파리 승률 W (보조)", 4)):
        L += ["", f"## 짝 차이: {title}", "",
              f"| 구분 | 앞 − 뒤 | 평균 Δ | {lvl}% 신뢰구간 | 0 제외 | 앞이 더 낮은 쌍 수 |", "|---|---|---|---|---|---|"]
        for r in res["comparisons"]:
            ci = r[m]
            if ci is None:
                L.append(f"| {r['kind']} | {r['a']} − {r['b']} | - | - | - | - |")
                continue
            lower = sum(d < 0 for d in ci["deltas"])
            L.append(f"| {r['kind']} | {r['a']} − {r['b']} | {_f(ci['mean'], nd)} | [{_f(ci['lo'], nd)}, {_f(ci['hi'], nd)}] | "
                     f"{'예' if ci['excludes_zero'] else '아니오'} | {lower}/{ci['n']} |")
    L += ["", "## 시드별 중단 수", "", "| 조건 | " + " | ".join(str(s) for s in res["eval_seeds"]) + " |",
          "|---|" + "---|" * len(res["eval_seeds"])]
    for c, mm in res["metrics"].items():
        L.append(f"| {c} | " + " | ".join(str(v) for v in mm["censored"]) + " |")
    L += ["", "## 시드별 값 (10쌍 모두)", "", "| 조건 | 지표 | " + " | ".join(str(s) for s in res["eval_seeds"]) + " |",
          "|---|---|" + "---|" * len(res["eval_seeds"])]
    for c, mm in res["metrics"].items():
        for m, nd in (("score", 1), ("W", 3)):
            L.append(f"| {c} | {m} | " + " | ".join(_f(v, nd) for v in mm[m]) + " |")
    L += ["", "## 짝 차이 (Δ_s, 평가 시드 순서)", ""]
    for r in res["comparisons"]:
        for m, nd in (("score", 1), ("W", 4)):
            if r[m]:
                L.append(f"- {r['a']} − {r['b']} [{m}]: " + ", ".join(_f(d, nd) for d in r[m]["deltas"]))
    return "\n".join(L) + "\n"


def report(bundle: Path) -> int:
    res = build_results(bundle)
    (bundle / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    text = render_report(res)
    (bundle / "report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


# ---------------------------------------------------------------- 평가 실행

def eval_config(base: dict, condition: str) -> Config:
    """조건 하나의 평가 설정: condition, ghosts, evaluate, experiment를 채운다."""
    d = json.loads(json.dumps(base))
    fe = d["final_eval"]
    d["condition"] = condition
    d["ghosts"] = dict(fe["ghosts"]["b1" if condition == "b1" else "flybrain"])
    d["evaluate"] = True
    d["experiment"] = f"{base['experiment']}_{condition}"
    return Config(d)


def run(config_path: Path, train_dirs: list[Path]) -> int:
    from flypac.core.runner import build_runner
    from flypac.recording.recorder import Recorder, make_run_dir
    from flypac.recording.weights import read_weight_file

    base = load_config(config_path).to_dict()
    fe = base["final_eval"]
    conditions, train_seeds = list(fe["conditions"]), [int(s) for s in fe["train_seeds"]]
    eval_seeds = [int(s) for s in base["seeds"]]
    out_dir = Path("runs") / str(base["experiment"])
    try:
        check_threads()
        if len(train_seeds) != len(eval_seeds):
            raise EvalSetupError("train_seeds와 seeds의 길이가 다르다 (위치로 짝짓는다)")
        weights = collect_weights(train_dirs, conditions, train_seeds)
        check_train_dirs(base, train_dirs)
        ckpt = Path(base["pacman"]["checkpoint"])
        if not ckpt.exists() or select_eps._sha256(ckpt) != select_eps.CHECKPOINT_SHA256:
            raise EvalSetupError(f"체크포인트가 없거나 sha256이 DECISIONS 14.6의 값과 다르다: {ckpt}")
        from flypac.pacman.trainer import resolve_device  # torch는 여기서 처음 필요하다

        if resolve_device(str(base["pacman"]["device"])).type != "cuda":
            raise EvalSetupError("실제 장치가 cuda가 아니다")
        if (out_dir / "index.json").exists():
            raise EvalSetupError(f"{out_dir / 'index.json'}가 이미 있다")
    except EvalSetupError as e:
        print(f"멈춤: {e}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for cond in conditions:
        cfg = eval_config(base, cond)
        run_dir = make_run_dir(cfg, Path("runs"))
        print(f"[{cond}] 시작: {run_dir}", flush=True)
        runner = build_runner(cfg, Recorder(run_dir, cfg))
        manifest = []
        if cond in LEARNING:
            for i, (e, s) in enumerate(zip(eval_seeds, train_seeds)):
                hit = weights[(cond, s)]
                runner.ghosts.load_slot_state(i, read_weight_file(hit["path"]))
                manifest.append({"slot": i, "eval_seed": e, "train_seed": s, "file": str(hit["path"]),
                                 "game_no": hit["game_no"], "sha256": _sha256(hit["path"])})
        (run_dir / "eval_manifest.json").write_text(
            json.dumps({"condition": cond, "files": manifest}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summary = runner.run()
        wins = [g["winner"] for g in summary["games"]]
        print(f"[{cond}] 끝: {summary['ticks']}틱, {len(wins)}판 (초파리 {wins.count('FLIES')}, "
              f"팩맨 {wins.count('PACMAN')}, 중단 {wins.count('CENSORED')})", flush=True)
        runs.append({"condition": cond, "run_dir": run_dir.name})
        (out_dir / "index.json").write_text(json.dumps({"runs": runs}, ensure_ascii=False, indent=2) + "\n",
                                            encoding="utf-8")
    return report(out_dir)


def main(argv: list[str]) -> int:
    if len(argv) >= 4 and argv[1] == "run":
        return run(Path(argv[2]), [Path(p) for p in argv[3:]])
    if len(argv) == 3 and argv[1] == "report":
        return report(Path(argv[2]))
    print("사용법: final_eval.py run <평가 설정.yaml> <학습 결과 폴더> ... | final_eval.py report <평가 묶음 폴더>",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
