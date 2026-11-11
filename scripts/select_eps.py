"""팩맨 난이도 ε 선정 도구 (DECISIONS 24).

  select_eps.py run <설정.yaml>     후보 × 조건(b0, b1) 실행을 한 프로세스에서 순서대로 돌린다 (체크포인트 필요)
  select_eps.py report <묶음 폴더>   games.csv만 다시 읽어 표와 판정을 낸다 (예: runs/eps_select_v1)

판정(`decide`)은 순수 함수다. 숫자(구간, 문턱, 후보)는 설정의 `eps_select`에서만 읽는다.
"""

from __future__ import annotations

import copy
import csv
import json
import statistics
import sys
from collections.abc import Mapping
from pathlib import Path

import yaml

from flypac.core.config import Config, load_config

CSV_ENCODING = "utf-8-sig"
OUT_DIR = Path("runs") / "eps_select_v1"
# DECISIONS 14.6의 본 실험 팩맨 체크포인트 (best.pt)
CHECKPOINT_SHA256 = "4670fad239ee6324464b231cbe944afe8993167fbe20e1909fc23648b61e07f8"


# ---------------------------------------------------------------- 집계 (순수)

def read_games(path: Path) -> list[dict]:
    with open(path, encoding=CSV_ENCODING, newline="") as fh:
        return list(csv.DictReader(fh))


def _ratio(win: int, lose: int) -> float | None:
    return win / (win + lose) if (win + lose) > 0 else None


def summarize(rows: list[dict]) -> dict:
    """한 실행(조건 × ε)의 games.csv 행들에서 24.1 · 24.3에 쓰는 값을 낸다."""
    n = len(rows)
    fly = sum(r["승자"] == "FLIES" for r in rows)
    pac = sum(r["승자"] == "PACMAN" for r in rows)
    cens = sum(r["승자"] == "CENSORED" for r in rows)
    per_seed: dict[str, list[int]] = {}
    for r in rows:
        w = per_seed.setdefault(r["시드"], [0, 0])
        w[0] += r["승자"] == "FLIES"
        w[1] += r["승자"] == "PACMAN"
    seed_w = [x for x in (_ratio(a, b) for a, b in per_seed.values()) if x is not None]
    return {
        "games": n, "fly_wins": fly, "pacman_wins": pac, "censored": cens,
        "censored_fraction": cens / n if n else None,
        "W": _ratio(fly, pac),
        "seed_W_min": min(seed_w) if seed_w else None, "seed_W_max": max(seed_w) if seed_w else None,
        "active_catches_per_game": sum(int(r["활성 후 포획 수"]) for r in rows) / n if n else None,
        "pacman_mean_score": statistics.fmean(int(r["점수"]) for r in rows) if n else None,
        "game_length_median": statistics.median(int(r["틱 수"]) for r in rows) if n else None,
        "active_eaten_per_game": sum(int(r["활성 후 먹힘 수"]) for r in rows) / n if n else None,
    }


# ---------------------------------------------------------------- 판정 (순수, 24.3 · 24.4)

def decide(stats: Mapping[float, Mapping], sel: Mapping) -> dict:
    """B0의 후보별 값 `stats[ε]`(summarize의 결과)와 설정의 `eps_select`로 ε을 고른다.

    반환: {"eps": 고른 값 또는 None, "action": "select" | "stop", "branch": 24.3 / 24.4의 어느 줄, "reason": 설명,
           "flags": [...]}. 고르지 않으면 eps는 None이다.
    """
    cands = sorted(float(c) for c in sel["candidates"])
    lo, hi = (float(x) for x in sel["win_rate_band"])
    min_catch = float(sel["min_active_catches_per_game"])
    max_cens = float(sel["max_censored_fraction"])

    def stop(branch: str, reason: str) -> dict:
        return {"eps": None, "action": "stop", "branch": branch, "reason": reason, "flags": []}

    missing = [c for c in cands if c not in stats]
    if missing:
        return stop("입력 부족", f"후보 {missing}의 값이 없다")
    # 24.4 마지막 줄: 어느 후보든 중단 비율 > 상한이면 고르지 않는다 (다른 줄보다 먼저 본다)
    over = [c for c in cands if stats[c]["censored_fraction"] is None or stats[c]["censored_fraction"] > max_cens]
    if over:
        return stop("24.4 중단 비율", f"중단 비율이 {max_cens}를 넘는 후보 {over}. 원인을 먼저 본다")
    undefined = [c for c in cands if stats[c]["W"] is None]
    if undefined:
        return stop("24.4 그 밖", f"W가 정의되지 않은 후보 {undefined} (초파리 승 + 팩맨 승 = 0)")

    ok = [c for c in cands
          if lo <= stats[c]["W"] <= hi and stats[c]["active_catches_per_game"] >= min_catch]
    if ok:
        return {"eps": ok[0], "action": "select", "branch": "24.3",
                "reason": f"조건을 모두 만족하는 후보 {ok} 중 가장 작은 값", "flags": []}
    if all(stats[c]["W"] > hi for c in cands):
        return {"eps": cands[0], "action": "select", "branch": "24.4 모두 W > 상한",
                "reason": f"세 후보 모두 W > {hi}: 가장 작은 후보를 고른다",
                "flags": ["천장 위험"]}
    if all(stats[c]["W"] < lo for c in cands):
        return {"eps": cands[-1], "action": "select", "branch": "24.4 모두 W < 하한",
                "reason": f"세 후보 모두 W < {lo}: 가장 큰 후보를 고른다",
                "flags": ["관리자 보고: 후보를 늘릴지 결정"]}
    return stop("24.4 그 밖", "조건을 만족하는 후보가 없고 모두 상한 초과도 모두 하한 미만도 아니다 (관리자 결정)")


# ---------------------------------------------------------------- 보고

def _fmt(x, nd=3):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def _load_index(bundle: Path) -> list[dict]:
    return json.loads((bundle / "index.json").read_text(encoding="utf-8"))["runs"]


def build_results(bundle: Path) -> dict:
    runs = _load_index(bundle)
    root = bundle.parent
    out_runs, sel = [], None
    for r in runs:
        d = root / r["run_dir"]
        cfg = yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8"))
        sel = sel or cfg["eps_select"]
        env = json.loads((d / "environment.json").read_text(encoding="utf-8"))
        commit = (d / "git_commit.txt").read_text(encoding="utf-8").strip()
        out_runs.append({
            **r, "stats": summarize(read_games(d / "games.csv")), "config_hash": Config(cfg).hash(),
            "git_commit": commit, "device": env.get("pacman_device"),
        })
    select_cond = sel["select_condition"]
    b0 = {r["eps"]: r["stats"] for r in out_runs if r["condition"] == select_cond}
    return {"selection": decide(b0, sel), "eps_select": sel, "runs": out_runs}


def render_report(res: dict) -> str:
    sel = res["selection"]
    L = ["# ε 선정 결과 (DECISIONS 24)", "",
         "## 판정", "",
         f"- 선정 조건: {res['eps_select']['select_condition']}",
         f"- 결과: **{'ε = ' + str(sel['eps']) if sel['eps'] is not None else '고르지 않고 멈춘다'}**",
         f"- 적용된 분기: {sel['branch']}", f"- 사유: {sel['reason']}"]
    if sel["flags"]:
        L.append(f"- 기록: {', '.join(sel['flags'])}")
    L += ["", "## 조건 × ε", "",
          "| 조건 | ε | 판 수 | 초파리 승 | 팩맨 승 | 중단 | 중단 비율 | W | 시드별 W 최소 | 시드별 W 최대 | "
          "판당 활성 후 포획 | 팩맨 평균 점수 | 판 길이 중앙값 | 판당 활성 후 먹힘 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["runs"]:
        s = r["stats"]
        L.append(f"| {r['condition']} | {r['eps']} | {s['games']} | {s['fly_wins']} | {s['pacman_wins']} | "
                 f"{s['censored']} | {_fmt(s['censored_fraction'])} | {_fmt(s['W'])} | {_fmt(s['seed_W_min'])} | "
                 f"{_fmt(s['seed_W_max'])} | {_fmt(s['active_catches_per_game'])} | "
                 f"{_fmt(s['pacman_mean_score'], 1)} | {_fmt(s['game_length_median'], 1)} | "
                 f"{_fmt(s['active_eaten_per_game'])} |")
    L += ["", "## 실행", "", "| 조건 | ε | 결과 폴더 | 설정 해시 | 커밋 | 장치 |", "|---|---|---|---|---|---|"]
    for r in res["runs"]:
        L.append(f"| {r['condition']} | {r['eps']} | {r['run_dir']} | {r['config_hash']} | "
                 f"{r['git_commit']} | {r['device']} |")
    return "\n".join(L) + "\n"


def report(bundle: Path) -> int:
    res = build_results(bundle)
    (bundle / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    text = render_report(res)
    (bundle / "report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


# ---------------------------------------------------------------- 측정

def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_config(base: dict, condition: str, eps: float) -> Config:
    """실행 하나의 설정: condition, ghosts, pacman.eps, experiment를 채운다."""
    d = copy.deepcopy(base)
    d["condition"] = condition
    d["ghosts"] = copy.deepcopy(d["eps_select"]["ghosts"][condition])
    d["pacman"]["eps"] = float(eps)
    d["experiment"] = f"{base['experiment']}_{condition}_eps{eps}"
    return Config(d)


def run(config_path: Path) -> int:
    from flypac.core.runner import build_runner
    from flypac.pacman.trainer import resolve_device
    from flypac.recording.recorder import Recorder, make_run_dir

    cfg0 = load_config(config_path)
    base = cfg0.to_dict()
    sel = base["eps_select"]
    ckpt = Path(base["pacman"]["checkpoint"])
    if not ckpt.exists():
        print(f"멈춤: 체크포인트가 없다: {ckpt}", file=sys.stderr)
        return 1
    if _sha256(ckpt) != CHECKPOINT_SHA256:
        print(f"멈춤: 체크포인트 sha256이 DECISIONS 14.6의 값과 다르다: {ckpt}", file=sys.stderr)
        return 1
    device = resolve_device(str(base["pacman"]["device"])).type
    if device != "cuda":
        print(f"멈춤: 실제 장치가 cuda가 아니다 ({device})", file=sys.stderr)
        return 1
    if (OUT_DIR / "index.json").exists():
        print(f"멈춤: {OUT_DIR / 'index.json'}가 이미 있다. 측정은 한 번만 한다 (DECISIONS 24.2)", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    conditions = [sel["select_condition"], *sel["record_conditions"]]
    done: list[dict] = []
    for cond in conditions:
        for eps in sel["candidates"]:
            cfg = run_config(base, cond, float(eps))
            run_dir = make_run_dir(cfg, Path("runs"))
            print(f"[{cond} ε={eps}] 시작: {run_dir}", flush=True)
            summary = build_runner(cfg, Recorder(run_dir, cfg)).run()
            wins = [g["winner"] for g in summary["games"]]
            print(f"[{cond} ε={eps}] 끝: {summary['ticks']}틱, {len(wins)}판 (초파리 {wins.count('FLIES')}, "
                  f"팩맨 {wins.count('PACMAN')}, 중단 {wins.count('CENSORED')})", flush=True)
            done.append({"condition": cond, "eps": float(eps), "run_dir": run_dir.name})
            (OUT_DIR / "index.json").write_text(json.dumps({"runs": done}, ensure_ascii=False, indent=2) + "\n",
                                                encoding="utf-8")
    return report(OUT_DIR)


def main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[1] not in ("run", "report"):
        print("사용법: select_eps.py run <설정.yaml> | select_eps.py report <묶음 폴더>", file=sys.stderr)
        return 2
    return run(Path(argv[2])) if argv[1] == "run" else report(Path(argv[2]))


if __name__ == "__main__":
    sys.exit(main(sys.argv))
