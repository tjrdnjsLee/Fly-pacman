"""초파리 동시 도착(순서 효과) 빈도 측정 (DECISIONS 25.3 · 25.4).

  measure_ties.py run <eps_select_v1.yaml> <ε 결과 묶음 폴더>   B0 · B1 실행 (체크포인트 필요)
  measure_ties.py report <ties 묶음 폴더>                      counts와 CSV만 다시 읽어 표와 판정

세는 기록기는 실제 Recorder를 감싸고 `on_life_end`에서만 센다. 판정(`decide_ties`)은 순수 함수이고 문턱은
configs/ties_v1.yaml에서만 읽는다.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import select_eps  # noqa: E402

from flypac.core.config import Config, load_config  # noqa: E402
from flypac.core.events import Kind  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TIES_CONFIG = ROOT / "configs" / "ties_v1.yaml"
INTEG_CONFIG = ROOT / "configs" / "integ_v1.yaml"
OUT_DIR = Path("runs") / "ties_v1"
CSV_FILES = ("games.csv", "lives.csv", "flies.csv")
F = 4
KINDS = ("T1", "T2", "T3")


# ---------------------------------------------------------------- 세기 (순수)

def new_counts() -> dict:
    return {
        "catches": 0, "active_catches": 0,
        "events": {t: {"all": 0, "active": 0} for t in KINDS},
        "order_effect_active": 0,                       # 활성 포획 중 T1 또는 T2가 있는 것
        "caught_fly_active": [0] * F,                   # 활성 포획에서 잡은 초파리 번호
        "pairs": {t: {} for t in KINDS},                # "j,k" -> 횟수 (전체 포획)
        "pairs_active": {t: {} for t in KINDS},         # 활성 포획만
        "eaten_same_tick": 0, "catches_with_eaten_same_tick": 0,
    }


def classify(view, j: int, nbr) -> dict[str, list[int]]:
    """종료 상태에서 j가 아닌, 집 밖 초파리 k의 범주 (DECISIONS 25.3 2.2)."""
    out = {t: [] for t in KINDS}
    pac = int(view.pac_cell)
    adjacent = {int(c) for c in nbr[pac] if c >= 0}
    for k in range(len(view.fly_cell)):
        if k == j or bool(view.fly_in_house[k]):
            continue
        cell, z = int(view.fly_cell[k]), bool(view.fly_z[k])
        if cell == pac:
            out["T2" if z else "T1"].append(k)
        elif cell in adjacent and not z:
            out["T3"].append(k)
    return out


def count_life(counts: dict, view, events, nbr) -> None:
    """목숨 하나가 끝날 때 부른다. 목숨을 끝낸 CATCH가 없으면(클리어 · 중단) 세지 않는다."""
    catch = next((e for e in events if e.kind is Kind.CATCH), None)
    if catch is None:
        return
    j, active = int(catch.fly), bool(catch.active)
    eaten = sum(e.kind is Kind.EATEN for e in events)
    counts["catches"] += 1
    counts["eaten_same_tick"] += eaten
    counts["catches_with_eaten_same_tick"] += eaten > 0
    if active:
        counts["active_catches"] += 1
        counts["caught_fly_active"][j] += 1
    found = classify(view, j, nbr)
    for t, ks in found.items():
        if not ks:
            continue
        counts["events"][t]["all"] += 1
        if active:
            counts["events"][t]["active"] += 1
        for k in ks:
            key = f"{j},{k}"
            counts["pairs"][t][key] = counts["pairs"][t].get(key, 0) + 1
            if active:
                counts["pairs_active"][t][key] = counts["pairs_active"][t].get(key, 0) + 1
    if active and (found["T1"] or found["T2"]):
        counts["order_effect_active"] += 1


class CountingRecorder:
    """실제 기록기를 감싼다. 모든 호출을 그대로 넘기고 on_life_end에서만 센다."""

    def __init__(self, inner, nbr):
        self._inner, self._nbr = inner, nbr
        self.counts = new_counts()

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def on_life_end(self, game, view, events):
        count_life(self.counts, view, events, self._nbr)
        return self._inner.on_life_end(game, view, events)


# ---------------------------------------------------------------- 판정 (순수, 25.4)

def order_effect_fraction(counts: dict) -> float | None:
    n = counts["active_catches"]
    return counts["order_effect_active"] / n if n else None


def decide_ties(counts_b0: dict, ties: dict, regression_ok: bool) -> dict:
    """B0의 순서 효과 비율로 25.4의 안 가 / 안 나를 정한다. 회귀 대조가 다르면 판정하지 않는다."""
    if not regression_ok:
        return {"verdict": "stop", "reason": "회귀 대조가 다르다: 빈도를 판정에 쓰지 않고 멈춘다 (25.3)", "fraction": None}
    frac = order_effect_fraction(counts_b0)
    if frac is None:
        return {"verdict": "stop", "reason": "B0의 활성 포획이 0이라 비율을 정할 수 없다", "fraction": None}
    limit = float(ties["order_effect_max_fraction"])
    if frac <= limit:
        return {"verdict": "가", "fraction": frac, "limit": limit,
                "reason": f"B0 순서 효과 비율 {frac:.4f} <= {limit}: 현재 동작을 유지하고 기록만 한다"}
    return {"verdict": "나", "fraction": frac, "limit": limit,
            "reason": f"B0 순서 효과 비율 {frac:.4f} > {limit}: 팩맨 칸의 초파리를 게임 난수로 섞은 순서로 검사 (엔진 수정은 별도 지시서)"}


# ---------------------------------------------------------------- 회귀 대조 · lives.csv

def compare_csvs(new_dir: Path, ref_dir: Path) -> dict:
    out = {}
    for name in CSV_FILES:
        a, b = (new_dir / name).read_bytes(), (ref_dir / name).read_bytes()
        if a == b:
            out[name] = {"same": True}
            continue
        la, lb = a.decode("utf-8-sig").split("\n"), b.decode("utf-8-sig").split("\n")
        first = next((i + 1 for i, (x, y) in enumerate(zip(la, lb)) if x != y), min(len(la), len(lb)) + 1)
        out[name] = {"same": False, "first_diff_line": first}
    return out


def lives_active_catch_dist(lives_csv: Path) -> list[int]:
    dist = [0] * F
    with open(lives_csv, encoding=select_eps.CSV_ENCODING, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["종료 원인"] == "포획" and r["출발 전 포획 여부"] == "0":
                dist[int(r["잡은 초파리"])] += 1
    return dist


# ---------------------------------------------------------------- 보고

def _f(x, nd=4):
    return "-" if x is None else f"{x:.{nd}f}"


def render_report(res: dict) -> str:
    d = res["decision"]
    L = ["# 동시 도착 빈도 (DECISIONS 25.3 · 25.4)", "", "## 판정", "",
         f"- 안: **{d['verdict']}**", f"- 사유: {d['reason']}", f"- 판정은 B0로 한다. B1은 기록만 한다.", "",
         "## 회귀 대조 (24절 측정과 바이트 단위)", "", "| 조건 | games.csv | lives.csv | flies.csv |", "|---|---|---|---|"]
    for cond, r in res["regression"].items():
        L.append(f"| {cond} | " + " | ".join(
            "같음" if r[n]["same"] else f"다름 (첫 차이 {r[n]['first_diff_line']}행)" for n in CSV_FILES) + " |")
    for cond, c in res["counts"].items():
        n = c["active_catches"]
        L += ["", f"## 조건 {cond}", "", "| 값 | 값 |", "|---|---|",
              f"| 포획 수 | {c['catches']} |", f"| 활성 포획 수 | {n} |"]
        for t in KINDS:
            L.append(f"| {t} 사건 수 (전체 / 활성) | {c['events'][t]['all']} / {c['events'][t]['active']} |")
        L += [f"| 순서 효과 비율 (활성 포획 중 T1 또는 T2) | {_f(order_effect_fraction(c))} ({c['order_effect_active']}/{n}) |",
              f"| 같은 틱 EATEN 수 (합 / 있는 포획 수) | {c['eaten_same_tick']} / {c['catches_with_eaten_same_tick']} |",
              "| 잡은 초파리 번호 분포 (활성, 0 ~ 3) | " + ", ".join(
                  f"{_f(v / n if n else None, 3)} ({v})" for v in c["caught_fly_active"]) + " |",
              f"| lives.csv와 분포 일치 | {'예' if res['lives_match'][cond] else '아니오'} |"]
        for t in ("T1", "T2", "T3"):
            pairs = c["pairs_active"][t]
            L.append(f"| {t} 쌍 (잡은 j,경쟁자 k; 활성) | " + (
                ", ".join(f"{k}: {v}" for k, v in sorted(pairs.items())) or "없음") + " |")
    L += ["", "T3는 상한 지표이며 판정에 쓰지 않는다.", "", "## 실행", "", "| 조건 | ε | 결과 폴더 | 설정 해시 | 커밋 | 장치 |",
          "|---|---|---|---|---|---|"]
    for r in res["runs"]:
        L.append(f"| {r['condition']} | {r['eps']} | {r['run_dir']} | {r['config_hash']} | {r['git_commit']} | {r['device']} |")
    return "\n".join(L) + "\n"


def report(bundle: Path) -> int:
    index = json.loads((bundle / "index.json").read_text(encoding="utf-8"))
    ties = yaml.safe_load(TIES_CONFIG.read_text(encoding="utf-8"))["ties"]
    root = bundle.parent
    counts, lives_match, runs = {}, {}, []
    for r in index["runs"]:
        cond = r["condition"]
        counts[cond] = json.loads((bundle / f"counts_{cond}.json").read_text(encoding="utf-8"))
        d = root / r["run_dir"]
        lives_match[cond] = lives_active_catch_dist(d / "lives.csv") == counts[cond]["caught_fly_active"]
        cfg = yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8"))
        env = json.loads((d / "environment.json").read_text(encoding="utf-8"))
        runs.append({**r, "config_hash": Config(cfg).hash(),
                     "git_commit": (d / "git_commit.txt").read_text(encoding="utf-8").strip(),
                     "device": env.get("pacman_device")})
    regression = index["regression"]
    reg_ok = all(v["same"] for r in regression.values() for v in r.values())
    decision = decide_ties(counts["b0"], ties, reg_ok and all(lives_match.values()))
    res = {"decision": decision, "regression": regression, "counts": counts, "lives_match": lives_match,
           "runs": runs, "ties": ties}
    (bundle / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    text = render_report(res)
    (bundle / "report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


# ---------------------------------------------------------------- 측정

def run(eps_cfg_path: Path, eps_bundle: Path) -> int:
    from flypac.core.runner import build_runner
    from flypac.maps.original import load_original
    from flypac.pacman.trainer import resolve_device
    from flypac.recording.recorder import Recorder, make_run_dir

    base = load_config(eps_cfg_path).to_dict()
    ties = yaml.safe_load(TIES_CONFIG.read_text(encoding="utf-8"))["ties"]
    eps = float(load_config(INTEG_CONFIG).pacman.eps)
    ckpt = Path(base["pacman"]["checkpoint"])
    if not ckpt.exists() or select_eps._sha256(ckpt) != select_eps.CHECKPOINT_SHA256:
        print(f"멈춤: 체크포인트가 없거나 sha256이 DECISIONS 14.6의 값과 다르다: {ckpt}", file=sys.stderr)
        return 1
    device = resolve_device(str(base["pacman"]["device"])).type
    if device != "cuda":
        print(f"멈춤: 실제 장치가 cuda가 아니다 ({device})", file=sys.stderr)
        return 1
    if (OUT_DIR / "index.json").exists():
        print(f"멈춤: {OUT_DIR / 'index.json'}가 이미 있다. 측정은 한 번만 한다", file=sys.stderr)
        return 1
    ref_runs = {(r["condition"], float(r["eps"])): r["run_dir"]
                for r in json.loads((eps_bundle / "index.json").read_text(encoding="utf-8"))["runs"]}

    nbr = load_original().nbr
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs, regression = [], {}
    for cond in ties["conditions"]:
        cfg = select_eps.run_config(base, cond, eps)
        d = cfg.to_dict()
        d["experiment"] = f"ties_v1_{cond}"
        cfg = Config(d)
        run_dir = make_run_dir(cfg, Path("runs"))
        print(f"[{cond} ε={eps}] 시작: {run_dir}", flush=True)
        rec = CountingRecorder(Recorder(run_dir, cfg), nbr)
        summary = build_runner(cfg, rec).run()
        wins = [g["winner"] for g in summary["games"]]
        print(f"[{cond}] 끝: {summary['ticks']}틱, {len(wins)}판 (초파리 {wins.count('FLIES')}, "
              f"팩맨 {wins.count('PACMAN')}, 중단 {wins.count('CENSORED')})", flush=True)
        (OUT_DIR / f"counts_{cond}.json").write_text(json.dumps(rec.counts, ensure_ascii=False, indent=2) + "\n",
                                                     encoding="utf-8")
        ref = ref_runs.get((cond, eps))
        if ref is None:
            print(f"멈춤: ε 결과 묶음에 {cond} ε={eps} 실행이 없다", file=sys.stderr)
            return 1
        regression[cond] = compare_csvs(run_dir, eps_bundle.parent / ref)
        runs.append({"condition": cond, "eps": eps, "run_dir": run_dir.name, "eps_select_run_dir": ref})
        (OUT_DIR / "index.json").write_text(
            json.dumps({"runs": runs, "regression": regression}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report(OUT_DIR)


def main(argv: list[str]) -> int:
    if len(argv) == 4 and argv[1] == "run":
        return run(Path(argv[2]), Path(argv[3]))
    if len(argv) == 3 and argv[1] == "report":
        return report(Path(argv[2]))
    print("사용법: measure_ties.py run <eps_select_v1.yaml> <ε 결과 묶음 폴더> | measure_ties.py report <ties 묶음 폴더>",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
