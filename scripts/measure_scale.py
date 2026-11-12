"""동시 실행 결정론 · 처리량 · 본 실험 규모 측정 (DECISIONS 27).

  measure_scale.py run <기본 설정.yaml>     세 배치(S, A, B)를 차례로 돌고, 배치마다 해시 · 게임-틱 합 · 시간만 남긴다
  measure_scale.py report <묶음 폴더>        남긴 값으로 결정론 대조 · 처리량 · 규모 규칙을 계산한다 (예: runs/scale_v1)

27.2: 이 측정은 학습 결과를 보지 않는다. 승패 · 점수 · 보상 · 포획 수는 화면에도 파일에도 내지 않는다. 배치가 끝나면
곧바로 games.csv · lives.csv · flies.csv · 재생 로그 · 가중치를 지우고 config.yaml, environment.json, git_commit.txt,
해시, 게임-틱 합, 시간만 남긴다. 규칙 계산(`decide_scale`)은 순수 함수이고 숫자는 설정의 `scale`에서만 읽는다.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_parallel  # noqa: E402
import select_eps  # noqa: E402

CSV_ENCODING = "utf-8-sig"
OUT_DIR = Path("runs") / "scale_v1"
# (파일, 정렬 열): 행을 (시드, 판, [목숨 번호 | 초파리]) 순으로 정렬해 해시한다 (27.3)
CSV_SPECS = {
    "games": ("games.csv", ("시드", "판")),
    "lives": ("lives.csv", ("시드", "판", "목숨 번호")),
    "flies": ("flies.csv", ("시드", "판", "초파리")),
}
WEIGHT_FILE = "s{seed}_g{game:05d}.npz"
CHECKPOINT_SHA256 = select_eps.CHECKPOINT_SHA256


# ---------------------------------------------------------------- 정렬 해시 (순수에 가까움: 파일만 읽는다)

def sorted_rows_hash(paths: list[Path], key_cols: tuple[str, ...]) -> str:
    """CSV 여러 개(시드를 나눈 폴더들)의 행을 합쳐 key_cols 순으로 정렬하고 sha256을 낸다. 머리글도 해시에 든다."""
    header, rows = None, []
    for p in paths:
        with open(p, encoding=CSV_ENCODING, newline="") as fh:
            r = csv.reader(fh)
            h = next(r)
            if header is None:
                header = h
            elif h != header:
                raise ValueError(f"CSV 머리글이 다르다: {p}")
            rows.extend(r)
    if header is None:
        raise ValueError("읽을 CSV가 없다")
    idx = [header.index(c) for c in key_cols]
    rows.sort(key=lambda row: tuple(int(row[i]) for i in idx))
    h = hashlib.sha256()
    for row in [header, *rows]:
        h.update(("\x1f".join(row) + "\n").encode("utf-8"))
    return h.hexdigest()


def weights_hash(dirs: list[Path]) -> str | None:
    """가중치 파일들의 불러온 배열(dw_masked, rbar) 바이트로 sha256 (시드, 판 순). 파일이 없으면 None."""
    from flypac.recording.weights import parse_weight_filename

    items = []
    for d in dirs:
        for p in (Path(d) / "weights").glob("*.npz"):
            parsed = parse_weight_filename(p.name)
            if parsed:
                items.append((parsed, p))
    if not items:
        return None
    h = hashlib.sha256()
    for (seed, game), p in sorted(items):
        with np.load(p, allow_pickle=False) as z:
            h.update(f"{seed}:{game}\n".encode())
            for name in ("dw_masked", "rbar"):
                a = np.ascontiguousarray(z[name])
                h.update(f"{name}:{a.dtype.str}:{a.shape}\n".encode())
                h.update(a.tobytes())
    return h.hexdigest()


def game_tick_sum(paths: list[Path]) -> int:
    total = 0
    for p in paths:
        with open(p, encoding=CSV_ENCODING, newline="") as fh:
            total += sum(int(r["틱 수"]) for r in csv.DictReader(fh))
    return total


def condition_digest(run_dirs: list[Path], learning: bool) -> dict:
    """조건 하나(시드를 나눈 폴더들)의 해시 넷과 합친 해시, 게임-틱 합. 학습 조건인데 가중치가 없으면 오류."""
    out = {}
    for key, (fname, cols) in CSV_SPECS.items():
        out[key] = sorted_rows_hash([d / fname for d in run_dirs], cols)
    out["weights"] = weights_hash(run_dirs)
    if learning and out["weights"] is None:
        raise ValueError(f"학습 조건인데 가중치 파일이 없다: {[str(d) for d in run_dirs]}")
    out["combined"] = hashlib.sha256(
        "|".join(f"{k}={out[k]}" for k in ("games", "lives", "flies", "weights")).encode()).hexdigest()
    out["game_ticks"] = game_tick_sum([d / CSV_SPECS["games"][0] for d in run_dirs])
    return out


def erase_results(run_dir: Path) -> None:
    """CSV · 재생 로그 · 가중치를 지운다 (27.2). config.yaml, environment.json, git_commit.txt는 남긴다."""
    for fname, _ in CSV_SPECS.values():
        (run_dir / fname).unlink(missing_ok=True)
    for sub in ("replay", "weights"):
        shutil.rmtree(run_dir / sub, ignore_errors=True)


# ---------------------------------------------------------------- 규모 규칙 (순수, 27.3 · 27.4)

def decide_scale(hashes: Mapping[str, Mapping[str, str]], wall_seconds: Mapping[str, float], scale: Mapping) -> dict:
    """hashes[배치][조건] = 합친 해시, wall_seconds[배치] = 배치 벽시계 초, scale = 설정의 `scale` 블록.

    S와 A가 한 조건이라도 다르면 {"action": "stop"} (규칙을 적용하지 않는다). 아니면 B가 A와 같고 더 빠를 때만 B,
    아니면 A를 골라 T̂(n) = T × (s·n + n_eval) / 30 이 budget_hours 이하인 후보 중 가장 큰 n을 고른다 (경계 포함).
    모두 넘으면 가장 작은 후보로 하고 두 밤에 나눈다. 시간 비교는 분수로 해서 부동소수 오차를 피한다.
    """
    conds = list(scale["conditions"])
    sa_diff = [c for c in conds if hashes["S"][c] != hashes["A"][c]]
    ab_diff = [c for c in conds if hashes["A"][c] != hashes["B"][c]]
    res = {"sa_equal": not sa_diff, "sa_diff": sa_diff, "ab_equal": not ab_diff, "ab_diff": ab_diff}
    if sa_diff:
        return {**res, "action": "stop",
                "reason": f"S와 A의 정렬 해시가 다른 조건 {sa_diff}: 동시 실행이 결과를 바꿨다. 규칙을 적용하지 않는다"}
    use_b = (not ab_diff) and wall_seconds["B"] < wall_seconds["A"]
    batch = "B" if use_b else "A"
    if ab_diff:
        why = f"A와 B가 다른 조건 {ab_diff}: B를 쓰지 않는다"
    elif use_b:
        why = "B가 A와 같고 더 빠르다"
    else:
        why = "B가 A와 같지만 더 빠르지 않다"
    T = Fraction(wall_seconds[batch])
    s, n_eval = Fraction(str(scale["safety_factor"])), int(scale["eval_games"])
    limit = Fraction(str(scale["budget_hours"])) * 3600
    table = []
    for n in sorted(int(x) for x in scale["n_games_candidates"]):
        t_hat = T * (s * n + n_eval) / 30
        table.append({"n_games": n, "t_hat_hours": float(t_hat / 3600), "within_budget": t_hat <= limit})
    ok = [r["n_games"] for r in table if r["within_budget"]]
    n_pick = max(ok) if ok else table[0]["n_games"]
    return {**res, "action": "decide", "batch": batch, "batch_reason": why, "t_seconds": float(T), "table": table,
            "n_games": n_pick, "nights": 1 if ok else 2,
            "reason": (f"예산 안의 후보 중 가장 큰 값 {n_pick}" if ok
                       else f"모든 후보가 {scale['budget_hours']}시간을 넘는다: {n_pick}으로 하고 두 밤에 나눈다")}


# ---------------------------------------------------------------- 측정

def measure_batch(spec: Mapping, base: dict, scale: Mapping, root: Path) -> dict:
    """배치 하나를 돌고, 해시 · 게임-틱 합 · 시간을 root/<배치>/measure.json에 쓰고, 결과를 지운다."""
    out = root / spec["name"]
    index = run_parallel.run_batch(base, list(scale["conditions"]), int(spec["split"]), bool(spec["sequential"]), out)
    run_dirs = [out / r["run_dir"] for r in index["runs"]]
    try:
        if index["failed"]:
            return {"name": spec["name"], "failed": index["failed"], "wall_seconds": index["wall_seconds"]}
        conds = {}
        for cond in scale["conditions"]:
            runs = [r for r in index["runs"] if r["condition"] == cond]
            digest = condition_digest([out / r["run_dir"] for r in runs], cond in run_parallel_learning())
            digest["wall_seconds"] = max(r["wall_seconds"] for r in runs)  # 한 조건의 조각들은 같이 시작한다
            digest["run_dirs"] = [r["run_dir"] for r in runs]
            conds[cond] = digest
        meas = {"name": spec["name"], "split": int(spec["split"]), "sequential": bool(spec["sequential"]),
                "wall_seconds": index["wall_seconds"], "conditions": conds, "failed": []}
        (out / "measure.json").write_text(json.dumps(meas, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return meas
    finally:
        for d in run_dirs:
            erase_results(d)


def run_parallel_learning() -> tuple[str, ...]:
    from flypac.core.runner import LEARNING_CONDITIONS
    return LEARNING_CONDITIONS


def measure_all(base: dict, root: Path) -> list[dict]:
    """설정의 scale.batches를 차례로 돈다. index.json을 먼저 써서 측정이 한 번만 되게 한다 (실패해도 남는다)."""
    scale = base["scale"]
    root.mkdir(parents=True, exist_ok=True)
    idx_path = root / "index.json"
    idx = {"batches": []}
    idx_path.write_text(json.dumps(idx, indent=2) + "\n", encoding="utf-8")
    done = []
    for spec in scale["batches"]:
        print(f"[배치 {spec['name']}] 시작 (프로세스 분할 {spec['split']}, "
              f"{'차례로' if spec['sequential'] else '동시에'})", flush=True)
        meas = measure_batch(spec, base, scale, root)
        done.append(meas)
        idx["batches"].append(spec["name"])
        idx_path.write_text(json.dumps(idx, indent=2) + "\n", encoding="utf-8")
        if meas["failed"]:
            print(f"멈춤: 배치 {spec['name']}의 프로세스가 실패했다: {meas['failed']} "
                  f"(오류 출력: {root / spec['name'] / 'logs'})", file=sys.stderr)
            return done
        print(f"[배치 {spec['name']}] 끝: {meas['wall_seconds']:.1f}초, 게임-틱 합 "
              f"{sum(c['game_ticks'] for c in meas['conditions'].values())}", flush=True)
    return done


def run(config_path: Path) -> int:
    from flypac.core.config import load_config
    from flypac.pacman.trainer import resolve_device

    base = load_config(config_path).to_dict()
    ckpt = Path(base["pacman"]["checkpoint"])
    if not ckpt.exists():
        print(f"멈춤: 체크포인트가 없다: {ckpt}", file=sys.stderr)
        return 1
    if select_eps._sha256(ckpt) != CHECKPOINT_SHA256:
        print(f"멈춤: 체크포인트 sha256이 DECISIONS 14.6의 값과 다르다: {ckpt}", file=sys.stderr)
        return 1
    device = resolve_device(str(base["pacman"]["device"])).type
    if device != "cuda":
        print(f"멈춤: 실제 장치가 cuda가 아니다 ({device})", file=sys.stderr)
        return 1
    if (OUT_DIR / "index.json").exists():
        print(f"멈춤: {OUT_DIR / 'index.json'}가 이미 있다. 측정은 한 번만 한다 (DECISIONS 27.1)", file=sys.stderr)
        return 1
    done = measure_all(base, OUT_DIR)
    if any(m["failed"] for m in done):
        return 1
    return report(OUT_DIR)


# ---------------------------------------------------------------- 보고

def build_results(bundle: Path) -> dict:
    idx = json.loads((bundle / "index.json").read_text(encoding="utf-8"))
    meas = {n: json.loads((bundle / n / "measure.json").read_text(encoding="utf-8")) for n in idx["batches"]}
    first = next(iter(meas.values()))["conditions"]
    d0 = bundle / idx["batches"][0] / next(iter(first.values()))["run_dirs"][0]
    cfg = yaml.safe_load((d0 / "config.yaml").read_text(encoding="utf-8"))
    commit = (d0 / "git_commit.txt").read_text(encoding="utf-8").strip()
    env = json.loads((d0 / "environment.json").read_text(encoding="utf-8"))
    scale = cfg["scale"]
    hashes = {n: {c: v["combined"] for c, v in m["conditions"].items()} for n, m in meas.items()}
    walls = {n: m["wall_seconds"] for n, m in meas.items()}
    return {"decision": decide_scale(hashes, walls, scale), "scale": scale, "git_commit": commit,
            "device": env.get("pacman_device"), "batches": meas}


def _hours(sec: float) -> str:
    return f"{sec / 3600:.3f}"


def render_report(res: dict) -> str:
    dec, meas = res["decision"], res["batches"]
    conds = list(res["scale"]["conditions"])
    L = ["# 본 실험 규모 측정 결과 (DECISIONS 27)", "", f"- 측정 커밋: {res['git_commit']}",
         f"- 팩맨 장치: {res['device']}", "",
         "## 처리량", "", "| 배치 | 분할 | 방식 | 벽시계(초) | 벽시계(시간) | 게임-틱 합 | 게임-틱/초 |", "|---|---|---|---|---|---|---|"]
    for n, m in meas.items():
        tot = sum(c["game_ticks"] for c in m["conditions"].values())
        L.append(f"| {n} | {m['split']} | {'차례로' if m['sequential'] else '동시에'} | {m['wall_seconds']:.1f} | "
                 f"{_hours(m['wall_seconds'])} | {tot} | {tot / m['wall_seconds']:.1f} |")
    L += ["", "조건별 (게임-틱 합 / 벽시계 초):", "", "| 조건 | " + " | ".join(meas) + " |",
          "|---|" + "---|" * len(meas)]
    for c in conds:
        L.append(f"| {c} | " + " | ".join(f"{m['conditions'][c]['game_ticks']} / {m['conditions'][c]['wall_seconds']:.1f}"
                                          for m in meas.values()) + " |")
    L += ["", "## 결정론 대조 (정렬 해시: games · lives · flies · 가중치)", "",
          f"- S–A: {'같음' if dec['sa_equal'] else '**다름** ' + str(dec['sa_diff'])}",
          f"- A–B: {'같음' if dec['ab_equal'] else '**다름** ' + str(dec['ab_diff'])}", "",
          "| 조건 | " + " | ".join(meas) + " |", "|---|" + "---|" * len(meas)]
    for c in conds:
        L.append(f"| {c} | " + " | ".join(m["conditions"][c]["combined"][:12] for m in meas.values()) + " |")
    L += ["", "## 규모 규칙", ""]
    if dec["action"] == "stop":
        L += [f"- **멈춘다.** {dec['reason']}"]
    else:
        L += [f"- 고른 배치: **{dec['batch']}** ({dec['batch_reason']}), T = {dec['t_seconds']:.1f}초 "
              f"({_hours(dec['t_seconds'])}시간)", "", "| 판 수 n | T̂(n) 시간 | 예산 안 |", "|---|---|---|"]
        for r in dec["table"]:
            L.append(f"| {r['n_games']} | {r['t_hat_hours']:.3f} | {'예' if r['within_budget'] else '아니오'} |")
        L += ["", f"- 본 실험 판 수: **{dec['n_games']}** ({dec['reason']}), 밤 수 {dec['nights']}"]
    return "\n".join(L) + "\n"


def report(bundle: Path) -> int:
    res = build_results(bundle)
    (bundle / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    text = render_report(res)
    (bundle / "report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


def main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[1] not in ("run", "report"):
        print("사용법: measure_scale.py run <설정.yaml> | measure_scale.py report <묶음 폴더>", file=sys.stderr)
        return 2
    return run(Path(argv[2])) if argv[1] == "run" else report(Path(argv[2]))


if __name__ == "__main__":
    sys.exit(main(sys.argv))
