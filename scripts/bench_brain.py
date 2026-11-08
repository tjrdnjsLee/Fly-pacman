"""뇌 속도 실측 (M1 완료 조건, M1 뇌 작업 지시서 ② 3.3).

스레드 수마다 새 프로세스를 띄운다. 스레드 제한은 numpy를 불러오기 **전에** 환경변수
OMP_NUM_THREADS · OPENBLAS_NUM_THREADS · MKL_NUM_THREADS를 n으로 두고, 리눅스에서는 CPU 친화도도
os.sched_setaffinity로 n개 논리 코어(0 ~ n-1)로 묶는다. numpy의 원소별 연산은 원래 단일 스레드이고,
BLAS(OpenBLAS)를 쓰는 행렬곱만 스레드 수의 영향을 받는다.

측정 (게임 B = 10, 초파리 F = 4, 원본 회로 N_KC = 1575, N_MBON = 47):
- 가짜 엔진 + 규칙 기반 팩맨 + 초파리 뇌로 실행기를 끝까지 돌린다 (기록기 없음, 체제 1).
- propose와 commit의 호출 시간을 따로 잰다. 실행기 전체 틱당 시간도 잰다.
- ρ ∈ {0.02, 0.05, 0.10}. 나머지 값은 configs/brain_v1.yaml (자리표시) 그대로.

    uv run python scripts/bench_brain.py --threads 1 3 16 --out runs/brain/bench/bench.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def child(threads: int, rhos: list[float], seeds: int, n_games: int, repeats: int) -> dict:
    """이 프로세스 안에서 잰다 (스레드 제한은 부모가 환경변수로 이미 걸었다)."""
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(range(threads)))
    import time

    import numpy as np

    from flypac.brain.flybrain import FlyBrain, circuit_from_config, load_brain_config
    from flypac.core.config import Config, load_config
    from flypac.core.runner import build_runner

    base = load_config(REPO / "configs" / "m0_smoke.yaml").to_dict()
    base.update(seeds=list(range(seeds)), n_games=n_games)
    run_cfg = Config(base)
    bcfg = load_brain_config("configs/brain_v1.yaml")
    circuit = circuit_from_config(bcfg)
    out = []
    for rho in rhos:
        d = bcfg.to_dict()
        d["kc"] = {**d["kc"], "rho": rho}
        cfg = Config(d)
        best = None
        for _ in range(repeats):
            r = build_runner(run_cfg)
            brain = FlyBrain(cfg, r.B, list(run_cfg.seeds), condition=str(run_cfg.condition), circuit=circuit)
            r.ghosts = brain
            acc = {"propose": 0.0, "commit": 0.0, "n_committed": 0, "n_decide": 0}
            propose, commit = brain.propose, brain.commit

            def timed_propose(obs, alive, views=None, _p=propose, _a=acc):
                t = time.perf_counter()
                a = _p(obs, alive, views)
                _a["propose"] += time.perf_counter() - t
                _a["n_decide"] += int((a >= 0).sum())
                return a

            def timed_commit(committed, active, _c=commit, _a=acc):
                t = time.perf_counter()
                _c(committed, active)
                _a["commit"] += time.perf_counter() - t
                _a["n_committed"] += int((committed & active).sum())

            brain.propose, brain.commit = timed_propose, timed_commit
            t0 = time.perf_counter()
            res = r.run()
            total = time.perf_counter() - t0
            ticks = res["ticks"]
            row = {
                "rho": rho, "k": brain.k, "ticks": ticks,
                "runner_ms_per_tick": 1e3 * total / ticks,
                "propose_ms_per_tick": 1e3 * acc["propose"] / ticks,
                "commit_ms_per_tick": 1e3 * acc["commit"] / ticks,
                "brain_ms_per_tick": 1e3 * (acc["propose"] + acc["commit"]) / ticks,
                "commit_ms_per_learning_decision": (1e3 * acc["commit"] / acc["n_committed"]
                                                    if acc["n_committed"] else None),
                "decisions_per_tick": acc["n_decide"] / ticks,
                "learning_commits_per_tick": acc["n_committed"] / ticks,
                "learned": bool(np.any(brain.dw)),
            }
            if best is None or row["runner_ms_per_tick"] < best["runner_ms_per_tick"]:
                best = row
        out.append(best)
    b = FlyBrain(bcfg, seeds, list(range(seeds)), circuit=circuit)
    mem = {"w(dw)_MB": b.dw.nbytes / 2**20, "e_MB": b.e.nbytes / 2**20, "dtype": str(b.dw.dtype),
           "shape": list(b.dw.shape)}
    return {"threads": threads, "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
            else None, "env": {v: os.environ.get(v) for v in THREAD_VARS}, "results": out, "memory": mem}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 3, 16])
    ap.add_argument("--rho", type=float, nargs="+", default=[0.02, 0.05, 0.10])
    ap.add_argument("--seeds", type=int, default=10, help="게임 수 B (칸마다 실행 시드 하나)")
    ap.add_argument("--games", type=int, default=2, help="칸마다 판 수")
    ap.add_argument("--repeats", type=int, default=3, help="반복 중 가장 빠른 값을 쓴다")
    ap.add_argument("--out", type=Path, default=REPO / "runs" / "brain" / "bench" / "bench.json")
    ap.add_argument("--child", type=int, default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.child is not None:
        print(json.dumps(child(a.child, a.rho, a.seeds, a.games, a.repeats)))
        return

    import numpy as np  # 부모는 장치 정보만

    runs = []
    for n in a.threads:
        env = {**os.environ, **{v: str(n) for v in THREAD_VARS}}
        cmd = [sys.executable, str(Path(__file__).resolve()), "--child", str(n), "--rho", *map(str, a.rho),
               "--seeds", str(a.seeds), "--games", str(a.games), "--repeats", str(a.repeats)]
        res = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8", check=True)
        runs.append(json.loads(res.stdout.strip().splitlines()[-1]))
        print(f"threads={n} 끝", file=sys.stderr)
    cpu = platform.processor()
    try:
        cpu = next(line.split(":", 1)[1].strip() for line in
                   Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines() if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    report = {"cpu": cpu, "logical_cpus": os.cpu_count(), "python": platform.python_version(),
              "numpy": np.__version__, "blas": "OpenBLAS (numpy 내장)",
              "thread_limit": "환경변수 " + ", ".join(THREAD_VARS) + " = n, 그리고 os.sched_setaffinity(0~n-1)",
              "B": a.seeds, "games_per_slot": a.games, "repeats": a.repeats, "runs": runs}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
