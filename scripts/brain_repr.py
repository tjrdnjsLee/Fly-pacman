"""표현력 시험 (M1 뇌 ② 단계 3). 규칙은 DECISIONS 13절. 개발 자료(1000 ~ 1019)만 만든다.

  uv run --extra analysis python scripts/brain_repr.py data      # 자료 생성 + 표본 수 확인 (측정 전)
  uv run --extra analysis python scripts/brain_repr.py measure   # 180개 조합 측정 + 선정 + 보고서
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import pickle
import sys
import time
from pathlib import Path

from flypac.brain import representation as R
from flypac.core.config import load_config

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "runs" / "brain" / "representation"


def phase_data() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config(REPO / "configs" / "brain_repr_v1.yaml")
    t0 = time.perf_counter()
    rows = R.generate_rows(cfg)
    print(f"실행 {time.perf_counter() - t0:.1f}s, 행 {len(rows['seed'])}")
    data = {}
    bad = []
    for name, (lo, hi) in R.SPLITS.items():
        sd = R.build_split(name, R.split_rows(rows, lo, hi))
        counts = R.sample_counts(sd)
        short = R.count_shortages(counts)
        print(f"[{name}] hash {sd.data_hash[:16]} 행 {len(sd.rows['seed'])}")
        print("  제외·유일:", sd.exclusions)
        print("  표본 수:", counts)
        print("  미달:", short or "없음")
        bad += [f"{name}:{x}" for x in short]
        data[name] = sd
    with (OUT / "data.pkl").open("wb") as f:
        pickle.dump(data, f)
    print("자료 부족:" if bad else "표본 수 통과", bad)
    return 1 if bad else 0


def phase_measure(workers: int, threads: int) -> int:
    """180개 조합을 모두 측정한다 (13.7 1단계). 작업 하나 = (배정 시드, ρ). 결과는 줄 단위로 바로 저장한다."""
    data_path = OUT / "data.pkl"
    with data_path.open("rb") as f:
        data = pickle.load(f)
    tr, ev = data["dev_train"], data["dev_eval"]
    short = R.count_shortages(R.sample_counts(tr)) + R.count_shortages(R.sample_counts(ev))
    if short:
        print("자료 부족, 측정하지 않는다:", short)
        return 1
    ctrl = R.control_scores(tr, ev)
    (OUT / "control.json").write_text(json.dumps(ctrl, ensure_ascii=False, indent=1), encoding="utf-8")
    print("대조:", {k: v for k, v in ctrl.items() if "general" in k or k == "ok"})
    if not ctrl["ok"]:
        print("대조 초과: 자료 구성 코드의 버그다. 회로를 판정하지 않고 멈춘다.")
        return 1
    med = R.reference_odor(tr.rows)
    print("기준 냄새 중앙값(좌·앞·우·뒤):", med.tolist())
    lines = OUT / "tasks.jsonl"
    done = set()
    if lines.exists():
        for ln in lines.read_text(encoding="utf-8").splitlines():
            r = json.loads(ln)
            done.add((r["seed"], r["rho"]))
    tasks = [(s, rho, str(data_path), med.tolist()) for s in R.ASSIGN_SEEDS for rho in R.RHOS if (s, rho) not in done]
    print(f"작업 {len(tasks)}개 (이미 끝난 {len(done)}개), 프로세스 {workers}, 스레드 {threads}")
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = str(threads)   # 자식 프로세스가 numpy를 불러오기 전에 상속한다
    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool, lines.open("a", encoding="utf-8") as fh:
        for i, r in enumerate(pool.imap_unordered(R.measure_task, tasks), 1):
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
            ps = [c["pass"] for c in r["combos"]]
            print(f"[{i}/{len(tasks)}] 시드 {r['seed']} ρ {r['rho']} 통과 {ps}  {time.perf_counter() - t0:.0f}s", flush=True)
    return 0


def _git_commit() -> str:
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def phase_report(with_aux: bool) -> int:
    """작업 결과를 모아 선정하고 results.json, report.md를 쓴다. 후보가 있으면 보조 ②도 한다 (13.4)."""
    import numpy as np
    import sklearn
    with (OUT / "data.pkl").open("rb") as f:
        data = pickle.load(f)
    tasks = [json.loads(ln) for ln in (OUT / "tasks.jsonl").read_text(encoding="utf-8").splitlines()]
    ctrl = json.loads((OUT / "control.json").read_text(encoding="utf-8"))
    med = R.reference_odor(data["dev_train"].rows)
    res = R.build_results(tasks, data, ctrl, med)
    if res["n_combos"] != len(R.ASSIGN_SEEDS) * len(R.RHOS) * len(R.CS):
        print(f"조합 {res['n_combos']}개: 180개가 아니다. 측정을 마친 뒤 다시 한다.")
        return 1
    aux = None
    cand = res["candidate"]
    if with_aux and cand is not None:
        from flypac.brain.flybrain import build_circuit, load_brain_config, repo_path
        bcfg = load_brain_config("configs/brain_v1.yaml")
        circ = build_circuit(bcfg.circuit.raw, load_config(repo_path(bcfg.circuit.assign)), cand["seed"])
        t0 = time.perf_counter()
        aux = R.aux_h_test(circ, cand["rho"], cand["theta"], data["dev_train"], data["dev_eval"])
        print(f"보조 ② {time.perf_counter() - t0:.0f}s")
        res["aux_h"] = {t: {str(k): v for k, v in d.items()} for t, d in aux.items()}
        aux = {t: {str(k): v for k, v in d.items()} for t, d in aux.items()}
    (OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "report.md").write_text(R.format_report(res, aux, sklearn.__version__, _git_commit()), encoding="utf-8")
    print("후보:", cand, "| 통과 시드 수:", res["passing_seed_count"], "/ 20")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "data":
        sys.exit(phase_data())
    if cmd == "measure":
        sys.exit(phase_measure(int(os.environ.get("REPR_WORKERS", 5)), int(os.environ.get("REPR_THREADS", 3))))
    if cmd == "report":
        sys.exit(phase_report("--no-aux" not in sys.argv))
    print(__doc__)
    sys.exit(2)
