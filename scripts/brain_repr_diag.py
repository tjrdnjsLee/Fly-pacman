"""표현력 시험 진단 v1 (판정에 쓰지 않음, 규칙 변경 아님). 기존 개발 자료(data.pkl)를 그대로 쓴다.

먼저 `scripts/brain_repr.py data`로 data.pkl을 만들어 두어야 한다 (기존 측정과 같은 자료). 다른 폴더의 자료를 쓰려면 REPR_DIR.

  uv run --extra analysis python scripts/brain_repr_diag.py measure   # 배정 시드 0~4 × 격자 9개
  uv run --extra analysis python scripts/brain_repr_diag.py report    # 표 만들기

분류기 설정, 학습/평가 분할, 특징 생성은 brain/representation.py 그대로. 평가 표본만 나눠 센다.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from flypac.brain import representation as R

REPO = Path(__file__).resolve().parents[1]
SRC = Path(os.environ.get("REPR_DIR", REPO / "runs" / "brain" / "representation"))   # data.pkl 등이 있는 폴더 (읽기만)
OUT = SRC / "diagnostics_v1"
SEEDS = (0, 1, 2, 3, 4)
GRADIENT = {"weak": (-np.inf, 0.1), "mid": (0.1, 0.3), "strong": (0.3, np.inf)}   # r = (o1 - o2) / o1 (DECISIONS 13.9)


def rec(y, pred, classes=R.DIRS):
    b, per = R.balanced_recall(y, pred, classes)
    return {"balanced": b, "general": float((pred == y).mean()) if len(y) else None, "n": int(len(y)),
            "per_dir": per}


def conf(y, pred):
    m = np.zeros((4, 4), dtype=np.int64)          # [정답 0..3, 예측 0..3]
    np.add.at(m, (y, pred), 1)
    return m.tolist()


def gradient_r(p):
    best, second, _ = R._top_two(p.odor, p.legal)
    return (best - second) / best


def measure_task(args):
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    from flypac.brain.flybrain import build_circuit, load_brain_config, repo_path
    from flypac.core.config import load_config
    seed, rho, data_path, med = args
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    tr, ev = data["dev_train"], data["dev_eval"]
    bcfg = load_brain_config("configs/brain_v1.yaml")
    circ = build_circuit(bcfg.circuit.raw, load_config(repo_path(bcfg.circuit.assign)), seed)
    uref = R.u_ref(circ, np.asarray(med))
    out = {"seed": seed, "rho": rho, "uref": uref, "combos": []}
    for c in R.CS:
        t0 = time.perf_counter()
        theta = c * uref
        res = {"seed": seed, "rho": rho, "c": c}
        # ---- 시험 1 (약한 · 강한 구배)
        clf, conv = R.fit_clf(R._probe_feats(circ, rho, theta, tr.t1), tr.t1.y)
        p1 = ev.t1
        pred = R.predict_masked(clf, R._probe_feats(circ, rho, theta, p1), p1.legal)
        r = gradient_r(p1)
        t1 = {"converged": conv, "all": rec(p1.y, pred)}
        for name, (lo, hi) in GRADIENT.items():
            m = (r >= lo) & (r < hi)
            t1[name] = rec(p1.y[m], pred[m])
        res["t1"] = t1
        # ---- 시험 4 (선택지 수별, z = 1 방향별 · 혼동, 집합 정답)
        clf, conv = R.fit_clf(R._probe_feats(circ, rho, theta, tr.t4), tr.t4.y)
        p4 = ev.t4
        pred = R.predict_masked(clf, R._probe_feats(circ, rho, theta, p4), p4.legal)
        nl = p4.legal.sum(1)
        t4 = {"converged": conv}
        for zv in (0, 1):
            mz = p4.z == zv
            t4[f"z{zv}"] = {"all": rec(p4.y[mz], pred[mz]), "conf": conf(p4.y[mz], pred[mz])}
            for n_leg in (2, 3):
                m = mz & (nl == n_leg)
                t4[f"z{zv}"][f"L{n_leg}"] = rec(p4.y[m], pred[m])
        # 집합 정답: z = 1에서 정답 = 합법 방향 중 "가장 센 방향이 아닌 것" (참고용 진단)
        o = np.where(p4.legal, p4.odor, -np.inf)
        strongest = o.argmax(1)
        ok = p4.legal[np.arange(len(pred)), pred] & (pred != strongest)
        mz = p4.z == 1
        chance = (nl - 1) / nl                                   # 합법 방향 중 임의로 고를 때 집합 정답일 확률
        sa = {"all": {"acc": float(ok[mz].mean()), "chance": float(chance[mz].mean()), "n": int(mz.sum())}}
        for n_leg in (2, 3):
            m = mz & (nl == n_leg)
            sa[f"L{n_leg}"] = {"acc": float(ok[m].mean()), "chance": float(chance[m].mean()), "n": int(m.sum())}
        t4["z1_set"] = sa
        res["t4"] = t4
        res["sec"] = time.perf_counter() - t0
        out["combos"].append(res)
    return out


def phase_measure(workers: int, threads: int) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    data_path = SRC / "data.pkl"
    with data_path.open("rb") as f:
        data = pickle.load(f)
    tr, ev = data["dev_train"], data["dev_eval"]
    med = R.reference_odor(tr.rows)
    # ---- 합법 방향 수 분포 (결정 표본), 자료 구성 확인
    dist = {}
    for name, sd in (("dev_train", tr), ("dev_eval", ev)):
        nl = sd.rows["legal"][R.decision_mask(sd.rows)].sum(1)
        dist[name] = {"decision_samples": {str(k): int((nl == k).sum()) for k in np.unique(nl)}}
        for tname, p in (("t1", sd.t1), ("t3", sd.t3)):
            q = p.legal.sum(1)
            dist[name][tname] = {str(k): int((q == k).sum()) for k in np.unique(q)}
        q = sd.t4.legal.sum(1)[sd.t4.meta["var"] == 0]
        dist[name]["t4_base"] = {str(k): int((q == k).sum()) for k in np.unique(q)}
    (OUT / "legal_count.json").write_text(json.dumps(dist, ensure_ascii=False, indent=1), encoding="utf-8")
    print(dist)
    lines = OUT / "tasks.jsonl"
    done = set()
    if lines.exists():
        for ln in lines.read_text(encoding="utf-8").splitlines():
            r = json.loads(ln)
            done.add((r["seed"], r["rho"]))
    tasks = [(s, rho, str(data_path), med.tolist()) for s in SEEDS for rho in R.RHOS if (s, rho) not in done]
    print(f"작업 {len(tasks)}개 (끝난 {len(done)}개), 프로세스 {workers}, 스레드 {threads}", flush=True)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = str(threads)
    t0 = time.perf_counter()
    with mp.get_context("spawn").Pool(workers) as pool, lines.open("a", encoding="utf-8") as fh:
        for i, r in enumerate(pool.imap_unordered(measure_task, tasks), 1):
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[{i}/{len(tasks)}] 시드 {r['seed']} ρ {r['rho']}  {time.perf_counter() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "measure":
        sys.exit(phase_measure(int(os.environ.get("REPR_WORKERS", 5)), int(os.environ.get("REPR_THREADS", 3))))
    if cmd == "report":
        from brain_repr_diag_report import main
        sys.exit(main())
    print(__doc__)
    sys.exit(2)
