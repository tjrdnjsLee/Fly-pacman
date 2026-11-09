"""표현력 진단 v2 (판정에 쓰지 않음, 규칙 변경 아님): z 입력 크기 p_z = z × m × 배수.

  uv run --extra analysis python scripts/brain_repr_diag2.py measure | report
기존 개발 자료(data.pkl), 시드 0~4, ρ ∈ {0.1, 0.05}, c = 0 (θ = 0). 분류기·분할·특징 생성은 representation.py 그대로.
m = 개발-학습 결정 표본에서 열린 방향(벽 마스크 거짓)의 냄새 값 전체의 중앙값 (13.10 항목 4와 같은 선택, 네 채널을 합쳐 하나).
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import pickle
import sys
import time

import numpy as np

import brain_repr_diag as D1
from flypac.brain import representation as R

SRC, REPO = D1.SRC, D1.REPO
OUT = SRC / "diagnostics_v2"
RHOS = (0.1, 0.05)
MULTS = (0.5, 1.0, 2.0)
SEEDS = D1.SEEDS


def reference_m(train_rows) -> tuple[float, list[float]]:
    cand = np.flatnonzero(R.decision_mask(train_rows))
    o, wall = train_rows["odor"][cand], train_rows["wall"][cand]
    return float(np.median(o[~wall])), [float(np.median(o[~wall[:, c], c])) for c in range(4)]


def feats(circ, rho, p, zval):
    return R.kc_features(circ, rho, 0.0, p.odor, zval, p.phero, p.seed, p.game)


def task(args):
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    from flypac.brain.flybrain import build_circuit, load_brain_config, repo_path
    from flypac.core.config import load_config
    seed, rho, data_path, m = args
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    tr, ev = data["dev_train"], data["dev_eval"]
    bcfg = load_brain_config("configs/brain_v1.yaml")
    circ = build_circuit(bcfg.circuit.raw, load_config(repo_path(bcfg.circuit.assign)), seed)
    out = {"seed": seed, "rho": rho, "m": m, "mults": []}
    for mult in MULTS:
        t0 = time.perf_counter()
        pz = m * mult
        res = {"seed": seed, "rho": rho, "mult": mult, "p_z": pz}
        # 시험 4: z 채널 값 = z × m × 배수
        p4tr, p4 = tr.t4, ev.t4
        clf, conv = R.fit_clf(feats(circ, rho, p4tr, p4tr.z * pz), p4tr.y)
        pred = R.predict_masked(clf, feats(circ, rho, p4, p4.z * pz), p4.legal)
        nl = p4.legal.sum(1)
        t4 = {"converged": conv}
        for zv in (0, 1):
            mz = p4.z == zv
            t4[f"z{zv}"] = {"all": D1.rec(p4.y[mz], pred[mz]), "conf": D1.conf(p4.y[mz], pred[mz])}
            for n_leg in (2, 3):
                mm = mz & (nl == n_leg)
                t4[f"z{zv}"][f"L{n_leg}"] = D1.rec(p4.y[mm], pred[mm])
        strongest = np.where(p4.legal, p4.odor, -np.inf).argmax(1)
        ok = p4.legal[np.arange(len(pred)), pred] & (pred != strongest)
        chance = (nl - 1) / nl
        mz = p4.z == 1
        sa = {"all": {"acc": float(ok[mz].mean()), "chance": float(chance[mz].mean()), "n": int(mz.sum())}}
        for n_leg in (2, 3):
            mm = mz & (nl == n_leg)
            sa[f"L{n_leg}"] = {"acc": float(ok[mm].mean()), "chance": float(chance[mm].mean()), "n": int(mm.sum())}
        t4["z1_set"] = sa
        res["t4"] = t4
        # J_K: 시험 2 평가 짝 (앞 절반 z = 0, 뒤 절반 z = 1)
        p2 = ev.t2
        X2 = feats(circ, rho, p2, p2.z * pz)
        n = p2.meta["pairs"]
        jk = R._jaccard_rows(X2[:n], X2[n:])
        res["jk"] = {"mean": float(np.nanmean(jk)) if np.isfinite(jk).any() else None,
                     "excluded_both_empty": int(np.isnan(jk).sum()), "n": int(n)}
        # 시험 1: z = 1 고정(학습·평가 모두), z 채널 값 = m × 배수
        t1tr, t1ev = tr.t1, ev.t1
        clf1, conv1 = R.fit_clf(feats(circ, rho, t1tr, np.full(len(t1tr), pz)), t1tr.y)
        pr1 = R.predict_masked(clf1, feats(circ, rho, t1ev, np.full(len(t1ev), pz)), t1ev.legal)
        r = D1.gradient_r(t1ev)
        t1 = {"converged": conv1, "all": D1.rec(t1ev.y, pr1)}
        for name, (lo, hi) in D1.GRADIENT.items():
            mm = (r >= lo) & (r < hi)
            t1[name] = D1.rec(t1ev.y[mm], pr1[mm])
        res["t1_z1"] = t1
        res["sec"] = time.perf_counter() - t0
        out["mults"].append(res)
    return out


def measure(workers=5, threads=3) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    dp = SRC / "data.pkl"
    with dp.open("rb") as f:
        tr = pickle.load(f)["dev_train"]
    m, per = reference_m(tr.rows)
    (OUT / "m.json").write_text(json.dumps({"m": m, "per_channel_open_median": per,
                                            "note": "열린 방향 냄새 값 전체의 중앙값(네 채널 합침). 기존 채널별 중앙값은 per_channel"},
                                           ensure_ascii=False, indent=1), encoding="utf-8")
    print("m", m, per, flush=True)
    lines = OUT / "tasks.jsonl"
    done = set()
    if lines.exists():
        for ln in lines.read_text(encoding="utf-8").splitlines():
            r = json.loads(ln)
            done.add((r["seed"], r["rho"]))
    tasks = [(s, rho, str(dp), m) for s in SEEDS for rho in RHOS if (s, rho) not in done]
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = str(threads)
    t0 = time.perf_counter()
    with mp.get_context("spawn").Pool(workers) as pool, lines.open("a", encoding="utf-8") as fh:
        for i, r in enumerate(pool.imap_unordered(task, tasks), 1):
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[{i}/{len(tasks)}] 시드 {r['seed']} ρ {r['rho']} {time.perf_counter() - t0:.0f}s", flush=True)
    return 0


def ms(vals, nd=3):
    v = [x for x in vals if x is not None and not np.isnan(x)]
    return f"{np.mean(v):.{nd}f} [{min(v):.{nd}f}–{max(v):.{nd}f}]" if v else "—"


def report() -> int:
    rows = [json.loads(ln) for ln in (OUT / "tasks.jsonl").read_text(encoding="utf-8").splitlines()]
    by = {}
    for t in rows:
        for r in t["mults"]:
            by.setdefault((r["rho"], r["mult"]), []).append(r)
    old = [json.loads(ln) for ln in (SRC / "tasks.jsonl").read_text(encoding="utf-8").splitlines()]
    oldc = {(r["seed"], r["rho"]): [c for c in r["combos"] if c["c"] == 0][0] for r in old}
    mj = json.loads((OUT / "m.json").read_text(encoding="utf-8"))
    L = ["# 표현력 진단 v2 (판정에 쓰지 않음, 규칙 변경 아님)", "",
         f"- m = {mj['m']:.4f} (열린 방향 냄새 전체의 중앙값). 채널별(좌·앞·우·뒤): " + ", ".join(f"{v:.4f}" for v in mj["per_channel_open_median"]),
         "- 시드 0~4, c = 0 (θ = 0), 값은 시드 평균 [최소–최대]. 현행 = z 채널 값 z (z=1일 때 1.0).", "",
         "## 1. 현행 z 짝 J_K (기존 results, 시험 2 평가 짝)", "", "| ρ | J_K | 비교용: 둘 다 빈 짝 수 |", "|---|---|---|"]
    for rho in RHOS:
        cs = [oldc[(s, rho)] for s in SEEDS]
        L.append(f"| {rho} | {ms([c['diag_jk']['mean'] for c in cs])} | {[c['diag_jk']['excluded_both_empty'] for c in cs]} |")
    hdr = ["| ρ | z 채널 값 | J_K | t4 z0 전체 | z0 L2 | z0 L3 | t4 z1 전체 | z1 L2 | z1 L3 | 집합 전체 | 집합 L2 | 집합 L3 | t1(z=1) 전체 | t1 약함 | t1 강함 |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    L += ["", "## 2. z 채널 값 = z × m × 배수", ""] + hdr
    for rho in RHOS:
        # 현행 행 (diagnostics_v1 + 기존 results)
        d1 = [json.loads(ln) for ln in (SRC / "diagnostics_v1" / "tasks.jsonl").read_text(encoding="utf-8").splitlines()]
        cur = [c for t in d1 for c in t["combos"] if t["rho"] == rho and c["c"] == 0]
        g = lambda f: ms([f(c) for c in cur])
        L.append(f"| {rho} | 현행(1.0) | {ms([oldc[(c['seed'], rho)]['diag_jk']['mean'] for c in cur])} | "
                 f"{g(lambda c: c['t4']['z0']['all']['balanced'])} | {g(lambda c: c['t4']['z0']['L2']['balanced'])} | {g(lambda c: c['t4']['z0']['L3']['balanced'])} | "
                 f"{g(lambda c: c['t4']['z1']['all']['balanced'])} | {g(lambda c: c['t4']['z1']['L2']['balanced'])} | {g(lambda c: c['t4']['z1']['L3']['balanced'])} | "
                 f"{g(lambda c: c['t4']['z1_set']['all']['acc'])} | {g(lambda c: c['t4']['z1_set']['L2']['acc'])} | {g(lambda c: c['t4']['z1_set']['L3']['acc'])} | "
                 f"{ms([oldc[(c['seed'], rho)]['diag_t1_z1_only']['acc'] for c in cur])} | — | — |")
        for mult in MULTS:
            rs = by[(rho, mult)]
            g = lambda f: ms([f(r) for r in rs])
            L.append(f"| {rho} | {mult}×m = {mj['m'] * mult:.3f} | {g(lambda r: r['jk']['mean'])} | "
                     f"{g(lambda r: r['t4']['z0']['all']['balanced'])} | {g(lambda r: r['t4']['z0']['L2']['balanced'])} | {g(lambda r: r['t4']['z0']['L3']['balanced'])} | "
                     f"{g(lambda r: r['t4']['z1']['all']['balanced'])} | {g(lambda r: r['t4']['z1']['L2']['balanced'])} | {g(lambda r: r['t4']['z1']['L3']['balanced'])} | "
                     f"{g(lambda r: r['t4']['z1_set']['all']['acc'])} | {g(lambda r: r['t4']['z1_set']['L2']['acc'])} | {g(lambda r: r['t4']['z1_set']['L3']['acc'])} | "
                     f"{g(lambda r: r['t1_z1']['all']['balanced'])} | {g(lambda r: r['t1_z1']['weak']['balanced'])} | {g(lambda r: r['t1_z1']['strong']['balanced'])} |")
    L += ["", "집합 정답 우연 수준: 전체 0.536, 합법 방향 2개 0.5, 3개 0.667. 현행 행의 t1(z=1)은 기존 `diag_t1_z1_only`(전체만).",
          "시험 4 균형 정확도의 표본 수: z별 전체 34,986, 합법 2개 27,528, 3개 7,458. 시험 1 평가 17,546 (약함 6,939, 강함 3,630).", ""]
    L += ["## 시험 4 z=1 방향별 재현율 (좌·앞·우)", "", "| ρ | 배수 | 좌 | 앞 | 우 |", "|---|---|---|---|---|"]
    for rho in RHOS:
        for mult in MULTS:
            rs = by[(rho, mult)]
            L.append(f"| {rho} | {mult} | " + " | ".join(ms([r['t4']['z1']['all']['per_dir'][str(d)] for r in rs]) for d in R.DIRS) + " |")
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "measure":
        sys.exit(measure(int(os.environ.get("REPR_WORKERS", 5)), int(os.environ.get("REPR_THREADS", 3))))
    if cmd == "report":
        sys.exit(report())
    print(__doc__)
