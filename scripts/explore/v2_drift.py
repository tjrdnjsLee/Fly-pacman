"""표류 대 확산: 학습 팔의 가중치는 막걸음보다 빠르게, 한 방향으로 움직였는가 (2026-11-21, 사후 탐색, 실행 전 고정).

위치. 42.14는 learn과 부호 대조군(ctrl)을 승률로 갈랐다. 이 스크립트는 같은 두 팔이 100판마다 남긴 가중치만 읽어
가중치 공간의 움직임을 물리의 말로 잰다. 새로 돌리는 판은 없다. 37.2(v1 본 실험 진단)와 같은 모형이다:

    Δw(t) = μ t + ξ(t),   Var ξ(t) = σ² t (독립 증분),   a = μ²/σ²
    ‖Δw(t)‖ ∝ t^H:  막걸음(μ = 0)이면 H = 1/2, 표류가 이기면 H → 1
    cos(Δw_s, Δw_t) = √(s/t) · √((a t + 1)/(a s + 1))
    s = t/2를 넣어 풀면  x ≡ a t = (2c² − 1)/(1 − c²),  표류 몫 f = x/(1 + x)   (c = cos(Δw_{t/2}, Δw_t))

입력: runs/vision_confirm_v1_learn · _ctrl/results.json (42.12, 학습 시드 6400 ~ 6419, 3000판, 가중치 100판마다).
  runs/v2_sparse_v1_s0 · _s1/results.json이 둘 다 있으면 sparse(λ = 0) 팔도 같이 잰다 (보고).
가중치 = 파일의 dw_masked (초파리 4마리 × 실제 시냅스), 시드마다 한 벡터로 잇는다. 정규화 ‖w0‖ = ‖Δw_3000‖ / results.json의 dw_rel
  (42.14 표의 ‖Δw‖/‖w0‖와 같은 단위가 되게. 시드마다 같은 값이 나와야 하며, 어긋남을 출력한다).

지표 (시드마다):
  H     ln‖Δw(t)‖를 ln t에 최소제곱 (t = 100, 200, …, 3000)
  c½    cos(Δw_1500, Δw_3000),  x · f는 위 식 (x < 0이면 f = 0으로 두고 x는 그대로 적는다)
  c⅒    cos(Δw_300, Δw_3000)과 x로 예측한 값 (모형 점검, 37.2와 같은 방식)
  공유  같은 초파리 번호끼리 다른 시드와의 cos 평균 (20개의 독립 학습이 같은 방향을 찾았는가. 입력과 무관한 행동 편향도 섞일 수 있다)

읽기 (결과 전 고정, 2026-11-21. 판정이 아니라 기술이다. 다중 비교 보정 없음):
  D1. ctrl의 H 95% 신뢰구간이 0.5를 포함하면 "대조군은 막걸음" (예상). learn − ctrl의 H 짝 차이(학습 시드 위치로 짝, 20쌍)가
      0 제외 양수이면 "학습 팔의 가중치는 막걸음보다 빠르게 멀어졌다 (표류가 있다)".
  D2. learn − ctrl의 f 짝 차이 (보고).
  D3. 공유 cos: learn과 ctrl (보고). ctrl은 0 근처가 예상.
미리 적은 예측 (Claude): ctrl H ≈ 0.5 · f ≈ 0 · 공유 ≈ 0, learn H > 0.5 · f > 0 · 공유 > 0. 틀려도 그대로 적는다.

  저장소 루트에서 (몇 분, 학습이 돌고 있어도 된다):
  uv run python scripts/explore/v2_drift.py
  sparse 팔이 끝난 뒤 다시 돌리면 sparse까지 넣은 새 기록을 쓴다 (V2D_TAG=with_sparse).

COOP_TEST=1이면 COOP_ROOT의 시험 결과를 읽는다 (가중치 2판마다, 4판).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
import final_eval as fe  # noqa: E402
from flypac.recording.weights import parse_weight_filename, read_weight_file  # noqa: E402

TEST = os.environ.get("COOP_TEST") == "1"
ROOT = Path(os.environ.get("COOP_ROOT", REPO / "runs"))
TAG = os.environ.get("V2D_TAG", "")
OUT = ROOT / (f"v2_drift_v1_{TAG}.json" if TAG else "v2_drift_v1.json")
if OUT.exists():
    sys.exit(f"멈춤: {OUT}가 이미 있다 (다른 이름이 필요하면 V2D_TAG)")


def load_arm(results_paths):
    """[(학습 폴더, 학습 시드들, n_train, dw_rel)] → 시드마다 {t: Δw 벡터}, ‖w0‖ 추정."""
    seeds, traj, w0n, n_train = [], {}, [], None
    for p in results_paths:
        r = json.loads(p.read_text(encoding="utf-8"))
        d = ROOT / r["train_dir"]
        nt = int(r["n_train"])
        if n_train is None:
            n_train = nt
        if nt != n_train:
            sys.exit(f"멈춤: {p}의 n_train {nt}이 앞 조각({n_train})과 다르다")
        files = {}
        for f in (d / "weights").glob("*.npz"):
            parsed = parse_weight_filename(f.name)
            if parsed:
                files[parsed] = f
        for i, s in enumerate(int(x) for x in r["train_seeds"]):
            ts = sorted(t for (ss, t) in files if ss == s)
            if not ts or ts[-1] != nt:
                sys.exit(f"멈춤: {d}에 시드 {s}의 마지막 판({nt}) 가중치가 없다")
            traj[s] = {t: read_weight_file(files[(s, t)])["dw_masked"].astype(np.float64) for t in ts}
            seeds.append(s)
            last = np.linalg.norm(traj[s][nt])
            if float(r["dw_rel"][i]) > 0:
                w0n.append(last / float(r["dw_rel"][i]))
    return seeds, traj, n_train, w0n


def cos(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    return float(u.ravel() @ v.ravel() / (nu * nv)) if nu > 0 and nv > 0 else float("nan")


def nearest(ts, want):
    return min(ts, key=lambda t: (abs(t - want), t))


def arm_metrics(seeds, traj, n_train, W0N):
    per = {k: [] for k in ("H", "c_half", "x", "f", "c_tenth", "c_tenth_pred", "shared")}
    curves = {}
    for s in seeds:
        ts = sorted(traj[s])
        D = np.array([np.linalg.norm(traj[s][t]) / W0N for t in ts])
        curves[s] = {"t": ts, "D": D.tolist()}
        ok = D > 0
        H = float(np.polyfit(np.log(np.array(ts)[ok]), np.log(D[ok]), 1)[0]) if ok.sum() >= 2 else float("nan")
        th, tt = nearest(ts, n_train / 2), nearest(ts, n_train / 10)
        ch = cos(traj[s][th], traj[s][n_train])
        x = (2 * ch ** 2 - 1) / (1 - ch ** 2) if ch < 1 else float("inf")
        f = max(x, 0.0) / (1 + max(x, 0.0)) if np.isfinite(x) else 1.0
        r = tt / n_train
        ct = cos(traj[s][tt], traj[s][n_train])
        xp = max(x, 0.0)
        ctp = float(np.sqrt(r) * np.sqrt((xp + 1) / (r * xp + 1))) if np.isfinite(xp) else 1.0
        for k, v in (("H", H), ("c_half", ch), ("x", x), ("f", f), ("c_tenth", ct), ("c_tenth_pred", ctp)):
            per[k].append(float(v))
    # 공유 방향: 같은 초파리 번호, 다른 시드
    U = np.stack([traj[s][n_train] for s in seeds])   # (시드, F, 시냅스)
    nS, F = U.shape[0], U.shape[1]
    for i in range(nS):
        vals = [cos(U[i, f], U[j, f]) for f in range(F) for j in range(nS) if j != i]
        per["shared"].append(float(np.mean(vals)) if vals else float("nan"))
    return per, curves


ARMS = {"learn": [ROOT / "vision_confirm_v1_learn" / "results.json"],
        "ctrl": [ROOT / "vision_confirm_v1_ctrl" / "results.json"]}
sp = [ROOT / f"v2_sparse_v1_s{k}" / "results.json" for k in (0, 1)]
if all(p.exists() for p in sp):
    ARMS["sparse"] = sp
for a, ps in ARMS.items():
    for p in ps:
        if not p.exists():
            sys.exit(f"멈춤: {p}가 없다")

res = {"model": "DECISIONS 37.2", "arms": {}}
W0_all = []
loaded = {}
for a, ps in ARMS.items():
    seeds, traj, nt, w0n = load_arm(ps)
    loaded[a] = (seeds, traj, nt)
    W0_all += w0n
W0N = float(np.median(W0_all))
dev = float(np.max(np.abs(np.array(W0_all) / W0N - 1))) if W0_all else float("nan")
print(f"‖w0‖ 추정 {W0N:.4f} (시드 · 팔 사이 최대 상대 어긋남 {dev:.2e})")
for a, (seeds, traj, nt) in loaded.items():
    per, curves = arm_metrics(seeds, traj, nt, W0N)
    res["arms"][a] = {"seeds": seeds, "n_train": nt, "per_seed": per, "curves": curves}
res["w0_norm"] = W0N
res["w0_max_rel_dev"] = dev


def ci0(v):
    v = [x for x in v if np.isfinite(x)]
    return fe.paired_ci(v, [0.0] * len(v), 0.95) if len(v) >= 2 else None


def fmt(c, nd=3):
    return "—" if c is None else f"{c['mean']:+.{nd}f} [{c['lo']:+.{nd}f}, {c['hi']:+.{nd}f}]"


print("\n  팔      H (성장 지수)                 표류 몫 f                     공유 cos                      c⅒ 측정 / 예측")
for a in res["arms"]:
    p = res["arms"][a]["per_seed"]
    print(f"  {a:<7} {fmt(ci0(p['H'])):<29} {fmt(ci0(p['f'])):<29} {fmt(ci0(p['shared'])):<29} "
          f"{np.nanmean(p['c_tenth']):.3f} / {np.nanmean(p['c_tenth_pred']):.3f}")
res["read"] = {}
c = ci0(res["arms"]["ctrl"]["per_seed"]["H"])
d1a = bool(c is not None and c["lo"] <= 0.5 <= c["hi"])
print(f"\n  D1a. ctrl H 신뢰구간이 0.5를 포함: {'예 (대조군은 막걸음)' if d1a else '아니오'}")
res["read"]["D1a_ctrl_H_contains_half"] = d1a


def paired(a, b, key):
    A = res["arms"][a]["per_seed"][key]; B = res["arms"][b]["per_seed"][key]
    if res["arms"][a]["seeds"] != res["arms"][b]["seeds"]:
        sys.exit(f"멈춤: {a}와 {b}의 학습 시드가 다르다 (짝 불가)")
    ok = [i for i in range(len(A)) if np.isfinite(A[i]) and np.isfinite(B[i])]
    return fe.paired_ci([A[i] for i in ok], [B[i] for i in ok], 0.95) if len(ok) >= 2 else None


r1 = paired("learn", "ctrl", "H")
d1b = bool(r1 is not None and r1["excludes_zero"] and r1["mean"] > 0)
print(f"  D1b. learn − ctrl H: {fmt(r1)} → " + ("표류가 있다 (막걸음보다 빠르게 멀어졌다)" if d1b else "구별하지 못했다"))
r2 = paired("learn", "ctrl", "f")
print(f"  D2.  learn − ctrl f: {fmt(r2)} (보고)")
print(f"  D3.  공유 cos learn {np.nanmean(res['arms']['learn']['per_seed']['shared']):+.4f}, "
      f"ctrl {np.nanmean(res['arms']['ctrl']['per_seed']['shared']):+.4f} (보고)")
res["read"].update(D1b={"ci": r1, "met": d1b}, D2={"ci": r2})
if "sparse" in res["arms"]:
    for key in ("H", "f"):
        r = paired("sparse", "ctrl", key)
        print(f"  보고: sparse − ctrl {key}: {fmt(r)}")
        res["read"][f"sparse_ctrl_{key}"] = r
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float) + "\n", encoding="utf-8")
print(f"기록: {OUT}")
