"""걸음 진단: 두 팔의 성장 지수가 왜 1/2보다 작은가, 대조군의 공유 방향은 어디서 왔나 (2026-11-21, 사후 진단).

위치. v2_drift.py(커밋 7c77540, 2026-11-21 20:3x)의 결과를 본 뒤에 만든 진단이다. 그 결과:
  learn H 0.453, ctrl H 0.318 (둘 다 1/2보다 작다), ctrl 공유 cos 0.158 (0이 아니다), 37.2 모형의 c⅒ 예측이 측정보다 크다.
이 스크립트의 숫자는 보기 전에 아래 읽기를 고정했다. 새로 돌리는 판은 없다: 42.12 확인 실행이 남긴
  가중치(100판마다)와 재생(판 1 · 3000, 뇌 기록 포함)만 읽는다. sparse 팔(v2_sparse.py)이 끝나 있으면 같이 잰다 (보고).

지표 (시드마다, Δ_k = Δw(t_k) − Δw(t_{k−1}), 100판 구간의 걸음):
  s_k   ‖Δ_k‖/‖w0‖.  ν = −(ln s_k를 ln t_k에 최소제곱한 기울기, t_k = 구간 가운데).
        증분이 서로 독립이고 s ∝ t^{−ν}이면 ‖Δw(T)‖² ∝ Σ s_k² ∝ T^{1−2ν}, 곧 H_예측 = 1/2 − ν.
  지속  cos(Δ_k, Δ_{k+1})의 평균 (이어진 두 구간의 걸음이 같은 쪽인가. 막걸음이면 0, 표류가 있으면 양수.
        37.2 모형과 달리 걸음 크기가 일정하다는 가정이 필요 없다)
  바닥  판 3000에서 w = 0(dw = −w0)인 시냅스 몫. 공유 cos를 "어느 시드에서든 바닥에 닿은 시냅스"를 빼고 다시 잰다.
  굳음  재생의 확정 결정(갈 수 있는 길 2개 이상)에서 1 − π(고른 행동). 학습 규칙의 (δ − π) 크기다.
        균등 정책이면 1 − 1/갈래 수. 판 1(Δw ≈ 0, 거의 균등)과 판 3000을 비교한다.
  DAN   같은 결정의 |R_n − R̄| (재생의 dan, 부호 대조 전 값). 판 1과 판 3000.

읽기 (고정, 2026-11-21. 사후 진단이며 판정이 아니다. 다중 비교 보정 없음):
  G1 걸음 감소: ν의 95% 신뢰구간이 0 제외 양수 → "걸음이 시간이 갈수록 작아졌다". ctrl의 H_예측과 v2_drift의 H(0.318)를 나란히 적는다.
  G2 원인: (1 − π)의 판 3000/판 1 비와 |DAN|의 판 3000/판 1 비를, 걸음의 끝/처음 비(마지막 3구간 / 처음 3구간)와 나란히 적는다.
     (1 − π) 비가 0 제외로 1보다 작으면 "정책이 굳어 학습 신호가 작아졌다".
  G3 표류: learn의 지속이 0 제외 양수이고 ctrl의 지속이 0을 포함 → "학습 팔만 걸음의 방향이 이어진다 (표류)".
  G4 바닥: ctrl의 바닥 뺀 공유 cos가 0을 포함 → "대조군의 공유 방향은 w ≥ 0 바닥에서 왔다". 바닥 몫이 1% 미만이면 "바닥 아님".
미리 적은 예측 (Claude): G1 두 팔 모두 예, ctrl H_예측 ≈ 0.32. G2 learn은 굳음, ctrl은 덜 굳음. G3 예. G4 예. 틀려도 그대로 적는다.

  저장소 루트에서 (몇 분, 학습이 돌고 있어도 된다. 메모리 약 2 GB):
  uv run python scripts/explore/v2_stepdiag.py
  sparse가 끝난 뒤 다시: SD_TAG=with_sparse uv run python scripts/explore/v2_stepdiag.py

COOP_TEST=1이면 COOP_ROOT의 시험 결과를 읽는다 (가중치 2판마다, 4판, 재생 판 1 · 4).
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
from flypac.replay.reader import read_replay  # noqa: E402

TEST = os.environ.get("COOP_TEST") == "1"
ROOT = Path(os.environ.get("COOP_ROOT", REPO / "runs"))
TAG = os.environ.get("SD_TAG", "")
OUT = ROOT / (f"v2_stepdiag_v1_{TAG}.json" if TAG else "v2_stepdiag_v1.json")
if OUT.exists():
    sys.exit(f"멈춤: {OUT}가 이미 있다 (다른 이름이 필요하면 SD_TAG)")
CIRC = np.load(REPO / "data" / "circuit" / "mb_v2.npz")
W0M = CIRC["w0"][CIRC["M"]].astype(np.float64)         # 연결 마스크 안의 w0, dw_masked와 같은 순서
F = 4
W0N = float(np.linalg.norm(CIRC["w0"])) * np.sqrt(F)   # vision_confirm의 ‖w0‖ 정규화와 같다
print(f"회로 v2: 시냅스 {W0M.size}개, ‖w0‖·√F = {W0N:.4f} (v2_drift의 추정 343.7066과 같아야 한다)")

ARMS = {"learn": [ROOT / "vision_confirm_v1_learn" / "results.json"],
        "ctrl": [ROOT / "vision_confirm_v1_ctrl" / "results.json"]}
sp = [ROOT / f"v2_sparse_v1_s{k}" / "results.json" for k in (0, 1)]
if all(p.exists() for p in sp):
    ARMS["sparse"] = sp
for ps in ARMS.values():
    for p in ps:
        if not p.exists():
            sys.exit(f"멈춤: {p}가 없다")


def cos(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    return float(u.ravel() @ v.ravel() / (nu * nv)) if nu > 0 and nv > 0 else float("nan")


def replay_stats(path):
    """확정 결정(갈래 ≥ 2)의 1 − π(고른 행동), 균등이면의 값, |DAN|."""
    _, a = read_replay(path)
    com = a["committed"] & (a["legal"].sum(-1) >= 2) & (a["action"] >= 0)
    t, f = np.nonzero(com)
    act = a["action"][t, f].astype(int)
    pi = a["pi"][t, f].astype(np.float64)
    nl = a["legal"][t, f].sum(-1)
    return {"one_minus_pi": float(np.mean(1 - pi[np.arange(len(act)), act])) if len(act) else float("nan"),
            "uniform": float(np.mean(1 - 1 / nl)) if len(act) else float("nan"),
            "dan_abs": float(np.mean(np.abs(a["dan"][t, f].astype(np.float64)))) if len(act) else float("nan"),
            "n": int(len(act))}


def arm(paths):
    per = {k: [] for k in ("nu", "H", "H_pred", "step_ratio", "persist", "floor", "omp1", "omp3000", "uni",
                           "omp_ratio", "dan1", "dan3000", "dan_ratio")}
    finals, seeds = [], []
    for p in paths:
        r = json.loads(p.read_text(encoding="utf-8"))
        d = ROOT / r["train_dir"]; nt = int(r["n_train"])
        files = {}
        for fp in (d / "weights").glob("*.npz"):
            parsed = parse_weight_filename(fp.name)
            if parsed:
                files[parsed] = fp
        for s in (int(x) for x in r["train_seeds"]):
            ts = sorted(t for (ss, t) in files if ss == s)
            if not ts or ts[-1] != nt:
                sys.exit(f"멈춤: {d}에 시드 {s}의 마지막 판 가중치가 없다")
            W = [read_weight_file(files[(s, t)])["dw_masked"].astype(np.float64) for t in ts]
            if W[0].shape != (F, W0M.size):
                sys.exit(f"멈춤: 가중치 모양 {W[0].shape}이 회로({F}, {W0M.size})와 다르다")
            steps = [W[0]] + [W[k] - W[k - 1] for k in range(1, len(W))]
            s_k = np.array([np.linalg.norm(x) / W0N for x in steps])
            mid = np.array([t - (ts[0] if k == 0 else ts[k] - ts[k - 1]) / 2 for k, t in enumerate(ts)], dtype=float)
            ok = s_k > 0
            nu = -float(np.polyfit(np.log(mid[ok]), np.log(s_k[ok]), 1)[0]) if ok.sum() >= 2 else float("nan")
            D = np.array([np.linalg.norm(x) / W0N for x in W])
            H = float(np.polyfit(np.log(np.array(ts, float)[D > 0]), np.log(D[D > 0]), 1)[0]) if (D > 0).sum() >= 2 \
                else float("nan")
            m = min(3, len(s_k))
            per["nu"].append(nu); per["H"].append(H); per["H_pred"].append(0.5 - nu)
            per["step_ratio"].append(float(np.mean(s_k[-m:]) / np.mean(s_k[:m])))
            per["persist"].append(float(np.nanmean([cos(steps[k], steps[k + 1]) for k in range(len(steps) - 1)]))
                                  if len(steps) >= 2 else float("nan"))
            per["floor"].append(float(np.mean(W[-1] <= -W0M[None, :] + 1e-12)))
            rp = {g: d / "replay" / f"regime1_s{s}_g{g}.npz" for g in (1, nt)}
            if all(x.exists() for x in rp.values()):
                a1, a2 = replay_stats(rp[1]), replay_stats(rp[nt])
                per["omp1"].append(a1["one_minus_pi"]); per["omp3000"].append(a2["one_minus_pi"])
                per["uni"].append(a2["uniform"])
                per["omp_ratio"].append(a2["one_minus_pi"] / a1["one_minus_pi"])
                per["dan1"].append(a1["dan_abs"]); per["dan3000"].append(a2["dan_abs"])
                per["dan_ratio"].append(a2["dan_abs"] / a1["dan_abs"] if a1["dan_abs"] > 0 else float("nan"))
            else:
                for k in ("omp1", "omp3000", "uni", "omp_ratio", "dan1", "dan3000", "dan_ratio"):
                    per[k].append(float("nan"))
            finals.append(W[-1]); seeds.append(s)
    U = np.stack(finals)                                   # (시드, F, 시냅스)
    at_floor = (U <= -W0M[None, None, :] + 1e-12).any(0)   # (F, 시냅스): 어느 시드에서든 바닥
    shared, shared_nf = [], []
    for i in range(len(seeds)):
        a = [cos(U[i, f], U[j, f]) for f in range(F) for j in range(len(seeds)) if j != i]
        b = [cos(U[i, f][~at_floor[f]], U[j, f][~at_floor[f]]) for f in range(F) for j in range(len(seeds)) if j != i]
        shared.append(float(np.mean(a)) if a else float("nan"))
        shared_nf.append(float(np.mean(b)) if b else float("nan"))
    per["shared"], per["shared_no_floor"] = shared, shared_nf
    return {"seeds": seeds, "per_seed": per, "floor_any_seed_frac": float(at_floor.mean())}


def ci0(v, ref=0.0):
    v = [x - ref for x in v if np.isfinite(x)]
    return fe.paired_ci(v, [0.0] * len(v), 0.95) if len(v) >= 2 else None


def fmt(c, ref=0.0, nd=3):
    if c is None:
        return "—"
    return f"{c['mean'] + ref:+.{nd}f} [{c['lo'] + ref:+.{nd}f}, {c['hi'] + ref:+.{nd}f}]"


res = {"after": "v2_drift_v1 (learn H 0.453, ctrl H 0.318, ctrl 공유 0.158)", "w0n": W0N, "arms": {}}
for a, ps in ARMS.items():
    res["arms"][a] = arm(ps)
    print(f"[{a}] 읽음: 시드 {len(res['arms'][a]['seeds'])}개", flush=True)

print("\n  팔      ν (걸음 감소)            H_예측 = ½ − ν     H (v2_drift와 같은 계산)   걸음 끝/처음")
for a, v in res["arms"].items():
    p = v["per_seed"]
    print(f"  {a:<7} {fmt(ci0(p['nu'])):<24} {np.nanmean(p['H_pred']):.3f}              {np.nanmean(p['H']):.3f}"
          f"                      {np.nanmean(p['step_ratio']):.3f}")
print("\n  팔      1−π 판1 / 판3000 (균등)        1−π 비 (3000/1)            |DAN| 판1 / 판3000     |DAN| 비")
for a, v in res["arms"].items():
    p = v["per_seed"]
    print(f"  {a:<7} {np.nanmean(p['omp1']):.3f} / {np.nanmean(p['omp3000']):.3f} ({np.nanmean(p['uni']):.3f})"
          f"          {fmt(ci0(p['omp_ratio'], 1.0), 1.0):<26} {np.nanmean(p['dan1']):.3f} / {np.nanmean(p['dan3000']):.3f}"
          f"        {np.nanmean(p['dan_ratio']):.3f}")
print("\n  팔      지속 cos(Δ_k, Δ_k+1)          바닥 몫 (판 3000)   어느 시드든 바닥   공유 cos → 바닥 빼고")
for a, v in res["arms"].items():
    p = v["per_seed"]
    print(f"  {a:<7} {fmt(ci0(p['persist'])):<29} {np.nanmean(p['floor']):.4f}              {v['floor_any_seed_frac']:.4f}"
          f"           {np.nanmean(p['shared']):+.3f} → {fmt(ci0(p['shared_no_floor']))}")

L, C = res["arms"]["learn"]["per_seed"], res["arms"]["ctrl"]["per_seed"]
read = {}
for a, p in (("learn", L), ("ctrl", C)):
    c = ci0(p["nu"]); read[f"G1_{a}"] = bool(c is not None and c["excludes_zero"] and c["mean"] > 0)
    c = ci0(p["omp_ratio"], 1.0); read[f"G2_{a}"] = bool(c is not None and c["excludes_zero"] and c["mean"] < 0)
cl, cc = ci0(L["persist"]), ci0(C["persist"])
read["G3"] = bool(cl is not None and cl["excludes_zero"] and cl["mean"] > 0 and cc is not None and not cc["excludes_zero"])
floor_c = float(np.nanmean(C["floor"]))
cn = ci0(C["shared_no_floor"])
read["G4"] = "바닥 아님 (바닥 몫 1% 미만)" if floor_c < 0.01 else (
    "대조군의 공유 방향은 바닥에서 왔다" if cn is not None and not cn["excludes_zero"] else "바닥만으로는 설명되지 않는다")
print("\n  G1 걸음 감소:  learn " + ("예" if read["G1_learn"] else "아니오") + ", ctrl " + ("예" if read["G1_ctrl"] else "아니오"))
print("  G2 정책 굳음:  learn " + ("예" if read["G2_learn"] else "아니오") + ", ctrl " + ("예" if read["G2_ctrl"] else "아니오"))
print("  G3 표류:       " + ("학습 팔만 걸음의 방향이 이어진다" if read["G3"] else "조건 미충족 (위 지속 cos를 본다)"))
print("  G4 바닥:       " + read["G4"])
res["read"] = read
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float) + "\n", encoding="utf-8")
print(f"기록: {OUT}")
