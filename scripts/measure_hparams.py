"""하이퍼파라미터 측정 (DECISIONS 18절, TRD 10.7). 작은 과제 A ~ D는 돌리지 않고, 정답률은 재지도 보지도 않는다.

  uv run python scripts/measure_hparams.py            # 18.3의 순서대로 잰다 → runs/hparam/measure_v1.json
  uv run python scripts/measure_hparams.py confirm    # 18.2의 η 확인 격자를 두 조건으로 끝까지 적용 → confirm_v1.json
  uv run python scripts/measure_hparams.py whatif 0.8 1.0   # 참고용: η만 바꿔 8단계 값을 본다 (규칙의 선택이 아니다)

순서: κ → 회로 통계 → Λ0 → β → g → η → α 확인 → H와 20시행 비 검증. 범위가 비거나 18.4의 대안이 필요하면 거기서 멈춘다.
측정 시드 5000 ~ 5004 (뇌 · 환경 난수). 회로는 configs/brain_v1.yaml의 배정 시드 0, ρ = 0.05, θ = 0.
8단계는 과제 C의 앞 50시행을 돌리지만 엔트로피 H와 가중치 변화량만 기록한다 (정답 여부는 읽지 않는다).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

from flypac.brain.flybrain import FlyBrain, kc_drive, kc_select, load_brain_config, pn_input
from flypac.core.config import load_config
from flypac.core.types import FlyObs
from flypac.tasks import runner
from flypac.tasks.tmaze import F, FWD, LEFT, RIGHT, TMaze

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "runs" / "hparam"
BRAIN = "configs/brain_v1.yaml"
SEEDS = [5000, 5001, 5002, 5003, 5004]
TASKS = load_config(REPO / "configs" / "tasks_v1.yaml")
L = int(TASKS.env.corridor_ticks)
M_ODOR = float(TASKS.env.amp)
N_CELL = 500                      # 18.2: 과제 C 셀당 시행 수 (2000 / 4)
LN3 = math.log(3.0)
# 18.2 상수 (규칙에 적힌 값)
KAPPA_GRID = (0.15, 0.2, 0.25, 0.3)
BETA_GRID = (1, 2, 4, 8, 16)
ALPHA_GRID = (0.005, 0.01, 0.02)
ETA_FACTORS = (0.25, 0.5, 1.0, 2.0, 4.0)
P0_RANGE = (0.57, 0.68)
F_MIN, KAPPA_OVER_F_MAX = 0.90, 0.3
H_PRE_MIN, H_EARLY_MIN = 0.90, 0.85
LOCAL_RATIO = (0.015, 0.3)
ALPHA_NOISE_MAX, ALPHA_TIME_MAX = 0.1, 100
CELL_S1_SPREAD_MAX = 3.0

# 입력 칸: (이름, 냄새 [좌, 앞, 우, 뒤], z). C의 z = 0 칸은 B와 같은 입력이다.
def cells():
    a = M_ODOR
    return [("A", [a, 0, a, 0], 0.0), ("B/C z0 냄새 좌", [a, 0, 0, 0], 0.0), ("B/C z0 냄새 우", [0, 0, a, 0], 0.0),
            ("C z1 냄새 좌", [a, 0, 0, 0], 1.0), ("C z1 냄새 우", [0, 0, a, 0], 1.0)]


class Stop(Exception):
    """범위가 비었거나 대안이 필요하다: 멈추고 보고한다."""


def sigma(x):
    return 1.0 / (1.0 + math.exp(-x))


def h2(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


class _Rng:
    def __init__(self):
        self.g = np.random.default_rng(0)

    def __getitem__(self, r):
        return self.g


def obs_for(B, odor, z, junction):
    o = np.zeros((B, F, 4))
    o[:, 0] = odor
    zz = np.zeros((B, F))
    zz[:, 0] = z
    legal = np.zeros((B, F, 4), dtype=bool)
    decide = np.zeros((B, F), dtype=bool)
    if junction:
        legal[:, 0, LEFT] = legal[:, 0, RIGHT] = True
        decide[:, 0] = True
    else:
        legal[:, 0, FWD] = True
    return FlyObs(odor=o, phero=np.zeros((B, F, 4)), z=zz, legal=legal, decide=decide)


def run_trial(brain, odor, z, reward=None):
    """같은 입력의 시행 하나: 복도 L틱 + 갈림길. 갈림길 행동 (B,)을 돌려준다. reward가 있으면 결정 직후 준다."""
    B = brain.B
    alive = np.ones(B, dtype=bool)
    active = np.ones((B, F), dtype=bool)
    a = None
    for t in range(L + 1):
        o = obs_for(B, odor, z, t == L)
        act = brain.propose(o, alive)
        brain.commit(o.decide & (act >= 0), active)
        if t == L:
            a = act[:, 0].copy()
            if reward is not None:
                R = np.zeros((B, F))
                R[:, 0] = reward
                brain.give_reward(R)
    return a


def probe_pi(brain, odor, z):
    """뇌 복제본에 같은 입력을 L + 1틱 넣어 갈림길의 π를 읽는다 (원래 뇌와 난수는 건드리지 않는다)."""
    c = runner.clone_brain(brain)
    c.eta = 0.0
    c.x[:] = 0.0
    c.h[:] = 0.0
    B = c.B
    alive = np.ones(B, dtype=bool)
    for t in range(L + 1):
        o = obs_for(B, odor, z, t == L)
        c.propose(o, alive)
        if t < L:
            c.commit(np.zeros((B, F), dtype=bool), np.ones((B, F), dtype=bool))
    return c.last_pi[:, 0].copy(), c.last_kc[:, 0].copy()


def make_brain(circuit, **learn):
    return runner.build_brain(BRAIN, SEEDS, circuit=circuit, **learn)


# ------------------------------------------------------------------ 단계

def step1_kappa(rep):
    rows = []
    for k in KAPPA_GRID:
        f = 1 - (1 - k) ** (L + 1)
        rows.append({"kappa": k, "f": f, "kappa_over_f": k / f, "ok": f >= F_MIN and k / f <= KAPPA_OVER_F_MAX})
    ok = [r["kappa"] for r in rows if r["ok"]]
    if not ok:
        raise Stop("κ 격자에 조건을 만족하는 값이 없다")
    kappa = min(ok)
    rep["step1"] = {"grid": rows, "kappa": kappa, "S": (1 - (1 - kappa) ** (L + 1)) / kappa}
    return kappa, rep["step1"]["S"]


def step2_stats(circuit, kappa, rep):
    cfg = load_brain_config(BRAIN)
    rho, theta = float(cfg.kc.rho), float(cfg.kc.theta)
    k = int(math.ceil(rho * circuit.n_kc - 1e-9))
    bundle = circuit.mbon_bundle
    sizes = np.bincount(bundle, minlength=4)
    M, w0 = circuit.M, circuit.w0
    out = {"rho": rho, "theta": theta, "k": k, "n_kc": circuit.n_kc, "n_mbon": circuit.n_mbon,
           "bundle_sizes": sizes.tolist(), "nnz_M": int(M.sum()), "density_M": float(M.mean()),
           "w0_positive_mean": float(w0[M].mean()), "w0_positive_cv": float(w0[M].std() / w0[M].mean()),
           "w0_frobenius_on_M": float(np.linalg.norm(w0[M])), "cells": {}}
    sets = {}
    for name, odor, z in cells():
        p = pn_input(np.asarray(odor, float)[None], np.array([z]), np.zeros((1, 4)))
        u = kc_drive(p, circuit)
        kv = kc_select(u, theta, k, _Rng())[0]
        sets[name] = kv
        idx = np.flatnonzero(kv)
        dens, mean_w = {}, {}
        for a, nm in ((LEFT, "좌"), (RIGHT, "우")):
            cols = bundle == a
            dens[nm] = float(M[np.ix_(idx, cols)].mean())
            mean_w[nm] = float(w0[np.ix_(idx, cols)].mean())
        out["cells"][name] = {"k_eff": int(kv.sum()), "n_u_positive": int((u[0] > theta).sum()),
                              "mu_hat_density": dens, "mu_hat_mean_w0": mean_w}
    jk = {}
    for side in ("좌", "우"):
        a, b = sets[f"B/C z0 냄새 {side}"], sets[f"C z1 냄새 {side}"]
        jk[side] = float((a & b).sum() / (a | b).sum())
    out["J_K"] = jk
    out["q"] = {s: 2 * j / (1 + j) for s, j in jk.items()}
    out["q_worst"] = max(out["q"].values())
    out["mu_hat_worst_density"] = min(min(c["mu_hat_density"].values()) for c in out["cells"].values())
    out["mu_hat_worst_mean_w0"] = min(min(c["mu_hat_mean_w0"].values()) for c in out["cells"].values())
    out["k_eff_min"] = min(c["k_eff"] for c in out["cells"].values())
    rep["step2"] = out
    return out


def step3_lambda0(rep):
    lo, hi = (math.log(p / (1 - p)) for p in P0_RANGE)
    # 18.2의 "Λ0 ∈ [0.28, 0.75]"는 p0 범위의 로짓을 반올림한 값이다. 목표는 그 범위의 기하평균이다.
    lam_lo, lam_hi = 0.28, 0.75
    lam0 = math.sqrt(lam_lo * lam_hi)
    rep["step3"] = {"p0_range": P0_RANGE, "logit_of_p0_range": [lo, hi], "lambda0_range": [lam_lo, lam_hi],
                    "lambda0_target": lam0, "p0_target": sigma(lam0), "H_pre_target_over_ln2": float(h2(sigma(lam0))),
                    "lambda_need": lam0 + LN3}
    return lam0, lam0 + LN3


def step4_beta(st2, S, lam_need, rep):
    mu, q = st2["mu_hat_worst_density"], st2["q_worst"]
    bound = lam_need / ((1 - q) * S * mu * 1.0)
    ok = [b for b in BETA_GRID if b >= bound]
    out = {"mu_hat_density_worst_cell": mu, "q_worst": q, "beta_lower_bound": bound,
           "reference_bound_with_mean_w0": lam_need / ((1 - q) * S * st2["mu_hat_worst_mean_w0"]),
           "grid": list(BETA_GRID)}
    if not ok:
        out["verdict"] = "격자 상한에서도 부족"
        rep["step4"] = out
        raise Stop(f"β 하한 {bound:.2f}가 격자 상한 {BETA_GRID[-1]}를 넘는다: 18.4 a가 필요하다")
    beta = min(ok)
    out["beta"] = beta
    out["binding"] = bool(beta > BETA_GRID[0])        # 하한이 격자 최솟값(1)보다 β를 끌어올렸나
    rep["step4"] = out
    return float(beta)


def step5_g(lam0, beta, S, rep):
    g = lam0 / (beta * M_ODOR * S)
    rep["step5"] = {"g": g, "lambda0_check": beta * g * M_ODOR * S}
    return g


def entropy_model_s(lam0, n_trials=50, target=H_EARLY_MIN):
    """초기 50시행 평균 H/ln2 ≥ target을 지키는 s의 최댓값. 모형: 네 셀을 블록으로 돌고, 셀을 방문할 때마다 정답 로짓이
    s씩 선형으로 오른다 (상한 모형). 본능은 z = 0 셀에서 정답 쪽, z = 1 셀에서 오답 쪽으로 Λ0."""
    start = [lam0, lam0, -lam0, -lam0]

    def avg(s):
        vis = [0, 0, 0, 0]
        hs = []
        for t in range(n_trials):
            c = t % 4
            hs.append(float(h2(sigma(start[c] + s * vis[c]))))
            vis[c] += 1
        return float(np.mean(hs))

    lo, hi = 0.0, 5.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if avg(mid) >= target else (lo, mid)
    return lo, avg


def measure_s(circuit, kappa, beta, eta, odor, z, g=0.0):
    """η, g에서 같은 입력을 두 시행(보상 +1을 줌) 돌린 뒤, 학습된 가중치로 같은 입력의 로짓 ln(π_a1/π_다른쪽)을 잰다.
    g = 0이면 학습 전 π = 0.5라 s의 정의 (π = 0.5, |R - R̄| = 1)와 같다. 반환 (시드별 s, 클리핑 있나)."""
    b = make_brain(circuit, kappa=kappa, beta=beta, g=g, eta=eta)
    runner.start(b)
    a1 = run_trial(b, odor, z, reward=1.0)
    run_trial(b, odor, z)                      # 둘째 시행 첫 결정의 확정에서 학습한다
    pi, _ = probe_pi(b, odor, z)
    other = np.where(a1 == LEFT, RIGHT, LEFT)
    n = np.arange(b.B)
    s = np.log(pi[n, a1]) - np.log(pi[n, other])
    clipped = bool(((b.dw <= -circuit.w0 + 1e-12) & circuit.M).any())
    return s, clipped


def step6_eta(circuit, kappa, beta, g, lam0, lam_need, S, st2, rep):
    q = st2["q_worst"]
    s_lo = lam_need / ((1 - q) * N_CELL / 4)
    s_h, _ = entropy_model_s(lam0)
    s_hi = min(lam_need / 10, s_h)
    out = {"s_lo": s_lo, "s_hi_lambda_need_over_10": lam_need / 10, "s_hi_entropy_model": s_h, "s_hi": s_hi,
           "note_entropy_model": "초기 50시행 평균 H/ln2 >= 0.85를 지키는 최대 s (셀 방문마다 정답 로짓이 s씩 선형 상승, 상한 모형). 18.2의 '약 0.08'을 이 정의로 계산했다"}
    if s_lo > s_hi:
        rep["step6"] = out
        raise Stop(f"η 범위가 비었다: s_lo {s_lo:.4f} > s_hi {s_hi:.4f}")
    s_target = math.sqrt(s_lo * s_hi)
    s1 = {}
    for name, odor, z in cells():
        v, clipped = measure_s(circuit, kappa, beta, 1.0, odor, z)
        s1[name] = {"s1_per_seed": v.tolist(), "s1_mean": float(v.mean()), "clipped": clipped}
    means = np.array([c["s1_mean"] for c in s1.values()])
    if (means <= 0).any():
        rep["step6"] = {**out, "s1": s1}
        raise Stop("s1이 0 이하인 셀이 있다: 학습 방향이 맞지 않는다")
    spread = float(means.max() / means.min())
    out["s1"] = s1
    out["s1_spread_max_over_min"] = spread
    if spread > CELL_S1_SPREAD_MAX:
        rep["step6"] = out
        raise Stop(f"셀 사이 s1의 최대/최소 {spread:.2f}가 3배를 넘는다: η 하나로 맞출 수 없다")
    s1_geo = float(np.exp(np.log(means).mean()))
    eta = s_target / s1_geo
    out.update({"s_target": s_target, "s1_geomean": s1_geo, "eta": eta})
    grid = {}
    for f in ETA_FACTORS:
        row = {}
        for name, odor, z in cells():
            v, clipped = measure_s(circuit, kappa, beta, eta * f, odor, z)
            row[name] = {"s_mean": float(v.mean()), "clipped": clipped}
        grid[str(f)] = row
    out["eta_grid"] = grid
    out["clipping_in_grid"] = any(c["clipped"] for r in grid.values() for c in r.values())
    out["linearity_check"] = {str(f): [grid[str(f)][nm]["s_mean"] / (eta * f * s1[nm]["s1_mean"]) for nm, _, _ in cells()]
                              for f in ETA_FACTORS}
    # 식으로 예측한 s와 비교 (참고)
    mu = st2["mu_hat_worst_density"]
    out["formula_s_per_eta_worst_cell"] = beta ** 2 * S ** 2 * mu / (st2["k_eff_min"] * max(st2["bundle_sizes"][LEFT], 1))
    out["formula_note"] = "s ~ eta beta^2 S^2 mu_hat / (k_eff |a|), |a| = 좌 묶음 크기. 측정값과의 비는 참고"
    rep["step6"] = out
    return eta


def step7_alpha(rep):
    rows = []
    for a in ALPHA_GRID:
        noise = math.sqrt(a / (2 - a))
        rows.append({"alpha": a, "noise": noise, "time_const": 1 / a,
                     "ok": noise <= ALPHA_NOISE_MAX and 1 / a <= ALPHA_TIME_MAX + 1e-9})
    ok = [r["alpha"] for r in rows if r["ok"]]
    if not ok:
        raise Stop("α 격자에 조건을 만족하는 값이 없다")
    alpha = max(ok) if False else min(ok, key=lambda a: abs(a - 0.01))   # 두 조건을 만족하는 값 (0.01)
    rep["step7"] = {"grid": rows, "satisfying": ok, "alpha": alpha}
    return alpha


def entropy_pre(circuit, kappa, beta, g, rep_key, rep):
    b = make_brain(circuit, kappa=kappa, beta=beta, g=g, eta=0.0)
    runner.start(b)
    out = {}
    for name, odor, z in cells():
        pi, _ = probe_pi(b, odor, z)
        pm = np.maximum(pi[:, LEFT], pi[:, RIGHT])
        out[name] = {"p_instinct_side": float(pm.mean()), "H_over_ln2": float(h2(pm).mean())}
    rep[rep_key] = out
    return out


def step8_verify(circuit, kappa, beta, g, eta, alpha, rep, check=True, key="step8"):
    pre = entropy_pre(circuit, kappa, beta, g, "step8_pre", rep)
    if min(c["H_over_ln2"] for c in pre.values()) < H_PRE_MIN:
        raise Stop("학습 전 H/ln2가 0.90 미만이다")
    env = TMaze(TASKS, "C", SEEDS)
    b = make_brain(circuit, kappa=kappa, beta=beta, g=g, eta=eta, alpha_rbar=alpha)
    runner.start(b)
    B = b.B
    alive = np.ones(B, dtype=bool)
    active = np.ones((B, F), dtype=bool)
    Hs = []
    union = np.zeros((B, b.n_kc), dtype=bool)
    snap = None
    n_done = 0
    while n_done < 50:
        o = env.obs()
        act = b.propose(o, alive)
        if o.decide.any():
            pi = b.last_pi[:, 0]
            Hs.append(h2(np.maximum(pi[:, LEFT], pi[:, RIGHT])))     # 엔트로피만 기록 (정답 여부는 읽지 않는다)
            if n_done < 20:
                union |= b.last_kc[:, 0]
        b.commit(o.decide & (act >= 0), active)
        before = env.trials_done
        b.give_reward(env.step(act))
        if env.trials_done > before:
            n_done += 1
            if n_done == 20:
                snap = runner.clone_brain(b)
                snap.flush_learning(np.ones(B, dtype=bool))
    H = np.stack(Hs)                                    # (50, B)
    w0, dw = circuit.w0, snap.dw[:, 0]
    local, glob = [], []
    for s in range(B):
        r = union[s]
        local.append(float(np.linalg.norm(dw[s][r]) / np.linalg.norm((w0 + dw[s])[r])))
        glob.append(float(np.linalg.norm(dw[s]) / np.linalg.norm(w0 + dw[s])))
    out = {"H_over_ln2_mean_per_seed": H.mean(0).tolist(), "H_over_ln2_mean": float(H.mean()),
           "H_over_ln2_min_seed": float(H.mean(0).min()),
           "local_ratio_20_per_seed": local, "global_ratio_20_per_seed": glob,
           "union_kc_rows_per_seed": union.sum(1).tolist(), "trials": 50,
           "note": "과제 C 앞 50시행. 엔트로피와 가중치 변화량만 기록했다 (정답 여부는 읽지 않음)."}
    rep[key] = out
    if not check:
        return out
    bad = []
    if out["H_over_ln2_mean"] < H_EARLY_MIN:
        bad.append(f"초기 50시행 평균 H/ln2 {out['H_over_ln2_mean']:.3f} < {H_EARLY_MIN}")
    if not all(LOCAL_RATIO[0] <= r <= LOCAL_RATIO[1] for r in local):
        bad.append(f"20시행 로컬 비 {np.round(local, 4).tolist()}가 {LOCAL_RATIO} 밖")
    if bad:
        raise Stop("; ".join(bad))
    return out


def whatif(etas) -> int:
    """참고용: 같은 κ, β, g, α에서 η만 바꿔 앞 50시행의 H와 20시행 비를 본다 (규칙의 선택이 아니다, 정답률은 읽지 않는다)."""
    rep = json.loads((OUT / "measure_v1.json").read_text(encoding="utf-8"))
    cfg = load_brain_config(BRAIN)
    circuit = FlyBrain(cfg, 1, [0]).circuit
    ch = {"kappa": rep["step1"]["kappa"], "beta": rep["step4"]["beta"], "g": rep["step5"]["g"]}
    out = {}
    for eta in etas:
        r: dict = {}
        v = step8_verify(circuit, ch["kappa"], ch["beta"], ch["g"], float(eta), 0.01, r, check=False, key="x")
        out[str(eta)] = {"H_over_ln2_mean": v["H_over_ln2_mean"], "H_min_seed": v["H_over_ln2_min_seed"],
                         "local_ratio_20": v["local_ratio_20_per_seed"], "global_ratio_20": v["global_ratio_20_per_seed"]}
    (OUT / "whatif_eta.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in out.items():
        print(k, round(v["H_over_ln2_mean"], 3), round(v["H_min_seed"], 3), np.round(v["local_ratio_20"], 4).tolist())
    return 0


def confirm() -> int:
    """18.2의 확인 격자 ×{¼, ½, 1, 2, 4}를 두 조건으로 끝까지 적용한다 (DECISIONS 18.7).
    조건: s = η·s1 ∈ [s_lo, s_hi], 5시드 모두 20시행 로컬 비 ∈ [0.015, 0.3], 앞 50시행 평균 H/ln2 ≥ 0.85.
    η 기준값 = 첫 측정(measure_v1.json)의 s_target / s1. g는 소수 넷째 자리까지 반올림한 값을 쓴다 (설정에 적는 값과 같다)."""
    rep = json.loads((OUT / "measure_v1.json").read_text(encoding="utf-8"))
    cfg = load_brain_config(BRAIN)
    circuit = FlyBrain(cfg, 1, [0]).circuit
    kappa, beta, alpha = rep["step1"]["kappa"], rep["step4"]["beta"], rep["step7"]["alpha"]
    g = round(rep["step5"]["g"], 5)
    s6 = rep["step6"]
    eta0 = s6["eta"]
    s1 = s6["s1_geomean"]
    rows = []
    for f in ETA_FACTORS:
        eta = round(eta0 * f, 3)
        r: dict = {}
        v = step8_verify(circuit, kappa, beta, g, eta, alpha, r, check=False, key="x")
        s_val = eta * s1
        ok_s = s6["s_lo"] <= s_val <= s6["s_hi"]
        ok_ratio = all(LOCAL_RATIO[0] <= x <= LOCAL_RATIO[1] for x in v["local_ratio_20_per_seed"])
        ok_h = v["H_over_ln2_mean"] >= H_EARLY_MIN
        rows.append({"factor": f, "eta": eta, "s": s_val, "ok_s": bool(ok_s), "ok_ratio": bool(ok_ratio),
                     "ok_H": bool(ok_h), "ok": bool(ok_s and ok_ratio and ok_h), "step8": v})
    good = [r for r in rows if r["ok"]]
    out = {"eta_base": eta0, "g_used": g, "rows": rows, "satisfying_factors": [r["factor"] for r in good]}
    if len(good) == 1:
        c = good[0]
        out["chosen"] = {"kappa": kappa, "g": g, "beta": beta, "eta": c["eta"], "alpha_rbar": alpha, "gamma": 0.9}
        out["status"] = "ok"
        code = 0
    else:
        out["status"] = "stopped"
        out["stop_reason"] = f"두 조건을 모두 만족하는 격자값이 {len(good)}개다 (하나여야 한다)"
        code = 2
    (OUT / "confirm_v1.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for r in rows:
        v = r["step8"]
        print(f"x{r['factor']:<5} eta {r['eta']:<7} s {r['s']:.4f} s조건 {r['ok_s']} | H평균 {v['H_over_ln2_mean']:.3f} 최소시드 "
              f"{v['H_over_ln2_min_seed']:.3f} H {r['ok_H']} | 로컬비 {np.round(v['local_ratio_20_per_seed'], 4).tolist()} {r['ok_ratio']} | 통과 {r['ok']}")
    print(out["status"], out.get("chosen"), out.get("stop_reason"))
    return code


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rep: dict = {"seeds": SEEDS, "L": L, "m": M_ODOR, "status": "running"}
    cfg = load_brain_config(BRAIN)
    circuit = FlyBrain(cfg, 1, [0]).circuit
    rep["circuit"] = {"assign_seed": circuit.seed, "source_hash": circuit.source_hash}
    try:
        kappa, S = step1_kappa(rep)
        st2 = step2_stats(circuit, kappa, rep)
        lam0, lam_need = step3_lambda0(rep)
        beta = step4_beta(st2, S, lam_need, rep)
        g = step5_g(lam0, beta, S, rep)
        eta = step6_eta(circuit, kappa, beta, g, lam0, lam_need, S, st2, rep)
        alpha = step7_alpha(rep)
        step8_verify(circuit, kappa, beta, g, eta, alpha, rep)
        rep["chosen"] = {"kappa": kappa, "g": g, "beta": beta, "eta": eta, "alpha_rbar": alpha, "gamma": 0.9}
        rep["alternatives_a_to_d_used"] = "쓰지 않음"
        rep["status"] = "ok"
        code = 0
    except Stop as e:
        rep["status"] = "stopped"
        rep["stop_reason"] = str(e)
        code = 2
    (OUT / "measure_v1.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1)[:12000])
    return code


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "confirm":
        sys.exit(confirm())
    if len(sys.argv) > 1 and sys.argv[1] == "whatif":
        sys.exit(whatif([float(x) for x in sys.argv[2:]]))
    sys.exit(main())
