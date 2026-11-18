"""보상 지형: 페로몬을 피하는 세기 k_ph에 따라 승률 W와 보상이 어떻게 변하는가 (2026-11-17, 탐색, 실행 전 고정. 한 번만 돌린다).

위치. 관리자의 방향(2026-11-17): 체제를 비교하지 않고, LLM(Claude)이 결과를 보며 가장 좋은 학습 조건을 찾는다.
그 첫 단계로 학습 없이 지형부터 잰다. 학습 조건을 고르기 전에 "무엇을 올리면 승률이 오르는가"를 안다.

동기 (관문 5, 41.5). k_ph = 0에서 초파리 한 마리가 받는 갱신량은 잡음에 묻혔다 (q = 0: t −0.52, q = 1: t +0.38).
가설: 체제 3의 보상은 팀 합의 1/4이고, 한 마리만 k_ph를 바꾸면 팀 효과의 약 1/4만 생긴다. 그러면 한 마리가 보는 기울기는

    G_fly ≈ (1/16) · dR_team/dk_ph      (R_team = 판당 활성 후 포획 − q · 활성 후 먹힘, 선형 근사)

이다. 이 스크립트는 dR_team/dk_ph를 직접 재서 이 가설이 관문 5의 값과 맞는지 본다.

  저장소 루트에서:
  D=$(ls ~/다운로드/phero_land.py ~/Downloads/phero_land.py 2>/dev/null | head -1); echo $D
  systemd-inhibit --what=sleep:idle env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --extra pacman python - < $D

조건: 평가 모드(학습 없음, Δw = 0), 조건 regime3, 뇌 configs/brain_kph_v1.yaml, k = 10, phero.emission marker,
phero.view others (41.4의 선택), k_ph ∈ {0, −1, −3, −10, −30}. 네 마리 모두 같은 k_ph. k_ph = 0은 측정 0의 solo와 같은 정책.
평가 시드 9820 ~ 9829 (다른 곳에 쓰지 않는다), 칸마다 200판, 재생은 판 1만. 팩맨은 본 실험과 같다 (best.pt, ε = 0, cuda).

결과 전에 고정한 읽기 (판정이 아니라 다음 학습 조건을 고르는 규칙):
  R1. 승률 최적점 k* = 격자에서 W 평균이 가장 큰 k_ph.
  R2. 보상 설계 q_design = {1, 0.5, 0.25, 0} 중, R_q의 격자 최댓점이 k*와 같은 가장 큰 q (원래 보상 q = 1에 가장 가까운 것).
      없으면 q_design = 0으로 두고 "어긋남"을 적는다.
  R3. 한 마리 학습의 가능성: q_design에서 0과 −3 사이의 기울기로 G_fly = (dR_team/dk_ph)/16을 예측하고,
      t = 2에 필요한 판 수 n = (2 σ / |G_fly|)²를 낸다 (σ: 관문 5의 판 하나 잡음, q = 0이면 0.072, q = 1이면 0.139,
      그 사이 q는 직선 보간). n > 3000이면 "독립 학습으로는 3000판에 기대하기 어렵다".
  R4. 관문 5와의 대조 (보고만): q = 0, 1에서 예측한 G_fly와 관측 (q = 0: −0.00085 ± 0.00162, q = 1: +0.00117 ± 0.00310).
다중 비교 보정은 하지 않는다. 이 자료(9820대)는 확인의 근거로 쓰지 않는다.

COOP_TEST=1이면 시드 0 · 1, 3판, 규칙 팩맨 (시험용). COOP_ROOT로 결과 폴더의 뿌리를 바꾼다 (기본 runs).
"""
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path.cwd()
if not (REPO / "configs" / "brain_kph_v1.yaml").exists():
    sys.exit("멈춤: 저장소 루트에서 돌린다 (configs/brain_kph_v1.yaml이 없다)")
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
import final_eval as fe  # noqa: E402
from flypac.core.config import load_config  # noqa: E402
from flypac.core.runner import build_runner  # noqa: E402
from flypac.recording.recorder import Recorder, make_run_dir  # noqa: E402

TEST = os.environ.get("COOP_TEST") == "1"
SEEDS = [0, 1] if TEST else list(range(9820, 9830))
N = 3 if TEST else 200
BRAIN = "configs/brain_kph_v1.yaml"
K = 10.0
GRID = [0.0, -1.0, -3.0, -10.0, -30.0]
QS = [1.0, 0.5, 0.25, 0.0]
SIGMA = {0.0: 0.072, 1.0: 0.139}   # 관문 5의 판 하나 잡음 (시드 sd × √200)
GATE5 = {0.0: (-0.00085, 0.00162), 1.0: (+0.00117, 0.00310)}
ROOT = Path(os.environ.get("COOP_ROOT", "runs")); OUT = ROOT / "phero_land_v1"
if (OUT / "results.json").exists():
    sys.exit(f"멈춤: {OUT / 'results.json'}가 이미 있다")
if not TEST:
    fe.check_threads()
base = load_config(REPO / "configs" / "final_eval_v1.yaml").to_dict()
base.update(seeds=SEEDS, n_games=N, experiment="phero_land_v1")
base["run"] = {"max_loop_ticks": N * 20001}
base["replay"] = {"keep_games": [1], "brain": True}
base["final_eval"]["ghosts"]["flybrain"] = {"kind": "flybrain", "config": BRAIN}
base["phero"]["emission"] = "marker"
base["phero"]["view"] = "others"
if TEST:
    base["pacman"] = {"kind": "rule_based"}; base["rule_based"] = {"danger_dist": 6, "chase_dist": 8}

OUT.mkdir(parents=True, exist_ok=True)
res = {"seeds": SEEDS, "n_games": N, "brain": BRAIN, "k": K, "emission": "marker", "view": "others", "grid": GRID, "points": {}}
for kph in GRID:
    name = f"kph{kph:g}"
    d = copy.deepcopy(base); d["experiment"] = f"phero_land_v1_{name}"
    cfg = fe.eval_config(d, "regime3")
    run_dir = make_run_dir(cfg, ROOT)
    runner = build_runner(cfg, Recorder(run_dir, cfg))
    g = runner.ghosts
    g.k_gain[:] = K
    g.kph_gain[:] = kph
    assert g.eta == 0.0 and g.eta_k == 0.0 and g.eta_ph == 0.0 and not g.dw.any()
    assert runner.regime.emission_kind == "marker" and runner.phero_view == "others"
    (run_dir / "coop.json").write_text(json.dumps({"k": K, "k_ph": kph, "view": "others"}) + "\n", encoding="utf-8")
    runner.run()
    rows = fe.read_games(run_dir / "games.csv")
    m = fe.per_seed_metrics(rows, SEEDS)
    per = lambda col: [float(np.mean([float(r[col]) for r in rows if int(r["시드"]) == s])) for s in SEEDS]  # noqa: E731
    res["points"][name] = {"run_dir": run_dir.name, "k_ph": kph, "score": m["score"], "W": m["W"],
                           "Nc": per("활성 후 포획 수"), "Ne": per("활성 후 먹힘 수"), "censored": m["censored"]}
    print(f"[{name}] 끝: {run_dir}", flush=True)

P = res["points"]
names = [f"kph{k:g}" for k in GRID]
mean = lambda v: float(np.mean(v))  # noqa: E731
Rq = lambda p, q: [c - q * e for c, e in zip(p["Nc"], p["Ne"])]  # noqa: E731

print(f"\n보상 지형 (평가 시드 {SEEDS[0]} ~ {SEEDS[-1]}, 칸마다 {N}판, k = {K:g}, marker, view others). 95% CI, * = 0 제외")
print("  k_ph     W      점수     포획/판  먹힘/판  " + "  ".join(f"R(q={q:g})" for q in QS) + "   W − (k_ph = 0)")
ref = P["kph0"]
for nm in names:
    p = P[nm]
    w = fe.paired_ci(p["W"], ref["W"], 0.95) if nm != "kph0" else None
    wtxt = "" if w is None else f"{w['mean']:+.3f} [{w['lo']:+.3f}, {w['hi']:+.3f}]{' *' if w['excludes_zero'] else ''}"
    print(f"  {p['k_ph']:+5g}  {mean(p['W']):.3f}  {mean(p['score']):7.1f}  {mean(p['Nc']):.3f}    {mean(p['Ne']):.3f}    "
          + "  ".join(f"{mean(Rq(p, q)):+.3f}  " for q in QS) + f"   {wtxt}   (중단 {sum(p['censored'])})")

# R1
kstar = max(GRID, key=lambda k: mean(P[f"kph{k:g}"]["W"]))
# R2
qdesign, aligned = None, {}
for q in QS:
    kq = max(GRID, key=lambda k: mean(Rq(P[f"kph{k:g}"], q)))
    aligned[f"{q:g}"] = kq
    if qdesign is None and kq == kstar:
        qdesign = q
mismatch = qdesign is None
if mismatch:
    qdesign = 0.0
print(f"\n  R1. 승률 최적점 k* = {kstar:g}")
print("  R2. q별 보상 최댓점: " + ", ".join(f"q = {q}: {k:g}" for q, k in aligned.items())
      + f" → q_design = {qdesign:g}" + ("  (어긋남: 어느 q도 k*와 맞지 않음)" if mismatch else ""))


def slope_fly(q):
    """0과 −3 사이의 팀 보상 기울기 dR/dk_ph (시드 짝 차이 ÷ Δk)와 그 1/16."""
    ci = fe.paired_ci(Rq(P["kph-3"], q), Rq(P["kph0"], q), 0.95)
    s = {k: ci[k] / -3.0 for k in ("mean", "lo", "hi")}
    lo, hi = sorted((s["lo"], s["hi"]))
    return {"team_slope": s["mean"], "team_slope_ci": [lo, hi], "G_fly": s["mean"] / 16, "G_fly_ci": [lo / 16, hi / 16]}


def sigma(q):
    return SIGMA[0.0] + (SIGMA[1.0] - SIGMA[0.0]) * q


res["read"] = {"kstar": kstar, "aligned": aligned, "q_design": qdesign, "mismatch": mismatch, "slopes": {}}
for q in sorted({qdesign, 0.0, 1.0}):
    s = slope_fly(q)
    n = (2 * sigma(q) / abs(s["G_fly"])) ** 2 if s["G_fly"] != 0 else float("inf")
    s["sigma"], s["n_games_t2"] = sigma(q), n
    res["read"]["slopes"][f"{q:g}"] = s
    tag = "  ← q_design" if q == qdesign else ""
    print(f"  R3. q = {q:g}: dR_team/dk_ph = {s['team_slope']:+.4f} [{s['team_slope_ci'][0]:+.4f}, {s['team_slope_ci'][1]:+.4f}], "
          f"예측 G_fly = {s['G_fly']:+.5f}, σ {sigma(q):.3f} → t = 2에 필요한 판 수 ≈ {n:,.0f}"
          + (" (> 3000: 독립 학습으로는 기대하기 어렵다)" if n > 3000 else " (≤ 3000)") + tag)
for q in (0.0, 1.0):
    s = res["read"]["slopes"][f"{q:g}"]
    o, se = GATE5[q]
    print(f"  R4. q = {q:g}: 예측 G_fly {s['G_fly']:+.5f} [{s['G_fly_ci'][0]:+.5f}, {s['G_fly_ci'][1]:+.5f}] "
          f"대 관문 5 관측 {o:+.5f} ± {se:.5f} (표준오차)")
(OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float) + "\n", encoding="utf-8")
print(f"기록: {OUT / 'results.json'}")
