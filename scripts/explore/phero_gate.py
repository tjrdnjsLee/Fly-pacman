"""관문 5: 보상은 k_ph를 어느 쪽으로 미는가 (DECISIONS 41.5, 학습 없음, 실행 전 고정. 한 번만 돌린다).

학습 모드(평가 아님)에서 k = 10, k_ph = 0을 고정하고(헛돌기 dry_run), 41.3의 규칙이 냈을 갱신량
G_ph = Σ (R_n − R̄) e_ph를 초파리 · 판마다 센다 (38.2 관문 4와 같은 양, η_ph를 곱하지 않은 값).

  env PH_VIEW=others OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
      uv run --extra pacman python scripts/explore/phero_gate.py

- 조건 regime3, 학습 모드. 뇌는 configs/brain_kph_v1.yaml에 phero_gain.learn: true, dry_run: true, eta_ph: 1.0을
  덮어쓴 임시 설정 (결과 폴더에 `brain_kph_gate.yaml`로 쓴다). 시냅스는 얼렸고(Δw = 0) η_k = 0이라 k = 10에 머물고,
  헛돌기라 k_ph는 0에 머문다. 정책은 판 내내 solo와 같다.
- 방출은 phero.emission: marker. phero.view는 환경변수 PH_VIEW (total 또는 others, 없으면 멈춤. 41.4의 선택 규칙).
- 보상은 체제 3의 식에서 reward: {q: 1.0}(본 실험)과 reward: {q: 0.0}(먹혀도 고통 없음) 두 번.
- 시드 9810 ~ 9819, 칸마다 200판. 가중치 · 재생은 저장하지 않는다.
- 출력: runs/phero_gate_v1/results.json. q마다 시드별 Ḡ_ph (초파리 4마리 · 판 수 평균), 시드 10개의 평균 · 표준오차 · t.
- 미리 적은 예측 (41.2): q = 1에서 Ḡ_ph ≥ 0 (피하지 않는 쪽), q = 0에서 Ḡ_ph < 0 (피하는 쪽). 틀려도 그대로 기록한다.
- η_ph의 규칙 (41.5): η_ph = 10 / (|Ḡ_ph(q = 0)| × 3000). q = 0의 |t|가 1에 못 미치면 이 식을 쓰지 않는다.

COOP_TEST=1이면 시드 0 · 1, 3판, 규칙 팩맨 (시험용). COOP_ROOT로 결과 폴더의 뿌리를 바꾼다 (기본 runs).
"""
import copy
import json
import os
import statistics
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
import final_eval as fe  # noqa: E402
from flypac.core.config import Config, load_config  # noqa: E402
from flypac.core.runner import build_runner  # noqa: E402
from flypac.recording.recorder import Recorder, make_run_dir  # noqa: E402

TEST = os.environ.get("COOP_TEST") == "1"
SEEDS = [0, 1] if TEST else list(range(9810, 9820))
N = 3 if TEST else 200
BRAIN = "configs/brain_kph_v1.yaml"
QS = (1.0, 0.0)
ETA_PH_TARGET, ETA_PH_GAMES = 10.0, 3000   # 41.5: 3000판의 기대 표류 |Δk_ph| = 10
VIEW = os.environ.get("PH_VIEW")
if VIEW not in ("total", "others"):
    sys.exit(f"멈춤: 환경변수 PH_VIEW를 total 또는 others로 준다 (41.4의 선택 규칙). 지금: {VIEW!r}")
ROOT = Path(os.environ.get("COOP_ROOT", "runs")); OUT = ROOT / "phero_gate_v1"
if (OUT / "results.json").exists():
    sys.exit(f"멈춤: {OUT / 'results.json'}가 이미 있다")
if not TEST:
    fe.check_threads()

OUT.mkdir(parents=True, exist_ok=True)
brain = load_config(REPO / BRAIN).to_dict()
brain["phero_gain"].update(learn=True, dry_run=True, eta_ph=1.0)
brain_path = (OUT / "brain_kph_gate.yaml").resolve()
brain_path.write_text("# phero_gate.py가 만든 임시 뇌 설정: configs/brain_kph_v1.yaml에 phero_gain.learn: true, "
                      "dry_run: true, eta_ph: 1.0을 덮어씀 (DECISIONS 41.5)\n"
                      + yaml.safe_dump(brain, allow_unicode=True, sort_keys=False), encoding="utf-8")

base = load_config(REPO / "configs" / "final_eval_v1.yaml").to_dict()
del base["final_eval"]
base.update(seeds=SEEDS, n_games=N, condition="regime3", evaluate=False,
            ghosts={"kind": "flybrain", "config": str(brain_path)})
base["run"] = {"max_loop_ticks": N * 20001}
base["replay"] = {"keep_games": [], "brain": False}
base["phero"]["emission"] = "marker"
base["phero"]["view"] = VIEW
if TEST:
    base["pacman"] = {"kind": "rule_based"}; base["rule_based"] = {"danger_dist": 6, "chase_dist": 8}


def seed_stats(values: list[float]) -> dict:
    n = len(values)
    mean, sd = statistics.fmean(values), statistics.stdev(values)
    se = sd / n ** 0.5
    return {"n": n, "mean": mean, "sd": sd, "se": se, "t": mean / se if se > 0 else None}


res = {"seeds": SEEDS, "n_games": N, "brain": BRAIN, "brain_overrides": {"learn": True, "dry_run": True, "eta_ph": 1.0},
       "emission": "marker", "view": VIEW, "q": {}}
for q in QS:
    d = copy.deepcopy(base)
    d["experiment"] = f"phero_gate_v1_q{q:g}"
    d["reward"] = {"q": q}
    cfg = Config(d)
    run_dir = make_run_dir(cfg, ROOT)
    runner = build_runner(cfg, Recorder(run_dir, cfg))
    g = runner.ghosts
    assert not runner.ghosts.evaluate and runner.weight_writer is None and runner.regime.q == q
    assert g.eta == 0.0 and g.eta_k == 0.0 and g.ph_learn and g.dry_run and (g.k_gain == 10.0).all() and (g.kph_gain == 0.0).all()
    assert runner.regime.emission_kind == "marker" and runner.phero_view == VIEW
    runner.run()
    assert (g.k_gain == 10.0).all() and (g.kph_gain == 0.0).all() and not g.dw.any()
    games = [sum(1 for r in runner.results if r["slot"] == b) for b in range(len(SEEDS))]
    assert games == [N] * len(SEEDS), games
    per_seed = [float(g.G_ph[b].mean() / N) for b in range(len(SEEDS))]   # 초파리 4마리 · 판 수 평균
    st = seed_stats(per_seed)
    res["q"][f"{q:g}"] = {"run_dir": run_dir.name, "G_ph_by_seed": per_seed, "G_ph_by_seed_fly": (g.G_ph / N).tolist(), **st}
    print(f"[q = {q:g}] 끝: {run_dir}", flush=True)

print(f"\n관문 5: 보상은 k_ph를 어느 쪽으로 미는가 (시드 {SEEDS[0]} ~ {SEEDS[-1]}, 칸마다 {N}판, view = {VIEW}, marker, k = 10, k_ph = 0)")
print("  Ḡ_ph = 초파리 · 판당 Σ (R_n − R̄) e_ph. 양수면 k_ph를 키우는 쪽(피하지 않는 쪽), 음수면 줄이는 쪽(피하는 쪽)")
for q in QS:
    r = res["q"][f"{q:g}"]
    t = "-" if r["t"] is None else f"{r['t']:+.2f}"
    print(f"  q = {q:g}: 평균 {r['mean']:+.5f}, 표준오차 {r['se']:.5f}, t {t}, 표준편차 {r['sd']:.5f}")
    print("         시드별: " + ", ".join(f"{v:+.5f}" for v in r["G_ph_by_seed"]))
p1, p0 = res["q"]["1"]["mean"] >= 0, res["q"]["0"]["mean"] < 0
res["prediction"] = {"q1_nonneg": bool(p1), "q0_neg": bool(p0)}
print(f"  미리 적은 예측 (41.2): q = 1에서 Ḡ_ph ≥ 0 → {'맞음' if p1 else '틀림'}, q = 0에서 Ḡ_ph < 0 → {'맞음' if p0 else '틀림'}")
r0 = res["q"]["0"]
if r0["t"] is not None and abs(r0["t"]) >= 1 and r0["mean"] != 0:
    eta = ETA_PH_TARGET / (abs(r0["mean"]) * ETA_PH_GAMES)
    res["eta_ph_rule"] = {"applies": True, "eta_ph": eta}
    print(f"  η_ph의 규칙 (41.5): |t(q = 0)| ≥ 1 → η_ph = 10 / (|Ḡ_ph(q = 0)| × 3000) = {eta:.6g}")
else:
    res["eta_ph_rule"] = {"applies": False, "eta_ph": None}
    print("  η_ph의 규칙 (41.5): |t(q = 0)| < 1 → 식을 쓰지 않는다. 개발 실행 전에 값을 판단으로 정하고 사유를 적는다")
(OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float) + "\n", encoding="utf-8")
print(f"기록: {OUT / 'results.json'}")
