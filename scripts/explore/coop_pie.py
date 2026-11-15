"""협동의 몫 측정 (사후 탐색, 기준은 실행 전에 여기 적는다): 동료의 페로몬에 끌리거나 피하는 본능을 더하면 사냥이 나아지는가.

  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --extra pacman python - < coop.py   (저장소 루트에서)

모든 조건은 학습 없는 뇌다 (조건 regime3의 평가 모드: η = 0, Δw = 0, 페로몬 방출은 D§29 그대로). Δw = 0이면 MBON 항이 0이라
결정은 본능 항 g·odor만으로 정해진다. 뇌에 넘기는 냄새 관측을  k·odor + k_ph·phero  로 바꾸면 본능 항이
    g (k·o_a + k_ph·φ_a)        (φ: 네 방향의 페로몬 관측)
가 된다. k_ph > 0이면 동료의 페로몬 쪽으로(중계를 따라 모임), k_ph < 0이면 반대쪽으로(흩어짐). 기록 · 재생은 원래 관측을 받는다.

조건: b0 (k = 1, k_ph = 0), solo (k = 10, k_ph = 0: 혼자 냄새만 따르는 설계의 기준),
      k = 10에 k_ph = −10, −3, +3, +10.
평가 시드 9500 ~ 9509 (다른 곳에 쓰지 않는다), 칸마다 100판, 팩맨은 본 실험과 같다 (best.pt, ε = 0, cuda).
결과 전 고정 기준: (조건 − solo)의 팩맨 점수 짝 차이(평가 시드 10쌍, t 95% CI)가 0을 제외하고 음수이면
"페로몬 본능이 혼자 냄새를 따르는 것보다 낫다 (협동의 몫이 있다)". 포획 · 먹힘 · 승률 W는 함께 보고한다. 다중 비교 보정은 하지 않는다.
"""
import sys, json, copy, dataclasses, os
from pathlib import Path
import numpy as np
sys.path[:0] = ["src", "scripts"]
import final_eval as fe
from flypac.core.config import load_config
from flypac.core.runner import build_runner
from flypac.recording.recorder import Recorder, make_run_dir

TEST = os.environ.get("COOP_TEST") == "1"
SEEDS = [0, 1] if TEST else list(range(9500, 9510))
N = 3 if TEST else 100
CONDS = {"b0": (1.0, 0.0), "solo": (10.0, 0.0), "away10": (10.0, -10.0), "away3": (10.0, -3.0),
         "toward3": (10.0, 3.0), "toward10": (10.0, 10.0)}
ROOT = Path(os.environ.get("COOP_ROOT", "runs")); OUT = ROOT / "coop_v1"
if (OUT / "results.json").exists():
    sys.exit(f"멈춤: {OUT / 'results.json'}가 이미 있다")
if not TEST:
    fe.check_threads()
base = load_config("configs/final_eval_v1.yaml").to_dict()
base.update(seeds=SEEDS, n_games=N, experiment="coop_v1")
base["run"] = {"max_loop_ticks": N * 20001}
base["replay"] = {"keep_games": [1, 2, 3, 4, 5], "brain": True}
if TEST:
    base["pacman"] = {"kind": "rule_based"}; base["rule_based"] = {"danger_dist": 6, "chase_dist": 8}

def wrap(brain, k, kph):
    orig = brain.propose
    def propose(obs, alive, views=None):
        mix = k * np.asarray(obs.odor, float) + kph * np.asarray(obs.phero, float)
        return orig(dataclasses.replace(obs, odor=mix), alive, views)
    brain.propose = propose

OUT.mkdir(parents=True, exist_ok=True)
res = {"seeds": SEEDS, "n_games": N, "conditions": {}}
for name, (k, kph) in CONDS.items():
    d = copy.deepcopy(base); d["experiment"] = f"coop_v1_{name}"
    cfg = fe.eval_config(d, "regime3")
    run_dir = make_run_dir(cfg, ROOT)
    runner = build_runner(cfg, Recorder(run_dir, cfg))
    assert runner.ghosts.eta == 0.0 and not runner.ghosts.dw.any() and runner.phero is not None
    wrap(runner.ghosts, k, kph)
    (run_dir / "coop.json").write_text(json.dumps({"k": k, "k_ph": kph}) + "\n", encoding="utf-8")
    runner.run()
    rows = fe.read_games(run_dir / "games.csv")
    m = fe.per_seed_metrics(rows, SEEDS)
    per = lambda col: [float(np.mean([float(r[col]) for r in rows if int(r["시드"]) == s])) for s in SEEDS]
    res["conditions"][name] = {"run_dir": run_dir.name, "k": k, "k_ph": kph, "score": m["score"], "W": m["W"],
                               "Nc": per("활성 후 포획 수"), "Ne": per("활성 후 먹힘 수"), "ticks": per("틱 수"), "censored": m["censored"]}
    print(f"[{name}] 끝: {run_dir}", flush=True)

def line(c, ref):
    out = []
    for key, nd in (("score", 1), ("W", 3), ("Nc", 3), ("Ne", 3)):
        ci = fe.paired_ci(c[key], ref[key], 0.95)
        out.append(f"{key} {ci['mean']:+.{nd}f} [{ci['lo']:+.{nd}f}, {ci['hi']:+.{nd}f}]{' *' if ci['excludes_zero'] else ''}")
    return " | ".join(out), fe.paired_ci(c["score"], ref["score"], 0.95)

C = res["conditions"]
print(f"\n협동의 몫 (평가 시드 {SEEDS[0]} ~ {SEEDS[-1]}, 칸마다 {N}판). 95% CI, * = 0 제외")
for name in ("b0", "solo"):
    c = C[name]; print(f"  {name:9s} 평균: 점수 {np.mean(c['score']):.1f}, W {np.mean(c['W']):.3f}, 포획/판 {np.mean(c['Nc']):.3f}, 먹힘/판 {np.mean(c['Ne']):.3f}, 중단 {sum(c['censored'])}")
print("  [solo − b0]  " + line(C["solo"], C["b0"])[0])
print("  --- 조건 − solo (기준 비교) ---")
for name, c in C.items():
    if name in ("b0", "solo"): continue
    txt, s = line(c, C["solo"]); c["vs_solo"] = s; c["coop"] = bool(s["hi"] < 0)
    print(f"  {name:9s} " + txt + f"   → 기준 충족: {'예' if c['coop'] else '아니오'}")
(OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float) + "\n", encoding="utf-8")
print(f"기록: {OUT / 'results.json'}")
