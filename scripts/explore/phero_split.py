"""측정 0: 누구의 페로몬을 피하는가 (DECISIONS 41.4, 설계한 초파리, 학습 없음, 실행 전 고정. 한 번만 돌린다).

40.6에서 페로몬을 피하는 초파리(spread10)의 초파리 사이 거리는 줄었다. 피한 것이 동료의 페로몬인지 자기 흔적인지
가른다. 페로몬 장을 초파리별로 나눠 풀고(41.3), 뇌가 보는 페로몬을 전체(total) · 동료 것(others) · 자기 것(self)으로 바꾼다.

  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --extra pacman python scripts/explore/phero_split.py

틀은 spread_confirm.py와 같다 (평가 모드, 조건 regime3, 결과 파일이 있으면 멈춤). 다른 점:
- 뇌는 configs/brain_kph_v1.yaml (k와 k_ph가 뇌 안의 본능 항이다. propose를 감싸지 않는다). build_runner 뒤에
  runner.ghosts.k_gain, runner.ghosts.kph_gain을 조건의 값으로 채운다. 평가 모드라 η_k = η_ph = 0, Δw = 0.
- 방출은 phero.emission: marker (39.3의 위치 표지), phero.view는 조건의 값.

조건 (모두 k = 10): solo (k_ph = 0, total), total10 (−10, total), others10 (−10, others), self10 (−10, self).
평가 시드 9800 ~ 9809, 칸마다 200판, 재생 판 1 ~ 5. 팩맨은 본 실험과 같다 (best.pt, ε = 0, cuda).
판정 (각각 초파리 승률 W의 짝 차이, t 95% 신뢰구간, 자유도 9, 0을 제외하고 양수):
  (1) others10 − solo: "동료의 페로몬만 피해도 더 자주 이긴다."
  (2) self10 − solo: "자기 흔적만 피해도 더 자주 이긴다."
total10 − solo는 39.4의 재현으로 보고만 한다. 점수 · 포획 · 먹힘의 짝 차이도 보고한다. 다중 비교 보정은 하지 않는다.
뒤의 선택 규칙: (1)이 충족이면 41.5 ~ 41.7은 view: others, 아니면 total.

COOP_TEST=1이면 시드 0 · 1, 3판, 규칙 팩맨 (시험용). COOP_ROOT로 결과 폴더의 뿌리를 바꾼다 (기본 runs).
"""
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
import final_eval as fe  # noqa: E402
from flypac.core.config import load_config  # noqa: E402
from flypac.core.runner import build_runner  # noqa: E402
from flypac.recording.recorder import Recorder, make_run_dir  # noqa: E402

TEST = os.environ.get("COOP_TEST") == "1"
SEEDS = [0, 1] if TEST else list(range(9800, 9810))
N = 3 if TEST else 200
BRAIN = "configs/brain_kph_v1.yaml"
K = 10.0
CONDS = {"solo": (0.0, "total"), "total10": (-10.0, "total"), "others10": (-10.0, "others"), "self10": (-10.0, "self")}
ROOT = Path(os.environ.get("COOP_ROOT", "runs")); OUT = ROOT / "phero_split_v1"
if (OUT / "results.json").exists():
    sys.exit(f"멈춤: {OUT / 'results.json'}가 이미 있다")
if not TEST:
    fe.check_threads()
base = load_config(REPO / "configs" / "final_eval_v1.yaml").to_dict()
base.update(seeds=SEEDS, n_games=N, experiment="phero_split_v1")
base["run"] = {"max_loop_ticks": N * 20001}
base["replay"] = {"keep_games": [1, 2, 3, 4, 5], "brain": True}
base["final_eval"]["ghosts"]["flybrain"] = {"kind": "flybrain", "config": BRAIN}
base["phero"]["emission"] = "marker"
if TEST:
    base["pacman"] = {"kind": "rule_based"}; base["rule_based"] = {"danger_dist": 6, "chase_dist": 8}

OUT.mkdir(parents=True, exist_ok=True)
res = {"seeds": SEEDS, "n_games": N, "brain": BRAIN, "k": K, "emission": "marker", "conditions": {}}
for name, (kph, view) in CONDS.items():
    d = copy.deepcopy(base); d["experiment"] = f"phero_split_v1_{name}"
    d["phero"]["view"] = view
    cfg = fe.eval_config(d, "regime3")
    run_dir = make_run_dir(cfg, ROOT)
    runner = build_runner(cfg, Recorder(run_dir, cfg))
    g = runner.ghosts
    g.k_gain[:] = K
    g.kph_gain[:] = kph
    assert g.eta == 0.0 and g.eta_k == 0.0 and g.eta_ph == 0.0 and not g.dw.any() and runner.phero is not None
    assert runner.regime.emission_kind == "marker" and runner.phero_view == view
    (run_dir / "coop.json").write_text(json.dumps({"k": K, "k_ph": kph, "view": view}) + "\n", encoding="utf-8")
    runner.run()
    rows = fe.read_games(run_dir / "games.csv")
    m = fe.per_seed_metrics(rows, SEEDS)
    per = lambda col: [float(np.mean([float(r[col]) for r in rows if int(r["시드"]) == s])) for s in SEEDS]  # noqa: E731
    res["conditions"][name] = {"run_dir": run_dir.name, "k": K, "k_ph": kph, "view": view, "score": m["score"], "W": m["W"],
                               "Nc": per("활성 후 포획 수"), "Ne": per("활성 후 먹힘 수"), "ticks": per("틱 수"),
                               "censored": m["censored"]}
    print(f"[{name}] 끝: {run_dir}", flush=True)


def line(c, ref):
    out = []
    for key, nd in (("score", 1), ("W", 3), ("Nc", 3), ("Ne", 3)):
        ci = fe.paired_ci(c[key], ref[key], 0.95)
        out.append(f"{key} {ci['mean']:+.{nd}f} [{ci['lo']:+.{nd}f}, {ci['hi']:+.{nd}f}]{' *' if ci['excludes_zero'] else ''}")
    return " | ".join(out)


C = res["conditions"]
solo = C["solo"]
print(f"\n측정 0: 누구의 페로몬을 피하는가 (평가 시드 {SEEDS[0]} ~ {SEEDS[-1]}, 칸마다 {N}판, k = {K:g}, marker). 95% CI, * = 0 제외")
print(f"  solo      평균: 점수 {np.mean(solo['score']):.1f}, W {np.mean(solo['W']):.3f}, 포획/판 {np.mean(solo['Nc']):.3f}, "
      f"먹힘/판 {np.mean(solo['Ne']):.3f}, 중단 {sum(solo['censored'])}")
print("  --- 조건 − solo ---")
for name in ("total10", "others10", "self10"):
    c = C[name]
    c["vs_solo"] = {key: fe.paired_ci(c[key], solo[key], 0.95) for key in ("score", "W", "Nc", "Ne")}
    print(f"  {name:9s} " + line(c, solo) + f"   (중단 {sum(c['censored'])})")
verdicts = {}
for no, name, text in ((1, "others10", "동료의 페로몬만 피해도 더 자주 이긴다."), (2, "self10", "자기 흔적만 피해도 더 자주 이긴다.")):
    w = C[name]["vs_solo"]["W"]
    ok = bool(w["lo"] > 0)
    verdicts[str(no)] = {"comparison": f"{name} − solo", "W": w, "met": ok, "text": text}
    print(f"  41.4 판정 ({no}) {name} − solo의 W {w['mean']:+.3f} [{w['lo']:+.3f}, {w['hi']:+.3f}] → "
          f"{'충족' if ok else '미충족'}: \"{text}\"")
res["verdicts"] = verdicts
res["next_view"] = "others" if verdicts["1"]["met"] else "total"
print(f"  41.4의 선택 규칙: 41.5 ~ 41.7의 view = {res['next_view']}")
(OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float) + "\n", encoding="utf-8")
print(f"기록: {OUT / 'results.json'}")
