"""드문 보상 팔: 같은 회로 · 같은 학습 규칙에서 냄새 퍼텐셜 보상(λΔΦ)을 빼면 배우는가 (2026-11-21, 사후 탐색, 실행 전 고정).

위치. 42.14(확인 실행)의 learn 팔은 체제 1 보상 + λΔΦ로 배웠다 (learn − ctrl W +0.154 [+0.091, +0.217]).
42.1의 진단은 v1에서 배우지 못한 원인으로 (1) 냄새는 꼬리, (2) 보상은 포획 때뿐(드묾), (3) 본능이 공을 가로챔을 들었다.
v2는 (3)을 없애고(본능 끔) 동시에 (2)를 바꿨다(λΔΦ). 그래서 "보상의 밀도가 갈랐다"는 지금 v1과 v2를 넘나든 해석이다.
이 팔은 v2에서 λΔΦ **하나만** 뺀다. 나머지(뇌 configs/brain_v2_nolh.yaml, η = 1.106, 본능 끔, 시각 끔, 체제 1, 학습 시드,
평가 시드, 판 수, 팩맨)는 42.12의 learn 팔과 같다. 그래서 learn 팔 · ctrl 팔 · b0_nolh(무작위)와 시드 위치로 짝이 된다.

팔 (SP_SHARD, 두 프로세스로 나눈다. 시드마다 난수 흐름이 따로라 나누어도 같은 실험이다):
  0       학습 시드 6400 ~ 6409, 평가 시드 9730 ~ 9739
  1       학습 시드 6410 ~ 6419, 평가 시드 9740 ~ 9749
  report  판정 (학습 없음). 두 조각과 runs/vision_confirm_v1_learn · _ctrl의 results.json을 읽는다.
공통: 학습 3000판, 가중치 100판마다, 재생 판 1 · 3000. 평가 200판, η = 0. reward.shaping 블록 없음 (λ = 0).

판정 (결과 전 고정, 2026-11-21):
  S1 (주). (learn − sparse)의 W 짝 차이, 평가 시드 20쌍, t 95% 신뢰구간 (자유도 19)이 0 제외 양수
      → "같은 회로 · 같은 규칙에서 냄새 퍼텐셜 보상이 있어야 배웠다 (보상의 밀도가 갈랐다)".
      0을 포함하면 → "드문 보상과 구별하지 못했다 (v1의 실패는 밀도가 아니라 본능 쪽일 수 있다)".
  S2 (부). (sparse − b0_nolh)의 W 짝 차이: 0 제외 양수 → "포획 보상만으로도 무작위보다 나아졌다".
      0 포함 → "포획 보상만으로는 무작위와 구별되지 않았다". 0 제외 음수도 그대로 적는다.
  보고: sparse − ctrl, 점수 · 포획 · 먹힘, 세 팔의 ‖Δw‖/‖w0‖.
미리 적은 예측 (Claude): S1 충족, S2 0 포함. 확신은 낮다: 본능을 끈 것만으로 (3)이 사라졌으므로 드문 보상으로도
  배울 수 있다. 어느 쪽이어도 포스터 문장이 된다. 틀려도 그대로 적는다. 한 번만 돌린다. 다중 비교 보정은 하지 않는다.

  저장소 루트에서 (두 조각을 나란히):
  systemd-inhibit --what=sleep:idle bash -c 'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1;
    SP_SHARD=0 uv run --extra pacman python scripts/explore/v2_sparse.py > ~/v2_sparse_0.log 2>&1 &
    SP_SHARD=1 uv run --extra pacman python scripts/explore/v2_sparse.py > ~/v2_sparse_1.log 2>&1 & wait'
  SP_SHARD=report uv run python scripts/explore/v2_sparse.py

COOP_TEST=1이면 학습 시드 0 · 1(조각마다 하나), 평가 2 · 3, 학습 4판 · 평가 3판, 가중치 2판마다, 규칙 팩맨, 뇌 COOP_BRAIN.
COOP_ROOT로 결과 폴더의 뿌리를 바꾼다 (기본 runs).
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
from flypac.core.config import Config, load_config  # noqa: E402
from flypac.core.runner import build_runner  # noqa: E402
from flypac.recording.recorder import Recorder, make_run_dir  # noqa: E402

SHARD = os.environ.get("SP_SHARD")
if SHARD not in ("0", "1", "report"):
    sys.exit(f"멈춤: SP_SHARD는 0, 1, report 중 하나다. 지금: {SHARD!r}")
TEST = os.environ.get("COOP_TEST") == "1"
TRAIN_ALL = [0, 1] if TEST else list(range(6400, 6420))
EVAL_ALL = [2, 3] if TEST else list(range(9730, 9750))
NT, NE = (4, 3) if TEST else (3000, 200)
EVERY = 2 if TEST else 100
BRAIN = os.environ.get("COOP_BRAIN", "configs/brain_v2_nolh.yaml") if TEST else "configs/brain_v2_nolh.yaml"
COND = "regime1"
ROOT = Path(os.environ.get("COOP_ROOT", REPO / "runs"))
HALF = len(TRAIN_ALL) // 2


def mean(v):
    return float(np.mean(v))


def shard_out(k):
    return ROOT / f"v2_sparse_v1_s{k}"


if SHARD == "report":
    S = []
    for k in (0, 1):
        p = shard_out(k) / "results.json"
        if not p.exists():
            sys.exit(f"멈춤: {p}가 없다 (두 조각이 끝나야 한다)")
        S.append(json.loads(p.read_text(encoding="utf-8")))
    R = {}
    for a in ("learn", "ctrl"):
        p = ROOT / f"vision_confirm_v1_{a}" / "results.json"
        if not p.exists():
            sys.exit(f"멈춤: {p}가 없다 (42.12의 확인 실행 결과)")
        R[a] = json.loads(p.read_text(encoding="utf-8"))
    for a in ("learn", "ctrl"):
        if [int(s) for s in R[a]["train_seeds"]] != TRAIN_ALL or [int(s) for s in R[a]["eval_seeds"]] != EVAL_ALL:
            sys.exit(f"멈춤: {a} 팔의 시드가 이 스크립트와 다르다")
    sp_eval = {"score": [], "W": [], "Nc": [], "Ne": [], "censored": []}
    sp_rel, sp_seeds = [], []
    for s in S:
        for key in sp_eval:
            sp_eval[key] += list(s["eval"]["sparse"][key])
        sp_rel += list(s["dw_rel"]); sp_seeds += list(s["eval_seeds"])
    if sp_seeds != EVAL_ALL:
        sys.exit(f"멈춤: 두 조각의 평가 시드를 이으면 {sp_seeds}다 (기대 {EVAL_ALL})")
    E = {**R["learn"]["eval"], **R["ctrl"]["eval"], "sparse": sp_eval}
    rel = {"learn": R["learn"]["dw_rel"], "ctrl": R["ctrl"]["dw_rel"], "sparse": sp_rel}
    print(f"드문 보상 팔 (λ = 0) · 확인 실행과 짝 (학습 시드 {TRAIN_ALL[0]} ~ {TRAIN_ALL[-1]}, "
          f"평가 시드 {EVAL_ALL[0]} ~ {EVAL_ALL[-1]}, {NE}판)")
    print("  칸        W      점수     포획/판  먹힘/판  중단   ‖Δw‖/‖w0‖ 평균")
    for k in ("b0_nolh", "ctrl", "sparse", "learn"):
        c = E[k]
        dwr = f"{mean(rel[k]):.3f}" if k in rel else ""
        print(f"  {k:<8}  {mean(c['W']):.3f}  {mean(c['score']):7.1f}  {mean(c['Nc']):.3f}    {mean(c['Ne']):.3f}    "
              f"{sum(c['censored'])}     {dwr}")

    def ci(a, b, key, nd=3):
        r = fe.paired_ci(E[a][key], E[b][key], 0.95)
        return r, f"{r['mean']:+.{nd}f} [{r['lo']:+.{nd}f}, {r['hi']:+.{nd}f}]{' *' if r['excludes_zero'] else ''}"

    r1, t1 = ci("learn", "sparse", "W")
    s1 = bool(r1["excludes_zero"] and r1["mean"] > 0)
    print(f"\n  S1. learn − sparse W: {t1} (자유도 {r1['df']}) → "
          + ("충족: 냄새 퍼텐셜 보상이 있어야 배웠다 (보상의 밀도가 갈랐다)" if s1
             else "미충족: 드문 보상과 구별하지 못했다"))
    r2, t2 = ci("sparse", "b0_nolh", "W")
    if r2["excludes_zero"] and r2["mean"] > 0:
        s2 = "포획 보상만으로도 무작위보다 나아졌다"
    elif r2["excludes_zero"]:
        s2 = "포획 보상만으로는 무작위보다 나빠졌다"
    else:
        s2 = "포획 보상만으로는 무작위와 구별되지 않았다"
    print(f"  S2. sparse − b0_nolh W: {t2} → {s2}")
    for a, b in (("learn", "sparse"), ("sparse", "b0_nolh"), ("sparse", "ctrl")):
        print(f"\n  보고: {a} − {b}")
        for key, nd in (("W", 3), ("score", 1), ("Nc", 3), ("Ne", 3)):
            print(f"    {key}: {ci(a, b, key, nd)[1]}")
    out = {"S1": {"W": r1["mean"], "ci": [r1["lo"], r1["hi"]], "met": s1},
           "S2": {"W": r2["mean"], "ci": [r2["lo"], r2["hi"]], "reading": s2},
           "mean_W": {k: mean(E[k]["W"]) for k in ("b0_nolh", "ctrl", "sparse", "learn")},
           "dw_rel_mean": {k: mean(v) for k, v in rel.items()},
           "per_seed": {k: {key: list(map(float, E[k][key])) for key in ("W", "score", "Nc", "Ne")}
                        for k in ("b0_nolh", "ctrl", "sparse", "learn")},
           "eval_seeds": EVAL_ALL, "train_seeds": TRAIN_ALL}
    path = ROOT / "v2_sparse_v1_report.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"기록: {path}")
    sys.exit(0)

k = int(SHARD)
TRAIN = TRAIN_ALL[k * HALF:(k + 1) * HALF]
EVAL = EVAL_ALL[k * HALF:(k + 1) * HALF]
OUT = shard_out(k)
if (OUT / "results.json").exists():
    sys.exit(f"멈춤: {OUT / 'results.json'}가 이미 있다")
if not TEST:
    fe.check_threads()
bcfg = load_config(REPO / BRAIN)
lh = bcfg.get("lateral_horn"); sc = bcfg.get("sign_control")
assert lh is not None and not lh["enabled"], "본능을 끈 뇌여야 한다"
assert not (sc is not None and sc["enabled"]), "부호 대조군 뇌가 아니어야 한다"
OUT.mkdir(parents=True, exist_ok=True)

base = load_config(REPO / "configs" / "final_eval_v1.yaml").to_dict()
if TEST:
    base["pacman"] = {"kind": "rule_based"}; base["rule_based"] = {"danger_dist": 6, "chase_dist": 8}
tr = copy.deepcopy(base)
del tr["final_eval"]
tr.update(experiment=f"v2_sparse_v1_s{k}_train", seeds=TRAIN, n_games=NT, condition=COND, evaluate=False,
          ghosts={"kind": "flybrain", "config": BRAIN})
tr["run"] = {"max_loop_ticks": NT * 20001}
tr["replay"] = {"keep_games": [1, NT], "brain": True}
tr["weights"] = {"every_games": EVERY}
tr.pop("reward", None)   # λ = 0: reward.shaping 블록 없음 (42.5: 없으면 끔). learn 팔과 다른 유일한 곳
cfg = Config(tr)
train_dir = make_run_dir(cfg, ROOT)
runner = build_runner(cfg, Recorder(train_dir, cfg))
g = runner.ghosts
assert not g.evaluate and g.eta > 0 and not g.lateral_horn and g.g == 0.0 and not g.sign_control
assert runner.shaping is None and runner.vision_R == 0, "냄새 퍼텐셜 보상과 시각이 모두 꺼져 있어야 한다"
runner.run()
assert g.dw.any()
states = [g.slot_state(i) for i in range(len(TRAIN))]
W0N = float(np.linalg.norm(g.circuit.w0)) * np.sqrt(g.F)
rel = [float(np.linalg.norm(g.dw[i]) / W0N) for i in range(len(TRAIN))]
print(f"[학습 sparse 조각 {k}] 끝: {train_dir}", flush=True)

d = copy.deepcopy(base)
d.update(seeds=EVAL, n_games=NE, experiment=f"v2_sparse_v1_s{k}")
d["run"] = {"max_loop_ticks": NE * 20001}
d["replay"] = {"keep_games": [1], "brain": True}
d["final_eval"]["ghosts"]["flybrain"] = {"kind": "flybrain", "config": BRAIN}
c = fe.eval_config(d, COND)
rd = make_run_dir(c, ROOT)
r = build_runner(c, Recorder(rd, c))
gg = r.ghosts
assert gg.evaluate and gg.eta == 0.0 and not gg.dw.any() and r.shaping is None and not gg.lateral_horn
for i, s in enumerate(states):
    gg.load_slot_state(i, s)
r.run()
rows = fe.read_games(rd / "games.csv")
m = fe.per_seed_metrics(rows, EVAL)
per = lambda col: [float(np.mean([float(x[col]) for x in rows if int(x["시드"]) == s])) for s in EVAL]  # noqa: E731
print(f"[평가 sparse 조각 {k}] 끝: {rd}", flush=True)
ev = {"run_dir": rd.name, "score": m["score"], "W": m["W"], "Nc": per("활성 후 포획 수"), "Ne": per("활성 후 먹힘 수"),
      "censored": m["censored"]}
res = {"arm": "sparse", "shard": k, "brain": BRAIN, "lambda": 0.0, "train_seeds": TRAIN, "eval_seeds": EVAL,
       "n_train": NT, "n_eval": NE, "train_dir": train_dir.name, "dw_rel": rel, "eval": {"sparse": ev}}
print(f"\n드문 보상 조각 {k} (λ = 0, 학습 시드 {TRAIN[0]} ~ {TRAIN[-1]}, {NT}판)")
print(f"  ‖Δw‖/‖w0‖ 평균: {mean(rel):.3f}")
print(f"  평가: W {mean(ev['W']):.3f}, 점수 {mean(ev['score']):.1f}, 포획/판 {mean(ev['Nc']):.3f}, "
      f"먹힘/판 {mean(ev['Ne']):.3f}, 중단 {sum(ev['censored'])}")
(OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float) + "\n", encoding="utf-8")
print(f"기록: {OUT / 'results.json'}  (두 조각이 끝나면 SP_SHARD=report)")
