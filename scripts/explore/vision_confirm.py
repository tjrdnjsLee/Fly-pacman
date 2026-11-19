"""확인 실행: 본능 없는 버섯체는 냄새 퍼텐셜 보상으로 "배운" 것인가, 가중치가 흔들려서 나아진 것인가 (DECISIONS 42.12, 실행 전 고정).

위치. 42.7 개발 실행에서 novision 팔(뇌 v2, 본능 끔, 체제 1 + λΔΦ)이 무작위(Δw = 0)보다 W +0.194 [+0.136, +0.252] 높았다.
그러나 41.14 · 41.15에서 보상과 무관한 막걸음만으로도 W가 올랐다. 그래서 확인은 무작위가 아니라 **부호 대조군**과 비교한다:
같은 η · 같은 흔적 · 같은 R̄로 움직이되, 가중치 갱신의 (R_n − R̄)에 ξ = ±1(갱신마다 무작위)을 곱한 버섯체
(뇌 설정 sign_control, configs/brain_v2_nolh_ctrl.yaml). 움직이는 크기는 같고 보상과의 상관만 없다.

팔 (VC_ARM, 두 프로세스):
  learn  뇌 configs/brain_v2_nolh.yaml        (42.7 novision 팔과 같은 조건)
  ctrl   뇌 configs/brain_v2_nolh_ctrl.yaml   (부호 대조군, ξ 시드 20261120)
  report 판정 (학습 없음)
공통: 시각 끔, 체제 1 보상 + λΔΦ, λ = 42.7 측정값(runs/vision_dev_v1_measure.json의 lambda, 1.5109) 그대로.
  학습 시드 6400 ~ 6419 (20개), 3000판, 가중치 100판마다, 재생 판 1 · 3000. 평가 시드 9730 ~ 9749 (학습 시드와 위치로 짝), 200판, η = 0.
  learn 팔은 b0_nolh(무작위: 뇌 v2, 본능 끔, Δw = 0)도 같은 평가 시드로 돈다 (보고).
판정 (결과 전 고정):
  K1 (주). (learn − ctrl)의 W 짝 차이, 평가 시드 20쌍, t 95% 신뢰구간 (자유도 19)이 0 제외 양수
      → "본능 없는 버섯체가 냄새 보상으로 배워 이긴다 (막걸음이 아니다)". 아니면 "구별하지 못했다".
  보고: learn − b0_nolh, ctrl − b0_nolh, 점수 · 포획 · 먹힘, ‖Δw‖/‖w0‖ (두 팔이 비슷한 크기로 움직였는지).
시드 수 20의 근거: 42.7의 (novision − 무작위) W 짝 차이의 반폭 0.058(시드 10개)에서 시드 간 sd ≈ 0.081을 역산하면 20개의 반폭은
  약 0.038이다. 실행 시간(시드 30개면 약 8시간)과 함께 정한 판단이다.
미리 적은 예측 (Claude): K1 충족. ctrl − b0_nolh는 양수일 수 있다 (41.14처럼). 틀려도 그대로 적는다. 한 번만 돌린다.

  저장소 루트에서 (두 프로세스를 나란히, 약 5 ~ 6시간 추정):
  systemd-inhibit --what=sleep:idle bash -c 'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1;
    VC_ARM=learn uv run --extra pacman python scripts/explore/vision_confirm.py > ~/vision_confirm_learn.log 2>&1 &
    VC_ARM=ctrl uv run --extra pacman python scripts/explore/vision_confirm.py > ~/vision_confirm_ctrl.log 2>&1 & wait'
  VC_ARM=report uv run python scripts/explore/vision_confirm.py

COOP_TEST=1이면 학습 시드 0 · 1, 평가 2 · 3, 학습 4판 · 평가 3판, 규칙 팩맨, 뇌 COOP_BRAIN / COOP_BRAIN_CTRL, λ는 COOP_ROOT의 측정 파일.
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

ARM = os.environ.get("VC_ARM")
if ARM not in ("learn", "ctrl", "report"):
    sys.exit(f"멈춤: VC_ARM은 learn, ctrl, report 중 하나다. 지금: {ARM!r}")
TEST = os.environ.get("COOP_TEST") == "1"
TRAIN = [0, 1] if TEST else list(range(6400, 6420))
EVAL = [2, 3] if TEST else list(range(9730, 9750))
NT, NE = (4, 3) if TEST else (3000, 200)
EVERY = 2 if TEST else 100
BRAINS = {"learn": os.environ.get("COOP_BRAIN", "configs/brain_v2_nolh.yaml") if TEST else "configs/brain_v2_nolh.yaml",
          "ctrl": os.environ.get("COOP_BRAIN_CTRL", "configs/brain_v2_nolh_ctrl.yaml") if TEST
          else "configs/brain_v2_nolh_ctrl.yaml"}
LAM_WANT = None if TEST else 1.5109
COND = "regime1"
ROOT = Path(os.environ.get("COOP_ROOT", REPO / "runs"))
OUTS = {a: ROOT / f"vision_confirm_v1_{a}" for a in ("learn", "ctrl")}


def mean(v):
    return float(np.mean(v))


if ARM == "report":
    R = {}
    for a, o in OUTS.items():
        if not (o / "results.json").exists():
            sys.exit(f"멈춤: {o / 'results.json'}가 없다")
        R[a] = json.loads((o / "results.json").read_text(encoding="utf-8"))
    E = {**R["learn"]["eval"], **R["ctrl"]["eval"]}
    print(f"확인 실행 (λ = {R['learn']['lambda']:.4f}, 학습 시드 {TRAIN[0]} ~ {TRAIN[-1]}, 평가 시드 {EVAL[0]} ~ {EVAL[-1]}, {NE}판)")
    print("  칸        W      점수     포획/판  먹힘/판  중단   ‖Δw‖/‖w0‖ 평균")
    for k in ("b0_nolh", "ctrl", "learn"):
        c = E[k]
        dwr = f"{mean(R[k]['dw_rel']):.3f}" if k in R else ""
        print(f"  {k:<8}  {mean(c['W']):.3f}  {mean(c['score']):7.1f}  {mean(c['Nc']):.3f}    {mean(c['Ne']):.3f}    "
              f"{sum(c['censored'])}     {dwr}")

    def ci(a, b, key, nd=3):
        r = fe.paired_ci(E[a][key], E[b][key], 0.95)
        return r, f"{r['mean']:+.{nd}f} [{r['lo']:+.{nd}f}, {r['hi']:+.{nd}f}]{' *' if r['excludes_zero'] else ''}"

    r, t = ci("learn", "ctrl", "W")
    met = bool(r["excludes_zero"] and r["mean"] > 0)
    print(f"\n  K1. learn − ctrl W: {t} (자유도 {r['df']}) → "
          + ("충족: 냄새 보상으로 배웠다 (막걸음이 아니다)" if met else "미충족: 구별하지 못했다"))
    for a, b in (("learn", "ctrl"), ("learn", "b0_nolh"), ("ctrl", "b0_nolh")):
        print(f"\n  보고: {a} − {b}")
        for key, nd in (("W", 3), ("score", 1), ("Nc", 3), ("Ne", 3)):
            print(f"    {key}: {ci(a, b, key, nd)[1]}")
    out = {"K1": {"W": r["mean"], "ci": [r["lo"], r["hi"]], "met": met}}
    (ROOT / "vision_confirm_v1_report.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"기록: {ROOT / 'vision_confirm_v1_report.json'}")
    sys.exit(0)

OUT = OUTS[ARM]
if (OUT / "results.json").exists():
    sys.exit(f"멈춤: {OUT / 'results.json'}가 이미 있다")
if not TEST:
    fe.check_threads()
meas = json.loads((ROOT / "vision_dev_v1_measure.json").read_text(encoding="utf-8"))
LAM = float(meas["lambda"])
if LAM_WANT is not None and abs(LAM - LAM_WANT) > 5e-5:
    sys.exit(f"멈춤: 측정 파일의 λ {LAM}가 고정값 {LAM_WANT}와 다르다")
BRAIN = BRAINS[ARM]
bcfg = load_config(REPO / BRAIN)
lh = bcfg.get("lateral_horn"); sc = bcfg.get("sign_control")
assert lh is not None and not lh["enabled"], "본능을 끈 뇌여야 한다"
assert (sc is not None and sc["enabled"]) == (ARM == "ctrl"), "ctrl 팔만 부호 대조군이어야 한다"
OUT.mkdir(parents=True, exist_ok=True)

base = load_config(REPO / "configs" / "final_eval_v1.yaml").to_dict()
if TEST:
    base["pacman"] = {"kind": "rule_based"}; base["rule_based"] = {"danger_dist": 6, "chase_dist": 8}
tr = copy.deepcopy(base)
del tr["final_eval"]
tr.update(experiment=f"vision_confirm_v1_{ARM}_train", seeds=TRAIN, n_games=NT, condition=COND, evaluate=False,
          ghosts={"kind": "flybrain", "config": BRAIN})
tr["run"] = {"max_loop_ticks": NT * 20001}
tr["replay"] = {"keep_games": [1, NT], "brain": True}
tr["weights"] = {"every_games": EVERY}
tr["reward"] = {"shaping": {"lam": LAM}}
cfg = Config(tr)
train_dir = make_run_dir(cfg, ROOT)
runner = build_runner(cfg, Recorder(train_dir, cfg))
g = runner.ghosts
assert not g.evaluate and g.eta > 0 and not g.lateral_horn and g.g == 0.0 and g.sign_control == (ARM == "ctrl")
assert runner.shaping is not None and abs(runner.shaping.lam - LAM) < 1e-12 and runner.vision_R == 0
runner.run()
assert g.dw.any()
states = [g.slot_state(i) for i in range(len(TRAIN))]
W0N = float(np.linalg.norm(g.circuit.w0)) * np.sqrt(g.F)
rel = [float(np.linalg.norm(g.dw[i]) / W0N) for i in range(len(TRAIN))]
print(f"[학습 {ARM}] 끝: {train_dir}", flush=True)


def evaluate(name, cond, load, brain):
    d = copy.deepcopy(base)
    d.update(seeds=EVAL, n_games=NE, experiment=f"vision_confirm_v1_{name}")
    d["run"] = {"max_loop_ticks": NE * 20001}
    d["replay"] = {"keep_games": [1], "brain": True}
    d["final_eval"]["ghosts"]["flybrain"] = {"kind": "flybrain", "config": brain}   # 평가는 η = 0이라 sign_control은 아무것도 하지 않는다
    c = fe.eval_config(d, cond)
    rd = make_run_dir(c, ROOT)
    r = build_runner(c, Recorder(rd, c))
    gg = r.ghosts
    assert gg.evaluate and gg.eta == 0.0 and not gg.dw.any() and r.shaping is None and not gg.lateral_horn
    if load:
        for i, s in enumerate(states):
            gg.load_slot_state(i, s)
    r.run()
    rows = fe.read_games(rd / "games.csv")
    m = fe.per_seed_metrics(rows, EVAL)
    per = lambda col: [float(np.mean([float(x[col]) for x in rows if int(x["시드"]) == s])) for s in EVAL]  # noqa: E731
    print(f"[평가 {name}] 끝: {rd}", flush=True)
    return {"run_dir": rd.name, "score": m["score"], "W": m["W"], "Nc": per("활성 후 포획 수"), "Ne": per("활성 후 먹힘 수"),
            "censored": m["censored"]}


res = {"arm": ARM, "brain": BRAIN, "lambda": LAM, "train_seeds": TRAIN, "eval_seeds": EVAL, "n_train": NT, "n_eval": NE,
       "train_dir": train_dir.name, "dw_rel": rel, "eval": {}}
res["eval"][ARM] = evaluate(ARM, COND, True, BRAIN)
if ARM == "learn":
    res["eval"]["b0_nolh"] = evaluate("b0_nolh", "b0", False, BRAINS["learn"])
L = res["eval"][ARM]
print(f"\n확인 실행 {ARM} (λ = {LAM:.4f}, 학습 시드 {TRAIN[0]} ~ {TRAIN[-1]}, {NT}판)")
print("  ‖Δw‖/‖w0‖ 평균: " + f"{mean(rel):.3f}")
print(f"  평가: W {mean(L['W']):.3f}, 점수 {mean(L['score']):.1f}, 포획/판 {mean(L['Nc']):.3f}, 먹힘/판 {mean(L['Ne']):.3f}, "
      f"중단 {sum(L['censored'])}")
(OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float) + "\n", encoding="utf-8")
print(f"기록: {OUT / 'results.json'}  (두 팔이 끝나면 VC_ARM=report)")
