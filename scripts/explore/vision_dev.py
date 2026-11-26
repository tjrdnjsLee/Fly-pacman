"""개발 실행: 본능 없이, 시각을 가진 버섯체가 사냥을 배우는가 (DECISIONS 42.7, 탐색 · 개발, 실행 전 고정. 팔마다 한 번).

위치. 36절: 사냥은 외측뿔 본능 항이 했고 버섯체 학습은 점수를 바꾸지 못했다. 관리자의 결정(2026-11-19):
본능(행동 규칙)을 끄고, 냄새는 목표(퍼텐셜 보상)에 넣고, 실제 버섯체 시각 경로(hemibrain 부속 칼릭스)로
"가까이에서 보이는 팩맨"을 입력으로 준다. 따라갈지는 버섯체가 배운다. 회로는 v2 (configs/brain_v2_nolh.yaml).

모두 저장소 코드다 (스크립트가 뇌 · 실행기를 감싸지 않는다). 측정 단계만 관측을 읽는 감시(spy)를 붙이고 행동은 바꾸지 않는다.

  시각:        vision: {R: 3, s: s_v}       (sense.see: 겹눈, 네 방향, 벽에서 끊김, R칸 안에 보이면 s_v)
  냄새 보상:   reward.shaping.lam: λ        (core.shaping: R += λ ΔΦ, Φ = ±ln(1 + c/c0))
  본능 끔:     lateral_horn.enabled: false  (g = 0)

규칙 (결과 전 고정):
  측정 (VS_ARM=measure): 뇌 v2 · 본능 끔 · Δw = 0 (조건 b0, 갈림길에서 균등 무작위), 시각 R = 3 · s = 1, 냄새 보상 λ = 1,
    측정 시드 9790 ~ 9799 × 20판.
    s_v = 팩맨이 보이는 방향(시각 > 0)의 냄새 입력 ln(1 + c/c0)의 중앙값 ("보일 때 냄새와 같은 세기").
    λ   = 1 / sd(활성 후 확정된 결정마다 냄새 항의 합) (결정당 크기를 포획 +1 · T자 미로 ±1의 척도에).
  팔 (두 프로세스): vision (R = 3, s = s_v)  /  novision (R = 0).  공통: 체제 1 보상 + λ ΔΦ, 학습 시드 6300 ~ 6309, 3000판,
    가중치 100판마다, 재생 판 1 · 1000 · 2000 · 3000. 평가 시드 9780 ~ 9789 (위치로 짝), 200판, η = 0, 학습과 같은 시각 설정.
    평가에는 냄새 보상이 없다 (보상을 쓰지 않는다).
    vision 팔은 같은 가중치를 시각을 끄고도 평가한다 (vision_eval_off, 보고만). novision 팔은 기준 둘도 돈다:
    b0_nolh (뇌 v2, 본능 끔, Δw = 0: 무작위), b0_lh (뇌 v2, 본능 켬, Δw = 0: 본능만).
  판정 (VS_ARM=report):
    J1 (주). vision − novision 의 W 짝 차이 95% t 신뢰구간 (자유도 9)이 0 제외 양수 → "시각을 받은 버섯체가 더 잘 사냥한다".
    J2. novision − b0_nolh 의 W가 0 제외 양수 → "본능 없이 냄새 보상만으로 사냥을 배웠다" (회로 v2에서).
    보고: vision − b0_lh (본능 수준과), vision_eval_off − vision (배운 정책이 시각을 쓰는가), 포획 · 먹힘 · 점수.
    다중 비교 보정은 하지 않는다. 이 개발 자료(6300 · 9780 · 9790대)는 확인의 근거로 쓰지 않는다.
  미리 적은 예측 (Claude): J2 충족. J1 충족 (냄새가 주지 못하는 "지금 위치"를 시각이 준다). vision_eval_off는 vision보다 낮다.
    vision의 W는 b0_lh(본능만)에 못 미칠 수 있다 (3000판, 시각 KC 수가 적다). 틀려도 그대로 적는다.

선행 조건: 회로 v2(data/circuit/mb_v2.npz)를 하이퍼파라미터 재측정 · 작은 과제 재통과 뒤의 값으로 만든 상태. 시작할 때 meta의
status를 출력한다.

  저장소 루트에서:
  uv run --extra pacman python scripts/explore/vision_dev.py   (VS_ARM=measure, 먼저 한 번)
  systemd-inhibit --what=sleep:idle bash -c 'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1;
    VS_ARM=vision uv run --extra pacman python scripts/explore/vision_dev.py > ~/vision_dev_vision.log 2>&1 &
    VS_ARM=novision uv run --extra pacman python scripts/explore/vision_dev.py > ~/vision_dev_novision.log 2>&1 & wait'
  VS_ARM=report uv run python scripts/explore/vision_dev.py

COOP_TEST=1이면 학습 시드 0 · 1, 평가 2 · 3, 측정 4 · 5, 학습 4판 · 평가 3판 · 측정 2판, 규칙 팩맨, 뇌 COOP_BRAIN(기본 v2_nolh).
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

ARM = os.environ.get("VS_ARM", "measure")
if ARM not in ("measure", "vision", "novision", "report"):
    sys.exit(f"멈춤: VS_ARM은 measure, vision, novision, report 중 하나다. 지금: {ARM!r}")
TEST = os.environ.get("COOP_TEST") == "1"
TRAIN = [0, 1] if TEST else list(range(6300, 6310))
EVAL = [2, 3] if TEST else list(range(9780, 9790))
MEAS = [4, 5] if TEST else list(range(9790, 9799 + 1))
NT, NE, NM = (4, 3, 2) if TEST else (3000, 200, 20)
EVERY = 2 if TEST else 100
R_VIS = 3
BRAIN = os.environ.get("COOP_BRAIN", "configs/brain_v2_nolh.yaml") if TEST else "configs/brain_v2_nolh.yaml"
BRAIN_LH = os.environ.get("COOP_BRAIN_LH", "configs/brain_v2.yaml") if TEST else "configs/brain_v2.yaml"
COND = "regime1"
ROOT = Path(os.environ.get("COOP_ROOT", REPO / "runs"))
MEAS_OUT = ROOT / "vision_dev_v1_measure.json"
OUTS = {a: ROOT / f"vision_dev_v1_{a}" for a in ("vision", "novision")}


def mean(v):
    return float(np.mean(v))


def base_cfg() -> dict:
    b = load_config(REPO / "configs" / "final_eval_v1.yaml").to_dict()
    if TEST:
        b["pacman"] = {"kind": "rule_based"}; b["rule_based"] = {"danger_dist": 6, "chase_dist": 8}
    return b


def eval_cfg(base, name, cond, brain, seeds, n, vision):
    d = copy.deepcopy(base)
    d.update(seeds=seeds, n_games=n, experiment=f"vision_dev_v1_{name}")
    d["run"] = {"max_loop_ticks": n * 20001}
    d["replay"] = {"keep_games": [1], "brain": True}
    d["final_eval"]["ghosts"]["flybrain"] = {"kind": "flybrain", "config": brain}
    if vision is not None:
        d["vision"] = vision
    return fe.eval_config(d, cond)


def metrics(rd, seeds):
    rows = fe.read_games(rd / "games.csv")
    m = fe.per_seed_metrics(rows, seeds)
    per = lambda col: [float(np.mean([float(x[col]) for x in rows if int(x["시드"]) == s])) for s in seeds]  # noqa: E731
    return {"run_dir": rd.name, "score": m["score"], "W": m["W"], "Nc": per("활성 후 포획 수"), "Ne": per("활성 후 먹힘 수"),
            "censored": m["censored"]}


def circuit_status(brain) -> str:
    cfg = load_config(REPO / brain)
    f = REPO / cfg.circuit.file
    meta = json.loads(f.with_name(f.stem + ".meta.json").read_text(encoding="utf-8"))
    return f"{meta.get('format')} · {meta.get('status')} · 통과: {meta.get('passed')}"


# ---------------------------------------------------------------- 판정
if ARM == "report":
    R = {}
    for a, o in OUTS.items():
        if not (o / "results.json").exists():
            sys.exit(f"멈춤: {o / 'results.json'}가 없다")
        R[a] = json.loads((o / "results.json").read_text(encoding="utf-8"))
    E = {**R["novision"]["eval"], **R["vision"]["eval"]}
    M = json.loads(MEAS_OUT.read_text(encoding="utf-8"))
    print(f"시각 개발 실행 (s_v = {M['s_v']:.4f}, λ = {M['lambda']:.4f}, R = {R_VIS}, 평가 시드 {EVAL[0]} ~ {EVAL[-1]}, {NE}판)")
    print("  칸               W      점수     포획/판  먹힘/판  중단")
    for k in ("b0_nolh", "novision", "vision", "vision_eval_off", "b0_lh"):
        c = E[k]
        print(f"  {k:<15}  {mean(c['W']):.3f}  {mean(c['score']):7.1f}  {mean(c['Nc']):.3f}    {mean(c['Ne']):.3f}    "
              f"{sum(c['censored'])}")

    def ci(a, b, key, nd=3):
        r = fe.paired_ci(E[a][key], E[b][key], 0.95)
        return r, f"{r['mean']:+.{nd}f} [{r['lo']:+.{nd}f}, {r['hi']:+.{nd}f}]{' *' if r['excludes_zero'] else ''}"

    judge = {}
    for name, a, b in (("J1", "vision", "novision"), ("J2", "novision", "b0_nolh")):
        r, t = ci(a, b, "W")
        judge[name] = {"W": r["mean"], "ci": [r["lo"], r["hi"]], "met": bool(r["excludes_zero"] and r["mean"] > 0)}
        print(f"\n  {name}. {a} − {b} W: {t} → {'충족' if judge[name]['met'] else '미충족'}")
    for a, b in (("vision", "b0_lh"), ("vision_eval_off", "vision"), ("vision", "b0_nolh")):
        print(f"\n  보고: {a} − {b}")
        for key, nd in (("W", 3), ("score", 1), ("Nc", 3), ("Ne", 3)):
            print(f"    {key}: {ci(a, b, key, nd)[1]}")
    (ROOT / "vision_dev_v1_report.json").write_text(json.dumps(judge, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"기록: {ROOT / 'vision_dev_v1_report.json'}")
    sys.exit(0)

if not TEST:
    fe.check_threads()
print(f"회로: {circuit_status(BRAIN)}", flush=True)
base = base_cfg()

# ---------------------------------------------------------------- 측정
if ARM == "measure":
    if MEAS_OUT.exists():
        sys.exit(f"멈춤: {MEAS_OUT}가 이미 있다")
    c = eval_cfg(base, "measure", "b0", BRAIN, MEAS, NM, {"R": R_VIS, "s": 1.0})
    d = c.to_dict(); d["reward"] = {"shaping": {"lam": 1.0}}
    c = Config(d)
    rd = make_run_dir(c, ROOT)
    r = build_runner(c, Recorder(rd, c))
    g = r.ghosts
    assert g.eta == 0.0 and not g.dw.any() and not g.lateral_horn and g.g == 0.0 and r.shaping is not None
    seen_odor, dec = [], []
    acc = np.zeros((g.B, g.F))
    o_prop, o_give, o_commit = g.propose, g.give_reward, g.commit
    o_reset, o_release = g.reset_life, g.on_release

    def propose(obs, alive, views=None):   # 감시: 보이는 방향의 냄새 입력 (행동은 바꾸지 않는다)
        m = (obs.vision > 0) & np.asarray(alive, bool)[:, None, None]
        seen_odor.extend(obs.odor[m].tolist())
        return o_prop(obs, alive, views)

    def give_reward(Rv):                    # b0는 체제 보상이 0이므로 Rv = 냄새 항(λ = 1)
        acc[:] += np.asarray(Rv)
        return o_give(Rv)

    def commit(committed, active):
        sel = np.asarray(committed, bool) & np.asarray(active, bool)
        dec.extend(acc[sel].tolist())
        acc[np.asarray(committed, bool)] = 0.0
        return o_commit(committed, active)
    def reset_life(mask):                   # 뇌의 R_n처럼 목숨 · 판 시작과 방출에서 비운다
        acc[np.asarray(mask, bool)] = 0.0
        return o_reset(mask)

    def on_release(mask):
        acc[np.asarray(mask, bool)] = 0.0
        return o_release(mask)
    g.propose, g.give_reward, g.commit, g.reset_life, g.on_release = propose, give_reward, commit, reset_life, on_release
    r.run()
    so, dv = np.asarray(seen_odor), np.asarray(dec)
    out = {"run_dir": rd.name, "seeds": MEAS, "n_games": NM, "R": R_VIS,
           "n_seen": int(so.size), "s_v": float(np.median(so)) if so.size else float("nan"),
           "seen_odor_q": [float(np.quantile(so, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)] if so.size else [],
           "n_decisions": int(dv.size), "dphi_mean": float(dv.mean()), "dphi_sd": float(dv.std(ddof=1)),
           "lambda": float(1.0 / dv.std(ddof=1))}
    if not (out["s_v"] > 0):
        sys.exit(f"멈춤: s_v가 양수가 아니다 ({out['s_v']}). 보인 횟수 {so.size}")
    MEAS_OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[측정] 보인 칸 {so.size}개 → s_v = {out['s_v']:.4f}; 결정 {dv.size}개, ΔΦ sd {out['dphi_sd']:.4f} → λ = {out['lambda']:.4f}")
    print(f"기록: {MEAS_OUT}")
    sys.exit(0)

# ---------------------------------------------------------------- 학습 + 평가
OUT = OUTS[ARM]
if (OUT / "results.json").exists():
    sys.exit(f"멈춤: {OUT / 'results.json'}가 이미 있다")
if not MEAS_OUT.exists():
    sys.exit(f"멈춤: {MEAS_OUT}가 없다. 먼저 VS_ARM=measure")
M = json.loads(MEAS_OUT.read_text(encoding="utf-8"))
LAM, SV = float(M["lambda"]), float(M["s_v"])
VIS = {"R": R_VIS, "s": SV} if ARM == "vision" else None
OUT.mkdir(parents=True, exist_ok=True)

tr = copy.deepcopy(base)
del tr["final_eval"]
tr.update(experiment=f"vision_dev_v1_{ARM}_train", seeds=TRAIN, n_games=NT, condition=COND, evaluate=False,
          ghosts={"kind": "flybrain", "config": BRAIN})
tr["run"] = {"max_loop_ticks": NT * 20001}
tr["replay"] = {"keep_games": sorted({1, 1000, 2000, NT}) if not TEST else [1, NT], "brain": True}
tr["weights"] = {"every_games": EVERY}
tr["reward"] = {"shaping": {"lam": LAM}}
if VIS is not None:
    tr["vision"] = VIS
cfg = Config(tr)
train_dir = make_run_dir(cfg, ROOT)
runner = build_runner(cfg, Recorder(train_dir, cfg))
g = runner.ghosts
assert not g.evaluate and g.eta > 0 and not g.lateral_horn and g.g == 0.0 and runner.weight_writer is not None
assert runner.shaping is not None and abs(runner.shaping.lam - LAM) < 1e-12
assert runner.vision_R == (R_VIS if VIS else 0) and g.circuit.has_vision
runner.run()
assert g.dw.any()
states = [g.slot_state(i) for i in range(len(TRAIN))]
print(f"[학습 {ARM}] 끝: {train_dir}", flush=True)
W0N = float(np.linalg.norm(g.circuit.w0)) * np.sqrt(g.F)
rel = [float(np.linalg.norm(g.dw[i]) / W0N) for i in range(len(TRAIN))]


def evaluate(name, cond, brain, load, vision):
    c = eval_cfg(base, name, cond, brain, EVAL, NE, vision)
    rd = make_run_dir(c, ROOT)
    r = build_runner(c, Recorder(rd, c))
    gg = r.ghosts
    assert gg.evaluate and gg.eta == 0.0 and not gg.dw.any() and r.shaping is None
    if load:
        for i, s in enumerate(states):
            gg.load_slot_state(i, s)
    r.run()
    print(f"[평가 {name}] 끝: {rd} (본능 {'켬' if gg.lateral_horn else '끔'}, 시각 R = {r.vision_R})", flush=True)
    return metrics(rd, EVAL)


res = {"arm": ARM, "lambda": LAM, "s_v": SV, "R": R_VIS if VIS else 0, "train_seeds": TRAIN, "eval_seeds": EVAL,
       "n_train": NT, "n_eval": NE, "train_dir": train_dir.name, "dw_rel": rel, "eval": {}}
res["eval"][ARM] = evaluate(ARM, COND, BRAIN, True, VIS)
if ARM == "vision":
    res["eval"]["vision_eval_off"] = evaluate("vision_eval_off", COND, BRAIN, True, None)
else:
    res["eval"]["b0_nolh"] = evaluate("b0_nolh", "b0", BRAIN, False, None)
    res["eval"]["b0_lh"] = evaluate("b0_lh", "b0", BRAIN_LH, False, None)
L = res["eval"][ARM]
print(f"\n개발 실행 {ARM} (λ = {LAM:.4f}, s_v = {SV:.4f}, R = {res['R']}, 학습 시드 {TRAIN[0]} ~ {TRAIN[-1]}, {NT}판, 본능 끔)")
print("  ‖Δw‖/‖w0‖ (시드별): " + ", ".join(f"{v:.3f}" for v in rel))
print(f"  평가: W {mean(L['W']):.3f}, 점수 {mean(L['score']):.1f}, 포획/판 {mean(L['Nc']):.3f}, 먹힘/판 {mean(L['Ne']):.3f}, "
      f"중단 {sum(L['censored'])}")
(OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float) + "\n", encoding="utf-8")
print(f"기록: {OUT / 'results.json'}  (두 팔이 끝나면 VS_ARM=report)")
