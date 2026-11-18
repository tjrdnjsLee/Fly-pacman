"""대조군: 보상과 무관하게 움직이는 초파리 (2026-11-18, 개발 결과를 본 뒤의 탐색, 실행 전 고정. 한 번만 돌린다).

동기. 개발 실행(phero_dev.py)에서 q = 0은 k̄_ph −6.22 [−9.62, −2.82], 학습 − solo W +0.136 [+0.056, +0.217]이었다.
그런데 보상이 회피를 밀지 않는 q = 1도 학습 − solo W +0.123 [+0.032, +0.213]이었다 (q0 − q1 W +0.014 [−0.081, +0.109]).
가설: 이득은 보상으로 배운 것이 아니라 막걸음으로 흩어진 k_ph와 비대칭 지형에서 나온다. 이를 가른다.

대조군. q = 0 개발 실행과 모두 같다 (보상 q = 0, η_ph = 4.1152, 학습 시드 7600 ~ 7609, 3000판, 평가 시드 9900 ~ 9909, 200판).
다른 것 하나: k_ph의 갱신  k_ph ← clip(k_ph + η_ph ξ (R_n − R̄) e_ph)  에서 ξ = ±1을 갱신마다 무작위로 뽑는다
(전용 난수, 시드 20261118). 움직이는 크기는 같고 보상과의 상관만 없다. R̄ 갱신과 흔적 e_ph는 그대로다.
CTRL_SIGN=plus이면 ξ = +1로 고정한다 (구현 검사용: q = 0 개발 실행과 같은 k_ph가 나와야 한다).

판정 (탐색, 실행 전 고정. 개발 시드라 확인의 근거로 쓰지 않는다):
  (A) 평가 W: (q = 0 학습 − 대조군) 짝 차이 (평가 시드 10쌍, t 95% CI)가 0을 제외하고 양수이면
      "보상과의 상관이 승률 이득을 만들었다". 아니면 "개발 실행의 승률 이득은 막걸음과 구별되지 않는다".
  (B) k̄_ph: (q = 0 − 대조군) 판 3000의 시드 평균 짝 차이 (학습 시드 10쌍)가 0을 제외하고 음수이면 "보상이 k_ph를 밀었다".
  함께 보고: 대조군 − solo의 W (막걸음만으로 승률이 오르는가), 대조군의 k̄_ph와 초파리별 범위.
q = 0 개발 실행의 결과 파일(runs/phero_dev_v1_q0/results.json)이 있어야 끝의 비교를 한다.

  저장소 루트에서:
  D=$(ls ~/다운로드/phero_ctrl.py 2>/dev/null | head -1); echo $D
  systemd-inhibit --what=sleep:idle env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --extra pacman python - < $D > ~/phero_ctrl.log 2>&1

COOP_TEST=1이면 시드 0 · 1 (학습) / 2 · 3 (평가), 학습 4판 · 평가 3판, 규칙 팩맨. COOP_ROOT로 결과 폴더의 뿌리를 바꾼다.
"""
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path.cwd()
if not (REPO / "configs" / "brain_kph_v1.yaml").exists():
    sys.exit("멈춤: 저장소 루트에서 돌린다 (configs/brain_kph_v1.yaml이 없다)")
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
import final_eval as fe  # noqa: E402
from flypac.core.config import Config, load_config  # noqa: E402
from flypac.core.runner import build_runner  # noqa: E402
from flypac.recording.recorder import Recorder, make_run_dir  # noqa: E402
from flypac.recording.weights import read_weight_file  # noqa: E402

Q = 0.0
SIGN = os.environ.get("CTRL_SIGN", "random")
if SIGN not in ("random", "plus"):
    sys.exit(f"멈춤: CTRL_SIGN은 random 또는 plus. 지금: {SIGN!r}")
TAG = "ctrl" if SIGN == "random" else "ctrlplus"
TEST = os.environ.get("COOP_TEST") == "1"
TRAIN = [0, 1] if TEST else list(range(7600, 7610))
EVAL = [2, 3] if TEST else list(range(9900, 9910))
NT, NE = (4, 3) if TEST else (3000, 200)
EVERY = 2 if TEST else 100
BRAIN = "configs/brain_kph_v1.yaml"
K, ETA_PH = 10.0, 10.0 / (0.00081 * 3000)
ROOT = Path(os.environ.get("COOP_ROOT", "runs")); OUT = ROOT / f"phero_dev_v1_{TAG}"
if (OUT / "results.json").exists():
    sys.exit(f"멈춤: {OUT / 'results.json'}가 이미 있다")
if not TEST:
    fe.check_threads()
OUT.mkdir(parents=True, exist_ok=True)

brain = load_config(REPO / BRAIN).to_dict()
assert brain["instinct_gain"]["eta_k"] == 0.0 and brain["instinct_gain"]["k_init"] == K and brain["instinct_gain"]["freeze_synapses"]
brain["phero_gain"].update(learn=True, dry_run=False, eta_ph=ETA_PH, kph_init=0.0)
brain_path = (OUT / "brain_kph_dev.yaml").resolve()
brain_path.write_text("# phero_dev.py가 만든 뇌 설정: configs/brain_kph_v1.yaml에 phero_gain.learn: true, dry_run: false, "
                      f"eta_ph: {ETA_PH!r}, kph_init: 0.0을 덮어씀\n"
                      + yaml.safe_dump(brain, allow_unicode=True, sort_keys=False), encoding="utf-8")

base = load_config(REPO / "configs" / "final_eval_v1.yaml").to_dict()
base["phero"]["emission"] = "marker"
base["phero"]["view"] = "others"
base["reward"] = {"q": Q}
if TEST:
    base["pacman"] = {"kind": "rule_based"}; base["rule_based"] = {"danger_dist": 6, "chase_dist": 8}

# ---- 학습 ----
tr = copy.deepcopy(base)
del tr["final_eval"]
tr.update(experiment=f"phero_dev_v1_{TAG}_train", seeds=TRAIN, n_games=NT, condition="regime3", evaluate=False,
          ghosts={"kind": "flybrain", "config": str(brain_path)})
tr["run"] = {"max_loop_ticks": NT * 20001}
tr["replay"] = {"keep_games": sorted({1, 1000, 2000, NT}) if not TEST else [1, NT], "brain": True}
tr["weights"] = {"every_games": EVERY}
cfg = Config(tr)
train_dir = make_run_dir(cfg, ROOT)
runner = build_runner(cfg, Recorder(train_dir, cfg))
g = runner.ghosts
assert not g.evaluate and g.eta == 0.0 and g.eta_k == 0.0 and g.ph_learn and not g.dry_run and abs(g.eta_ph - ETA_PH) < 1e-12
assert (g.k_gain == K).all() and (g.kph_gain == 0.0).all() and runner.regime.q == Q and runner.weight_writer is not None
assert runner.regime.emission_kind == "marker" and runner.phero_view == "others"
xi_rng = np.random.default_rng(20261118)
orig_learn = g._learn
n_flip = [0, 0]


def learn(b, f):
    """k_ph 갱신만 ξ (R_n − R̄)로 하고, 나머지(R̄ 갱신)는 원래 _learn에 맡긴다 (그동안 η_ph = 0)."""
    xi = 1.0 if SIGN == "plus" else (1.0 if xi_rng.random() < 0.5 else -1.0)
    n_flip[xi < 0] += 1
    k = g.kph_gain[b, f] + ETA_PH * xi * (g.Rn[b, f] - g.rbar[b, f]) * g.e_ph[b, f]
    g.kph_gain[b, f] = min(max(k, g.kph_min), g.kph_max)
    g.eta_ph = 0.0
    try:
        orig_learn(b, f)
    finally:
        g.eta_ph = ETA_PH


g._learn = learn
runner.run()
print(f"  ξ: +1 {n_flip[0]}번, −1 {n_flip[1]}번", flush=True)
assert not g.dw.any() and (g.k_gain == K).all()
learned = g.kph_gain.copy()
print(f"[학습 {TAG}] 끝: {train_dir}", flush=True)

curve = {}
for p in sorted((train_dir / "weights").glob("s*_g*.npz")):
    w = read_weight_file(p)
    curve.setdefault(int(w["seed"]), {})[int(w["game_no"])] = float(np.mean(w["kph_gain"]))
games = sorted(next(iter(curve.values())))
last = [curve[s][NT] for s in TRAIN]
assert np.allclose(last, learned.mean(axis=1))


def t_ci(v):
    return fe.paired_ci(list(v), [0.0] * len(v), 0.95)


# ---- 평가 ----
def evaluate(name, kph):
    d = copy.deepcopy(base)
    d.update(seeds=EVAL, n_games=NE, experiment=f"phero_dev_v1_{name}")
    d["run"] = {"max_loop_ticks": NE * 20001}
    d["replay"] = {"keep_games": [1], "brain": True}
    d["final_eval"]["ghosts"]["flybrain"] = {"kind": "flybrain", "config": BRAIN}
    c = fe.eval_config(d, "regime3")
    rd = make_run_dir(c, ROOT)
    r = build_runner(c, Recorder(rd, c))
    gg = r.ghosts
    gg.k_gain[:] = K
    gg.kph_gain[:] = kph
    assert gg.evaluate and gg.eta_ph == 0.0 and not gg.dw.any() and r.phero_view == "others"
    r.run()
    rows = fe.read_games(rd / "games.csv")
    m = fe.per_seed_metrics(rows, EVAL)
    per = lambda col: [float(np.mean([float(x[col]) for x in rows if int(x["시드"]) == s])) for s in EVAL]  # noqa: E731
    print(f"[평가 {name}] 끝: {rd}", flush=True)
    return {"run_dir": rd.name, "score": m["score"], "W": m["W"], "Nc": per("활성 후 포획 수"), "Ne": per("활성 후 먹힘 수"),
            "censored": m["censored"]}


res = {"q": Q, "eta_ph": ETA_PH, "train_seeds": TRAIN, "eval_seeds": EVAL, "n_train": NT, "n_eval": NE,
       "train_dir": train_dir.name, "kph_final": learned.tolist(), "kph_curve": {str(s): curve[s] for s in TRAIN},
       "kph_last_seed_mean": last, "kph_last_ci": t_ci(last), "eval": {}}
res["sign"], res["xi_counts"] = SIGN, {"plus": n_flip[0], "minus": n_flip[1]}
res["eval"][f"learned_{TAG}"] = evaluate(f"learned_{TAG}", learned)

print(f"\n대조군 ({TAG}: ξ {SIGN}, q = 0, η_ph = {ETA_PH:.4f}, 학습 시드 {TRAIN[0]} ~ {TRAIN[-1]}, {NT}판)")
step = max(1, len(games) // 10)
print("  k̄_ph (시드 평균): " + ", ".join(f"판 {n}: {np.mean([curve[s][n] for s in TRAIN]):+.2f}" for n in games[step - 1::step]))
ci = res["kph_last_ci"]
print(f"  판 {NT}의 k̄_ph: {ci['mean']:+.2f} [{ci['lo']:+.2f}, {ci['hi']:+.2f}]{' * (0 제외)' if ci['excludes_zero'] else ''}")
print(f"  초파리별 k_ph 범위: {learned.min():+.2f} ~ {learned.max():+.2f}, 시드별 평균: " + ", ".join(f"{v:+.1f}" for v in last))
C = res["eval"][f"learned_{TAG}"]
print(f"  평가 (시드 {EVAL[0]} ~ {EVAL[-1]}, {NE}판): 대조군 W {np.mean(C['W']):.3f}, 점수 {np.mean(C['score']):.1f}, "
      f"포획/판 {np.mean(C['Nc']):.3f}, 먹힘/판 {np.mean(C['Ne']):.3f}, 중단 {sum(C['censored'])}")
(OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float) + "\n", encoding="utf-8")
print(f"기록: {OUT / 'results.json'}")
q0p = ROOT / "phero_dev_v1_q0" / "results.json"
if not q0p.exists():
    sys.exit(f"비교 생략: {q0p}가 없다")
a = json.loads(q0p.read_text(encoding="utf-8"))
L0, S = a["eval"]["learned_q0"], a["eval"]["solo"]
assert a["eval_seeds"] == EVAL and a["train_seeds"] == TRAIN
for name, x, y in (("q0 학습 − 대조군", L0, C), ("대조군 − solo", C, S)):
    for key, nd in (("W", 3), ("score", 1), ("Nc", 3), ("Ne", 3)):
        c = fe.paired_ci(x[key], y[key], 0.95)
        print(f"  {name} {key}: {c['mean']:+.{nd}f} [{c['lo']:+.{nd}f}, {c['hi']:+.{nd}f}]{' *' if c['excludes_zero'] else ''}")
kd = fe.paired_ci(a["kph_last_seed_mean"], last, 0.95)
print(f"  (B) k̄_ph q0 − 대조군 (학습 시드 짝): {kd['mean']:+.2f} [{kd['lo']:+.2f}, {kd['hi']:+.2f}]{' *' if kd['excludes_zero'] else ''}")
wa = fe.paired_ci(L0["W"], C["W"], 0.95)
print(f"  판정 (A) {'충족: 보상과의 상관이 승률 이득을 만들었다' if wa['lo'] > 0 else '미충족: 개발 실행의 승률 이득은 막걸음과 구별되지 않는다'}")
print(f"  판정 (B) {'충족: 보상이 k_ph를 밀었다' if kd['hi'] < 0 else '미충족'}")
