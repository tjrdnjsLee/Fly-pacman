"""확인 실행: 보상으로 배운 초파리가 막걸음보다 더 자주 이기는가 (DECISIONS 41.15, 실행 전 고정. 팔마다 한 번).

두 팔은 모두 41.13의 학습 조건이다: 체제 3의 식에서 q = 0, η_ph = 4.1152, 시냅스 얼림, k = 10 고정, k_ph는 0에서 시작,
범위 [−50, 50], 초파리마다 따로, phero.emission marker, phero.view others.
  learn: k_ph ← clip(k_ph + η_ph (R_n − R̄) e_ph)
  ctrl:  k_ph ← clip(k_ph + η_ph ξ (R_n − R̄) e_ph),  ξ = ±1을 갱신마다 무작위 (전용 난수, 시드 20261119). R̄ · e_ph는 그대로.
학습 시드 7700 ~ 7729, 칸마다 3000판, 가중치 100판마다, 재생 판 1 · 1000 · 2000 · 3000.
평가 시드 9920 ~ 9949 (학습 시드와 위치로 짝), 칸마다 200판, 평가 모드에 학습 끝의 k_ph를 넣는다.
learn 팔은 solo(k_ph = 0)도 같은 평가 시드로 평가한다 (보고용).

판정 (41.15): (learn − ctrl)의 초파리 승률 W 짝 차이 (평가 시드 30쌍, t 95% 신뢰구간, 자유도 29)가 0을 제외하고 양수이면
"보상으로 배운 초파리가 보상과 무관하게 움직인 초파리보다 더 자주 이긴다". 보고: k̄_ph의 (learn − ctrl) 학습 시드 짝 차이,
learn − solo, ctrl − solo, 점수 · 포획 · 먹힘. 판정은 PH_ARM=report가 두 팔의 결과 파일로 낸다.

  저장소 루트에서 (두 팔을 나란히, 끝나면 보고):
  systemd-inhibit --what=sleep:idle bash -c 'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1;
    PH_ARM=learn uv run --extra pacman python scripts/explore/phero_confirm.py > ~/phero_confirm_learn.log 2>&1 &
    PH_ARM=ctrl  uv run --extra pacman python scripts/explore/phero_confirm.py > ~/phero_confirm_ctrl.log 2>&1 & wait'
  PH_ARM=report uv run python scripts/explore/phero_confirm.py

COOP_TEST=1이면 시드 0 · 1 (학습) / 2 · 3 (평가), 학습 4판 · 평가 3판, 규칙 팩맨. COOP_ROOT로 결과 폴더의 뿌리를 바꾼다.
CTRL_SIGN=plus는 시험 전용이다 (ctrl 팔의 ξ를 +1로 고정해 learn 팔과 같은 결과가 나오는지 본다). 실제 실행에서는 쓰지 않는다.
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

ARM = os.environ.get("PH_ARM")
if ARM not in ("learn", "ctrl", "report"):
    sys.exit(f"멈춤: 환경변수 PH_ARM을 learn, ctrl, report 중 하나로 준다. 지금: {ARM!r}")
TEST = os.environ.get("COOP_TEST") == "1"
SIGN = os.environ.get("CTRL_SIGN", "random")
if SIGN != "random" and not TEST:
    sys.exit("멈춤: CTRL_SIGN은 시험(COOP_TEST=1)에서만 쓴다")
TRAIN = [0, 1] if TEST else list(range(7700, 7730))
EVAL = [2, 3] if TEST else list(range(9920, 9950))
NT, NE = (4, 3) if TEST else (3000, 200)
EVERY = 2 if TEST else 100
BRAIN = "configs/brain_kph_v1.yaml"
K, Q, ETA_PH = 10.0, 0.0, 10.0 / (0.00081 * 3000)
XI_SEED = 20261119
ROOT = Path(os.environ.get("COOP_ROOT", "runs"))
OUTS = {a: ROOT / f"phero_confirm_v1_{a}" for a in ("learn", "ctrl")}


def report() -> None:
    files = {a: OUTS[a] / "results.json" for a in OUTS}
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        sys.exit(f"멈춤: 결과 파일이 없다: {missing}")
    out = ROOT / "phero_confirm_v1" / "report.json"
    if out.exists():
        sys.exit(f"멈춤: {out}가 이미 있다")
    R = {a: json.loads(p.read_text(encoding="utf-8")) for a, p in files.items()}
    for key in ("train_seeds", "eval_seeds", "n_train", "n_eval", "eta_ph", "q"):
        assert R["learn"][key] == R["ctrl"][key], key
    L, C, S = R["learn"]["eval"]["learned"], R["ctrl"]["eval"]["learned"], R["learn"]["eval"]["solo"]
    rep = {"comparisons": {}}
    print(f"확인 실행 (41.15). 학습 시드 {TRAIN[0]} ~ {TRAIN[-1]}, 평가 시드 {EVAL[0]} ~ {EVAL[-1]}, 95% CI, * = 0 제외")
    for a, r in (("learn", R["learn"]), ("ctrl", R["ctrl"])):
        ci = r["kph_last_ci"]
        print(f"  {a:5s}: 판 {NT}의 k̄_ph {ci['mean']:+.2f} [{ci['lo']:+.2f}, {ci['hi']:+.2f}], "
              f"평가 W {np.mean(r['eval']['learned']['W']):.3f}, 점수 {np.mean(r['eval']['learned']['score']):.1f}")
    print(f"  solo : 평가 W {np.mean(S['W']):.3f}, 점수 {np.mean(S['score']):.1f}")
    for name, x, y in (("learn − ctrl", L, C), ("learn − solo", L, S), ("ctrl − solo", C, S)):
        rep["comparisons"][name] = {}
        for key, nd in (("W", 3), ("score", 1), ("Nc", 3), ("Ne", 3)):
            c = fe.paired_ci(x[key], y[key], 0.95)
            rep["comparisons"][name][key] = c
            print(f"  {name} {key}: {c['mean']:+.{nd}f} [{c['lo']:+.{nd}f}, {c['hi']:+.{nd}f}]{' *' if c['excludes_zero'] else ''}")
    kd = fe.paired_ci(R["learn"]["kph_last_seed_mean"], R["ctrl"]["kph_last_seed_mean"], 0.95)
    rep["kph_learn_minus_ctrl"] = kd
    print(f"  k̄_ph learn − ctrl (학습 시드 짝): {kd['mean']:+.2f} [{kd['lo']:+.2f}, {kd['hi']:+.2f}]{' *' if kd['excludes_zero'] else ''}")
    w = rep["comparisons"]["learn − ctrl"]["W"]
    rep["verdict"] = bool(w["lo"] > 0)
    print("  판정 (41.15): " + ("충족: 보상으로 배운 초파리가 보상과 무관하게 움직인 초파리보다 더 자주 이긴다" if rep["verdict"]
                           else "미충족: 보상으로 배운 초파리와 막걸음 초파리의 승률을 구별하지 못했다"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float) + "\n", encoding="utf-8")
    print(f"기록: {out}")


if ARM == "report":
    report()
    sys.exit(0)

from flypac.core.config import Config, load_config  # noqa: E402
from flypac.core.runner import build_runner  # noqa: E402
from flypac.recording.recorder import Recorder, make_run_dir  # noqa: E402
from flypac.recording.weights import read_weight_file  # noqa: E402

OUT = OUTS[ARM]
if (OUT / "results.json").exists():
    sys.exit(f"멈춤: {OUT / 'results.json'}가 이미 있다")
if not TEST:
    fe.check_threads()
OUT.mkdir(parents=True, exist_ok=True)

brain = load_config(REPO / BRAIN).to_dict()
assert brain["instinct_gain"]["eta_k"] == 0.0 and brain["instinct_gain"]["k_init"] == K and brain["instinct_gain"]["freeze_synapses"]
brain["phero_gain"].update(learn=True, dry_run=False, eta_ph=ETA_PH, kph_init=0.0)
brain_path = (OUT / "brain_kph_confirm.yaml").resolve()
brain_path.write_text("# phero_confirm.py가 만든 뇌 설정: configs/brain_kph_v1.yaml에 phero_gain.learn: true, dry_run: false, "
                      f"eta_ph: {ETA_PH!r}, kph_init: 0.0을 덮어씀 (DECISIONS 41.15)\n"
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
tr.update(experiment=f"phero_confirm_v1_{ARM}_train", seeds=TRAIN, n_games=NT, condition="regime3", evaluate=False,
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
counts = [0, 0]
if ARM == "ctrl":
    xi_rng = np.random.default_rng(XI_SEED)
    orig_learn = g._learn

    def learn(b, f):
        """k_ph 갱신만 ξ (R_n − R̄)로 하고, R̄ 갱신은 원래 _learn에 맡긴다 (그동안 η_ph = 0)."""
        xi = 1.0 if SIGN == "plus" else (1.0 if xi_rng.random() < 0.5 else -1.0)
        counts[xi < 0] += 1
        k = g.kph_gain[b, f] + ETA_PH * xi * (g.Rn[b, f] - g.rbar[b, f]) * g.e_ph[b, f]
        g.kph_gain[b, f] = min(max(k, g.kph_min), g.kph_max)
        g.eta_ph = 0.0
        try:
            orig_learn(b, f)
        finally:
            g.eta_ph = ETA_PH

    g._learn = learn
runner.run()
assert not g.dw.any() and (g.k_gain == K).all()
learned = g.kph_gain.copy()
print(f"[학습 {ARM}] 끝: {train_dir}" + (f"  (ξ: +1 {counts[0]}번, −1 {counts[1]}번)" if ARM == "ctrl" else ""), flush=True)

curve = {}
for p in sorted((train_dir / "weights").glob("s*_g*.npz")):
    w = read_weight_file(p)
    curve.setdefault(int(w["seed"]), {})[int(w["game_no"])] = float(np.mean(w["kph_gain"]))
games = sorted(next(iter(curve.values())))
last = [curve[s][NT] for s in TRAIN]
assert np.allclose(last, learned.mean(axis=1))


# ---- 평가 ----
def evaluate(name, kph):
    d = copy.deepcopy(base)
    d.update(seeds=EVAL, n_games=NE, experiment=f"phero_confirm_v1_{name}")
    d["run"] = {"max_loop_ticks": NE * 20001}
    d["replay"] = {"keep_games": [1, 2, 3, 4, 5], "brain": True}
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


res = {"arm": ARM, "q": Q, "eta_ph": ETA_PH, "train_seeds": TRAIN, "eval_seeds": EVAL, "n_train": NT, "n_eval": NE,
       "train_dir": train_dir.name, "kph_final": learned.tolist(), "kph_curve": {str(s): curve[s] for s in TRAIN},
       "kph_last_seed_mean": last, "kph_last_ci": fe.paired_ci(last, [0.0] * len(last), 0.95), "eval": {}}
if ARM == "ctrl":
    res["xi"] = {"seed": XI_SEED, "sign": SIGN, "plus": counts[0], "minus": counts[1]}
res["eval"]["learned"] = evaluate(f"{ARM}_learned", learned)
if ARM == "learn":
    res["eval"]["solo"] = evaluate("solo", 0.0)

ci = res["kph_last_ci"]
step = max(1, len(games) // 10)
print(f"\n{ARM} (q = {Q:g}, η_ph = {ETA_PH:.4f}, 학습 시드 {TRAIN[0]} ~ {TRAIN[-1]}, {NT}판)")
print("  k̄_ph (시드 평균): " + ", ".join(f"판 {n}: {np.mean([curve[s][n] for s in TRAIN]):+.2f}" for n in games[step - 1::step]))
print(f"  판 {NT}의 k̄_ph: {ci['mean']:+.2f} [{ci['lo']:+.2f}, {ci['hi']:+.2f}]{' *' if ci['excludes_zero'] else ''}, "
      f"초파리별 {learned.min():+.2f} ~ {learned.max():+.2f}")
for name, e in res["eval"].items():
    print(f"  평가 {name}: W {np.mean(e['W']):.3f}, 점수 {np.mean(e['score']):.1f}, 포획/판 {np.mean(e['Nc']):.3f}, "
          f"먹힘/판 {np.mean(e['Ne']):.3f}, 중단 {sum(e['censored'])}")
(OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float) + "\n", encoding="utf-8")
print(f"기록: {OUT / 'results.json'}")
