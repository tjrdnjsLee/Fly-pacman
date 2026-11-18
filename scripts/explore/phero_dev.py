"""개발 실행: 초파리가 페로몬을 피하는 세기 k_ph를 보상으로 배우는가 (2026-11-17, 탐색 · 개발, 실행 전 고정. 조건마다 한 번).

위치. 관리자의 방향(2026-11-17): 체제를 비교하지 않고 LLM(Claude)이 결과를 보며 가장 좋은 학습 조건을 찾는다.
이 실행은 그 첫 개발 실행이다. 개발 자료(학습 7600 ~ 7609, 평가 9900 ~ 9909)는 확인의 근거로 쓰지 않는다 (41.6).

근거 (지형 측정 phero_land_v1, 시드 9820 ~ 9829, 2026-11-17).
  W: k_ph 0 → −1 → −3 → −10 → −30에서 0.396, 0.377, 0.513, 0.625, 0.646.
  q = 0 (보상 = 활성 후 포획)의 최댓점은 −10, q = 1의 최댓점은 −1 (q = 1은 회피를 벌한다).
  한 마리 기울기 예측 (팀 기울기 / 16) q = 0: −0.00081 [−0.00161, −0.00002], 관문 5 관측 −0.00085 ± 0.00162와 맞음.
학습 조건 (Claude의 선택, 결과 전 고정):
  - 보상: 체제 3의 식에서 q = 0 (PH_Q=0)과 대조 q = 1 (PH_Q=1). 같은 스크립트를 두 프로세스로 나란히 돌린다.
  - η_ph = 10 / (0.00081 × 3000) = 4.115 (41.5의 식에 관문 5 대신 지형의 유의한 기울기를 넣음. 3000판의 기대 표류 −10).
  - 그 밖: 시냅스 얼림, k = 10 고정(η_k = 0), k_ph는 0에서 시작, 범위 [−50, 50], marker, view others, 초파리마다 따로 배운다.
학습: 학습 시드 7600 ~ 7609, 칸마다 3000판, 가중치 100판마다 (k_ph 곡선), 재생 판 1 · 1000 · 2000 · 3000.
평가: 평가 시드 9900 ~ 9909 (학습 시드와 위치로 짝), 칸마다 200판, 평가 모드에 학습 끝의 k_ph를 넣는다.
      PH_Q=0 프로세스만 solo(k_ph = 0)도 같은 시드로 평가한다 (기준).
미리 적는 예측 (판정이 아니다): q = 0에서 판 3000의 k̄_ph < 0 (시드 10개 평균의 95% 신뢰구간이 0 제외),
  q = 1에서 k̄_ph는 0을 포함. q = 0의 학습 − solo의 W 짝 차이 > 0. 한 마리의 막걸음 폭은 약 ±16으로 크다.
확인에 올릴 조건 (41.6): 개발 평가의 W 평균이 더 높은 쪽. q = 0이 q = 1보다 높지 않으면 이 설계는 확인에 올리지 않고
  다음 설계(공유 본능 등)로 간다.

  저장소 루트에서 (두 프로세스를 나란히, 끝나면 각각 화면 출력을 ~/phero_dev_q0.log, ~/phero_dev_q1.log에서 본다):
  D=$(ls ~/다운로드/phero_dev.py 2>/dev/null | head -1); echo $D; export D
  systemd-inhibit --what=sleep:idle bash -c 'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1;
    PH_Q=0 uv run --extra pacman python - < "$D" > ~/phero_dev_q0.log 2>&1 &
    PH_Q=1 uv run --extra pacman python - < "$D" > ~/phero_dev_q1.log 2>&1 & wait'

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

Q = os.environ.get("PH_Q")
if Q not in ("0", "1"):
    sys.exit(f"멈춤: 환경변수 PH_Q를 0 또는 1로 준다. 지금: {Q!r}")
Q = float(Q)
TEST = os.environ.get("COOP_TEST") == "1"
TRAIN = [0, 1] if TEST else list(range(7600, 7610))
EVAL = [2, 3] if TEST else list(range(9900, 9910))
NT, NE = (4, 3) if TEST else (3000, 200)
EVERY = 2 if TEST else 100
BRAIN = "configs/brain_kph_v1.yaml"
K, ETA_PH = 10.0, 10.0 / (0.00081 * 3000)
ROOT = Path(os.environ.get("COOP_ROOT", "runs")); OUT = ROOT / f"phero_dev_v1_q{Q:g}"
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
tr.update(experiment=f"phero_dev_v1_q{Q:g}_train", seeds=TRAIN, n_games=NT, condition="regime3", evaluate=False,
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
runner.run()
assert not g.dw.any() and (g.k_gain == K).all()
learned = g.kph_gain.copy()
print(f"[학습 q = {Q:g}] 끝: {train_dir}", flush=True)

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
res["eval"][f"learned_q{Q:g}"] = evaluate(f"learned_q{Q:g}", learned)
if Q == 0.0:
    res["eval"]["solo"] = evaluate("solo", 0.0)

print(f"\n개발 실행 (q = {Q:g}, η_ph = {ETA_PH:.4f}, 학습 시드 {TRAIN[0]} ~ {TRAIN[-1]}, {NT}판)")
step = max(1, len(games) // 10)
print("  k̄_ph (시드 평균): " + ", ".join(f"판 {n}: {np.mean([curve[s][n] for s in TRAIN]):+.2f}" for n in games[step - 1::step]))
ci = res["kph_last_ci"]
print(f"  판 {NT}의 k̄_ph: {ci['mean']:+.2f} [{ci['lo']:+.2f}, {ci['hi']:+.2f}]{' * (0 제외)' if ci['excludes_zero'] else ''}")
print(f"  초파리별 k_ph 범위: {learned.min():+.2f} ~ {learned.max():+.2f}, 시드별 평균: " + ", ".join(f"{v:+.1f}" for v in last))
L = res["eval"][f"learned_q{Q:g}"]
print(f"  평가 (시드 {EVAL[0]} ~ {EVAL[-1]}, {NE}판): 학습한 초파리 W {np.mean(L['W']):.3f}, 점수 {np.mean(L['score']):.1f}, "
      f"포획/판 {np.mean(L['Nc']):.3f}, 먹힘/판 {np.mean(L['Ne']):.3f}, 중단 {sum(L['censored'])}")
if "solo" in res["eval"]:
    S = res["eval"]["solo"]
    print(f"  solo: W {np.mean(S['W']):.3f}, 점수 {np.mean(S['score']):.1f}")
    for key, nd in (("W", 3), ("score", 1), ("Nc", 3), ("Ne", 3)):
        c = fe.paired_ci(L[key], S[key], 0.95)
        print(f"  학습 − solo {key}: {c['mean']:+.{nd}f} [{c['lo']:+.{nd}f}, {c['hi']:+.{nd}f}]{' *' if c['excludes_zero'] else ''}")
    print("  (q = 1의 학습 − solo는 두 결과 파일로 계산한다: q = 1 프로세스는 solo를 돌리지 않는다)")
(OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float) + "\n", encoding="utf-8")
print(f"기록: {OUT / 'results.json'}")
