"""결정 단위 분석: 버섯체는 "보이면 따라가기 · 겁먹으면 피하기 · 안 보이면 냄새"를 배웠는가 (DECISIONS 42.11, 실행 전 고정).

위치. 42.7(vision_dev.py)의 판정은 승률 W다. 관리자의 질문은 행동이다: "보이면 따라가고, 모퉁이로 사라지면 다시 후각".
승률로는 이를 직접 볼 수 없으므로, 학습이 끝난 가중치로 평가 판을 돌리며 갈림길의 정책 π를 읽는다.
학습 결과(판정)를 본 뒤가 아니라, 학습이 도는 동안(2026-11-19 19시) 규칙을 고정했다.

방법. 42.7 두 팔(vision · novision)의 판 3000 가중치(학습 시드 6300 ~ 6309, 칸 i = 6300 + i)를 평가 모드(η = 0)로 불러
평가 시드 9720 ~ 9729에서 칸마다 50판을 돈다. 뇌 v2 · 본능 끔, 시각 설정은 그 팔의 학습과 같다 (vision: R = 3, s = s_v; novision: 끔).
뇌의 propose를 감시(spy)해 결정마다 기록한다 (행동은 바꾸지 않는다):
  π: 그 결정의 정책 (뇌가 계산한 last_pi), legal: 갈 수 있는 방향, z. 엔진이 결정으로 받아들인(committed) 제안만 센다,
  seen: 팩맨이 보이는 방향 (sense.see, R = 3. novision 팔도 같은 규칙으로 계산만 한다: 뇌는 받지 않는다),
  odor: 냄새 입력 (자기중심).
지표 (결정마다 정책 확률로 계산, 표본 추출의 잡음을 피한다):
  보임 결정 = seen ∩ legal이 비지 않고 legal 전부도 아님.
    P_seen = Σ_{a∈seen∩legal} π_a,   우연 = |seen∩legal| / |legal|   (Δw = 0, g = 0이면 π는 legal 위 균등이라 정확히 우연이다)
  안 보임 결정 = seen ∩ legal이 비었고, legal 안의 냄새 최댓값이 하나뿐이며 0보다 크다.
    P_odor = π_{argmax odor},  우연 = 1 / |legal|
  시드(칸)마다 결정 평균의 (P − 우연)을 내고, 시드 10개로 t 95% 신뢰구간 (자유도 9). 결정이 5개 미만인 칸은 뺀다 (보고).
판정 (결과 전 고정):
  B1 vision 팔, z = 0, 보임: (P_seen − 우연) 0 제외 양수 → "보이면 그쪽으로 가기를 배웠다".
  B2 vision 팔, z = 1, 보임: (P_seen − 우연) 0 제외 음수 → "겁먹으면 보이는 쪽을 피하기를 배웠다".
  B3 vision 팔, z = 0, 안 보임: (P_odor − 우연) 0 제외 양수 → "안 보이면 냄새 쪽을 배웠다".
  보고: novision 팔의 같은 세 지표 (보지 못하므로 B1 꼴의 값은 냄새를 거쳐 생긴 것), 두 팔의 차이(짝: 같은 칸 번호).
미리 적은 예측 (Claude): B1 충족, B3 충족, B2는 불확실 (겁먹은 결정이 드물다). 틀려도 그대로 적는다.
다중 비교 보정은 하지 않는다. 이 자료(9720대)는 다른 판정에 쓰지 않는다.

  vision_dev.py의 두 팔이 끝난 뒤, 저장소 루트에서 (약 20분 추정):
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --extra pacman python scripts/explore/vision_behavior.py

COOP_TEST=1이면 vision_dev.py 시험 모드의 결과(COOP_ROOT)를 쓰고, 평가 시드 6 · 7, 칸마다 2판, 규칙 팩맨.
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
from flypac.recording.weights import read_weight_file  # noqa: E402
from flypac.senses.sense import see  # noqa: E402

TEST = os.environ.get("COOP_TEST") == "1"
EVAL = [6, 7] if TEST else list(range(9720, 9730))
N = 2 if TEST else 50
R_SEE = 3
MIN_DEC = 5
BRAIN = os.environ.get("COOP_BRAIN", "configs/brain_v2_nolh.yaml") if TEST else "configs/brain_v2_nolh.yaml"
ROOT = Path(os.environ.get("COOP_ROOT", REPO / "runs"))
OUT = ROOT / "vision_behavior_v1"
if (OUT / "results.json").exists():
    sys.exit(f"멈춤: {OUT / 'results.json'}가 이미 있다")
if not TEST:
    fe.check_threads()

base = load_config(REPO / "configs" / "final_eval_v1.yaml").to_dict()
if TEST:
    base["pacman"] = {"kind": "rule_based"}; base["rule_based"] = {"danger_dist": 6, "chase_dist": 8}


def arm_info(arm):
    r = json.loads((ROOT / f"vision_dev_v1_{arm}" / "results.json").read_text(encoding="utf-8"))
    return r


def run_arm(arm):
    info = arm_info(arm)
    train_dir = ROOT / info["train_dir"]
    seeds_tr = [int(s) for s in info["train_seeds"]]
    if len(seeds_tr) != len(EVAL):
        sys.exit(f"멈춤: 학습 시드 {len(seeds_tr)}개와 평가 시드 {len(EVAL)}개가 다르다")
    w = fe.collect_weights([train_dir], ["regime1"], seeds_tr, train_n_games=int(info["n_train"]))
    vis = {"R": int(info["R"]), "s": float(info["s_v"])} if int(info["R"]) > 0 else None
    d = copy.deepcopy(base)
    d.update(seeds=EVAL, n_games=N, experiment=f"vision_behavior_v1_{arm}")
    d["run"] = {"max_loop_ticks": N * 20001}
    d["replay"] = {"keep_games": [], "brain": False}
    d["final_eval"]["ghosts"]["flybrain"] = {"kind": "flybrain", "config": BRAIN}
    if vis is not None:
        d["vision"] = vis
    cfg = fe.eval_config(d, "regime1")
    rd = make_run_dir(cfg, ROOT)
    r = build_runner(cfg, Recorder(rd, cfg))
    g = r.ghosts
    assert g.evaluate and g.eta == 0.0 and not g.lateral_horn and g.g == 0.0
    files = []
    for i, s in enumerate(seeds_tr):
        hit = w[("regime1", s)]
        g.load_slot_state(i, read_weight_file(hit["path"]))
        files.append(str(hit["path"]))
    rec = []   # (칸, z, kind, P, 우연): 엔진이 결정으로 받아들인(committed) 것만
    pending = {}
    o_prop, o_commit = g.propose, g.commit

    def propose(obs, alive, views=None):
        pending.clear()
        out = o_prop(obs, alive, views)
        vw = r.game.views()
        cells = np.stack([v.fly_cell for v in vw]).astype(np.int64)
        head = np.stack([v.fly_dir for v in vw]).astype(np.int64)
        pac = np.array([v.pac_cell for v in vw], dtype=np.int64)
        seen = see(cells, head, pac, r.map.nbr, R_SEE, 1.0) > 0
        for b in np.flatnonzero(np.asarray(alive, bool)):
            for f in np.flatnonzero(obs.decide[b]):
                legal = np.asarray(obs.legal[b, f], bool)
                pi = g.last_pi[b, f]
                nl = int(legal.sum())
                if nl < 2:
                    continue
                sl = seen[b, f] & legal
                z = int(obs.z[b, f])
                if 0 < sl.sum() < nl:
                    pending[(int(b), int(f))] = (int(b), z, "seen", float(pi[sl].sum()), float(sl.sum() / nl))
                elif sl.sum() == 0:
                    od = np.where(legal, obs.odor[b, f], -np.inf)
                    top = od.max()
                    if top > 0 and int((od == top).sum()) == 1:
                        pending[(int(b), int(f))] = (int(b), z, "odor", float(pi[int(np.argmax(od))]), 1.0 / nl)
        return out

    def commit(committed, active):
        cm = np.asarray(committed, bool)
        for (b, f), row in pending.items():
            if cm[b, f]:
                rec.append(row)
        pending.clear()
        return o_commit(committed, active)
    g.propose, g.commit = propose, commit
    r.run()
    print(f"[{arm}] 끝: {rd} (결정 기록 {len(rec)}개)", flush=True)
    return {"run_dir": rd.name, "files": files, "vision": vis, "rec": rec}


def summarize(rec, z, kind):
    per, counts = [], []
    for b in range(len(EVAL)):
        x = [p - c for (bb, zz, kk, p, c) in rec if bb == b and zz == z and kk == kind]
        counts.append(len(x))
        per.append(float(np.mean(x)) if len(x) >= MIN_DEC else None)
    ok = [v for v in per if v is not None]
    ci = fe.paired_ci(ok, [0.0] * len(ok), 0.95) if len(ok) >= 2 else None
    return {"per_seed": per, "n_decisions": counts, "ci": ci}


OUT.mkdir(parents=True, exist_ok=True)
res = {"eval_seeds": EVAL, "n_games": N, "R_see": R_SEE, "arms": {}}
for arm in ("vision", "novision"):
    a = run_arm(arm)
    res["arms"][arm] = {"run_dir": a["run_dir"], "files": a["files"], "vision": a["vision"],
                        "B1_seen_z0": summarize(a["rec"], 0, "seen"), "B2_seen_z1": summarize(a["rec"], 1, "seen"),
                        "B3_odor_z0": summarize(a["rec"], 0, "odor")}


def txt(s, sign):
    c = s["ci"]
    if c is None:
        return "결정 부족", False
    met = c["excludes_zero"] and (c["mean"] > 0 if sign > 0 else c["mean"] < 0)
    return (f"{c['mean']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]{' *' if c['excludes_zero'] else ''} "
            f"(칸 {c['n']}개, 결정 {sum(s['n_decisions'])}개)"), met


print(f"\n결정 단위 분석 (평가 시드 {EVAL[0]} ~ {EVAL[-1]}, 칸마다 {N}판, 시야 R = {R_SEE}). 값 = 정책 확률 − 우연")
res["judge"] = {}
for key, name, sign in (("B1_seen_z0", "B1 보이면 그쪽 (평소)", +1), ("B2_seen_z1", "B2 보이면 피함 (겁먹음)", -1),
                        ("B3_odor_z0", "B3 안 보이면 냄새 쪽 (평소)", +1)):
    tv, met = txt(res["arms"]["vision"][key], sign)
    tn, _ = txt(res["arms"]["novision"][key], sign)
    res["judge"][key] = bool(met)
    print(f"  {name}\n    vision 팔:   {tv} → {'충족' if met else '미충족'}\n    novision 팔: {tn} (보고)")
(OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float) + "\n", encoding="utf-8")
print(f"기록: {OUT / 'results.json'}")
