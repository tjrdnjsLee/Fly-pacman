"""갈림길 반응 곡선: 본능만(W 0.387)과 배운 버섯체(0.558)는 둘 다 냄새를 따르는데 왜 차이가 큰가 (2026-11-21, 사후 탐색, 실행 전 고정).

위치. 42.15의 "모르는 것"에 답하려는 기술 분석이다. 승률이 아니라 갈림길의 정책 π를 읽는다 (42.11과 같은 감시 방식).
가중치는 42.12 확인 실행의 판 3000 (학습 시드 6400 ~ 6419). 평가 시드 9850 ~ 9869 (다른 곳에 쓰지 않았다, 학습 시드와 위치로 짝),
칸마다 25판, η = 0, 시각 끔, 팩맨은 본 실험과 같다. 행동은 바꾸지 않는다. 엔진이 결정으로 받아들인(committed) 제안만 센다.

팔 (PS_ARMS, 쉼표로 여럿. 프로세스를 나누어 나란히 돌릴 수 있다):
  b0_nolh  뇌 configs/brain_v2_nolh.yaml, Δw = 0 (무작위: π는 갈 수 있는 길 위 균등)
  b0_lh    뇌 configs/brain_v2.yaml, Δw = 0 (본능만, g = 0.03747)
  ctrl     부호 대조군 가중치 (뇌 configs/brain_v2_nolh_ctrl.yaml, 평가에서는 대조가 아무것도 하지 않는다)
  learn    냄새 퍼텐셜 보상 학습 가중치 (뇌 configs/brain_v2_nolh.yaml)
  sparse   v2_sparse.py의 λ = 0 가중치 (그 팔이 끝난 뒤, 보고)
  report   아래 읽기를 계산 (b0_nolh, b0_lh, learn은 필수)
결정마다 남기는 것 (runs/v2_psycho_v1/<팔>.npz): 칸, 초파리, z, 갈 수 있는 길, 냄새 입력 o = ln(1 + c/c0) (자기중심 4), π (4),
  팩맨까지의 미로 거리 d (초파리가 걸을 수 있는 이웃표의 최단 경로 칸 수).

읽기 (결과 전 고정, 2026-11-21. 탐색이며 다중 비교 보정 없음. 값은 시드(칸)마다 결정 평균을 낸 뒤 20개로 t 95% 신뢰구간):
  두 갈래 결정 = 갈 수 있는 길이 정확히 2개이고 두 냄새가 다르다. hi = 냄새가 더 진한 쪽, Δ = o_hi − o_lo > 0, p = π_hi.
  곡선: Δ를 6구간으로 (경계 = b0_nolh 팔, z = 0 두 갈래 Δ의 1/6 ~ 5/6 분위수: 행동과 무관한 미로 · 냄새의 기하),
        구간마다 p − 1/2. z = 0, z = 1 따로.
  S = 6구간 (p − 1/2)의 단순 평균 (구간마다 결정 분포가 팔마다 달라도 같은 무게로 비교하려고). z = 0.
  H1 세기:     S(learn) − S(b0_lh) 짝 차이가 0 제외 양수 → "같은 냄새 차에서 더 확실히 따른다"
  H2 거리:     [S_가까움 − S_멂](learn) − [같은 것](b0_lh)이 0 제외 양수 → "가까울 때 더 확실히 따른다" (경계 = b0_nolh z = 0 두 갈래 d의 중앙값)
  H3 직진 편향: 직진이 갈 수 있는 z = 0 결정(두 갈래 이상)에서 π_직진 − 1/갈래 수, learn − b0_lh가 0 제외 → "냄새와 별개로 방향 편향이 있다" (양쪽 검정, 부호 보고)
  H4 겁먹음:   z = 1 두 갈래에서 mean(p − 1/2), learn − b0_lh가 0 제외 음수 → "겁먹으면 냄새를 덜 따른다 (피한다)"
  보고: 각 팔의 z = 0 · z = 1 곡선과 S, ctrl · sparse의 같은 지표, 결정 수.
미리 적은 예측 (Claude): H4 지지 (개발 실행에서 학습이 본능보다 덜 먹혔다: 1.775 대 1.960). H1 · H2 · H3는 모른다.
  H1에 대한 두 단서는 방향이 엇갈린다: (가) 본능의 설계 목표(두 갈래에서 약 0.61, 18절)는 한쪽에만 냄새가 있는 T자 기준이라
  두 길 모두 냄새가 있는 게임 갈림길에서는 훨씬 약할 수 있다 (H1 쪽). (나) v1 38절에서 본능 세기가 약 6배로 커져도
  W는 그대로였다 (세기만으로는 승률이 오르지 않는다, H1이 지지돼도 0.17의 차이를 다 설명하지 못할 수 있다). 틀려도 그대로 적는다.
이 자료(9850대)는 다른 판정에 쓰지 않는다.

  저장소 루트에서 (두 프로세스로 나란히, 각 30분 안팎 추정):
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  PS_ARMS=b0_nolh,b0_lh uv run --extra pacman python scripts/explore/v2_psycho.py > ~/v2_psycho_a.log 2>&1 &
  PS_ARMS=ctrl,learn    uv run --extra pacman python scripts/explore/v2_psycho.py > ~/v2_psycho_b.log 2>&1 &
  wait; PS_ARMS=report uv run python scripts/explore/v2_psycho.py
  (sparse가 끝나면: PS_ARMS=sparse ... 뒤 report를 다시. 같은 이름의 기록이 있으면 PSY_TAG로 새 이름)

COOP_TEST=1이면 COOP_ROOT의 시험 결과, 평가 시드 6 · 7, 칸마다 2판, 규칙 팩맨, 뇌 COOP_BRAIN · COOP_BRAIN_LH · COOP_BRAIN_CTRL.
"""
import copy
import json
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
import final_eval as fe  # noqa: E402
from flypac.core.config import load_config  # noqa: E402
from flypac.core.runner import build_runner  # noqa: E402
from flypac.recording.recorder import Recorder, make_run_dir  # noqa: E402
from flypac.recording.weights import read_weight_file  # noqa: E402

TEST = os.environ.get("COOP_TEST") == "1"
EVAL = [6, 7] if TEST else list(range(9850, 9870))
N = 2 if TEST else 25
NBIN = 6
FORWARD = 1   # 자기중심 번호: 좌 0, 직진 1, 우 2, 반전 3 (core/directions.py)
ROOT = Path(os.environ.get("COOP_ROOT", REPO / "runs"))
OUT = ROOT / "v2_psycho_v1"
TAG = os.environ.get("PSY_TAG", "")
env = (lambda k, d: os.environ.get(k, d) if TEST else d)  # noqa: E731
BRAINS = {"b0_nolh": env("COOP_BRAIN", "configs/brain_v2_nolh.yaml"),
          "b0_lh": env("COOP_BRAIN_LH", "configs/brain_v2.yaml"),
          "ctrl": env("COOP_BRAIN_CTRL", "configs/brain_v2_nolh_ctrl.yaml"),
          "learn": env("COOP_BRAIN", "configs/brain_v2_nolh.yaml"),
          "sparse": env("COOP_BRAIN", "configs/brain_v2_nolh.yaml")}
SOURCES = {"ctrl": ["vision_confirm_v1_ctrl"], "learn": ["vision_confirm_v1_learn"],
           "sparse": ["v2_sparse_v1_s0", "v2_sparse_v1_s1"]}
ARMS = [a.strip() for a in os.environ.get("PS_ARMS", "").split(",") if a.strip()]
if not ARMS or any(a not in (*BRAINS, "report") for a in ARMS) or ("report" in ARMS and len(ARMS) > 1):
    sys.exit(f"멈춤: PS_ARMS는 {', '.join(BRAINS)} 중 여럿(쉼표) 또는 report 하나다. 지금: {ARMS!r}")


def geodesic(nbr):
    """초파리 이웃표(벽 −1)로 모든 칸 쌍의 최단 경로 칸 수. 닿지 않으면 −1."""
    n = nbr.shape[0]
    D = np.full((n, n), -1, dtype=np.int32)
    for s in range(n):
        D[s, s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in nbr[u]:
                if v >= 0 and D[s, v] < 0:
                    D[s, v] = D[s, u] + 1
                    q.append(v)
    return D


def run_arm(arm):
    path = OUT / f"{arm}.npz"
    if path.exists():
        sys.exit(f"멈춤: {path}가 이미 있다")
    base = load_config(REPO / "configs" / "final_eval_v1.yaml").to_dict()
    if TEST:
        base["pacman"] = {"kind": "rule_based"}; base["rule_based"] = {"danger_dist": 6, "chase_dist": 8}
    files, seeds_tr = [], None
    states = None
    if arm in SOURCES:
        infos = [json.loads((ROOT / s / "results.json").read_text(encoding="utf-8")) for s in SOURCES[arm]]
        seeds_tr = [int(x) for i in infos for x in i["train_seeds"]]
        nts = {int(i["n_train"]) for i in infos}
        if len(nts) != 1:
            sys.exit(f"멈춤: {arm}의 조각마다 n_train이 다르다 {nts}")
        if len(seeds_tr) != len(EVAL):
            sys.exit(f"멈춤: {arm} 학습 시드 {len(seeds_tr)}개와 평가 시드 {len(EVAL)}개가 다르다")
        w = fe.collect_weights([ROOT / i["train_dir"] for i in infos], ["regime1"], seeds_tr, train_n_games=nts.pop())
        states = [read_weight_file(w[("regime1", s)]["path"]) for s in seeds_tr]
        files = [str(w[("regime1", s)]["path"]) for s in seeds_tr]
    cond = "b0" if arm.startswith("b0") else "regime1"
    d = copy.deepcopy(base)
    d.update(seeds=EVAL, n_games=N, experiment=f"v2_psycho_v1_{arm}")
    d["run"] = {"max_loop_ticks": N * 20001}
    d["replay"] = {"keep_games": [], "brain": False}
    d["final_eval"]["ghosts"]["flybrain"] = {"kind": "flybrain", "config": BRAINS[arm]}
    cfg = fe.eval_config(d, cond)
    rd = make_run_dir(cfg, ROOT)
    r = build_runner(cfg, Recorder(rd, cfg))
    g = r.ghosts
    assert g.evaluate and g.eta == 0.0 and not g.dw.any() and r.shaping is None and r.vision_R == 0
    assert g.lateral_horn == (arm == "b0_lh"), "본능은 b0_lh 팔에서만 켜져 있어야 한다"
    if states is not None:
        for i, s in enumerate(states):
            g.load_slot_state(i, s)
    DG = geodesic(np.asarray(r.map.nbr))
    rec = {k: [] for k in ("b", "f", "z", "legal", "odor", "pi", "d")}
    pending = {}
    o_prop, o_commit = g.propose, g.commit

    def propose(obs, alive, views=None):
        pending.clear()
        out = o_prop(obs, alive, views)
        vw = r.game.views()
        for b in np.flatnonzero(np.asarray(alive, bool)):
            fc = np.asarray(vw[b].fly_cell, dtype=np.int64); pc = int(vw[b].pac_cell)
            for f in np.flatnonzero(obs.decide[b]):
                legal = np.asarray(obs.legal[b, f], bool)
                if int(legal.sum()) < 2:
                    continue
                pending[(int(b), int(f))] = (int(b), int(f), int(obs.z[b, f]), legal.copy(),
                                             np.asarray(obs.odor[b, f], np.float64).copy(),
                                             np.asarray(g.last_pi[b, f], np.float64).copy(), int(DG[fc[f], pc]))
        return out

    def commit(committed, active):
        cm = np.asarray(committed, bool)
        for (b, f), row in pending.items():
            if cm[b, f]:
                for k, v in zip(rec, row):
                    rec[k].append(v)
        pending.clear()
        return o_commit(committed, active)
    g.propose, g.commit = propose, commit
    r.run()
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, b=np.array(rec["b"], np.int16), f=np.array(rec["f"], np.int8), z=np.array(rec["z"], np.int8),
                        legal=np.array(rec["legal"], bool).reshape(-1, 4), odor=np.array(rec["odor"], np.float32).reshape(-1, 4),
                        pi=np.array(rec["pi"], np.float32).reshape(-1, 4), d=np.array(rec["d"], np.int16),
                        eval_seeds=np.array(EVAL), run_dir=np.str_(rd.name), files=np.array(files, dtype=str))
    print(f"[{arm}] 끝: {rd} (결정 {len(rec['b'])}개) → {path}", flush=True)


if ARMS != ["report"]:
    if not TEST:
        fe.check_threads()
    for a in ARMS:
        run_arm(a)
    sys.exit(0)

# ---------------------------------------------------------------- report
need = ("b0_nolh", "b0_lh", "learn")
have = [a for a in ("b0_nolh", "b0_lh", "ctrl", "learn", "sparse") if (OUT / f"{a}.npz").exists()]
for a in need:
    if a not in have:
        sys.exit(f"멈춤: {OUT / (a + '.npz')}가 없다")
D = {a: dict(np.load(OUT / f"{a}.npz")) for a in have}
nS = len(EVAL)


def two_way(x, z):
    """두 갈래 · 냄새가 다른 결정: (칸, Δ, p_hi, d)."""
    lg, od, pi = x["legal"], x["odor"].astype(np.float64), x["pi"].astype(np.float64)
    sel = (lg.sum(1) == 2) & (x["z"] == z)
    idx = np.flatnonzero(sel)
    a2 = np.array([np.flatnonzero(lg[i]) for i in idx]).reshape(-1, 2)
    o = od[idx[:, None], a2]; p = pi[idx[:, None], a2]
    keep = o[:, 0] != o[:, 1]
    hi = np.argmax(o, 1)
    delta = np.abs(o[:, 0] - o[:, 1])
    p_hi = p[np.arange(len(p)), hi]
    return x["b"][idx][keep].astype(int), delta[keep], p_hi[keep], x["d"][idx][keep].astype(int)


b0z = two_way(D["b0_nolh"], 0)
EDGES = np.quantile(b0z[1], np.arange(1, NBIN) / NBIN)
D_SPLIT = float(np.median(b0z[3][b0z[3] >= 0]))


def per_seed_mean(b, v, mask=None):
    out = []
    for s in range(nS):
        m = (b == s) if mask is None else (b == s) & mask
        out.append(float(np.mean(v[m])) if m.sum() >= 5 else float("nan"))
    return out


def curve(x, z, dmask=None):
    b, dl, p, dd = two_way(x, z)
    k = np.digitize(dl, EDGES)
    m0 = np.ones_like(b, bool) if dmask is None else dmask(dd)
    per_bin = [per_seed_mean(b, p - 0.5, (k == j) & m0) for j in range(NBIN)]
    counts = [int(((k == j) & m0).sum()) for j in range(NBIN)]
    S = [float(np.nanmean([per_bin[j][s] for j in range(NBIN)])) for s in range(nS)]
    flat = per_seed_mean(b, p - 0.5, m0)
    return {"per_bin": per_bin, "counts": counts, "S": S, "mean": flat, "n": int(m0.sum())}


def fwd_bias(x):
    lg, pi, z = x["legal"], x["pi"], x["z"]   # float32 그대로: 균등 정책이면 정확히 0이 되게
    sel = (z == 0) & lg[:, FORWARD]
    v = (pi[sel, FORWARD] - (1.0 / lg[sel].sum(1)).astype(np.float32)).astype(np.float64)
    return per_seed_mean(x["b"][sel].astype(int), v)


rep = {"edges": EDGES.tolist(), "d_split": D_SPLIT, "eval_seeds": EVAL, "arms": {}}
for a in have:
    x = D[a]
    rep["arms"][a] = {
        "n_decisions": int(len(x["b"])),
        "z0": curve(x, 0), "z1": curve(x, 1),
        "z0_near": curve(x, 0, lambda dd: (dd >= 0) & (dd <= D_SPLIT)),
        "z0_far": curve(x, 0, lambda dd: dd > D_SPLIT),
        "fwd_bias": fwd_bias(x)}


def ok_pairs(A, B):
    i = [k for k in range(nS) if np.isfinite(A[k]) and np.isfinite(B[k])]
    return [A[k] for k in i], [B[k] for k in i]


def pci(A, B):
    A2, B2 = ok_pairs(A, B)
    return fe.paired_ci(A2, B2, 0.95) if len(A2) >= 2 else None


def one(A):
    A2 = [v for v in A if np.isfinite(v)]
    return fe.paired_ci(A2, [0.0] * len(A2), 0.95) if len(A2) >= 2 else None


def fmt(c):
    if c is not None and c["lo"] == 0 and c["hi"] == 0:
        return "0 (정확히, 균등 정책)"
    return "—" if c is None else f"{c['mean']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]{' *' if c['excludes_zero'] else ''}"


R = rep["arms"]
print(f"갈림길 반응 곡선 (평가 시드 {EVAL[0]} ~ {EVAL[-1]}, 칸마다 {N}판). Δ 경계 {np.round(EDGES, 4).tolist()}, "
      f"거리 경계 {D_SPLIT:g}칸")
print("\n  z = 0 두 갈래, 구간마다 p_hi − 1/2 (시드 평균)")
for a in have:
    vals = " ".join(f"{np.nanmean(R[a]['z0']['per_bin'][j]):+.3f}" for j in range(NBIN))
    print(f"    {a:<8} {vals}   S {fmt(one(R[a]['z0']['S']))}   결정 {R[a]['z0']['n']}")
print("  z = 1 두 갈래, mean(p_hi − 1/2)")
for a in have:
    print(f"    {a:<8} {fmt(one(R[a]['z1']['mean']))}   결정 {R[a]['z1']['n']}")
nf = lambda a: [n - f for n, f in zip(R[a]["z0_near"]["S"], R[a]["z0_far"]["S"])]  # noqa: E731
H = {}
H["H1"] = pci(R["learn"]["z0"]["S"], R["b0_lh"]["z0"]["S"])
H["H2"] = pci(nf("learn"), nf("b0_lh"))
H["H3"] = pci(R["learn"]["fwd_bias"], R["b0_lh"]["fwd_bias"])
H["H4"] = pci(R["learn"]["z1"]["mean"], R["b0_lh"]["z1"]["mean"])
want = {"H1": +1, "H2": +1, "H3": 0, "H4": -1}
name = {"H1": "세기: 같은 냄새 차에서 더 확실히 따른다", "H2": "거리: 가까울 때 더 확실히 따른다",
        "H3": "직진 편향: 냄새와 별개의 방향 편향", "H4": "겁먹음: 겁먹으면 냄새를 덜 따른다"}
print("\n  learn − b0_lh (본능만)")
rep["H"] = {}
for k in ("H1", "H2", "H3", "H4"):
    c = H[k]
    met = bool(c is not None and c["excludes_zero"] and (want[k] == 0 or np.sign(c["mean"]) == want[k]))
    rep["H"][k] = {"ci": c, "supported": met}
    print(f"    {k} {name[k]}: {fmt(c)} → {'지지' if met else '지지 안 함'}")
print("\n  보고: 직진 편향 π_직진 − 1/갈래 (z = 0)")
for a in have:
    print(f"    {a:<8} {fmt(one(R[a]['fwd_bias']))}")
print("  보고: S 가까움 / 멂 (z = 0)")
for a in have:
    print(f"    {a:<8} {np.nanmean(R[a]['z0_near']['S']):+.3f} / {np.nanmean(R[a]['z0_far']['S']):+.3f}")
for a in ("ctrl", "sparse"):
    if a in have:
        rep[f"learn_minus_{a}_S"] = pci(R["learn"]["z0"]["S"], R[a]["z0"]["S"])
        print(f"  보고: learn − {a} S: {fmt(rep[f'learn_minus_{a}_S'])}")
p = ROOT / (f"v2_psycho_v1_report_{TAG}.json" if TAG else "v2_psycho_v1_report.json")
if p.exists():
    sys.exit(f"멈춤: {p}가 이미 있다 (PSY_TAG로 새 이름)")
p.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float) + "\n", encoding="utf-8")
print(f"기록: {p}")
