"""산개 거리의 측정 (D§40.3). 기준은 이 스크립트와 함께 측정 전에 저장소에 넣었다. 병합 뒤 한 번만 돌린다.

질문 (D§39.5가 재지 않았다고 적은 것): 동료가 있는 길을 피하는 본능(spread10)에서 초파리 사이 거리가 실제로 벌어졌는가.
새 게임 실행은 없다. spread_v2의 남은 재생만 읽는다.

자료: runs/spread_v2/results.json의 solo · spread10 실행 폴더, 평가 시드 9700 ~ 9709, 판 1 ~ 5 (keep_games), 조건마다 50판.
거리 d(i, j): 유령 집과 문을 뺀 걸을 수 있는 칸의 그래프에서의 최단 걸음 수 (터널 연결 포함, 팩맨의 이동 그래프와 같다,
              flypac.maps.original). 너비 우선 탐색으로 모든 칸 쌍을 미리 잰다.
틱의 값 D_t: 집 밖(fly_in_house = 거짓) 초파리가 2마리 이상인 틱에서 그 초파리 쌍들의 d 평균. 2마리 미만인 틱은 뺀다.
판의 값: 그 판에서 뺀 틱을 제외한 D_t의 평균. 시드의 값 D̄_s: 판 1 ~ 5 값의 단순 평균.
판정: Δ_s = D̄_s(spread10) − D̄_s(solo), 시드 10쌍, t 95% 신뢰구간 (자유도 9, final_eval.paired_ci).
      0을 제외하고 양수이면 "동료를 피하는 본능으로 초파리가 실제로 흩어졌다". 포위는 재지 않는다.
함께 보고 (판정 아님): 조건별 D̄의 평균, 판마다 유효 틱의 비율, 유효 틱의 집 밖 초파리 수 평균 (각 시드 평균의 짝 차이).

  저장소 루트에서:  uv run python - < scripts/explore/spread_distance.py
  CHECK=1 이면 거리표만 만들고 검사한 뒤 끝낸다 (runs/를 읽지 않는다).
"""
import sys, json, os
from collections import deque
from itertools import combinations
from pathlib import Path
import numpy as np
sys.path[:0] = ["src", "scripts"]
from flypac.maps.original import load_original

CHECK = os.environ.get("CHECK") == "1"
ROOT = Path(os.environ.get("DIST_ROOT", "runs"))
SRC = ROOT / "spread_v2" / "results.json"
OUT = ROOT / "spread_distance_v1"
SEEDS = list(range(9700, 9710))
GAMES = [1, 2, 3, 4, 5]
CONDS = ("solo", "spread10")


def stop(msg):
    sys.exit(f"멈춤: {msg}")


def distance_table(m):
    """(N, N) 최단 걸음 수. 집 · 문 칸이 끼는 행 · 열은 -1."""
    n = m.nbr.shape[0]
    house = np.zeros(n, dtype=bool); house[m.house_cells] = True; house[m.door_cells] = True
    dist = np.full((n, n), -1, dtype=np.int32)
    for src in np.flatnonzero(~house):
        dist[src, src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            for v in m.nbr[u]:
                if v >= 0 and not house[v] and dist[src, v] < 0:
                    dist[src, v] = dist[src, u] + 1
                    q.append(v)
    return dist, house


m = load_original()
DIST, HOUSE = distance_table(m)
out_cells = np.flatnonzero(~HOUSE)
sub = DIST[np.ix_(out_cells, out_cells)]
if (sub < 0).any():
    stop("집 밖 칸의 그래프가 이어져 있지 않다")
if not (sub == sub.T).all():
    stop("거리표가 대칭이 아니다")
tun = sorted(int(c) for c in m.tunnel_cells)
row = {}
for c in tun:
    row.setdefault(int(m.cell_rc[c][0]), []).append(c)
ends = [(min(cs, key=lambda c: m.cell_rc[c][1]), max(cs, key=lambda c: m.cell_rc[c][1])) for cs in row.values()]
print(f"거리표: 집 밖 칸 {len(out_cells)}개, 최대 거리 {int(sub.max())}, 터널 양 끝 거리 {[int(DIST[a, b]) for a, b in ends]}")
if any(DIST[a, b] != 1 for a, b in ends):
    stop("터널의 양 끝이 한 걸음으로 이어지지 않는다")
if CHECK:
    sys.exit(0)

import final_eval as fe

if (OUT / "results.json").exists():
    stop(f"{OUT / 'results.json'}가 이미 있다")
src = json.loads(SRC.read_text(encoding="utf-8"))
if [int(s) for s in src["seeds"]] != SEEDS:
    stop(f"spread_v2의 시드가 {SEEDS}가 아니다: {src['seeds']}")


def game_value(path):
    r = np.load(path)
    cell = np.asarray(r["fly_cell"], dtype=np.int64)
    inh = np.asarray(r["fly_in_house"], dtype=bool)
    if cell.shape != inh.shape or cell.ndim != 2 or cell.shape[1] != 4:
        stop(f"{path}: fly_cell · fly_in_house 모양이 (T, 4)가 아니다")
    vals, counts = [], []
    for t in range(cell.shape[0]):
        cs = cell[t][~inh[t]]
        if len(cs) < 2:
            continue
        if HOUSE[cs].any():
            stop(f"{path}: 틱 {t}에 집 밖이라는데 집 · 문 칸에 있는 초파리가 있다")
        vals.append(float(np.mean([DIST[i, j] for i, j in combinations(cs, 2)])))
        counts.append(len(cs))
    if not vals:
        stop(f"{path}: 유효한 틱이 없다")
    return {"D": float(np.mean(vals)), "ticks": int(cell.shape[0]), "valid": len(vals),
            "valid_frac": len(vals) / cell.shape[0], "out_flies": float(np.mean(counts))}


res = {"rule": "D§40.3", "seeds": SEEDS, "games": GAMES, "conditions": {}}
for c in CONDS:
    rdir = ROOT / src["conditions"][c]["run_dir"] / "replay"
    per_seed = {"D": [], "valid_frac": [], "out_flies": []}
    games = {}
    for s in SEEDS:
        gv = []
        for g in GAMES:
            p = rdir / f"regime3_s{s}_g{g}.npz"
            if not p.exists():
                stop(f"재생 파일이 없다: {p}")
            v = game_value(p); gv.append(v); games[f"s{s}_g{g}"] = v
        for k in per_seed:
            per_seed[k].append(float(np.mean([v[k] for v in gv])))
    res["conditions"][c] = {"run_dir": src["conditions"][c]["run_dir"], "per_seed": per_seed, "games": games}
    print(f"[{c}] 시드 평균 D̄: " + ", ".join(f"{x:.2f}" for x in per_seed["D"]), flush=True)

S, P = res["conditions"]["spread10"]["per_seed"], res["conditions"]["solo"]["per_seed"]
ci = fe.paired_ci(S["D"], P["D"], 0.95)
res["judgement"] = {"D": ci, "dispersed": bool(ci["lo"] > 0)}
res["also"] = {k: fe.paired_ci(S[k], P[k], 0.95) for k in ("valid_frac", "out_flies")}
print(f"\n산개 거리 (평가 시드 {SEEDS[0]} ~ {SEEDS[-1]}, 판 {GAMES[0]} ~ {GAMES[-1]}). 95% CI")
print(f"  solo D̄ {np.mean(P['D']):.3f}, spread10 D̄ {np.mean(S['D']):.3f}")
print(f"  spread10 − solo: D {ci['mean']:+.3f} [{ci['lo']:+.3f}, {ci['hi']:+.3f}]  → 기준 충족: {'예' if ci['lo'] > 0 else '아니오'}")
for k, name in (("valid_frac", "유효 틱 비율"), ("out_flies", "집 밖 초파리 수")):
    a = res["also"][k]
    print(f"  (판정 아님) {name}: solo {np.mean(P[k]):.3f}, spread10 {np.mean(S[k]):.3f}, "
          f"차이 {a['mean']:+.3f} [{a['lo']:+.3f}, {a['hi']:+.3f}]")
print("  시드별 Δ: " + ", ".join(f"{d:+.3f}" for d in ci["deltas"]))
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float) + "\n", encoding="utf-8")
print(f"기록: {OUT / 'results.json'}")
