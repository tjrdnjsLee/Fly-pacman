"""표현력 시험 (M1 뇌 ② 단계 3). 규칙은 DECISIONS 13절(지시서 4절, 동결)과 13.10(구현 해석)을 따른다.

뇌 가중치는 고정하고 KC 표현 k를 만든 뒤 진단용 선형 분류기(scikit-learn)만 학습한다.
이 모듈에는 결과를 본 뒤 바꿀 수 있는 선택지가 없다: 분할, 합격선, 격자, 분류기 설정은 모두 상수다.

구성: 자료 수집(Collector) → 시험별 표본(Probe) → KC 특징 → 분류기 → 조합 하나의 결과 → 진단 → 보조 ②.
"""

from __future__ import annotations

import hashlib
import time
import math
import warnings
from dataclasses import dataclass, field

import numpy as np

from flypac.brain import assign as A
from flypac.brain.flybrain import Circuit, kc_drive, kc_select, pn_input
from flypac.core import directions as d
from flypac.core.events import Kind
from flypac.core.rng import make_rng
from flypac.core.runner import NullRecorder

GEN_VERSION = "repr_data_v1"
SPLITS = {"dev_train": (1000, 1013), "dev_eval": (1014, 1019)}  # 닫힌 구간 (13.2)
FINAL_SEEDS = (2000, 2009)                                       # 이 단계에서는 만들지 않는다
RHOS = (0.05, 0.02, 0.10)                                        # 이 순서 (13.6)
CS = (0, 1, 2)                                                   # θ = c · ū_ref, 이 순서
ASSIGN_SEEDS = tuple(range(20))                                  # 배정 시드 0 ~ 19
TOL = 1e-6                                                       # 동률 허용오차 (로그 변환 뒤)
MIN_SAMPLES = 50                                                 # 13.5
# 합법 행동이 둘 이상일 때 가능한 정답 방향: 좌 · 앞 · 우. 가짜 엔진은 반전 금지라 뒤(3)는 |L| ≥ 2에 없다 (13.10 항목 15)
DIRS = (0, 1, 2)
PASS = {"t1": 0.90, "t2": 0.95, "t3": 0.90, "t4": 0.90}         # 13.5
CONTROL_MAX = 0.50
KAPPAS = (0.1, 0.2, 0.5)                                         # 보조 ②
CLF_SETTINGS = dict(solver="lbfgs", C=1.0, class_weight="balanced", max_iter=2000, tol=1e-4,
                    fit_intercept=True, random_state=0)         # 13.4. 입력은 float64, 표준화 없음
F = 4


# =============================================================== 자료 수집

class Collector(NullRecorder):
    """기록기 자리에서 틱마다 초파리별 관측을 모은다 (13.10 항목 2). 칸 b의 시드는 seeds[b].

    on_events가 먼저 오고 replay_frame이 뒤에 온다 (실행기 12번). 초기화 시점: 목숨 끝(LIFE_LOST · GAME_OVER)은
    모든 초파리, EATEN은 그 초파리, 확정(committed)은 그 초파리 — 뇌가 x, h를 비우는 때와 같다 (flybrain.commit,
    on_eaten, reset_life).
    """

    def __init__(self, seeds, nbr):
        self.seeds = [int(s) for s in seeds]
        self.nbr = nbr
        B = len(self.seeds)
        self.game_no = [1] * B
        self.tick = [0] * B
        self._pend = [(False, ()) for _ in range(B)]
        self._cols = [[] for _ in range(B)]

    def on_events(self, game, events, view):
        life_end = any(e.kind in (Kind.LIFE_LOST, Kind.GAME_OVER) for e in events)
        eaten = tuple(e.fly for e in events if e.kind is Kind.EATEN)
        self._pend[game] = (life_end, eaten)

    def replay_frame(self, game, views, obs, odor, phero, brain):
        v = views[game]
        life_end, eaten = self._pend[game]
        reset = np.full(F, life_end, dtype=bool)
        for k in eaten:
            reset[k] = True
        reset |= np.asarray(brain["committed"][game], dtype=bool)
        ego = np.arange(4)
        wall = np.stack([self.nbr[int(v.fly_cell[k]), d.ego_to_abs(int(v.fly_dir[k]), ego)] < 0 for k in range(F)])
        self._cols[game].append((
            self.game_no[game], self.tick[game],
            np.asarray(obs.odor[game], dtype=np.float64).copy(), np.asarray(obs.z[game], dtype=np.float64).copy(),
            np.asarray(obs.legal[game], dtype=bool).copy(), np.asarray(obs.decide[game], dtype=bool).copy(),
            np.asarray(brain["committed"][game], dtype=bool).copy(), bool(v.frozen), wall, reset))
        self.tick[game] += 1

    def replay_end(self, game, keep):
        self.game_no[game] += 1
        self.tick[game] = 0

    def rows(self) -> dict:
        out: dict[str, list] = {k: [] for k in ("seed", "game", "tick", "fly", "odor", "z", "legal", "decide",
                                                "committed", "frozen", "wall", "reset")}
        for b, seed in enumerate(self.seeds):
            cols = self._cols[b]
            if not cols:
                continue
            g = np.array([c[0] for c in cols]); t = np.array([c[1] for c in cols])
            T = len(cols)
            out["seed"].append(np.full(T * F, seed, dtype=np.int64))
            out["game"].append(np.repeat(g, F)); out["tick"].append(np.repeat(t, F))
            out["fly"].append(np.tile(np.arange(F), T))
            out["odor"].append(np.stack([c[2] for c in cols]).reshape(T * F, 4))
            out["z"].append(np.stack([c[3] for c in cols]).reshape(T * F))
            out["legal"].append(np.stack([c[4] for c in cols]).reshape(T * F, 4))
            out["decide"].append(np.stack([c[5] for c in cols]).reshape(T * F))
            out["committed"].append(np.stack([c[6] for c in cols]).reshape(T * F))
            out["frozen"].append(np.repeat(np.array([c[7] for c in cols]), F))
            out["wall"].append(np.stack([c[8] for c in cols]).reshape(T * F, 4))
            out["reset"].append(np.stack([c[9] for c in cols]).reshape(T * F))
        return {k: np.concatenate(v) for k, v in out.items()}


def generate_rows(cfg) -> dict:
    """설정(configs/brain_repr_v1.yaml)대로 실행기를 돌려 모든 시드의 틱별 행을 모은다."""
    from flypac.core.runner import build_runner
    r = build_runner(cfg)
    col = Collector([int(s) for s in cfg.seeds], r.map.nbr)
    r.recorder = col
    r.run()
    return col.rows()


def split_rows(rows: dict, lo: int, hi: int) -> dict:
    m = (rows["seed"] >= lo) & (rows["seed"] <= hi)
    return {k: v[m] for k, v in rows.items()}


def rows_hash(rows: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(rows):
        h.update(k.encode())
        h.update(np.ascontiguousarray(rows[k]).tobytes())
    return h.hexdigest()


# =============================================================== 시험별 표본

@dataclass
class Probe:
    """KC 특징을 만들 입력과 정답. 모든 배열의 앞 축이 표본."""
    odor: np.ndarray
    z: np.ndarray
    phero: np.ndarray
    seed: np.ndarray
    game: np.ndarray
    legal: np.ndarray
    y: np.ndarray
    meta: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.y)


def decision_mask(rows: dict) -> np.ndarray:
    """결정 표본: decide 참, |L| ≥ 2, 동결 아님 (13.10 항목 3)."""
    return rows["decide"] & ~rows["frozen"] & (rows["legal"].sum(1) >= 2)


def _top_two(odor, legal, largest=True):
    o = np.where(legal, odor, -np.inf if largest else np.inf)
    s = np.sort(o, axis=1)
    if largest:
        return s[:, -1], s[:, -2], o.argmax(1)
    return s[:, 0], s[:, 1], o.argmin(1)


def _sub(rows, idx, odor=None, z=None, phero=None, y=None, meta=None) -> Probe:
    n = len(idx)
    meta = dict(meta or {})
    meta.setdefault("src", np.asarray(idx))
    return Probe(rows["odor"][idx] if odor is None else odor,
                 rows["z"][idx] if z is None else z,
                 np.zeros((n, 4)) if phero is None else phero,
                 rows["seed"][idx], rows["game"][idx], rows["legal"][idx], y, meta)


def build_test1(rows: dict):
    """시험 1. 냄새 방향. 반환: (유일 표본 Probe, 동률 표본 Probe(meta['opt']), 제외 통계)."""
    dec = rows["decide"] & ~rows["frozen"]
    cand = np.flatnonzero(decision_mask(rows))
    best, second, arg = _top_two(rows["odor"][cand], rows["legal"][cand])
    zero = best <= 0.0
    unique = (best - second > TOL) & ~zero
    tie = ~zero & ~unique
    stats = {"decide_samples": int(dec.sum()), "lt2": int((dec & (rows["legal"].sum(1) < 2)).sum()),
             "candidates": int(len(cand)), "zero_input": int(zero.sum()), "tie": int(tie.sum()),
             "unique": int(unique.sum())}
    uniq = _sub(rows, cand[unique], y=arg[unique])
    ti = cand[tie]
    opt = rows["legal"][ti] & (rows["odor"][ti] >= (best[tie] - TOL)[:, None])
    tiep = _sub(rows, ti, y=arg[tie], meta={"opt": opt})
    return uniq, tiep, stats


def build_test2(rows: dict):
    """시험 2. 상태. 같은 관측의 z = 0 ↔ 1 짝. 앞 절반 z = 0, 뒤 절반 z = 1."""
    cand = np.flatnonzero(decision_mask(rows))
    m = len(cand)
    base = _sub(rows, cand)
    z = np.concatenate([np.zeros(m), np.ones(m)])
    p = Probe(np.concatenate([base.odor, base.odor]), z, np.zeros((2 * m, 4)),
              np.concatenate([base.seed, base.seed]), np.concatenate([base.game, base.game]),
              np.concatenate([base.legal, base.legal]), z.astype(np.int64),
              {"pairs": m, "src": np.concatenate([cand, cand])})
    return p, {"candidates": m, "unique": m}


def build_test3(rows: dict):
    """시험 3. 페로몬 채널의 방향 전달. 같은 (L, 벽 마스크)의 다른 판 표본의 냄새를 페로몬으로 넣는다 (13.10 항목 7)."""
    cand = np.flatnonzero(decision_mask(rows))
    key = (rows["legal"][cand] * np.array([1, 2, 4, 8])).sum(1) * 16 + (rows["wall"][cand] * np.array([1, 2, 4, 8])).sum(1)
    sg = rows["seed"][cand].astype(np.int64) * 1000 + rows["game"][cand]
    pools: dict[int, np.ndarray] = {}
    for kv in np.unique(key):
        pools[int(kv)] = np.flatnonzero(key == kv)
    rngs: dict[tuple[int, int], np.random.Generator] = {}
    partner = np.full(len(cand), -1, dtype=np.int64)
    for j in range(len(cand)):
        pool = pools[int(key[j])]
        elig = pool[sg[pool] != sg[j]]
        if elig.size == 0:
            continue
        k2 = (int(rows["seed"][cand[j]]), int(rows["game"][cand[j]]))
        if k2 not in rngs:
            rngs[k2] = make_rng(k2[0], k2[1], "repr_synth")
        partner[j] = elig[rngs[k2].integers(elig.size)]
    has = partner >= 0
    ph = np.zeros((len(cand), 4))
    ph[has] = rows["odor"][cand[partner[has]]]
    legal = rows["legal"][cand]
    best, second, arg = _top_two(ph, legal)
    ok = has & (best > 0.0) & (best - second > TOL)
    ob, osec, oarg = _top_two(rows["odor"][cand], legal)
    own_unique = (ob > 0.0) & (ob - osec > TOL)
    same = np.where(own_unique, np.where(oarg == arg, 1, 0), -1)
    stats = {"candidates": int(len(cand)), "no_partner": int((~has).sum()),
             "no_input_or_tie": int((has & ~ok).sum()), "unique": int(ok.sum())}
    idx = cand[ok]
    return _sub(rows, idx, phero=ph[ok], y=arg[ok], meta={"same": same[ok], "partner": cand[partner[ok]]}), stats


def build_test4(rows: dict):
    """시험 4. 과제 C 결합 통제 자료. 기본 하나에서 네 변형 (원본·평소→A, 원본·겁먹음→B, 교환·평소→B, 교환·겁먹음→A)."""
    cand = np.flatnonzero(decision_mask(rows))
    odor, legal = rows["odor"][cand], rows["legal"][cand]
    best, second, A_ = _top_two(odor, legal, True)
    low, low2, B_ = _top_two(odor, legal, False)
    ok = (best - second > TOL) & (low2 - low > TOL) & (A_ != B_)
    base = cand[ok]
    A_, B_ = A_[ok], B_[ok]
    m = len(base)
    o = rows["odor"][base]
    sw = o.copy()
    ar = np.arange(m)
    sw[ar, A_], sw[ar, B_] = o[ar, B_], o[ar, A_]
    odors = np.concatenate([o, o, sw, sw])
    zs = np.tile(np.concatenate([np.zeros(m), np.ones(m)]), 2)
    y = np.concatenate([A_, B_, B_, A_])
    var = np.repeat(np.arange(4), m)
    seeds = np.tile(rows["seed"][base], 4)
    games = np.tile(rows["game"][base], 4)
    p = Probe(odors, zs, np.zeros((4 * m, 4)), seeds, games, np.tile(rows["legal"][base], (4, 1)), y,
              {"var": var, "base": np.tile(np.arange(m), 4), "rows": base, "A": A_, "B": B_,
               "src": np.tile(base, 4), "swap": var >= 2, "swapA": np.tile(A_, 4), "swapB": np.tile(B_, 4)})
    return p, {"candidates": int(len(cand)), "unique": int(m)}


def build_real_c(rows: dict):
    """실제 관측(통제 없음)의 과제 C: 유일 최댓값이 있는 결정 표본, z는 자료 그대로 (가짜 엔진은 0뿐)."""
    cand = np.flatnonzero(decision_mask(rows))
    best, second, arg = _top_two(rows["odor"][cand], rows["legal"][cand])
    ok = (best > 0.0) & (best - second > TOL)
    return _sub(rows, cand[ok], y=arg[ok])


def reference_odor(train_rows: dict) -> np.ndarray:
    """기준 입력의 냄새 중앙값 (13.10 항목 4): 개발-학습 결정 표본에서 채널마다 그 방향이 열려 있을 때의 값만."""
    cand = np.flatnonzero(decision_mask(train_rows))
    o, wall = train_rows["odor"][cand], train_rows["wall"][cand]
    return np.array([np.median(o[~wall[:, c], c]) for c in range(4)])


@dataclass
class SplitData:
    name: str
    rows: dict
    t1: Probe
    t1_tie: Probe
    t2: Probe
    t3: Probe
    t4: Probe
    real_c: Probe
    exclusions: dict
    data_hash: str


def build_split(name: str, rows: dict) -> SplitData:
    t1, t1_tie, s1 = build_test1(rows)
    t2, s2 = build_test2(rows)
    t3, s3 = build_test3(rows)
    t4, s4 = build_test4(rows)
    return SplitData(name, rows, t1, t1_tie, t2, t3, t4, build_real_c(rows),
                     {"t1": s1, "t2": s2, "t3": s3, "t4": s4}, rows_hash(rows))


def sample_counts(sd: SplitData) -> dict:
    """13.5의 표본 수: 시험 1 · 3 방향별, 시험 2 z별, 시험 4 (상태 × 정답 방향)별 유일 표본."""
    t4 = sd.t4
    return {"t1": {int(dd): int((sd.t1.y == dd).sum()) for dd in DIRS},
            "t2": {0: int(len(sd.t2) // 2), 1: int(len(sd.t2) // 2)},
            "t3": {int(dd): int((sd.t3.y == dd).sum()) for dd in DIRS},
            "t4": {f"z{int(zv)}_dir{dd}": int(((t4.z == zv) & (t4.y == dd)).sum()) for zv in (0, 1) for dd in DIRS}}


def count_shortages(counts: dict) -> list[str]:
    return [f"{t}:{k}={v}" for t, cs in counts.items() for k, v in cs.items() if v < MIN_SAMPLES]


# =============================================================== KC 특징

class RowRngs:
    """표본 행마다 make_rng(실행 시드, 판 번호, "kc_tiebreak") (13.10 항목 5). (시험, 변형)마다 새로 만든다."""

    def __init__(self, seeds, games):
        self.seeds, self.games = seeds, games
        self._g: dict[tuple[int, int], np.random.Generator] = {}

    def __getitem__(self, r):
        key = (int(self.seeds[r]), int(self.games[r]))
        g = self._g.get(key)
        if g is None:
            g = self._g[key] = make_rng(key[0], key[1], "kc_tiebreak")
        return g


class _Offset:
    def __init__(self, rr, off):
        self.rr, self.off = rr, off

    def __getitem__(self, r):
        return self.rr[self.off + r]


def n_active_k(rho: float, n_kc: int) -> int:
    return int(math.ceil(rho * n_kc - 1e-9))


def kc_features(circ: Circuit, rho: float, theta: float, odor, z, phero, seeds, games, chunk: int = 4000, rr=None):
    """k (n, N_KC) bool. 상위 k의 동률은 행마다 RowRngs로 끊는다."""
    n = len(z)
    k = n_active_k(rho, circ.n_kc)
    out = np.zeros((n, circ.n_kc), dtype=bool)
    rr = RowRngs(seeds, games) if rr is None else rr
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        u = kc_drive(pn_input(odor[s:e], z[s:e], phero[s:e]), circ)
        out[s:e] = kc_select(u, theta, k, _Offset(rr, s))
    return out


def u_ref(circ: Circuit, med_odor: np.ndarray) -> float:
    """기준 입력(냄새 네 채널 = 중앙값, z = 0, 페로몬 0, α 적용)에서 KC 전체 u_i의 평균 (13.6)."""
    p = pn_input(med_odor[None, :], np.zeros(1), np.zeros((1, 4)))
    return float(kc_drive(p, circ).mean())


# =============================================================== 분류기

def fit_clf(X, y):
    """(분류기, 수렴 여부). 수렴 경고는 실패 이유 '분류기 비수렴'으로 구분한다 (13.4)."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(**CLF_SETTINGS)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        clf.fit(np.asarray(X, dtype=np.float64), y)
    return clf, not any(issubclass(x.category, ConvergenceWarning) for x in w)


def predict_masked(clf, X, legal):
    """결정 함수의 합법 행동 중 최댓값 (13.4). 분류기에게 합법성을 맞히게 하지 않는다."""
    dec = clf.decision_function(np.asarray(X, dtype=np.float64))
    if dec.ndim == 1:
        dec = np.stack([-dec, dec], axis=1)
    full = np.full((len(dec), 4), -np.inf)
    full[:, clf.classes_] = dec
    full = np.where(legal, full, -np.inf)
    return full.argmax(1)


def balanced_recall(y, pred, classes) -> tuple[float, dict]:
    rec = {}
    for c in classes:
        m = y == c
        if m.any():
            rec[int(c)] = float((pred[m] == c).mean())
    return (float(np.mean(list(rec.values()))) if rec else float("nan")), rec


def _verdict(acc, thr, conv):
    if not conv:
        return "classifier_not_converged"
    return "pass" if acc >= thr else "accuracy_below"


def control_scores(tr: SplitData, ev: SplitData) -> dict:
    """대조 분류기 (13.3): KC 없이 (z, 합법 마스크)만. 자료 분할당 한 번. 시험 4 대조의 z별 일반 정확도는
    완전한 네 변형 묶음에서 정확히 ≤ 0.5이어야 한다. 아니면 자료 구성 버그다."""
    def feat4(p):
        return np.column_stack([p.z, p.legal.astype(float)])
    clf, conv = fit_clf(feat4(tr.t4), tr.t4.y)
    out = {"t4_converged": conv}
    for name, sd in (("train", tr), ("eval", ev)):
        p = sd.t4
        pred = predict_masked(clf, feat4(p), p.legal)
        for zv in (0, 1):
            m = p.z == zv
            out[f"t4_{name}_general_acc_z{zv}"] = float((pred[m] == p.y[m]).mean())
        out[f"t4_{name}_balanced"] = {int(zv): balanced_recall(p.y[p.z == zv], pred[p.z == zv], DIRS)[0]
                                      for zv in (0, 1)}
    for t in ("t1", "t3"):
        clf, conv = fit_clf(tr.__dict__[t].legal.astype(float), tr.__dict__[t].y)
        p = ev.__dict__[t]
        pred = predict_masked(clf, p.legal.astype(float), p.legal)
        out[f"{t}_mask_only_balanced"] = balanced_recall(p.y, pred, DIRS)[0]
    out["ok"] = all(out[f"t4_{n}_general_acc_z{zv}"] <= CONTROL_MAX + 1e-12 for n in ("train", "eval") for zv in (0, 1))
    return out


# =============================================================== 조합 하나의 측정

def _probe_feats(circ, rho, theta, p: Probe, z_override=None):
    z = p.z if z_override is None else np.full(len(p), float(z_override))
    return kc_features(circ, rho, theta, p.odor, z, p.phero, p.seed, p.game)


def _jaccard_rows(a: np.ndarray, b: np.ndarray):
    """행마다 |a∩b| / |a∪b|. 둘 다 빈 행은 nan."""
    inter = (a & b).sum(1).astype(float)
    union = (a | b).sum(1).astype(float)
    return np.where(union > 0, inter / np.maximum(union, 1), np.nan)


def measure_combo(circ: Circuit, rho: float, c: int, uref: float, tr: SplitData, ev: SplitData, kc_class=None) -> tuple[dict, dict]:
    """조합 (배정 시드, ρ, c) 하나의 시험 1 ~ 4 + 기록용 진단. 반환 (결과, 진단용 특징 {t1_eval 켜진 집합})."""
    theta = c * uref
    res: dict = {"seed": circ.seed, "rho": rho, "c": c, "theta": theta, "k": n_active_k(rho, circ.n_kc)}
    feats: dict[str, np.ndarray] = {}

    # --- 시험 1
    Xtr, Xev = _probe_feats(circ, rho, theta, tr.t1), _probe_feats(circ, rho, theta, ev.t1)
    clf, conv = fit_clf(Xtr, tr.t1.y)
    pred = predict_masked(clf, Xev, ev.t1.legal)
    acc, per = balanced_recall(ev.t1.y, pred, DIRS)
    t1 = {"acc": acc, "per_dir": per, "converged": conv, "verdict": _verdict(acc, PASS["t1"], conv)}
    Xtie = _probe_feats(circ, rho, theta, ev.t1_tie)
    if len(ev.t1_tie):
        pt = predict_masked(clf, Xtie, ev.t1_tie.legal)
        t1["tie_hit_rate"] = float(ev.t1_tie.meta["opt"][np.arange(len(pt)), pt].mean())
        t1["tie_n"] = int(len(pt))
    res["t1"] = t1
    feats["t1_eval"] = Xev
    # 진단: z = 1 표본만으로 한 시험 1
    Xtr1, Xev1 = _probe_feats(circ, rho, theta, tr.t1, 1.0), _probe_feats(circ, rho, theta, ev.t1, 1.0)
    clf1, conv1 = fit_clf(Xtr1, tr.t1.y)
    a1, _ = balanced_recall(ev.t1.y, predict_masked(clf1, Xev1, ev.t1.legal), DIRS)
    res["diag_t1_z1_only"] = {"acc": a1, "converged": conv1}
    del Xtr, Xtr1, Xev1

    # --- 시험 2
    Xtr, Xev = _probe_feats(circ, rho, theta, tr.t2), _probe_feats(circ, rho, theta, ev.t2)
    clf, conv = fit_clf(Xtr, tr.t2.y)
    pred = clf.predict(Xev.astype(np.float64))
    acc, per = balanced_recall(ev.t2.y, pred, (0, 1))
    res["t2"] = {"acc": acc, "per_z": per, "converged": conv, "verdict": _verdict(acc, PASS["t2"], conv)}
    m = ev.t2.meta["pairs"]
    jk = _jaccard_rows(Xev[:m], Xev[m:])
    res["diag_jk"] = {"mean": float(np.nanmean(jk)) if np.isfinite(jk).any() else None,
                      "excluded_both_empty": int(np.isnan(jk).sum()), "n": int(m)}
    del Xtr, Xev

    # --- 시험 3
    Xtr, Xev = _probe_feats(circ, rho, theta, tr.t3), _probe_feats(circ, rho, theta, ev.t3)
    clf, conv = fit_clf(Xtr, tr.t3.y)
    pred = predict_masked(clf, Xev, ev.t3.legal)
    acc, per = balanced_recall(ev.t3.y, pred, DIRS)
    same = ev.t3.meta["same"]
    hit = pred == ev.t3.y
    res["t3"] = {"acc": acc, "per_dir": per, "converged": conv, "verdict": _verdict(acc, PASS["t3"], conv),
                 "same_dir_acc": float(hit[same == 1].mean()) if (same == 1).any() else None,
                 "diff_dir_acc": float(hit[same == 0].mean()) if (same == 0).any() else None,
                 "same_n": int((same == 1).sum()), "diff_n": int((same == 0).sum())}
    del Xtr, Xev

    # --- 시험 4
    Xtr, Xev = _probe_feats(circ, rho, theta, tr.t4), _probe_feats(circ, rho, theta, ev.t4)
    clf, conv = fit_clf(Xtr, tr.t4.y)
    pred = predict_masked(clf, Xev, ev.t4.legal)
    p4 = ev.t4
    zacc = {int(zv): balanced_recall(p4.y[p4.z == zv], pred[p4.z == zv], DIRS)[0] for zv in (0, 1)}
    ok = pred == p4.y
    base = p4.meta["base"]
    nb = int(base.max()) + 1
    all4 = float(np.bincount(base, weights=ok.astype(float), minlength=nb).__eq__(4).mean())
    acc = min(zacc.values())
    res["t4"] = {"acc_z0": zacc[0], "acc_z1": zacc[1], "all4_correct": all4, "converged": conv,
                 "verdict": _verdict(acc, PASS["t4"], conv)}
    # 실제 관측(통제 없음)의 과제 C: 시험 4 분류기를 z = 자료 그대로의 유일 최댓값 표본에 (판정에 안 씀)
    pr = ev.real_c
    Xr = _probe_feats(circ, rho, theta, pr)
    res["real_task_c"] = {"acc_z0": balanced_recall(pr.y, predict_masked(clf, Xr, pr.legal), DIRS)[0],
                          "n": int(len(pr)), "note": "가짜 엔진은 z = 0뿐"}
    del Xtr, Xev, Xr

    res["pass"] = all(res[t]["verdict"] == "pass" for t in ("t1", "t2", "t3", "t4"))
    res["diag"] = diagnostics(circ, rho, theta, feats["t1_eval"], ev, kc_class)
    return res, {"t1_eval_on": np.packbits(feats["t1_eval"], axis=1)}


def diagnostics(circ: Circuit, rho: float, theta: float, X: np.ndarray, ev: SplitData, kc_class=None) -> dict:
    """기록만 하는 진단 (13.9, 판정에 안 씀). X: 시험 1 평가 표본의 켜진 KC (n, N_KC)."""
    n_on = X.sum(1)
    out = {"n_on_mean": float(n_on.mean()), "n_on_min": int(n_on.min()), "n_on_max": int(n_on.max()),
           "n_on_q": [float(v) for v in np.quantile(n_on, [0.05, 0.5, 0.95])],
           "sample_silence_rate": float((n_on == 0).mean()),
           "kc_silence_rate": float((X.sum(0) == 0).mean())}
    raw_cls = kc_class
    if raw_cls is not None:
        names = sorted(set(map(str, raw_cls)))
        share = {}
        for nm in names:
            msk = np.asarray([str(v) == nm for v in raw_cls])
            share[nm] = float((X[:, msk].sum(1) / np.maximum(n_on, 1)).mean())
        out["kc_class_share_of_active"] = share
    # 행동별 도달 비율: 켜진 KC 중 묶음 a의 MBON과 연결된 비율
    bundles = np.unique(circ.mbon_bundle)
    reach = {}
    for a in bundles:
        conn = circ.M[:, circ.mbon_bundle == a].any(1)
        reach[int(a)] = float((X[:, conn].sum(1) / np.maximum(n_on, 1)).mean())
    out["reach_ratio_by_action"] = reach
    # 채널별 상위 k 점유율 (실제 입력): 켜진 KC의 가장 큰 기여 채널 (최대 2,000 표본)
    sub = ev.t1
    nn = min(2000, len(sub))
    p = pn_input(sub.odor[:nn], sub.z[:nn], sub.phero[:nn]) * circ.alpha
    share = np.zeros(9)
    cnt = 0
    for s in range(0, nn, 250):
        e = min(nn, s + 250)
        contrib = circ.J[None, :, :] * p[s:e, None, :]              # (b, N_KC, 9)
        top = contrib.argmax(-1)                                    # (b, N_KC)
        on = X[s:e]
        for ch in range(9):
            share[ch] += ((top == ch) & on).sum()
        cnt += on.sum()
    out["topk_share_by_channel_real"] = [float(v / max(cnt, 1)) for v in share]
    return out


def single_channel_diag(circ: Circuit, med_odor: np.ndarray, rho: float, theta: float) -> list[int]:
    """채널 하나만 기준 크기로 넣었을 때 켜진 KC 수 (z는 1, 나머지 채널은 같은 냄새 중앙값 크기)."""
    k = n_active_k(rho, circ.n_kc)
    out = []
    for ch in range(9):
        p = np.zeros((1, 9))
        p[0, ch] = 1.0 if ch == 4 else float(med_odor[ch % 5 if ch < 4 else ch - 5])
        u = kc_drive(p, circ)
        on = kc_select(u, theta, k, _Fixed(make_rng(0, 0, "kc_tiebreak")))
        out.append(int(on.sum()))
    return out


class _Fixed:
    def __init__(self, g):
        self.g = g

    def __getitem__(self, r):
        return self.g


def seed_diag(circ: Circuit) -> dict:
    J = circ.J
    reach = (J > 0).sum(0)
    return {"alpha": [float(v) for v in circ.alpha], "reach_kc_per_channel": [int(v) for v in reach],
            "junction_kc_ratio": float(((J > 0).sum(1) >= 2).mean()), "n_kc": int(circ.n_kc)}


def measure_task(args) -> dict:
    """작업 하나 = (배정 시드, ρ): c ∈ CS 세 조합. 프로세스 풀에서 돈다."""
    import os
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    import pickle

    from flypac.brain.flybrain import build_circuit, load_brain_config, repo_path
    from flypac.core.config import load_config
    seed, rho, data_path, med = args
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    tr, ev = data["dev_train"], data["dev_eval"]
    bcfg = load_brain_config("configs/brain_v1.yaml")
    circ = build_circuit(bcfg.circuit.raw, load_config(repo_path(bcfg.circuit.assign)), seed)
    kc_class = A.load_raw(repo_path(bcfg.circuit.raw)).kc_class
    uref = u_ref(circ, np.asarray(med))
    out = {"seed": seed, "rho": rho, "uref": uref, "combos": []}
    packed = {}
    for c in CS:
        t0 = time.perf_counter()
        r, extra = measure_combo(circ, rho, c, uref, tr, ev, kc_class)
        r["sec"] = time.perf_counter() - t0
        r["diag"]["single_channel_n_on"] = single_channel_diag(circ, np.asarray(med), rho, c * uref)
        packed[c] = extra["t1_eval_on"]
        out["combos"].append(r)
    # 같은 ρ에서 θ 격자 간 활성 집합의 중복
    n = packed[0].shape[0]
    ov = {}
    for a, b in ((0, 1), (0, 2), (1, 2)):
        xa = np.unpackbits(packed[a], axis=1, count=circ.n_kc).astype(bool)
        xb = np.unpackbits(packed[b], axis=1, count=circ.n_kc).astype(bool)
        j = _jaccard_rows(xa, xb)
        ov[f"c{a}_c{b}"] = float(np.nanmean(j)) if np.isfinite(j).any() else None
    out["theta_overlap"] = ov
    out["seed_diag"] = seed_diag(circ)
    return out


# =============================================================== 보조 ②: 누적 h (후보 하나, 판정에 안 씀)

def interval_layout(rows: dict):
    """초파리별 시간순 배열 order, 그 안의 위치 pos, 직전 초기화 뒤 구간의 시작 위치 istart (13.10 항목 11)."""
    order = np.lexsort((rows["tick"], rows["fly"], rows["game"], rows["seed"]))
    key = (rows["seed"][order].astype(np.int64) * 1000 + rows["game"][order]) * 10 + rows["fly"][order]
    reset = rows["reset"][order]
    istart = np.empty(len(order), dtype=np.int64)
    start, prev = 0, None
    for j in range(len(order)):
        if key[j] != prev:
            start, prev = j, key[j]
        istart[j] = start
        if reset[j]:
            start = j + 1
    pos = np.empty(len(order), dtype=np.int64)
    pos[order] = np.arange(len(order))
    return order, pos, istart


def h_features(circ: Circuit, rho: float, theta: float, rows: dict, layout, probe: Probe, kappas=KAPPAS,
               chunk_targets: int = 3000) -> dict[float, np.ndarray]:
    """표본마다 누적 h ← (1-κ)h + k 를 직전 초기화 뒤 구간 전체에서 재현한 h (n, N_KC) float32, κ별.

    z는 표본의 값(probe.z)으로 구간 전체를 바꾸고, 시험 4의 교환 변형은 구간 전체에서 두 감각 채널(A, B)을 맞바꾼다
    (구간 중 초파리가 방향을 틀면 자기중심 채널의 뜻이 바뀌므로 '구간 동안 두 채널이 뒤바뀐 경우'의 근사다).
    동결된 틱은 갱신하지 않는다 (뇌의 propose는 alive 행만 갱신한다).
    """
    order, pos, istart = layout
    src = probe.meta["src"]
    swap = probe.meta.get("swap")
    n = len(probe)
    out = {kp: np.zeros((n, circ.n_kc), dtype=np.float32) for kp in kappas}
    frozen = rows["frozen"]
    for s0 in range(0, n, chunk_targets):
        s1 = min(n, s0 + chunk_targets)
        ids, tgt, lens = [], [], []
        for t in range(s0, s1):
            j = pos[src[t]]
            seg = order[istart[j]:j + 1]
            seg = seg[~frozen[seg]]                 # alive 행만
            ids.append(seg)
            tgt.append(np.full(len(seg), t - s0))
            lens.append(len(seg))
        ids, tgt, lens = np.concatenate(ids), np.concatenate(tgt), np.array(lens)
        odor = rows["odor"][ids].copy()
        if swap is not None:
            sw = swap[s0:s1][tgt]
            a, b = probe.meta["swapA"][s0:s1][tgt], probe.meta["swapB"][s0:s1][tgt]
            r_ = np.flatnonzero(sw)
            oa, ob = odor[r_, a[r_]].copy(), odor[r_, b[r_]].copy()
            odor[r_, a[r_]], odor[r_, b[r_]] = ob, oa
        z = probe.z[s0:s1][tgt]
        K = kc_features(circ, rho, theta, odor, z, np.zeros((len(ids), 4)), rows["seed"][ids], rows["game"][ids],
                        rr=RowRngs(rows["seed"][ids], rows["game"][ids]))
        starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
        # 구간 안에서 끝에서부터 센 거리 e (마지막 alive 행이 0)
        ends = np.repeat(np.cumsum(lens) - 1, lens)
        dist = ends - np.arange(len(ids))
        for kp in kappas:
            w = ((1.0 - kp) ** dist).astype(np.float32)
            out[kp][s0:s1] = np.add.reduceat(K.astype(np.float32) * w[:, None], starts, axis=0)
    return out


def aux_h_test(circ: Circuit, rho: float, theta: float, tr: SplitData, ev: SplitData, kappas=KAPPAS) -> dict:
    """시험 1, 2, 4를 누적 h의 판독으로 (13.4 보조 ②). 시험 3은 건너뛴다."""
    ltr, lev = interval_layout(tr.rows), interval_layout(ev.rows)
    res: dict = {}
    for name in ("t1", "t2", "t4"):
        Htr = h_features(circ, rho, theta, tr.rows, ltr, getattr(tr, name), kappas)
        Hev = h_features(circ, rho, theta, ev.rows, lev, getattr(ev, name), kappas)
        res[name] = {}
        for kp in kappas:
            clf, conv = fit_clf(Htr[kp], getattr(tr, name).y)
            p = getattr(ev, name)
            if name == "t2":
                acc, per = balanced_recall(p.y, clf.predict(Hev[kp]), (0, 1))
                res[name][kp] = {"acc": acc, "converged": conv}
            elif name == "t1":
                acc, per = balanced_recall(p.y, predict_masked(clf, Hev[kp], p.legal), DIRS)
                res[name][kp] = {"acc": acc, "converged": conv}
            else:
                pred = predict_masked(clf, Hev[kp], p.legal)
                za = {int(zv): balanced_recall(p.y[p.z == zv], pred[p.z == zv], DIRS)[0] for zv in (0, 1)}
                ok = pred == p.y
                nb = int(p.meta["base"].max()) + 1
                res[name][kp] = {"acc_z0": za[0], "acc_z1": za[1], "converged": conv,
                                 "all4_correct": float((np.bincount(p.meta["base"], weights=ok.astype(float),
                                                                    minlength=nb) == 4).mean())}
    return res


# =============================================================== 선정과 보고서

def grid_order():
    return [(rho, c) for rho in RHOS for c in CS]


def select_candidate(combos: list[dict]):
    """사전 순서(시드가 바깥, 격자가 안쪽)에서 처음으로 시험 1 ~ 4를 모두 통과한 조합 (13.7)."""
    index = {(r["seed"], r["rho"], r["c"]): r for r in combos}
    for s in ASSIGN_SEEDS:
        for rho, c in grid_order():
            r = index.get((s, rho, c))
            if r is not None and r["pass"]:
                return r
    return None


def passing_seeds(combos: list[dict]) -> list[int]:
    return sorted({r["seed"] for r in combos if r["pass"]})


def build_results(tasks: list[dict], data: dict, control: dict, med: np.ndarray) -> dict:
    """작업 결과를 모아 results.json의 내용을 만든다. 조합은 (시드, 격자) 사전 순서."""
    combos = sorted((c for t in tasks for c in t["combos"]),
                    key=lambda r: (r["seed"], RHOS.index(r["rho"]), r["c"]))
    uref = {t["seed"]: t["uref"] for t in tasks}
    cand = select_candidate(combos)
    return {
        "generator_version": GEN_VERSION,
        "splits": {k: {"seeds": list(v), "hash": data[k].data_hash, "rows": int(len(data[k].rows["seed"])),
                       "exclusions": data[k].exclusions, "sample_counts": sample_counts(data[k])}
                   for k, v in SPLITS.items()},
        "control": control,
        "reference_odor_median": [float(v) for v in med],
        "u_ref_by_seed": {str(k): v for k, v in sorted(uref.items())},
        "classifier": {**CLF_SETTINGS, "input_dtype": "float64", "standardize": False},
        "n_combos": len(combos),
        "combos": combos,
        "candidate": None if cand is None else {k: cand[k] for k in ("seed", "rho", "c", "theta", "k")},
        "passing_seeds": passing_seeds(combos),
        "passing_seed_count": len(passing_seeds(combos)),
        "theta_overlap": {f"{t['seed']}_{t['rho']}": t["theta_overlap"] for t in tasks},
        "seed_diag": {str(t["seed"]): t["seed_diag"] for t in tasks},
    }


def _f(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def _grid_table(res: dict, getter, thr=None, reason=None) -> list[str]:
    cols = [f"ρ{rho} c{c}" for rho, c in grid_order()]
    lines = ["| 시드 | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    idx = {(r["seed"], r["rho"], r["c"]): r for r in res["combos"]}
    for s in ASSIGN_SEEDS:
        cells = []
        for rho, c in grid_order():
            r = idx.get((s, rho, c))
            if r is None:
                cells.append("·")
                continue
            v = getter(r)
            txt = _f(v)
            if reason is not None and reason(r) != "pass":
                txt += {"classifier_not_converged": " (비수렴)", "accuracy_below": ""}.get(reason(r), "")
            if thr is not None and reason is not None and reason(r) == "pass":
                txt = f"**{txt}**"
            cells.append(txt)
        lines.append(f"| {s} | " + " | ".join(cells) + " |")
    return lines


def format_report(res: dict, aux: dict | None = None, sklearn_version: str = "", commit: str = "") -> str:
    L: list[str] = []
    cand = res["candidate"]
    L += ["# 표현력 시험 보고서 (M1 뇌 ② 단계 3, 개발 자료)", "",
          "규칙은 DECISIONS 13절(지시서 4절, 동결)과 13.10을 따른다. **결과를 본 뒤 규칙을 바꾸지 않았다.** "
          "최종 검증(지시서 5절)은 하지 않았고 최종 검증 자료(2000 ~ 2009)는 만들지 않았다.", ""]
    L += ["## 요약", "",
          f"- 측정한 조합: {res['n_combos']}개 (배정 시드 0 ~ 19 × 격자 9개)",
          f"- 통과한 조합: {sum(r['pass'] for r in res['combos'])}개",
          f"- **통과 시드 수: {res['passing_seed_count']} / 20** (격자 9개 중 하나 이상에서 시험 1 ~ 4를 모두 통과한 배정 시드)",
          f"- 통과 시드: {res['passing_seeds']}",
          "- **후보 (사전 순서에서 처음 통과한 조합): " + (
              "없음" if cand is None else f"배정 시드 {cand['seed']}, ρ = {cand['rho']} (k = {cand['k']}), "
              f"c = {cand['c']} (θ = {cand['theta']:.4f})") + "**",
          f"- 자료 생성기 `{res['generator_version']}`, sklearn {sklearn_version}, 코드 커밋 `{commit}`", ""]
    L += ["## 자료", "", "| 분할 | 실행 시드 | 행 | sha256 |", "|---|---|---|---|"]
    for k, v in res["splits"].items():
        L.append(f"| {k} | {v['seeds'][0]} ~ {v['seeds'][1]} | {v['rows']} | `{v['hash'][:16]}…` |")
    L += ["", "### 제외와 유일 표본", "", "| 분할 | 시험 | 결정 표본(후보) | \\|L\\|<2 | 무입력 | 동률 | 짝 없음 | 입력없음·동률(시험3) | 유일 표본 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k, v in res["splits"].items():
        for t, e in v["exclusions"].items():
            L.append(f"| {k} | {t} | {e.get('candidates', '—')} | {e.get('lt2', '—')} | {e.get('zero_input', '—')} | "
                     f"{e.get('tie', '—')} | {e.get('no_partner', '—')} | {e.get('no_input_or_tie', '—')} | {e.get('unique', '—')} |")
    L += ["", "### 표본 수 (13.5, 기준 ≥ 50)", ""]
    for k, v in res["splits"].items():
        L.append(f"- {k}: {v['sample_counts']}")
    L += ["", "가짜 엔진은 반전을 금지하므로 \\|L\\| ≥ 2인 결정에서 '뒤'는 합법이 아니다. 방향별 기준은 좌 · 앞 · 우 세 방향에 적용했다 (13.10 항목 15). "
          "가짜 엔진은 z가 항상 0이라 z = 1은 시험 2 · 4의 합성 변형에만 있다.", ""]
    c = res["control"]
    L += ["### 대조 분류기 (자료 구성 검사, 분할당 한 번)", "",
          f"- 시험 4 대조(z, 합법 마스크만)의 z별 **일반** 정확도: 개발-학습 {_f(c['t4_train_general_acc_z0'])} / {_f(c['t4_train_general_acc_z1'])}, "
          f"개발-평가 {_f(c['t4_eval_general_acc_z0'])} / {_f(c['t4_eval_general_acc_z1'])} (상한 0.50: 통과)",
          f"- 시험 4 대조의 균형 정확도(기록만): 개발-평가 {c['t4_eval_balanced']}",
          f"- 시험 1 · 3 대조(마스크만) 균형 정확도(기록만): {_f(c['t1_mask_only_balanced'])} / {_f(c['t3_mask_only_balanced'])}", ""]
    L += ["## 기준 입력", "",
          f"- 냄새 중앙값(좌 · 앞 · 우 · 뒤, 개발-학습 결정 표본에서 벽 마스크가 거짓인 값만): {[round(v, 4) for v in res['reference_odor_median']]}",
          f"- ū_ref (시드별, ρ와 무관): {{" + ", ".join(f"{k}: {v:.2f}" for k, v in res['u_ref_by_seed'].items()) + "}", ""]
    L += ["## 시험별 결과 (개발-평가). 굵은 글씨는 통과, '(비수렴)'은 분류기 비수렴", ""]
    L += ["### 시험 1. 냄새 방향 (방향별 균형 정확도, ≥ 0.90)", ""] + _grid_table(res, lambda r: r["t1"]["acc"], True, lambda r: r["t1"]["verdict"]) + [""]
    L += ["### 시험 2. 상태 (z별 균형 정확도, ≥ 0.95)", ""] + _grid_table(res, lambda r: r["t2"]["acc"], True, lambda r: r["t2"]["verdict"]) + [""]
    L += ["### 시험 3. 페로몬 채널의 방향 전달 (방향별 균형 정확도, ≥ 0.90)", ""] + _grid_table(res, lambda r: r["t3"]["acc"], True, lambda r: r["t3"]["verdict"]) + [""]
    L += ["### 시험 4. 과제 C 결합: z = 0 / z = 1 중 낮은 쪽 (각각 ≥ 0.90)", ""] + _grid_table(
        res, lambda r: min(r["t4"]["acc_z0"], r["t4"]["acc_z1"]), True, lambda r: r["t4"]["verdict"]) + [""]
    L += ["### 통과 (시험 1 ~ 4 모두)", ""]
    idx = {(r["seed"], r["rho"], r["c"]): r for r in res["combos"]}
    cols = [f"ρ{rho} c{c}" for rho, c in grid_order()]
    L += ["| 시드 | " + " | ".join(cols) + " | 통과 격자 수 |", "|---|" + "---|" * (len(cols) + 1)]
    for s in ASSIGN_SEEDS:
        flags = [idx[(s, rho, c)]["pass"] for rho, c in grid_order() if (s, rho, c) in idx]
        L.append(f"| {s} | " + " | ".join("✓" if f else "✗" for f in flags) + f" | {sum(flags)} |")
    L += ["", "### 실패 이유 집계", ""]
    for t in ("t1", "t2", "t3", "t4"):
        cnt: dict[str, int] = {}
        for r in res["combos"]:
            cnt[r[t]["verdict"]] = cnt.get(r[t]["verdict"], 0) + 1
        L.append(f"- 시험 {t[1]}: {cnt}")
    L += ["", "자료 부족은 측정 전 확인에서 걸러지므로(13.10 항목 9) 이 표에는 없다.", ""]
    L += ["## 기록만 하는 진단 (13.9, 판정에 쓰지 않음)", ""]
    L += ["### 격자별 평균 (20개 시드)", "",
          "| 격자 | 켜진 KC 수(평균) | 표본 침묵률 | KC 침묵률 | z=1 표본만 시험 1 | J_K | 시험 1 동률 적중률 | 실제 관측 과제 C(z=0) | 시험 3 같은 방향 | 시험 3 다른 방향 | 시험 4 네 변형 모두 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for rho, c in grid_order():
        rs = [r for r in res["combos"] if r["rho"] == rho and r["c"] == c]

        def mean(fn):
            v = [fn(r) for r in rs]
            v = [x for x in v if x is not None and np.isfinite(x)]
            return float(np.mean(v)) if v else None
        L.append(f"| ρ{rho} c{c} | {_f(mean(lambda r: r['diag']['n_on_mean']), 1)} | {_f(mean(lambda r: r['diag']['sample_silence_rate']))} | "
                 f"{_f(mean(lambda r: r['diag']['kc_silence_rate']))} | {_f(mean(lambda r: r['diag_t1_z1_only']['acc']))} | "
                 f"{_f(mean(lambda r: r['diag_jk']['mean']))} | {_f(mean(lambda r: r['t1'].get('tie_hit_rate')))} | "
                 f"{_f(mean(lambda r: r['real_task_c']['acc_z0']))} | {_f(mean(lambda r: r['t3']['same_dir_acc']))} | "
                 f"{_f(mean(lambda r: r['t3']['diff_dir_acc']))} | {_f(mean(lambda r: r['t4']['all4_correct']))} |")
    L += ["", "### 같은 ρ에서 θ 격자 간 활성 집합의 중복 (평균 Jaccard, 시험 1 평가 표본, 20개 시드 평균)", "",
          "| ρ | c0 ∩ c1 | c0 ∩ c2 | c1 ∩ c2 |", "|---|---|---|---|"]
    for rho in RHOS:
        row = [v for k, v in res["theta_overlap"].items() if k.endswith(f"_{rho}")]
        m = {p: [x[p] for x in row if x[p] is not None] for p in ("c0_c1", "c0_c2", "c1_c2")}
        L.append(f"| {rho} | " + " | ".join(_f(float(np.mean(m[p])) if m[p] else None) for p in m) + " |")
    L += ["", "### 행동별 도달 비율 (켜진 KC 중 그 행동 묶음의 MBON과 연결된 비율, 20개 시드 평균)", ""]
    reach: dict[str, list[float]] = {}
    for r in res["combos"]:
        for a, v in r["diag"]["reach_ratio_by_action"].items():
            reach.setdefault(str(a), []).append(v)
    L.append("- " + ", ".join(f"묶음 {a}: {_f(float(np.mean(v)))}" for a, v in sorted(reach.items())))
    L += ["", "### 채널별 (채널 순서: 냄새 좌 · 앞 · 우 · 뒤, z, 페로몬 좌 · 앞 · 우 · 뒤)", "",
          "- 실제 입력에서 켜진 KC의 가장 큰 기여 채널 점유율 (평균): " +
          str([round(float(np.mean([r['diag']['topk_share_by_channel_real'][i] for r in res['combos']])), 3) for i in range(9)]),
          "- 채널 하나만 기준 크기로 넣었을 때 켜진 KC 수, 격자별 평균 (z는 1, 나머지는 냄새 중앙값 크기):"]
    for rho, c in grid_order():
        rs = [r for r in res["combos"] if r["rho"] == rho and r["c"] == c]
        L.append(f"  - ρ{rho} c{c}: " + str([round(float(np.mean([r['diag']['single_channel_n_on'][i] for r in rs])), 1) for i in range(9)]))
    L += ["", "### 켜진 KC 종류별 구성 (켜진 KC 중 비율, 전체 평균)", ""]
    comp: dict[str, list[float]] = {}
    for r in res["combos"]:
        for k, v in r["diag"].get("kc_class_share_of_active", {}).items():
            comp.setdefault(k, []).append(v)
    L.append("- " + ", ".join(f"{k}: {_f(float(np.mean(v)))}" for k, v in sorted(comp.items())))
    L += ["", "### 배정 시드별 회로 (α_c, 채널별 도달 KC 수, 접합 KC 비율 = 입력 채널이 둘 이상인 KC)", "",
          "| 시드 | α_c | 채널별 도달 KC 수 | 접합 KC 비율 |", "|---|---|---|---|"]
    for s, v in res["seed_diag"].items():
        L.append(f"| {s} | {[round(a, 3) for a in v['alpha']]} | {v['reach_kc_per_channel']} | {_f(v['junction_kc_ratio'])} |")
    if aux:
        L += ["", "## 보조 ②: 누적 h의 판독 (후보 하나, 판정에 쓰지 않음)", "",
              "합성 z와 시험 4의 냄새 맞바꿈은 직전 초기화 이후 구간 전체에서 바꿨다. 구간 중 초파리가 방향을 틀면 자기중심 채널의 뜻이 바뀌므로 "
              "시험 4의 교환 변형은 '그 구간 동안 두 감각 채널이 뒤바뀐 경우'의 근사다. 시험 3은 건너뛴다 (13.10 항목 11).", "",
              "| κ | 시험 1 | 시험 2 | 시험 4 z=0 | 시험 4 z=1 | 시험 4 네 변형 모두 | 비수렴 |", "|---|---|---|---|---|---|---|"]
        for kp in sorted(aux["t1"], key=float):
            nc = [t for t in ("t1", "t2", "t4") if not aux[t][kp]["converged"]]
            L.append(f"| {kp} | {_f(aux['t1'][kp]['acc'])} | {_f(aux['t2'][kp]['acc'])} | {_f(aux['t4'][kp]['acc_z0'])} | "
                     f"{_f(aux['t4'][kp]['acc_z1'])} | {_f(aux['t4'][kp]['all4_correct'])} | {nc or '없음'} |")
        L += ["", "보조 ①(모든 KC를 입력으로)은 모든 시드에서 KC 1,575개가 모두 MBON에 연결되어 주 판정과 같다 (13.10 항목 13).", ""]
    L += ["", "## 해석의 한계", "",
          "- 통과의 뜻은 '이 자료 분포에서 자유 선형 판독기로 해독 가능'이다 (13.1). 실제 판독(행동별 연결 제약, w − w⁰ ≥ −w⁰, 편향항 없음) 아래의 학습 가능성은 작은 과제 A ~ D가 판정한다.",
          "- 선정 결과는 최적 회로가 아니라 사전 우선순위에서 처음 통과한 회로다. 개발-평가 점수에는 180개 중 고른 선택 효과가 남는다.",
          "- 가짜 엔진 분포다. 진짜 엔진 분포 검증과 체제 3 페로몬 검증은 나중에 한다 (13.8).", ""]
    return "\n".join(L)
