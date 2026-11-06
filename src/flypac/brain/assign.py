"""MBON 행동 묶음 · PN 채널 배정 (M1 뇌 작업 지시서 ① 6절, 미결 2의 결정).

원본 회로 파일(data/circuit/raw/mb_raw_v1.npz)만 읽는다. 토큰 불필요.
같은 입력 · 같은 시드 → 같은 결과.

배정은 이 프로젝트가 부여한 대응이다. 생물학적 근거를 주장하지 않는다.
학습 전 방향 편향은 기준선 빼기(DECISIONS)로 구조적으로 없애므로, 여기서는 편향을 최적화하지 않는다.

순서 규약:
- 행동 묶음 a: 자기중심 번호 (좌 0, 직진 1, 우 2, 반전 3) = core/directions.py.
- PN 채널 c: 냄새 좌 · 앞 · 우 · 뒤 (0 ~ 3), z (4), 페로몬 좌 · 앞 · 우 · 뒤 (5 ~ 8).
- KC 종류 τ: gamma, apbp(α′β′), ab(αβ).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from flypac.core.directions import EGO_NAMES

N_ACTIONS = len(EGO_NAMES)
CHANNEL_NAMES = ("odor_left", "odor_forward", "odor_right", "odor_back", "z",
                 "phero_left", "phero_forward", "phero_right", "phero_back")
N_CHANNELS = len(CHANNEL_NAMES)
ODOR_CHANNELS = (0, 1, 2, 3)
Z_CHANNEL = 4
PHERO_CHANNELS = (5, 6, 7, 8)
KC_CLASSES = ("gamma", "apbp", "ab")


# ---------------------------------------------------------------- 원본

@dataclass(frozen=True)
class RawCircuit:
    pn_body: np.ndarray
    pn_type: np.ndarray
    pn_glom: np.ndarray
    kc_body: np.ndarray
    kc_type: np.ndarray
    kc_class: np.ndarray
    mbon_body: np.ndarray
    mbon_type: np.ndarray
    J_pn: np.ndarray
    W0: np.ndarray
    mbon_class_profile: np.ndarray
    glom_names: np.ndarray


def load_raw(path: str | Path) -> RawCircuit:
    with np.load(Path(path), allow_pickle=False) as z:
        return RawCircuit(**{f: z[f] for f in RawCircuit.__dataclass_fields__})


# ---------------------------------------------------------------- MBON 4묶음 (규칙 2)

def normalized_w0(W0: np.ndarray) -> np.ndarray:
    """w0 = W0 / <W0>_{W0>0}."""
    pos = W0[W0 > 0]
    if pos.size == 0:
        raise ValueError("W0에 양수 연결이 없다")
    return W0.astype(np.float64) / pos.mean()


def class_mass(w0: np.ndarray, kc_class: np.ndarray) -> np.ndarray:
    """C[j, τ] = Σ_{i∈τ} w0_ij. 모양 (N_MBON, 3)."""
    return np.stack([w0[kc_class == t].sum(axis=0) for t in KC_CLASSES], axis=1)


def group_sizes(n: int, n_groups: int = N_ACTIONS) -> np.ndarray:
    return np.array([n // n_groups + (1 if g < n % n_groups else 0) for g in range(n_groups)])


def class_table(C: np.ndarray, labels: np.ndarray, n_groups: int = N_ACTIONS) -> np.ndarray:
    """T[a, τ] = (1/|a|) Σ_{j∈a} C[j, τ]. 빈 묶음은 0."""
    T = np.zeros((n_groups, C.shape[1]))
    for a in range(n_groups):
        m = labels == a
        if m.any():
            T[a] = C[m].mean(axis=0)
    return T


def table_objective(T: np.ndarray) -> float:
    """각 τ에서 네 묶음 간 상대 편차의 최댓값 max_a |T_aτ - mean_a| / mean_a, 그 최댓값."""
    mean = T.mean(axis=0)
    ok = mean > 0
    if not ok.any():
        return 0.0
    return float(np.max(np.abs(T[:, ok] - mean[ok]) / mean[ok]))


def assign_mbon_groups(C: np.ndarray, seed: int, max_passes: int,
                       n_groups: int = N_ACTIONS) -> tuple[np.ndarray, dict]:
    """고정 시드 탐욕 배정 → 맞바꾸기 국소 탐색. 묶음 크기 차이 ≤ 1.

    탐욕: 시드로 섞은 순서대로, 자리가 남은 묶음 중 목적값이 가장 작은 곳에 넣는다
    (목적값은 최종 묶음 크기로 나눈 평균으로 계산, 동률이면 작은 번호).
    국소 탐색: 매 회 모든 쌍 맞바꾸기 중 가장 좋은 개선 하나를 받아들인다. 개선이 없거나
    max_passes에 이르면 멈춘다.
    """
    n = C.shape[0]
    if n < n_groups:
        raise ValueError(f"MBON {n}개로 {n_groups}묶음을 만들 수 없다")
    sizes = group_sizes(n, n_groups)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)

    sums = np.zeros((n_groups, C.shape[1]))
    count = np.zeros(n_groups, dtype=int)
    labels = np.full(n, -1)
    for j in order:
        best, best_val = -1, np.inf
        for a in range(n_groups):
            if count[a] >= sizes[a]:
                continue
            trial = sums.copy()
            trial[a] += C[j]
            val = table_objective(trial / sizes[:, None])
            if val < best_val - 1e-15:
                best, best_val = a, val
        labels[j] = best
        sums[best] += C[j]
        count[best] += 1
    greedy_obj = table_objective(class_table(C, labels, n_groups))

    cur = greedy_obj
    passes = 0
    for passes in range(1, max_passes + 1):
        best_gain, best_pair = 0.0, None
        for j in range(n):
            for k in range(j + 1, n):
                if labels[j] == labels[k]:
                    continue
                aj, ak = labels[j], labels[k]
                trial = sums.copy()
                trial[aj] += C[k] - C[j]
                trial[ak] += C[j] - C[k]
                val = table_objective(trial / sizes[:, None])
                if cur - val > best_gain + 1e-12:
                    best_gain, best_pair = cur - val, (j, k)
        if best_pair is None:
            passes -= 1
            break
        j, k = best_pair
        aj, ak = labels[j], labels[k]
        sums[aj] += C[k] - C[j]
        sums[ak] += C[j] - C[k]
        labels[j], labels[k] = ak, aj
        cur -= best_gain
    final = table_objective(class_table(C, labels, n_groups))
    return labels, {"greedy_objective": greedy_obj, "final_objective": final, "swaps": passes}


def mbon_diagnostics(W0: np.ndarray, kc_class: np.ndarray, labels: np.ndarray) -> dict:
    """기록만 하는 진단 (합격 기준 아님): T, 크기, μ_a, d_ia 분포, μ_a/|a|."""
    w0 = normalized_w0(W0)
    C = class_mass(w0, kc_class)
    M = (W0 > 0)
    T = class_table(C, labels)
    size = np.array([(labels == a).sum() for a in range(N_ACTIONS)])
    mu = np.array([M[:, labels == a].mean() if size[a] else 0.0 for a in range(N_ACTIONS)])
    d = np.stack([M[:, labels == a].sum(axis=1) for a in range(N_ACTIONS)], axis=1)
    return {"T": T, "objective": table_objective(T), "size": size, "mu": mu,
            "mu_over_size": np.where(size > 0, mu / np.maximum(size, 1), 0.0), "d": d}


# ---------------------------------------------------------------- PN 9묶음 (규칙 3)

def _balanced_labels(n_items: int, channels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """항목 n개를 채널들에 크기 차이 ≤ 1로 무작위 분할. 어느 채널이 하나 더 받는지도 무작위."""
    k = len(channels)
    slots = np.arange(n_items) % k
    rng.shuffle(slots)
    return channels[rng.permutation(k)][slots]


def assign_pn_channels(glom_names: np.ndarray, seed: int, branch: str,
                       pheromone: tuple[str, ...] | list[str]) -> np.ndarray:
    """사구체별 채널 번호 (N_glom,). 같은 사구체의 PN은 같은 채널 (단위 = 사구체).

    분기 P: 페로몬 사구체 4개를 페로몬 4채널에 하나씩 (방향 무작위), 나머지를 냄새 4 + z에 분할.
    분기 G: 모든 사구체를 9채널에 분할.
    """
    glom_names = np.asarray(glom_names)
    rng = np.random.default_rng(seed)
    out = np.full(len(glom_names), -1, dtype=np.int64)
    if branch == "P":
        pher = list(pheromone)
        if len(pher) != len(PHERO_CHANNELS):
            raise ValueError("분기 P는 페로몬 사구체가 정확히 4개여야 한다")
        idx = [int(np.nonzero(glom_names == g)[0][0]) if (glom_names == g).any() else -1 for g in pher]
        if min(idx) < 0:
            raise ValueError("분기 P: 원본에 없는 페로몬 사구체가 있다 (전환 조건 1)")
        out[idx] = np.asarray(PHERO_CHANNELS)[rng.permutation(len(PHERO_CHANNELS))]
        rest = np.setdiff1d(np.arange(len(glom_names)), idx)
        out[rest] = _balanced_labels(len(rest), np.array([*ODOR_CHANNELS, Z_CHANNEL]), rng)
    elif branch == "G":
        out[:] = _balanced_labels(len(glom_names), np.arange(N_CHANNELS), rng)
    else:
        raise ValueError(f"알 수 없는 분기: {branch}")
    return out


def pn_channels(pn_glom: np.ndarray, glom_names: np.ndarray, glom_channel: np.ndarray) -> np.ndarray:
    pos = {g: i for i, g in enumerate(glom_names)}
    return glom_channel[[pos[g] for g in pn_glom]]


def channel_matrix(J_pn: np.ndarray, pn_channel: np.ndarray) -> np.ndarray:
    """J_ic = Σ_{p∈c} J^pn_ip. 모양 (N_KC, 9)."""
    J = np.zeros((J_pn.shape[0], N_CHANNELS), dtype=np.int64)
    for c in range(N_CHANNELS):
        J[:, c] = J_pn[:, pn_channel == c].sum(axis=1)
    return J


# ---------------------------------------------------------------- 채널 이득 (규칙 4)

def channel_gains(J: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """S_c = Σ_i J_ic, α_c = S̄ / S_c. S_c = 0인 채널이 있으면 ok = False (α는 NaN)."""
    S = J.sum(axis=0).astype(np.float64)
    ok = bool((S > 0).all())
    with np.errstate(divide="ignore"):
        alpha = np.where(S > 0, S.mean() / np.where(S > 0, S, 1.0), np.nan)
    return S, alpha, ok


# ---------------------------------------------------------------- 분기 P 전환 조건 (규칙 5)

def branch_p_conditions(raw: RawCircuit, pheromone: tuple[str, ...] | list[str],
                        max_ratio: float) -> dict:
    """조건 1: 페로몬 후보 사구체 중 하나라도 원본(4.1 통과)에 없음.
    조건 2: 네 페로몬 사구체의 도달 KC 수 max/min > max_ratio (원본 J_pn 기준, 시드와 무관)."""
    present = {g: bool((raw.glom_names == g).any()) for g in pheromone}
    reach = {}
    for g in pheromone:
        m = raw.pn_glom == g
        reach[g] = int((raw.J_pn[:, m].sum(axis=1) > 0).sum()) if m.any() else 0
    lo = min(reach.values())
    ratio = max(reach.values()) / lo if lo > 0 else float("inf")
    cond1 = not all(present.values())
    cond2 = ratio > max_ratio
    return {"present": present, "reach": reach, "ratio": ratio,
            "cond1": cond1, "cond2": cond2, "use_g": cond1 or cond2}


# ---------------------------------------------------------------- 최종 유효성 검사 (6.5)

def jaccard_matrix(reach: np.ndarray) -> np.ndarray:
    """reach (N_KC, C) bool → (C, C) 자카드. 둘 다 빈 집합이면 NaN."""
    r = reach.astype(np.int64)
    inter = r.T @ r
    size = r.sum(axis=0)
    union = size[:, None] + size[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(union > 0, inter / np.maximum(union, 1), np.nan)


def validate_seed(raw: RawCircuit, J: np.ndarray, labels: np.ndarray) -> dict:
    S, alpha, s_ok = channel_gains(J)
    reach = J > 0
    jac = jaccard_matrix(reach)
    off = jac[~np.eye(N_CHANNELS, dtype=bool)]
    groups_ok = all((labels == a).any() for a in range(N_ACTIONS))
    conj = reach[:, list(ODOR_CHANNELS)].any(axis=1) & reach[:, Z_CHANNEL]
    return {
        "ok": s_ok and groups_ok,
        "fail_reasons": ([] if s_ok else ["S_c = 0인 채널"]) + ([] if groups_ok else ["빈 행동 묶음"]),
        "S": S, "alpha": alpha,
        "reach_count": reach.sum(axis=0),
        "jaccard": jac,
        "jaccard_offdiag_mean": float(np.nanmean(off)) if np.isfinite(off).any() else float("nan"),
        "jaccard_offdiag_max": float(np.nanmax(off)) if np.isfinite(off).any() else float("nan"),
        "conjunction_fraction": float(conj.mean()),
    }


def circuit_checks(raw: RawCircuit) -> dict:
    """시드와 무관한 검사 (보고용)."""
    pn_kc = (raw.J_pn > 0).sum(axis=0)
    kc_in = raw.J_pn.sum(axis=1)
    return {
        "pn_without_kc": int((pn_kc == 0).sum()),
        "pn_kc_count": pn_kc,
        "kc_input": kc_in,
        "kc_without_mbon": int(((raw.W0 > 0).sum(axis=1) == 0).sum()),
    }
