"""초파리 뇌 (TRD 5.5, 10절; M1 뇌 작업 지시서 ② 2절; DECISIONS 10.3 · 12절).

한 틱 계산 (살아 있는 게임의 네 초파리 모두, 집 안 · 방출 전 포함):
    p̃_c = α_c p_c                    (PN 입력 사본에만 α_c. 본능항에는 곱하지 않는다)
    u_i = Σ_c J_ic p̃_c
    k_i = 1[u_i > θ 그리고 상위 k]     (k = ceil(ρ N_KC), 동률은 게임별 난수 "kc_tiebreak")
    m_a = (1/(k|a|)) Σ_{j∈a} Σ_i (w_ij - w0_ij) k_i     (기준선 빼기, DECISIONS 10.3)
    x_a ← (1-κ) x_a + m_a + g odor_a,   h_i ← (1-κ) h_i + k_i
decide가 참이면 π_a ∝ exp(β x_a) (합법 행동만), a* ~ π (게임별 난수 "brain_action").

`propose`는 틱당 한 번 증거(x, h)를 누적하고 결정 시점이면 (π, a*)를 보관한다. 학습 상태(w, e, R̄, R_n)는
바꾸지 않는다. 같은 틱 중복 호출은 오류 (깃발은 조종기 전체에 하나: propose가 세우고 commit이 내린다).

학습 상태는 변화분 dw = w - w0 하나뿐이다 (DECISIONS 12.3). 갱신은 dw ← max(dw + Δw, -w0) (w ≥ 0과 같다).
w가 필요하면(시각화 등) w0 + dw로 만든다. 저장 · 재개도 dw만 쓴다. w0은 불변 참조 배열 하나를 모든 게임 ·
초파리가 읽기 전용으로 공유한다. 회로는 실험 회로 파일 `circuit.file`(mb_v1.npz)을 직접 읽고 meta를 대조한다
(DECISIONS 25.1; 저장 형식 3). B0 = 같은 코드, w = w0에서 시작, η = 0 (지시서 2.5).
"""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from flypac.brain import assign as A
from flypac.brain.base import GhostController
from flypac.core.config import Config, load_config
from flypac.core.rng import RngGroup
from flypac.core.types import FlyObs, GameView

REPO = Path(__file__).resolve().parents[3]
RNG_PURPOSES = ("kc_tiebreak", "brain_action")
LEARNING_CONDITIONS = ("regime1", "regime2", "regime3")
B0 = "b0"
STATE_FORMAT = 3  # 3: circuit_hash = 실험 회로 파일(mb_v1.npz)의 sha256 (DECISIONS 25.1). 2: w 대신 dw만 저장, 설정 변경 허용 기록 (PR #7 검토)


def repo_path(path) -> Path:
    """설정의 상대 경로는 저장소 루트 기준."""
    p = Path(path)
    return p if p.is_absolute() else REPO / p


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(repo_path(path), "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_brain_config(path) -> Config:
    return load_config(repo_path(path))


# ---------------------------------------------------------------- 회로

@dataclass(frozen=True)
class Circuit:
    """뇌가 쓰는 회로. 배열은 모두 읽기 전용."""
    J: np.ndarray            # (N_KC, 9) 채널 단위
    alpha: np.ndarray        # (9,) 채널 이득 α_c
    w0: np.ndarray           # (N_KC, N_MBON) 정규화된 초기 가중치 (불변 참조)
    M: np.ndarray            # (N_KC, N_MBON) bool 연결 마스크
    mbon_bundle: np.ndarray  # (N_MBON,) 행동 묶음 번호 (자기중심)
    pn_channel: np.ndarray   # (N_PN,) 채널 번호
    source_hash: str         # 회로 파일 sha256
    seed: int                # 배정 시드

    @property
    def n_kc(self) -> int:
        return self.J.shape[0]

    @property
    def n_mbon(self) -> int:
        return self.w0.shape[1]


def _readonly(a: np.ndarray) -> np.ndarray:
    a = np.array(a, copy=True)
    a.setflags(write=False)
    return a


def build_circuit(raw_path, assign_cfg: Config | dict, seed: int) -> Circuit:
    """원본 회로 파일과 배정 설정 · 시드로 채널 단위 회로를 만든다 (DECISIONS 10.1 규칙 2 · 3 · 4)."""
    raw = A.load_raw(repo_path(raw_path))
    acfg = assign_cfg
    pher = list(acfg["pheromone_glomeruli"])
    branch = str(acfg["branch"])
    if branch == "auto":
        branch = "G" if A.branch_p_conditions(raw, pher, float(acfg["branch_p_max_ratio"]))["use_g"] else "P"
    w0 = A.normalized_w0(raw.W0)
    labels, _ = A.assign_mbon_groups(A.class_mass(w0, raw.kc_class), int(acfg["mbon_groups"]["seed"]),
                                     int(acfg["mbon_groups"]["max_passes"]))
    gch = A.assign_pn_channels(raw.glom_names, int(seed), branch, pher)
    pch = A.pn_channels(raw.pn_glom, raw.glom_names, gch)
    J = A.channel_matrix(raw.J_pn, pch)
    _, alpha, ok = A.channel_gains(J)
    if not ok:
        raise ValueError(f"배정 시드 {seed}: S_c = 0인 채널이 있어 α_c를 정할 수 없다 (DECISIONS 10.1 규칙 4)")
    return Circuit(J=_readonly(J.astype(np.float64)), alpha=_readonly(alpha), w0=_readonly(w0),
                   M=_readonly(raw.W0 > 0), mbon_bundle=_readonly(labels.astype(np.int64)),
                   pn_channel=_readonly(pch), source_hash=file_sha256(raw_path), seed=int(seed))


def circuit_from_config(cfg: Config) -> Circuit:
    """원본 + 배정 설정 + 시드로 다시 조립한다. 닫힌 시험 · 진단 · 회로 파일 생성용 (뇌는 쓰지 않는다)."""
    c = cfg.circuit
    return build_circuit(c.raw, load_config(repo_path(c.assign)), int(c.seed))


def experiment_circuit_meta_path(npz_path) -> Path:
    """`mb_v1.npz` -> `mb_v1.meta.json`."""
    p = repo_path(npz_path)
    return p.with_name(p.stem + ".meta.json")


def check_experiment_circuit_meta(cfg: Config, meta: dict) -> None:
    """회로 파일의 meta를 뇌 설정 · 실제 파일과 대조한다. 하나라도 다르면 ValueError (DECISIONS 25.1)."""
    c = cfg.circuit

    def need(item: str, got, want) -> None:
        if got != want:
            raise ValueError(f"실험 회로 meta 불일치 [{item}]: meta {got!r} != 현재 {want!r}")

    need("npz_sha256", meta.get("npz_sha256"), file_sha256(c.file))
    need("raw_sha256", meta.get("raw_sha256"), file_sha256(c.raw))
    need("assign_config_hash", meta.get("assign_config_hash"), load_config(repo_path(c.assign)).hash())
    need("assign_seed", meta.get("assign_seed"), int(c.seed))
    kc = meta.get("kc", {})
    need("kc.rho", kc.get("rho"), float(cfg.kc.rho))
    need("kc.theta", kc.get("theta"), float(cfg.kc.theta))
    need("hyperparameter_hash", meta.get("hyperparameter_hash"), Config(cfg.to_dict()["learn"]).hash())


def load_experiment_circuit(cfg: Config) -> Circuit:
    """`circuit.file`(mb_v1.npz)을 읽어 회로를 만든다 (DECISIONS 25.1). 읽기 전에 meta를 대조한다.

    source_hash = npz 파일의 sha256, seed = meta의 assign_seed. 배열은 파일 그대로(dtype 포함) 읽기 전용.
    """
    meta = json.loads(experiment_circuit_meta_path(cfg.circuit.file).read_text(encoding="utf-8"))
    check_experiment_circuit_meta(cfg, meta)
    with np.load(repo_path(cfg.circuit.file)) as z:
        arr = {k: _readonly(z[k]) for k in ("J", "alpha", "w0", "M", "mbon_bundle", "pn_channel")}
    return Circuit(**arr, source_hash=file_sha256(cfg.circuit.file), seed=int(meta["assign_seed"]))


# ---------------------------------------------------------------- 순수 계산 (표현력 시험에서도 쓴다)

def pn_input(odor: np.ndarray, z: np.ndarray, phero: np.ndarray) -> np.ndarray:
    """채널 순서 (냄새 좌 · 앞 · 우 · 뒤, z, 페로몬 좌 · 앞 · 우 · 뒤). 모양 (..., 9). α_c는 곱하지 않는다."""
    z = np.asarray(z, dtype=np.float64)[..., None]
    return np.concatenate([np.asarray(odor, np.float64), z, np.asarray(phero, np.float64)], axis=-1)


def kc_drive(p: np.ndarray, circuit: Circuit) -> np.ndarray:
    """u = J (α ⊙ p). p (..., 9) → (..., N_KC)."""
    return (p * circuit.alpha) @ circuit.J.T


def kc_select(u: np.ndarray, theta: float, k: int, row_rngs) -> np.ndarray:
    """k_i = 1[u_i > θ 그리고 u_i가 상위 k 안]. u (R, N_KC) → (R, N_KC) bool.

    부등호는 엄격하다. 상위 k 경계의 동률은 행마다 row_rngs[r]로 끊는다 (u에 잡음을 더하지 않는다).
    난수는 경계에 실제로 동률이 걸린 행에서만 뽑는다.
    """
    u = np.asarray(u)
    out = u > theta
    n_on = out.sum(axis=1)
    over = np.flatnonzero(n_on > k)
    if over.size == 0:
        return out
    kth = np.partition(u[over], -k, axis=1)[:, -k]  # k번째로 큰 값 (> θ: 켜질 후보가 k개를 넘으므로)
    for r, v in zip(over, kth):
        row = u[r]
        above = row > v
        ties = np.flatnonzero(row == v)
        slots = k - int(above.sum())
        sel = above
        if slots == ties.size:
            sel = sel.copy()
            sel[ties] = True
        else:
            sel = sel.copy()
            sel[row_rngs[r].choice(ties, size=slots, replace=False)] = True
        out[r] = sel
    return out


# ---------------------------------------------------------------- 뇌

class FlyBrain(GhostController):
    """views를 받지 않는다 (TRD 원칙 2)."""

    uses_views = False

    def __init__(self, cfg: Config, B: int, seeds, *, condition: str = "regime1",
                 circuit: Circuit | None = None):
        """cfg: 뇌 설정 (configs/brain_v1.yaml). seeds: 칸별 실행 시드 (B,). condition: 실행 조건.

        조건 b0은 w = w0에서 시작하고 η = 0이다 (지시서 2.5). b0p는 없앴다 (오류). b1은 뇌가 아니다.
        """
        condition = str(condition)
        if condition == "b0p":
            raise ValueError("조건 b0p는 없앴다: 기준선 빼기로 B0와 같은 정책이다 (DECISIONS 10.4, 본 실험 5조건)")
        if condition not in LEARNING_CONDITIONS + (B0,):
            raise ValueError(f"초파리 뇌가 받지 않는 조건: {condition!r}")
        seeds = [int(s) for s in seeds]
        if len(seeds) != B:
            raise ValueError("seeds의 길이는 B와 같아야 한다")
        self.cfg, self.B, self.F = cfg, int(B), 4
        self.condition = condition
        self.config_hash = cfg.hash()
        self.circuit = circuit if circuit is not None else load_experiment_circuit(cfg)
        c = self.circuit
        self.n_kc, self.n_mbon = c.n_kc, c.n_mbon
        self.rho, self.theta = float(cfg.kc.rho), float(cfg.kc.theta)
        self.k = int(math.ceil(self.rho * self.n_kc - 1e-9))  # 부동소수점 올림 오차 방지
        if not 1 <= self.k <= self.n_kc:
            raise ValueError(f"k = ceil(ρ N_KC) = {self.k}가 범위 밖이다")
        L = cfg.learn
        self.kappa, self.g, self.beta = float(L.kappa), float(L.g), float(L.beta)
        self.eta = 0.0 if condition == B0 else float(L.eta)
        self.alpha_rbar, self.gamma = float(L.alpha_rbar), float(L.gamma)

        n_act = A.N_ACTIONS
        size = np.bincount(c.mbon_bundle, minlength=n_act)
        if (size == 0).any():
            raise ValueError("빈 행동 묶음이 있다")
        # bundle_mat[j, a] = 1/(k|a|) (j ∈ a): m = (k_vec · dw) @ bundle_mat. 흔적 계수는 같은 값을 쓴다
        self._bundle_mat = np.zeros((self.n_mbon, n_act))
        self._bundle_mat[np.arange(self.n_mbon), c.mbon_bundle] = 1.0 / (self.k * size[c.mbon_bundle])
        self._coef = self._bundle_mat[np.arange(self.n_mbon), c.mbon_bundle]  # (N_MBON,) 1/(k|a(j)|)
        self._neg_w0 = -c.w0  # w ≥ 0 ⇔ dw ≥ -w0

        self.rngs = RngGroup(seeds, RNG_PURPOSES)
        B, F = self.B, self.F
        self.dw = np.zeros((B, F, self.n_kc, self.n_mbon))   # w - w0: 유일한 학습 상태
        self.e = np.zeros_like(self.dw)
        self.rbar = np.zeros((B, F))
        self.Rn = np.zeros((B, F))
        self.x = np.zeros((B, F, n_act))
        self.h = np.zeros((B, F, self.n_kc))
        self._pi = np.zeros((B, F, n_act))
        self._a = np.full((B, F), -1, dtype=np.int64)
        self._has_prop = np.zeros((B, F), dtype=bool)
        self._active = np.zeros((B, F), dtype=bool)  # 마지막 commit에서 받은 학습 활성 (flush_learning용)
        self._proposed = False                        # 같은 틱 중복 propose 감지 (조종기 전체에 하나)
        self.last_pi = np.zeros((B, F, n_act))
        self.last_kc = np.zeros((B, F, self.n_kc), dtype=bool)
        self.last_m = np.zeros((B, F, n_act))
        self._snap = None                              # 이번 틱 propose의 재생 기록용 사본 (replay_info)
        self._dan = np.zeros((B, F))                   # 이번 틱 확정 시점의 R_n - R̄ (확정 아닌 곳은 0)
        self.config_changes: list[dict] = []          # 명시적으로 허용한 설정 변경 재개의 기록 (저장에 따라감)

    # ---- 읽기 ----
    @property
    def w(self) -> np.ndarray:
        """현재 가중치 w = w0 + dw (보기용 새 배열. 상태는 dw뿐이다)."""
        return self.circuit.w0 + self.dw

    @property
    def channel_names(self) -> tuple[str, ...]:
        return A.channel_names()

    def forward(self, obs: FlyObs, rows_b: np.ndarray):
        """게임 rows_b의 (p, u, k, m). 상태는 바꾸지 않는다 (동률 난수만 쓴다)."""
        p = pn_input(obs.odor[rows_b], obs.z[rows_b], obs.phero[rows_b])        # (nb, F, 9)
        u = kc_drive(p, self.circuit)                                            # (nb, F, N_KC)
        nb = len(rows_b)
        tie = self.rngs["kc_tiebreak"]
        row_rngs = [tie[int(b)] for b in rows_b for _ in range(self.F)]          # 행 순서 (b, f)
        kv = kc_select(u.reshape(nb * self.F, self.n_kc), self.theta, self.k, row_rngs)
        kv = kv.reshape(nb, self.F, self.n_kc)
        drive = np.einsum("bfi,bfij->bfj", kv.astype(np.float64), self.dw[rows_b])
        m = drive @ self._bundle_mat
        self._fwd_drive = drive  # replay_info용: Σ_i dw_ij k_i
        return p, u, kv, m

    # ---- 조종기 인터페이스 ----
    def propose(self, obs: FlyObs, alive: np.ndarray, views: list[GameView] | None = None) -> np.ndarray:
        if views is not None:
            raise ValueError("초파리 뇌는 views를 받지 않는다 (TRD 원칙 2)")
        if self._proposed:
            raise RuntimeError("propose를 commit 없이 다시 불렀다: 틱당 한 번만 부른다 (지시서 2.3)")
        self._proposed = True
        alive = np.asarray(alive, dtype=bool)
        out = np.full((self.B, self.F), -1, dtype=np.int64)
        self.last_pi = np.zeros((self.B, self.F, A.N_ACTIONS))
        self._snap = None
        rows = np.flatnonzero(alive)
        if rows.size == 0:
            return out
        p, _, kv, m = self.forward(obs, rows)
        self._snap = (rows, p, kv, self._fwd_drive)
        odor = np.asarray(obs.odor[rows], dtype=np.float64)
        self.x[rows] = (1.0 - self.kappa) * self.x[rows] + m + self.g * odor
        self.h[rows] = (1.0 - self.kappa) * self.h[rows] + kv
        self.last_kc[rows], self.last_m[rows] = kv, m
        act = self.rngs["brain_action"]
        for b in rows:
            for f in np.flatnonzero(obs.decide[b]):
                legal = np.asarray(obs.legal[b, f], dtype=bool)
                if not legal.any():
                    raise ValueError(f"게임 {b} 초파리 {f}: decide가 참인데 합법 행동이 없다")
                s = np.where(legal, self.beta * self.x[b, f], -np.inf)
                s = np.exp(s - s[legal].max())
                pi = s / s.sum()
                a = int(act[int(b)].choice(A.N_ACTIONS, p=pi))
                self._pi[b, f], self._a[b, f], self._has_prop[b, f] = pi, a, True
                self.last_pi[b, f] = pi
                out[b, f] = a
        return out

    def _learn(self, b: int, f: int) -> None:
        """확정 1번: Δw = η (R_n - R̄) e M, dw ← max(dw + Δw, -w0) (w ≥ 0과 같다), R̄ 갱신."""
        if self.eta != 0.0:
            delta = self.eta * (self.Rn[b, f] - self.rbar[b, f])
            if delta != 0.0:
                d = self.dw[b, f]
                d += delta * self.e[b, f] * self.circuit.M
                np.maximum(d, self._neg_w0, out=d)
        self.rbar[b, f] = (1.0 - self.alpha_rbar) * self.rbar[b, f] + self.alpha_rbar * self.Rn[b, f]

    def commit(self, committed: np.ndarray, active: np.ndarray) -> None:
        if not self._proposed:
            raise RuntimeError("commit 전에 이번 틱의 propose가 없다")
        self._proposed = False
        committed = np.asarray(committed, dtype=bool)
        active = np.asarray(active, dtype=bool)
        self._active = active.copy()
        self._dan = np.zeros((self.B, self.F))
        bad = committed & ~self._has_prop
        if bad.any():
            raise ValueError(f"제안하지 않은 초파리를 확정했다: {np.argwhere(bad).tolist()}")
        for b, f in np.argwhere(committed):
            if active[b, f]:
                self._dan[b, f] = self.Rn[b, f] - self.rbar[b, f]          # DAN 신호: 학습 직전의 R_n - R̄
                self._learn(b, f)                                          # 1
                g = -self._pi[b, f]                                        # 2: 1[a = a*] - π_a
                g[self._a[b, f]] += 1.0
                gj = self.beta * g[self.circuit.mbon_bundle] * self._coef  # (N_MBON,)
                e = self.e[b, f]
                e *= self.gamma
                e += np.outer(self.h[b, f], gj)
            self.x[b, f] = 0.0                                             # 3
            self.h[b, f] = 0.0
            self.Rn[b, f] = 0.0
        self._has_prop[:] = False
        self._a[:] = -1

    def replay_info(self) -> dict:
        """재생 로그의 뇌 필드 (TRD 19.13). 실행기가 이번 틱의 propose · commit 뒤에 부른다. 앞 축은 B.

        kc_active (B,F,k) int16: 켜진 KC 번호, 모자라는 자리는 -1.
        pn_input (B,F,9): 채널 9묶음의 입력 p (α_c를 곱하기 전).
        mbon_out (B,F,N_MBON): 개별 MBON의 원래 활동 Σ_i w_ij k_i, w = w0 + dw (기준선 빼기 전. 제안 시점의 w).
        dan (B,F): 확정 시점의 R_n - R̄ (학습 직전 값). 학습이 일어난 확정이 아니면 0.
        brain_on (B,F) bool: 이 틱에 뇌 계산을 한 초파리 (동결된 게임은 거짓).
        """
        B, F = self.B, self.F
        kc = np.full((B, F, self.k), -1, dtype=np.int16)
        pn = np.zeros((B, F, 9))
        mbon = np.zeros((B, F, self.n_mbon))
        on = np.zeros((B, F), dtype=bool)
        if self._snap is not None:
            rows, p, kv, drive = self._snap
            on[rows] = True
            pn[rows] = p
            mbon[rows] = drive + kv.astype(np.float64) @ self.circuit.w0
            for i, b in enumerate(rows):
                for f in range(F):
                    idx = np.flatnonzero(kv[i, f])
                    kc[b, f, :idx.size] = idx
        return {"kc_active": kc, "pn_input": pn, "mbon_out": mbon, "dan": self._dan.copy(), "brain_on": on}

    def give_reward(self, R: np.ndarray) -> None:
        self.Rn += np.asarray(R, dtype=np.float64)

    def on_release(self, mask: np.ndarray) -> None:
        mask = np.asarray(mask, dtype=bool)
        self.e[mask] = 0.0
        self.Rn[mask] = 0.0
        self._active[mask] = True

    def on_eaten(self, mask: np.ndarray) -> None:
        mask = np.asarray(mask, dtype=bool)
        self.x[mask] = 0.0
        self.h[mask] = 0.0

    def flush_learning(self, mask: np.ndarray) -> None:
        """확정 1번만 한 번 (마지막 commit의 활성 값으로). 새 결정은 없다."""
        for b in np.flatnonzero(np.asarray(mask, dtype=bool)):
            for f in range(self.F):
                if self._active[b, f]:
                    self._learn(int(b), f)

    def reset_life(self, mask: np.ndarray) -> None:
        """x, h, e, R_n 초기화. w, R̄ 유지."""
        mask = np.asarray(mask, dtype=bool)
        for arr in (self.x, self.h, self.e, self.Rn):
            arr[mask] = 0.0
        self._has_prop[mask] = False
        self._active[mask] = False

    # ---- 저장 · 재개 (판 경계에서만, TRD 13.6) ----
    def state_dict(self) -> dict:
        """학습 상태는 dw뿐이다 (w는 저장하지 않는다. w0은 회로 파일에서 다시 읽는다). circuit_hash는 mb_v1.npz의 sha256 (형식 3)."""
        return {
            "format": STATE_FORMAT,
            "dw": self.dw.copy(),
            "rbar": self.rbar.copy(),
            "rngs": self.rngs.state_dict(),
            "game_no": self.rngs.game_no.copy(),
            "circuit_hash": self.circuit.source_hash,
            "circuit_seed": self.circuit.seed,
            "config_hash": self.config_hash,
            "condition": self.condition,
            "config_changes": [dict(c) for c in self.config_changes],
        }

    def load_state_dict(self, state: dict, *, allow_config_change: bool = False) -> None:
        """판 경계 재개. x, h, e, R_n과 보관한 제안은 비운다.

        - 회로 파일 해시 · 배정 시드가 다르면 오류.
        - 설정 해시가 다르면 오류 (DECISIONS 12.5: 재개 중 설정이 바뀌면 한 실험의 앞뒤가 다른 조건이 되는데
          드러나지 않는다). allow_config_change=True로 명시할 때만 허용하고, 허용한 사실과 두 해시를
          config_changes에 남긴다 (이후 state_dict에 따라간다).
        """
        if state.get("format") != STATE_FORMAT:
            raise ValueError(f"저장 형식 {state.get('format')} != {STATE_FORMAT}")
        if state["circuit_hash"] != self.circuit.source_hash or int(state["circuit_seed"]) != self.circuit.seed:
            raise ValueError("회로 파일 해시 또는 배정 시드가 현재 회로와 다르다: 다른 회로의 가중치는 불러오지 않는다")
        changes = [dict(c) for c in state.get("config_changes", [])]
        if state["config_hash"] != self.config_hash:
            if not allow_config_change:
                raise ValueError(
                    f"설정 해시가 다르다 (저장 {state['config_hash']}, 현재 {self.config_hash}): 재개 중 설정이 바뀌면 "
                    "한 실험의 앞뒤가 다른 조건이 된다. 의도한 변경이면 allow_config_change=True로 명시한다")
            changes.append({"allowed": True, "saved_config_hash": str(state["config_hash"]),
                            "current_config_hash": self.config_hash,
                            "game_no": [int(g) for g in np.asarray(state["game_no"])]})
            warnings.warn(f"설정 변경을 허용하고 재개한다: {state['config_hash']} → {self.config_hash}", stacklevel=2)
        dw = np.asarray(state["dw"], dtype=np.float64).copy()
        if dw.shape != self.dw.shape:
            raise ValueError(f"dw의 모양 {dw.shape} != {self.dw.shape}")
        if (dw[..., ~self.circuit.M] != 0).any() or (dw < self._neg_w0).any():
            raise ValueError("dw가 연결 마스크 · 비음수(dw ≥ -w0) 조건을 어긴다")
        if self.condition == B0 and (dw != 0).any():
            raise ValueError("B0는 w = w0에서 시작한다: 학습된 w를 불러온 η = 0 정책은 B0가 아니다 (지시서 2.5)")
        self.dw = dw
        self.rbar = np.asarray(state["rbar"], dtype=np.float64).copy()
        self.rngs.load_state_dict(state["rngs"])
        self.config_changes = changes
        self.reset_life(np.ones(self.B, dtype=bool))
        self._proposed = False
