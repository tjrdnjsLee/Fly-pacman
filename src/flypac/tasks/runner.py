"""작은 과제 실행기: 환경(TMaze)과 FlyBrain을 틱마다 연결한다 (TRD 10.8). 과제 A ~ D는 여기서 돌리지 않는다.

틱 순서는 core/tick.py와 같다: 관측 → propose → commit(결정이 확정됨) → 환경 진행 → give_reward.
학습 활성은 명시적으로 켠다 (start). 평가는 뇌의 복사본에서 η = 0으로 하므로 학습 상태를 바꾸지 않는다.
"""

from __future__ import annotations

import copy

import numpy as np

from flypac.brain.flybrain import Circuit, FlyBrain, load_brain_config
from flypac.core.config import Config
from flypac.tasks.tmaze import F, TMaze, combos


def build_brain(brain_cfg_path, seeds, *, circuit: Circuit | None = None, **learn) -> FlyBrain:
    """학습 시드마다 한 칸인 뇌. learn: 학습 설정 덮어쓰기 (예: eta=0.0). 조건은 학습이 켜지는 regime1."""
    cfg = load_brain_config(brain_cfg_path)
    if learn:
        dd = cfg.to_dict()
        dd["learn"] = {**dd["learn"], **learn}
        cfg = Config(dd)
    return FlyBrain(cfg, len(seeds), [int(s) for s in seeds], condition="regime1", circuit=circuit)


def start(brain: FlyBrain) -> None:
    """학습 활성 시작 (e = 0, R_n = 0, x와 h와 w는 유지)."""
    brain.on_release(np.ones(brain.B, dtype=bool))


def run_trials(env: TMaze, brain: FlyBrain, n_trials: int) -> dict:
    """n_trials 시행을 돌린다. 반환: 시행별 기록 (n_trials, B) 배열들.

    cue, z, correct_action, choice, ok, p_correct (결정 때의 π[정답], 표본 오차 없는 정답 확률),
    entropy (결정 때의 π의 엔트로피, 자연로그).
    """
    B = env.B
    alive = np.ones(B, dtype=bool)
    active = np.ones((B, F), dtype=bool)
    start_n = env.trials_done
    p_main = np.zeros(B)
    h_main = np.zeros(B)
    pc: list[np.ndarray] = []
    hs: list[np.ndarray] = []
    while env.trials_done - start_n < n_trials:
        obs = env.obs()
        act = brain.propose(obs, alive)
        if obs.decide.any() and env.dec == 0:
            pi = brain.last_pi[:, env.learner]
            p_main = pi[np.arange(B), env.correct].copy()
            h_main = -(pi * np.log(np.where(pi > 0, pi, 1.0))).sum(1)
        before = env.trials_done
        brain.commit(obs.decide & (act >= 0), active)
        R = env.step(act)
        brain.give_reward(R)
        if env.trials_done > before:
            pc.append(p_main.copy())
            hs.append(h_main.copy())
    log = env.trial_log[start_n:start_n + n_trials]
    out = {k: np.stack([t[k] for t in log]) for k in log[0]}
    out["p_correct"] = np.stack(pc)
    out["entropy"] = np.stack(hs)
    return out


def finish(brain: FlyBrain) -> None:
    """종료 직전 남은 보상을 한 번 반영 (flush_learning)."""
    brain.flush_learning(np.ones(brain.B, dtype=bool))


def clone_brain(brain: FlyBrain) -> FlyBrain:
    """학습 상태 · 난수 상태까지 복사한 뇌 (설정과 회로는 읽기 전용이라 공유)."""
    b = copy.copy(brain)
    for k, v in vars(brain).items():
        if isinstance(v, np.ndarray):
            setattr(b, k, v.copy())
    b.rngs = copy.deepcopy(brain.rngs)
    b.config_changes = [dict(c) for c in brain.config_changes]
    return b


def evaluate(env: TMaze, brain: FlyBrain, n_trials: int, *, g: float | None = None, game_no: int = 2) -> dict:
    """가중치를 고정한 평가. 뇌를 복사(clone_brain)해 η = 0으로 돌리고 (원래 뇌는 그대로), 새 판 번호의 시행 열을 쓴다."""
    b = clone_brain(brain)
    b.eta = 0.0
    if g is not None:
        b.g = float(g)
    b.x[:] = 0.0
    b.h[:] = 0.0
    b.Rn[:] = 0.0
    e = env.fresh(game_no)
    return run_trials(e, b, n_trials)


def summarize(rec: dict, task: str) -> dict:
    """조합(냄새 방향, z)별 정답률. 반환: {'combos': [(cue, z)], 'rate': (B, n_combo), 'p_correct': (B, n_combo),
    'n': (n_combo,) 한 시드의 조합별 시행 수}."""
    cs = combos(task)
    ok, pc = rec["ok"], rec["p_correct"]
    rate, pm, n = [], [], []
    for cue, z in cs:
        m = (rec["cue"] == cue) & (rec["z"] == z)          # (n, B)
        n.append(m.sum(0))
        rate.append((ok & m).sum(0) / np.maximum(m.sum(0), 1))
        pm.append((pc * m).sum(0) / np.maximum(m.sum(0), 1))
    return {"combos": cs, "rate": np.stack(rate, 1), "p_correct": np.stack(pm, 1), "n": np.stack(n, 1)}
