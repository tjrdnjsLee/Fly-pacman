"""작은 과제 환경 시험 (TRD 10.8, DECISIONS 17.4): 정답 정의, 네 조합 균등, 보상 시점, 난수, 뇌와의 연결.

과제 A ~ D를 학습시켜 성공률을 보는 시험은 여기 없다 (하이퍼파라미터를 정한 뒤에 돌린다).
"""

from __future__ import annotations

import numpy as np
import pytest

from flypac.brain.flybrain import FlyBrain, load_brain_config
from flypac.core.config import load_config
from flypac.tasks import runner
from flypac.tasks.tmaze import FWD, LEFT, RIGHT, TMaze, combos, correct_action

CFG = load_config("configs/tasks_v1.yaml")
BRAIN = "configs/brain_v1.yaml"


@pytest.fixture(scope="module")
def circuit():
    return FlyBrain(load_brain_config(BRAIN), 1, [0]).circuit


def play(env, n_trials, act=None):
    """뇌 없이 환경만: act(obs)가 없으면 좌 · 우를 번갈아 낸다. 반환: 틱별 (obs, R)."""
    out = []
    while env.trials_done < n_trials:
        o = env.obs()
        a = np.full((env.B, 4), -1)
        if o.decide.any():
            a[:, env.learner] = LEFT if act is None else act(o)
        out.append((o, env.step(a)))
    return out


# ---- 정답 정의 ----

@pytest.mark.parametrize("task,cue,z,side,rev,want", [
    ("A", 0, 0, "L", False, LEFT), ("A", 1, 0, "L", False, LEFT),
    ("A", 0, 0, "R", False, RIGHT), ("A", 0, 0, "L", True, RIGHT), ("A", 0, 0, "R", True, LEFT),
    ("B", 0, 0, "L", False, LEFT), ("B", 1, 0, "L", False, RIGHT),
    ("B", 0, 0, "L", True, RIGHT), ("B", 1, 0, "L", True, LEFT),
    ("D", 0, 0, "L", False, LEFT), ("D", 1, 0, "L", False, RIGHT),
    ("C", 0, 0, "L", False, LEFT), ("C", 1, 0, "L", False, RIGHT),      # z = 0: 냄새 쪽
    ("C", 0, 1, "L", False, RIGHT), ("C", 1, 1, "L", False, LEFT),      # z = 1: 반대쪽
    ("C", 0, 0, "L", True, RIGHT), ("C", 0, 1, "L", True, LEFT),
])
def test_correct_action(task, cue, z, side, rev, want):
    assert correct_action(task, cue, z, side, rev) == want


def test_env_correct_matches_definition_and_reward():
    for task in "ABCD":
        env = TMaze(CFG, task, [3, 4], reward_side="R")
        for _ in range(12):
            want = env.correct.copy()
            cue, z = env.cue.copy(), env.z.copy()
            for b in range(2):
                assert want[b] == correct_action(task, int(cue[b]), int(z[b]), "R")
            R = np.zeros((2, 4))
            while True:
                o = env.obs()
                a = np.full((2, 4), -1)
                if o.decide.any():
                    a[:, 0] = want                      # 정답을 낸다
                R = env.step(a)
                if R.any():
                    break
            assert (R[:, 0] == CFG.env.reward_correct).all()


def test_wrong_choice_gets_wrong_reward():
    env = TMaze(CFG, "B", [1])
    wrong = RIGHT if env.correct[0] == LEFT else LEFT
    while True:
        o = env.obs()
        a = np.full((1, 4), -1)
        if o.decide.any():
            a[:, 0] = wrong
        R = env.step(a)
        if R.any():
            break
    assert R[0, 0] == CFG.env.reward_wrong and not env.trial_log[-1]["ok"][0]


# ---- 입력 ----

def junction_obs(task, **kw):
    env = TMaze(CFG, task, [0, 1, 2, 3], **kw)
    for _ in range(env.L):
        env.step(np.full((4, 4), -1))
    return env, env.obs()


def test_task_a_equal_nonzero_both_sides():
    env, o = junction_obs("A")
    f = env.learner
    assert (o.odor[:, f, LEFT] == CFG.env.amp).all() and (o.odor[:, f, RIGHT] == CFG.env.amp).all()
    assert (o.odor[:, f, FWD] == 0).all() and (o.odor[:, f, 3] == 0).all()


def test_task_b_odor_on_one_side_only():
    env, o = junction_obs("B")
    f = env.learner
    for b in range(4):
        side = LEFT if env.cue[b] == 0 else RIGHT
        other = RIGHT if side == LEFT else LEFT
        assert o.odor[b, f, side] == CFG.env.amp and o.odor[b, f, other] == CFG.env.amp * CFG.env.other_side_ratio


def test_non_learner_flies_have_no_input_and_never_decide():
    env, o = junction_obs("C")
    others = [i for i in range(4) if i != env.learner]
    assert (o.odor[:, others] == 0).all() and (o.z[:, others] == 0).all()
    assert not o.legal[:, others].any() and not o.decide[:, others].any()
    assert o.decide[:, env.learner].all()


def test_tick_structure_legal_and_decide():
    env = TMaze(CFG, "B", [0])
    ticks = play(env, 1)
    assert len(ticks) == env.ticks_per_trial == CFG.env.corridor_ticks + 1
    for i, (o, _) in enumerate(ticks):
        junction = i == len(ticks) - 1
        assert o.decide[0, 0] == junction
        assert o.legal[0, 0].tolist() == ([True, False, True, False] if junction else [False, True, False, False])


def test_c_states_have_identical_choices_and_differ_only_in_z():
    env = TMaze(CFG, "C", list(range(8)))
    o = env.obs()
    z = o.z[:, 0]
    assert set(z.tolist()) == {0.0, 1.0}                       # 여덟 칸에 두 상태가 모두 나옴 (시드 고정)
    # 같은 냄새 방향에서 z = 0과 1의 합법 행동은 같다
    envs = TMaze(CFG, "C", list(range(40)))
    for _ in range(envs.L):
        envs.step(np.full((40, 4), -1))
    o3 = envs.obs()
    assert (o3.legal[:, 0] == o3.legal[0, 0]).all()


# ---- 네 조합 균등 ----

@pytest.mark.parametrize("task", ["B", "C"])
def test_balanced_combos_exact(task):
    env = TMaze(CFG, task, [10, 11, 12])
    n_eval = CFG.budget.eval_trials
    play(env, n_eval)
    log = env.trial_log
    cs = combos(task)
    for b in range(3):
        for cue, z in cs:
            n = sum(1 for t in log if t["cue"][b] == cue and t["z"][b] == z)
            assert n == n_eval // len(cs), (task, b, cue, z, n)


def test_balanced_each_block_has_every_combo():
    env = TMaze(CFG, "C", [7])
    play(env, 40)
    for i in range(0, 40, 4):
        blk = {(int(t["cue"][0]), int(t["z"][0])) for t in env.trial_log[i:i + 4]}
        assert len(blk) == 4


# ---- 보상 시점 ----

@pytest.mark.parametrize("n", [0, 1, 5])
def test_reward_arrives_after_n_more_decisions(n):
    env = TMaze(CFG, "D", [0, 1], n_delay=n)
    ticks = play(env, 1)
    n_dec = [i for i, (o, _) in enumerate(ticks) if o.decide[0, 0]]
    rew = [i for i, (_, R) in enumerate(ticks) if R.any()]
    assert len(n_dec) == n + 1                                # 첫 결정 + n번 더
    assert rew == [n_dec[-1]]                                 # 보상은 마지막 결정 직후 (그 틱)
    assert len(ticks) == env.ticks_per_trial
    R = ticks[-1][1]
    assert not R[:, [1, 2, 3]].any() and R[:, 0].all()


def test_delay_junction_spacing_and_filler_state():
    env = TMaze(CFG, "D", [0], n_delay=3)
    ticks = play(env, 1)
    idx = [i for i, (o, _) in enumerate(ticks) if o.decide[0, 0]]
    assert idx[0] == CFG.env.corridor_ticks
    assert np.diff(idx).tolist() == [CFG.env.filler_corridor_ticks + 1] * 3
    assert all(o.z[0, 0] == 0 for o, _ in ticks[idx[0] + 1:])  # 필러는 z = 0


def test_reward_only_for_main_choice_not_filler():
    env = TMaze(CFG, "D", [0], n_delay=2)
    want = env.correct[0]
    seq = []
    while env.trials_done < 1:
        o = env.obs()
        a = np.full((1, 4), -1)
        if o.decide.any():
            a[0, 0] = want if not seq else (RIGHT if want == LEFT else LEFT)   # 첫 결정만 정답, 필러는 오답
            seq.append(1)
        env.step(a)
    assert env.trial_log[0]["ok"][0]


def test_delay_only_for_task_d():
    with pytest.raises(ValueError):
        TMaze(CFG, "B", [0], n_delay=5)
    with pytest.raises(ValueError):
        TMaze(CFG, "A", [0], reward_side="X")


def test_junction_action_must_be_left_or_right():
    env = TMaze(CFG, "A", [0])
    for _ in range(env.L):
        env.step(np.full((1, 4), -1))
    with pytest.raises(ValueError):
        env.step(np.full((1, 4), FWD))


# ---- 난수 ----

def stream(seeds, task="C", n=30, game_no=1):
    env = TMaze(CFG, task, seeds, game_no=game_no)
    play(env, n)
    return {s: [(int(t["cue"][i]), int(t["z"][i])) for t in env.trial_log] for i, s in enumerate(seeds)}


def test_trial_stream_depends_only_on_seed_and_game_no():
    a = stream([5, 6, 7])
    b = stream([7, 5])
    assert a[5] == b[5] and a[7] == b[7]
    assert a[5] != a[6]
    assert stream([5], game_no=2)[5] != a[5]
    assert stream([5])[5] == a[5]


# ---- 뇌와의 연결 ----

def run_ticks(env, brain, n_ticks, hook=None):
    alive = np.ones(env.B, bool)
    active = np.ones((env.B, 4), bool)
    for _ in range(n_ticks):
        o = env.obs()
        a = brain.propose(o, alive)
        brain.commit(o.decide & (a >= 0), active)
        R = env.step(a)
        brain.give_reward(R)
        if hook:
            hook(o, R)


def test_learning_is_applied_at_next_trials_first_decision(circuit):
    env = TMaze(CFG, "B", [0])
    brain = runner.build_brain(BRAIN, [0], circuit=circuit, eta=0.05)
    runner.start(brain)
    n = env.ticks_per_trial
    run_ticks(env, brain, n)                                  # 1시행: 결정과 보상까지
    assert brain.Rn[0, 0] != 0.0                              # 보상은 R_n에 쌓였다
    assert (brain.dw == 0).all()                              # 아직 학습 전 (R_n은 다음 결정 때 쓰인다)
    run_ticks(env, brain, n - 1)                              # 다음 시행의 복도
    assert (brain.dw == 0).all()
    run_ticks(env, brain, 1)                                  # 다음 시행의 첫 결정 확정
    assert (brain.dw != 0).any()                              # 첫 시행의 R_n으로 학습했다
    assert abs(brain.Rn[0, 0]) == 1.0                         # 둘째 시행의 보상이 새로 R_n에 쌓였다


def test_eta_zero_never_changes_weights(circuit):
    env = TMaze(CFG, "C", [0, 1])
    brain = runner.build_brain(BRAIN, [0, 1], circuit=circuit, eta=0.0)
    runner.start(brain)
    runner.run_trials(env, brain, 20)
    runner.finish(brain)
    assert (brain.dw == 0).all()


def test_evaluate_does_not_change_the_brain(circuit):
    env = TMaze(CFG, "C", [0, 1])
    brain = runner.build_brain(BRAIN, [0, 1], circuit=circuit, eta=0.05)
    runner.start(brain)
    runner.run_trials(env, brain, 12)
    dw, rb = brain.dw.copy(), brain.rbar.copy()
    rng_state = brain.rngs.state_dict()["brain_action"]["states"][0]["state"]["state"]
    rec = runner.evaluate(env, brain, 8, g=0.0)
    assert np.array_equal(brain.dw, dw) and np.array_equal(brain.rbar, rb)
    assert brain.rngs.state_dict()["brain_action"]["states"][0]["state"]["state"] == rng_state
    assert rec["ok"].shape == (8, 2)


def test_run_is_deterministic(circuit):
    outs = []
    for _ in range(2):
        env = TMaze(CFG, "C", [3, 4])
        brain = runner.build_brain(BRAIN, [3, 4], circuit=circuit, eta=0.05)
        runner.start(brain)
        rec = runner.run_trials(env, brain, 25)
        outs.append((rec["choice"], brain.dw.copy()))
    assert np.array_equal(outs[0][0], outs[1][0]) and np.array_equal(outs[0][1], outs[1][1])


def test_commits_between_main_decision_and_reward_equal_n(circuit):
    n = 3
    env = TMaze(CFG, "D", [0], n_delay=n)
    brain = runner.build_brain(BRAIN, [0], circuit=circuit, eta=0.05)
    runner.start(brain)
    commits = []
    run_ticks(env, brain, env.ticks_per_trial,
              hook=lambda o, R: commits.append(bool(o.decide[0, 0])))
    assert sum(commits) == n + 1 and commits[-1]


def test_summarize_counts_and_probabilities(circuit):
    env = TMaze(CFG, "C", [0, 1])
    brain = runner.build_brain(BRAIN, [0, 1], circuit=circuit, eta=0.0)
    runner.start(brain)
    rec = runner.run_trials(env, brain, 16)
    s = runner.summarize(rec, "C")
    assert (s["n"] == 4).all() and ((0 <= s["rate"]) & (s["rate"] <= 1)).all()
    assert ((0 < rec["p_correct"]) & (rec["p_correct"] < 1)).all()


def test_run_trials_records_decision_entropy(circuit):
    """결정 때의 π의 엔트로피(자연로그)를 시행마다 기록한다: 0 ≤ H ≤ ln 4, 학습 전 (η = 0)에는 최대에 가깝다."""
    env = TMaze(CFG, "B", [0, 1])
    brain = runner.build_brain(BRAIN, [0, 1], circuit=circuit, eta=0.0)
    runner.start(brain)
    rec = runner.run_trials(env, brain, 8)
    h = rec["entropy"]
    assert h.shape == rec["ok"].shape
    assert ((h >= 0) & (h <= np.log(4) + 1e-9)).all()
    # 학습 전 H/ln2는 0.96 가량 (DECISIONS 18.7, 측정): 여유를 두고 0.85 이상
    assert h.mean() / np.log(2) > 0.85
