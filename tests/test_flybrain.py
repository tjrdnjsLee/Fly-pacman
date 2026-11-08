"""초파리 뇌 시험 (TRD 10.6 표, M1 뇌 작업 지시서 ② 3.1 표, DECISIONS 12절)."""

from __future__ import annotations

import copy

import numpy as np
import pytest
from helpers import MAP, make_cfg, place

from flypac.brain import assign as A
from flypac.brain.flybrain import (Circuit, FlyBrain, kc_select, load_brain_config, pn_input)
from flypac.core import directions as d
from flypac.core.config import Config
from flypac.core.rng import make_rng
from flypac.core.runner import build_runner
from flypac.core.types import FlyObs
from flypac.senses.sense import sense

CFG = load_brain_config("configs/brain_v1.yaml")
F = 4
FWD, LEFT = d.EGO_NAMES.index("forward"), d.EGO_NAMES.index("left")


@pytest.fixture(scope="module")
def circuit():
    return FlyBrain(CFG, 1, [0]).circuit


def brain(circuit, B=2, seeds=None, condition="regime1", cfg=CFG, **learn):
    if learn:
        dd = cfg.to_dict()
        dd["learn"] = {**dd["learn"], **learn}
        cfg = Config(dd)
    return FlyBrain(cfg, B, seeds if seeds is not None else list(range(B)), condition=condition,
                    circuit=circuit)


def obs(B, odor=None, phero=None, z=None, legal=None, decide=None) -> FlyObs:
    return FlyObs(
        odor=np.zeros((B, F, 4)) if odor is None else np.asarray(odor, float),
        phero=np.zeros((B, F, 4)) if phero is None else np.asarray(phero, float),
        z=np.zeros((B, F), bool) if z is None else np.asarray(z, bool),
        legal=np.ones((B, F, 4), bool) if legal is None else np.asarray(legal, bool),
        decide=np.zeros((B, F), bool) if decide is None else np.asarray(decide, bool),
    )


def random_obs(rng, B, p_decide=0.5) -> FlyObs:
    legal = rng.random((B, F, 4)) < 0.6
    legal[..., FWD] |= ~legal.any(axis=-1)
    return obs(B, odor=rng.random((B, F, 4)) * 3, phero=rng.random((B, F, 4)) * 2,
               z=rng.random((B, F)) < 0.5, legal=legal, decide=rng.random((B, F)) < p_decide)


def step(b: FlyBrain, o: FlyObs, alive=None, commit_p=1.0, active=True, R=None, rng=None):
    """한 틱: propose → commit(결정한 초파리 중 일부) → give_reward. 행동을 돌려준다."""
    B = b.B
    alive = np.ones(B, bool) if alive is None else alive
    a = b.propose(o, alive)
    keep = np.ones((B, F), bool) if rng is None else rng.random((B, F)) < commit_p
    b.commit((a >= 0) & keep, np.broadcast_to(np.asarray(active, bool), (B, F)).copy())
    if R is not None:
        b.give_reward(R)
    return a


def randomize_dw(b: FlyBrain, rng, scale=0.5):
    """학습된 것처럼 w를 흔든다 (마스크 · 비음수 유지)."""
    dw = rng.normal(0, scale, b.dw.shape) * b.circuit.M
    b.dw = np.maximum(dw, -b.circuit.w0)


# ---------------------------------------------------------------- TRD 10.6

def test_shapes(circuit):
    b = brain(circuit, B=3)
    n_kc, n_mbon = circuit.n_kc, circuit.n_mbon
    assert b.dw.shape == b.e.shape == (3, F, n_kc, n_mbon) and b.w.shape == (3, F, n_kc, n_mbon)
    assert b.x.shape == (3, F, 4) and b.h.shape == (3, F, n_kc)
    assert b.rbar.shape == b.Rn.shape == (3, F)
    a = b.propose(random_obs(np.random.default_rng(0), 3), np.ones(3, bool))
    assert a.shape == (3, F) and b.last_pi.shape == (3, F, 4) and b.last_kc.shape == (3, F, n_kc)
    assert circuit.J.shape == (n_kc, 9) and circuit.alpha.shape == (9,) and circuit.mbon_bundle.shape == (n_mbon,)


@pytest.mark.parametrize("ego", ["forward", "left"])
def test_direction_mapping_odor_only_raises_that_action(circuit, ego):
    a_ego = d.EGO_NAMES.index(ego)
    b = brain(circuit, B=1)
    odor = np.zeros((1, F, 4))
    odor[..., a_ego] = 2.0
    b.propose(obs(1, odor=odor, decide=np.ones((1, F))), np.ones(1, bool))
    for f in range(F):
        others = np.delete(b.x[0, f], a_ego)
        assert b.x[0, f, a_ego] > others.max()
        assert b.last_pi[0, f].argmax() == a_ego


def test_sparsity_at_most_k_and_zero_without_input(circuit):
    b = brain(circuit, B=2)
    rng = np.random.default_rng(1)
    for _ in range(20):
        step(b, random_obs(rng, 2))
        assert (b.last_kc.sum(axis=-1) <= b.k).all()
    b.propose(obs(2), np.ones(2, bool))  # 무입력 (z = 0, 냄새 · 페로몬 0)
    assert b.last_kc.sum() == 0


def _tiny_circuit() -> Circuit:
    rng = np.random.default_rng(7)
    n_kc, n_mbon = 8, 6
    J = rng.integers(1, 30, size=(n_kc, 9)).astype(float)
    J[rng.random(J.shape) < 0.3] = 0
    w0 = rng.random((n_kc, n_mbon)) + 0.2
    M = rng.random((n_kc, n_mbon)) < 0.8
    w0 = w0 * M
    return Circuit(J=J, alpha=np.linspace(0.5, 1.5, 9), w0=w0, M=M,
                   mbon_bundle=np.array([0, 1, 2, 3, 0, 2]), pn_channel=np.zeros(1, int),
                   source_hash="tiny", seed=0)


def _tiny_cfg(**learn) -> Config:
    dd = copy.deepcopy(CFG.to_dict())
    dd["kc"] = {"rho": 0.5, "theta": 0.0}
    dd["learn"].update(learn)
    return Config(dd)


def test_gradient_trace_matches_finite_difference_of_log_pi():
    """첫 결정의 흔적 e_ij = ∂ ln π(a*)/∂w_ij (e는 0에서 시작, 증거는 한 틱)."""
    circ = _tiny_circuit()
    cfg = _tiny_cfg(beta=1.3, g=0.7, kappa=0.2, gamma=0.9)
    rng = np.random.default_rng(3)
    o = obs(1, odor=rng.random((1, F, 4)), phero=rng.random((1, F, 4)), z=np.ones((1, F)),
            decide=np.ones((1, F)))
    dw0 = np.maximum(rng.normal(0, 0.3, (1, F) + circ.w0.shape) * circ.M, -circ.w0)

    b = FlyBrain(cfg, 1, [0], circuit=circ)
    b.dw = dw0.copy()
    a = b.propose(o, np.ones(1, bool))
    b.commit(np.ones((1, F), bool), np.ones((1, F), bool))

    def log_pi(dw):
        bb = FlyBrain(cfg, 1, [0], circuit=circ)
        bb.dw = dw
        bb.propose(o, np.ones(1, bool))
        return np.log(bb.last_pi[0, np.arange(F), a[0]])

    eps = 1e-6
    fd = np.zeros_like(b.e)
    for i in range(circ.n_kc):
        for j in range(circ.n_mbon):
            up, dn = dw0.copy(), dw0.copy()
            up[:, :, i, j] += eps
            dn[:, :, i, j] -= eps
            fd[0, :, i, j] = (log_pi(up) - log_pi(dn)) / (2 * eps)
    assert np.allclose(b.e, fd, atol=1e-7), np.abs(b.e - fd).max()


def test_no_learning_in_b0(circuit):
    b = brain(circuit, condition="b0")
    assert b.eta == 0.0 and not b.dw.any()
    rng = np.random.default_rng(2)
    for _ in range(50):
        step(b, random_obs(rng, 2), R=rng.normal(0, 1, (2, F)))
    b.flush_learning(np.ones(2, bool))
    assert not b.dw.any()


def test_inactive_decisions_leave_w_rbar_e_unchanged(circuit):
    b = brain(circuit)
    rng = np.random.default_rng(4)
    randomize_dw(b, rng)
    w, rbar, e = b.dw.copy(), b.rbar.copy(), b.e.copy()
    for _ in range(30):
        step(b, random_obs(rng, 2), active=False, R=rng.normal(0, 1, (2, F)))
    b.flush_learning(np.ones(2, bool))  # 마지막 commit의 활성이 거짓
    assert np.array_equal(b.dw, w) and np.array_equal(b.rbar, rbar) and np.array_equal(b.e, e)


def test_mask_unconnected_weights_stay_zero(circuit):
    b = brain(circuit, eta=0.5)
    rng = np.random.default_rng(5)
    for _ in range(200):
        step(b, random_obs(rng, 2), R=rng.normal(0, 1, (2, F)))
    assert b.dw.any()
    assert not b.w[..., ~circuit.M].any()
    assert (b.w >= 0).all()


def test_brain_does_not_take_views(circuit):
    b = brain(circuit, B=1)
    assert not b.uses_views
    with pytest.raises(ValueError, match="views"):
        b.propose(obs(1), np.ones(1, bool), views=[object()])


def test_flush_learning_applies_step1_once_with_reward_after_last_commit(circuit):
    b = brain(circuit, B=1, eta=0.1)
    rng = np.random.default_rng(6)
    step(b, random_obs(rng, 1, p_decide=1.0))  # 흔적이 생긴다
    e = b.e.copy()
    assert e.any()
    b.give_reward(np.full((1, F), 1.0))       # 포획 보상은 commit 뒤(11번)에 들어온다
    b.flush_learning(np.ones(1, bool))
    expect = np.maximum(0.1 * 1.0 * e * circuit.M, -circuit.w0)
    assert np.allclose(b.dw, expect)
    assert np.allclose(b.rbar, b.alpha_rbar * 1.0)
    b.reset_life(np.ones(1, bool))
    assert not b.e.any() and not b.Rn.any() and not b.x.any() and not b.h.any()
    assert np.allclose(b.dw, expect)  # w, R̄는 목숨을 넘어 유지


def test_unexecuted_proposal_does_not_touch_trace_and_trace_uses_sampling_pi(circuit):
    b = brain(circuit, B=1, gamma=0.0)
    rng = np.random.default_rng(8)
    o = random_obs(rng, 1, p_decide=1.0)
    a = b.propose(o, np.ones(1, bool))
    pi = b.last_pi.copy()
    b.commit(np.zeros((1, F), bool), np.ones((1, F), bool))  # 실행되지 않음 → 버림
    assert not b.e.any()
    a = b.propose(o, np.ones(1, bool))
    pi, h = b.last_pi.copy(), b.h.copy()
    b.commit(np.ones((1, F), bool), np.ones((1, F), bool))
    for f in range(F):
        g = -pi[0, f]
        g[a[0, f]] += 1
        coef = 1.0 / (b.k * np.bincount(circuit.mbon_bundle)[circuit.mbon_bundle])
        expect = b.beta * np.outer(h[0, f], g[circuit.mbon_bundle] * coef)
        assert np.allclose(b.e[0, f], expect)


def test_release_resets_e_and_reward_but_keeps_x_h(circuit):
    b = brain(circuit)
    rng = np.random.default_rng(9)
    step(b, random_obs(rng, 2, p_decide=1.0))
    b.propose(random_obs(rng, 2, p_decide=0.0), np.ones(2, bool))
    b.commit(np.zeros((2, F), bool), np.ones((2, F), bool))
    b.give_reward(np.ones((2, F)))
    x, h, e = b.x.copy(), b.h.copy(), b.e.copy()
    assert e.any() and x.any() and h.any()
    b.on_release(np.array([True, False]))
    assert not b.e[0].any() and not b.Rn[0].any()
    assert np.array_equal(b.e[1], e[1]) and b.Rn[1].all()
    assert np.array_equal(b.x, x) and np.array_equal(b.h, h)


def test_eaten_clears_x_h_keeps_e(circuit):
    b = brain(circuit)
    rng = np.random.default_rng(10)
    step(b, random_obs(rng, 2, p_decide=1.0))
    b.propose(random_obs(rng, 2, p_decide=0.0), np.ones(2, bool))
    b.commit(np.zeros((2, F), bool), np.ones((2, F), bool))
    e = b.e.copy()
    mask = np.zeros((2, F), bool)
    mask[1, 2] = True
    x, h = b.x.copy(), b.h.copy()
    b.on_eaten(mask)
    assert not b.x[1, 2].any() and not b.h[1, 2].any()
    assert np.array_equal(b.x[~mask], x[~mask]) and np.array_equal(b.h[~mask], h[~mask])
    assert np.array_equal(b.e, e)


# ---------------------------------------------------------------- 지시서 3.1

def test_initial_zero_output_on_random_inputs(circuit):
    b = brain(circuit, B=10)
    rng = np.random.default_rng(11)
    for _ in range(25):  # 25 × 10 게임 × 4 초파리 = 1,000 입력
        o = random_obs(rng, 10)
        _, _, kv, m = b.forward(o, np.arange(10))
        assert kv.any()
        assert np.abs(m).max() <= 1e-12


def test_b0_equivalence_same_pi_as_untrained_brain(circuit):
    b1, b0 = brain(circuit), brain(circuit, condition="b0")
    rng = np.random.default_rng(12)
    for _ in range(100):
        o = random_obs(rng, 2)
        a1 = step(b1, o, R=np.zeros((2, F)))   # 학습 전: 보상 0이면 w = w0 유지
        a0 = step(b0, o, R=np.zeros((2, F)))
        assert np.array_equal(a1, a0)
        assert np.allclose(b1.last_pi, b0.last_pi, rtol=0, atol=1e-12)


def test_alpha_scales_pn_copy_only_not_instinct():
    circ = _tiny_circuit()
    alt = Circuit(**{**circ.__dict__, "alpha": circ.alpha * np.r_[3.0, np.ones(8)]})  # 냄새 좌 채널만
    cfg = _tiny_cfg()
    rng = np.random.default_rng(13)
    o = obs(1, odor=rng.random((1, F, 4)) + 0.5, decide=np.zeros((1, F)))
    xs = []
    for c in (circ, alt):
        b = FlyBrain(cfg, 1, [0], circuit=c)
        b.dw = np.maximum(rng.normal(0, 0.5, b.dw.shape) * c.M, -c.w0)
        b.propose(o, np.ones(1, bool))
        xs.append(b.x - b.last_m)  # 한 틱, 0에서 시작: x = m + g odor
    assert np.allclose(xs[0], xs[1]) and np.allclose(xs[0], cfg.learn.g * o.odor)
    u0 = (pn_input(o.odor, o.z, o.phero) * circ.alpha) @ circ.J.T
    u1 = (pn_input(o.odor, o.z, o.phero) * alt.alpha) @ alt.J.T
    assert not np.allclose(u0, u1)  # PN 입력 사본에는 곱해졌다


def test_w0_unchanged_after_learning_and_readonly(circuit):
    w0 = circuit.w0.copy()
    b = brain(circuit, eta=0.5)
    rng = np.random.default_rng(14)
    for _ in range(300):
        step(b, random_obs(rng, 2), R=rng.normal(0, 1, (2, F)))
    assert b.dw.any()
    assert np.array_equal(circuit.w0, w0)
    assert not np.shares_memory(b.dw, circuit.w0)
    with pytest.raises(ValueError):
        circuit.w0[0, 0] = 1.0


def test_only_committed_decisions_learn_but_evidence_accumulates(circuit):
    b = brain(circuit, eta=0.5)
    rng = np.random.default_rng(15)
    randomize_dw(b, rng)
    step(b, random_obs(rng, 2, p_decide=1.0), R=np.ones((2, F)))  # 흔적 · 보상이 쌓인 상태
    dw, e, rbar = b.dw.copy(), b.e.copy(), b.rbar.copy()
    for _ in range(5):
        x, h = b.x.copy(), b.h.copy()
        o = random_obs(rng, 2, p_decide=1.0)
        b.propose(o, np.ones(2, bool))
        b.commit(np.zeros((2, F), bool), np.ones((2, F), bool))  # 확정 없음 (속도 부족 · 도중 종료 등)
        assert np.array_equal(b.dw, dw) and np.array_equal(b.e, e) and np.array_equal(b.rbar, rbar)
        assert np.allclose(b.x, (1 - b.kappa) * x + b.last_m + b.g * o.odor)
        assert np.allclose(b.h, (1 - b.kappa) * h + b.last_kc)


def test_propose_twice_in_one_tick_is_an_error(circuit):
    b = brain(circuit)
    o = obs(2)
    b.propose(o, np.ones(2, bool))
    with pytest.raises(RuntimeError, match="propose"):
        b.propose(o, np.ones(2, bool))
    b.commit(np.zeros((2, F), bool), np.zeros((2, F), bool))
    b.propose(o, np.ones(2, bool))  # 다음 틱은 된다
    b.commit(np.zeros((2, F), bool), np.zeros((2, F), bool))
    with pytest.raises(RuntimeError, match="commit"):
        b.commit(np.zeros((2, F), bool), np.zeros((2, F), bool))  # propose 없는 commit도 오류


def test_frozen_game_is_untouched_and_gets_minus_one(circuit):
    b = brain(circuit)
    rng = np.random.default_rng(16)
    step(b, random_obs(rng, 2))
    x, h = b.x.copy(), b.h.copy()
    a = b.propose(random_obs(rng, 2, p_decide=1.0), np.array([False, True]))
    assert (a[0] == -1).all() and (a[1] >= 0).all()
    assert np.array_equal(b.x[0], x[0]) and np.array_equal(b.h[0], h[0])
    assert not np.array_equal(b.x[1], x[1])


def test_tie_at_top_k_boundary_broken_by_game_rng():
    u = np.array([[5.0, 3, 3, 3, 3, 1, 0], [5.0, 3, 3, 3, 3, 1, 0]])
    k = 3
    picks = []
    for seed in range(12):
        rngs = [make_rng(seed, 1, "kc_tiebreak"), make_rng(seed, 1, "kc_tiebreak")]
        out = kc_select(u, 0.0, k, rngs)
        assert (out.sum(axis=1) == k).all() and out[:, 0].all() and not out[:, 5:].any()
        assert np.array_equal(out[0], out[1])  # 같은 시드 → 같은 결과
        again = kc_select(u, 0.0, k, [make_rng(seed, 1, "kc_tiebreak")] * 2)
        assert np.array_equal(again[0], out[0])
        picks.append(tuple(np.flatnonzero(out[0])))
    assert len(set(picks)) > 1  # 시드에 따라 동률이 다르게 끊긴다
    # 경계 아래 동률 · 문턱: 엄격한 부등호
    assert not kc_select(np.array([[1.0, 1.0, 0.0]]), 1.0, 2, [None]).any()
    assert kc_select(np.array([[2.0, 1.0, 1.0]]), 0.0, 3, [None]).sum() == 3  # 동률이 경계에 안 걸리면 난수 없음


def test_brain_reproducible_with_same_seeds(circuit):
    runs = []
    for _ in range(2):
        b = brain(circuit, seeds=[3, 4], eta=0.3)
        rng = np.random.default_rng(17)
        acts = [step(b, random_obs(rng, 2), R=rng.normal(0, 1, (2, F))) for _ in range(40)]
        runs.append((np.array(acts), b.w.copy()))
    assert np.array_equal(runs[0][0], runs[1][0]) and np.array_equal(runs[0][1], runs[1][1])


def _play_game(b, rng, T=60):
    acts = []
    for _ in range(T):
        acts.append(step(b, random_obs(rng, b.B), rng=rng, commit_p=0.8, R=rng.normal(0, 1, (b.B, F))))
    every = np.ones(b.B, bool)
    b.flush_learning(every)   # 종료 처리(6.2): flush → reset_life → 새 판 난수
    b.reset_life(every)
    b.rngs.renew(every, b.rngs.game_no + 1)
    return np.array(acts)


def test_resume_at_game_boundary_matches_uninterrupted_run(circuit):
    straight = brain(circuit, seeds=[5, 6], eta=0.3)
    g1 = _play_game(straight, np.random.default_rng(18))
    g2 = _play_game(straight, np.random.default_rng(19))

    first = brain(circuit, seeds=[5, 6], eta=0.3)
    assert np.array_equal(_play_game(first, np.random.default_rng(18)), g1)
    state = first.state_dict()
    resumed = brain(circuit, seeds=[5, 6], eta=0.3)
    resumed.load_state_dict(state)
    assert np.array_equal(_play_game(resumed, np.random.default_rng(19)), g2)
    assert np.array_equal(resumed.w, straight.w) and np.array_equal(resumed.rbar, straight.rbar)
    assert (resumed.rngs.game_no == 3).all()


def test_resume_refuses_other_circuit(circuit):
    state = brain(circuit).state_dict()
    bad = dict(state, circuit_hash="0" * 64)
    with pytest.raises(ValueError, match="회로"):
        brain(circuit).load_state_dict(bad)
    other = brain(circuit).state_dict()
    other["circuit_seed"] = circuit.seed + 1
    with pytest.raises(ValueError, match="회로"):
        brain(circuit).load_state_dict(other)


def test_b0_starts_from_w0_and_refuses_learned_weights(circuit):
    b0 = brain(circuit, condition="b0")
    assert not b0.dw.any() and b0.eta == 0.0
    learned = brain(circuit)
    randomize_dw(learned, np.random.default_rng(20))
    with pytest.raises(ValueError, match="B0"):
        brain(circuit, condition="b0").load_state_dict(learned.state_dict())
    brain(circuit, condition="b0").load_state_dict(brain(circuit).state_dict())  # w = w0이면 된다


def test_b0p_and_b1_are_not_brain_conditions(circuit):
    with pytest.raises(ValueError, match="b0p"):
        brain(circuit, condition="b0p")
    with pytest.raises(ValueError):
        brain(circuit, condition="b1")


def test_committed_without_proposal_is_an_error(circuit):
    b = brain(circuit)
    b.propose(obs(2), np.ones(2, bool))  # decide 없음
    with pytest.raises(ValueError, match="제안"):
        b.commit(np.ones((2, F), bool), np.ones((2, F), bool))


def test_stored_reversal_application_tick_does_not_learn_again(circuit):
    """반전 예약 (뇌 쪽, DECISIONS 4.1 행 7 · 8). 저장 틱에 확정(흔적 · 학습)하고, 예약을 적용하는 틱은
    엔진이 committed 거짓을 주므로(decide도 거짓, 행동 -1) 다시 일어나지 않는다.
    엔진 쪽(저장 · 적용 · 대기 · 목숨 끝 폐기)은 가짜 엔진에 z 변화가 없어 *진짜 엔진에서 시험할 것*."""
    b = brain(circuit, B=1, eta=0.2)
    rng = np.random.default_rng(21)
    o = random_obs(rng, 1, p_decide=1.0)
    a = b.propose(o, np.ones(1, bool))
    b.commit(np.ones((1, F), bool), np.ones((1, F), bool))   # 저장 틱: moved 거짓, committed 참
    b.give_reward(np.ones((1, F)))
    dw, e, rbar = b.dw.copy(), b.e.copy(), b.rbar.copy()
    a2 = b.propose(random_obs(rng, 1, p_decide=0.0), np.ones(1, bool))  # 적용 틱: decide 거짓
    assert (a2 == -1).all()
    b.commit(np.zeros((1, F), bool), np.ones((1, F), bool))  # moved 참, committed 거짓
    assert np.array_equal(b.dw, dw) and np.array_equal(b.e, e) and np.array_equal(b.rbar, rbar)
    assert b.Rn.all()  # 보상은 다음 확정(또는 flush)까지 남는다


def test_channel_names_follow_ego_names(monkeypatch):
    assert A.CHANNEL_NAMES == A.channel_names(d.EGO_NAMES)
    assert A.CHANNEL_NAMES[A.Z_CHANNEL] == "z"
    assert [A.CHANNEL_NAMES[c] for c in A.ODOR_CHANNELS] == [f"odor_{n}" for n in d.EGO_NAMES]
    monkeypatch.setattr(d, "EGO_NAMES", ("L", "F", "R", "B"))
    names = A.channel_names()
    assert names[:4] == ("odor_L", "odor_F", "odor_R", "odor_B") and names[5:] == (
        "phero_L", "phero_F", "phero_R", "phero_B")


def test_sense_to_pn_to_action_use_same_ego_order(circuit):
    """감각 → PN 입력 → 행동 판독 연결: 앞 칸에만 냄새 → 앞 냄새 채널만 켜지고 본능항이 직진을 올린다."""
    r = build_runner(make_cfg(seeds=[0], n_games=1))
    r._reset_all()
    g = r.game.games[0]
    here = int(MAP.cell_id[5, 6])
    place(g, flies={0: here}, fly_dirs={0: d.EAST})
    ahead = int(MAP.nbr[here, d.EAST])
    assert ahead >= 0
    c = np.zeros((1, MAP.nbr.shape[0]))
    c[0, ahead] = 1.0
    views = r.game.views()
    o = sense(c, np.zeros_like(c), views, r.game.decision_info(), 0.01, nbr=MAP.nbr)
    assert o.odor[0, 0, FWD] > 0 and np.count_nonzero(o.odor[0, 0]) == 1
    p = pn_input(o.odor, o.z, o.phero)[0, 0]
    assert p[A.CHANNEL_NAMES.index("odor_forward")] > 0 and np.count_nonzero(p) == 1
    b = brain(circuit, B=1)
    b.propose(o, np.ones(1, bool))
    assert b.x[0, 0].argmax() == FWD


# ---------------------------------------------------------------- 실행기와 함께

def _brain_runner_cfg(**over):
    return make_cfg(ghosts={"kind": "flybrain", "config": "configs/brain_v1.yaml"}, **over)


def test_runner_carries_w_and_rbar_across_games_and_renews_rngs():
    """난수는 판마다 다시 만들지만 w와 R̄은 판을 넘어 이어진다 (관리자 결정, DECISIONS 12절)."""
    r = build_runner(_brain_runner_cfg(seeds=[0, 1], n_games=2))
    b = r.ghosts
    snap = {}
    flush = b.flush_learning

    def spy_flush(mask):
        flush(mask)
        snap["w"], snap["rbar"] = b.w.copy(), b.rbar.copy()

    b.flush_learning = spy_flush
    end = r._end_of_tick
    seen = {"boundaries": 0}

    def spy_end(events, running):
        before = r.game_no.copy()
        end(events, running)
        new = r.game_no != before
        if new.any():
            seen["boundaries"] += 1
            assert np.array_equal(b.w, snap["w"]) and np.array_equal(b.rbar, snap["rbar"])
            for s in np.flatnonzero(new):
                for p in ("kc_tiebreak", "brain_action"):
                    want = make_rng(int(r.cfg.seeds[s]), int(r.game_no[s]), p).bit_generator.state
                    assert b.rngs[p][s].bit_generator.state == want
            assert not b.x[new].any() and not b.e[new].any()

    r._end_of_tick = spy_end
    r.run()
    assert seen["boundaries"] >= 1
    assert b.dw.any()  # 학습이 실제로 일어나 시험이 의미 있다


def test_runner_builds_brain_and_rejects_b0p():
    out = build_runner(_brain_runner_cfg(seeds=[0], n_games=1)).run()
    assert len(out["games"]) == 1
    with pytest.raises(ValueError, match="b0p"):
        build_runner(_brain_runner_cfg(condition="b0p"))
