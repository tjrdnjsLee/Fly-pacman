"""페로몬 본능 세기 k_ph와 페로몬 방출 · 관측 선택의 시험 (DECISIONS 41.3). torch · 체크포인트는 쓰지 않는다.
시드는 0 ~ 99, 임시 폴더.

1 기본값 불변, 2 방출(marker), 3 겹침(초파리별 장), 4 점수 함수, 5 방향, 6 고정 · 범위, 7 초기화, 8 헛돌기,
9 39절(spread_confirm.py의 감싸기)과 같은 정책, 10 꺼진 것과 같음, 11 저장, 12 스크립트, 13 시드.
"""

from __future__ import annotations

import dataclasses
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from flypac.brain.flybrain import FlyBrain, load_brain_config
from flypac.core.config import Config, load_config
from flypac.core.runner import build_runner
from flypac.core.types import FlyObs
from flypac.recording.recorder import Recorder
from flypac.recording.weights import read_weight_file, weight_filename
from flypac.regimes.regime3 import Regime3
from flypac.senses.sense import sense

ROOT = Path(__file__).resolve().parents[1]
BRAIN, BRAIN_K, BRAIN_KPH = "configs/brain_v1.yaml", "configs/brain_k_v1.yaml", "configs/brain_kph_v1.yaml"
CFG, CFG_K, CFG_KPH = load_brain_config(BRAIN), load_brain_config(BRAIN_K), load_brain_config(BRAIN_KPH)
F = 4
CSVS = ("games.csv", "lives.csv", "flies.csv")
PHERO = {"D": 4.0, "lam": 0.05, "gain": 0.5922}   # final_eval_v1 · main_v1의 phero 블록 (D§29.5)


def _run_cfg(brain=BRAIN_K, condition="regime3", phero=None, **over) -> Config:
    """진짜 엔진(integ_v1의 블록) + 규칙 팩맨, 시드 0 · 1, 3판, 판마다 가중치 저장."""
    d = load_config(ROOT / "configs" / "integ_v1.yaml").to_dict()
    d.update(experiment="phgain", condition=condition, seeds=[0, 1], n_games=3,
             pacman={"kind": "rule_based"}, rule_based={"danger_dist": 6, "chase_dist": 8},
             ghosts={"kind": "flybrain", "config": str(brain)}, replay={"keep_games": [], "brain": False},
             weights={"every_games": 1}, phero=dict(PHERO if phero is None else phero))
    d.update(over)
    return Config(d)


def _run(tmp_path, name, cfg):
    runner = build_runner(cfg, Recorder(tmp_path / name, cfg))
    runner.run()
    return tmp_path / name, runner


def _weights(d: Path) -> dict:
    out = {}
    for p in sorted((d / "weights").glob("*.npz")):
        with np.load(p, allow_pickle=False) as z:
            out[p.name] = {k: z[k] for k in z.files}
    return out


@pytest.fixture(scope="module")
def circuit():
    return FlyBrain(CFG, 1, [0]).circuit


def pbrain(circuit, B=1, condition="regime3", evaluate=False, gain=None, **ph) -> FlyBrain:
    """brain_kph_v1 (k = 10 고정)에 phero_gain을 덮어쓴 뇌. 기본은 k_ph를 배운다 (learn 참, η_ph = 1)."""
    d = CFG_KPH.to_dict()
    d["instinct_gain"] = {**d["instinct_gain"], **(gain or {})}
    d["phero_gain"] = {**d["phero_gain"], "learn": True, "eta_ph": 1.0, **ph}
    return FlyBrain(Config(d), B, list(range(B)), condition=condition, circuit=circuit, evaluate=evaluate)


def obs(B, odor=None, phero=None, legal=None, decide=True):
    return FlyObs(odor=np.zeros((B, F, 4)) if odor is None else np.asarray(odor, float),
                  phero=np.zeros((B, F, 4)) if phero is None else np.asarray(phero, float), z=np.zeros((B, F), bool),
                  legal=np.ones((B, F, 4), bool) if legal is None else np.asarray(legal, bool),
                  decide=np.full((B, F), bool(decide)) if np.ndim(decide) == 0 else np.asarray(decide, bool))


def tick(b: FlyBrain, o, active=True, R=None):
    a = b.propose(o, np.ones(b.B, bool))
    b.commit(a >= 0, np.full((b.B, F), bool(active)))
    if R is not None:
        b.give_reward(R)
    return a


# ---------------------------------------------------------------- 1. 기본값 불변

def test_explicit_relay_total_equals_no_keys(tmp_path):
    """`emission: relay`, `view: total`을 명시한 실행과 키가 없는 실행의 CSV · 가중치가 같다 (실행 설정 해시만 다르다)."""
    a, ra = _run(tmp_path, "nokey", _run_cfg())
    b, rb = _run(tmp_path, "explicit", _run_cfg(phero={**PHERO, "emission": "relay", "view": "total"}))
    assert ra.phero_split is None and rb.phero_split is None and rb.phero_view == "total"
    assert rb.regime.emission_kind == "relay" and rb.regime.gain == PHERO["gain"]
    for name in CSVS:
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
    wa, wb = _weights(a), _weights(b)
    assert wa and list(wa) == list(wb)
    for name in wa:
        assert list(wa[name]) == list(wb[name]) and "kph_gain" not in wa[name]
        for k in wa[name]:
            if k != "run_config_hash":
                assert np.array_equal(wa[name][k], wb[name][k]), (name, k)


def test_default_brains_have_no_phero_gain_state(circuit):
    for cfg in (CFG, CFG_K):
        g = FlyBrain(cfg, 1, [0], circuit=circuit)
        assert "phero_gain" not in cfg and not g.ph_on
        assert not any(hasattr(g, n) for n in ("kph_gain", "e_ph", "xph", "G_ph", "eta_ph"))
        assert "kph_gain" not in g.slot_state(0) and "kph_gain" not in g.state_dict()


def test_view_requires_pheromone_field():
    with pytest.raises(ValueError, match="view"):
        build_runner(_run_cfg(condition="regime1", phero={**PHERO, "view": "others"}))
    with pytest.raises(ValueError, match="view"):
        build_runner(_run_cfg(phero={**PHERO, "view": "mine"}))


# ---------------------------------------------------------------- 2. 방출

def test_marker_emission_is_ones_regardless_of_observation():
    reg = Regime3(Config({"phero": {"D": 4.0, "lam": 0.05, "emission": "marker"}, "odor": {"c0": 0.01}}))
    assert reg.emission_kind == "marker" and not hasattr(reg, "gain")         # marker는 gain을 읽지 않는다
    rng = np.random.default_rng(10)
    for _ in range(20):
        o = obs(1, rng.random((1, F, 4)) * 6, rng.random((1, F, 4)) * 6)
        o = dataclasses.replace(o, z=rng.random((1, F)) < 0.5)
        one = FlyObs(o.odor[0], o.phero[0], o.z[0], o.legal[0], o.decide[0])
        em = reg.emission(one)
        assert np.array_equal(em, np.ones(F)) and em.shape == (F,)
    relay = Regime3(Config({"phero": PHERO, "odor": {"c0": 0.01}}))
    assert relay.emission_kind == "relay"
    with pytest.raises(ValueError, match="gain"):
        Regime3(Config({"phero": {"D": 4.0, "lam": 0.05, "emission": "relay"}, "odor": {"c0": 0.01}}))
    with pytest.raises(ValueError, match="emission"):
        Regime3(Config({"phero": {**PHERO, "emission": "pulse"}, "odor": {"c0": 0.01}}))


# ---------------------------------------------------------------- 3. 겹침

def _split_runner(view, emission, ticks=20, seed=11):
    cfg = _run_cfg(phero={**PHERO, "view": view}, weights=None)
    cfg = Config({k: v for k, v in cfg.to_dict().items() if k != "weights"})
    runner = build_runner(cfg)
    rng = np.random.default_rng(seed)
    runner.regime.emission = lambda o: emission(rng)
    runner._reset_all()
    for _ in range(ticks):
        runner.tick()
    return runner


def _view_obs(runner, view):
    views, info = runner.game.views(), runner.game.decision_info()
    old = runner.phero_view
    runner.phero_view = view
    try:
        return runner._split_view(runner.odor.c, views, info)
    finally:
        runner.phero_view = old


def test_per_fly_fields_sum_to_total_field():
    r = _split_runner("others", lambda rng: rng.random(F) * 3)
    B, N = r.B, r.phero.c.shape[1]
    cs = r.phero_split.c.reshape(B, F, N)
    assert r.phero.c.max() > 0 and (cs.sum(axis=1) > 0).any()
    assert np.abs(cs.sum(axis=1) - r.phero.c).max() <= 1e-10 * np.abs(r.phero.c).max()
    # 관측: 읽기가 칸 농도의 ln(1 + c/c0)이므로 c0 (e^φ − 1)은 선형이다 → others + self = total
    c0 = r.c0
    views, info = r.game.views(), r.game.decision_info()
    total = sense(r.odor.c, r.phero.c, views, info, c0, nbr=r.map.nbr).phero
    lin = lambda p: c0 * np.expm1(p)  # noqa: E731
    others, self_ = _view_obs(r, "others"), _view_obs(r, "self")
    assert total.max() > 0 and others.max() > 0 and self_.max() > 0
    np.testing.assert_allclose(lin(others) + lin(self_), lin(total), rtol=1e-10, atol=1e-14)


def test_self_view_equals_total_when_only_this_fly_emits():
    r = _split_runner("self", lambda rng: np.array([rng.random() * 3 + 0.5, 0.0, 0.0, 0.0]))
    views, info = r.game.views(), r.game.decision_info()
    total = sense(r.odor.c, r.phero.c, views, info, r.c0, nbr=r.map.nbr).phero
    self_, others = _view_obs(r, "self"), _view_obs(r, "others")
    assert total[:, 0].max() > 0
    np.testing.assert_allclose(self_[:, 0], total[:, 0], rtol=1e-12, atol=0)
    assert not others[:, 0].any() and not self_[:, 1:].any()


def test_brain_and_emission_receive_the_view_observation():
    """뇌(propose)와 방출 규칙이 받는 페로몬 관측은 초파리별 장에서 읽은 self 관측이다."""
    cfg = _run_cfg(phero={**PHERO, "view": "self", "emission": "marker"})
    cfg = Config({k: v for k, v in cfg.to_dict().items() if k != "weights"})
    r = build_runner(cfg)
    seen, emitted = [], []
    orig_propose, orig_em = r.ghosts.propose, r.regime.emission

    def propose(o, alive, views=None):
        views_now, B = r.game.views(), r.B
        cs = r.phero_split.c.reshape(B, F, -1)
        want = np.stack([sense(r.odor.c, cs[:, f], views_now, r.game.decision_info(), r.c0, nbr=r.map.nbr).phero[:, f]
                         for f in range(F)], axis=1)
        seen.append((o.phero.copy(), want))
        return orig_propose(o, alive, views)

    def emission(one):
        emitted.append(np.asarray(one.phero).copy())
        return orig_em(one)

    r.ghosts.propose, r.regime.emission = propose, emission
    r._reset_all()
    for _ in range(30):
        r.tick()
    assert any(got.any() for got, _ in seen)
    for got, want in seen:
        assert np.array_equal(got, want)
    assert np.array_equal(np.stack(emitted[-r.B:]), seen[-1][0])


# ---------------------------------------------------------------- 4. 점수 함수

def test_score_matches_finite_difference_of_log_pi_in_kph(circuit):
    """s_ph = β g (xph_a* − Σ_b π_b xph_b) = ∂ ln π_a* / ∂k_ph (합법 방향 안)."""
    rng = np.random.default_rng(12)
    legal = rng.random((1, F, 4)) < 0.6
    legal[0, :, 1] = True
    legal[0, 0] = True
    odors = [rng.random((1, F, 4)) * 3 for _ in range(3)]
    pheros = [rng.random((1, F, 4)) * 3 for _ in range(3)]
    kph0 = rng.uniform(-5.0, 5.0, (1, F))
    b = pbrain(circuit)
    b.kph_gain = kph0.copy()
    for od, ph in zip(odors[:-1], pheros[:-1]):
        b.propose(obs(1, od, ph, legal, decide=False), np.ones(1, bool))
        b.commit(np.zeros((1, F), bool), np.ones((1, F), bool))
    a = b.propose(obs(1, odors[-1], pheros[-1], legal), np.ones(1, bool))
    xo, xph = b.xo.copy(), b.xph.copy()
    b.commit(a >= 0, np.ones((1, F), bool))
    k = 1 - b.kappa
    assert np.allclose(xph, k ** 2 * pheros[0] + k * pheros[1] + pheros[2]) and not b.xph.any()
    eps = 1e-5
    for f in range(F):
        def log_pi(kph):
            z = np.where(legal[0, f], b.beta * b.g * (b.k_gain[0, f] * xo[0, f] + kph * xph[0, f]), -np.inf)
            z = z - z[legal[0, f]].max()
            return z[a[0, f]] - np.log(np.exp(z).sum())

        fd = (log_pi(kph0[0, f] + eps) - log_pi(kph0[0, f] - eps)) / (2 * eps)
        assert b.e_ph[0, f] == pytest.approx(fd, abs=1e-8) and abs(fd) > 0
        assert np.exp(log_pi(kph0[0, f])) == pytest.approx(b.last_pi[0, f, a[0, f]], abs=1e-12)


# ---------------------------------------------------------------- 5. 방향

def _phero_task(circuit, away: bool, steps=400, **ph):
    """페로몬이 가장 약한 쪽(away) 또는 가장 센 쪽을 고르면 +1, 아니면 −1. 냄새는 0."""
    b = pbrain(circuit, **ph)
    rng = np.random.default_rng(13)
    for _ in range(steps):
        phero = rng.random((1, F, 4)) * 5
        a = tick(b, obs(1, None, phero))
        target = phero.argmin(axis=-1) if away else phero.argmax(axis=-1)
        b.give_reward(np.where(a == target, 1.0, -1.0))
    return b


def test_kph_shrinks_when_avoiding_is_rewarded_and_grows_when_following(circuit):
    away, toward = _phero_task(circuit, True), _phero_task(circuit, False)
    assert (away.kph_gain < -0.5).all(), away.kph_gain
    assert (toward.kph_gain > 0.5).all(), toward.kph_gain
    for b in (away, toward):
        assert not b.dw.any() and (b.k_gain == 10.0).all()                 # 시냅스 · k는 그대로


# ---------------------------------------------------------------- 6. 고정 · 범위

@pytest.mark.parametrize("kw", [{"condition": "b0"}, {"evaluate": True}, {"learn": False}])
def test_kph_does_not_change_in_b0_evaluate_or_learn_false(circuit, kw):
    ph = {k: v for k, v in kw.items() if k == "learn"}
    b = pbrain(circuit, condition=kw.get("condition", "regime3"), evaluate=kw.get("evaluate", False), **ph)
    assert b.eta_ph == 0.0 and not b.ph_learn
    if kw.get("condition") != "b0":
        b.kph_gain[:] = -3.0
    before = b.kph_gain.copy()
    rng = np.random.default_rng(14)
    for _ in range(60):
        tick(b, obs(1, rng.random((1, F, 4)) * 3, rng.random((1, F, 4)) * 5), R=rng.normal(0, 1, (1, F)))
    b.flush_learning(np.ones(1, bool))
    assert np.array_equal(b.kph_gain, before) and b.e_ph.any()


def test_kph_is_clipped_to_range(circuit):
    for reward, kw, want in ((+100.0, {"kph_max": 1.5}, 1.5), (-100.0, {"kph_min": -0.5}, -0.5)):
        b = pbrain(circuit, **kw)
        tick(b, obs(1, None, np.random.default_rng(15).random((1, F, 4)) * 5))
        b.e_ph[:] = 1.0
        b.give_reward(np.full((1, F), reward))
        b.flush_learning(np.ones(1, bool))
        assert (b.kph_gain == want).all(), b.kph_gain
    with pytest.raises(ValueError, match="kph_min"):
        pbrain(circuit, kph_init=60.0)


def test_phero_gain_needs_instinct_gain(circuit):
    d = CFG.to_dict()
    d["phero_gain"] = dict(CFG_KPH.to_dict()["phero_gain"])
    with pytest.raises(ValueError, match="instinct_gain"):
        FlyBrain(Config(d), 1, [0], circuit=circuit)
    d["phero_gain"]["enabled"] = False                                    # 꺼진 블록은 괜찮다
    assert not FlyBrain(Config(d), 1, [0], circuit=circuit).ph_on


# ---------------------------------------------------------------- 7. 초기화

def test_xph_and_eph_reset_where_xo_and_ek_reset(circuit):
    b = pbrain(circuit, B=2)
    rng = np.random.default_rng(16)

    def same_zeros():
        assert np.array_equal(b.xph == 0, b.xo == 0) and np.array_equal(b.e_ph == 0, b.e_k == 0)

    o = lambda decide=True: obs(2, rng.random((2, F, 4)) * 3 + 0.1, rng.random((2, F, 4)) * 3 + 0.1, decide=decide)  # noqa: E731
    tick(b, o(), R=np.ones((2, F)))
    tick(b, o())
    same_zeros()
    assert not b.xph.any()                                               # 확정에서 0
    b.propose(o(False), np.ones(2, bool))
    b.commit(np.zeros((2, F), bool), np.ones((2, F), bool))
    assert b.xph.all() and b.e_ph.all()
    same_zeros()
    eaten = np.zeros((2, F), bool)
    eaten[1, 2] = True
    b.on_eaten(eaten)
    assert not b.xph[1, 2].any() and b.xph[~eaten].all()
    same_zeros()
    b.on_release(np.array([True, False]))
    assert not b.e_ph[0].any() and b.e_ph[1].all()
    same_zeros()
    kph, G = b.kph_gain.copy(), b.G_ph.copy()
    b.reset_life(np.array([False, True]))
    assert not b.xph[1].any() and not b.e_ph[1].any() and b.xph[0].all()
    same_zeros()
    assert np.array_equal(b.kph_gain, kph) and np.array_equal(b.G_ph, G)


# ---------------------------------------------------------------- 8. 헛돌기

def test_dry_run_accumulates_the_update_without_changing_kph(circuit):
    eps = 1e-7
    dry, real = pbrain(circuit, dry_run=True, eta_ph=1.0), pbrain(circuit, dry_run=False, eta_ph=eps)
    rng = np.random.default_rng(17)
    first = None
    for _ in range(200):
        o = obs(1, rng.random((1, F, 4)) * 3, rng.random((1, F, 4)) * 5)
        R = rng.normal(0, 1, (1, F))
        G0, k0 = dry.G_ph.copy(), real.kph_gain.copy()
        assert np.array_equal(tick(dry, o, R=R), tick(real, o, R=R))
        if first is None and dry.G_ph.any():
            first = (dry.G_ph - G0, real.kph_gain - k0)
            break
    assert first is not None
    dG, dk = first
    assert dG.any() and np.allclose(dk / eps, dG, rtol=1e-6, atol=1e-12)
    for _ in range(100):                                                 # 오래 돌아도 k_ph는 그대로
        tick(dry, obs(1, rng.random((1, F, 4)) * 3, rng.random((1, F, 4)) * 5), R=rng.normal(0, 1, (1, F)))
    assert (dry.kph_gain == 0.0).all() and dry.rbar.any() and dry.e_ph.any() and dry.G_ph.any()


def test_dry_run_in_evaluate_mode_counts_nothing(circuit):
    b = pbrain(circuit, dry_run=True, evaluate=True)
    rng = np.random.default_rng(18)
    for _ in range(50):
        tick(b, obs(1, rng.random((1, F, 4)) * 3, rng.random((1, F, 4)) * 5), R=rng.normal(0, 1, (1, F)))
    assert not b.G_ph.any() and (b.kph_gain == 0.0).all()


# ---------------------------------------------------------------- 9. 39절과 같은 정책

def test_same_policy_as_spread_confirm_wrapper(circuit):
    """새 뇌(k = 10, k_ph = −10)의 π와 spread_confirm.py의 감싸기(k o + k_ph φ를 냄새로 넘김)의 π가 1e-9 안에서 같다."""
    k, kph, B = 10.0, -10.0, 2
    new = FlyBrain(CFG_KPH, B, [0, 1], condition="regime3", circuit=circuit, evaluate=True)
    new.k_gain[:], new.kph_gain[:] = k, kph
    old = FlyBrain(CFG, B, [0, 1], condition="regime3", circuit=circuit, evaluate=True)
    orig = old.propose

    def propose(o, alive, views=None):                                  # spread_confirm.py의 wrap 그대로
        mix = k * np.asarray(o.odor, float) + kph * np.asarray(o.phero, float)
        return orig(dataclasses.replace(o, odor=mix), alive, views)

    old.propose = propose
    rng = np.random.default_rng(19)
    decided = 0
    for t in range(300):
        legal = rng.random((B, F, 4)) < 0.7
        legal[..., 1] = True
        decide = rng.random((B, F)) < 0.4
        o = obs(B, rng.random((B, F, 4)) * 5, rng.random((B, F, 4)) * 5, legal, decide)
        a_new, a_old = tick(new, o, R=rng.normal(0, 1, (B, F))), tick(old, o, R=rng.normal(0, 1, (B, F)))
        assert np.abs(new.last_pi - old.last_pi).max() < 1e-9, t
        assert np.array_equal(a_new, a_old), t
        decided += int(decide.sum())
        if t % 37 == 36:                                                # 가끔 먹힘 · 목숨 끝
            mask = rng.random((B, F)) < 0.3
            new.on_eaten(mask)
            old.on_eaten(mask)
    assert decided > 100 and not new.dw.any() and not old.dw.any()


# ---------------------------------------------------------------- 10. 꺼진 것과 같음

def test_enabled_with_zero_kph_plays_same_games_as_no_block(tmp_path):
    d = CFG_K.to_dict()
    d["phero_gain"] = {**CFG_KPH.to_dict()["phero_gain"]}               # 켜짐, k_ph = 0, learn: false
    path = tmp_path / "brain_k_plus_ph.yaml"
    path.write_text(yaml.safe_dump(d, allow_unicode=True), encoding="utf-8")
    a, ra = _run(tmp_path, "off", _run_cfg(brain=BRAIN_K))
    b, rb = _run(tmp_path, "on", _run_cfg(brain=path))
    assert rb.ghosts.ph_on and not ra.ghosts.ph_on and (rb.ghosts.kph_gain == 0).all()
    assert (ra.ghosts.k_gain != 1.0).any()                              # k는 실제로 배웠다
    for name in CSVS:
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
    wa, wb = _weights(a), _weights(b)
    for name in wa:
        assert np.array_equal(wa[name]["k_gain"], wb[name]["k_gain"]) and np.array_equal(wa[name]["rbar"], wb[name]["rbar"])
        assert (wb[name]["kph_gain"] == 0).all() and "kph_gain" not in wa[name]


# ---------------------------------------------------------------- 11. 저장

def test_weight_file_and_state_dict_roundtrip_kph_gain(tmp_path, circuit):
    d = CFG_KPH.to_dict()
    d["phero_gain"].update(learn=True, eta_ph=5.0)
    path = tmp_path / "brain_kph_learn.yaml"
    path.write_text(yaml.safe_dump(d, allow_unicode=True), encoding="utf-8")
    out, runner = _run(tmp_path, "r", _run_cfg(brain=path, phero={**PHERO, "emission": "marker", "view": "others"}))
    g = runner.ghosts
    assert g.ph_on and (g.kph_gain != 0).any()
    f = read_weight_file(out / "weights" / weight_filename(1, 3))
    assert f["kph_gain"].shape == (F,) and f["kph_gain"].dtype == np.float64 and np.array_equal(f["kph_gain"], g.kph_gain[1])
    fresh = FlyBrain(load_brain_config(path), 2, [0, 1], condition="regime3", evaluate=True)
    fresh.load_slot_state(1, f)
    assert np.array_equal(fresh.kph_gain[1], g.kph_gain[1]) and (fresh.kph_gain[0] == 0).all()
    with pytest.raises(ValueError, match="kph_gain"):                   # 켜졌는데 없다
        fresh.load_slot_state(0, {k: v for k, v in f.items() if k != "kph_gain"})
    off = FlyBrain(CFG_K, 1, [0], condition="regime3", circuit=circuit)
    with pytest.raises(ValueError, match="kph_gain"):                   # 꺼졌는데 있다
        off.load_slot_state(0, {**off.slot_state(0), "kph_gain": np.zeros(F)})
    for bad in (np.ones(F - 1), np.full(F, 51.0), np.full(F, np.nan)):
        with pytest.raises(ValueError, match="kph_gain"):
            fresh.load_slot_state(0, {**f, "kph_gain": bad})
    # state_dict
    st = g.state_dict()
    assert np.array_equal(st["kph_gain"], g.kph_gain) and not np.shares_memory(st["kph_gain"], g.kph_gain)
    other = FlyBrain(load_brain_config(path), 2, [0, 1], condition="regime3", circuit=circuit)
    other.load_state_dict(st)
    assert np.array_equal(other.kph_gain, g.kph_gain) and not other.e_ph.any() and not other.xph.any()
    with pytest.raises(ValueError, match="kph_gain"):
        other.load_state_dict({k: v for k, v in st.items() if k != "kph_gain"})
    with pytest.raises(ValueError, match="kph_gain"):
        FlyBrain(CFG_K, 2, [0, 1], condition="regime3", circuit=circuit).load_state_dict(
            {**st, "config_hash": CFG_K.hash()})
    b0 = FlyBrain(load_brain_config(path), 2, [0, 1], condition="b0", circuit=circuit)
    with pytest.raises(ValueError, match="B0"):
        b0.load_state_dict({**st, "condition": "b0"})


def test_k_curve_reads_kph_gain(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("k_curve_ph", ROOT / "scripts" / "k_curve.py")
    kc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kc)
    for cond in ("regime3",):
        dd = tmp_path / "bundle" / f"{cond}_p1"
        (dd / "weights").mkdir(parents=True)
        (dd / "config.yaml").write_text(yaml.safe_dump({"condition": cond}), encoding="utf-8")
        for seed, vals in ((0, [-1, -2, -3, -2]), (1, [-4, -4, -4, -4]), (2, [0, 0, 0, 0])):
            np.savez(dd / "weights" / weight_filename(seed, 10), k_gain=np.full(F, 10.0), kph_gain=np.asarray(vals, float))
    bundle = tmp_path / "bundle"
    assert kc.main(["k_curve.py", str(bundle), "--key", "kph_gain"]) == 0
    res = kc.build(bundle, "kph_gain")
    ci = res["conditions"]["regime3"]["last_minus_ref"]
    assert res["key"] == "kph_gain" and res["k_ref"] == 0.0 and ci["mean"] == pytest.approx(-2.0)
    assert (bundle / "kph_curve.json").exists() and (bundle / "kph_curve.md").exists() and not (bundle / "k_curve.json").exists()
    assert "key" not in kc.build(bundle)                                # 기본 출력은 그대로
    assert kc.main(["k_curve.py", str(bundle), "--key", "x"]) == 2


# ---------------------------------------------------------------- 12. 스크립트

def _script(name, tmp_path, **env):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "explore" / name)], cwd=ROOT,
                          env={**os.environ, "COOP_TEST": "1", "COOP_ROOT": str(tmp_path), **env},
                          capture_output=True, text=True, encoding="utf-8")


def test_phero_split_script_runs_to_the_end(tmp_path):
    done = _script("phero_split.py", tmp_path)
    assert done.returncode == 0, done.stderr
    res = yaml.safe_load((tmp_path / "phero_split_v1" / "results.json").read_text(encoding="utf-8"))
    assert set(res["conditions"]) == {"solo", "total10", "others10", "self10"} and res["seeds"] == [0, 1]
    assert {c["view"] for c in res["conditions"].values()} == {"total", "others", "self"}
    assert set(res["verdicts"]) == {"1", "2"} and res["next_view"] in ("others", "total")
    assert "41.4 판정 (1)" in done.stdout and "41.4 판정 (2)" in done.stdout
    again = _script("phero_split.py", tmp_path)                         # 결과 파일이 있으면 멈춤
    assert again.returncode != 0 and "멈춤" in again.stderr


def test_phero_gate_script_runs_to_the_end(tmp_path):
    missing = _script("phero_gate.py", tmp_path)
    assert missing.returncode != 0 and "PH_VIEW" in missing.stderr
    done = _script("phero_gate.py", tmp_path, PH_VIEW="others")
    assert done.returncode == 0, done.stderr
    res = yaml.safe_load((tmp_path / "phero_gate_v1" / "results.json").read_text(encoding="utf-8"))
    assert res["view"] == "others" and set(res["q"]) == {"1", "0"}
    for r in res["q"].values():
        assert len(r["G_ph_by_seed"]) == 2 and any(v != 0 for v in r["G_ph_by_seed"])
    runs = sorted(tmp_path.glob("*_phero_gate_v1_q*"))
    assert len(runs) == 2 and not any((d / "weights").exists() for d in runs)


# ---------------------------------------------------------------- 13. 시드

def test_new_seeds_are_used_nowhere_else():
    """9800 ~ 9819는 phero_split.py (9800 ~ 9809) · phero_gate.py (9810 ~ 9819)에만 있다."""
    pat = re.compile(r"(?<!\d)98[01]\d(?!\d)")
    own = {"phero_split.py", "phero_gate.py"}
    for path in sorted((ROOT / "configs").glob("*.yaml")):
        d = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        used = list(d.get("seeds") or []) + list((d.get("final_eval") or {}).get("train_seeds") or [])
        assert not {s for s in used if isinstance(s, int)} & set(range(9800, 9820)), path.name
        assert not pat.search(path.read_text(encoding="utf-8")), path.name
    for path in sorted((ROOT / "scripts").rglob("*.py")):
        if path.name not in own:
            assert not pat.search(path.read_text(encoding="utf-8")), path
    assert "range(9800, 9810)" in (ROOT / "scripts/explore/phero_split.py").read_text(encoding="utf-8")
    assert "range(9810, 9820)" in (ROOT / "scripts/explore/phero_gate.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------- 설정

def test_brain_kph_differs_from_brain_k_only_in_gain_blocks():
    old = yaml.safe_load((ROOT / BRAIN_K).read_text(encoding="utf-8"))
    new = yaml.safe_load((ROOT / BRAIN_KPH).read_text(encoding="utf-8"))
    assert {k for k in set(old) | set(new) if old.get(k) != new.get(k)} == {"instinct_gain", "phero_gain"}
    assert new["instinct_gain"] == {"learn": True, "eta_k": 0.0, "k_init": 10.0, "k_min": 0.0, "k_max": 50.0,
                                    "freeze_synapses": True}
    assert new["phero_gain"] == {"enabled": True, "learn": False, "eta_ph": 0.0, "kph_init": 0.0, "kph_min": -50.0,
                                 "kph_max": 50.0, "dry_run": False}
