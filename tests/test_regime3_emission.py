"""체제 3 페로몬 방출 규칙 (DECISIONS 29): E_k = κ ĉ_k 1[z_k = 0], ĉ_k = c0 (exp(max_a o_{k,a}) − 1)."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml
from helpers import BASE, make_cfg

from flypac.core import directions as d
from flypac.core.config import Config
from flypac.core.runner import build_runner
from flypac.core.types import FlyObs, GameView
from flypac.field.odor import Field
from flypac.maps.original import load_original
from flypac.regimes import make_regime
from flypac.regimes.regime3 import Regime3
from flypac.senses.sense import sense

ROOT = Path(__file__).resolve().parents[1]
MAIN = yaml.safe_load((ROOT / "configs" / "main_v1.yaml").read_text(encoding="utf-8"))
M = load_original()
F = 4
GAIN, C0 = MAIN["phero"]["gain"], MAIN["odor"]["c0"]


def regime(**over) -> Regime3:
    return Regime3(make_cfg(condition="regime3", odor={"c0": C0}, phero={"gain": GAIN}, **over))


def obs_one(odor, z=None, phero=None) -> FlyObs:
    """한 게임의 관측 조각: odor (F,4), z (F,)."""
    odor = np.asarray(odor, dtype=float)
    return FlyObs(odor=odor, phero=np.zeros((F, 4)) if phero is None else np.asarray(phero, dtype=float),
                  z=np.zeros(F, bool) if z is None else np.asarray(z),
                  legal=np.ones((F, 4), bool), decide=np.zeros(F, bool))


def self_concentration(D: float, lam: float) -> np.ndarray:
    """세기 1인 정지 원천이 자기 칸에 만드는 정상 상태 농도 [(D L + λ I)^{-1}]_jj (N,)."""
    nbr = np.asarray(M.nbr)
    n = nbr.shape[0]
    adj = np.zeros((n, n))
    rows, dirs = np.nonzero(nbr >= 0)
    adj[rows, nbr[rows, dirs]] = 1.0
    adj = np.maximum(adj, adj.T)
    L = np.diag(adj.sum(axis=1)) - adj
    return np.diag(np.linalg.inv(D * L + lam * np.eye(n)))


def test_shape_dtype_nonnegative_and_deterministic():
    r = regime()
    o = obs_one(np.random.default_rng(0).random((F, 4)) * 5, z=[0, 1, 0, 1])
    e = r.emission(o)
    assert e.shape == (F,) and e.dtype == np.float64 and (e >= 0).all()
    assert np.array_equal(r.emission(o), e) and np.array_equal(regime().emission(o), e)


def test_hand_made_observation_follows_the_formula():
    odor = np.array([[0.0, 0.0, 0.0, 0.0],     # 최댓값 0 → 0
                     [0.5, 3.0, 1.0, 0.0],     # z = 1 → 0
                     [0.5, 3.0, 1.0, 0.0],     # 최댓값 3.0
                     [2.0, 0.0, 0.0, 4.5]])    # 최댓값 4.5
    e = regime().emission(obs_one(odor, z=[0, 1, 0, 0]))
    assert e[0] == 0.0 and e[1] == 0.0
    np.testing.assert_allclose(e[2:], [GAIN * C0 * (np.exp(3.0) - 1), GAIN * C0 * (np.exp(4.5) - 1)])


@pytest.mark.parametrize("ztype", [bool, int, float])
def test_frightened_fly_does_not_emit_for_any_z_dtype(ztype):
    odor = np.full((F, 4), 2.0)
    e = regime().emission(obs_one(odor, z=np.array([0, 1, 1, 0]).astype(ztype)))
    assert e[1] == e[2] == 0.0 and e[0] == e[3] > 0


def test_pheromone_observation_is_not_used():
    r = regime()
    odor = np.random.default_rng(1).random((F, 4)) * 4
    base = r.emission(obs_one(odor))
    assert np.array_equal(r.emission(obs_one(odor, phero=np.full((F, 4), 7.0))), base)


def test_inverse_of_sense_transform_recovers_concentration():
    c = np.array([1e-4, 0.03, 1.7, 25.0])                     # 초파리마다 가장 진한 이웃 칸의 농도
    odor = np.zeros((F, 4))
    odor[np.arange(F), [0, 1, 2, 3]] = np.log1p(c / C0)       # sense와 같은 식
    odor[np.arange(F), [1, 2, 3, 0]] = np.log1p(0.5 * c / C0)  # 더 옅은 이웃
    np.testing.assert_allclose(regime().emission(obs_one(odor)) / GAIN, c, rtol=1e-12)


def test_emission_from_real_sense_is_gain_times_strongest_neighbor():
    rng = np.random.default_rng(2)
    field = rng.random((1, M.nbr.shape[0])) * 3
    cells = np.array([M.pacman_start, 0, 5, 40])
    view = GameView(
        pac_cell=M.pacman_start, pac_dir=0, lives=3, power=False, power_ticks=0, ghost_streak=0,
        fly_cell=cells, fly_dir=np.full(F, d.NORTH), fly_z=np.zeros(F, bool),
        fly_in_house=np.zeros(F, bool), dots=M.dots.copy(), energizers=M.energizers.copy(),
        fruit=False, tick=0, score=0, learning_active=False, elroy_stage=0, frozen=False,
    )
    obs = sense(field, np.zeros_like(field), [view], (np.ones((1, F, 4), bool), np.zeros((1, F), bool)), C0,
                nbr=M.nbr)
    nbr = np.asarray(M.nbr)[cells]                                    # (F,4), 벽 -1
    strongest = np.where(nbr >= 0, field[0, np.clip(nbr, 0, None)], 0.0).max(axis=1)
    e = regime().emission(obs_one(obs.odor[0], z=obs.z[0]))
    np.testing.assert_allclose(e, GAIN * strongest, rtol=1e-10)


def test_gain_is_derived_from_odor_parameters():
    """κ = 1 / c̄, c̄ = mean_j [(D L + λ I)^{-1}]_jj (29.2). 냄새 파라미터가 바뀌면 여기서 걸린다."""
    cbar = self_concentration(MAIN["odor"]["D"], MAIN["odor"]["lam"]).mean()
    print(f"c_bar = {cbar:.6f}, 1 / c_bar = {1 / cbar:.6f}")
    assert abs(cbar - 1.6885) < 1e-4
    assert abs(1 / cbar - MAIN["phero"]["gain"]) < 1e-4


def test_steady_state_fly_next_to_source_emits_source_strength_on_average():
    """진짜 Field(냄새 파라미터)에 칸 j의 정지 원천 s를 두면, 그 칸의 농도를 읽은 초파리의 방출은 s·c_jj/c̄."""
    D, lam, s = MAIN["odor"]["D"], MAIN["odor"]["lam"], MAIN["odor"]["s"]
    cjj = self_concentration(D, lam)
    n = cjj.shape[0]
    field = Field(M, D, lam)
    every = np.ones(n, bool)                 # 게임 j는 칸 j에 원천을 둔다 (모든 칸을 한꺼번에)
    field.reset(every)
    S = s * np.eye(n)
    for _ in range(1000):                    # e^{-λ·1000} = e^{-50}: 정상 상태
        field.step(S, every)
    conc = np.diag(field.c)                  # 원천 칸의 농도
    np.testing.assert_allclose(conc, s * cjj, rtol=1e-9)
    r = regime()
    emitted = np.empty(n)
    for start in range(0, n, F):             # 한 번에 초파리 F마리씩
        idx = np.arange(start, start + F) % n
        odor = np.zeros((F, 4))
        odor[:, 1] = np.log1p(conc[idx] / C0)
        emitted[idx] = r.emission(obs_one(odor))
    np.testing.assert_allclose(emitted, s * cjj / cjj.mean(), rtol=1e-4)   # gain을 넷째 자리까지 적은 만큼
    np.testing.assert_allclose(emitted.mean(), s, rtol=1e-4)               # 평균 칸에서 s


@pytest.mark.parametrize("section,key", [("phero", "gain"), ("odor", "c0")])
def test_missing_key_raises_with_key_name(section, key):
    data = copy.deepcopy(BASE)
    data["condition"] = "regime3"
    del data[section][key]
    with pytest.raises(ValueError, match=rf"{section}\.{key}"):
        make_regime(Config(data))
    del data[section]
    with pytest.raises(ValueError, match=rf"{section}\.{key}"):
        make_regime(Config(data))


def test_emission_and_sense_read_the_same_c0_key():
    """실행기는 cfg.odor.c0를 sense에 넘기고 (core/runner.py), 체제 3도 같은 키를 읽는다."""
    runner = build_runner(make_cfg(condition="regime3", odor={"c0": 0.037}))
    assert runner.c0 == runner.regime.c0 == 0.037
    assert runner.regime.gain == BASE["phero"]["gain"]


def test_runner_feeds_emission_into_pheromone_field():
    runner = build_runner(make_cfg(condition="regime3", n_games=1))
    runner._reset_all()
    for _ in range(40):
        runner.tick()
    assert runner.phero.c.sum() > 0 and np.isfinite(runner.phero.c).all()
