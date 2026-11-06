"""배정 코드 검사 (M1 뇌 작업 지시서 ① 6절 · 7절). 작은 가짜 원본으로 돈다. 토큰 불필요."""

from __future__ import annotations

import numpy as np
import pytest

from flypac.brain import assign as A

PHER = ("DA1", "VA1v", "VA1d", "DL3")


def fake_raw(n_kc=60, n_mbon=9, seed=0) -> A.RawCircuit:
    rng = np.random.default_rng(seed)
    gloms = np.array(sorted(["DA1", "VA1v", "VA1d", "DL3", "DM1", "DM2", "DC1", "VM2",
                             "VA2", "DL5", "DP1m", "VC1", "V"]))
    pn_glom = np.repeat(gloms, 2)
    n_pn = len(pn_glom)
    J = (rng.random((n_kc, n_pn)) < 0.15) * rng.integers(3, 20, (n_kc, n_pn))
    J[np.arange(n_kc), rng.integers(0, n_pn, n_kc)] = 5          # 모든 KC에 입력 하나 이상
    W0 = (rng.random((n_kc, n_mbon)) < 0.4) * rng.integers(3, 30, (n_kc, n_mbon))
    kc_class = np.array(["gamma", "apbp", "ab"])[np.arange(n_kc) % 3]
    return A.RawCircuit(
        pn_body=np.arange(n_pn, dtype=np.int64) + 100, pn_type=np.char.add(pn_glom, "_adPN"),
        pn_glom=pn_glom, kc_body=np.arange(n_kc, dtype=np.int64) + 1000,
        kc_type=np.array(["KC"] * n_kc), kc_class=kc_class,
        mbon_body=np.arange(n_mbon, dtype=np.int64) + 5000, mbon_type=np.array(["MBON"] * n_mbon),
        J_pn=J.astype(np.int32), W0=W0.astype(np.int32),
        mbon_class_profile=np.full((n_mbon, 3), 1 / 3), glom_names=gloms)


@pytest.fixture
def raw():
    return fake_raw()


@pytest.mark.parametrize("branch", ["P", "G"])
def test_same_seed_same_assignment(raw, branch):
    a = A.assign_pn_channels(raw.glom_names, 3, branch, PHER)
    b = A.assign_pn_channels(raw.glom_names, 3, branch, PHER)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("branch", ["P", "G"])
def test_different_seed_different_assignment(raw, branch):
    outs = {tuple(A.assign_pn_channels(raw.glom_names, s, branch, PHER)) for s in range(5)}
    assert len(outs) > 1


def test_same_glomerulus_same_channel(raw):
    gch = A.assign_pn_channels(raw.glom_names, 0, "G", PHER)
    pch = A.pn_channels(raw.pn_glom, raw.glom_names, gch)
    for g in raw.glom_names:
        assert len(set(pch[raw.pn_glom == g])) == 1


@pytest.mark.parametrize("seed", range(5))
def test_branch_p_pheromone_in_pheromone_channels(raw, seed):
    gch = A.assign_pn_channels(raw.glom_names, seed, "P", PHER)
    pos = {g: i for i, g in enumerate(raw.glom_names)}
    pher_ch = sorted(gch[pos[g]] for g in PHER)
    assert pher_ch == list(A.PHERO_CHANNELS)                      # 하나씩
    rest = np.array([gch[i] for g, i in pos.items() if g not in PHER])
    assert set(rest) <= {*A.ODOR_CHANNELS, A.Z_CHANNEL}
    sizes = np.bincount(rest, minlength=A.N_CHANNELS)[[*A.ODOR_CHANNELS, A.Z_CHANNEL]]
    assert sizes.max() - sizes.min() <= 1


@pytest.mark.parametrize("seed", range(5))
def test_branch_g_balanced(raw, seed):
    gch = A.assign_pn_channels(raw.glom_names, seed, "G", PHER)
    sizes = np.bincount(gch, minlength=A.N_CHANNELS)
    assert sizes.max() - sizes.min() <= 1 and sizes.sum() == len(raw.glom_names)


def test_branch_p_missing_pheromone_raises(raw):
    with pytest.raises(ValueError):
        A.assign_pn_channels(raw.glom_names, 0, "P", ("DA1", "VA1v", "VA1d", "XX9"))


def test_mbon_groups_sizes_and_determinism(raw):
    C = A.class_mass(A.normalized_w0(raw.W0), raw.kc_class)
    l1, info = A.assign_mbon_groups(C, seed=0, max_passes=50)
    l2, _ = A.assign_mbon_groups(C, seed=0, max_passes=50)
    assert np.array_equal(l1, l2)
    size = np.bincount(l1, minlength=A.N_ACTIONS)
    assert size.max() - size.min() <= 1 and (size > 0).all()
    assert info["final_objective"] <= info["greedy_objective"] + 1e-12


def test_channel_gain_identity(raw):
    gch = A.assign_pn_channels(raw.glom_names, 1, "G", PHER)
    J = A.channel_matrix(raw.J_pn, A.pn_channels(raw.pn_glom, raw.glom_names, gch))
    assert J.shape == (len(raw.kc_body), A.N_CHANNELS)
    assert J.sum() == raw.J_pn.sum()
    S, alpha, ok = A.channel_gains(J)
    assert ok
    assert np.allclose(alpha * S, S.mean())


def test_zero_channel_is_seed_failure_not_exception(raw):
    J = np.ones((5, A.N_CHANNELS), dtype=np.int64)
    J[:, 4] = 0
    S, alpha, ok = A.channel_gains(J)
    assert not ok and np.isnan(alpha[4]) and np.isfinite(np.delete(alpha, 4)).all()
    labels = np.arange(8) % A.N_ACTIONS
    v = A.validate_seed(raw, J, labels)
    assert not v["ok"] and v["fail_reasons"]


def test_empty_action_group_fails(raw):
    J = np.ones((5, A.N_CHANNELS), dtype=np.int64)
    v = A.validate_seed(raw, J, np.zeros(8, dtype=int))
    assert not v["ok"]


def test_branch_conditions(raw):
    c = A.branch_p_conditions(raw, PHER, 1.5)
    assert not c["cond1"] and set(c["reach"]) == set(PHER)
    assert c["cond2"] == (c["ratio"] > 1.5)
    c2 = A.branch_p_conditions(raw, ("DA1", "VA1v", "VA1d", "XX9"), 1.5)
    assert c2["cond1"] and c2["use_g"]


def test_conjunction_and_jaccard():
    reach = np.zeros((4, A.N_CHANNELS), dtype=bool)
    reach[0, [0, 4]] = True          # 접합
    reach[1, 0] = True
    reach[2, 4] = True
    jac = A.jaccard_matrix(reach)
    assert jac[0, 4] == pytest.approx(1 / 3)
    assert np.isnan(jac[1, 2])


def test_load_raw_roundtrip(tmp_path, raw):
    p = tmp_path / "raw.npz"
    np.savez_compressed(p, **{f: getattr(raw, f) for f in A.RawCircuit.__dataclass_fields__})
    back = A.load_raw(p)
    assert np.array_equal(back.J_pn, raw.J_pn) and list(back.glom_names) == list(raw.glom_names)
