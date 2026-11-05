import numpy as np
import pytest

from flypac.field.odor import Field
from flypac.maps.original import load_original


@pytest.fixture(scope="module")
def m():
    return load_original()


class Chain:
    """길이 n 일직선 통로 (동서 이웃만)."""

    def __init__(self, n):
        nbr = -np.ones((n, 4), dtype=int)
        nbr[:-1, 1] = np.arange(1, n)
        nbr[1:, 3] = np.arange(n - 1)
        self.nbr = nbr


def _run(f, src_cell, s, ticks, B=1):
    S = np.zeros((B, f.N))
    S[:, src_cell] = s
    f.reset(np.ones(B, bool))
    out = []
    for _ in range(ticks):
        f.step(S, np.ones(B, bool))
        out.append(f.total().copy())
    return np.array(out)


def test_mass_conservation_fixed_source(m):
    D, lam, s = 1.0, 0.05, 2.0
    f = Field(m, D, lam)
    M = _run(f, m.pacman_start, s, 60)[:, 0]
    t = np.arange(1, 61)
    assert np.allclose(M, s / lam * (1 - np.exp(-lam * t)), rtol=1e-9)


def test_p_symmetric_and_nonnegative(m):
    f = Field(m, 1.0, 0.1)
    assert np.allclose(f.P, f.P.T, atol=1e-12)
    _run(f, m.pacman_start, 1.0, 30)
    assert f.c.min() >= -1e-12


def test_lam_must_be_positive(m):
    for lam in (0, -1):
        with pytest.raises(ValueError):
            Field(m, 1.0, lam)


def test_tunnel_odor_reaches_other_side(m):
    left, right = m.cell_id[14, 0], m.cell_id[14, 27]
    f = Field(m, 1.0, 0.05)
    _run(f, left, 1.0, 40)
    c = f.c[0]
    assert c[right] > 0
    # 터널로 이어져 있으므로 반대편 끝칸이 그 옆 칸들보다 진하다
    assert c[right] > c[m.cell_id[14, 26]] > c[m.cell_id[14, 25]]


@pytest.mark.parametrize("D,lam", [(1.0, 0.05), (1.0, 0.2), (0.5, 0.02)])
def test_corridor_decay_length_matches_lattice_formula(D, lam):
    n = 400
    f = Field(Chain(n), D, lam)
    _run(f, 0, 1.0, 3000)  # 정상 상태까지
    c = f.c[0]
    inv_ell = np.arccosh(1 + lam / (2 * D))
    for k in (20, 30, 40):  # 원천과 끝에서 충분히 떨어진 가운데
        assert np.isclose(np.log(c[k] / c[k + 1]), inv_ell, rtol=1e-3)
