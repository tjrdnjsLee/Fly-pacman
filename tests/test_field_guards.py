import numpy as np
import pytest

from flypac.field.odor import Field


class RingMap:
    """N칸 원형 복도 (4방향 이웃표: 동 = i+1, 서 = i-1, 나머지 벽)."""

    def __init__(self, n=12):
        nbr = -np.ones((n, 4), dtype=int)
        nbr[:, 1] = (np.arange(n) + 1) % n
        nbr[:, 3] = (np.arange(n) - 1) % n
        self.nbr = nbr


def test_lam_must_be_positive():
    for lam in (0.0, -0.1):
        with pytest.raises(ValueError):
            Field(RingMap(), 1.0, lam)


def test_step_before_reset_raises():
    f = Field(RingMap(), 1.0, 0.1)
    with pytest.raises(RuntimeError):
        f.step(np.zeros((2, 12)), np.ones(2, bool))
    with pytest.raises(RuntimeError):
        _ = f.c
    with pytest.raises(RuntimeError):
        f.total()


def test_mask_length_must_match_first_reset():
    f = Field(RingMap(), 1.0, 0.1)
    f.reset(np.ones(3, bool))
    assert f.c.shape == (3, 12)
    with pytest.raises(ValueError):
        f.reset(np.ones(2, bool))
    with pytest.raises(ValueError):
        f.step(np.zeros((3, 12)), np.ones(2, bool))
    with pytest.raises(ValueError):
        f.step(np.zeros((2, 12)), np.ones(3, bool))  # S 모양 불일치


def test_step_skips_frozen_games_and_reset_is_per_game():
    f = Field(RingMap(), 1.0, 0.1)
    f.reset(np.ones(2, bool))
    S = np.zeros((2, 12))
    S[:, 0] = 1.0
    f.step(S, np.array([True, False]))
    assert f.total()[0] > 0 and f.total()[1] == 0
    f.reset(np.array([True, False]))
    assert f.total()[0] == 0
