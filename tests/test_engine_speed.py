"""TRD 7.2의 속도 누적과 점 감속. 이동 횟수는 해석적 기대값으로 검증한다."""

import pytest

from flypac.engine import game as engine


@pytest.mark.parametrize("speed", [0.0, 0.1, 0.4, 0.5, 0.6, 0.75, 0.8, 0.9, 1.0])
def test_average_speed_and_at_most_one_move_per_tick(speed):
    accumulator = 0.0
    moves = 0
    for tick in range(1, 10001):
        accumulator, moved = engine.advance_accumulator(accumulator, speed)
        assert isinstance(moved, bool)
        moves += moved
        assert abs(moves - tick * speed) <= 1.000001
        assert 0 <= accumulator < 1


@pytest.mark.parametrize("speed,pellet_speed", [(0.8, 0.71), (0.9, 0.79)])
def test_pellet_debt_produces_expected_average_speed(speed, pellet_speed):
    accumulator = 0.0
    moves = 0
    for _ in range(10000):
        accumulator, moved = engine.advance_accumulator(accumulator, speed)
        if moved:
            moves += 1
            accumulator -= speed / pellet_speed - 1
    assert abs(moves / 10000 - pellet_speed) < 0.0002


def test_negative_pellet_debt_and_fraction_are_preserved():
    assert engine.advance_accumulator(-0.2, 0.75) == pytest.approx((0.55, False))
    assert engine.advance_accumulator(0.7, 0.75) == pytest.approx((0.45, True))
