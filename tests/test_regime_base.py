import pytest

from flypac.core.events import Event, Kind
from flypac.regimes.base import fly_index


def test_fly_index_accepts_valid_range():
    for f in range(4):
        assert fly_index(Event(Kind.CATCH, 0, 0, f, True), 4) == f


def test_fly_index_rejects_out_of_range():
    with pytest.raises(ValueError):
        fly_index(Event(Kind.CATCH, 0, 0, 4, True), 4)
    # 사건 생성 단계에서 막히지 않는 종류(DOT, fly=-1)를 잘못 넘긴 경우도 막는다
    with pytest.raises(ValueError):
        fly_index(Event(Kind.DOT, 0, 0, -1, True), 4)
