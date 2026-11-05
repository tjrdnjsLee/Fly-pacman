import numpy as np
import pytest

from flypac.core import directions as d
from flypac.core.config import load_config
from flypac.core.events import Event, Kind, Winner, ZCause
from flypac.core.rng import make_rng
from flypac.core.types import FlyObs


def test_ego_abs_roundtrip():
    for h in range(4):
        for e in range(4):
            assert d.abs_to_ego(h, d.ego_to_abs(h, e)) == e
        for a in range(4):
            assert d.ego_to_abs(h, d.abs_to_ego(h, a)) == a


def test_ego_order_left_is_odor0():
    # 동쪽을 보면 왼쪽은 북, 앞은 동, 오른쪽은 남, 뒤는 서
    assert [int(d.ego_to_abs(d.EAST, e)) for e in range(4)] == [d.NORTH, d.EAST, d.SOUTH, d.WEST]
    # 북쪽을 보면 왼쪽은 서
    assert int(d.ego_to_abs(d.NORTH, d.LEFT)) == d.WEST
    assert int(d.ego_to_abs(d.NORTH, d.BACK)) == d.SOUTH


def test_ego_vectorized():
    h = np.array([[0, 1], [2, 3]])
    assert d.ego_to_abs(h, d.FORWARD).tolist() == h.tolist()


def test_pacman_ai_roundtrip_and_mapping():
    # 북->상 0, 동->우 3, 남->하 1, 서->좌 2
    assert [int(d.to_pacman_ai(a)) for a in range(4)] == [0, 3, 1, 2]
    for a in range(4):
        assert int(d.from_pacman_ai(d.to_pacman_ai(a))) == a


def test_event_format():
    ev = Event("CATCH", game=0, tick=5, fly=2, active=True)
    assert ev.kind == "CATCH" and ev.kind is Kind.CATCH and ev.cause is None
    over = Event(Kind.GAME_OVER, 0, 9, -1, False, cause="FLIES")
    assert over.winner is Winner.FLIES
    assert Event(Kind.Z_CHANGE, 0, 1, 3, True, ZCause.REVIVE).cause is ZCause.REVIVE
    with pytest.raises(ValueError):
        Event(Kind.CATCH, 0, 1, -1, True)
    with pytest.raises(ValueError):
        Event(Kind.DOT, 0, 1, 0, True)
    with pytest.raises(ValueError):
        Event(Kind.GAME_OVER, 0, 1, -1, True, cause="NOBODY")
    with pytest.raises(ValueError):
        Event(Kind.Z_CHANGE, 0, 1, 0, True)


def test_flyobs_validate():
    ok = FlyObs(np.zeros((2, 4, 4)), np.zeros((2, 4, 4)), np.zeros((2, 4)),
                np.ones((2, 4, 4), bool), np.zeros((2, 4), bool))
    ok.validate()
    bad = FlyObs(np.zeros((2, 4, 3)), ok.phero, ok.z, ok.legal, ok.decide)
    with pytest.raises(ValueError):
        bad.validate()


def test_rng_same_seed_same_stream_and_independent():
    a = make_rng(1, 2, "brain").random(5)
    assert np.array_equal(a, make_rng(1, 2, "brain").random(5))
    assert not np.array_equal(a, make_rng(1, 3, "brain").random(5))
    assert not np.array_equal(a, make_rng(1, 2, "engine").random(5))
    with pytest.raises(KeyError):
        make_rng(1, 2, "nope")


def test_config_utf8_roundtrip(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("experiment: 시험\nodor: {D: 1.0, lam: 0.05}\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.experiment == "시험" and cfg.odor.lam == 0.05 and cfg["odor"]["D"] == 1.0
    assert len(cfg.hash()) == 16
