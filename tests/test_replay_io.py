import numpy as np
import pytest

from flypac.core.events import Kind
from flypac.replay import schema as s
from flypac.replay.reader import kind_names, read_replay
from flypac.replay.synth import make_synthetic_arrays, write_synthetic
from flypac.replay.writer import ReplayBuffer, replay_filename, write_replay


@pytest.mark.parametrize("brain,phero", [(False, False), (True, False), (False, True), (True, True)])
def test_synthetic_log_passes_format_check(brain, phero):
    a = make_synthetic_arrays(T=60, brain=brain, phero=phero, k=7, n_mbon=5)
    meta = s.validate(a)
    assert meta["format_version"] == s.FORMAT_VERSION and meta["condition"] == "synthetic"
    assert ("kc_active" in a) == brain and ("phero_field" in a) == phero
    assert a["odor_field"].shape == (60, 320) and a["sense"].shape == (60, 4, 8)
    if brain:
        assert a["kc_active"].shape == (60, 4, 7) and a["mbon_out"].shape == (60, 4, 5)
        assert a["pn_input"].shape == (60, 4, 9) and a["dan"].astype(bool).any()


def test_synthetic_is_deterministic_and_plausible():
    a, b = make_synthetic_arrays(T=50, seed=3), make_synthetic_arrays(T=50, seed=3)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    c = make_synthetic_arrays(T=50, seed=4)
    assert not np.array_equal(a["fly_cell"], c["fly_cell"])
    # 칸 번호는 모두 맵 안, 마지막 틱에 포획 사건
    assert a["fly_cell"].min() >= 0 and a["fly_cell"].max() < 320
    kinds = kind_names(a["ev_kind"])
    assert "CATCH" in kinds and a["ev_tick"].max() == 49
    # 결정 아닌 틱의 행동은 -1, 결정 틱의 π는 합 1
    dec = a["action"] >= 0
    assert np.allclose(a["pi"][dec].astype(float).sum(-1), 1.0, atol=1e-2)
    assert np.allclose(a["pi"][~dec], 0.0)


def test_synthetic_flies_move_at_most_one_cell_per_tick():
    from flypac.maps.original import load_original

    m = load_original()
    a = make_synthetic_arrays(T=120, seed=1)
    cells = a["fly_cell"].astype(int)
    in_house = a["fly_in_house"]
    for t in range(1, 119):  # 마지막 틱의 포획 이동은 제외하고 이웃표로 확인
        for j in range(4):
            if cells[t, j] != cells[t - 1, j] and not in_house[t - 1, j]:  # 집에서 나오는 순간은 제외
                assert cells[t, j] in m.nbr[cells[t - 1, j]]


def test_write_read_roundtrip_exact(tmp_path):
    a = make_synthetic_arrays(T=40, brain=True, phero=True, seed=2)
    path = write_replay(tmp_path / "x" / replay_filename("synthetic", 2, 1), a)
    assert path.name == "synthetic_s2_g1.npz"
    meta, got = read_replay(path)
    assert meta["seed"] == 2 and meta["game_no"] == 1
    assert set(got) == set(a) - {s.META_KEY}
    for k, v in got.items():
        assert v.dtype == a[k].dtype and np.array_equal(v, a[k]), k


def test_write_synthetic_helper_and_no_pickle(tmp_path):
    path = write_synthetic(tmp_path, T=20, seed=5, game_no=7)
    assert path.name == "synthetic_s5_g7.npz"
    with np.load(path, allow_pickle=False) as z:  # pickle 없이 열린다
        assert "meta" in z.files


def test_writer_rejects_bad_arrays_and_reader_rejects_bad_file(tmp_path):
    a = make_synthetic_arrays(T=10)
    bad = dict(a)
    bad["fly_cell"] = bad["fly_cell"].astype("int32")
    with pytest.raises(s.SchemaError):
        write_replay(tmp_path / "bad.npz", bad)
    np.savez(tmp_path / "junk.npz", foo=np.zeros(3))
    with pytest.raises(s.SchemaError):
        read_replay(tmp_path / "junk.npz")


def test_buffer_checks_fields_and_event_codes():
    buf = ReplayBuffer()
    with pytest.raises(ValueError):
        buf.add_frame(pac_cell=1)
    buf.add_event(3, "CATCH", 2)
    buf.add_event(3, Kind.LIFE_LOST, -1)  # StrEnum도 받는다
    assert buf._events == [(3, 0, 2), (3, 5, -1)]
    with pytest.raises(ValueError):
        buf.arrays("{}")  # 프레임 없음
