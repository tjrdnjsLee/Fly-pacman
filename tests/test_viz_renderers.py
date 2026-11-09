import json
from pathlib import Path

import numpy as np
import pytest

from flypac.core import directions as d
from flypac.maps.original import load_original
from flypac.replay import schema as s
from flypac.replay.synth import make_synthetic_arrays, write_synthetic
from flypac.replay.writer import write_replay

pygame = pytest.importorskip("pygame")
pytest.importorskip("imageio_ffmpeg")

# viz is a repository tool outside the installed flypac package.
with pytest.MonkeyPatch.context() as patch:
    patch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from viz.game import GameRenderer
    from viz.palette import PELLET
    from viz.raw import RawRenderer
    from viz.replay import FPS, SIZE, TILE, Replay
    from viz.video import frame_ticks


@pytest.mark.parametrize("version", (2, 3))
def test_synthetic_replay_aligns_both_offscreen_renderers(tmp_path, version):
    if version == 3:
        path = write_synthetic(tmp_path, T=25, phero=True)
    else:
        arrays = make_synthetic_arrays(T=25)
        meta = s.parse_meta(arrays[s.META_KEY])
        meta["format_version"] = 2
        arrays[s.META_KEY] = np.array(json.dumps(meta))
        del arrays["committed"], arrays["moved"]
        path = tmp_path / "v2.npz"
        np.savez_compressed(path, **arrays)

    replay = Replay(path)
    game = GameRenderer(replay)
    raw = RawRenderer(Replay(path), c0=0.01)
    for tick in (3.0, 16.5):
        assert len(game.actors_at(tick)) == 5
        assert game.actors_at(tick) == raw.actors_at(tick)
        game_frame, raw_frame = game.render(tick), raw.render(tick)
        for frame in (game_frame, raw_frame):
            assert frame.shape == (SIZE[1], SIZE[0], 3)
            assert frame.dtype == np.uint8 and frame.flags.c_contiguous
        assert not np.array_equal(game_frame, raw_frame)
    assert pygame.display.get_surface() is None


def test_interpolation_stays_continuous_through_tunnel_and_stops_at_reset(tmp_path):
    amap = load_original()
    left = next(cell for cell in amap.tunnel_cells if amap.cell_rc[cell, 1] == 0)
    right = amap.nbr[left, d.WEST]
    arrays = make_synthetic_arrays(T=5)
    arrays["pac_cell"][:] = [amap.nbr[left, d.EAST], left, right, amap.nbr[right, d.WEST], amap.pacman_start]
    arrays["pac_dir"][:] = d.WEST
    arrays["pac_acc"][:] = 0.25
    arrays["ev_tick"] = np.array([3], dtype=np.int32)
    arrays["ev_kind"] = np.array([s.KIND_CODES["LIFE_LOST"]], dtype=np.int8)
    arrays["ev_fly"] = np.array([-1], dtype=np.int8)
    replay = Replay(write_replay(tmp_path / "tunnel.npz", arrays))

    width = replay.board_size[0]
    samples = frame_ticks(3, 9.47, 0, None)
    points = np.array([replay.positions(tick)[0] for tick in samples])
    steps = np.abs(np.diff(points, axis=0))
    assert (steps[:, 0] > width - TILE).any()
    steps[:, 0] = np.minimum(steps[:, 0], width - steps[:, 0])
    assert np.all(np.linalg.norm(steps, axis=1) < TILE)
    assert replay.positions(1.0)[0][0] == replay.cell_center(left)[0] - TILE // 4
    assert replay.positions(3.9) == replay.positions(3.0)
    assert replay.positions(4.0)[0] != replay.positions(3.9)[0]


def test_game_reconstructs_pellets_when_seeking(tmp_path):
    arrays = make_synthetic_arrays(T=8)
    amap = load_original()
    cells = [np.flatnonzero(amap.dots)[0], np.flatnonzero(amap.energizers)[0]]
    for tick, cell in zip((2, 4), cells):
        arrays["pac_cell"][tick] = cell
    arrays[s.META_KEY] = np.array(s.make_meta("b1", 0, 1, "engine", "config", "original"))
    arrays["ev_tick"] = np.array([2, 4], dtype=np.int32)
    arrays["ev_kind"] = np.array([s.KIND_CODES[kind] for kind in ("DOT", "ENERGIZER")], dtype=np.int8)
    arrays["ev_fly"] = np.array([-1, -1], dtype=np.int8)
    replay = Replay(write_replay(tmp_path / "pellets.npz", arrays))

    assert [replay.pellet_eaten_at[cell] for cell in cells] == [2, 4]
    game = GameRenderer(replay)
    x, y = replay.cell_center(cells[0])
    assert tuple(game.render(5)[y, x]) != PELLET
    assert tuple(game.render(1)[y, x]) == PELLET


def test_frame_count_matches_duration_and_clip_bounds():
    assert len(frame_ticks(200, 9.47, 8, 3)) == 3 * FPS
    assert len(frame_ticks(200, 9.47, 0.1, 0.2)) == 6
    ticks = frame_ticks(20, 10.0, 1.9, 3)
    assert len(ticks) == 3 and ticks[0] == pytest.approx(19) and ticks[-1] < 20
