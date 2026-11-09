from pathlib import Path

import numpy as np
import pytest

from flypac.core.directions import FORWARD, RIGHT
from flypac.replay import schema
from flypac.replay.synth import make_synthetic_arrays, write_synthetic
from flypac.replay.writer import write_replay

pygame = pytest.importorskip("pygame")
pytest.importorskip("imageio_ffmpeg")

with pytest.MonkeyPatch.context() as patch:
    patch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from viz import palette as p
    from viz.anatomy import Anatomy
    from viz.brain import BrainPairRenderer, BrainRenderer
    from viz.replay import SIZE, TILE, Replay


@pytest.fixture(scope="module")
def anatomy():
    return Anatomy()


def synthetic_arrays(anatomy, **kwargs):
    return make_synthetic_arrays(T=12, brain=True, n_kc=len(anatomy.coordinates["KC"]),
                                 n_mbon=len(anatomy.mbon_bundles), **kwargs)


def renderer(replay, anatomy, fly=1):
    return BrainRenderer(replay, anatomy, fly=fly, tick_rate=9.47, decay_s=0.18)


def test_anatomy_covers_circuit_or_has_reason(anatomy):
    assert {name: len(xy) for name, xy in anatomy.coordinates.items()} == {
        "PN": 106, "KC": 1575, "MBON": 47, "DAN": 317,
    }
    assert all(np.isfinite(anatomy.coordinates[kind]).all() for kind in ("PN", "KC", "MBON"))
    assert len(anatomy.missing) == 2
    assert all(kind == "DAN" and reason for kind, _, reason in anatomy.missing)
    assert {region.name for region in anatomy.regions if region.model_term} == {"LH(R)"}
    assert len(anatomy.regions) == 18 and len(anatomy.dan_regions) == 13


def test_only_confirmed_active_ticks_flash_same_compartments(anatomy, tmp_path):
    a = synthetic_arrays(anatomy)
    a["dan"][:, 0] = 0
    a["dan"][1:5, 0] = [0.5, -0.5, 0.5, -0.5]
    a["committed"][1:5, 0] = [True, True, False, True]
    a["action"][1:5, 0] = [FORWARD, RIGHT, FORWARD, RIGHT]
    a["brain_on"][4, 0] = False
    a["moved"][1:4, 0] = [False, True, True]  # 1 is saved; 3 is applied without another commit.
    path = write_replay(tmp_path / "activity.npz", a)
    brain = renderer(Replay(path), anatomy)

    for tick, sign, action in ((1, 1, FORWARD), (2, -1, RIGHT), (3, 0, None), (4, 0, None)):
        state = brain.activity_at(tick)
        if tick == 4:
            assert state is None and brain.status_at(tick)
        else:
            assert np.sign(state.dan) == sign and state.action == action
            expected = [state.mbon[anatomy.mbon_bundles == bundle].sum() for bundle in range(4)]
            np.testing.assert_allclose(state.bundles, expected)

        frame = brain.render(tick + 0.25)
        assert frame.shape == (SIZE[1], SIZE[0], 3) and frame.flags.c_contiguous
        plot = brain._layout(False).plot
        pixels = frame[plot.top:plot.bottom, plot.left:plot.right]
        for direction, color in ((1, p.DAN_POSITIVE), (-1, p.DAN_NEGATIVE)):
            assert np.all(pixels == color, axis=-1).any() == (sign == direction)

    assert len(brain._layout(False).dan) == len(anatomy.dan_regions)
    assert pygame.display.get_surface() is None


def test_pair_passes_same_tick_and_highlights_selected_fly(anatomy, tmp_path, monkeypatch):
    path = write_synthetic(tmp_path, T=25, brain=True, n_kc=len(anatomy.coordinates["KC"]),
                           n_mbon=len(anatomy.mbon_bundles))
    replay = Replay(path)
    brain = renderer(replay, anatomy, fly=3)
    pair = BrainPairRenderer(brain)
    ticks = []

    def spy(owner):
        original = owner.surface_at

        def wrapped(tick, **kwargs):
            ticks.append((tick, owner.replay.tick_index(tick)))
            return original(tick, **kwargs)

        monkeypatch.setattr(owner, "surface_at", wrapped)

    spy(pair.game)
    spy(brain)
    frame = pair.render(17.5)
    assert ticks == [(17.5, 17), (17.5, 17)]
    x, y = replay.positions(17.5)[3]
    x += pair._left - pair._crop.left
    radius = TILE // 2
    patch = frame[y - radius:y + radius + 1, x - radius:x + radius + 1]
    assert np.all(patch == p.SELECTED_FLY, axis=-1).any()
    assert frame.shape == (SIZE[1], SIZE[0], 3)


def test_b1_status_and_b0_learning_label(anatomy, tmp_path, monkeypatch):
    b1 = renderer(Replay(write_synthetic(tmp_path, T=5, condition="b1")), anatomy)
    assert b1.status_at(0) == ("뇌 없음(원작 유령)", "No brain (original ghost)")
    assert b1.activity_at(0) is None
    b0_path = write_replay(tmp_path / "b0.npz", synthetic_arrays(anatomy, condition="b0"))
    b0 = renderer(Replay(b0_path), anatomy)
    labels = []

    def capture(surface, korean, english, xy):
        labels.append(english)

    monkeypatch.setattr(b0, "_caption", capture)
    b0.render(0)
    assert "Learning off" in labels


def test_mismatched_replay_mbon_width_is_rejected(anatomy, tmp_path):
    path = write_synthetic(tmp_path, T=5, brain=True, n_mbon=4)
    replay = Replay(path)
    assert replay.meta["format_version"] == schema.FORMAT_VERSION
    with pytest.raises(ValueError, match="MBON width"):
        renderer(replay, anatomy)
