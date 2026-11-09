from dataclasses import replace
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
    from viz.clouds import PointClouds, add_light
    from viz.replay import SIZE, TILE, Replay


@pytest.fixture(scope="module")
def anatomy():
    return Anatomy()


def synthetic_arrays(anatomy, **kwargs):
    return make_synthetic_arrays(T=12, brain=True, n_kc=len(anatomy.coordinates["KC"]),
                                 n_mbon=len(anatomy.mbon_bundles), **kwargs)


def renderer(replay, anatomy, fly=1, view="b"):
    return BrainRenderer(replay, anatomy, fly=fly, tick_rate=9.47, decay_s=0.18, view=view)


def test_anatomy_covers_circuit_or_has_reason(anatomy):
    assert {name: len(xy) for name, xy in anatomy.coordinates.items()} == {
        "PN": 106, "KC": 1575, "MBON": 47, "DAN": 317,
    }
    assert all(np.isfinite(anatomy.coordinates[kind]).all() for kind in ("PN", "KC", "MBON"))
    assert len(anatomy.missing) == 2
    assert all(kind == "DAN" and reason for kind, _, reason in anatomy.missing)
    assert {region.name for region in anatomy.regions if region.model_term} == {"LH(R)"}
    assert len(anatomy.regions) == 18 and len(anatomy.dan_regions) == 13


@pytest.mark.parametrize("view", ("a", "b"))
def test_only_confirmed_active_ticks_flash_same_compartments(anatomy, tmp_path, view):
    a = synthetic_arrays(anatomy)
    a["dan"][:, 0] = 0
    a["dan"][1:5, 0] = [0.5, -0.5, 0.5, -0.5]
    a["committed"][1:5, 0] = [True, True, False, True]
    a["action"][1:5, 0] = [FORWARD, RIGHT, FORWARD, RIGHT]
    a["brain_on"][4, 0] = False
    a["moved"][1:4, 0] = [False, True, True]  # 1 is saved; 3 is applied without another commit.
    path = write_replay(tmp_path / "activity.npz", a)
    brain = renderer(Replay(path), anatomy, view=view)

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


@pytest.mark.parametrize("view", ("a", "b"))
def test_pair_passes_same_tick_and_highlights_selected_fly(anatomy, tmp_path, monkeypatch, view):
    path = write_synthetic(tmp_path, T=25, brain=True, n_kc=len(anatomy.coordinates["KC"]),
                           n_mbon=len(anatomy.mbon_bundles))
    replay = Replay(path)
    brain = renderer(replay, anatomy, fly=3, view=view)
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
    assert (brain._layout(True).mini is not None) == (view == "b")


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


def test_cloud_data_and_active_kc_light_both_regions(anatomy, tmp_path):
    clouds = PointClouds(anatomy)
    assert len(clouds.context) == 23008 and clouds.neuron_count == 1728
    assert {name: len(cloud.xy) for name, cloud in clouds.groups.items()} == {
        "pn_calyx": 9856, "kc_calyx": 62425, "kc_lobe": 31500, "mbon_lobe": 9400,
    }
    replay = Replay(write_replay(tmp_path / "kc.npz", synthetic_arrays(anatomy)))
    brain = renderer(replay, anatomy)
    frame = brain.activity_at(1)
    kc = np.zeros_like(frame.kc)
    kc[0] = 1.0
    frame = replace(frame, kc=kc, pn=np.zeros_like(frame.pn), mbon=np.zeros_like(frame.mbon))
    activity = brain._cloud_activity(frame)
    np.testing.assert_array_equal(activity["kc_calyx"], activity["kc_lobe"])
    raster = brain._layout(False).raster
    image = raster.render(activity)
    signal = (image[raster.window].astype(int) - raster.background[raster.window]).reshape(-1, 3)
    for group in ("kc_calyx", "kc_lobe"):
        pixels, owners = raster.groups[group]
        assert (signal[pixels[owners == 0]].max(axis=1) > 0).all()


def test_bloom_and_brightness_cap():
    base = np.full((19, 19, 3), 80, dtype=np.uint8)
    light = np.zeros_like(base, dtype=np.float32)
    light[9, 9] = [30, 45, 120]
    one = add_light(base, light, sigma=2, gain=8, ceiling=195)
    two = add_light(base, light * 2, sigma=2, gain=8, ceiling=195)
    saturated = add_light(base, light * 10000, sigma=2, gain=8, ceiling=195)
    assert two[9, 9, 2] > one[9, 9, 2] > base[9, 9, 2]
    assert one[9, 10, 2] > base[9, 10, 2]
    assert saturated.max() <= 240 and saturated[9, 9, 2] > saturated[9, 9, 0]


def test_probabilities_hold_between_decisions_and_reset_at_boundary(anatomy, tmp_path):
    arrays = synthetic_arrays(anatomy)
    arrays["action"][:, 0] = -1
    arrays["committed"][:, 0] = False
    arrays["pi"][:, 0] = 0
    arrays["action"][[1, 3, 6], 0] = [FORWARD, RIGHT, FORWARD]
    arrays["committed"][[1, 6], 0] = True
    arrays["pi"][[1, 3, 6], 0] = [[.25, .75, 0, 0], [0, .5, .5, 0], [0, 1, 0, 0]]
    arrays["brain_on"][4, 0] = False
    arrays["lives"][8:] = 2
    replay = Replay(write_replay(tmp_path / "pi.npz", arrays))
    brain = renderer(replay, anatomy)

    assert brain.activity_at(0).pi is None
    assert brain.activity_at(1).action == FORWARD
    np.testing.assert_array_equal(brain.activity_at(2).pi, [.25, .75, 0, 0])
    assert brain.activity_at(2).pi_tick == 1 and brain.activity_at(2).action is None
    assert brain.activity_at(3).pi_tick == 3 and brain.activity_at(3).action is None
    assert brain.activity_at(4) is None and brain.activity_at(5).pi is None
    assert brain.activity_at(7).pi_tick == 6 and brain.activity_at(8).pi is None

    frame = brain.render(1)
    layout = brain._layout(False)
    x, y = layout.metrics
    spacing = layout.bar_width + 11
    selected = frame[y + 54:y + 63, x + spacing:x + spacing + layout.bar_width]
    illegal = frame[y + 54:y + 63, x + 2 * spacing:x + 2 * spacing + layout.bar_width]
    assert np.all(selected == p.DECISION, axis=-1).any()
    assert not np.all(illegal == p.MBON_ACTIVITY, axis=-1).any()
