"""해부 점구름 · 뇌 형태 자료 검사 (뇌 렌더러 v2). 파일이 없으면 건너뛴다.

토큰이나 neuprint-python 없이 돈다. docs/ 에 의존하지 않는다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CLOUDS = ROOT / "data" / "anatomy" / "mb_synapse_clouds_v1.npz"
CLOUDS_META = CLOUDS.with_name("mb_synapse_clouds_v1.meta.json")
CONTEXT = ROOT / "data" / "anatomy" / "brain_context_v1.npz"
CONTEXT_META = CONTEXT.with_name("brain_context_v1.meta.json")
ANAT = ROOT / "data" / "anatomy" / "mb_anatomy_v1.npz"
RAW = ROOT / "data" / "circuit" / "raw" / "mb_raw_v1.npz"

pytestmark = pytest.mark.skipif(
    not (CLOUDS.exists() and CONTEXT.exists() and ANAT.exists() and RAW.exists()), reason="해부 자료 파일이 없다")

KINDS = {"pn_calyx": "PN", "kc_calyx": "KC", "kc_lobe": "KC", "mbon_lobe": "MBON"}
ORDER_KEY = {"PN": "pn_body", "KC": "kc_body", "MBON": "mbon_body"}


def load(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


@pytest.fixture(scope="module")
def cl():
    return load(CLOUDS)


@pytest.fixture(scope="module")
def meta():
    return json.loads(CLOUDS_META.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def raw():
    return load(RAW)


@pytest.fixture(scope="module")
def anat():
    return load(ANAT)


def cloud_of(cl, kind, i):
    off = cl[f"{kind}_offsets"]
    return cl[f"{kind}_xy_um"][off[i]:off[i + 1]]


def test_neuron_order_matches_circuit(cl, raw):
    for kind, k in KINDS.items():
        assert np.array_equal(cl[f"{kind}_body"], raw[ORDER_KEY[k]]), kind


def test_layout_is_consistent(cl, raw):
    for kind, k in KINDS.items():
        off, xy, n0 = cl[f"{kind}_offsets"], cl[f"{kind}_xy_um"], cl[f"{kind}_n_orig"]
        n = len(raw[ORDER_KEY[k]])
        assert off.shape == (n + 1,) and n0.shape == (n,)
        assert off[0] == 0 and off[-1] == len(xy) and (np.diff(off) >= 0).all()
        assert xy.ndim == 2 and xy.shape[1] == 2 and np.isfinite(xy).all()


def test_every_neuron_has_points_or_reason(cl):
    for kind in KINDS:
        cnt, reason = np.diff(cl[f"{kind}_offsets"]), cl[f"{kind}_missing_reason"]
        assert ((cnt > 0) | (reason != "")).all(), kind
        assert (reason[cnt > 0] == "").all(), kind


def test_caps_and_counts(cl, meta):
    for kind in KINDS:
        cap = meta["caps_per_neuron"][kind]
        cnt, n0 = np.diff(cl[f"{kind}_offsets"]), cl[f"{kind}_n_orig"]
        assert (cnt <= cap).all(), kind
        assert np.array_equal(cnt, np.minimum(n0, cap)), kind   # 상한 이하이면 전부, 넘으면 정확히 상한
        assert meta["stats"][kind]["points"] == len(cl[f"{kind}_xy_um"])


def test_same_coordinate_system_as_anatomy(cl, anat):
    """기존 대표 좌표(시냅스 위치 중심)가 그 뉴런 점구름의 범위 안에 있다 (상한으로 줄인 경우만 약간 여유)."""
    margin = 1.0  # μm: 뽑아서 줄인 점구름의 범위가 전체보다 좁아질 수 있는 정도
    pairs = {"pn_calyx": "PN", "kc_calyx": "KC", "mbon_lobe": "MBON"}   # KC 엽 점은 대표 좌표가 없다
    for kind, k in pairs.items():
        m = anat["kind"] == k
        assert np.array_equal(anat["body"][m], cl[f"{kind}_body"]), kind
        rep = anat["xy_um"][m]
        capped = cl[f"{kind}_n_orig"] > np.diff(cl[f"{kind}_offsets"])
        for i in range(len(rep)):
            p = cloud_of(cl, kind, i)
            tol = margin if capped[i] else 1e-3
            lo, hi = p.min(axis=0) - tol, p.max(axis=0) + tol
            assert ((rep[i] >= lo) & (rep[i] <= hi)).all(), (kind, i, rep[i], lo, hi)


def test_context_is_same_frame_and_documented():
    ctx = load(CONTEXT)
    m = json.loads(CONTEXT_META.read_text(encoding="utf-8"))
    xy = ctx["xy_um"]
    assert xy.ndim == 2 and xy.shape[1] == 2 and np.isfinite(xy).all()
    assert set(ctx) == {"xy_um"}                      # bodyId와 종류는 저장하지 않는다
    assert m["n_with_soma"] == len(xy) and m["n_with_soma"] + m["n_without_soma"] == m["n_neurons"]
    assert "시뮬레이션하지 않음" in m["note"] and "절반" in m["hemibrain_is_half_brain"]
    assert m["dataset"] == "hemibrain:v1.2.1" and "projection" in m
    # 같은 좌표계: 버섯체 대표 좌표 전부가 뇌 형태 점들의 범위 안에 든다
    a = load(ANAT)
    rep = a["xy_um"][a["has_coord"]]
    assert (rep >= xy.min(axis=0)).all() and (rep <= xy.max(axis=0)).all()
    # 투영 설정이 기존 해부 자료와 같다
    am = json.loads((ANAT.with_suffix("").parent / "mb_anatomy_v1.meta.json").read_text(encoding="utf-8"))
    assert m["projection"]["u (가로)"] == am["axis_transform"]["u (가로)"]
    assert m["projection"]["v (세로, 위 = 등쪽)"] == am["axis_transform"]["v (세로, 위 = 등쪽)"]


def test_meta_records_reproducibility(meta):
    assert meta["dataset"] == "hemibrain:v1.2.1"
    assert isinstance(meta["sampling_seed"], int)
    assert meta["sources"] and all(len(h) == 64 for h in meta["sources"].values())
    assert set(meta["caps_per_neuron"]) == set(KINDS)
