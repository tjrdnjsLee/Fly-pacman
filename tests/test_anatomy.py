"""해부 자료 검사 (TRD 19.13, M1 뇌 작업 지시서 ① 5절 · 7절). 파일이 없으면 건너뛴다.

토큰이나 neuprint-python 없이 돈다. docs/ 에 의존하지 않는다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ANAT = ROOT / "data" / "anatomy" / "mb_anatomy_v1.npz"
ANAT_META = ROOT / "data" / "anatomy" / "mb_anatomy_v1.meta.json"
RAW = ROOT / "data" / "circuit" / "raw" / "mb_raw_v1.npz"

pytestmark = pytest.mark.skipif(not ANAT.exists(), reason="해부 자료 파일이 없다")


@pytest.fixture(scope="module")
def anat():
    with np.load(ANAT, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def lm(anat, name):
    names = list(anat["landmark_names"])
    return anat["landmark_xy"][names.index(name)]


def test_every_neuron_has_coord_or_reason(anat):
    has, reason = anat["has_coord"], anat["missing_reason"]
    assert np.isfinite(anat["xy_um"][has]).all()
    assert np.isnan(anat["xy_um"][~has]).all()
    assert all(r != "" for r in reason[~has]), "좌표가 없는데 이유가 비어 있다"
    assert all(r == "" for r in reason[has])


@pytest.mark.skipif(not RAW.exists(), reason="원본 회로 파일이 없다")
def test_circuit_neurons_covered_in_order(anat):
    with np.load(RAW, allow_pickle=False) as z:
        raw = {k: z[k] for k in ("pn_body", "kc_body", "mbon_body")}
    for kind, key in (("PN", "pn_body"), ("KC", "kc_body"), ("MBON", "mbon_body")):
        m = anat["kind"] == kind
        assert np.array_equal(anat["body"][m], raw[key]), f"{kind} 순서가 원본 회로와 다르다"


def test_frontal_orientation(anat):
    root, alpha, beta = lm(anat, "root"), lm(anat, "alpha_tip"), lm(anat, "beta_tip")
    calyx, mid = lm(anat, "calyx"), lm(anat, "midline")
    assert alpha[1] > root[1], "(가) α엽 끝이 꽃자루 뿌리보다 위가 아니다"
    assert abs(beta[0] - mid[0]) < abs(root[0] - mid[0]), "(나) β엽 끝이 뿌리보다 안쪽이 아니다"
    assert calyx[1] > root[1], "(다) 칼릭스가 뿌리보다 위가 아니다"


def test_regions(anat):
    names = list(anat["region_names"])
    assert len(names) == 18  # 칼릭스, 꽃자루, 엽 구획 15개, 외측뿔
    model = dict(zip(names, anat["region_model_term"]))
    assert model["LH(R)"] and sum(model.values()) == 1, "외측뿔에만 model_term 표시"
    off = anat["ring_offsets"]
    assert off[0] == 0 and off[-1] == len(anat["ring_xy"]) and np.all(np.diff(off) >= 3)
    assert len(off) == len(anat["ring_region"]) + 1
    assert set(anat["ring_region"]) == set(range(len(names))), "다각형이 없는 영역이 있다"


def test_dan_compartments(anat):
    m = anat["kind"] == "DAN"
    comps = set(anat["dan_compartment"][m & anat["has_coord"]])
    assert "" not in comps and comps <= set(anat["region_names"])


def test_meta():
    meta = json.loads(ANAT_META.read_text(encoding="utf-8"))
    for key in ("dataset", "view", "axis_transform", "units", "scale_um_per_voxel",
                "regions", "attribution", "orientation_checks"):
        assert key in meta, key
    assert meta["view"] == "frontal"
    assert all(meta["orientation_checks"].values())
