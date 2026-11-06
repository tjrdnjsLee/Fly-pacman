"""원본 회로 파일 검사 (M1 뇌 작업 지시서 ① 4.2 · 4.3, 7절). 원본이 없으면 건너뛴다.

토큰이나 neuprint-python 없이 돈다. docs/ 에 의존하지 않는다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "circuit" / "raw" / "mb_raw_v1.npz"
META = ROOT / "data" / "circuit" / "raw" / "mb_raw_v1.meta.json"

pytestmark = pytest.mark.skipif(not RAW.exists(), reason="원본 회로 파일이 없다")

NEURON_ARRAYS = {
    "pn": ("pn_body", "pn_type", "pn_glom"),
    "kc": ("kc_body", "kc_type", "kc_class"),
    "mbon": ("mbon_body", "mbon_type"),
}


@pytest.fixture(scope="module")
def raw():
    with np.load(RAW, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


@pytest.fixture(scope="module")
def meta():
    return json.loads(META.read_text(encoding="utf-8"))


def test_arrays_present(raw):
    want = {a for group in NEURON_ARRAYS.values() for a in group}
    want |= {"J_pn", "W0", "mbon_class_profile", "mbon_class_profile_candidates", "glom_names"}
    assert want <= set(raw)


def test_shapes(raw):
    n_pn, n_kc, n_mbon = len(raw["pn_body"]), len(raw["kc_body"]), len(raw["mbon_body"])
    for kind, names in NEURON_ARRAYS.items():
        n = len(raw[names[0]])
        for name in names:
            assert raw[name].shape == (n,), name
    assert raw["J_pn"].shape == (n_kc, n_pn)
    assert raw["W0"].shape == (n_kc, n_mbon)
    assert raw["mbon_class_profile"].shape == (n_mbon, 3)
    assert raw["mbon_class_profile_candidates"].shape == (n_mbon, 3)
    assert raw["J_pn"].dtype == np.int32 and raw["W0"].dtype == np.int32
    assert raw["mbon_class_profile"].dtype == np.float64


def test_sorted_by_body_id(raw):
    for names in NEURON_ARRAYS.values():
        b = raw[names[0]]
        assert np.all(np.diff(b) > 0), f"{names[0]} 가 bodyId 오름차순이 아니다"


def test_no_negative_and_s_min(raw, meta):
    s_min = meta["thresholds"]["s_min"]
    for name in ("J_pn", "W0"):
        m = raw[name]
        assert (m >= 0).all()
        assert ((m == 0) | (m >= s_min)).all(), f"{name} 에 s_min 미만 연결이 있다"


def test_glomeruli_and_classes(raw):
    assert list(raw["glom_names"]) == sorted(set(raw["pn_glom"]))
    assert set(raw["kc_class"]) <= {"gamma", "apbp", "ab"}
    for key in ("mbon_class_profile", "mbon_class_profile_candidates"):
        prof = raw[key]
        assert np.allclose(prof.sum(axis=1), 1.0) and (prof >= 0).all()


def test_profile_is_selected_kc_basis(raw):
    """mbon_class_profile = 선택 KC(W0) 기준의 KC 종류별 입력 비율."""
    W0, cls = raw["W0"], raw["kc_class"]
    want = np.stack([W0[cls == c].sum(axis=0) for c in ("gamma", "apbp", "ab")], axis=1).astype(float)
    assert np.allclose(raw["mbon_class_profile"], want / want.sum(axis=1, keepdims=True))


def test_meta_fields(meta):
    for key in ("dataset", "server", "extracted_at", "neuprint_python_version", "rois",
                "thresholds", "exclude", "counts", "sort_rule", "snapshot_sha256",
                "raw_sha256", "attribution"):
        assert key in meta, key
    assert meta["dataset"] == "hemibrain:v1.2.1"
    for key in ("s_min", "theta_pn", "theta_kc", "kc_min_selected_pn_input", "theta_mbon"):
        assert key in meta["thresholds"], key
    assert len(meta["rois"]["compartments"]) == 15


def test_meta_counts_match(raw, meta):
    sel = meta["counts"]["selected"]
    assert sel["pn"] == len(raw["pn_body"])
    assert sel["kc"] == len(raw["kc_body"])
    assert sel["mbon"] == len(raw["mbon_body"])
    assert sel["glomeruli"] == len(raw["glom_names"])
