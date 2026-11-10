"""실험용 확정 회로 파일(data/circuit/mb_v1.npz)이 뇌가 만드는 회로와 같고, meta가 설정과 맞는지 확인한다."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from flypac.brain.flybrain import build_circuit, file_sha256, load_brain_config
from flypac.core.config import Config, load_config

REPO = Path(__file__).resolve().parents[1]
NPZ, META = REPO / "data" / "circuit" / "mb_v1.npz", REPO / "data" / "circuit" / "mb_v1.meta.json"


def test_file_equals_built_circuit():
    cfg = load_brain_config("configs/brain_v1.yaml")
    c = build_circuit(REPO / cfg.circuit.raw, load_config(REPO / cfg.circuit.assign), int(cfg.circuit.seed))
    with np.load(NPZ, allow_pickle=False) as z:
        assert np.array_equal(z["J"], c.J) and np.array_equal(z["alpha"], c.alpha)
        assert np.array_equal(z["w0"], c.w0) and np.array_equal(z["M"], c.M)
        assert np.array_equal(z["mbon_bundle"], c.mbon_bundle) and np.array_equal(z["pn_channel"], c.pn_channel)
        assert np.array_equal(z["M"], z["W0"] > 0)
        assert z["J_pn"].shape[0] == z["kc_body"].size == c.n_kc and z["W0"].shape[1] == z["mbon_body"].size


def test_meta_matches_config_and_file():
    cfg = load_brain_config("configs/brain_v1.yaml")
    m = json.loads(META.read_text(encoding="utf-8"))
    assert m["status"] == "실험용 확정 (작은 과제 A~D 통과)"
    assert m["assign_seed"] == int(cfg.circuit.seed) and m["kc"] == {"rho": float(cfg.kc.rho), "theta": float(cfg.kc.theta)}
    assert m["learn"] == cfg.to_dict()["learn"] and m["hyperparameter_hash"] == Config(cfg.to_dict()["learn"]).hash()
    assert m["npz_sha256"] == file_sha256(NPZ) and m["raw_sha256"] == file_sha256(REPO / cfg.circuit.raw)
