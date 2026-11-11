"""뇌가 실험 회로 파일(mb_v1.npz)을 직접 읽는다 (DECISIONS 25.1). 저장소의 data 파일은 고치지 않는다."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from helpers import make_cfg

from flypac.brain.flybrain import FlyBrain, build_circuit, file_sha256, load_brain_config, load_experiment_circuit
from flypac.core.config import Config, load_config
from flypac.core.runner import build_runner
from flypac.recording.recorder import Recorder

ROOT = Path(__file__).resolve().parents[1]
CFG = load_brain_config("configs/brain_v1.yaml")
NPZ = ROOT / "data" / "circuit" / "mb_v1.npz"


def test_default_brain_circuit_equals_rebuilt_circuit():
    b = FlyBrain(CFG, 1, [0])
    ref = build_circuit(CFG.circuit.raw, load_config(ROOT / CFG.circuit.assign), int(CFG.circuit.seed))
    for name in ("J", "alpha", "w0", "M", "mbon_bundle", "pn_channel"):
        got = getattr(b.circuit, name)
        assert np.array_equal(got, getattr(ref, name)), name
        assert not got.flags.writeable, name
    assert b.circuit.source_hash == file_sha256(NPZ) and b.circuit.seed == 0
    assert b.circuit.source_hash != ref.source_hash  # 원본 파일 해시가 아니다


def _tmp_circuit(tmp_path, meta_edit=None, cfg_edit=None):
    shutil.copy(NPZ, tmp_path / "mb_v1.npz")
    meta = json.loads((ROOT / "data" / "circuit" / "mb_v1.meta.json").read_text(encoding="utf-8"))
    if meta_edit:
        meta_edit(meta)
    (tmp_path / "mb_v1.meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    d = CFG.to_dict()
    d["circuit"]["file"] = str(tmp_path / "mb_v1.npz")
    if cfg_edit:
        cfg_edit(d)
    return Config(d)


def test_unmodified_copy_loads(tmp_path):
    c = load_experiment_circuit(_tmp_circuit(tmp_path))
    assert c.source_hash == file_sha256(NPZ)


META_EDITS = {
    "npz_sha256": lambda m: m.update(npz_sha256="0" * 64),
    "raw_sha256": lambda m: m.update(raw_sha256="0" * 64),
    "assign_config_hash": lambda m: m.update(assign_config_hash="0" * 16),
    "assign_seed": lambda m: m.update(assign_seed=1),
    "kc.rho": lambda m: m["kc"].update(rho=0.10),
    "kc.theta": lambda m: m["kc"].update(theta=0.5),
    "hyperparameter_hash": lambda m: m.update(hyperparameter_hash="0" * 16),
}


@pytest.mark.parametrize("item", list(META_EDITS))
def test_meta_mismatch_raises(tmp_path, item):
    cfg = _tmp_circuit(tmp_path, meta_edit=META_EDITS[item])
    with pytest.raises(ValueError, match=item.replace(".", r"\.")):
        load_experiment_circuit(cfg)
    with pytest.raises(ValueError, match=item.replace(".", r"\.")):
        FlyBrain(cfg, 1, [0])


CFG_EDITS = {
    "kc.rho": lambda d: d["kc"].update(rho=0.10),
    "kc.theta": lambda d: d["kc"].update(theta=0.5),
    "assign_seed": lambda d: d["circuit"].update(seed=1),
    "hyperparameter_hash": lambda d: d["learn"].update(eta=2.0),
}


@pytest.mark.parametrize("item", list(CFG_EDITS))
def test_config_side_change_raises(tmp_path, item):
    cfg = _tmp_circuit(tmp_path, cfg_edit=CFG_EDITS[item])
    with pytest.raises(ValueError, match=item.replace(".", r"\.")):
        FlyBrain(cfg, 1, [0])


def test_state_format_3_and_format_2_rejected():
    b = FlyBrain(CFG, 1, [0])
    st = b.state_dict()
    assert st["format"] == 3 and st["circuit_hash"] == file_sha256(NPZ)
    FlyBrain(CFG, 1, [0]).load_state_dict(st)
    with pytest.raises(ValueError):
        FlyBrain(CFG, 1, [0]).load_state_dict({**st, "format": 2})


def test_environment_json_records_circuit(tmp_path):
    cfg = make_cfg(ghosts={"kind": "flybrain", "config": "configs/brain_v1.yaml"}, n_games=1, seeds=[0])
    rec = Recorder(tmp_path / "run", cfg)
    build_runner(cfg, rec).run()
    env = json.loads((tmp_path / "run" / "environment.json").read_text(encoding="utf-8"))
    assert env["circuit_file"] == "data/circuit/mb_v1.npz" and env["circuit_sha256"] == file_sha256(NPZ)
    assert env["brain_config"] == "configs/brain_v1.yaml" and env["brain_config_hash"] == CFG.hash()


def test_environment_json_circuit_null_for_other_ghosts(tmp_path):
    cfg = make_cfg(n_games=1, seeds=[0])
    build_runner(cfg, Recorder(tmp_path / "run", cfg)).run()
    env = json.loads((tmp_path / "run" / "environment.json").read_text(encoding="utf-8"))
    for k in ("circuit_file", "circuit_sha256", "brain_config", "brain_config_hash"):
        assert k in env and env[k] is None
