"""실험용 확정 회로 파일 만들기 (TRD 10.1 표, DECISIONS 19). 작은 과제 A~D 통과 뒤에 한 번 만든다.

  uv run python scripts/make_experiment_circuit.py     # data/circuit/mb_v1.npz + mb_v1.meta.json

원본(mb_raw_v1.npz) + 배정 설정(configs/circuit_assign_v1.yaml) + 배정 시드(configs/brain_v1.yaml의 circuit.seed)로
`build_circuit`이 만드는 회로를 그대로 저장한다 (다시 계산하지 않고 같은 함수를 부른다). 뉴런 식별(bodyId · 종류)은
회로 순서 그대로다. 파일은 읽기 전용 참고용이며, 뇌는 지금도 raw + 배정으로 같은 회로를 만든다 (tests/test_circuit_v1.py가 일치를 확인).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from flypac.brain import assign as A
from flypac.brain.flybrain import build_circuit, file_sha256, load_brain_config
from flypac.core.config import Config, load_config

REPO = Path(__file__).resolve().parents[1]
BRAIN = "configs/brain_v1.yaml"
OUT_NPZ = REPO / "data" / "circuit" / "mb_v1.npz"
OUT_META = REPO / "data" / "circuit" / "mb_v1.meta.json"
STATUS = "실험용 확정 (작은 과제 A~D 통과)"
GENERATOR = "make_experiment_circuit_v1"


def build() -> tuple[dict, dict]:
    cfg = load_brain_config(BRAIN)
    acfg = load_config(REPO / cfg.circuit.assign)
    raw_path = REPO / cfg.circuit.raw
    seed = int(cfg.circuit.seed)
    raw = A.load_raw(raw_path)
    c = build_circuit(raw_path, acfg, seed)
    arrays = {
        "J": np.asarray(c.J), "alpha": np.asarray(c.alpha), "w0": np.asarray(c.w0), "M": np.asarray(c.M),
        "J_pn": raw.J_pn, "W0": raw.W0,
        "mbon_bundle": np.asarray(c.mbon_bundle), "pn_channel": np.asarray(c.pn_channel),
        "pn_body": raw.pn_body, "pn_type": raw.pn_type, "pn_glom": raw.pn_glom,
        "kc_body": raw.kc_body, "kc_type": raw.kc_type, "kc_class": raw.kc_class,
        "mbon_body": raw.mbon_body, "mbon_type": raw.mbon_type,
    }
    size = np.bincount(c.mbon_bundle, minlength=A.N_ACTIONS)
    learn_hash = Config(cfg.to_dict()["learn"]).hash()
    meta = {
        "format": "mb_v1", "status": STATUS, "generator": GENERATOR,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": "hemibrain:v1.2.1", "raw_file": str(cfg.circuit.raw), "raw_sha256": file_sha256(raw_path),
        "assign_file": str(cfg.circuit.assign), "assign_config_hash": acfg.hash(),
        "assign_seed": seed, "branch": str(acfg["branch"]),
        "brain_config": BRAIN, "kc": {"rho": float(cfg.kc.rho), "theta": float(cfg.kc.theta)},
        "learn": cfg.to_dict()["learn"], "hyperparameter_hash": learn_hash,
        "hyperparameter_rule": "DECISIONS 18.7",
        "counts": {"n_pn": int(raw.pn_body.size), "n_kc": int(c.n_kc), "n_mbon": int(c.n_mbon),
                   "bundle_sizes": size.tolist()},
        "connection_density": float(c.M.mean()),
        "arrays": {k: [list(v.shape), str(v.dtype)] for k, v in arrays.items()},
        "channel_order": "냄새 좌 · 앞 · 우 · 뒤, 상태 z, 페로몬 좌 · 앞 · 우 · 뒤 (J의 열)",
        "bundle_order": "좌 · 직진 · 우 · 반전 (mbon_bundle의 값)",
        "identity_order": "pn_* · kc_* · mbon_*는 J_pn · W0 · J · w0의 행 · 열 순서 그대로 (bodyId · 종류)",
        "passed": "작은 과제 A~C (DECISIONS 19), D (지연 n = 0, 5 합격; n = 10, 30, 50은 저하 곡선 기록)",
    }
    return arrays, meta


def main() -> int:
    arrays, meta = build()
    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_NPZ, **arrays)
    meta["npz_sha256"] = file_sha256(OUT_NPZ)
    OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(OUT_NPZ.name, meta["npz_sha256"][:12], meta["counts"], "밀도", round(meta["connection_density"], 4))
    return 0


if __name__ == "__main__":
    sys.exit(main())
