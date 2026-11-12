"""실행 환경 기록: 결과 폴더의 environment.json. 설정이 auto여도 실제로 쓰인 값을 남긴다.

같은 장치에서는 결과가 바이트 단위로 같지만 cpu와 cuda는 다르다 (DECISIONS 22). 짝지은 비교는 모든 조건이
같은 장치일 때만 성립하므로 팩맨 장치를 기록한다. PPOPacman이 쓰는 resolve_device와 같은 규칙으로 정한다.
"""

from __future__ import annotations

import hashlib
import platform
from pathlib import Path

import numpy as np


def checkpoint_sha256(path) -> str:
    h = hashlib.sha256()
    with open(Path(path), "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _brain_info(cfg) -> dict:
    """초파리 뇌를 쓰는 실행이면 실험 회로 파일과 뇌 설정을 기록한다 (DECISIONS 25.1). 아니면 모두 null."""
    info = {"circuit_file": None, "circuit_sha256": None, "brain_config": None, "brain_config_hash": None}
    ghosts = cfg.get("ghosts", None)
    if ghosts is None or str(ghosts.kind) != "flybrain":
        return info
    from flypac.brain.flybrain import file_sha256, load_brain_config

    brain_path = str(ghosts.config)
    brain = load_brain_config(brain_path)
    info.update(circuit_file=str(brain.circuit.file), circuit_sha256=file_sha256(brain.circuit.file),
                brain_config=brain_path, brain_config_hash=brain.hash())
    return info


def environment_info(cfg) -> dict:
    pacman = cfg.get("pacman", None)  # 최소 설정(시험)에는 pacman 절이 없을 수 있다
    kind = str(pacman.kind) if pacman is not None else None
    info = {
        "python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(),
        "pacman_kind": kind,
        "pacman_checkpoint": None, "pacman_checkpoint_sha256": None,
        "pacman_device_setting": None, "pacman_device": None,
        "torch": None, "torch_cuda": None, "gpu": None,
    }
    info.update(_brain_info(cfg))
    if kind != "ppo":
        return info
    import torch

    from flypac.pacman.trainer import resolve_device

    ckpt = pacman.get("checkpoint", None)  # 최소 설정(시험)에는 체크포인트가 없을 수 있다
    if ckpt is not None:
        info.update(pacman_checkpoint=str(ckpt), pacman_checkpoint_sha256=checkpoint_sha256(ckpt))

    setting = str(pacman.device)
    device = resolve_device(setting)
    info.update(pacman_device_setting=setting, pacman_device=device.type, torch=str(torch.__version__),
                torch_cuda=torch.version.cuda)
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device)
    return info
