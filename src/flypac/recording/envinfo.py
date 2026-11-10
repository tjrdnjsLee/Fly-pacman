"""실행 환경 기록: 결과 폴더의 environment.json. 설정이 auto여도 실제로 쓰인 값을 남긴다.

같은 장치에서는 결과가 바이트 단위로 같지만 cpu와 cuda는 다르다 (DECISIONS 22). 짝지은 비교는 모든 조건이
같은 장치일 때만 성립하므로 팩맨 장치를 기록한다. PPOPacman이 쓰는 resolve_device와 같은 규칙으로 정한다.
"""

from __future__ import annotations

import platform

import numpy as np


def environment_info(cfg) -> dict:
    pacman = cfg.get("pacman", None)  # 최소 설정(시험)에는 pacman 절이 없을 수 있다
    kind = str(pacman.kind) if pacman is not None else None
    info = {
        "python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(),
        "pacman_kind": kind,
        "pacman_device_setting": None, "pacman_device": None,
        "torch": None, "torch_cuda": None, "gpu": None,
    }
    if kind != "ppo":
        return info
    import torch

    from flypac.pacman.trainer import resolve_device

    setting = str(pacman.device)
    device = resolve_device(setting)
    info.update(pacman_device_setting=setting, pacman_device=device.type, torch=str(torch.__version__),
                torch_cuda=torch.version.cuda)
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device)
    return info
