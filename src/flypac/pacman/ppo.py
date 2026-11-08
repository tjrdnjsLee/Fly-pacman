"""PPO 팩맨 (TRD 5.7, 12.2): 학습한 체크포인트로 실험에서 행동한다. torch는 pacman 선택 묶음이다.

- 관측 · 프레임 이력 · 행동 마스크는 학습과 같은 코드(pacman/obs.py)를 쓴다. 관측 규격 버전이 다르면 멈춘다.
- 행동: 탐욕(가장 확률 높은 방향). 난이도 ε(TRD 12.2): 매 틱 확률 ε로 그 틱에 가능한 방향 중 균등하게 고른다.
- 프레임 이력은 실행기가 reset(mask)를 부르는 시점(목숨 · 판 종료)에 지운다. 학습 환경과 같은 시점이다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from flypac.core.config import Config
from flypac.core.rng import make_rng
from flypac.core.types import GameView
from flypac.maps.original import load_original
from flypac.pacman.base import PacmanController
from flypac.pacman.obs import NUM_CHANNELS, NUM_SCALARS, OBS_VERSION, FrameStack, ObsEncoder, legal_mask


class PPOPacman(PacmanController):
    def __init__(self, checkpoint: str, device: str, eps: float, rng: np.random.Generator | None = None):
        import torch

        from flypac.pacman.ppo_agent import build_network
        from flypac.pacman.trainer import resolve_device

        path = Path(checkpoint)
        s = torch.load(path, map_location="cpu", weights_only=False)
        if s.get("obs_version") != OBS_VERSION:
            raise ValueError(f"관측 규격 버전이 다르다: 체크포인트 {s.get('obs_version')}, 코드 {OBS_VERSION}")
        if not 0.0 <= float(eps) <= 1.0:
            raise ValueError(f"eps는 0 ~ 1이어야 한다: {eps}")
        cfg = Config(s["config"])
        p = cfg.pacman_train
        self.map = load_original()
        self.encoder = ObsEncoder(self.map, int(p.obs.power_ticks_max), int(cfg.engine.lives))
        self.k = int(p.obs.frame_stack)
        self.mask_reverse = bool(p.obs.mask_reverse)
        H, W = self.map.grid.shape
        self.device = resolve_device(device)
        self.net = build_network(p.network, (NUM_CHANNELS * self.k, H, W), NUM_SCALARS).to(self.device)
        self.net.load_state_dict(s["model"])
        self.net.eval()
        self.eps = float(eps)
        self.rng = rng if rng is not None else make_rng(0, 0, "pacman")
        self.checkpoint_update = int(s["update"])
        self.frames: FrameStack | None = None
        self._torch = torch

    def act(self, views: list[GameView]) -> np.ndarray:
        torch = self._torch
        B = len(views)
        if self.frames is None or self.frames.buf.shape[0] != B:
            self.frames = FrameStack(B, self.k, (NUM_CHANNELS, *self.map.grid.shape))
        grid, scalars = self.encoder.encode(views)
        stacked = self.frames.push(grid)
        legal = legal_mask(self.map, views, self.mask_reverse)
        with torch.no_grad():
            logits, _ = self.net(torch.as_tensor(stacked, device=self.device).float(),
                                 torch.as_tensor(scalars, device=self.device),
                                 torch.as_tensor(legal, device=self.device))
        act = logits.argmax(dim=-1).cpu().numpy().astype(np.int64)
        if self.eps > 0.0:
            for b in range(B):
                if self.rng.random() < self.eps:
                    act[b] = int(self.rng.choice(np.flatnonzero(legal[b])))
        return act

    def reset(self, mask: np.ndarray) -> None:
        if self.frames is None:
            return  # 첫 act에서 새로 만든다 (모두 비어 있음)
        self.frames.reset(mask)
