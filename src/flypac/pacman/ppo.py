"""PPO 팩맨 (TRD 5.7, 12.2): 학습한 체크포인트로 실험에서 행동한다. torch는 pacman 선택 묶음이다.

- 관측 · 프레임 이력 · 행동 마스크는 학습과 같은 코드(pacman/obs.py의 PacmanObserver)를 쓴다.
  관측 규격 버전 · 저장 형식 번호가 다르면 멈추고, 엔진 숫자(파워 분모 · 초기 목숨)가 실험 설정과 다르면 멈춘다.
  저장한 커밋이 지금 코드와 다르면 경고만 낸다.
- 행동: 탐욕(반전 금지 마스크 안에서 가장 확률 높은 방향). 난이도 ε(TRD 12.2): 매 틱 확률 ε로 엔진이 허용하는
  방향 전체(반전 포함) 중 균등하게 고른다 (DECISIONS 14.2절).
- 난수: 칸별 `GameRngs`("pacman")를 받는다. 칸 b는 rngs[b]만 쓰고, 판이 시작될 때 실행기(TickCore)가 다시 만든다.
- 프레임 이력은 실행기가 reset(mask)를 부르는 시점(목숨 · 판 종료)에 지운다. 학습 환경과 같은 시점이다.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from flypac.core.config import Config
from flypac.core.rng import GameRngs
from flypac.core.types import GameView
from flypac.maps.original import load_original
from flypac.pacman.base import PacmanController
from flypac.pacman.obs import NUM_CHANNELS, NUM_SCALARS, PacmanObserver, engine_constants


class PPOPacman(PacmanController):
    def __init__(self, checkpoint: str, device: str, eps: float, rngs: GameRngs, run_cfg):
        """run_cfg: 이 실험의 설정. 체크포인트에 기록된 엔진 숫자와 대조한다."""
        import torch

        from flypac.pacman.ppo_agent import build_network
        from flypac.pacman.trainer import check_checkpoint_header, resolve_device

        if not isinstance(rngs, GameRngs):
            raise TypeError("PPOPacman에는 GameRngs(core.rng.game_rngs(seeds, 'pacman'))를 넘긴다")
        if not 0.0 <= float(eps) <= 1.0:
            raise ValueError(f"eps는 0 ~ 1이어야 한다: {eps}")
        path = Path(checkpoint)
        s = torch.load(path, map_location="cpu", weights_only=False)
        check_checkpoint_header(s, path, warn=lambda m: warnings.warn(m, stacklevel=2))
        constants, now = s["engine_constants"], engine_constants(run_cfg)
        if constants != now:
            raise ValueError(f"엔진 숫자가 체크포인트와 다르다: 체크포인트 {constants}, 이 실험 {now} "
                             "(관측의 파워 · 목숨 분모가 어긋난다)")
        cfg = Config(s["config"])
        p = cfg.pacman_train
        self.map = load_original()
        self.k = int(p.obs.frame_stack)
        self.observer = PacmanObserver(self.map, self.k, bool(p.obs.mask_reverse), constants)
        H, W = self.map.grid.shape
        self.device = resolve_device(device)
        self.net = build_network(p.network, (NUM_CHANNELS * self.k, H, W), NUM_SCALARS).to(self.device)
        self.net.load_state_dict(s["model"])
        self.net.eval()
        self.eps = float(eps)
        self.rngs = rngs
        self.checkpoint_update = int(s["update"])
        self._torch = torch

    def act(self, views: list[GameView]) -> np.ndarray:
        torch = self._torch
        B = len(views)
        if len(self.rngs) != B:
            raise ValueError(f"rngs 길이 {len(self.rngs)} ≠ 게임 수 {B}")
        obs = self.observer.observe(views)
        legal = self.observer.legal(views)  # 탐욕: 반전 금지 마스크
        with torch.no_grad():
            logits, _ = self.net(torch.as_tensor(obs["grid"], device=self.device).float(),
                                 torch.as_tensor(obs["scalars"], device=self.device),
                                 torch.as_tensor(legal, device=self.device))
        act = logits.argmax(dim=-1).cpu().numpy().astype(np.int64)
        if self.eps > 0.0:
            free = self.observer.engine_legal(views)  # ε 무작위: 엔진이 허용하는 방향 전체 (반전 포함)
            for b in range(B):
                if self.rngs[b].random() < self.eps:
                    act[b] = int(self.rngs[b].choice(np.flatnonzero(free[b])))
        return act

    def reset(self, mask: np.ndarray) -> None:
        self.observer.reset(mask)
