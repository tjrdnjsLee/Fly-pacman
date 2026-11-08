"""PPO 팩맨 학습 환경 (TRD 12.2). 한 스텝 = 엔진 한 틱 (TRD 6.1 순서에서 냄새 · 페로몬 · 기록을 뺀 것).

실험 실행기(core/runner.py)와 같은 순서로 엔진을 부른다:
팩맨 행동 → 팩맨 이동 → 초파리 관측 → 유령 제안 → 초파리 이동 → 조종기 통지 → 타이머 → 종료 처리.
유령 조종기 관측(FlyObs)의 냄새 · 페로몬은 0이다 (학습 상대는 B1 또는 무작위 유령이라 쓰지 않는다).

종료 처리 (실행기 6.2와 같은 시점):
- 목숨이 끝나면 프레임 이력을 지우고 엔진 목숨을 다시 시작한다. PPO의 done은 아니다.
- 판이 끝나면(GAME_OVER) done = 참. 게임을 새로 시작한다 (자동 초기화).
"""

from __future__ import annotations

import pickle

import numpy as np

from flypac.core.events import Kind
from flypac.core.rng import game_rngs, make_rng
from flypac.core.types import FlyObs
from flypac.engine.fake import FakeVecGame
from flypac.engine.game import VecGame
from flypac.maps.original import load_original
from flypac.pacman.obs import NUM_CHANNELS, FrameStack, ObsEncoder, legal_mask
from flypac.pacman.rewards import RewardFn

GHOST_KINDS = ("random", "original")


def make_ghosts(kind: str, cfg, B: int, slot_seeds: list[int]):
    if kind == "random":
        from flypac.ghosts.random_ghost import RandomGhost
        return RandomGhost(cfg, B, game_rngs(slot_seeds, "ghost"))
    if kind == "original":
        from flypac.ghosts.original import OriginalGhosts
        return OriginalGhosts(cfg, B, make_rng(slot_seeds[0], 0, "ghost"))
    raise ValueError(f"알 수 없는 유령 종류: {kind} (가능: {GHOST_KINDS})")


class PacmanTrainEnv:
    """B개 게임을 나란히 돌리는 학습 환경. 행동은 절대 방향(시계방향 번호) (B,)."""

    def __init__(self, cfg, B: int, seed: int, ghosts: str):
        self.cfg = cfg
        self.B = int(B)
        self.seed = int(seed)
        self.map = load_original()
        p = cfg.pacman_train
        self.frame_stack = int(p.obs.frame_stack)
        self.mask_reverse = bool(p.obs.mask_reverse)
        self.encoder = ObsEncoder(self.map, int(p.obs.power_ticks_max), int(cfg.engine.lives))
        self.reward_fn = RewardFn(p.rewards, self.map.cell_rc)
        # 칸 b의 시드는 seed * 10000 + b. 엔진 난수는 칸마다 하나를 계속 쓴다 (판이 바뀌어도 이어진다)
        self.slot_seeds = [self.seed * 10000 + b for b in range(self.B)]
        rngs = [make_rng(s, 0, "engine") for s in self.slot_seeds]
        kind = cfg.engine.kind
        if kind == "fake":
            self.game = FakeVecGame(self.B, self.map, cfg, rngs)
        elif kind == "real":
            self.game = VecGame(self.B, self.map, cfg, rngs)
        else:
            raise ValueError(f"알 수 없는 engine.kind: {kind}")
        self.ghost_kind = ghosts
        self.ghosts = make_ghosts(ghosts, cfg, self.B, self.slot_seeds)
        self.F = 4
        H, W = self.map.grid.shape
        self.grid_shape = (NUM_CHANNELS * self.frame_stack, H, W)
        self.frames = FrameStack(self.B, self.frame_stack, (NUM_CHANNELS, H, W))
        self.game_no = np.ones(self.B, dtype=np.int64)
        self.streak = np.zeros(self.B, dtype=np.int64)
        self.ep_return = np.zeros(self.B)
        self.ep_len = np.zeros(self.B, dtype=np.int64)
        self._views = None

    # ---- 초기화 ----
    def reset(self) -> dict:
        every = np.ones(self.B, dtype=bool)
        self.game.reset_game(every)
        self.ghosts.reset_life(every)
        self.frames.reset(every)
        self.streak[:] = 0
        self.ep_return[:] = 0.0
        self.ep_len[:] = 0
        return self._observe()

    def set_ghosts(self, kind: str) -> dict:
        """상대 유령을 바꾸고 모든 게임을 새로 시작한다 (커리큘럼 단계 전환)."""
        self.ghost_kind = kind
        self.ghosts = make_ghosts(kind, self.cfg, self.B, self.slot_seeds)
        return self.reset()

    def _observe(self) -> dict:
        self._views = self.game.views()
        grid, scalars = self.encoder.encode(self._views)
        return {"grid": self.frames.push(grid), "scalars": scalars}

    def legal_masks(self) -> np.ndarray:
        """(B,4) bool, 절대 방향 순서. 마지막 관측과 같은 순간의 상태로 만든다."""
        return legal_mask(self.map, self._views, self.mask_reverse)

    # ---- 한 스텝 ----
    def step(self, actions: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray, dict]:
        """반환: (관측, 보상 (B,), done (B,), 정보). 정보의 "finished"는 이번 스텝에 끝난 판 목록."""
        B, F, game, ghosts = self.B, self.F, self.game, self.ghosts
        ev = game.step_pacman(np.asarray(actions, dtype=np.int64))
        views = game.views()
        legal, decide = game.decision_info()
        z = np.stack([np.asarray(v.fly_z, dtype=np.float64) for v in views])
        zeros = np.zeros((B, F, 4))
        fobs = FlyObs(zeros, zeros, z, legal, decide)
        alive = np.array([not v.frozen for v in views])
        fly_act = ghosts.propose(fobs, alive, views if ghosts.uses_views else None)
        ev2, moved, committed, active = game.step_flies(fly_act)
        released = np.zeros(B, dtype=bool)
        eaten = np.zeros((B, F), dtype=bool)
        for b in range(B):
            for e in ev[b] + ev2[b]:
                if e.kind is Kind.ALL_RELEASED:
                    released[b] = True
                elif e.kind is Kind.EATEN:
                    eaten[b, e.fly] = True
        if released.any():
            ghosts.on_release(released)
        ghosts.commit(committed, active)
        if eaten.any():
            ghosts.on_eaten(eaten)
        timers = game.tick_timers()
        events = [ev[b] + ev2[b] + timers[b] for b in range(B)]
        now = game.views()

        rewards = np.zeros(B, dtype=np.float32)
        ends = np.zeros(B, dtype=bool)
        over = np.zeros(B, dtype=bool)
        finished = []
        for b in range(B):
            r, self.streak[b] = self.reward_fn(events[b], now[b], int(self.streak[b]))
            rewards[b] = r
            self.ep_return[b] += r
            self.ep_len[b] += 1
            for e in events[b]:
                if e.kind is Kind.LIFE_LOST:
                    ends[b] = True
                elif e.kind is Kind.GAME_OVER:
                    ends[b] = over[b] = True
                    finished.append({"slot": b, "score": int(now[b].score), "winner": str(e.winner),
                                     "return": float(self.ep_return[b]), "length": int(self.ep_len[b])})
        if ends.any():
            ghosts.flush_learning(ends)
            ghosts.reset_life(ends)
            self.frames.reset(ends)
            self.streak[ends] = 0
            if over.any():
                self.game_no[over] += 1
                bundle = getattr(ghosts, "rngs", None)
                if hasattr(bundle, "renew"):
                    bundle.renew(over, self.game_no)
                self.ep_return[over] = 0.0
                self.ep_len[over] = 0
                game.reset_game(over)
            game.reset_life(ends & ~over)
        return self._observe(), rewards, over, {"finished": finished}

    # ---- 저장 · 복원 (체크포인트 재개) ----
    def state_bytes(self) -> bytes:
        """엔진 · 유령 · 프레임 이력 · 칸별 계수를 통째로 담는다. 재개 뒤 같은 궤적을 다시 만든다."""
        return pickle.dumps({
            "game": self.game, "ghosts": self.ghosts, "ghost_kind": self.ghost_kind,
            "frames": self.frames, "game_no": self.game_no, "streak": self.streak,
            "ep_return": self.ep_return, "ep_len": self.ep_len,
        })

    def load_state_bytes(self, blob: bytes) -> dict:
        s = pickle.loads(blob)
        self.game, self.ghosts, self.ghost_kind = s["game"], s["ghosts"], s["ghost_kind"]
        self.frames, self.game_no, self.streak = s["frames"], s["game_no"], s["streak"]
        self.ep_return, self.ep_len = s["ep_return"], s["ep_len"]
        self._views = self.game.views()
        # 프레임 이력은 이미 들어 있다: push 없이 현재 묶음을 돌려준다
        grid, scalars = self.encoder.encode(self._views)
        Bk = self.frames.buf
        return {"grid": Bk.reshape(self.B, -1, *Bk.shape[3:]).copy(), "scalars": scalars}
