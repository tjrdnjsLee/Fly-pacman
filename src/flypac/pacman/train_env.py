"""PPO 팩맨 학습 환경 (TRD 12.2). 한 스텝 = 엔진 한 틱.

한 틱 순서와 판 시작 처리는 core/tick.py의 `TickCore`가 가진 단 한 곳이다 (DECISIONS 6.1절). 학습 환경은 그것을
`begin()` → (밖에서 고른 팩맨 행동) → `finish(행동)` 두 단계로 쓴다. 냄새 · 페로몬 · 기록 · 체제 보상은 끈다
(odor=recorder=regime=None): 유령 관측의 냄새 · 페로몬은 0이다 (학습 상대는 B1 또는 무작위 유령이라 쓰지 않는다).

난수: 엔진 · 유령 모두 칸별 `GameRngs`(용도 "engine", "ghost")이고 판마다 (시드, 판 번호, 용도)로 다시 만든다.
실행기(core/runner.py)와 같은 이름 · 같은 칸 시드를 쓰므로 같은 시드 · 같은 팩맨 행동열이면 같은 사건열이 나온다
(tests/test_train_env_equiv.py). 칸 b의 시드는 `seed * 10000 + b`.

종료 처리 (실행기 6.2와 같은 시점, TickCore):
- 목숨이 끝나면 프레임 이력을 지우고 엔진 목숨을 다시 시작한다. PPO의 done은 아니다.
- 판이 끝나면(GAME_OVER) done = 참. 새 판이 곧바로 시작된다 (자동 초기화).

조건 이름은 추론하지 않는다: 커리큘럼 단계마다 설정에 적은 이름을 받는다. views를 쓰는 유령(B1)은 조건 "b1"이어야
views가 넘어간다 (원칙 2).
"""

from __future__ import annotations

import pickle

import numpy as np

from flypac.core.events import Kind
from flypac.core.rng import game_rngs
from flypac.core.tick import TickCore
from flypac.engine.fake import FakeVecGame
from flypac.engine.game import VecGame
from flypac.maps.original import load_original
from flypac.pacman.obs import NUM_CHANNELS, PacmanObserver, engine_constants
from flypac.pacman.rewards import RewardFn


def _random_ghosts(cfg, B, seeds):
    from flypac.ghosts.random_ghost import RandomGhost
    return RandomGhost(cfg, B, game_rngs(seeds, "ghost"))


def _original_ghosts(cfg, B, seeds):
    from flypac.ghosts.original import OriginalGhosts
    return OriginalGhosts(cfg, B, game_rngs(seeds, "ghost"))


# 유령 종류 → 만드는 함수. 실행기(core/runner.py)와 같은 용도 이름 "ghost"의 칸별 묶음을 준다.
GHOST_FACTORIES = {"random": _random_ghosts, "original": _original_ghosts}
GHOST_KINDS = tuple(GHOST_FACTORIES)


def make_ghosts(kind: str, cfg, B: int, slot_seeds: list[int]):
    if kind not in GHOST_FACTORIES:
        raise ValueError(f"알 수 없는 유령 종류: {kind} (가능: {GHOST_KINDS})")
    return GHOST_FACTORIES[kind](cfg, B, slot_seeds)


def make_game(cfg, B: int, amap, engine_rngs):
    kind = cfg.engine.kind
    rngs = [engine_rngs[b] for b in range(B)]
    if kind == "fake":
        return FakeVecGame(B, amap, cfg, rngs)
    if kind == "real":
        return VecGame(B, amap, cfg, rngs)
    raise ValueError(f"알 수 없는 engine.kind: {kind}")


def check_condition(ghosts, condition: str) -> None:
    """views를 쓰는 유령은 조건 "b1"에서만 허용한다 (TRD 5.5, 원칙 2). 실행기와 같은 규칙."""
    if bool(ghosts.uses_views) != (str(condition) == "b1"):
        raise ValueError(f"조건 {condition!r}에서 조종기 {type(ghosts).__name__}의 uses_views={ghosts.uses_views}: "
                         "views를 쓰는 조종기는 조건 'b1'에서만 허용한다 (커리큘럼의 condition을 확인한다)")


class PacmanTrainEnv:
    """B개 게임을 나란히 돌리는 학습 환경. 행동은 절대 방향(시계방향 번호) (B,)."""

    def __init__(self, cfg, B: int, seed: int, ghosts: str, condition: str):
        self.cfg = cfg
        self.B = int(B)
        self.seed = int(seed)
        self.map = load_original()
        p = cfg.pacman_train
        self.constants = engine_constants(cfg)  # 파워 분모 · 목숨 분모: 엔진 설정에서만
        self.observer = PacmanObserver(self.map, int(p.obs.frame_stack), bool(p.obs.mask_reverse), self.constants,
                                       self.B)
        self.reward_fn = RewardFn(p.rewards, self.map.cell_rc)
        self.slot_seeds = [self.seed * 10000 + b for b in range(self.B)]
        self.engine_rngs = game_rngs(self.slot_seeds, "engine")
        self.game = make_game(cfg, self.B, self.map, self.engine_rngs)
        self.ghost_kind, self.condition = ghosts, str(condition)
        self.ghosts = make_ghosts(ghosts, cfg, self.B, self.slot_seeds)
        check_condition(self.ghosts, self.condition)
        self.F = 4
        H, W = self.map.grid.shape
        self.grid_shape = (NUM_CHANNELS * int(p.obs.frame_stack), H, W)
        self.core = TickCore(self.map, self.game, self.engine_rngs, self.ghosts, condition=self.condition)
        self.ep_return = np.zeros(self.B)
        self.ep_len = np.zeros(self.B, dtype=np.int64)
        self._views = None

    @property
    def game_no(self) -> np.ndarray:
        return self.core.game_no

    # ---- 초기화 ----
    def reset(self) -> dict:
        self.core.start()
        self.observer.reset(np.ones(self.B, dtype=bool))
        self.ep_return[:] = 0.0
        self.ep_len[:] = 0
        return self._observe()

    def set_ghosts(self, kind: str, condition: str) -> dict:
        """상대 유령을 바꾸고 모든 게임을 새로 시작한다 (커리큘럼 단계 전환)."""
        ghosts = make_ghosts(kind, self.cfg, self.B, self.slot_seeds)
        check_condition(ghosts, condition)
        self.ghost_kind, self.condition, self.ghosts = kind, str(condition), ghosts
        self.core.ghosts, self.core.condition = ghosts, self.condition
        self.core.pass_views = self.condition == "b1"
        return self.reset()

    def _observe(self) -> dict:
        self._views = self.core.begin()
        return self.observer.observe(self._views)

    def legal_masks(self) -> np.ndarray:
        """(B,4) bool, 절대 방향 순서. 마지막 관측과 같은 순간의 상태로 만든다 (학습 · 탐욕용, 반전 금지 가능)."""
        return self.observer.legal(self._views)

    # ---- 한 스텝 ----
    def step(self, actions: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray, dict]:
        """반환: (관측, 보상 (B,), done (B,), 정보). 정보의 "finished"는 이번 스텝에 끝난 판 목록,
        "ends"는 목숨이 끝난 칸 (B,) (PPOPacman.reset을 부를 칸)."""
        before = self._views
        out = self.core.finish(actions)
        nxt = self.core.begin()  # 다음 틱의 views. 목숨 · 판이 끝난 칸은 초기화된 상태
        now = out.now
        rewards = np.zeros(self.B, dtype=np.float32)
        moved = np.zeros(self.B, dtype=bool)
        finished = []
        for b in range(self.B):
            events = out.events[b]
            moved[b] = int(now[b].pac_cell) != int(before[b].pac_cell)
            died = any(e.kind is Kind.LIFE_LOST for e in events)
            # 목숨을 잃었고 판이 계속되면 원 저장소처럼 위치를 되돌린 뒤의 배치로 근접 항을 계산한다
            prox = nxt[b] if (died and not out.over[b]) else now[b]
            r = self.reward_fn(events, now[b], prox, bool(moved[b]))
            rewards[b] = r
            self.ep_return[b] += r
            self.ep_len[b] += 1
            for e in events:
                if e.kind is Kind.GAME_OVER:
                    finished.append({"slot": b, "score": int(now[b].score), "winner": str(e.winner),
                                     "return": float(self.ep_return[b]), "length": int(self.ep_len[b])})
        self.ep_return[out.over] = 0.0
        self.ep_len[out.over] = 0
        self.observer.reset(out.ends)
        self._views = nxt
        obs = self.observer.observe(nxt)
        return obs, rewards, out.over.copy(), {"finished": finished, "ends": out.ends.copy(), "moved": moved}

    # ---- 저장 · 복원 (체크포인트 재개) ----
    def state_bytes(self) -> bytes:
        """TickCore(엔진 · 유령 · 난수 · 판 번호 · begin 상태) · 프레임 이력 · 칸별 계수를 통째로 담는다."""
        return pickle.dumps({
            "core": self.core, "ghost_kind": self.ghost_kind, "condition": self.condition,
            "frames": self.observer.frames, "ep_return": self.ep_return, "ep_len": self.ep_len,
        })

    def load_state_bytes(self, blob: bytes) -> dict:
        s = pickle.loads(blob)
        self.core = s["core"]
        self.game, self.ghosts, self.engine_rngs = self.core.game, self.core.ghosts, self.core.engine_rngs
        self.ghost_kind, self.condition = s["ghost_kind"], s["condition"]
        self.observer.frames = s["frames"]
        self.ep_return, self.ep_len = s["ep_return"], s["ep_len"]
        self._views = self.core._views  # 저장은 begin 뒤라 begin 상태 그대로 돌아온다
        # 프레임 이력은 이미 들어 있다: push 없이 현재 묶음을 돌려준다
        return {**self.observer.current(), "scalars": self.observer.encoder.encode(self._views)[1]}
