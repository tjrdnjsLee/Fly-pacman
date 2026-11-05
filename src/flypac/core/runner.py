"""실행기 (TRD 6). 한 틱 순서(6.1)와 종료 처리(6.2)는 이 파일 한 곳에만 있다.

B개 게임 칸을 나란히 돌린다. 칸마다 자기 시드로 `n_games`판을 연속으로 하고,
판 초기화는 기록 뒤에 여기서 한다. 엔진은 스스로 초기화하지 않는다.
"""

from __future__ import annotations

import numpy as np

from flypac.core.events import Event, Kind
from flypac.core.rng import make_rng
from flypac.core.types import FlyObs
from flypac.engine.fake import FakeVecGame
from flypac.engine.game import VecGame
from flypac.field.odor import Field
from flypac.ghosts.random_ghost import RandomGhost
from flypac.maps.original import load_original
from flypac.pacman.rule_based import RuleBasedPacman
from flypac.regimes.regime1 import Regime1
from flypac.regimes.regime2 import Regime2
from flypac.regimes.regime3 import Regime3
from flypac.senses.sense import sense

_END_KINDS = (Kind.LIFE_LOST, Kind.GAME_OVER)


class NullRecorder:
    """아무것도 기록하지 않는 기록기 (시험 · 연기 시험용). Recorder와 같은 메서드를 가진다."""

    def on_events(self, game, events, view): ...
    def on_life_end(self, game, view, events): ...
    def on_game_end(self, game, view, events): ...
    def replay_frame(self, game, views, obs, odor, phero, brain): ...
    def replay_end(self, game, keep): ...


def _slice_obs(obs: FlyObs, b: int) -> FlyObs:
    """한 게임의 관측 조각 (F,4) 모양. regime.emission에 넘긴다."""
    return FlyObs(obs.odor[b], obs.phero[b], obs.z[b], obs.legal[b], obs.decide[b])


class Runner:
    def __init__(self, cfg, map, game: VecGame, odor: Field, phero: Field | None, pacman, ghosts, regime,
                 recorder):
        self.cfg, self.map = cfg, map
        self.game, self.odor, self.phero = game, odor, phero
        self.pacman, self.ghosts, self.regime, self.recorder = pacman, ghosts, regime, recorder
        self.B = game.B
        self.F = 4
        self.n_games = int(cfg.n_games)
        self.c0 = float(cfg.odor.c0)
        self.s_odor = float(cfg.odor.s)
        self.reset_odor_on_life_loss = bool(cfg.odor.reset_on_life_loss)
        replay = cfg.get("replay", {})
        self.keep_games = set(replay.get("keep_games", [])) if replay else set()
        self.game_no = np.ones(self.B, dtype=np.int64)  # 칸별 현재 판 번호 (1부터)
        self.finished = np.zeros(self.B, dtype=bool)
        self.total_ticks = 0
        self.results: list[dict] = []  # 끝난 판마다 {slot, game_no, winner}

    # ---- 초기화 ----
    def _reset_all(self) -> None:
        every = np.ones(self.B, dtype=bool)
        self.odor.reset(every)
        if self.phero is not None:
            self.phero.reset(every)
        self.pacman.reset(every)
        self.ghosts.reset_life(every)

    # ---- 한 틱 (TRD 6.1) ----
    def tick(self) -> None:
        B, F = self.B, self.F
        running = ~self.finished
        game = self.game

        views = game.views()                                              # 1
        pac_act = self.pacman.act(views)                                  # 2
        ev = game.step_pacman(pac_act)                                    # 3
        views = game.views()                                              # 4 (다시 만든다)
        phero_c = self.phero.c if self.phero is not None else np.zeros_like(self.odor.c)
        obs = sense(self.odor.c, phero_c, views, game.decision_info(), self.c0, nbr=self.map.nbr)  # 5
        alive = np.array([not v.frozen for v in views]) & running
        # 원칙 2: views는 이를 쓰는 조종기(B1)에만 넘긴다
        fly_act = self.ghosts.propose(obs, alive, views if self.ghosts.uses_views else None)       # 6
        ev2, executed, active = game.step_flies(fly_act)                  # 7
        self.ghosts.commit(executed, active)                              # 8
        released = np.zeros(B, dtype=bool)
        eaten = np.zeros((B, F), dtype=bool)
        for b in range(B):
            for e in ev[b] + ev2[b]:
                if e.kind is Kind.ALL_RELEASED:
                    released[b] = True
                elif e.kind is Kind.EATEN:
                    eaten[b, e.fly] = True
        if released.any():
            self.ghosts.on_release(released)  # 학습 활성은 게임 단위
        if eaten.any():
            self.ghosts.on_eaten(eaten)
        timers = game.tick_timers()                                       # 9
        events: list[list[Event]] = [ev[b] + ev2[b] + timers[b] for b in range(B)]

        now = game.views()
        step_mask = np.array([not v.frozen for v in now]) & running       # 10 (동결 게임 제외)
        S = np.zeros_like(self.odor.c)
        for b in range(B):
            S[b, now[b].pac_cell] = self.s_odor
        self.odor.step(S, step_mask)
        if self.phero is not None:
            Sp = np.zeros_like(self.phero.c)
            for b in range(B):
                em = self.regime.emission(_slice_obs(obs, b))
                for k in range(F):
                    Sp[b, now[b].fly_cell[k]] += em[k]
            self.phero.step(Sp, step_mask)

        R = np.zeros((B, F))                                              # 11
        for b in range(B):
            if running[b]:
                R[b] = self.regime.rewards(events[b])
        self.ghosts.give_reward(R)

        for b in range(B):                                                # 12
            if running[b]:
                self.recorder.on_events(b, events[b], views[b])
                self.recorder.replay_frame(b, views, obs, self.odor.c, phero_c, None)

        self._end_of_tick(events, running)                                # 13
        self.total_ticks += 1

    # ---- 종료 처리 (TRD 6.2) ----
    def _end_of_tick(self, events: list[list[Event]], running: np.ndarray) -> None:
        ends = np.array([bool(running[b] and any(e.kind in _END_KINDS for e in events[b]))
                         for b in range(self.B)])
        if not ends.any():
            return
        # 1. 추가 행동 중단: 이미 동결되어 있다 (6, 7, 9, 10번에서 제외됨)
        # 2. 마지막 학습: 학습 활성 플래그가 아직 켜진 채로 (초기화는 4번)
        self.ghosts.flush_learning(ends)
        # 3. 종료 상태 기록: 사건 처리가 끝난 뒤 view를 다시 읽는다
        final = self.game.views()
        over = np.zeros(self.B, dtype=bool)
        for b in np.flatnonzero(ends):
            self.recorder.on_life_end(int(b), final[b], events[b])
            winner = next((e.winner for e in events[b] if e.kind is Kind.GAME_OVER), None)
            if winner is not None:
                over[b] = True
                self.recorder.on_game_end(int(b), final[b], events[b])
                self.recorder.replay_end(int(b), int(self.game_no[b]) in self.keep_games)
                self.results.append({"slot": int(b), "game_no": int(self.game_no[b]), "winner": str(winner)})
        # 4. 초기화: 조종기와 팩맨 이력, 냄새 장, 게임. 학습 활성은 게임 초기화에서 해제된다
        self.pacman.reset(ends)
        self.ghosts.reset_life(ends)
        field_reset = over | (ends & self.reset_odor_on_life_loss)
        if field_reset.any():
            self.odor.reset(field_reset)
            if self.phero is not None:
                self.phero.reset(field_reset)
        more = over & (self.game_no < self.n_games)
        self.finished |= over & ~more  # 마지막 판이 끝난 칸은 동결 상태로 둔다
        self.game_no[more] += 1
        self.game.reset_game(more)
        self.game.reset_life(ends & ~over)

    def run(self) -> dict:
        self._reset_all()
        guard = int(self.cfg.run.max_loop_ticks)
        while not self.finished.all():
            if self.total_ticks >= guard:
                raise RuntimeError(f"무한 루프 감시: {guard}틱 안에 끝나지 않았다")
            self.tick()
        return {"ticks": self.total_ticks, "games": list(self.results)}


def build_runner(cfg, recorder=None) -> Runner:
    """설정에서 부품을 만들어 Runner를 돌려준다. 기록기를 주지 않으면 NullRecorder."""
    amap = load_original()
    seeds = [int(s) for s in cfg.seeds]
    B = len(seeds)
    rngs = [make_rng(s, 0, "engine") for s in seeds]
    kind = cfg.engine.kind
    if kind == "fake":
        game = FakeVecGame(B, amap, cfg, rngs)
    elif kind == "real":
        game = VecGame(B, amap, cfg, rngs)
    else:
        raise ValueError(f"알 수 없는 engine.kind: {kind}")
    odor = Field(amap, float(cfg.odor.D), float(cfg.odor.lam))
    condition = cfg.condition
    phero = Field(amap, float(cfg.phero.D), float(cfg.phero.lam)) if condition == "regime3" else None
    if cfg.pacman.kind != "rule_based":
        raise NotImplementedError(f"pacman.kind={cfg.pacman.kind}")
    pacman = RuleBasedPacman(amap, cfg, make_rng(seeds[0], 0, "pacman"))
    if cfg.ghosts.kind != "random":
        raise NotImplementedError(f"ghosts.kind={cfg.ghosts.kind}")  # TODO: flybrain(뇌 담당), original(B1)
    ghosts = RandomGhost(cfg, B, make_rng(seeds[0], 0, "ghost"))
    # b0, b0p, b1은 학습하지 않으므로 보상 체제는 쓰이지 않는다. TODO: 기준선 조건의 체제 처리
    regime = {"regime2": Regime2, "regime3": Regime3}.get(condition, Regime1)(cfg)
    return Runner(cfg, amap, game, odor, phero, pacman, ghosts, regime, recorder or NullRecorder())
