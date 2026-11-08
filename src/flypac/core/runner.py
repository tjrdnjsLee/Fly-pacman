"""실행기 (TRD 6). 한 틱 순서(6.1)와 종료 처리(6.2)는 이 파일 한 곳에만 있다.

B개 게임 칸을 나란히 돌린다. 칸마다 자기 시드로 `n_games`판을 연속으로 하고,
판 초기화는 기록 뒤에 여기서 한다. 엔진은 스스로 초기화하지 않는다.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from flypac.core.events import Event, Kind
from flypac.core.rng import GameRngs, RngGroup, game_rngs, make_rng
from flypac.core.types import FlyObs
from flypac.engine.fake import FakeVecGame
from flypac.engine.game import VecGame
from flypac.field.odor import Field
from flypac.ghosts.random_ghost import RandomGhost
from flypac.maps.original import load_original
from flypac.pacman.rule_based import RuleBasedPacman
from flypac.regimes import make_regime
from flypac.senses.sense import sense

_END_KINDS = (Kind.LIFE_LOST, Kind.GAME_OVER)


class NullRecorder:
    """아무것도 기록하지 않는 기록기 (시험 · 연기 시험용). Recorder와 같은 메서드를 가진다."""

    def on_events(self, game, events, view): ...
    def on_life_end(self, game, view, events): ...
    def on_game_end(self, game, view, events): ...
    def replay_frame(self, game, views, obs, odor, phero, brain): ...
    def replay_end(self, game, keep): ...
    def close(self): ...


def _slice_obs(obs: FlyObs, b: int) -> FlyObs:
    """한 게임의 관측 조각 (F,4) 모양. regime.emission에 넘긴다."""
    return FlyObs(obs.odor[b], obs.phero[b], obs.z[b], obs.legal[b], obs.decide[b])


class Runner:
    def __init__(self, cfg, map, game: VecGame, odor: Field, phero: Field | None, pacman, ghosts, regime,
                 recorder):
        self.cfg, self.map = cfg, map
        self.condition = str(cfg.condition)
        self.pass_views = self.condition == "b1"  # 원칙 2: views는 B1 조건에서만 조종기에 넘긴다
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
        # 칸별 현재 판의 틱 번호 (0부터). 틱 번호의 주인은 실행기 하나다: 동결된 틱을 포함해 매 틱 올리고
        # 새 판에서 0으로 돌린다. 사건의 틱 번호는 이 값을 쓴다 = 재생 프레임 번호 (DECISIONS 8절)
        self.tick_no = np.zeros(self.B, dtype=np.int64)
        self._new_game = np.zeros(self.B, dtype=bool)
        self.finished = np.zeros(self.B, dtype=bool)
        self.total_ticks = 0
        self.results: list[dict] = []  # 끝난 판마다 {slot, game_no, winner}

    # ---- 초기화 ----
    def _check_views_rule(self) -> None:
        """실행 시작 때 한 번: 조종기가 views를 쓰는지가 조건과 맞아야 한다 (TRD 5.5, 원칙 2)."""
        if bool(self.ghosts.uses_views) != self.pass_views:
            raise ValueError(
                f"조건 {self.condition!r}에서 조종기 {type(self.ghosts).__name__}의 uses_views="
                f"{self.ghosts.uses_views}: views를 쓰는 조종기는 조건 'b1'에서만 허용한다")

    def _reset_all(self) -> None:
        self._check_views_rule()
        every = np.ones(self.B, dtype=bool)
        self.odor.reset(every)
        if self.phero is not None:
            self.phero.reset(every)
        self.pacman.reset(every)
        self.ghosts.reset_life(every)

    # ---- 한 틱 (TRD 6.1) ----
    def _stamp(self, events: list[list[Event]]) -> list[list[Event]]:
        """사건의 틱 번호를 실행기의 번호로 쓴다 (엔진이 붙인 번호를 믿지 않는다)."""
        return [[e if e.tick == t else replace(e, tick=t) for e in evs]
                for evs, t in zip(events, self.tick_no.tolist())]

    def _check_engine_tick(self, views, running) -> None:
        """엔진의 틱 번호(view.tick)는 실행기의 번호를 따라야 한다. 다르면 엔진 버그로 멈춘다."""
        for b in np.flatnonzero(running):
            if int(views[b].tick) != int(self.tick_no[b]):
                raise RuntimeError(f"칸 {b}: 엔진 틱 {views[b].tick} ≠ 실행기 틱 {self.tick_no[b]} "
                                   "(엔진은 동결된 틱에도 틱 번호를 올려야 한다, DECISIONS 8절)")

    def tick(self) -> None:
        B, F = self.B, self.F
        running = ~self.finished
        game = self.game
        self._new_game = np.zeros(B, dtype=bool)

        views = game.views()                                              # 1
        self._check_engine_tick(views, running)
        pac_act = self.pacman.act(views)                                  # 2
        ev = self._stamp(game.step_pacman(pac_act))                       # 3
        views = game.views()                                              # 4 (다시 만든다)
        phero_c = self.phero.c if self.phero is not None else np.zeros_like(self.odor.c)
        obs = sense(self.odor.c, phero_c, views, game.decision_info(), self.c0, nbr=self.map.nbr)  # 5
        # 한 프레임의 views, obs, 냄새 장이 같은 순간이 되게 5번의 값을 복사해 둔다 (10번에서 장이 바뀐다)
        frame_odor, frame_phero = self.odor.c.copy(), phero_c.copy()
        alive = np.array([not v.frozen for v in views]) & running
        fly_act = self.ghosts.propose(obs, alive, views if self.pass_views else None)             # 6
        ev2, moved, committed, active = game.step_flies(fly_act)          # 7 (DECISIONS 4절의 표)
        ev2 = self._stamp(ev2)
        released = np.zeros(B, dtype=bool)
        eaten = np.zeros((B, F), dtype=bool)
        for b in range(B):
            for e in ev[b] + ev2[b]:
                if e.kind is Kind.ALL_RELEASED:
                    released[b] = True
                elif e.kind is Kind.EATEN:
                    eaten[b, e.fly] = True
        # 8: on_release → commit → on_eaten (총책임자 결정, DECISIONS 4절). TRD의 commit → on_release 순서면
        # 같은 틱 ALL_RELEASED 뒤의 첫 활성 결정 흔적이 on_release(e=0)로 지워진다. commit은 결정별 active로
        # 거르므로 순서를 바꿔도 비활성 결정은 흔적에 들어가지 않는다.
        if released.any():
            self.ghosts.on_release(released)  # 학습 활성은 게임 단위
        self.ghosts.commit(committed, active)  # 확정된 결정만. 복도 · 집 출구 이동(moved만 참)은 넘기지 않는다
        if eaten.any():
            self.ghosts.on_eaten(eaten)
        timers = self._stamp(game.tick_timers())                          # 9
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

        brain = {"action": fly_act, "pi": getattr(self.ghosts, "last_pi", np.zeros((B, F, 4))),
                 "committed": committed, "moved": moved}  # 재생 로그 형식 3 (DECISIONS 6절)
        if hasattr(self.ghosts, "replay_info"):  # 뇌 필드 (TRD 19.13): 뇌 담당이 제공
            brain.update(self.ghosts.replay_info())
        R = np.zeros((B, F))                                              # 11
        for b in range(B):
            if running[b]:
                R[b] = self.regime.rewards(events[b])
        self.ghosts.give_reward(R)

        for b in range(B):                                                # 12
            if running[b]:
                self.recorder.on_events(b, events[b], views[b])
                self.recorder.replay_frame(b, views, obs, frame_odor, frame_phero, brain)

        self._end_of_tick(events, running)                                # 13
        self.total_ticks += 1
        self.tick_no[running] += 1   # 동결된 틱 포함, 진행 중인 칸은 매 틱
        self.tick_no[self._new_game] = 0

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
        # 4. 초기화: 조종기와 팩맨 이력, 냄새 장, 게임. 학습 활성은 게임 초기화(또는 은퇴)에서 해제된다
        self.pacman.reset(ends)
        self.ghosts.reset_life(ends)
        field_reset = over | (ends & self.reset_odor_on_life_loss)
        if field_reset.any():
            self.odor.reset(field_reset)
            if self.phero is not None:
                self.phero.reset(field_reset)
        more = over & (self.game_no < self.n_games)
        retired = over & ~more  # 모든 판을 끝낸 칸: 새 판은 시작하지 않고 학습 활성만 내린다
        self.finished |= retired
        self.game_no[more] += 1
        self._new_game = more
        bundle = getattr(self.ghosts, "rngs", None)
        if isinstance(bundle, (GameRngs, RngGroup)):  # 새 판: (시드, 판 번호, 용도)별 생성기로 바꾼다 (TRD 13.2)
            bundle.renew(more, self.game_no)
        self.game.retire(retired)
        self.game.reset_game(more)
        self.game.reset_life(ends & ~over)

    def run(self) -> dict:
        self._reset_all()
        guard = int(self.cfg.run.max_loop_ticks)
        while not self.finished.all():
            if self.total_ticks >= guard:
                raise RuntimeError(f"무한 루프 감시: {guard}틱 안에 끝나지 않았다")
            self.tick()
        close = getattr(self.recorder, "close", None)
        if close is not None:
            close()
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
    if cfg.ghosts.kind == "random":
        ghosts = RandomGhost(cfg, B, game_rngs(seeds, "ghost"))
    elif cfg.ghosts.kind == "flybrain":
        from flypac.brain.flybrain import FlyBrain, load_brain_config
        ghosts = FlyBrain(load_brain_config(cfg.ghosts.config), B, seeds, condition=str(condition))
    else:
        raise NotImplementedError(f"ghosts.kind={cfg.ghosts.kind}")  # TODO: original(B1)
    regime = make_regime(cfg)
    return Runner(cfg, amap, game, odor, phero, pacman, ghosts, regime, recorder or NullRecorder())
