"""한 틱 처리와 판 시작 처리 (TRD 6.1, 6.2). 실행기와 학습 환경이 함께 쓰는 단 한 곳이다 (DECISIONS 6.1절).

여기에만 있는 것: 틱 번호(실행기가 주인, 8절), view.tick 대조, 사건 틱 다시 쓰기, committed/moved 처리와
조종기 통지 순서(4절), 종료 처리 순서(6.2), 판마다 (시드, 판 번호, 용도) 난수 갱신(11절).

한 틱은 두 단계다. 순서는 내부 상태로 강제한다 (begin 없이 finish를 부르면 오류).

    views = core.begin()           # 1번: views를 만들고 엔진 틱을 대조한다
    out = core.finish(pac_actions) # 2~13번 + 종료 처리. pac_actions는 begin의 views로 고른 팩맨 행동

냄새 장(odor=None), 기록(recorder=None), 체제 보상(regime=None)은 인자로 끈다 (PPO 학습 환경용).
끄면 냄새 · 페로몬 관측은 0이고, 조종기에는 보상 0을 준다.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from flypac.core.events import Event, Kind
from flypac.core.rng import GameRngs, RngGroup
from flypac.core.types import FlyObs, GameView
from flypac.senses.sense import sense

_END_KINDS = (Kind.LIFE_LOST, Kind.GAME_OVER)


def _slice_obs(obs: FlyObs, b: int) -> FlyObs:
    """한 게임의 관측 조각 (F,4) 모양. regime.emission에 넘긴다."""
    return FlyObs(obs.odor[b], obs.phero[b], obs.z[b], obs.legal[b], obs.decide[b])


@dataclass
class TickOut:
    """finish의 결과. now는 종료 처리(초기화) 전의 views다."""

    events: list[list[Event]]  # 칸별 사건. 틱 번호는 실행기의 번호
    now: list[GameView]        # 10번의 views (타이머 뒤, 초기화 전)
    running: np.ndarray        # (B,) 이번 틱에 진행 중이던 칸
    ends: np.ndarray           # (B,) 목숨이 끝난 칸 (판 끝 포함)
    over: np.ndarray           # (B,) 판이 끝난 칸
    started: np.ndarray        # (B,) 이번 틱 끝에 새 판을 시작한 칸
    fly_act: np.ndarray        # (B,F)
    moved: np.ndarray          # (B,F)
    committed: np.ndarray      # (B,F)


class TickCore:
    """B개 게임 칸의 한 틱과 판 시작. 부품은 속성으로 들고 매번 읽는다 (시험에서 바꿔 끼울 수 있다).

    engine_rngs: 엔진 용도의 GameRngs. 판 시작 때 (시드, 판 번호, "engine")으로 다시 만들어 reset_game에 준다.
    pacman: 팩맨 조종기 또는 None (행동을 밖에서 주는 학습 환경). 목숨 끝에 reset을 부르고 rngs를 갱신한다.
    n_games: 칸마다 할 판 수. None이면 끝없이 새 판을 시작한다.
    """

    F = 4

    def __init__(self, map, game, engine_rngs: GameRngs, ghosts, *, condition: str, pacman=None,
                 odor=None, phero=None, regime=None, recorder=None, c0: float = 1.0, s_odor: float = 0.0,
                 reset_odor_on_life_loss: bool = False, n_games: int | None = None, keep_games=()):
        if not isinstance(engine_rngs, GameRngs) or len(engine_rngs) != game.B:
            raise TypeError("engine_rngs에는 길이 B의 GameRngs(core.rng.game_rngs(seeds, 'engine'))를 넘긴다")
        if engine_rngs.purpose != "engine":
            raise ValueError(f"engine_rngs의 용도가 'engine'이 아니다: {engine_rngs.purpose}")
        if phero is not None and regime is None:
            raise ValueError("페로몬 장에는 체제(regime)가 필요하다 (방출량)")
        self.map, self.game, self.engine_rngs = map, game, engine_rngs
        self.ghosts, self.pacman = ghosts, pacman
        self.odor, self.phero, self.regime, self.recorder = odor, phero, regime, recorder
        self.condition = str(condition)
        self.pass_views = self.condition == "b1"  # 원칙 2: views는 B1 조건에서만 조종기에 넘긴다
        self.B = game.B
        self.n_games = None if n_games is None else int(n_games)
        # c0 기본값 1은 냄새 장을 끈 경우용: 농도가 0이라 관측 ln(1 + 0/c0) = 0이고 c0 값과 무관하다
        self.c0, self.s_odor = float(c0), float(s_odor)
        self.reset_odor_on_life_loss = bool(reset_odor_on_life_loss)
        self.keep_games = set(keep_games)
        self.weight_writer = None  # 가중치 저장기 (recording/weights.py). 없으면 저장하지 않는다 (DECISIONS 26.1)
        self.game_no = np.ones(self.B, dtype=np.int64)  # 칸별 현재 판 번호 (1부터)
        # 칸별 현재 판의 틱 번호 (0부터). 틱 번호의 주인은 여기 하나다: 동결된 틱을 포함해 매 틱 올리고
        # 새 판에서 0으로 돌린다. 사건의 틱 번호는 이 값을 쓴다 = 재생 프레임 번호 (DECISIONS 8절)
        self.tick_no = np.zeros(self.B, dtype=np.int64)
        self.finished = np.zeros(self.B, dtype=bool)
        self.total_ticks = 0
        self.results: list[dict] = []  # 끝난 판마다 {slot, game_no, winner}
        self._views: list[GameView] | None = None  # begin의 views. None이면 begin 전
        self._running: np.ndarray | None = None

    # ---- 판 시작 ----
    def _rng_bundles(self) -> list:
        out = [self.engine_rngs]
        for owner in (self.ghosts, self.pacman):
            bundle = getattr(owner, "rngs", None)
            if isinstance(bundle, (GameRngs, RngGroup)):
                out.append(bundle)
        return out

    def _start_games(self, mask: np.ndarray) -> None:
        """mask 칸에서 판 self.game_no[mask]를 시작한다: 틱 번호 0, 모든 난수 묶음을 (시드, 판 번호, 용도)로
        다시 만들고, 엔진에 새 엔진 난수를 준다 (DECISIONS 3절 · 11절)."""
        if not mask.any():
            return
        self.tick_no[mask] = 0
        for bundle in self._rng_bundles():
            bundle.renew(mask, self.game_no)
        self.game.reset_game(mask, self.engine_rngs)

    def start(self) -> None:
        """모든 칸에서 판 1을 시작한다. 실행 시작 때 한 번."""
        every = np.ones(self.B, dtype=bool)
        self.game_no[:] = 1
        self.finished[:] = False
        self._views = None
        if self.odor is not None:
            self.odor.reset(every)
        if self.phero is not None:
            self.phero.reset(every)
        if self.pacman is not None:
            self.pacman.reset(every)
        self.ghosts.reset_life(every)
        self._start_games(every)

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

    def begin(self) -> list[GameView]:
        """1번: views를 만들고 엔진 틱을 대조한다. 팩맨 행동은 이 views로 고른다."""
        if self._views is not None:
            raise RuntimeError("begin을 두 번 불렀다: 먼저 finish(팩맨 행동)를 부른다")
        running = ~self.finished
        views = self.game.views()
        self._check_engine_tick(views, running)
        self._views, self._running = views, running
        return views

    def finish(self, pac_act) -> TickOut:
        """2~13번과 종료 처리. pac_act (B,) 절대 방향: begin의 views로 고른 팩맨 행동."""
        if self._views is None:
            raise RuntimeError("begin 없이 finish를 불렀다")
        running = self._running
        self._views = self._running = None
        B, F, game = self.B, self.F, self.game
        pac_act = np.asarray(pac_act, dtype=np.int64)
        if pac_act.shape != (B,):
            raise ValueError(f"팩맨 행동 모양 {pac_act.shape} ≠ ({B},)")

        ev = self._stamp(game.step_pacman(pac_act))                       # 3
        views = game.views()                                              # 4 (다시 만든다)
        odor_c = self.odor.c if self.odor is not None else np.zeros((B, self.map.nbr.shape[0]))
        phero_c = self.phero.c if self.phero is not None else np.zeros_like(odor_c)
        obs = sense(odor_c, phero_c, views, game.decision_info(), self.c0, nbr=self.map.nbr)  # 5
        if self.recorder is not None:
            # 한 프레임의 views, obs, 냄새 장이 같은 순간이 되게 5번의 값을 복사해 둔다 (10번에서 장이 바뀐다)
            frame_odor, frame_phero = odor_c.copy(), phero_c.copy()
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
        if self.odor is not None:
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
        if self.regime is not None:
            for b in range(B):
                if running[b]:
                    R[b] = self.regime.rewards(events[b])
        self.ghosts.give_reward(R)

        if self.recorder is not None:                                     # 12
            brain = {"action": fly_act, "pi": getattr(self.ghosts, "last_pi", np.zeros((B, F, 4))),
                     "committed": committed, "moved": moved}  # 재생 로그 형식 3 (DECISIONS 6절)
            if hasattr(self.ghosts, "replay_info"):  # 뇌 필드 (TRD 19.13): 뇌 담당이 제공
                brain.update(self.ghosts.replay_info())
            for b in range(B):
                if running[b]:
                    self.recorder.on_events(b, events[b], views[b])
                    self.recorder.replay_frame(b, views, obs, frame_odor, frame_phero, brain)

        self.total_ticks += 1
        self.tick_no[running] += 1   # 동결된 틱 포함, 진행 중인 칸은 매 틱. 새 판은 _start_games가 0으로
        self._ended = None
        self._end_of_tick(events, running)                                # 13
        ends, over, started = self._ended
        return TickOut(events, now, running, ends, over, started, fly_act, moved, committed)

    # ---- 종료 처리 (TRD 6.2) ----
    def _end_of_tick(self, events: list[list[Event]], running: np.ndarray) -> None:
        """결과 (ends, over, 새 판)는 self._ended에 둔다."""
        B = self.B
        none = np.zeros(B, dtype=bool)
        ends = np.array([bool(running[b] and any(e.kind in _END_KINDS for e in events[b])) for b in range(B)])
        self._ended = (ends, none, none)
        if not ends.any():
            return
        # 1. 추가 행동 중단: 이미 동결되어 있다 (6, 7, 9, 10번에서 제외됨)
        # 2. 마지막 학습: 학습 활성 플래그가 아직 켜진 채로 (초기화는 4번)
        self.ghosts.flush_learning(ends)
        # 3. 종료 상태 기록: 사건 처리가 끝난 뒤 view를 다시 읽는다
        over = np.zeros(B, dtype=bool)
        final = self.game.views() if self.recorder is not None else None
        for b in np.flatnonzero(ends):
            if self.recorder is not None:
                self.recorder.on_life_end(int(b), final[b], events[b])
            winner = next((e.winner for e in events[b] if e.kind is Kind.GAME_OVER), None)
            if winner is not None:
                over[b] = True
                if self.weight_writer is not None:  # flush_learning 뒤, reset_life 전 (DECISIONS 26.1)
                    self.weight_writer.on_game_end(int(b), int(self.game_no[b]), self.ghosts)
                if self.recorder is not None:
                    self.recorder.on_game_end(int(b), final[b], events[b])
                    self.recorder.replay_end(int(b), int(self.game_no[b]) in self.keep_games)
                self.results.append({"slot": int(b), "game_no": int(self.game_no[b]), "winner": str(winner)})
        # 4. 초기화: 조종기와 팩맨 이력, 냄새 장, 게임. 학습 활성은 게임 초기화(또는 은퇴)에서 해제된다
        if self.pacman is not None:
            self.pacman.reset(ends)
        self.ghosts.reset_life(ends)
        field_reset = over | (ends & self.reset_odor_on_life_loss)
        if field_reset.any():
            if self.odor is not None:
                self.odor.reset(field_reset)
            if self.phero is not None:
                self.phero.reset(field_reset)
        more = over if self.n_games is None else over & (self.game_no < self.n_games)
        retired = over & ~more  # 모든 판을 끝낸 칸: 새 판은 시작하지 않고 학습 활성만 내린다
        self.finished |= retired
        self.game.retire(retired)
        self.game_no[more] += 1
        self._start_games(more)  # 새 판: 틱 번호 0, 난수 갱신, 엔진 reset_game(rng)
        self.game.reset_life(ends & ~over)
        self._ended = (ends, over, more)
