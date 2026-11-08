"""가짜 엔진 (TRD 5.2, M0 지시서 4단계). 진짜 엔진(M1)과 같은 인터페이스를 가진 단순판.

단순화한 곳 (진짜 엔진에서 채운다):
- TODO(M1): 파워 쿠키 효과(겁먹음, z, 먹힘, 부활), 과일, 엘로이, 터널 감속은 생략. 파워 쿠키는 점수와
  ENERGIZER 사건만 만든다. 따라서 z는 항상 0이고 EATEN, Z_CHANGE, POWER_*는 생기지 않는다.
- 유령 집: 목숨이 시작되고 `release_ticks[k]` 틱 뒤에 k번 초파리가 출구를 통과한다. 엔진은
  `release_ticks`의 순서를 검사하지 않는다. 그래서 마지막으로 나오는 초파리가 3번이라는 보장은 없고,
  ALL_RELEASED의 초파리 번호는 마지막으로 나온 초파리의 번호다. 마지막 통과 순간 ALL_RELEASED가 생기고
  학습 활성이 켜진다. 0번 초파리는 집 밖에서 시작하고, 자기 차례에 "통과한 것으로" 표시만 한다.
- 모든 속도는 설정값(기본 1.0). 속도 누적기 구조는 그대로 둔다 (TRD 7.2).
엔진은 스스로 초기화하지 않는다 (reset_life, reset_game은 실행기만 부른다).
"""

from __future__ import annotations

import numpy as np

from flypac.core import directions as d
from flypac.core.events import Event, Kind, Winner
from flypac.core.types import GameView
from flypac.engine.game import Game, VecGame
from flypac.maps.original import MapSpec


class FakeGame(Game):
    """TRD 5.2 `Game`과 같은 인터페이스."""

    def __init__(self, map: MapSpec, cfg, rng: np.random.Generator):
        e = cfg.engine
        self.map = map
        self.rng = rng
        self.lives_max = int(e.lives)
        self.v_pac = float(e.pacman_speed)
        self.v_fly = float(e.fly_speed)
        self.release_ticks = [int(t) for t in e.release_ticks]
        self.score_dot = int(e.score_dot)
        self.score_energizer = int(e.score_energizer)
        self.watchdog = int(e.watchdog_ticks)
        self.F = len(map.fly_starts)
        if len(self.release_ticks) != self.F:
            raise ValueError("engine.release_ticks의 길이는 초파리 수와 같아야 한다")
        self._house = np.zeros(map.nbr.shape[0], dtype=bool)
        self._house[map.house_cells] = True
        self._house[map.door_cells] = True
        self.reset_game()

    # ---- 초기화 (실행기만 부른다) ----
    def reset_game(self) -> None:
        self.lives = self.lives_max
        self.dots = self.map.dots.copy()
        self.energizers = self.map.energizers.copy()
        self.score = 0
        self.tick = 0
        self.reset_life()

    def retire(self) -> None:
        """모든 판을 끝낸 칸을 은퇴시킨다. 새 판은 시작하지 않고 학습 활성만 내린다. 동결은 그대로."""
        self.learning_active = False

    def reset_life(self) -> None:
        m = self.map
        self.pac_cell = int(m.pacman_start)
        self.pac_dir = d.WEST
        self.pac_acc = 0.0
        self.fly_cell = m.fly_starts.astype(np.int64).copy()
        self.fly_dir = np.array([d.WEST] + [d.NORTH] * (self.F - 1), dtype=np.int64)
        self.fly_acc = np.zeros(self.F)
        self.released = np.zeros(self.F, dtype=bool)
        self.life_tick = 0
        self.learning_active = False
        self.frozen = False

    # ---- 사건 ----
    def _ev(self, kind: Kind, fly: int = -1, cause=None) -> Event:
        return Event(kind, self.index, self.tick, fly, self.learning_active, cause)

    def _catch(self, k: int, events: list) -> None:
        events.append(self._ev(Kind.CATCH, k))
        self.lives -= 1
        events.append(self._ev(Kind.LIFE_LOST))
        if self.lives <= 0:
            events.append(self._ev(Kind.GAME_OVER, cause=Winner.FLIES))
        self.frozen = True

    def _collision_with_pacman(self, events: list, fly: int | None = None) -> bool:
        """한 캐릭터가 움직일 때마다 부른다. 같은 칸이면 포획."""
        flies = range(self.F) if fly is None else [fly]
        for k in flies:
            if not self._house[self.fly_cell[k]] and self.fly_cell[k] == self.pac_cell:
                self._catch(k, events)
                return True
        return False

    # ---- 팩맨 ----
    def step_pacman(self, action: int) -> list[Event]:
        events: list[Event] = []
        if self.frozen:
            return events
        self.pac_acc += self.v_pac
        if self.pac_acc < 1.0:
            return events
        self.pac_acc -= 1.0
        mask = self.map.move_mask["pacman"][self.pac_cell]
        if 0 <= action < 4 and mask[action]:
            direction = int(action)
        elif mask[self.pac_dir]:
            direction = self.pac_dir  # 막히면 가던 방향
        else:
            return events  # 갈 곳이 없다
        self.pac_dir = direction
        self.pac_cell = int(self.map.nbr[self.pac_cell, direction])
        # 먹기가 충돌보다 먼저다 (PRD 3.7)
        if self.dots[self.pac_cell]:
            self.dots[self.pac_cell] = False
            self.score += self.score_dot
            events.append(self._ev(Kind.DOT))
        elif self.energizers[self.pac_cell]:
            self.energizers[self.pac_cell] = False
            self.score += self.score_energizer
            events.append(self._ev(Kind.ENERGIZER))
        if not (self.dots.any() or self.energizers.any()):
            events.append(self._ev(Kind.GAME_OVER, cause=Winner.PACMAN))
            self.frozen = True
            return events
        self._collision_with_pacman(events)
        return events

    # ---- 초파리 ----
    def _fly_options(self, k: int) -> list[int]:
        """초파리 k가 갈 수 있는 절대 방향. 반전 금지(막다른 길이면 반전), 집으로 못 돌아간다."""
        cell, h = self.fly_cell[k], self.fly_dir[k]
        nb = self.map.nbr[cell]
        ok = [dd for dd in range(4) if nb[dd] >= 0 and not self._house[nb[dd]]]
        fwd = [dd for dd in ok if dd != (h + 2) % 4]
        return fwd if fwd else ok

    def decision_info(self) -> tuple[np.ndarray, np.ndarray]:
        legal = np.zeros((self.F, 4), dtype=bool)
        decide = np.zeros(self.F, dtype=bool)
        for k in range(self.F):
            if not self.released[k] or self._house[self.fly_cell[k]]:
                continue
            opts = self._fly_options(k)
            for dd in opts:
                legal[k, int(d.abs_to_ego(self.fly_dir[k], dd))] = True
            will_move = self.fly_acc[k] + self.v_fly >= 1.0
            decide[k] = len(opts) >= 2 and will_move and not self.frozen
        return legal, decide

    def _release(self, k: int, events: list) -> None:
        """k번 초파리가 집 출구를 통과한다 (초파리 이동 단계, TRD 4.5)."""
        exit_cell = int(self.map.fly_starts[0])
        if k > 0:
            self.fly_cell[k] = exit_cell
            self.fly_dir[k] = d.NORTH
        self.released[k] = True
        if self.released.all():
            events.append(self._ev(Kind.ALL_RELEASED, k))  # 사건 자체는 활성 이전에 생긴다
            self.learning_active = True

    def _move_fly(self, k: int, action: int, events: list) -> tuple[bool, bool]:
        """반환 (moved, committed). DECISIONS 4절의 표: 집 출구 통과와 복도 이동은 moved만 참이다.

        committed는 결정 지점(반전 외 선택지 2개 이상)에서 받은 행동 ≥ 0으로 이동했을 때만 참이다.
        이동한 칸에서 포획되어도 참이다. 속도가 모자라 움직이지 않으면 둘 다 거짓 (제안은 버린다).
        가짜 엔진은 z가 항상 0이라 지연 반전(결정 지점 2)이 없다.
        """
        if not self.released[k]:
            if self.life_tick < self.release_ticks[k]:
                return False, False
            self._release(k, events)
            self._collision_with_pacman(events, k)
            return True, False
        self.fly_acc[k] += self.v_fly
        if self.fly_acc[k] < 1.0:
            return False, False
        self.fly_acc[k] -= 1.0
        opts = self._fly_options(k)
        if not opts:
            return False, False
        h = int(self.fly_dir[k])
        committed = action >= 0 and len(opts) >= 2
        if action >= 0:
            direction = int(d.ego_to_abs(h, action))
            if direction not in opts:
                raise ValueError(f"초파리 {k}: 불가능한 행동 {action} (가능한 절대 방향 {opts})")
        else:  # 유일한 경로를 따라간다. 갈림길이면 직진 우선
            direction = opts[0] if len(opts) == 1 or h not in opts else h
        self.fly_dir[k] = direction
        self.fly_cell[k] = int(self.map.nbr[self.fly_cell[k], direction])
        self._collision_with_pacman(events, k)
        return True, committed

    def step_flies(self, actions: np.ndarray) -> tuple[list[Event], np.ndarray, np.ndarray, np.ndarray]:
        events: list[Event] = []
        moved = np.zeros(self.F, dtype=bool)
        committed = np.zeros(self.F, dtype=bool)
        active = np.full(self.F, self.learning_active, dtype=bool)
        if self.frozen:
            return events, moved, committed, active
        for k in range(self.F):
            active[k] = self.learning_active  # 이 초파리가 움직이는 순간의 값
            moved[k], committed[k] = self._move_fly(k, int(actions[k]), events)
            if self.frozen:  # 도중에 종료되면 나머지는 실행하지 않는다 (제안은 버림)
                break
        return events, moved, committed, active

    def tick_timers(self) -> list[Event]:
        events: list[Event] = []
        if self.frozen:
            # 동결된 틱도 틱이다: 번호는 올린다 (틱 번호의 주인은 실행기, 엔진은 같은 번호를 따른다. DECISIONS 8절)
            self.tick += 1
            return events
        self.life_tick += 1
        if self.tick + 1 >= self.watchdog:  # 감시 장치 (TRD 7.3, 6.2)
            # 사건은 틱 번호를 올리기 전에 만든다: 같은 틱의 다른 사건과 같은 번호를 갖는다
            events.append(self._ev(Kind.GAME_OVER, cause=Winner.CENSORED))
            self.frozen = True
        self.tick += 1
        return events

    def view(self) -> GameView:
        return GameView(
            pac_cell=self.pac_cell,
            pac_dir=self.pac_dir,
            lives=self.lives,
            power=False,
            power_ticks=0,
            ghost_streak=0,
            fly_cell=self.fly_cell.copy(),
            fly_dir=self.fly_dir.copy(),
            fly_z=np.zeros(self.F, dtype=bool),
            fly_in_house=self._house[self.fly_cell].copy(),
            dots=self.dots.copy(),
            energizers=self.energizers.copy(),
            fruit=False,
            tick=self.tick,
            score=self.score,
            learning_active=self.learning_active,
            elroy_stage=0,
            frozen=self.frozen,
            pac_acc=float(self.pac_acc),
            fly_acc=self.fly_acc.copy(),
        )


class FakeVecGame(VecGame):
    """TRD 5.2 `VecGame`과 같은 인터페이스. 묶는 대상만 FakeGame이다."""

    game_cls = FakeGame
