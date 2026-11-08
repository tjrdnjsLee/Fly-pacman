"""엔진 인터페이스 (TRD 5.2). 자리 파일: 진짜 엔진은 M1(팩맨 담당)이 채운다.

엔진은 스스로 초기화하지 않는다. reset_life, reset_game은 실행기(core/tick.py)만 부른다.

난수 계약 (관리자 결정, DECISIONS 3절 · 11절):
- 판을 시작할 때마다(판 1 포함) 실행기가 `reset_game(rng)`로 그 판의 생성기를 준다.
  rng = make_rng(시드, 판 번호, "engine"), 판 번호는 1부터.
- 엔진은 받은 rng로 자기 난수를 바꾸고, 다음 reset_game까지 그것만 쓴다. 다른 생성기를 만들거나
  전역 난수(np.random.*, random.*)를 쓰지 않는다. 그래서 한 칸의 엔진 난수열은 (시드, 판 번호)로만 정해지고,
  묶음 크기 · 칸 순서와 무관하다.
- reset_life는 난수를 바꾸지 않는다 (목숨이 바뀌어도 그 판의 난수열은 이어진다).
- 생성자의 rng는 TRD 5.2 모양 때문에 남은 자리다. 실행기가 곧바로 reset_game(rng)를 부르므로
  엔진은 생성자의 rng로 난수를 뽑지 않는다.

겁먹는 시간 (관리자 승인, DECISIONS 14.3절):
- 설정 `engine.power_s`(초)를 `core.timing.config_ticks(cfg, power_s)`로 틱으로 바꿔 쓴다. 엔진이 따로 변환하지
  않는다. PPO 팩맨 관측의 파워 분모(`pacman/obs.py`)가 같은 함수를 쓰므로, 엔진이 다른 값을 쓰면 관측이 어긋난다.

`GameView.ghost_streak` 계약 (PPO 팩맨의 먹힘 보상 순번이 이 값에서 나온다, DECISIONS 14.3절):
- 파워 쿠키를 먹어 파워가 시작될 때 0으로 돌아가고, 초파리를 먹을 때마다 1씩 오른다. 파워가 끝나도 지우지 않는다
  (다음 파워 시작까지 그대로).
- 따라서 매 틱 `view.ghost_streak − (그 틱의 EATEN 사건 수) ≥ 0`이고, 그 틱의 k번째(0부터) EATEN의 순번은
  `ghost_streak − (EATEN 수) + k`이다. 위반하면 학습 환경이 멈춘다.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from flypac.core import directions as d
from flypac.core.config import load_config
from flypac.core.events import Event, Kind, Winner, ZCause
from flypac.core.timing import config_ticks
from flypac.core.types import GameView
from flypac.maps.original import MapSpec


@lru_cache(maxsize=1)
def _default_engine_settings():
    """새 엔진 키가 없는 기존 설정은 저장소의 공통 참고값을 읽는다."""
    return load_config(Path(__file__).resolve().parents[3] / "configs" / "m0_smoke.yaml").engine


def _engine_setting(engine, key):
    return engine[key] if key in engine else _default_engine_settings()[key]


def advance_accumulator(accumulator: float, speed: float) -> tuple[float, bool]:
    """TRD 7.2: 점 감속의 음수 잔액도 유지하고, 한 틱에는 최대 한 칸 이동한다."""
    accumulator += speed
    moved = accumulator >= 1.0
    if moved:
        accumulator -= 1.0
    return accumulator, moved


class Game:
    """한 판 (TRD 5.2)."""

    index = 0  # 묶음 안의 게임 칸 번호. 사건의 `game` 값. VecGame이 정해 준다.

    def __init__(self, map: MapSpec, cfg, rng: np.random.Generator):
        e = cfg.engine
        self.map = map
        self.lives_max = int(e.lives)
        pac_speed, fly_speed = float(e.pacman_speed), float(e.fly_speed)
        pac_cap = float(e.get("pacman_normal_speed_cap", pac_speed))
        fly_cap = float(e.get("fly_normal_speed_cap", fly_speed))
        self.v_pac = min(pac_speed, pac_cap)
        self.v_fly = min(fly_speed, fly_cap)
        self.v_dot = float(e.get("pacman_dot_speed", self.v_pac))
        self.v_pac_power = float(_engine_setting(e, "pacman_power_speed"))
        self.v_dot_power = float(_engine_setting(e, "pacman_power_dot_speed"))
        self.v_frightened = float(_engine_setting(e, "fly_frightened_speed"))
        self.v_tunnel = float(_engine_setting(e, "fly_tunnel_speed"))
        self.elroy_thresholds = tuple(_engine_setting(e, "elroy_dots_left"))
        self.elroy_speeds = tuple(_engine_setting(e, "elroy_speeds"))
        self.power_ticks_max = config_ticks(cfg, e.power_s)
        self.fruit_duration = config_ticks(cfg, _engine_setting(e, "fruit_s"))
        self.fruit_thresholds = tuple(_engine_setting(e, "fruit_dots_eaten"))
        self.score_dot = int(e.score_dot)
        self.score_energizer = int(e.score_energizer)
        self.score_ghosts = tuple(_engine_setting(e, "score_ghosts"))
        self.score_fruit = int(_engine_setting(e, "score_fruit"))
        self.watchdog = int(e.watchdog_ticks)
        self.house_personal_limits = _engine_setting(e, "house_personal_dots")
        self.house_global_limits = _engine_setting(e, "house_global_dots")
        self.house_no_dot_ticks = config_ticks(cfg, _engine_setting(e, "house_no_dot_s"))
        if not all(0 <= speed <= 1 for speed in
                   (pac_speed, fly_speed, pac_cap, fly_cap, self.v_pac_power, self.v_frightened,
                    self.v_tunnel, *self.elroy_speeds)):
            raise ValueError("속도는 0 이상 1 이하여야 한다 (TRD 7.2)")
        for speed, dot_speed in ((self.v_pac, self.v_dot), (self.v_pac_power, self.v_dot_power)):
            if not 0 <= dot_speed <= speed or (speed > 0 and dot_speed == 0):
                raise ValueError("점 이동 속도는 양수이고 해당 상태의 이동 속도 이하여야 한다")
        if self.power_ticks_max <= 0 or not self.score_ghosts:
            raise ValueError("파워 지속 틱과 초파리 점수 목록이 필요하다")
        self.F = len(map.fly_starts)
        self._fly_start_dirs = np.array([d.WEST] + [d.NORTH] * (self.F - 1), dtype=np.int64)
        self._house = np.zeros(map.nbr.shape[0], dtype=bool)
        self._house[map.house_cells] = True
        self._house[map.door_cells] = True
        self._tunnel = np.zeros_like(self._house)
        self._tunnel[map.tunnel_cells] = True
        self._new_game()

    def step_pacman(self, action: int) -> list[Event]:
        """팩맨 이동, 먹기, 충돌. 목숨이 끝나면 그 틱 동안 동결. action: 절대 방향."""
        if self.frozen:
            return []
        if 0 <= action < len(d.ABS_NAMES):
            self.pac_requested_dir = int(action)
        speed = self.v_pac_power if self.power_ticks else self.v_pac
        dot_speed = self.v_dot_power if self.power_ticks else self.v_dot
        self.pac_acc, will_move = advance_accumulator(self.pac_acc, speed)
        if not will_move:
            return []
        legal = self.map.move_mask["pacman"][self.pac_cell]
        if legal[self.pac_requested_dir]:
            self.pac_dir = self.pac_requested_dir
        if not legal[self.pac_dir]:
            return []
        self.pac_cell = int(self.map.nbr[self.pac_cell, self.pac_dir])
        events: list[Event] = []
        if self.dots[self.pac_cell]:
            self.dots[self.pac_cell] = False
            self.score += self.score_dot
            self.pac_acc -= speed / dot_speed - 1
            events.append(self._ev(Kind.DOT))
            self._count_pellet()
        elif self.energizers[self.pac_cell]:
            self.energizers[self.pac_cell] = False
            self.score += self.score_energizer
            events.append(self._ev(Kind.ENERGIZER))
            self._count_pellet()
            self.power_ticks = self.power_ticks_max
            self.ghost_streak = 0
            events.append(self._ev(Kind.POWER_START))
            self._set_frightened(True, ZCause.POWER_START, events)
        if self.fruit_ticks and self.pac_cell == self.map.fruit_cell:
            self.fruit_ticks = 0
            self.score += self.score_fruit
            events.append(self._ev(Kind.FRUIT))
        if not (self.dots.any() or self.energizers.any()):
            events.append(self._ev(Kind.GAME_OVER, cause=Winner.PACMAN))
            self.frozen = True
            return events
        self._collision_with_pacman(events)
        return events

    def _ev(self, kind: Kind, fly: int = -1, cause=None) -> Event:
        return Event(kind, self.index, self.tick, fly, self.learning_active, cause)

    def _set_frightened(self, frightened: bool, cause: ZCause, events: list[Event]) -> None:
        for k in np.flatnonzero(self.fly_z != frightened):
            self.fly_z[k] = frightened
            events.append(self._ev(Kind.Z_CHANGE, int(k), cause))

    def _fly_speed(self, k: int) -> float:
        if self._tunnel[self.fly_cell[k]]:
            return self.v_tunnel
        if self.fly_z[k]:
            return self.v_frightened
        if k == 0 and self.elroy_stage:
            return max(self.v_fly, self.elroy_speeds[self.elroy_stage - 1])
        return self.v_fly

    def _update_elroy(self) -> None:
        if self.use_global_house_count and not (self.exiting[-1] or self.released[-1]):
            return  # 목숨을 잃으면 클라이드의 재출발까지 가속을 정지한다.
        remaining = np.count_nonzero(self.dots) + np.count_nonzero(self.energizers)
        self.elroy_stage = sum(remaining <= threshold for threshold in self.elroy_thresholds)

    def _waiting_fly(self) -> int | None:
        return next((k for k in range(self.F) if not self.released[k] and not self.exiting[k]), None)

    def _count_pellet(self) -> None:
        self.pellets_eaten += 1
        if self.pellets_eaten in self.fruit_thresholds:
            self.fruit_ticks = self.fruit_duration
        self.last_house_dot_tick = self.tick
        if self.use_global_house_count:
            self.house_global_count += 1
        else:
            waiting = self._waiting_fly()
            if waiting is not None:
                self.house_personal_count[waiting] += 1

    def _step_house(self, k: int, events: list[Event]) -> bool:
        """문 열로 이동한 뒤 북쪽으로 한 칸씩 나간다. 문 밖에서만 방출을 기록한다."""
        if not self.exiting[k]:
            if self._waiting_fly() != k:
                return False
            if self.use_global_house_count:
                ready = self.house_global_count >= self.house_global_limits[k - 1]
            else:
                ready = self.house_personal_count[k] >= self.house_personal_limits[k]
            if not ready:
                return False
            self.exiting[k] = True

        self.fly_acc[k], will_move = advance_accumulator(self.fly_acc[k], self._fly_speed(k))
        if not will_move:
            return False
        _, column = self.map.cell_rc[self.fly_cell[k]]
        _, exit_column = self.map.cell_rc[self.map.fly_starts[0]]
        direction = d.NORTH
        if column != exit_column:
            direction = d.EAST if column < exit_column else d.WEST
        self.fly_dir[k] = direction
        self.fly_cell[k] = self.map.nbr[self.fly_cell[k], direction]
        if not self._house[self.fly_cell[k]]:
            self.exiting[k] = False
            self.released[k] = True
            if not self.learning_active and self.released.all():
                events.append(self._ev(Kind.ALL_RELEASED, k))
                self.learning_active = True
            self._collision_with_pacman(events, k)
        return True

    def _collision_with_pacman(self, events: list[Event], fly: int | None = None) -> None:
        """캐릭터마다 이동 직후 검사한다. 맞교환도 팩맨 단계에서 먼저 포획된다."""
        for k in range(self.F) if fly is None else [fly]:
            if self._house[self.fly_cell[k]] or self.fly_cell[k] != self.pac_cell:
                continue
            if self.fly_z[k]:
                events.append(self._ev(Kind.EATEN, k))
                self.score += self.score_ghosts[min(self.ghost_streak, len(self.score_ghosts) - 1)]
                self.ghost_streak += 1
                self.fly_cell[k] = self.map.fly_starts[k]
                self.fly_dir[k] = self._fly_start_dirs[k]
                self.fly_acc[k] = 0.0
                self.exiting[k] = bool(self._house[self.fly_cell[k]])
                self.fly_z[k] = False
                events.append(self._ev(Kind.Z_CHANGE, k, ZCause.REVIVE))
                continue
            events.append(self._ev(Kind.CATCH, k))
            self.lives -= 1
            events.append(self._ev(Kind.LIFE_LOST))
            if self.lives == 0:
                events.append(self._ev(Kind.GAME_OVER, cause=Winner.FLIES))
            self.frozen = True
            break

    def step_flies(self, actions: np.ndarray) -> tuple[list[Event], np.ndarray, np.ndarray, np.ndarray]:
        """동결이 아니면 초파리 1→4번 이동, 매번 충돌. actions (F,) 자기중심 행동, -1은 유일한 경로.

        반환: (events, moved (F,) bool, committed (F,) bool, active (F,) bool). 도중 종료 시 나머지는
        실행하지 않음. moved = 이번 틱에 칸을 옮겼다. committed = 이번 틱의 제안을 엔진이 그 초파리의
        결정으로 받아들였다 (복도 · 집 출구 이동은 거짓, 지연 반전은 저장한 틱에 참). 언제 참인지는
        DECISIONS 4절의 표를 그대로 따른다.
        """
        events: list[Event] = []
        moved = np.zeros(self.F, dtype=bool)
        committed = np.zeros(self.F, dtype=bool)
        active = np.full(self.F, self.learning_active, dtype=bool)
        if self.frozen:
            return events, moved, committed, active
        for k in range(self.F):
            active[k] = self.learning_active
            if not self.released[k] or self._house[self.fly_cell[k]]:
                moved[k] = self._step_house(k, events)
                if self.frozen:
                    break
                continue
            self.fly_acc[k], will_move = advance_accumulator(self.fly_acc[k], self._fly_speed(k))
            if not will_move:
                continue
            options = self._fly_options(k)
            if not options:
                continue
            action, heading = int(actions[k]), int(self.fly_dir[k])
            if action >= 0:
                if action >= len(d.EGO_NAMES):
                    raise ValueError(f"초파리 {k}: 불가능한 행동 {action}")
                direction = int(d.ego_to_abs(heading, action))
                if direction not in options:
                    raise ValueError(f"초파리 {k}: 불가능한 행동 {action} (가능한 절대 방향 {options})")
            else:
                direction = heading if heading in options else options[0]
            self.fly_dir[k] = direction
            self.fly_cell[k] = int(self.map.nbr[self.fly_cell[k], direction])
            moved[k] = True
            committed[k] = action >= 0 and len(options) >= 2
            self._collision_with_pacman(events, k)
            if self.frozen:  # 포획을 만든 결정은 유지하고, 뒤 초파리의 제안은 버린다.
                break
        return events, moved, committed, active

    def _fly_options(self, k: int) -> list[int]:
        """집 재진입과 반전을 막고, 막다른 길에서만 반전을 허용한다 (TRD 4.7)."""
        neighbors = self.map.nbr[self.fly_cell[k]]
        options = [direction for direction, neighbor in enumerate(neighbors)
                   if neighbor >= 0 and not self._house[neighbor]]
        forward = [direction for direction in options if direction != d.opposite(self.fly_dir[k])]
        return forward or options

    def tick_timers(self) -> list[Event]:
        """동결이 아니면 파워 · 과일 · 유령 집 · 엘로이. 틱 번호(view.tick)는 동결된 틱에도 1 올린다.

        틱 번호의 주인은 실행기다: 사건의 틱 번호는 실행기가 다시 쓰고, view.tick이 실행기 번호와 다르면
        실행기가 멈춘다 (DECISIONS 8절).
        """
        events: list[Event] = []
        if not self.frozen:
            self.life_tick += 1
            if self.power_ticks:
                self.power_ticks -= 1
                if not self.power_ticks:
                    events.append(self._ev(Kind.POWER_END))
                    self._set_frightened(False, ZCause.POWER_END, events)
            if self.fruit_ticks:
                self.fruit_ticks -= 1
            if self.tick - self.last_house_dot_tick >= self.house_no_dot_ticks:
                waiting = self._waiting_fly()
                if waiting is not None:
                    self.exiting[waiting] = True
                    self.last_house_dot_tick = self.tick
            self._update_elroy()
            if self.tick + 1 >= self.watchdog:
                events.append(self._ev(Kind.GAME_OVER, cause=Winner.CENSORED))
                self.frozen = True
        self.tick += 1
        return events

    def view(self) -> GameView:
        """현재 상태의 읽기 전용 사본."""
        return GameView(
            pac_cell=self.pac_cell,
            pac_dir=self.pac_dir,
            lives=self.lives,
            power=self.power_ticks > 0,
            power_ticks=self.power_ticks,
            ghost_streak=self.ghost_streak,
            fly_cell=self.fly_cell.copy(),
            fly_dir=self.fly_dir.copy(),
            fly_z=self.fly_z.copy(),
            fly_in_house=self._house[self.fly_cell].copy(),
            dots=self.dots.copy(),
            energizers=self.energizers.copy(),
            fruit=self.fruit_ticks > 0,
            tick=self.tick,
            score=self.score,
            learning_active=self.learning_active,
            elroy_stage=self.elroy_stage,
            frozen=self.frozen,
            pac_acc=float(self.pac_acc),
            fly_acc=self.fly_acc.copy(),
        )

    def decision_info(self) -> tuple[np.ndarray, np.ndarray]:
        """(legal (F,4) bool, decide (F,) bool)."""
        legal = np.zeros((self.F, len(d.EGO_NAMES)), dtype=bool)
        decide = np.zeros(self.F, dtype=bool)
        for k in range(self.F):
            if not self.released[k] or self._house[self.fly_cell[k]]:
                continue
            options = self._fly_options(k)
            for direction in options:
                legal[k, int(d.abs_to_ego(self.fly_dir[k], direction))] = True
            will_move = self.fly_acc[k] + self._fly_speed(k) >= 1.0
            decide[k] = len(options) >= 2 and will_move and not self.frozen
        return legal, decide

    def retire(self) -> None:
        """모든 판을 끝낸 칸을 은퇴시킨다. 새 판은 시작하지 않고 `learning_active`만 거짓으로 내린다. 동결은 유지.

        TRD 5.2에 없는 메서드다 (총책임자 결정, DECISIONS 3절). 실행기만 부른다.
        """
        self.learning_active = False

    def reset_life(self) -> None:
        """새 목숨. 점 · 점수 · 목숨 수 · 난수는 그대로."""
        m = self.map
        self.pac_cell = int(m.pacman_start)
        self.pac_dir = d.WEST
        self.pac_requested_dir = self.pac_dir
        self.pac_acc = 0.0
        self.fly_cell = m.fly_starts.astype(np.int64).copy()
        self.fly_dir = self._fly_start_dirs.copy()
        self.fly_acc = np.zeros(self.F)
        self.fly_z = np.zeros(self.F, dtype=bool)
        self.power_ticks = 0
        self.fruit_ticks = 0
        self.elroy_stage = 0
        self.released = ~self._house[self.fly_cell]  # 처음부터 집 밖인 1번만 이동한다.
        self.exiting = np.zeros(self.F, dtype=bool)
        self.house_personal_count = np.zeros(self.F, dtype=np.int64)
        self.house_global_count = 0
        self.use_global_house_count = self.lives < self.lives_max
        self.last_house_dot_tick = self.tick - 1
        # DECISIONS 4.1의 7~11행은 후속 범위. 절대 방향 예약 자리만 확보한다.
        self.pending_fly_dir = np.full(self.F, -1, dtype=np.int64)
        self.life_tick = 0
        self.learning_active = False
        self.frozen = False

    def reset_game(self, rng: np.random.Generator) -> None:
        """새 판의 생성기를 받아 다음 판까지 사용한다 (DECISIONS 3절 · 11절)."""
        self.rng = rng
        self._new_game()

    def _new_game(self) -> None:
        self.lives = self.lives_max
        self.dots = self.map.dots.copy()
        self.energizers = self.map.energizers.copy()
        self.score = 0
        self.ghost_streak = 0
        self.pellets_eaten = 0
        self.tick = 0
        self.reset_life()


class VecGame:
    """Game B개를 도는 얇은 묶음 (TRD 5.2, 원칙 1). 규칙은 모두 Game에 있고 여기는 반복과 배열 쌓기만 한다.

    자동 초기화 없음. 각 메서드는 Game의 같은 이름 메서드를 B개에 적용한다.
    FakeVecGame은 `game_cls`만 바꾼다.
    """

    game_cls = Game

    def __init__(self, B: int, map: MapSpec, cfg, rngs: list[np.random.Generator]):
        if len(rngs) != B:
            raise ValueError(f"rngs 길이 {len(rngs)}가 B={B}와 다르다")
        self.B = B
        self.games = [self.game_cls(map, cfg, rng) for rng in rngs]
        for i, g in enumerate(self.games):
            g.index = i

    def step_pacman(self, actions: np.ndarray) -> list[list[Event]]:
        """actions (B,) 절대 방향."""
        return [g.step_pacman(int(a)) for g, a in zip(self.games, actions)]

    def step_flies(self, actions: np.ndarray) -> tuple[list[list[Event]], np.ndarray, np.ndarray, np.ndarray]:
        """actions (B,F). 반환: (events 게임별 목록, moved (B,F), committed (B,F), active (B,F))."""
        events, moved, committed, active = [], [], [], []
        for g, a in zip(self.games, actions):
            ev, mv, cm, ac = g.step_flies(a)
            events.append(ev)
            moved.append(mv)
            committed.append(cm)
            active.append(ac)
        return events, np.stack(moved), np.stack(committed), np.stack(active)

    def tick_timers(self) -> list[list[Event]]:
        return [g.tick_timers() for g in self.games]

    def views(self) -> list[GameView]:
        return [g.view() for g in self.games]

    def decision_info(self) -> tuple[np.ndarray, np.ndarray]:
        """(legal (B,F,4) bool, decide (B,F) bool)."""
        infos = [g.decision_info() for g in self.games]
        return np.stack([i[0] for i in infos]), np.stack([i[1] for i in infos])

    def reset_life(self, mask: np.ndarray) -> None:
        """mask (B,) bool: 초기화할 게임."""
        for g, m in zip(self.games, mask):
            if m:
                g.reset_life()

    def reset_game(self, mask: np.ndarray, rngs) -> None:
        """mask (B,) bool: 새 판을 시작할 게임. rngs: 칸별 엔진 생성기 (GameRngs, rngs[b]가 칸 b의 이번 판 생성기)."""
        for b, (g, m) in enumerate(zip(self.games, mask)):
            if m:
                g.reset_game(rngs[b])

    def retire(self, mask: np.ndarray) -> None:
        """mask (B,) bool: 은퇴시킬 게임 (모든 판을 끝낸 칸)."""
        for g, m in zip(self.games, mask):
            if m:
                g.retire()
