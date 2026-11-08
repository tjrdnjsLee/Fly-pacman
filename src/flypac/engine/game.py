"""엔진 인터페이스 (TRD 5.2). 자리 파일: 진짜 엔진은 M1(팩맨 담당)이 채운다.

엔진은 스스로 초기화하지 않는다. reset_life, reset_game은 실행기만 부른다.
"""

from __future__ import annotations

import numpy as np

from flypac.core.events import Event
from flypac.core.types import GameView
from flypac.maps.original import MapSpec


class Game:
    """한 판 (TRD 5.2)."""

    index = 0  # 묶음 안의 게임 칸 번호. 사건의 `game` 값. VecGame이 정해 준다.

    def __init__(self, map: MapSpec, cfg, rng: np.random.Generator):
        raise NotImplementedError

    def step_pacman(self, action: int) -> list[Event]:
        """팩맨 이동, 먹기, 충돌. 목숨이 끝나면 그 틱 동안 동결. action: 절대 방향."""
        raise NotImplementedError

    def step_flies(self, actions: np.ndarray) -> tuple[list[Event], np.ndarray, np.ndarray, np.ndarray]:
        """동결이 아니면 초파리 1→4번 이동, 매번 충돌. actions (F,) 자기중심 행동, -1은 유일한 경로.

        반환: (events, moved (F,) bool, committed (F,) bool, active (F,) bool). 도중 종료 시 나머지는
        실행하지 않음. moved = 이번 틱에 칸을 옮겼다. committed = 이번 틱의 제안을 엔진이 그 초파리의
        결정으로 받아들였다 (복도 · 집 출구 이동은 거짓, 지연 반전은 저장한 틱에 참). 언제 참인지는
        DECISIONS 4절의 표를 그대로 따른다.
        """
        raise NotImplementedError

    def tick_timers(self) -> list[Event]:
        """동결이 아니면 파워 · 과일 · 유령 집 · 엘로이. 틱 번호(view.tick)는 동결된 틱에도 1 올린다.

        틱 번호의 주인은 실행기다: 사건의 틱 번호는 실행기가 다시 쓰고, view.tick이 실행기 번호와 다르면
        실행기가 멈춘다 (DECISIONS 8절).
        """
        raise NotImplementedError

    def view(self) -> GameView:
        """현재 상태의 읽기 전용 사본."""
        raise NotImplementedError

    def decision_info(self) -> tuple[np.ndarray, np.ndarray]:
        """(legal (F,4) bool, decide (F,) bool)."""
        raise NotImplementedError

    def retire(self) -> None:
        """모든 판을 끝낸 칸을 은퇴시킨다. 새 판은 시작하지 않고 `learning_active`만 거짓으로 내린다. 동결은 유지.

        TRD 5.2에 없는 메서드다 (총책임자 결정, DECISIONS 3절). 실행기만 부른다.
        """
        raise NotImplementedError

    def reset_life(self) -> None:
        raise NotImplementedError

    def reset_game(self) -> None:
        raise NotImplementedError


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

    def reset_game(self, mask: np.ndarray) -> None:
        for g, m in zip(self.games, mask):
            if m:
                g.reset_game()

    def retire(self, mask: np.ndarray) -> None:
        """mask (B,) bool: 은퇴시킬 게임 (모든 판을 끝낸 칸)."""
        for g, m in zip(self.games, mask):
            if m:
                g.retire()
