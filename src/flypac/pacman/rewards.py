"""PPO 학습 보상 (TRD 12.2 [권장]: 원 저장소 pacman-ai `engine/game.py` compute_reward의 설계).

원작 점수가 아니다 (TRD 18절 해석). 평가는 원작 점수 · 승률로 따로 기록한다.
값은 설정 파일 `rewards` 절에 둔다. 엔진 사건(core/events.py) → 보상:

| 사건 | 항목 |
|---|---|
| 팩맨이 움직인 틱 | time_step |
| DOT | eat_pellet |
| ENERGIZER | eat_power_pellet |
| EATEN | eat_ghost[이번 파워에서 몇 번째인가 - 1] (마지막 값에서 멈춘다) |
| FRUIT | eat_fruit |
| GAME_OVER 승자 PACMAN | clear_level |
| LIFE_LOST | death |
| GAME_OVER 승자 FLIES | game_over (death에 더한다) |
| 팩맨이 움직인 틱(또는 목숨을 잃은 틱), 위험한 유령이 맨해튼 거리 2칸 안 | 유령마다 ghost_proximity |

시점 규칙 (관리자 결정, DECISIONS 13.1절):
- 시간 항과 근접 항은 팩맨이 실제로 움직인 틱(칸이 바뀐 틱)에만 준다. 서 있는 틱에 시간 벌점이나 근접 벌점이
  쌓이지 않게 한다. 사건 항(먹기 · 사망 · 클리어)은 움직임과 무관하다.
- 목숨을 잃은 틱의 근접 항은 원 저장소처럼 계산한다 (이동 여부와 무관): 목숨이 남으면 원 저장소가 위치를 처음으로
  되돌린 뒤 보상을 계산하므로 초기화 뒤 배치(`prox_view`)로, 판이 끝나면(되돌리지 않음) 잡힌 순간의 배치로 계산한다.
  어느 배치를 쓸지는 호출하는 쪽(train_env)이 `prox_view`로 정한다.
- 먹힘 순번은 엔진 상태에서 얻는다: 그 틱의 k번째 EATEN의 순번 = `view.ghost_streak − (EATEN 수) + k`
  (engine/game.py의 ghost_streak 계약). 계약을 어기면 멈춘다.
"""

from __future__ import annotations

import numpy as np

from flypac.core.events import Event, Kind, Winner
from flypac.core.types import GameView
from flypac.pacman.obs import dangerous_flies


class RewardFn:
    def __init__(self, rcfg, cell_rc: np.ndarray):
        self.time_step = float(rcfg["time_step"])
        self.dot = float(rcfg["eat_pellet"])
        self.energizer = float(rcfg["eat_power_pellet"])
        self.ghost = [float(x) for x in rcfg["eat_ghost"]]
        self.fruit = float(rcfg["eat_fruit"])
        self.clear = float(rcfg["clear_level"])
        self.death = float(rcfg["death"])
        self.game_over = float(rcfg["game_over"])
        self.proximity = float(rcfg["ghost_proximity"])
        self.proximity_dist = int(rcfg["ghost_proximity_dist"])
        self.cell_rc = cell_rc

    def proximity_count(self, view: GameView) -> int:
        """view에서 팩맨과 맨해튼 거리 proximity_dist 이내인 위험한 유령 수 (원 저장소와 같이 터널로 감싸지 않는다)."""
        pr, pc = self.cell_rc[view.pac_cell]
        n = 0
        for cell in np.asarray(view.fly_cell)[dangerous_flies(view)]:
            gr, gc = self.cell_rc[cell]
            if abs(int(gr) - int(pr)) + abs(int(gc) - int(pc)) <= self.proximity_dist:
                n += 1
        return n

    def __call__(self, events: list[Event], now: GameView, prox_view: GameView, moved: bool) -> float:
        """한 게임 한 틱의 보상.

        now: 틱 끝(타이머 뒤, 초기화 전)의 view. 먹힘 순번(ghost_streak)을 읽는다.
        prox_view: 근접 항을 계산할 배치 (모듈 docstring). moved: 팩맨의 칸이 바뀌었는가.
        """
        died = any(e.kind is Kind.LIFE_LOST for e in events)
        r = self.time_step if moved else 0.0
        n_eaten = sum(e.kind is Kind.EATEN for e in events)
        order = int(now.ghost_streak) - n_eaten
        if order < 0:
            raise RuntimeError(f"엔진 계약 위반: ghost_streak {now.ghost_streak} < 이번 틱 EATEN 수 {n_eaten} "
                               "(engine/game.py의 ghost_streak 계약)")
        for e in events:
            k = e.kind
            if k is Kind.DOT:
                r += self.dot
            elif k is Kind.ENERGIZER:
                r += self.energizer
            elif k is Kind.EATEN:
                r += self.ghost[min(order, len(self.ghost) - 1)]
                order += 1
            elif k is Kind.FRUIT:
                r += self.fruit
            elif k is Kind.LIFE_LOST:
                r += self.death
            elif k is Kind.GAME_OVER:
                if e.winner is Winner.PACMAN:
                    r += self.clear
                elif e.winner is Winner.FLIES:
                    r += self.game_over
        if self.proximity != 0.0 and (moved or died):
            r += self.proximity * self.proximity_count(prox_view)
        return r
