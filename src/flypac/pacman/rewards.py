"""PPO 학습 보상 (TRD 12.2 [권장]: 원 저장소 pacman-ai `engine/game.py` compute_reward의 설계).

원작 점수가 아니다 (TRD 18절 해석). 평가는 원작 점수 · 승률로 따로 기록한다.
값은 설정 파일 `rewards` 절에 둔다. 엔진 사건(core/events.py) → 보상:

| 사건 | 항목 |
|---|---|
| 매 스텝 | time_step |
| DOT | eat_pellet |
| ENERGIZER | eat_power_pellet |
| EATEN | eat_ghost[이번 파워에서 몇 번째인가 - 1] (마지막 값에서 멈춘다) |
| FRUIT | eat_fruit |
| GAME_OVER 승자 PACMAN | clear_level |
| LIFE_LOST | death |
| GAME_OVER 승자 FLIES | game_over (death에 더한다) |
| 위험한 유령이 맨해튼 거리 2칸 안 | 유령마다 ghost_proximity |
"""

from __future__ import annotations

import numpy as np

from flypac.core.events import Event, Kind, Winner
from flypac.core.types import GameView


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

    def __call__(self, events: list[Event], view: GameView, streak: int) -> tuple[float, int]:
        """한 게임 한 스텝의 보상. streak: 이번 파워에서 지금까지 먹은 유령 수. 반환 (보상, 새 streak)."""
        r = self.time_step
        for e in events:
            k = e.kind
            if k is Kind.DOT:
                r += self.dot
            elif k is Kind.ENERGIZER:
                r += self.energizer
                streak = 0
            elif k is Kind.EATEN:
                r += self.ghost[min(streak, len(self.ghost) - 1)]
                streak += 1
            elif k is Kind.FRUIT:
                r += self.fruit
            elif k is Kind.LIFE_LOST:
                r += self.death
            elif k is Kind.GAME_OVER:
                if e.winner is Winner.PACMAN:
                    r += self.clear
                elif e.winner is Winner.FLIES:
                    r += self.game_over
        if self.proximity != 0.0 and not view.frozen:
            pr, pc = self.cell_rc[view.pac_cell]
            out = ~np.asarray(view.fly_in_house, dtype=bool) & ~np.asarray(view.fly_z, dtype=bool)
            for cell in np.asarray(view.fly_cell)[out]:
                gr, gc = self.cell_rc[cell]
                # 원 저장소와 같이 터널로 감싸지 않은 맨해튼 거리
                if abs(int(gr) - int(pr)) + abs(int(gc) - int(pc)) <= self.proximity_dist:
                    r += self.proximity
        return r, streak
