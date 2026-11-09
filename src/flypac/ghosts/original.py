"""기준선 B1. 원작 목표 칸에 가장 가까운 이웃을 고른다 (TRD 5.5)."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from flypac.brain.base import GhostController
from flypac.core import directions as d
from flypac.core.rng import GameRngs
from flypac.core.types import FlyObs, GameView
from flypac.engine.b1 import B1GameView, b1_settings
from flypac.maps.original import load_original


BLINKY, PINKY, INKY, CLYDE = range(4)
_PRIORITY = (d.NORTH, d.WEST, d.SOUTH, d.EAST)
_CLOCKWISE = (d.NORTH, d.EAST, d.SOUTH, d.WEST)


class OriginalGhosts(GhostController):
    """모드와 강제 반전은 엔진이 관리한다. 조종기는 정상 결정만 제안한다."""

    uses_views = True

    def __init__(self, cfg, B: int, rngs: GameRngs):
        if not isinstance(rngs, GameRngs) or len(rngs) != B:
            raise TypeError("OriginalGhosts에는 길이 B의 GameRngs를 넘긴다")
        if cfg.engine.kind != "real":
            raise ValueError("B1에는 engine.kind: real이 필요하다")
        settings = b1_settings(cfg)
        self.rngs, self.B = rngs, B
        self.map = load_original()
        self.corners = np.asarray(settings.scatter_targets, dtype=int)
        if self.corners.shape != (len(self.map.fly_starts), 2):
            raise ValueError("engine.b1.scatter_targets는 유령별 (행, 열) 네 쌍이어야 한다")
        self.pinky_ahead = int(settings.pinky_ahead)
        self.inky_ahead = int(settings.inky_ahead)
        self.clyde_distance_sq = int(settings.clyde_chase_distance) ** 2

    @staticmethod
    def _ahead(heading: int, tiles: int) -> np.ndarray:
        offset = tiles * np.asarray(d.STEP_RC[heading])
        if heading == d.NORTH:  # 원작의 북쪽 오프셋은 왼쪽에도 같은 거리만큼 치우친다.
            offset += tiles * np.asarray(d.STEP_RC[d.WEST])
        return offset

    def target_tile(self, view: B1GameView, fly: int) -> tuple[int, int]:
        pac = self.map.cell_rc[view.pac_cell]
        target = pac
        if view.scatter and not (fly == BLINKY and view.elroy_stage):
            target = self.corners[fly]
        elif fly == PINKY:
            target = pac + self._ahead(view.pac_dir, self.pinky_ahead)
        elif fly == INKY:
            pivot = pac + self._ahead(view.pac_dir, self.inky_ahead)
            target = 2 * pivot - self.map.cell_rc[view.fly_cell[BLINKY]]
        elif fly == CLYDE:
            diff = pac - self.map.cell_rc[view.fly_cell[CLYDE]]
            if diff @ diff < self.clyde_distance_sq:
                target = self.corners[fly]
        return int(target[0]), int(target[1])

    def _toward_target(self, view: B1GameView, fly: int, legal: np.ndarray) -> int:
        target = np.asarray(self.target_tile(view, fly))
        heading = view.fly_dir[fly]
        neighbors = self.map.nbr[view.fly_cell[fly]]
        best_action, best_distance = -1, float("inf")
        for direction in _PRIORITY:
            action = int(d.abs_to_ego(heading, direction))
            if legal[action]:
                delta = self.map.cell_rc[neighbors[direction]] - target
                distance = delta @ delta
                if distance < best_distance:
                    best_action, best_distance = action, distance
        return best_action

    @staticmethod
    def _frightened_action(heading: int, legal: np.ndarray, roll: float) -> int:
        start = int(roll * len(_CLOCKWISE))
        for direction in _CLOCKWISE[start:] + _CLOCKWISE[:start]:
            action = int(d.abs_to_ego(heading, direction))
            if legal[action]:
                return action
        return -1

    def propose(self, obs: FlyObs, alive: np.ndarray, views: list[GameView] | None = None) -> np.ndarray:
        if views is None or len(views) != self.B:
            raise ValueError("B1에는 칸별 엔진 view가 필요하다")
        out = np.full(obs.decide.shape, -1, dtype=np.int64)
        for b in np.flatnonzero(alive):
            view = views[b]
            if not isinstance(view, B1GameView):
                raise ValueError("B1에는 엔진이 관리하는 산개·추격 모드가 필요하다")
            flies = np.flatnonzero(obs.decide[b])
            # 확정될 결정 순서대로 난수를 미리 읽는다. 제안은 실제 생성기를 전진시키지 않는다.
            rolls = deepcopy(self.rngs[b]).random(len(flies)) if view.fly_z[flies].any() else None
            for i, fly in enumerate(flies):
                legal = obs.legal[b, fly]
                if view.fly_z[fly]:
                    out[b, fly] = self._frightened_action(view.fly_dir[fly], legal, rolls[i])
                else:
                    out[b, fly] = self._toward_target(view, fly, legal)
        return out

    def commit(self, committed: np.ndarray, active: np.ndarray) -> None:
        # B1은 학습 여부와 관계없이 확정된 결정에만 칸별 난수를 소비한다.
        for b, n in enumerate(np.count_nonzero(committed, axis=1)):
            if n:
                self.rngs[b].random(n)

    def give_reward(self, R: np.ndarray) -> None:
        pass

    def on_release(self, mask: np.ndarray) -> None:
        pass

    def on_eaten(self, mask: np.ndarray) -> None:
        pass

    def flush_learning(self, mask: np.ndarray) -> None:
        pass

    def reset_life(self, mask: np.ndarray) -> None:
        pass

    def state_dict(self) -> dict:
        return {"rngs": self.rngs.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.rngs.load_state_dict(state["rngs"])
