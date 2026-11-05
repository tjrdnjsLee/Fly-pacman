"""규칙 기반 팩맨 (TRD 5.7, 12.1). 팀원 노트북용 최소판.

가장 가까운 점으로 최단경로 이동. 위험한 초파리(z=0)가 가까우면 반대쪽으로, 겁먹은 초파리(z=1)가 가까우면 추격.
세부 규칙은 [미결 6] (팩맨 담당). 거리 기준값은 설정 파일의 rule_based 절에 둔다.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from flypac.core.types import GameView
from flypac.maps.original import MapSpec
from flypac.pacman.base import PacmanController

_FAR = 10**6


class RuleBasedPacman(PacmanController):
    def __init__(self, map: MapSpec, cfg, rng: np.random.Generator):
        # TODO(미결 6): 기준값은 자리표시. 팩맨 담당이 정한다.
        self.danger_dist = float(cfg.rule_based.danger_dist)
        self.chase_dist = float(cfg.rule_based.chase_dist)
        self.rng = rng
        self.map = map
        mask = map.move_mask["pacman"]
        rows, dirs = np.nonzero(mask)
        n = mask.shape[0]
        graph = csr_matrix((np.ones(len(rows)), (rows, map.nbr[rows, dirs])), shape=(n, n))
        dist = shortest_path(graph, unweighted=True, directed=True)
        self.dist = np.where(np.isfinite(dist), dist, _FAR)  # (N,N) 팩맨 이동 그래프의 칸 사이 거리

    def _best_dir(self, cell: int, target: int, toward: bool) -> int:
        """cell에서 갈 수 있는 방향 중 target과의 거리를 가장 줄이는(늘리는) 방향."""
        moves = np.flatnonzero(self.map.move_mask["pacman"][cell])
        if len(moves) == 0:
            return 0
        d = self.dist[self.map.nbr[cell, moves], target]
        return int(moves[np.argmin(d) if toward else np.argmax(d)])

    def _act_one(self, v: GameView) -> int:
        cell = v.pac_cell
        outside = ~v.fly_in_house
        d_fly = self.dist[cell, v.fly_cell]
        danger = outside & ~v.fly_z & (d_fly <= self.danger_dist)
        if danger.any():  # 가장 가까운 위험에서 멀어진다
            k = np.flatnonzero(danger)[np.argmin(d_fly[danger])]
            return self._best_dir(cell, int(v.fly_cell[k]), toward=False)
        prey = outside & v.fly_z & (d_fly <= self.chase_dist)
        if prey.any():
            k = np.flatnonzero(prey)[np.argmin(d_fly[prey])]
            return self._best_dir(cell, int(v.fly_cell[k]), toward=True)
        targets = np.flatnonzero(v.dots | v.energizers)
        if len(targets) == 0:
            return int(v.pac_dir)
        target = int(targets[np.argmin(self.dist[cell, targets])])
        return self._best_dir(cell, target, toward=True)

    def act(self, views: list[GameView]) -> np.ndarray:
        return np.array([self._act_one(v) for v in views], dtype=np.int64)

    def reset(self, mask: np.ndarray) -> None:
        pass  # 프레임 이력이 없다
