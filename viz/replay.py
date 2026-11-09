"""Load recorded state and place both views on the same replay timeline."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from flypac.core.directions import STEP_RC
from flypac.maps.original import load_original
from flypac.replay.reader import kind_names, read_replay

SIZE = (1920, 1080)
FPS = 30
TILE = 28
ORIGIN = (104, 146)
STEP_XY = np.asarray(STEP_RC)[:, ::-1]


class Replay:
    """Validated replay arrays, plus drawing data derived from the original map."""

    def __init__(self, path: Path):
        self.meta, self.arrays = read_replay(path)
        if self.meta["map_name"] != "original":
            raise ValueError(f"Unsupported replay map: {self.meta['map_name']!r}")
        self.map = load_original()
        a = self.arrays
        self.ticks = len(a["pac_cell"])
        if not self.ticks or a["odor_field"].shape[1] != len(self.map.cell_rc):
            raise ValueError("Expected a nonempty replay on the original map")

        self.cells = np.column_stack((a["pac_cell"], a["fly_cell"]))
        self.headings = np.column_stack((a["pac_dir"], a["fly_dir"]))
        if np.any((self.cells < 0) | (self.cells >= len(self.map.cell_rc))):
            raise ValueError("Replay contains a cell outside the map")
        if np.any((self.headings < 0) | (self.headings >= len(STEP_RC))):
            raise ValueError("Replay contains an invalid heading")
        acc = np.column_stack((a["pac_acc"], a["fly_acc"]))
        if not np.isfinite(acc).all():
            raise ValueError("Replay contains a non-finite movement accumulator")

        rows, self.cols = self.map.grid.shape
        self.board_size = (self.cols * TILE, rows * TILE)
        self.centers = self.map.cell_rc[:, ::-1] + 0.5  # tile units
        self._xy = self.centers[self.cells]
        # The real engine can record pellet speed debt (negative pac_acc).
        # Suppress forward projection when the recorded heading faces a wall.
        can_project = np.column_stack((
            self.map.move_mask["pacman"][self.cells[:, 0], self.headings[:, 0]],
            self.map.move_mask["fly"][self.cells[:, 1:], self.headings[:, 1:]],
        ))
        self._xy += (np.clip(acc.astype(float), 0, 1) * can_project)[..., None] * STEP_XY[self.headings]

        current, following = self.cells[:-1], self.cells[1:]
        same = current == following
        # Non-neighbor jumps include synthetic house release and eaten-fly resets.
        smooth = same | (self.map.nbr[current] == following[..., None]).any(axis=-1)
        wraps = smooth & (np.abs(self.map.cell_rc[following, 1] - self.map.cell_rc[current, 1]) > 1)
        self.events = defaultdict(list)
        self.pellet_eaten_at = np.full(len(self.centers), self.ticks, dtype=int)
        synthetic = self.meta["git_commit"] == "synthetic"
        for tick, kind, fly in zip(a["ev_tick"], kind_names(a["ev_kind"]), a["ev_fly"]):
            if not 0 <= tick < self.ticks:
                raise ValueError(f"Replay event is outside the timeline: {tick}")
            self.events[int(tick)].append((kind, int(fly)))
            if kind in ("DOT", "ENERGIZER"):
                # Runner captures Pac-Man after its move. Synth records before it.
                cell = self.cells[min(tick + int(synthetic), self.ticks - 1), 0]
                self.pellet_eaten_at[cell] = min(self.pellet_eaten_at[cell], tick)
            if tick < self.ticks - 1:
                if kind == "LIFE_LOST":
                    smooth[tick] = False
                if kind == "EATEN" and 0 <= fly < len(self.map.fly_starts):
                    if following[tick, fly + 1] == self.map.fly_starts[fly] and not same[tick, fly + 1]:
                        smooth[tick, fly + 1] = False

        self._delta = self._xy[1:] - self._xy[:-1]
        dx = self._delta[..., 0]
        dx[wraps] -= np.copysign(self.cols, dx[wraps])
        self._delta *= smooth[..., None]

    def tick_index(self, tick: float) -> int:
        if not 0 <= tick < self.ticks:
            raise ValueError(f"Replay tick must be in [0, {self.ticks}): {tick}")
        return int(tick)

    def cell_center(self, cell: int) -> tuple[int, int]:
        x, y = self.centers[cell] * TILE + ORIGIN
        return int(x), int(y)

    def positions(self, tick: float) -> tuple[tuple[int, int], ...]:
        """Screen centers of Pac-Man/source followed by the four flies."""
        index = self.tick_index(tick)
        xy = self._xy[index].copy()
        if index < self.ticks - 1:
            xy += (tick - index) * self._delta[index]
        pixels = np.rint(xy * TILE).astype(int)
        pixels[:, 0] %= self.board_size[0]
        pixels += ORIGIN
        return tuple((int(x), int(y)) for x, y in pixels)
