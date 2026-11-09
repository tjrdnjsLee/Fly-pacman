"""Raw replay view of the recorded fields and sensory observations (TRD 19.7)."""

from __future__ import annotations

import math

import numpy as np
import pygame

from flypac.core.directions import EGO_NAMES

from . import palette as p
from .draw import CONTENT_X, Renderer, diamond, state_color
from .replay import STEP_XY, TILE, Replay


class RawRenderer(Renderer):
    """Offscreen map, with logged observations in a separate panel."""

    def __init__(self, replay: Replay, *, c0: float, observations: bool = True):
        if not math.isfinite(c0) or c0 <= 0:
            raise ValueError("c0 must be positive and finite")
        super().__init__(replay, "FLY PAC-MAN   /   02", "Raw replay")
        self.observations = observations
        self.has_pheromone = "phero_field" in replay.arrays
        a = replay.arrays
        fields = ["odor_field"] + (["phero_field"] if self.has_pheromone else [])
        logs = [np.log1p(np.maximum(a[name].astype(np.float32), 0) / c0) for name in fields]
        # A fixed scale for the whole replay keeps different times comparable.
        self.field_strength = [field / (float(field.max()) or 1.0) for field in logs]
        self.sense_max = float(a["sense"].max()) or 1.0
        self.nodes = [replay.cell_center(cell) for cell in range(len(replay.centers))]
        self.graph = pygame.Surface(replay.board_size, pygame.SRCALPHA)
        self._maze()

        x = CONTENT_X
        self.text(self.base, "WORLD STATE", (x, 184), 28, p.MUTED)
        diamond(self.base, p.SOURCE, (x + 9, 236), 8)
        self.text(self.base, "Actual odor source", (x + 28, 225), 28)
        for label, color, offset in (("Odor field", p.ODOR, 284), ("Pheromone field", p.PHEROMONE, 482)):
            if label == "Odor field" or self.has_pheromone:
                pygame.draw.circle(self.base, color, (x + offset, 236), 7)
                self.text(self.base, label, (x + offset + 19, 225), 28)
        self.text(self.base, "Field intensity uses log(1 + c/c0). Dashed edges mark the tunnel.",
                  (x, 269), 24, p.MUTED)
        self.rule(self.base, 309)
        self.text(self.base, "RECORDED FLY OBSERVATIONS", (x, 331), 28, p.MUTED)
        subtitle = "Own state and four directions relative to heading."
        if replay.meta["condition"] == "b1":
            subtitle = "Sensory samples are logged. The B1 controller also uses game state."
        self.text(self.base, subtitle, (x, 367), 24, p.MUTED)
        if observations:
            for direction, label in enumerate(EGO_NAMES):
                self.text(self.base, label.upper(), (x + 174 + direction * 153, 409), 24, p.MUTED)

    def _maze(self) -> None:
        local = [(x - self.board.left, y - self.board.top) for x, y in self.nodes]
        for cell, neighbors in enumerate(self.replay.map.nbr):
            x, y = local[cell]
            for neighbor in neighbors[neighbors > cell]:
                nx, ny = local[neighbor]
                if abs(nx - x) <= TILE:
                    pygame.draw.line(self.graph, (*p.LINE, 160), (x, y), (nx, ny), 2)
                else:
                    for offset in (0, 6):
                        pygame.draw.line(self.graph, p.TUNNEL, (offset, y), (offset + 3, y), 2)
                        end = self.board.width - 1 - offset
                        pygame.draw.line(self.graph, p.TUNNEL, (end - 3, ny), (end, ny), 2)
            pygame.draw.circle(self.graph, p.TUNNEL, (x, y), 2)

    def _arrow(self, surface: pygame.Surface, center: tuple, heading: int, z: bool, decided: bool) -> None:
        x, y = center
        dx, dy = map(int, STEP_XY[heading])
        points = [(x + dx * 11, y + dy * 11),
                  (x - dx * 8 - dy * 7, y - dy * 8 + dx * 7),
                  (x - dx * 3, y - dy * 3),
                  (x - dx * 8 + dy * 7, y - dy * 8 - dx * 7)]
        pygame.draw.polygon(surface, state_color(z), points, 2 if z else 0)
        if decided:
            pygame.draw.circle(surface, p.DECISION, center, TILE // 2 - 1, 2)

    def _draw(self, surface: pygame.Surface, tick: float, index: int, positions: tuple) -> None:
        a = self.replay.arrays
        actions = a["action"][index]
        confirmed = a["committed"][index] if "committed" in a else actions >= 0
        moved = a["moved"][index] if "moved" in a else None
        highlight = self.observations and (tick - index < 0.7)

        surface.set_clip(self.board)
        field = pygame.Surface(self.replay.board_size, pygame.SRCALPHA)
        for strength, color, radius in zip(self.field_strength, (p.ODOR, p.PHEROMONE), (18, 11)):
            field.fill((0, 0, 0, 0))
            for cell in np.flatnonzero(strength[index] > 0):
                x, y = self.nodes[cell]
                local = (x - self.board.left, y - self.board.top)
                alpha = int(156 * strength[index, cell])
                pygame.draw.circle(field, (*color, alpha), local, radius)
            surface.blit(field, self.board.topleft)
        surface.blit(self.graph, self.board.topleft)

        for wrapped in self.wrapped_centers(positions[0]):
            diamond(surface, p.SOURCE, wrapped, 10)
        for fly, center in enumerate(positions[1:]):
            z = a["fly_z"][index, fly]
            for wrapped in self.wrapped_centers(center):
                self._arrow(surface, wrapped, self.replay.headings[index, fly + 1], z,
                            highlight and confirmed[fly])
                self.text(surface, str(fly + 1), (wrapped[0] + 7, wrapped[1] - 16), 24)
        surface.set_clip(None)

        for fly, z in enumerate(a["fly_z"][index]):
            self._observations(surface, index, fly, z, confirmed[fly], None if moved is None else moved[fly])

    def _observations(self, surface: pygame.Surface, index: int, fly: int, z: bool,
                      confirmed: bool, moved: bool | None) -> None:
        a = self.replay.arrays
        x, y = CONTENT_X, 451 + fly * 126
        self.rule(surface, y - 7)
        self.text(surface, f"Fly {fly + 1}   z={int(z)}", (x, y + 7), 28, state_color(z))
        if moved is not None:
            self.text(surface, "Moved" if moved else "Stationary", (x, y + 38), 24, p.MUTED)
        if not self.observations:
            return
        if confirmed:
            label = "Committed" if "committed" in a else "Action logged"
            self.text(surface, label, (x, y + 68), 24, p.DECISION)
        sense = a["sense"][index, fly]
        pi = a["pi"][index, fly]
        show_pi = confirmed and pi.any()
        for direction in range(len(EGO_NAMES)):
            column = x + 174 + direction * 153
            for value, color, offset in ((sense[direction], p.ODOR, 12),
                                         (sense[direction + len(EGO_NAMES)], p.PHEROMONE, 36)):
                if color == p.PHEROMONE and not self.has_pheromone:
                    continue
                self._bar(surface, column, y + offset, float(value) / self.sense_max, color)
                self.text(surface, f"{value:.2f}", (column + 96, y + offset - 5), 24, p.MUTED)
            if show_pi:
                prob = float(pi[direction])
                color = p.DECISION if a["action"][index, fly] == direction else p.TEXT
                self._bar(surface, column, y + 74, prob, color)
                self.text(surface, f"p {prob:.2f}", (column, y + 53), 24, color)

    @staticmethod
    def _bar(surface: pygame.Surface, x: int, y: int, fraction: float, color: tuple) -> None:
        pygame.draw.rect(surface, p.LINE, (x, y, 92, 7), border_radius=3)
        width = round(92 * min(max(fraction, 0.0), 1.0))
        if width:
            pygame.draw.rect(surface, color, (x, y, width, 7), border_radius=3)
