"""Game graphics for a replay (TRD 19.8).

Adapted from pacman-ai's pacman/viz/renderer.py and sprites.py, by
vikranthreddimasu, MIT as stated in the upstream README. See NOTICE.
"""

from __future__ import annotations

import math

import numpy as np
import pygame

from flypac.maps.original import DOOR, HOUSE, WALL

from . import palette as p
from .draw import CONTENT_X, Renderer, korean_font, state_color
from .replay import ORIGIN, STEP_XY, TILE, Replay


def _pacman(surface: pygame.Surface, center: tuple, heading: int, tick: float) -> None:
    """Fill a sector with an oscillating mouth, adapted from draw_pacman."""
    dx, dy = STEP_XY[heading]
    angle = math.atan2(dy, dx)
    mouth = 0.08 + 0.57 * abs(math.sin(tick * 1.2))
    radius = TILE // 2 - 2
    x, y = center
    points = [center]
    for theta in np.linspace(angle + mouth, angle + math.tau - mouth, 34):
        points.append((x + radius * math.cos(theta), y + radius * math.sin(theta)))
    pygame.draw.polygon(surface, p.SOURCE, points)


def _fly(surface: pygame.Surface, center: tuple, heading: int, z: bool) -> None:
    """Team-drawn fly glyph with wings, thorax, abdomen and antennae."""
    x, y = center
    dx, dy = map(int, STEP_XY[heading])

    def point(forward: int, lateral: int) -> tuple[int, int]:
        return x + forward * dx - lateral * dy, y + forward * dy + lateral * dx

    color = state_color(z)
    if z:
        pygame.draw.circle(surface, color, center, TILE // 2 - 1, 1)
    for side in (-1, 1):
        pygame.draw.polygon(surface, p.FLY_WING,
                            [point(3, side * 2), point(-3, side * 11), point(-9, side * 9), point(-5, side * 3)])
        pygame.draw.line(surface, color, point(7, side * 2), point(11, side * 5), 2)
    pygame.draw.circle(surface, color, point(-4, 0), 5)
    pygame.draw.circle(surface, color, center, 5)
    pygame.draw.circle(surface, color, point(5, 0), 4)


def _fruit(surface: pygame.Surface, center: tuple) -> None:
    """Two circles and stems, adapted from draw_fruit."""
    x, y = center
    pygame.draw.line(surface, p.FRUIT_STEM, (x - 5, y + 3), (x + 1, y - 8), 2)
    pygame.draw.line(surface, p.FRUIT_STEM, (x + 5, y + 4), (x + 1, y - 8), 2)
    pygame.draw.circle(surface, p.FRUIT, (x - 5, y + 3), 5)
    pygame.draw.circle(surface, p.FRUIT, (x + 5, y + 4), 5)


class GameRenderer(Renderer):
    """Offscreen game view driven by a recorded replay."""

    def __init__(self, replay: Replay, *, selected_fly: int | None = None):
        super().__init__(replay, "FLY PAC-MAN   /   01", "Game view")
        self.selected_fly = selected_fly  # zero-based, like the replay arrays
        self._maze()
        x = CONTENT_X
        self.text(self.base, "Same data, presented with game graphics.", (x, 184), 34)
        font = korean_font(22)
        if font:
            subtitle = font.render("같은 데이터를 게임 그래픽으로 입힌 것", True, p.MUTED)
            self.base.blit(subtitle, (x, 225))
        self.rule(self.base, 270)
        for label, offset in (("SCORE", 0), ("LIVES", 260), ("POWER LEFT", 485)):
            self.text(self.base, label, (x + offset, 304), 24, p.MUTED)
        self.rule(self.base, 430)
        self.text(self.base, "FLY STATUS", (x, 459), 28, p.MUTED)

    def _maze(self) -> None:
        grid = self.replay.map.grid
        for row, col in np.argwhere((grid == WALL) | (grid == HOUSE) | (grid == DOOR)):
            x, y = ORIGIN[0] + int(col) * TILE, ORIGIN[1] + int(row) * TILE
            rect = pygame.Rect(x, y, TILE, TILE)
            tile = grid[row, col]
            if tile == WALL:
                pygame.draw.rect(self.base, p.WALL, rect)
                pygame.draw.rect(self.base, p.WALL_EDGE, rect, 1)
            elif tile == HOUSE:
                pygame.draw.rect(self.base, p.HOUSE, rect)
            else:
                pygame.draw.line(self.base, p.DOOR, (x, y + TILE // 2), (x + TILE, y + TILE // 2), 3)

    def _draw(self, surface: pygame.Surface, tick: float, index: int, positions: tuple) -> None:
        replay = self.replay
        a = replay.arrays
        surface.set_clip(self.board)
        present = replay.pellet_eaten_at > index
        for cell in np.flatnonzero(replay.map.dots & present):
            pygame.draw.circle(surface, p.PELLET, replay.cell_center(cell), 2)
        if tick % 4 < 3:
            for cell in np.flatnonzero(replay.map.energizers & present):
                pygame.draw.circle(surface, p.POWER, replay.cell_center(cell), 6)
        if a["fruit"][index]:
            _fruit(surface, replay.cell_center(replay.map.fruit_cell))

        for fly, center in enumerate(positions[1:]):
            for wrapped in self.wrapped_centers(center):
                _fly(surface, wrapped, replay.headings[index, fly + 1], a["fly_z"][index, fly])
        if self.selected_fly is not None:
            for wrapped in self.wrapped_centers(positions[self.selected_fly + 1]):
                pygame.draw.circle(surface, p.SELECTED_FLY, wrapped, TILE // 2 - 1, 2)
        for wrapped in self.wrapped_centers(positions[0]):
            _pacman(surface, wrapped, replay.headings[index, 0], tick)

        event_rows = []
        for kind, fly in replay.events.get(index, ()):
            if kind in ("CATCH", "EATEN"):
                color = p.CATCH if kind == "CATCH" else p.EATEN
                for wrapped in self.wrapped_centers(positions[0]):
                    pygame.draw.circle(surface, color, wrapped, TILE // 2, 3)
                event_rows.append((f"{kind}  /  FLY {fly + 1}", color))
        surface.set_clip(None)

        x = CONTENT_X
        for value, offset in ((a["score"][index], 0), (a["lives"][index], 260), (a["power_ticks"][index], 485)):
            self.text(surface, str(value), (x + offset, 337), 62)
        self.text(surface, "ticks", (x + 595, 360), 24, p.MUTED)
        for fly, z in enumerate(a["fly_z"][index]):
            y = 524 + fly * 65
            _fly(surface, (x + 14, y + 13), replay.headings[index, fly + 1], z)
            self.text(surface, f"Fly {fly + 1}", (x + 46, y), 34)
            state = "Frightened" if z else "Normal"
            if a["fly_in_house"][index, fly]:
                state += "  /  In house"
            self.text(surface, state, (x + 204, y), 28, state_color(z))
        for row, (message, color) in enumerate(event_rows):
            self.text(surface, message, (x, 816 + row * 36), 34, color)
