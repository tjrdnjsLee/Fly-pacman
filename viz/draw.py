"""Offscreen canvas, layout, and primitives shared by both replay views."""

from __future__ import annotations

import numpy as np
import pygame

from . import palette as p
from .replay import ORIGIN, SIZE, TILE, Replay

PANEL_X = 934
CONTENT_X = PANEL_X + 36


def rgb_frame(surface: pygame.Surface) -> np.ndarray:
    """Copy a pygame surface into contiguous RGB rows for the encoder."""
    return np.ascontiguousarray(pygame.surfarray.array3d(surface).transpose(1, 0, 2))


def korean_font(size: int) -> pygame.font.Font | None:
    path = pygame.font.match_font(("notosanscjkkr", "nanumgothic", "malgungothic", "applegothic"))
    return pygame.font.Font(path, size) if path else None


def state_color(z: bool) -> tuple[int, int, int]:
    return p.FLY_FRIGHTENED if z else p.FLY_NORMAL


def diamond(surface: pygame.Surface, color: tuple, center: tuple, radius: int) -> None:
    x, y = center
    pygame.draw.polygon(surface, color, [(x, y - radius), (x + radius, y), (x, y + radius), (x - radius, y)])


class Renderer:
    """Draw into a pygame Surface without initializing video or audio."""

    def __init__(self, replay: Replay, label: str, title: str):
        self.replay = replay
        self.board = pygame.Rect(ORIGIN, replay.board_size)
        pygame.font.init()
        self.fonts = {size: pygame.font.Font(None, size) for size in (24, 28, 34, 44, 62)}
        self.base = pygame.Surface(SIZE)
        self.base.fill(p.BACKGROUND)
        pygame.draw.rect(self.base, p.LINE, self.board.inflate(4, 4), border_radius=3)
        self.base.fill(p.BOARD, self.board)
        pygame.draw.rect(self.base, p.PANEL, (PANEL_X, ORIGIN[1], 882, self.board.height), border_radius=20)
        self.text(self.base, label, (ORIGIN[0], 38), 24, p.MUTED)
        self.text(self.base, title, (ORIGIN[0], 66), 62)

    def text(self, surface: pygame.Surface, value: str, xy: tuple, size: int = 28,
             color: tuple = p.TEXT) -> None:
        surface.blit(self.fonts[size].render(value, True, color), xy)

    def rule(self, surface: pygame.Surface, y: int) -> None:
        pygame.draw.line(surface, p.LINE, (CONTENT_X, y), (SIZE[0] - ORIGIN[0] - 36, y))

    def actors_at(self, tick: float) -> tuple[tuple[int, int], ...]:
        return self.replay.positions(tick)

    def wrapped_centers(self, center: tuple[int, int]):
        """Draw the overlapping part of a tunnel sprite on the opposite edge."""
        x, y = center
        yield center
        if x - self.board.left < TILE // 2:
            yield (x + self.board.width, y)
        elif self.board.right - x < TILE // 2:
            yield (x - self.board.width, y)

    def surface_at(self, tick: float) -> pygame.Surface:
        index = self.replay.tick_index(tick)
        surface = self.base.copy()
        self._draw(surface, tick, index, self.actors_at(tick))
        self.text(surface, f"RECORDED REPLAY  /  {self.replay.meta['condition']}  /  "
                  f"seed {self.replay.meta['seed']}  /  game {self.replay.meta['game_no']}  /  tick {index}",
                  (ORIGIN[0], self.board.bottom + 24), 24, p.MUTED)
        return surface

    def render(self, tick: float) -> np.ndarray:
        return rgb_frame(self.surface_at(tick))
