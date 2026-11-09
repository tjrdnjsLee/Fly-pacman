"""Frontal anatomy and model activity, sampled from a recorded format-4 replay."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pygame

from flypac.core.directions import EGO_NAMES
from flypac.replay.schema import N_FLIES

from . import palette as p
from .anatomy import Anatomy
from .draw import korean_font, rgb_frame
from .game import GameRenderer
from .replay import ORIGIN, SIZE, Replay

PAIR_SPLIT = 858
ACTION_LABELS = ("LEFT", "FORWARD", "RIGHT", "REVERSE")
PN_COLORS = (p.ODOR,) * len(EGO_NAMES) + (p.PN_STATE,) + (p.PHEROMONE,) * len(EGO_NAMES)


@dataclass(frozen=True)
class BrainFrame:
    pn: np.ndarray
    kc: np.ndarray
    mbon: np.ndarray
    bundles: np.ndarray
    dan: float
    action: int | None
    instinct: float


def _tint(color: tuple, strength: float, base: tuple = p.LINE) -> pygame.Color:
    return pygame.Color(*base).lerp(pygame.Color(*color), min(1.0, max(0.0, strength)))


def _dashed(surface: pygame.Surface, color: tuple, points: np.ndarray, width: int) -> None:
    """Draw a closed dashed outline with phase carried across polygon edges."""
    dash, gap = 7, 5
    period = dash + gap
    distance = 0.0
    for start, end in zip(points, np.roll(points, -1, axis=0)):
        vector = end - start
        length = math.hypot(*vector)
        along = 0.0
        while along < length:
            phase = (distance + along) % period
            if phase < dash:
                stop = min(length, along + dash - phase)
                pygame.draw.line(surface, color, start + vector * (along / length),
                                 start + vector * (stop / length), width)
                along = stop
            else:
                along = min(length, along + period - phase)
            # Roundoff at a dash boundary must still advance the cursor.
            along += 1e-7
        distance += length


class _Layout:
    """Cached static surfaces and projected points for one canvas size."""

    def __init__(self, size: tuple[int, int], compact: bool, anatomy: Anatomy):
        self.surface = pygame.Surface(size)
        self.surface.fill(p.BACKGROUND)
        if compact:
            self.plot = pygame.Rect(26, 148, size[0] - 54, 562)
            self.metrics, self.notes = (50, 746), (510, 746)
            cards = (pygame.Rect(26, 728, size[0] - 54, 298),)
        else:
            self.plot = pygame.Rect(ORIGIN[0], 148, 1110, 866)
            self.metrics, self.notes = (1272, 182), (1272, 553)
            cards = (pygame.Rect(1246, 148, 570, 866),)
        pygame.draw.rect(self.surface, p.BOARD, self.plot, border_radius=18)
        for card in cards:
            pygame.draw.rect(self.surface, p.PANEL, card, border_radius=18)

        lo, hi = anatomy.bounds
        box = self.plot.inflate(-104, -96)
        scale = min(box.width / (hi[0] - lo[0]), box.height / (hi[1] - lo[1]))
        origin = np.asarray(box.center) - (lo + hi) * 0.5 * (scale, -scale)

        def project(xy):
            return np.rint(xy * (scale, -scale) + origin).astype(int)

        self.neurons = {}
        self.valid = {}
        for kind, xy in anatomy.coordinates.items():
            valid = np.isfinite(xy).all(axis=1)
            self.valid[kind] = valid
            self.neurons[kind] = np.zeros_like(xy, dtype=int)
            self.neurons[kind][valid] = project(xy[valid])

        self.lh = []
        self.dan = []
        self.labels = {}
        for region in anatomy.regions:
            for ring, hole in region.rings:
                points = project(ring)
                if region.model_term:
                    self.lh.append(points)
                    _dashed(self.surface, p.LINE, points, 2)
                else:
                    pygame.draw.lines(self.surface, p.LINE, True, points, 2)
                if region.name in anatomy.dan_regions:
                    self.dan.append((points, hole))
                if region.name in ("CA(R)", "PED(R)", "LH(R)") and not hole:
                    self.labels[region.name] = (int(points[:, 0].min()), int(points[:, 1].max()))
        for kind in ("KC", "PN", "DAN", "MBON"):
            radius = 1 if kind in ("KC", "DAN") else 2
            for point in self.neurons[kind][self.valid[kind]]:
                pygame.draw.circle(self.surface, p.LINE, point, radius)


class BrainRenderer:
    """Display recorded activity for one fly with no engine or controller calls."""

    def __init__(self, replay: Replay, anatomy: Anatomy, *, fly: int = 1, tick_rate: float, decay_s: float):
        if not 1 <= fly <= N_FLIES:
            raise ValueError(f"Fly number must be between 1 and {N_FLIES}")
        if not all(math.isfinite(v) and v > 0 for v in (tick_rate, decay_s)):
            raise ValueError("Tick rate and KC display decay must be positive and finite")
        self.replay, self.anatomy, self.fly = replay, anatomy, fly
        self.fly_index = fly - 1
        self.decay_s = decay_s
        self._decay_per_tick = 1 / (decay_s * tick_rate)
        a = replay.arrays
        self.has_activity = "brain_on" in a and "kc_active" in a
        if self.has_activity:
            self._on = a["brain_on"][:, self.fly_index]
            self._kc = a["kc_active"][:, self.fly_index]
            self._pn = a["pn_input"][:, self.fly_index]
            self._mbon = a["mbon_out"][:, self.fly_index]
            self._dan = a["dan"][:, self.fly_index]
            if not all(np.isfinite(field[self._on]).all() for field in (self._pn, self._mbon, self._dan)):
                raise ValueError("Recorded brain activity must be finite")
            if self._mbon.shape[1] != len(anatomy.mbon_bundles):
                raise ValueError("Replay MBON width does not match the anatomy circuit")
            if np.any((self._kc[self._on] < -1) | (self._kc[self._on] >= len(anatomy.coordinates["KC"]))):
                raise ValueError("Replay KC indices do not match the anatomy circuit")
            self._pn_scale = np.maximum(self._pn[self._on].max(axis=0, initial=0).astype(float), 1)
            self._mbon_scale = float(self._mbon[self._on].max(initial=0)) or 1.0
            n_odor = len(EGO_NAMES)
            self._instinct_scale = float(self._pn[self._on, :n_odor].max(initial=0)) * anatomy.instinct_gain or 1.0

        pygame.font.init()
        self.fonts = {size: pygame.font.Font(None, size) for size in (22, 24, 28, 34, 58)}
        self.ko = korean_font(18)
        self._layouts = {}
        self.learning_off = replay.meta["condition"].lower() == "b0"

    @lru_cache(maxsize=2)
    def activity_at(self, index: int) -> BrainFrame | None:
        """Return exactly one logged tick, with display-only KC afterglow."""
        self.replay.tick_index(index)
        if not self.has_activity or not self._on[index]:
            return None
        a, f = self.replay.arrays, self.fly_index
        glow = np.zeros(len(self.anatomy.coordinates["KC"]))
        first = max(0, index - math.ceil(4 / self._decay_per_tick))
        for t in range(index, first - 1, -1):
            if not self._on[t] or a["lives"][t] != a["lives"][index]:
                break
            kc = self._kc[t]
            glow[kc[kc >= 0]] += math.exp(-(index - t) * self._decay_per_tick)
        pn = self._pn[index].astype(float)
        mbon = self._mbon[index].astype(float)
        bundles = np.bincount(self.anatomy.mbon_bundles, weights=mbon, minlength=len(EGO_NAMES))
        committed = a["committed"][index, f]
        action = int(a["action"][index, f])
        return BrainFrame(pn, glow, mbon, bundles, float(self._dan[index]) if committed else 0.0,
                          action if committed and 0 <= action < len(EGO_NAMES) else None,
                          self.anatomy.instinct_gain * float(pn[:len(EGO_NAMES)].max()))

    def _text(self, surface, value, xy, size=24, color=p.TEXT):
        surface.blit(self.fonts[size].render(value, True, color), xy)

    def _caption(self, surface, korean, english, xy):
        font = self.ko or self.fonts[22]
        surface.blit(font.render(korean if self.ko else english, True, p.MUTED), xy)

    def status_at(self, index: int) -> tuple[str, str] | None:
        """Differentiate a baseline with no brain from a brain that did not run."""
        condition = self.replay.meta["condition"].lower()
        if self.has_activity and self._on[index]:
            return None
        if self.has_activity:
            return "이 틱에는 뇌 계산 없음", "Brain not evaluated on this tick"
        if condition == "b1":
            return "뇌 없음(원작 유령)", "No brain (original ghost)"
        if "kc_active" in self.replay.arrays:
            return "뇌 표시에 형식 4 로그 필요", "Brain display requires format 4"
        return "기록된 뇌 활동 없음", "No recorded brain activity"

    def _layout(self, compact: bool) -> _Layout:
        if compact not in self._layouts:
            size = (SIZE[0] - PAIR_SPLIT, SIZE[1]) if compact else SIZE
            layout = _Layout(size, compact, self.anatomy)
            x = layout.plot.left
            self._text(layout.surface, "FLY PAC-MAN   /   03", (x, 38), 24, p.MUTED)
            self._text(layout.surface, f"Fly {self.fly}  /  Brain activity", (x, 65), 58)
            self._text(layout.surface, "FRONTAL PROJECTION   /   FACING THE FLY",
                       (x + 23, layout.plot.top + 20), 22, p.MUTED)
            if self.learning_off:
                self._caption(layout.surface, "학습 꺼짐", "Learning off", (x + 427, 91))
            self._notes(layout.surface, *layout.notes)
            self._layouts[compact] = layout
        return self._layouts[compact]

    def _notes(self, surface, x, y):
        self._text(surface, "HOW TO READ THIS VIEW", (x, y), 22, p.MUTED)
        rows = (
            ("해부 좌표에 표시한 모델 활동 · 실제 측정 아님", "Simulated activity shown on anatomical coordinates."),
            ("PN 채널은 프로젝트가 배정한 대응", "PN channels are assigned by the model."),
            ("밝기는 원래 활동, 결정에는 처음 상태와의 차이가 쓰인다", "Raw brightness. Decisions use change from baseline."),
            ("모델은 DAN 신호 하나를 모든 구획에 보낸다", "The model sends one DAN signal to every compartment."),
            ("실제 뇌는 구획마다 PAM·PPL1 입력이 다르다", "Biological PAM and PPL1 inputs differ by compartment."),
            ("외측뿔은 모델 항, 뉴런 아님", "Lateral horn represents a model term, without neuron dots."),
        )
        for row, (ko, en) in enumerate(rows):
            self._caption(surface, ko, en, (x, y + 29 + row * 26))
        self._text(surface, f"Display-only KC afterglow  {self.decay_s:.2f} s", (x, y + 190), 22, p.MUTED)
        self._text(surface, f"Anatomy  /  {self.anatomy.dataset}  /  Janelia FlyEM", (x, y + 215), 22, p.MUTED)
        if self.anatomy.missing:
            count = len(self.anatomy.missing)
            self._caption(surface, f"DAN {count}개 좌표 없음 · 구획 내 시냅스 전 위치 없음",
                          f"{count} DAN omitted. No presynaptic sites in the compartments.", (x, y + 240))

    def _draw_activity(self, surface, layout, frame):
        if frame.dan:
            color = p.DAN_POSITIVE if frame.dan > 0 else p.DAN_NEGATIVE
            glow = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
            for points, hole in layout.dan:
                pygame.draw.polygon(glow, (0, 0, 0, 0) if hole else (*color, 61), points)
            surface.blit(glow, (0, 0))
            for points, _ in layout.dan:
                pygame.draw.lines(surface, color, True, points, 2)
            for point in layout.neurons["DAN"][layout.valid["DAN"]]:
                pygame.draw.circle(surface, color, point, 2)

        for index in np.flatnonzero((frame.kc > 0.05) & layout.valid["KC"]):
            color = _tint(p.KC_ACTIVE, frame.kc[index])
            pygame.draw.circle(surface, color, layout.neurons["KC"][index], 3)
        for index in np.flatnonzero(layout.valid["PN"]):
            channel = self.anatomy.pn_channels[index]
            strength = frame.pn[channel] / self._pn_scale[channel]
            if strength > 0:
                pygame.draw.circle(surface, _tint(PN_COLORS[channel], strength), layout.neurons["PN"][index], 3)
        for index in np.flatnonzero(layout.valid["MBON"]):
            strength = frame.mbon[index] / self._mbon_scale
            point = layout.neurons["MBON"][index]
            pygame.draw.circle(surface, _tint(p.MBON_ACTIVITY, strength), point, 4)
            if frame.action == self.anatomy.mbon_bundles[index]:
                pygame.draw.circle(surface, p.DECISION, point, 6, 1)
        if frame.instinct > 0:
            color = _tint(p.INSTINCT, frame.instinct / self._instinct_scale)
            for points in layout.lh:
                _dashed(surface, color, points, 3)

    def _draw_metrics(self, surface, xy, frame):
        x, y = xy
        self._text(surface, "RECORDED MODEL CHANNELS", (x, y), 22, p.MUTED)
        n = len(EGO_NAMES)
        odor = "   ".join(f"{v:.2f}" for v in frame.pn[:n])
        pheromone = "   ".join(f"{v:.2f}" for v in frame.pn[n + 1:])
        self._text(surface, "PN odor  " + odor, (x, y + 31), 24, p.ODOR)
        self._text(surface, "PN pheromone  " + pheromone, (x, y + 58), 24, p.PHEROMONE)
        self._text(surface, f"PN state  {frame.pn[n]:.0f}", (x, y + 85), 24, p.PN_STATE)
        self._text(surface, "MBON raw activity sums", (x, y + 116), 22, p.MUTED)
        for index, label in enumerate(ACTION_LABELS):
            left = x + index * 106
            color = p.DECISION if frame.action == index else p.MBON_ACTIVITY
            pygame.draw.rect(surface, p.LINE, (left, y + 143, 97, 50), 1, border_radius=6)
            self._text(surface, label, (left + 7, y + 148), 22, color)
            self._text(surface, f"{frame.bundles[index]:.2f}", (left + 7, y + 171), 22, color)
        dan_color = p.DAN_POSITIVE if frame.dan > 0 else p.DAN_NEGATIVE if frame.dan < 0 else p.MUTED
        self._text(surface, f"DAN  {frame.dan:+.3f}", (x, y + 212), 24, dan_color)
        self._text(surface, f"Instinct  {frame.instinct:.2f}", (x + 177, y + 212), 24, p.INSTINCT)
        if frame.action is not None:
            self._text(surface, f"CONFIRMED  /  {ACTION_LABELS[frame.action]}", (x, y + 242), 22, p.DECISION)

    def surface_at(self, tick: float, *, compact: bool = False) -> pygame.Surface:
        index = self.replay.tick_index(tick)
        layout = self._layout(compact)
        surface = layout.surface.copy()
        frame = self.activity_at(index)
        if frame is not None:
            self._draw_activity(surface, layout, frame)
            self._draw_metrics(surface, layout.metrics, frame)
        for roi, label in (("CA(R)", "CALYX"), ("PED(R)", "PEDUNCLE"), ("LH(R)", "LATERAL HORN")):
            if roi in layout.labels:
                px, py = layout.labels[roi]
                self._text(surface, label, (px, py + 8), 22, p.MUTED)
        status = self.status_at(index)
        x = layout.plot.left
        self._text(surface, f"RECORDED REPLAY   /   {self.replay.meta['condition']}   /   tick {index}",
                   (x, 1040), 24, p.MUTED)
        if status:
            self._caption(surface, *status, (x + 24, layout.plot.top + 55))
        return surface

    def render(self, tick: float) -> np.ndarray:
        return rgb_frame(self.surface_at(tick))


class BrainPairRenderer:
    """One recorded tick in the game and anatomical views, on the same canvas."""

    def __init__(self, brain: BrainRenderer):
        self.brain = brain
        self.game = GameRenderer(brain.replay, selected_fly=brain.fly_index)
        self._crop = pygame.Rect(self.game.board.left - 4, 0, self.game.board.width + 8, SIZE[1])
        self._left = (PAIR_SPLIT - self._crop.width) // 2

    def render(self, tick: float) -> np.ndarray:
        surface = pygame.Surface(SIZE)
        surface.fill(p.BACKGROUND)
        surface.blit(self.game.surface_at(tick), (self._left, 0), self._crop)
        self.game.text(surface, "Same data, presented with game graphics.", (self._left + 4, 121), 24, p.MUTED)
        self.game.text(surface, f"Fly {self.brain.fly} selected", (self._left + 590, 121), 24, p.SELECTED_FLY)
        surface.blit(self.brain.surface_at(tick, compact=True), (PAIR_SPLIT, 0))
        return rgb_frame(surface)
