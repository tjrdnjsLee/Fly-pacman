"""Frontal anatomy and model activity, sampled from a recorded format-4 replay."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pygame

from flypac.core.config import load_config
from flypac.core.directions import EGO_NAMES
from flypac.replay.schema import N_FLIES

from . import palette as p
from .anatomy import Anatomy
from .clouds import ActivityRaster, PointClouds, Projection, context_image
from .draw import korean_font, rgb_frame
from .game import GameRenderer
from .replay import SIZE, Replay

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
    pi: np.ndarray | None
    pi_tick: int | None


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
    """Cache each view's static canvas, projection and raster addresses."""

    def __init__(self, size: tuple[int, int], compact: bool, view: str, anatomy: Anatomy,
                 clouds: PointClouds, style):
        self.surface = pygame.Surface(size)
        self.surface.fill(p.BACKGROUND)
        margin = 25 if compact else 52
        self.plot = pygame.Rect(margin, 102, size[0] - 2 * margin, 692)
        self.metrics = (margin + 18, 826)
        self.notes = (520 if compact else 1130, 826)
        pygame.draw.rect(self.surface, p.BRAIN_VOID, self.plot)
        pygame.draw.rect(self.surface, p.PANEL, (margin, 807, size[0] - 2 * margin, 217), border_radius=12)
        self.bar_width = 93 if compact else 155

        bounds = clouds.context_bounds if view == "a" else clouds.detail_bounds
        projection = Projection(bounds, self.plot.width, self.plot.height, padding=27)
        self.raster = ActivityRaster(clouds, projection, self.plot.width, self.plot.height, style)
        self._last_tick = None
        self._last_surface = None

        self.dan = []
        self.lh = []
        for region in anatomy.regions:
            for ring, hole in region.rings:
                points = projection.pixels(ring) + self.plot.topleft
                if region.model_term:
                    self.lh.append(points)
                elif region.name in anatomy.dan_regions:
                    self.dan.append((points, hole))

        self.mini = None
        if view == "b":
            w, h = (174, 166) if compact else (220, 210)
            self.mini_rect = pygame.Rect(self.plot.right - w - 16, self.plot.top + 18, w, h)
            mini_projection = Projection(clouds.context_bounds, w, h, padding=12)
            image = context_image(clouds.context, mini_projection, w, h, style)
            self.mini = pygame.surfarray.make_surface(image.swapaxes(0, 1))
            lo, hi = mini_projection.pixels(clouds.detail_bounds)
            zoom = pygame.Rect(lo[0], hi[1], hi[0] - lo[0] + 1, lo[1] - hi[1] + 1)
            pygame.draw.rect(self.mini, p.KC_ACTIVE, zoom, 1)
            pygame.draw.rect(self.mini, p.LINE, self.mini.get_rect(), 1)
            self.mini_lh = [mini_projection.pixels(ring) + self.mini_rect.topleft
                            for region in anatomy.regions if region.model_term for ring, _ in region.rings]

    def cloud_surface(self, index: int, activity: dict[str, np.ndarray] | None) -> pygame.Surface:
        if self._last_tick != index:
            image = self.raster.background if activity is None else self.raster.render(activity)
            self._last_surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
            self._last_tick = index
        return self._last_surface


class BrainRenderer:
    """Display recorded activity for one fly with no engine or controller calls."""

    def __init__(self, replay: Replay, anatomy: Anatomy, *, fly: int = 1, tick_rate: float, decay_s: float,
                 view: str = "b"):
        if not 1 <= fly <= N_FLIES:
            raise ValueError(f"Fly number must be between 1 and {N_FLIES}")
        if not all(math.isfinite(v) and v > 0 for v in (tick_rate, decay_s)):
            raise ValueError("Tick rate and KC display decay must be positive and finite")
        if view not in ("a", "b"):
            raise ValueError("Brain view must be a or b")
        self.replay, self.anatomy, self.fly = replay, anatomy, fly
        self.view = view
        self.clouds = PointClouds(anatomy)
        self.style = load_config(Path(__file__).with_name("settings.yaml")).brain_cloud
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
            self._pi = a["pi"][:, self.fly_index]
            ticks = np.arange(replay.ticks)
            deciding = self._on & (a["action"][:, self.fly_index] >= 0)
            boundary = ~self._on | np.r_[True, np.diff(a["lives"]) != 0]
            latest = np.maximum.accumulate(np.where(deciding, ticks, -1))
            life_start = np.maximum.accumulate(np.where(boundary, ticks, -1))
            self._pi_tick = np.where(latest >= life_start, latest, -1)
            if not all(np.isfinite(field[self._on]).all() for field in (self._pn, self._mbon, self._dan, self._pi)):
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
        self.fonts = {size: pygame.font.Font(None, size) for size in (20, 22, 24, 28, 46)}
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
        pi_tick = int(self._pi_tick[index])
        return BrainFrame(pn, glow, mbon, bundles, float(self._dan[index]) if committed else 0.0,
                          action if committed and 0 <= action < len(EGO_NAMES) else None,
                          self.anatomy.instinct_gain * float(pn[:len(EGO_NAMES)].max()),
                          self._pi[pi_tick] if pi_tick >= 0 else None, pi_tick if pi_tick >= 0 else None)

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
            layout = _Layout(size, compact, self.view, self.anatomy, self.clouds, self.style)
            x = layout.plot.left
            view_label = "A  /  CONTEXT" if self.view == "a" else "B  /  MUSHROOM BODY DETAIL"
            self._text(layout.surface, f"FLY PAC-MAN   /   {view_label}", (x, 15), 22, p.MUTED)
            self._text(layout.surface, f"Fly {self.fly}  /  Brain activity", (x, 44), 46)
            if self.learning_off:
                self._caption(layout.surface, "학습 꺼짐", "Learning off", (x + 385, 57))
            self._notes(layout.surface, *layout.notes)
            notes_x, _ = layout.notes
            missing = len(self.anatomy.missing)
            self._caption(layout.surface, f"Janelia FlyEM · DAN {missing}개는 구획 내 시냅스 전 위치 없음",
                          f"Janelia FlyEM · {missing} DAN lack presynaptic sites in mapped lobes.", (notes_x, 1041))
            self._layouts[compact] = layout
        return self._layouts[compact]

    def _notes(self, surface, x, y):
        self._caption(surface, "hemibrain: 뇌의 절반 · 정면 투영", "FRONTAL HEMIBRAIN   /   HALF BRAIN", (x, y))
        rows = (
            ("회색 = 뇌의 형태만 (계산하지 않음)", "Gray dots show brain shape (not computed)."),
            (f"활동 점 = 시뮬레이션한 버섯체 뉴런 {self.clouds.neuron_count:,}개",
             f"Light shows activity of {self.clouds.neuron_count:,} model mushroom-body neurons."),
            ("시뮬레이션 표시 · 실제 측정 아님", "Simulated activity shown at anatomical positions."),
            ("모델은 DAN 신호 하나를 모든 구획에 보낸다", "The model sends one DAN signal to every compartment."),
            ("실제 PAM·PPL1 입력은 구획별로 다름", "Biological PAM and PPL1 inputs differ by compartment."),
            ("MBON 밝기 = 원래 활동 · 결정은 기준선과의 차이",
             "MBON brightness is raw. Decisions use change from baseline."),
            ("외측뿔 = 모델 항, 뉴런 아님", "Lateral horn represents a model term, without neuron dots."),
            (f"KC 잔광 {self.decay_s:.2f}초 · 표시용", f"KC afterglow {self.decay_s:.2f} s · display only"),
        )
        for row, (ko, en) in enumerate(rows):
            self._caption(surface, ko, en, (x, y + 23 + row * 21))

    def _cloud_activity(self, frame: BrainFrame) -> dict[str, np.ndarray]:
        style = self.style
        channel = self.anatomy.pn_channels
        pn = np.clip(frame.pn[channel] / self._pn_scale[channel], 0, 1)
        kc = np.clip(frame.kc, 0, 1)
        mbon = np.clip(frame.mbon / self._mbon_scale, 0, 1)
        kc_light = kc[:, None] * (np.asarray(p.KC_ACTIVE) * style.kc_gain)
        return {
            "pn_calyx": pn[:, None] * np.asarray(PN_COLORS)[channel] * style.pn_gain,
            "kc_calyx": kc_light,
            "kc_lobe": kc_light,
            "mbon_lobe": mbon[:, None] * (np.asarray(p.MBON_ACTIVITY) * style.mbon_gain),
        }

    def _draw_regions(self, surface, layout, frame):
        surface.set_clip(layout.plot)
        if frame is not None and frame.dan:
            color = p.DAN_POSITIVE if frame.dan > 0 else p.DAN_NEGATIVE
            for points, _ in layout.dan:
                pygame.draw.lines(surface, color, True, points, 2)
        instinct = frame.instinct / self._instinct_scale if frame is not None else 0
        outline = _tint(p.INSTINCT, instinct)
        for points in layout.lh:
            _dashed(surface, outline, points, 2)
        surface.set_clip(None)
        if layout.mini is not None:
            surface.blit(layout.mini, layout.mini_rect)
            for points in layout.mini_lh:
                _dashed(surface, outline, points, 1)
            self._text(surface, "HEMIBRAIN  /  VIEWPORT",
                       (layout.mini_rect.left + 4, layout.mini_rect.bottom + 6), 20, p.MUTED)

    def _draw_metrics(self, surface, layout, frame):
        x, y = layout.metrics
        self._text(surface, "DECISION PROBABILITY", (x, y), 24, p.TEXT)
        tick_label = f"tick {frame.pi_tick}" if frame.pi_tick is not None else "awaiting decision"
        self._text(surface, tick_label, (x + 228, y + 1), 22, p.MUTED)
        n = len(EGO_NAMES)
        width = layout.bar_width
        for index, label in enumerate(ACTION_LABELS):
            left = x + index * (width + 11)
            value = float(frame.pi[index]) if frame.pi is not None else 0
            selected = frame.action == index
            color = p.MBON_ACTIVITY if value > 0 else p.PROBABILITY_INACTIVE
            if selected:
                color = p.DECISION
            self._text(surface, label, (left, y + 33), 20, color)
            rect = pygame.Rect(left, y + 54, width, 9)
            pygame.draw.rect(surface, p.LINE, rect, 1, border_radius=3)
            filled = min(rect.width, max(0, round(rect.width * value)))
            if filled:
                pygame.draw.rect(surface, color, (left, rect.top, filled, rect.height), border_radius=3)
            if selected:
                pygame.draw.rect(surface, p.DECISION, rect.inflate(6, 6), 1, border_radius=3)
            label_value = f"{value:.1%}" if frame.pi is not None else "—"
            self._text(surface, label_value, (left, y + 70), 20, color)

        raw = "  ".join(f"{value:.1f}" for value in frame.bundles)
        self._text(surface, "MBON raw sums    " + raw, (x, y + 96), 20, p.MBON_ACTIVITY)
        odor = "   ".join(f"{v:.2f}" for v in frame.pn[:n])
        pheromone = "   ".join(f"{v:.2f}" for v in frame.pn[n + 1:])
        self._text(surface, "PN odor    " + odor, (x, y + 117), 20, p.ODOR)
        self._text(surface, "PN pheromone    " + pheromone, (x, y + 138), 20, p.PHEROMONE)
        self._text(surface, f"PN state  {frame.pn[n]:.0f}", (x, y + 159), 20, p.PN_STATE)
        dan_color = p.DAN_POSITIVE if frame.dan > 0 else p.DAN_NEGATIVE if frame.dan < 0 else p.MUTED
        self._text(surface, f"DAN  {frame.dan:+.3f}", (x + 123, y + 159), 20, dan_color)
        self._text(surface, f"Instinct  {frame.instinct:.2f}", (x + 260, y + 159), 20, p.INSTINCT)
        if frame.action is not None:
            self._text(surface, f"CONFIRMED  /  {ACTION_LABELS[frame.action]}", (x, y + 181), 20, p.DECISION)

    def surface_at(self, tick: float, *, compact: bool = False) -> pygame.Surface:
        index = self.replay.tick_index(tick)
        layout = self._layout(compact)
        surface = layout.surface.copy()
        frame = self.activity_at(index)
        activity = self._cloud_activity(frame) if frame is not None else None
        surface.blit(layout.cloud_surface(index, activity), layout.plot.topleft)
        self._draw_regions(surface, layout, frame)
        if frame is not None:
            self._draw_metrics(surface, layout, frame)
        status = self.status_at(index)
        x = layout.plot.left
        self._text(surface, f"RECORDED REPLAY   /   {self.replay.meta['condition']}   /   tick {index}",
                   (x, 1041), 24, p.MUTED)
        if status:
            self._caption(surface, *status, (x + 20, layout.plot.top + 20))
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
