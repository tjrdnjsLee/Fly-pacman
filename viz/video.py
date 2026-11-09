"""Stream short replay ranges to H.264 without retaining frames in memory."""

from __future__ import annotations

import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol

import imageio_ffmpeg
import numpy as np
from .replay import FPS, SIZE


class FrameRenderer(Protocol):
    def render(self, tick: float) -> np.ndarray: ...


def frame_ticks(ticks: int, tick_rate: float, start: float, duration: float | None) -> tuple[float, ...]:
    """Sample the requested half-open time range at 30 fps, clipped to the replay."""
    if not math.isfinite(tick_rate) or tick_rate <= 0:
        raise ValueError("Tick rate must be positive and finite")
    end = ticks / tick_rate
    if not 0 <= start < end:
        raise ValueError(f"Start must be in [0, {end:.3f}) seconds")
    if duration is not None and (not math.isfinite(duration) or duration <= 0):
        raise ValueError("Duration must be positive and finite")
    length = end - start if duration is None else min(duration, end - start)
    # Rounding removes floating-point noise at integer frame boundaries.
    count = max(1, math.ceil(round(length * FPS, 10)))
    last = math.nextafter(float(ticks), 0.0)
    return tuple(min((start + n / FPS) * tick_rate, last) for n in range(count))


def write_mp4(renderer: FrameRenderer, ticks: tuple[float, ...], path: Path) -> None:
    """Write RGB frames as H.264 at the exact canvas size and playback rate."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as work:
        output = Path(work) / path.name
        writer = imageio_ffmpeg.write_frames(
            str(output), SIZE, fps=FPS, codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
            quality=None, macro_block_size=1,
            input_params=["-filter_threads", "1"],
            output_params=["-preset", "veryfast", "-threads", "1", "-movflags", "+faststart"],
        )
        try:
            writer.send(None)
            for tick in ticks:
                writer.send(renderer.render(tick))
        finally:
            writer.close()
        if not output.stat().st_size:
            raise RuntimeError(f"Encoder produced an empty video: {path}")
        output.replace(path)
