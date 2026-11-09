"""Validate and rasterize the supplied hemibrain point clouds."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from . import palette as p
from .anatomy import DEFAULT_ANATOMY, Anatomy

GROUPS = (("pn_calyx", "PN"), ("kc_calyx", "KC"), ("kc_lobe", "KC"), ("mbon_lobe", "MBON"))


@dataclass(frozen=True)
class Cloud:
    xy: np.ndarray
    owners: np.ndarray


class PointClouds:
    """Read the sampled positions without resampling or mirroring either cloud."""

    def __init__(self, anatomy: Anatomy, directory: Path = DEFAULT_ANATOMY.parent):
        with np.load(directory / "brain_context_v1.npz", allow_pickle=False) as file:
            self.context = file["xy_um"]
        context_meta = json.loads((directory / "brain_context_v1.meta.json").read_text(encoding="utf-8"))
        synapse_meta = json.loads((directory / "mb_synapse_clouds_v1.meta.json").read_text(encoding="utf-8"))
        if not all(meta["dataset"] == anatomy.dataset and meta["projection"]["view"] == "frontal"
                   for meta in (context_meta, synapse_meta)):
            raise ValueError("The point clouds must use the frontal projection of the anatomy dataset")
        if synapse_meta["sources"]["circuit_raw"] != anatomy.source_hash:
            raise ValueError("Synapse point clouds do not match the configured circuit")
        if not _valid_points(self.context) or len(self.context) != context_meta["n_with_soma"]:
            raise ValueError("Invalid context coordinates")

        self.groups = {}
        with np.load(directory / "mb_synapse_clouds_v1.npz", allow_pickle=False) as file:
            for group, kind in GROUPS:
                xy, offsets, body = (file[f"{group}_{suffix}"] for suffix in ("xy_um", "offsets", "body"))
                cap = synapse_meta["caps_per_neuron"][group]
                counts = np.diff(offsets)
                if (not np.array_equal(body, anatomy.body_ids[kind]) or not _valid_points(xy)
                        or offsets.shape != (len(body) + 1,) or offsets[0] != 0 or offsets[-1] != len(xy)
                        or np.any((counts < 0) | (counts > cap))):
                    raise ValueError(f"Invalid synapse coordinates or neuron order in {group}")
                self.groups[group] = Cloud(xy, np.repeat(np.arange(len(body), dtype=np.intp), counts))

        self.neuron_count = sum(len(anatomy.body_ids[kind]) for kind in ("PN", "KC", "MBON"))
        self.context_bounds = _bounds(self.context)
        mb_rings = [points for region in anatomy.regions if not region.model_term for points, _ in region.rings]
        self.detail_bounds = _bounds(np.concatenate([cloud.xy for cloud in self.groups.values()] + mb_rings))


def _valid_points(xy: np.ndarray) -> bool:
    return xy.ndim == 2 and xy.shape[1] == 2 and len(xy) > 0 and np.isfinite(xy).all()


def _bounds(xy: np.ndarray) -> np.ndarray:
    return np.array([xy.min(axis=0), xy.max(axis=0)], dtype=float)


class Projection:
    """Map the common frontal (u, v) coordinates onto square screen pixels."""

    def __init__(self, bounds: np.ndarray, width: int, height: int, padding: int):
        lo, hi = bounds
        scale = min((width - 2 * padding) / (hi[0] - lo[0]), (height - 2 * padding) / (hi[1] - lo[1]))
        self.step = np.array([scale, -scale])
        self.origin = (width / 2, height / 2) - (lo + hi) * 0.5 * self.step

    def pixels(self, xy: np.ndarray) -> np.ndarray:
        return np.rint(xy * self.step + self.origin).astype(np.intp)


def _addresses(xy: np.ndarray, projection: Projection, width: int, height: int):
    x, y = projection.pixels(xy).T
    on_screen = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    return y[on_screen] * width + x[on_screen], on_screen


def context_image(context: np.ndarray, projection: Projection, width: int, height: int, style) -> np.ndarray:
    addresses, _ = _addresses(context, projection, width, height)
    density = np.bincount(addresses, minlength=width * height).reshape(height, width).astype(np.float32)
    soft = gaussian_filter(density, style.soma_sigma, mode="constant")
    gray = style.soma_ceiling * -np.expm1(-(soft + density * style.soma_core) * style.soma_gain)
    rgb = np.asarray(p.BRAIN_VOID, dtype=np.float32) + gray[..., None]
    return rgb.astype(np.uint8)


def add_light(base: np.ndarray, light: np.ndarray, *, sigma: float, gain: float, ceiling: float) -> np.ndarray:
    """Bloom in pixel space, then compress the peak while preserving hue."""
    glow = gaussian_filter(light, sigma=(sigma, sigma, 0), mode="constant")
    light = light + glow * gain
    peak = light.max(axis=2, keepdims=True)
    limit = np.minimum(ceiling, 240 - base.max(axis=2, keepdims=True).astype(np.float32))
    mapped = light * (limit / (limit + peak))
    return (base + mapped).astype(np.uint8)


class ActivityRaster:
    """Cache projection and background. Accumulate only within the activity bounds."""

    def __init__(self, clouds: PointClouds, projection: Projection, width: int, height: int, style):
        self.style = style
        self.background = context_image(clouds.context, projection, width, height, style)
        projected = {}
        bounds = []
        for group, cloud in clouds.groups.items():
            pixels, visible = _addresses(cloud.xy, projection, width, height)
            ys, xs = np.divmod(pixels, width)
            projected[group] = (xs, ys, cloud.owners[visible])
            if pixels.size:
                bounds.append((xs.min(), ys.min(), xs.max(), ys.max()))
        edges = np.asarray(bounds)
        pad = int(np.ceil(4 * style.bloom_sigma))
        left, top = np.maximum(0, edges[:, :2].min(axis=0) - pad)
        right, bottom = np.minimum((width, height), edges[:, 2:].max(axis=0) + pad + 1)
        self.window = (slice(top, bottom), slice(left, right))
        self.shape = (bottom - top, right - left, 3)
        self.groups = {key: ((ys - top) * self.shape[1] + xs - left, owners)
                       for key, (xs, ys, owners) in projected.items()}

    def render(self, activity: dict[str, np.ndarray]) -> np.ndarray:
        light = np.zeros(self.shape, dtype=np.float32)
        flat = light.reshape(-1, 3)
        for group, (pixels, owners) in self.groups.items():
            values = activity[group]
            active = np.any(values > 0, axis=1)[owners]
            np.add.at(flat, pixels[active], values[owners[active]])
        style = self.style
        frame = self.background.copy()
        frame[self.window] = add_light(self.background[self.window], light, sigma=style.bloom_sigma,
                                       gain=style.bloom_gain, ceiling=style.activity_ceiling)
        return frame
