"""Read the frontal projection and bind replay indices to anatomy by body ID."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from flypac.brain.assign import load_raw
from flypac.brain.flybrain import circuit_from_config, load_brain_config, repo_path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANATOMY = ROOT / "data" / "anatomy" / "mb_anatomy_v1.npz"
DEFAULT_BRAIN_CONFIG = ROOT / "configs" / "brain_v1.yaml"


@dataclass(frozen=True)
class Region:
    name: str
    model_term: bool
    rings: tuple[tuple[np.ndarray, bool], ...]


class Anatomy:
    def __init__(self, path: Path = DEFAULT_ANATOMY, brain_config: Path = DEFAULT_BRAIN_CONFIG):
        config = load_brain_config(brain_config)
        circuit = circuit_from_config(config)
        raw = load_raw(repo_path(config.circuit.raw))
        meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
        if meta["view"] != "frontal" or meta["raw_circuit"]["sha256"] != circuit.source_hash:
            raise ValueError("The frontal anatomy and the configured circuit must come from the same raw circuit")

        with np.load(path, allow_pickle=False) as file:
            a = {name: file[name] for name in file.files}
        xy, has_coord, reasons = a["xy_um"], a["has_coord"], a["missing_reason"]
        if not np.array_equal(has_coord, np.isfinite(xy).all(axis=1)) or np.any(reasons[~has_coord] == ""):
            raise ValueError("Every neuron must have a finite coordinate or a recorded reason")

        self.dataset = meta["dataset"]
        self.source_hash = circuit.source_hash
        self.coordinates = {}
        self.body_ids = {}
        for kind, ids in (("PN", raw.pn_body), ("KC", raw.kc_body), ("MBON", raw.mbon_body), ("DAN", None)):
            rows = a["kind"] == kind
            if ids is not None and not np.array_equal(a["body"][rows], ids):
                raise ValueError(f"Anatomy {kind} body IDs do not match replay circuit order")
            self.coordinates[kind] = xy[rows]
            self.body_ids[kind] = a["body"][rows]
        self.missing = tuple((str(kind), int(body), str(reason))
                             for kind, body, reason in zip(a["kind"][~has_coord], a["body"][~has_coord],
                                                           reasons[~has_coord]))
        self.dan_regions = frozenset(a["dan_compartment"][(a["kind"] == "DAN") & has_coord])
        rings = [[] for _ in a["region_names"]]
        for n, (region, hole) in enumerate(zip(a["ring_region"], a["ring_is_hole"])):
            first, last = a["ring_offsets"][n:n + 2]
            rings[region].append((a["ring_xy"][first:last], bool(hole)))
        self.regions = tuple(Region(str(name), bool(term), tuple(polygons))
                             for name, term, polygons in zip(a["region_names"], a["region_model_term"], rings))
        if not self.dan_regions <= {region.name for region in self.regions if not region.model_term}:
            raise ValueError("DAN compartment names must match anatomical regions")
        self.bounds = np.array([a["ring_xy"].min(axis=0), a["ring_xy"].max(axis=0)], dtype=float)
        self.pn_channels = circuit.pn_channel
        self.mbon_bundles = circuit.mbon_bundle
        self.instinct_gain = float(config.learn.g)
