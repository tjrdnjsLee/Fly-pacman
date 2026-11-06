"""해부 자료 내보내기 (TRD 19.13, M1 뇌 작업 지시서 ① 5절). 토큰 필요 (캐시가 있으면 캐시만 읽음).

    uv run --extra extract python scripts/export_anatomy.py --config configs/circuit_extract_v1.yaml

출력: data/anatomy/mb_anatomy_v1.npz + mb_anatomy_v1.meta.json, 보고서 runs/extract/anatomy_report.md.

- 대표 좌표: 뉴런마다 해당 ROI 안 시냅스 위치의 중심 (5.1).
- 정면 투영: 설정의 축 변환 가설로 앞뒤 축을 버리고 등쪽이 위가 되게 한다.
  확인 시험 (가) ~ (다)와 앞뒤 축 시험 (라)가 하나라도 틀리면 멈춘다 (5.2).
- 영역 경계: ROI 메시 삼각형을 정면으로 투영해 격자에 칠하고, 경계를 따라 다각형으로 만든 뒤
  Douglas–Peucker로 단순화한다 (shapely를 쓰지 않음, 새 의존성 없음).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_circuit as ec  # noqa: E402

ROOT = ec.ROOT
AXES = {"x": 0, "y": 1, "z": 2}
REGION_LABELS = {"CA(R)": "칼릭스", "PED(R)": "꽃자루", "LH(R)": "외측뿔"}


# ---------------------------------------------------------------- 조회 (캐시)

def safe_name(roi: str) -> str:
    return roi.replace("(", "_").replace(")", "").replace("'", "p")


def fetch_mesh(cache_dir: Path, roi: str, lazy_client) -> tuple[np.ndarray, np.ndarray]:
    path = cache_dir / f"mesh_{safe_name(roi)}.obj"
    if not path.exists():
        print(f"[조회] 메시 {roi}", flush=True)
        path.write_bytes(lazy_client().fetch_roi_mesh(roi))
    V, F = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            F.append([int(x.split("/")[0]) - 1 for x in line.split()[1:4]])
    return np.asarray(V, dtype=np.float64), np.asarray(F, dtype=np.int64)


def fetch_voxel_meta(client) -> dict[str, pd.DataFrame]:
    df = client.fetch_custom("MATCH (m:Meta) RETURN m.voxelSize AS voxelSize, m.voxelUnits AS voxelUnits")
    return {"meta": pd.DataFrame({"voxelSize": [json.dumps(list(df["voxelSize"].iloc[0]))],
                                  "voxelUnits": [str(df["voxelUnits"].iloc[0])]})}


def fetch_pn_kc_synapses(client, pn_ids, kc_ids) -> dict[str, pd.DataFrame]:
    from neuprint import NeuronCriteria as NC, SynapseCriteria as SC, fetch_synapse_connections

    df = fetch_synapse_connections(NC(bodyId=list(map(int, pn_ids))), NC(bodyId=list(map(int, kc_ids))),
                                   SC(rois=[ec.ROI_MAIN_CALYX]), client=client)
    cols = ["bodyId_pre", "bodyId_post", "x_pre", "y_pre", "z_pre", "x_post", "y_post", "z_post"]
    return {"syn": df[cols].copy()}


def fetch_kc_mbon_synapses(client, kc_ids, mbon_ids) -> dict[str, pd.DataFrame]:
    from neuprint import NeuronCriteria as NC, SynapseCriteria as SC, fetch_synapse_connections

    df = fetch_synapse_connections(NC(bodyId=list(map(int, kc_ids))), NC(bodyId=list(map(int, mbon_ids))),
                                   SC(rois=[ec.ROI_PED, *ec.ROI_COMPARTMENTS], primary_only=False),
                                   client=client)
    cols = ["bodyId_pre", "bodyId_post", "x_post", "y_post", "z_post"]
    return {"syn": df[cols].copy()}


def pair_synapses(cache: "ec.Cache", name: str, describe: str, fetch_fn, pre_ids, post_ids,
                  lazy_client) -> pd.DataFrame:
    """뉴런 쌍 시냅스 표를 캐시에서 읽어 지금 회로(pre_ids × post_ids)로 거른다.

    캐시는 조회한 bodyId 목록(ids_pre, ids_post)을 함께 둔다. 지금 회로에 캐시가 덮지 못한 뉴런이
    있으면 그 뉴런의 쌍만 서버에서 더 받아 붙인다 (토큰 필요). 목록이 없는 캐시는 처음부터 다시 받는다.
    """
    d = ec.CACHE_DIR
    f_syn, f_pre, f_post = (d / f"{name}__syn.parquet", d / f"{name}__ids_pre.parquet",
                            d / f"{name}__ids_post.parquet")
    pre_ids = np.asarray(pre_ids, dtype=np.int64)
    post_ids = np.asarray(post_ids, dtype=np.int64)
    if cache.refresh or not (f_syn.exists() and f_pre.exists() and f_post.exists()):
        syn = pd.DataFrame()
        have_pre = have_post = np.array([], dtype=np.int64)
    else:
        syn = pd.read_parquet(f_syn)
        have_pre = pd.read_parquet(f_pre)["bodyId"].to_numpy()
        have_post = pd.read_parquet(f_post)["bodyId"].to_numpy()
    miss_pre = np.setdiff1d(pre_ids, have_pre)
    miss_post = np.setdiff1d(post_ids, have_post)
    parts = [syn]
    if len(miss_pre) and len(post_ids):
        print(f"[조회] {name}: 새 pre {len(miss_pre)}개", flush=True)
        parts.append(fetch_fn(lazy_client(), miss_pre, post_ids)["syn"])
    if len(miss_post) and len(pre_ids):
        print(f"[조회] {name}: 새 post {len(miss_post)}개", flush=True)
        keep_pre = np.setdiff1d(pre_ids, miss_pre)          # miss_pre 쪽은 위에서 이미 받음
        if len(keep_pre):
            parts.append(fetch_fn(lazy_client(), keep_pre, miss_post)["syn"])
    if len(parts) > 1:
        syn = pd.concat([x for x in parts if len(x)], ignore_index=True).drop_duplicates()
        syn.to_parquet(f_syn, index=False)
        pd.DataFrame({"bodyId": np.union1d(have_pre, pre_ids)}).to_parquet(f_pre, index=False)
        pd.DataFrame({"bodyId": np.union1d(have_post, post_ids)}).to_parquet(f_post, index=False)
        cache.meta[name] = {"query": describe, "dataset": ec.DATASET, "server": ec.SERVER,
                            "fetched_at": datetime.now(timezone.utc).isoformat(),
                            "neuprint_python_version": ec.neuprint_version(),
                            "tables": {"syn": int(len(syn))}, "incremental": True}
        cache.meta_path.write_text(json.dumps(cache.meta, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        print(f"[캐시] {name}")
    return syn[syn["bodyId_pre"].isin(pre_ids) & syn["bodyId_post"].isin(post_ids)]


def fetch_dan_synapses(client, dan_ids) -> dict[str, pd.DataFrame]:
    from neuprint import NeuronCriteria as NC, SynapseCriteria as SC, fetch_synapses

    df = fetch_synapses(NC(bodyId=list(map(int, dan_ids))),
                        SC(type="pre", rois=list(ec.ROI_COMPARTMENTS), primary_only=False), client=client)
    return {"syn": df[["bodyId", "x", "y", "z"]].copy()}


# ---------------------------------------------------------------- 투영

class Projection:
    def __init__(self, acfg, scale_um: float):
        self.drop = AXES[acfg["drop_axis"]]
        self.up = AXES[acfg["up_axis"]]
        self.up_sign = float(acfg["up_sign"])
        self.h = AXES[acfg["horizontal_axis"]]
        self.h_sign = float(acfg["horizontal_sign"])
        if len({self.drop, self.up, self.h}) != 3:
            raise SystemExit("축 변환 설정이 세 축을 하나씩 쓰지 않는다.")
        self.scale = scale_um

    def __call__(self, xyz_vox: np.ndarray) -> np.ndarray:
        p = np.asarray(xyz_vox, dtype=np.float64) * self.scale
        return np.stack([self.h_sign * p[..., self.h], self.up_sign * p[..., self.up]], axis=-1)


# ---------------------------------------------------------------- 영역 다각형

def rasterize(tri: np.ndarray, res: float) -> tuple[np.ndarray, np.ndarray]:
    """투영한 삼각형들 (T,3,2)의 합집합을 격자에 칠한다. 반환: mask (H,W) bool, 원점 (2,)."""
    from scipy import ndimage

    lo = tri.reshape(-1, 2).min(axis=0) - 2 * res
    hi = tri.reshape(-1, 2).max(axis=0) + 2 * res
    W, H = (np.ceil((hi - lo) / res).astype(int) + 1)
    mask = np.zeros((H, W), dtype=bool)
    edge = np.max(np.linalg.norm(tri - np.roll(tri, 1, axis=1), axis=2), axis=1)
    n_sub = np.maximum(1, np.ceil(edge / (0.5 * res)).astype(int))
    for n in np.unique(n_sub):
        t = tri[n_sub == n]
        i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
        keep = i + j <= n
        a = (i[keep] / n)[:, None]
        b = (j[keep] / n)[:, None]
        c = 1.0 - a - b
        pts = a[None] * t[:, None, 0] + b[None] * t[:, None, 1] + c[None] * t[:, None, 2]
        ij = np.floor((pts.reshape(-1, 2) - lo) / res).astype(int)
        mask[ij[:, 1], ij[:, 0]] = True
    mask = ndimage.binary_closing(mask, structure=np.ones((3, 3), bool))
    return mask, lo


def trace_rings(mask: np.ndarray) -> list[np.ndarray]:
    """칠한 격자의 경계를 고리들로 (격자 모서리 좌표 (c, r)). 안쪽이 왼쪽 (바깥 = 반시계)."""
    H, W = mask.shape
    m = np.pad(mask, 1)
    inside = m[1:-1, 1:-1]
    r, c = np.nonzero(inside)
    edges = []
    below = ~m[r, c + 1]
    edges += [((cc, rr), (cc + 1, rr)) for rr, cc in zip(r[below], c[below])]
    right = ~m[r + 1, c + 2]
    edges += [((cc + 1, rr), (cc + 1, rr + 1)) for rr, cc in zip(r[right], c[right])]
    above = ~m[r + 2, c + 1]
    edges += [((cc + 1, rr + 1), (cc, rr + 1)) for rr, cc in zip(r[above], c[above])]
    left = ~m[r + 1, c]
    edges += [((cc, rr + 1), (cc, rr)) for rr, cc in zip(r[left], c[left])]
    out: dict[tuple, list] = {}
    for s, e in edges:
        out.setdefault(s, []).append(e)
    rings = []
    while out:
        start = next(iter(out))
        ring = [start]
        prev_dir = None
        cur = start
        while True:
            cands = out.get(cur)
            if not cands:
                break
            if len(cands) == 1 or prev_dir is None:
                nxt = cands[0]
            else:  # 꼭짓점에서 두 갈래: 왼쪽으로 꺾는 쪽을 먼저 (규칙을 고정해 재현 가능)
                def turn(e):
                    d = (e[0] - cur[0], e[1] - cur[1])
                    return -(prev_dir[0] * d[1] - prev_dir[1] * d[0])
                nxt = min(cands, key=turn)
            cands.remove(nxt)
            if not cands:
                del out[cur]
            prev_dir = (nxt[0] - cur[0], nxt[1] - cur[1])
            cur = nxt
            if cur == start:
                break
            ring.append(cur)
        rings.append(np.asarray(ring, dtype=np.float64))
    return rings


def signed_area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def rdp(points: np.ndarray, tol: float) -> np.ndarray:
    """열린 꺾은선의 Douglas–Peucker 단순화 (반복형)."""
    n = len(points)
    if n < 3:
        return points
    keep = np.zeros(n, dtype=bool)
    keep[[0, -1]] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = points[i], points[j]
        seg = b - a
        L = np.hypot(*seg)
        q = points[i + 1:j] - a
        if L == 0:
            d = np.hypot(q[:, 0], q[:, 1])
        else:
            d = np.abs(seg[0] * q[:, 1] - seg[1] * q[:, 0]) / L
        k = int(np.argmax(d))
        if d[k] > tol:
            keep[i + 1 + k] = True
            stack += [(i, i + 1 + k), (i + 1 + k, j)]
    return points[keep]


def simplify_ring(ring: np.ndarray, tol: float) -> np.ndarray:
    far = int(np.argmax(np.hypot(*(ring - ring[0]).T)))
    a = rdp(np.vstack([ring[:far + 1]]), tol)
    b = rdp(np.vstack([ring[far:], ring[:1]]), tol)
    return np.vstack([a[:-1], b[:-1]])


def region_polygons(V_vox, F, proj: Projection, acfg) -> list[tuple[np.ndarray, bool]]:
    tri = proj(V_vox)[F]
    res = float(acfg["raster_um"])
    mask, lo = rasterize(tri, res)
    out = []
    for ring in trace_rings(mask):
        xy = ring * res + lo
        area = signed_area(xy)
        if abs(area) < float(acfg["min_ring_area_um2"]):
            continue
        s = simplify_ring(xy, float(acfg["simplify_um"]))
        if len(s) >= 3:
            out.append((s, area < 0))
    return out


# ---------------------------------------------------------------- 기준점과 확인 시험

def far_mean(points: np.ndarray, ref: np.ndarray, q: float) -> np.ndarray:
    d = np.linalg.norm(points - ref, axis=1)
    return points[d >= np.quantile(d, q)].mean(axis=0)


def landmarks(meshes: dict, q: float) -> dict[str, np.ndarray]:
    """3차원 기준점 (복셀). 축 가설과 무관하게 거리로만 정한다."""
    calyx = meshes[ec.ROI_MAIN_CALYX][0].mean(axis=0)
    root = far_mean(meshes[ec.ROI_PED][0], calyx, q)          # 꽃자루에서 칼릭스와 가장 먼 끝
    alpha = far_mean(meshes["aL(R)"][0], root, q)               # α엽에서 뿌리와 가장 먼 끝
    beta = far_mean(meshes["bL(R)"][0], root, q)
    mid = meshes["EB"][0].mean(axis=0)                           # 타원체(EB)는 정중선 구조
    return {"root": root, "alpha_tip": alpha, "beta_tip": beta, "calyx": calyx, "midline": mid}


def orientation_checks(lm3: dict, proj: Projection) -> dict[str, bool]:
    p = {k: proj(v) for k, v in lm3.items()}
    ap_axis = int(np.argmax(np.abs(lm3["calyx"] - lm3["root"])))
    return {
        "가_alpha_tip_above_root": bool(p["alpha_tip"][1] > p["root"][1]),
        "나_beta_tip_medial_to_root": bool(abs(p["beta_tip"][0] - p["midline"][0])
                                           < abs(p["root"][0] - p["midline"][0])),
        "다_calyx_above_root": bool(p["calyx"][1] > p["root"][1]),
        "라_dropped_axis_is_calyx_root_axis": bool(ap_axis == proj.drop),
    }


# ---------------------------------------------------------------- 본체

def centroids(df: pd.DataFrame, key: str, cols: list[str]) -> pd.DataFrame:
    g = df.groupby(key)
    out = g[cols].mean()
    out.columns = ["x", "y", "z"]
    out["n"] = g.size()
    return out


def run(cfg_path: Path, refresh: bool) -> Path:
    from flypac.core.config import load_config

    cfg = load_config(cfg_path)
    ec.check_config_rois(cfg)
    acfg = cfg["anatomy"]
    raw_path = ROOT / cfg["output"]["raw"]
    with np.load(raw_path, allow_pickle=False) as z:
        raw = {k: z[k] for k in z.files}

    token = ec.read_token()
    client = None

    def lazy_client():
        nonlocal client
        if client is None:
            if not token:
                raise SystemExit("NEUPRINT_TOKEN 이 .env 에 없습니다.")
            client = ec.connect(token)
        return client

    def lazy(fn, *args):
        return lambda: fn(lazy_client(), *args)

    d = ec.load_data(lambda: lazy_client(), False)
    cache = d["cache"]
    cache.refresh = refresh
    dan = d["dan"].sort_values("bodyId")
    vm = cache.get("voxel_meta", "MATCH (m:Meta) RETURN voxelSize, voxelUnits", lazy(fetch_voxel_meta))["meta"]
    voxel = json.loads(vm["voxelSize"].iloc[0])
    units = vm["voxelUnits"].iloc[0]
    if not (np.allclose(voxel, [acfg["voxel_nm"]] * 3) and units in ("nanometers", "nm")):
        raise SystemExit(f"서버 복셀 크기 {voxel} {units} 가 설정 voxel_nm={acfg['voxel_nm']} 와 다르다. 멈춤.")
    scale = float(acfg["voxel_nm"]) / 1000.0
    proj = Projection(acfg, scale)

    pk = pair_synapses(cache, "anat_pn_kc", "fetch_synapse_connections(원본 PN -> 원본 KC, CA(R))",
                       fetch_pn_kc_synapses, raw["pn_body"], raw["kc_body"], lazy_client)
    km = pair_synapses(cache, "anat_kc_mbon", "fetch_synapse_connections(원본 KC -> 원본 MBON, PED + 15구획)",
                       fetch_kc_mbon_synapses, raw["kc_body"], raw["mbon_body"], lazy_client)
    ds = cache.get("anat_dan", "fetch_synapses(PAM · PPL1, 시냅스 전, 15구획)",
                   lazy(fetch_dan_synapses, dan["bodyId"].to_numpy()))["syn"]

    mesh_dir = ec.CACHE_DIR / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    region_rois = [ec.ROI_MAIN_CALYX, ec.ROI_PED, *ec.ROI_COMPARTMENTS, ec.ROI_LH]
    meshes = {r: fetch_mesh(mesh_dir, r, lazy_client) for r in [*region_rois, "aL(R)", "bL(R)", "EB"]}

    # 확인 시험 (5.2)
    lm3 = landmarks(meshes, float(acfg["landmark_quantile"]))
    checks = orientation_checks(lm3, proj)
    if not all(checks.values()):
        raise SystemExit(f"정면 투영 확인 시험 실패: {checks}. 멈추고 보고한다.")

    # 대표 좌표 (5.1)
    c_pn = centroids(pk, "bodyId_pre", ["x_pre", "y_pre", "z_pre"])
    c_kc = centroids(pk, "bodyId_post", ["x_post", "y_post", "z_post"])
    c_mb = centroids(km, "bodyId_post", ["x_post", "y_post", "z_post"])
    c_dn = centroids(ds, "bodyId", ["x", "y", "z"])
    reasons = {"PN": "본 칼릭스 안 PN→원본 KC 시냅스 없음", "KC": "본 칼릭스 안 원본 PN→KC 시냅스 없음",
               "MBON": "꽃자루 · 15구획 안 원본 KC→MBON 시냅스 없음", "DAN": "15구획 안 시냅스 전 위치 없음"}

    rc = d["roi_counts"]
    rc = rc[rc["roi"].isin(ec.ROI_COMPARTMENTS) & rc["bodyId"].isin(dan["bodyId"])]
    dpiv = rc.pivot_table(index="bodyId", columns="roi", values="pre", aggfunc="sum", fill_value=0)
    dpiv = dpiv.reindex(dan["bodyId"], fill_value=0)
    dan_comp = dpiv.idxmax(axis=1).where(dpiv.sum(axis=1) > 0, "")

    blocks = [("PN", raw["pn_body"], c_pn), ("KC", raw["kc_body"], c_kc),
              ("MBON", raw["mbon_body"], c_mb), ("DAN", dan["bodyId"].to_numpy(), c_dn)]
    body, kind, xyz, nsyn, reason, comp = [], [], [], [], [], []
    for k, ids, cdf in blocks:
        cd = cdf.reindex(ids)
        body.append(np.asarray(ids, dtype=np.int64))
        kind += [k] * len(ids)
        xyz.append(cd[["x", "y", "z"]].to_numpy(dtype=np.float64))
        nsyn.append(cd["n"].fillna(0).to_numpy(dtype=np.int32))
        reason += [reasons[k] if np.isnan(v) else "" for v in cd["x"].to_numpy()]
        comp += [str(dan_comp.get(b, "")) for b in ids] if k == "DAN" else [""] * len(ids)
    body = np.concatenate(body)
    xyz_vox = np.vstack(xyz)
    has = ~np.isnan(xyz_vox).any(axis=1)
    xy = proj(xyz_vox)
    xyz_um = xyz_vox * scale

    # 영역 경계 (5.3)
    reg_names, reg_labels, reg_model, ring_region, ring_hole, ring_xy, offsets = [], [], [], [], [], [], [0]
    for ri, roi in enumerate(region_rois):
        reg_names.append(roi)
        reg_labels.append(REGION_LABELS.get(roi, "엽 구획"))
        reg_model.append(roi == ec.ROI_LH)
        for ring, hole in region_polygons(*meshes[roi], proj, acfg):
            ring_region.append(ri)
            ring_hole.append(hole)
            ring_xy.append(ring.astype(np.float32))
            offsets.append(offsets[-1] + len(ring))

    lm_names = list(lm3)
    out = {
        "body": body,
        "kind": np.asarray(kind, dtype=str),
        "xyz_um": xyz_um,
        "xy_um": xy,
        "has_coord": has,
        "missing_reason": np.asarray(reason, dtype=str),
        "n_syn": np.concatenate(nsyn),
        "dan_compartment": np.asarray(comp, dtype=str),
        "region_names": np.asarray(reg_names, dtype=str),
        "region_labels": np.asarray(reg_labels, dtype=str),
        "region_model_term": np.asarray(reg_model, dtype=bool),
        "ring_region": np.asarray(ring_region, dtype=np.int32),
        "ring_is_hole": np.asarray(ring_hole, dtype=bool),
        "ring_offsets": np.asarray(offsets, dtype=np.int64),
        "ring_xy": np.vstack(ring_xy),
        "landmark_names": np.asarray(lm_names, dtype=str),
        "landmark_xyz_um": np.vstack([lm3[k] for k in lm_names]) * scale,
        "landmark_xy": np.vstack([proj(lm3[k]) for k in lm_names]),
    }
    out_path = ROOT / cfg["output"]["anatomy"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)

    counts = {k: {"total": int(sum(1 for x in kind if x == k)),
                  "with_coord": int(sum(1 for x, h in zip(kind, has) if x == k and h))}
              for k in ("PN", "KC", "MBON", "DAN")}
    meta = {
        "format": "mb_anatomy_v1",
        "dataset": ec.DATASET,
        "server": ec.SERVER,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "neuprint_python_version": ec.neuprint_version(),
        "view": acfg["view"],
        "axis_transform": {
            "dropped_axis": acfg["drop_axis"] + " (앞뒤)",
            "u (가로)": f"{'+' if acfg['horizontal_sign'] > 0 else '-'}{acfg['horizontal_axis']} × scale",
            "v (세로, 위 = 등쪽)": f"{'+' if acfg['up_sign'] > 0 else '-'}{acfg['up_axis']} × scale",
            "note": "앞에서 본 모습. 오른쪽 버섯체의 정중선 쪽이 화면 오른쪽(+u).",
        },
        "units": "μm",
        "source_units": f"복셀 {voxel} {units}",
        "scale_um_per_voxel": scale,
        "xyz_um": "원 좌표계(x, y, z) × scale. 축 순서는 바꾸지 않음",
        "orientation_checks": checks,
        "landmark_definitions": {
            "calyx": "CA(R) 메시 꼭짓점 평균",
            "root": f"PED(R) 꼭짓점 중 칼릭스 중심에서 가장 먼 {1 - acfg['landmark_quantile']:.0%}의 평균 (엽 갈림 쪽 끝)",
            "alpha_tip": "aL(R) 꼭짓점 중 root에서 가장 먼 쪽 평균",
            "beta_tip": "bL(R) 꼭짓점 중 root에서 가장 먼 쪽 평균",
            "midline": "EB 메시 꼭짓점 평균 (정중선 구조)",
        },
        "representative_coordinate": {
            "PN": "본 칼릭스 CA(R) 안 PN→원본 KC 시냅스의 시냅스 전 위치 중심",
            "KC": "본 칼릭스 CA(R) 안 원본 PN→KC 시냅스의 시냅스 후 위치 중심 (발톱)",
            "MBON": "꽃자루 + 15구획 안 원본 KC→MBON 시냅스의 시냅스 후 위치 중심",
            "DAN": "15구획 안 DAN 시냅스 전 위치 중심. dan_compartment = 시냅스 전 위치가 가장 많은 구획",
        },
        "neuron_order": "PN, KC, MBON (원본 회로 순서 = bodyId 오름차순), DAN (bodyId 오름차순)",
        "counts": counts,
        "regions": {
            "rois": region_rois,
            "calyx_note": "칼릭스 영역은 본 칼릭스 CA(R)만. 부속 칼릭스(dACA · lACA · vACA)는 그리지 않음",
            "model_term": {ec.ROI_LH: "외측뿔은 모델 항(본능 g·o_a)이며 뉴런이 아님"},
            "antennal_lobe": "그리지 않음 (hemibrain에 일부만 있음)",
            "method": "ROI 메시 삼각형 정면 투영 → 격자 칠하기 → 3×3 닫힘 → 경계 추적 → Douglas–Peucker",
            "raster_um": acfg["raster_um"],
            "simplify_tolerance_um": acfg["simplify_um"],
            "min_ring_area_um2": acfg["min_ring_area_um2"],
            "rings": "ring_offsets[k]:ring_offsets[k+1] 가 고리 k의 꼭짓점. ring_is_hole = 시계 방향(구멍)",
        },
        "raw_circuit": {"path": cfg["output"]["raw"], "sha256": ec.sha256_file(raw_path)},
        "attribution": "hemibrain v1.2.1, Janelia FlyEM. Scheffer et al. 2020, eLife 9:e57443, "
                       "\"A connectome and analysis of the adult Drosophila central brain\". CC BY 4.0 "
                       "(관리자 확인 후 확정).",
        "honesty_note": "화면의 빛은 모델의 활동을 실제 해부 위치에 표시한 것이며 실제 초파리의 측정이 아니다 (TRD 19.13).",
        "config_hash": cfg.hash(),
    }
    meta_path = out_path.with_name(out_path.name[:-4] + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # 보고서
    lines = ["# 단계 3: 해부 자료 보고서\n", f"- 출력: `{cfg['output']['anatomy']}`",
             f"- 복셀: {voxel} {units} → {scale} μm/복셀", "", "## 정면 투영 확인 시험 (5.2)\n"]
    lines += [f"- {k}: {'통과' if v else '실패'}" for k, v in checks.items()]
    lines += ["", "| 기준점 | u (μm) | v (μm) |", "|---|---|---|"]
    lines += [f"| {k} | {proj(v)[0]:.1f} | {proj(v)[1]:.1f} |" for k, v in lm3.items()]
    lines += ["", "## 좌표\n", "| 종류 | 전체 | 좌표 있음 | 시냅스 수 중앙값 |", "|---|---|---|---|"]
    n_syn = out["n_syn"]
    for k, v in counts.items():
        m = out["kind"] == k
        lines.append(f"| {k} | {v['total']} | {v['with_coord']} | {np.median(n_syn[m & has]) if (m & has).any() else 0:.0f} |")
    miss = [(b, k, r) for b, k, r, h in zip(body, kind, reason, has) if not h]
    lines += ["", f"좌표 없는 뉴런 {len(miss)}개: " + (", ".join(f"{k} {b} ({r})" for b, k, r in miss) or "없음")]
    lines += ["", "## 영역\n", "| ROI | 고리 수 | 꼭짓점 수 | 넓이 (μm²) |", "|---|---|---|---|"]
    rr = out["ring_region"]
    for ri, roi in enumerate(region_rois):
        ks = np.nonzero(rr == ri)[0]
        nv = sum(offsets[k + 1] - offsets[k] for k in ks)
        area = sum(signed_area(ring_xy[k].astype(np.float64)) for k in ks)
        lines.append(f"| {roi} | {len(ks)} | {nv} | {area:.0f} |")
    (ec.OUT_DIR / "anatomy_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"checks": checks, "counts": counts}, ensure_ascii=False))
    return out_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args(argv)
    scrub = ec.Scrubber(ec.read_token())
    try:
        path = run(args.config, args.refresh)
        print(f"해부 자료: {path.relative_to(ROOT)}")
        return 0
    except SystemExit as e:
        print(scrub(e), file=sys.stderr)
        return 1
    except Exception as e:
        print(f"실패: {type(e).__name__}: {scrub(e)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
