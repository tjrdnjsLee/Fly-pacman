"""뇌 렌더러 v2용 해부 자료: 뇌 전체 형태(세포체 위치)와 버섯체 시냅스 점구름. 토큰 필요 (시험은 부르지 않음).

    uv run --extra extract python scripts/export_anatomy_clouds.py --config configs/anatomy_clouds_v1.yaml \\
        [--env ~/dev/fly-pacman/.env] [--read-cache ~/dev/fly-pacman/runs/extract/cache]

- 기존 캐시(--read-cache)는 읽기만 한다. 새로 받은 표는 이 저장소의 runs/anat_clouds_cache/ 에 둔다.
- 투영은 export_anatomy.Projection 과 같은 함수, 같은 설정이다.
- 점은 뉴런별 고유 위치(같은 위치의 중복 시냅스 쌍은 한 점)이며, 상한을 넘으면 고정 시드로 균일하게 뽑는다.
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
import export_anatomy as ea  # noqa: E402
import extract_circuit as ec  # noqa: E402

ROOT = ec.ROOT
OWN_CACHE = ROOT / "runs" / "anat_clouds_cache"
KIND_CODE = {"pn_calyx": 1, "kc_calyx": 2, "kc_lobe": 3, "mbon_lobe": 4}
NOTE_CONTEXT = "형태 배경 전용, 시뮬레이션하지 않음"


def sha256(path: Path) -> str:
    return ec.sha256_file(path)


def fetch_soma(client) -> pd.DataFrame:
    """뉴런 정보만 한 번에: 모든 뉴런의 bodyId 와 세포체 위치 (없으면 빈 칸). 시냅스는 조회하지 않는다."""
    df = client.fetch_custom(
        "MATCH (n:Neuron) RETURN n.bodyId AS bodyId, n.somaLocation AS soma")
    return df


def soma_xyz(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, dict):
        c = v.get("coordinates")
        return None if c is None else [float(t) for t in c]
    if isinstance(v, (list, tuple, np.ndarray)) and len(v) == 3:
        return [float(t) for t in v]
    raise SystemExit(f"알 수 없는 somaLocation 형식: {type(v).__name__} {str(v)[:80]}")


def fetch_kc_mbon_full(client, kc_ids, mbon_ids) -> pd.DataFrame:
    """기존 kc_mbon 조회와 같은 쿼리, 시냅스 전·후 위치를 모두 받는다."""
    from neuprint import NeuronCriteria as NC, SynapseCriteria as SC, fetch_synapse_connections

    df = fetch_synapse_connections(NC(bodyId=list(map(int, kc_ids))), NC(bodyId=list(map(int, mbon_ids))),
                                   SC(rois=[ec.ROI_PED, *ec.ROI_COMPARTMENTS], primary_only=False),
                                   client=client)
    return df[["bodyId_pre", "bodyId_post", "x_pre", "y_pre", "z_pre", "x_post", "y_post", "z_post"]].copy()


def cloud(df: pd.DataFrame, key: str, cols: list[str], ids: np.ndarray, cap: int, seed: int, code: int,
          proj) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """뉴런 순서(ids)대로 점구름을 만든다. 반환: 점 (N,2) float32, 시작 위치 (len(ids)+1), 원래 점 수."""
    uniq = df[[key, *cols]].drop_duplicates()
    groups = {b: g[cols].to_numpy(dtype=np.float64) for b, g in uniq.groupby(key)}
    pts, offsets, n_orig = [], [0], []
    for b in ids:
        p = groups.get(int(b))
        if p is None:
            n_orig.append(0)
            offsets.append(offsets[-1])
            continue
        p = p[np.lexsort(p.T[::-1])]                       # 순서를 값으로 고정 (입력 순서와 무관)
        n_orig.append(len(p))
        if len(p) > cap:
            rng = np.random.default_rng([seed, code, int(b)])
            p = p[np.sort(rng.choice(len(p), size=cap, replace=False))]
        pts.append(proj(p).astype(np.float32))
        offsets.append(offsets[-1] + len(p))
    xy = np.vstack(pts) if pts else np.zeros((0, 2), np.float32)
    return xy, np.asarray(offsets, dtype=np.int64), np.asarray(n_orig, dtype=np.int32)


def run(cfg_path: Path, env: Path, read_cache: Path) -> None:
    from flypac.core.config import load_config

    cfg = load_config(cfg_path)
    pcfg = load_config(ROOT / cfg["projection_config"])
    acfg = pcfg["anatomy"]
    ec.ENV_PATH = env
    token = ec.read_token()
    if not token:
        raise SystemExit("NEUPRINT_TOKEN 이 .env 에 없습니다.")
    client = ec.connect(token)

    vm = pd.read_parquet(read_cache / "voxel_meta__meta.parquet")
    voxel = json.loads(vm["voxelSize"].iloc[0])
    if not np.allclose(voxel, [acfg["voxel_nm"]] * 3):
        raise SystemExit(f"복셀 크기 {voxel} 가 설정 {acfg['voxel_nm']} 와 다르다. 멈춤.")
    proj = ea.Projection(acfg, float(acfg["voxel_nm"]) / 1000.0)

    with np.load(ROOT / cfg["circuit_raw"], allow_pickle=False) as z:
        pn_ids, kc_ids, mbon_ids = z["pn_body"], z["kc_body"], z["mbon_body"]
    OWN_CACHE.mkdir(parents=True, exist_ok=True)

    # ---- 1. 뇌 전체 형태
    soma_path = OWN_CACHE / "soma_all.parquet"
    if soma_path.exists():
        soma = pd.read_parquet(soma_path)
        print("[캐시] soma_all")
    else:
        print("[조회] 모든 뉴런의 세포체 위치 (뉴런 정보만)", flush=True)
        soma = fetch_soma(client)
        soma["xyz"] = [soma_xyz(v) for v in soma["soma"]]
        soma["x"] = [np.nan if v is None else v[0] for v in soma["xyz"]]
        soma["y"] = [np.nan if v is None else v[1] for v in soma["xyz"]]
        soma["z"] = [np.nan if v is None else v[2] for v in soma["xyz"]]
        soma = soma[["bodyId", "x", "y", "z"]]
        soma.to_parquet(soma_path, index=False)
    has = ~soma[["x", "y", "z"]].isna().any(axis=1).to_numpy()
    ctx_xy = proj(soma.loc[has, ["x", "y", "z"]].to_numpy(dtype=np.float64)).astype(np.float32)
    ctx_meta = {
        "format": "brain_context_v1", "dataset": ec.DATASET, "server": ec.SERVER,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "neuprint_python_version": ec.neuprint_version(),
        "n_neurons": int(len(soma)), "n_with_soma": int(has.sum()), "n_without_soma": int((~has).sum()),
        "projection": {
            "function": "scripts/export_anatomy.py Projection (같은 함수, 같은 설정)",
            "config": cfg["projection_config"], "view": acfg["view"],
            "dropped_axis": acfg["drop_axis"] + " (앞뒤)",
            "u (가로)": f"{'+' if acfg['horizontal_sign'] > 0 else '-'}{acfg['horizontal_axis']} × scale",
            "v (세로, 위 = 등쪽)": f"{'+' if acfg['up_sign'] > 0 else '-'}{acfg['up_axis']} × scale",
            "units": "μm", "scale_um_per_voxel": proj.scale,
        },
        "query": "MATCH (n:Neuron) RETURN n.bodyId, n.somaLocation (뉴런 정보만, 시냅스 조회 없음)",
        "note": NOTE_CONTEXT,
        "hemibrain_is_half_brain": "hemibrain은 초파리 중앙 뇌의 한쪽 절반(주로 오른쪽)이다. 뇌 전체가 아니다.",
        "stored": "세포체 2차원 좌표(xy_um)만. bodyId와 종류는 저장하지 않는다.",
    }
    cpath = ROOT / cfg["output"]["context"]
    cpath.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cpath, xy_um=ctx_xy)
    cpath.with_name(cpath.name[:-4] + ".meta.json").write_text(
        json.dumps(ctx_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"뇌 형태: {cpath.relative_to(ROOT)} 점 {len(ctx_xy)} (세포체 없음 {int((~has).sum())})")

    # ---- 2. 시냅스 점구름
    f_pk = read_cache / "anat_pn_kc__syn.parquet"
    f_km = read_cache / "anat_kc_mbon__syn.parquet"
    pk = pd.read_parquet(f_pk)
    pk = pk[pk["bodyId_pre"].isin(pn_ids) & pk["bodyId_post"].isin(kc_ids)]
    km_old = pd.read_parquet(f_km)
    km_old = km_old[km_old["bodyId_pre"].isin(kc_ids) & km_old["bodyId_post"].isin(mbon_ids)]
    km_path = OWN_CACHE / "kc_mbon_pre_post.parquet"
    if km_path.exists():
        km = pd.read_parquet(km_path)
        print("[캐시] kc_mbon_pre_post")
    else:
        print("[조회] KC→MBON 시냅스 (시냅스 전·후 위치, 기존과 같은 쿼리)", flush=True)
        km = fetch_kc_mbon_full(client, kc_ids, mbon_ids)
        km.to_parquet(km_path, index=False)
    km = km[km["bodyId_pre"].isin(kc_ids) & km["bodyId_post"].isin(mbon_ids)]
    cols_post = ["bodyId_post", "x_post", "y_post", "z_post"]
    same = (len(km) == len(km_old)
            and km[cols_post].sort_values(cols_post).reset_index(drop=True)
            .equals(km_old[cols_post].sort_values(cols_post).reset_index(drop=True)))
    if not same:
        raise SystemExit(f"새로 받은 KC→MBON 표({len(km)}행)가 기존 캐시({len(km_old)}행)의 시냅스 후 위치와 다르다. 멈춤.")

    caps = cfg["clouds"]["caps"]
    seed = int(cfg["clouds"]["seed"])
    out, nums = {}, {}
    specs = [("pn_calyx", pk, "bodyId_pre", ["x_pre", "y_pre", "z_pre"], pn_ids),
             ("kc_calyx", pk, "bodyId_post", ["x_post", "y_post", "z_post"], kc_ids),
             ("kc_lobe", km, "bodyId_pre", ["x_pre", "y_pre", "z_pre"], kc_ids),
             ("mbon_lobe", km, "bodyId_post", ["x_post", "y_post", "z_post"], mbon_ids)]
    reasons = {"pn_calyx": "본 칼릭스 안 PN→원본 KC 시냅스 없음", "kc_calyx": "본 칼릭스 안 원본 PN→KC 시냅스 없음",
               "kc_lobe": "꽃자루·15구획 안 원본 KC→MBON 시냅스 전 위치 없음",
               "mbon_lobe": "꽃자루·15구획 안 원본 KC→MBON 시냅스 후 위치 없음"}
    for name, df, key, cols, ids in specs:
        xy, off, n0 = cloud(df, key, cols, ids, int(caps[name]), seed, KIND_CODE[name], proj)
        out[f"{name}_xy_um"] = xy
        out[f"{name}_offsets"] = off
        out[f"{name}_n_orig"] = n0
        out[f"{name}_body"] = np.asarray(ids, dtype=np.int64)
        out[f"{name}_missing_reason"] = np.asarray([reasons[name] if n == 0 else "" for n in n0], dtype=str)
        nums[name] = {"neurons": int(len(ids)), "no_points": int((n0 == 0).sum()), "points": int(len(xy)),
                      "orig_total": int(n0.sum()), "capped_neurons": int((n0 > int(caps[name])).sum()),
                      "orig_median": float(np.median(n0))}
    opath = ROOT / cfg["output"]["clouds"]
    np.savez_compressed(opath, **out)
    meta = {
        "format": "mb_synapse_clouds_v1", "dataset": ec.DATASET, "server": ec.SERVER,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "neuprint_python_version": ec.neuprint_version(),
        "caps_per_neuron": {k: int(v) for k, v in caps.items()}, "sampling_seed": seed,
        "sampling": "뉴런마다 고유 위치를 값 순서로 정렬한 뒤 default_rng([seed, 종류 번호, bodyId])로 상한 개를 비복원 균일 추출 (추출 뒤 원래 순서 유지)",
        "kind_codes": KIND_CODE,
        "layout": "{종류}_xy_um (N,2) 모든 점, {종류}_offsets (뉴런 수+1) 뉴런 i의 점 = xy[offsets[i]:offsets[i+1]], "
                  "{종류}_n_orig 뉴런별 원래 시냅스 점 수(고유 위치), {종류}_body 회로 뉴런 순서",
        "kinds": {
            "pn_calyx": "PN, 본 칼릭스 CA(R) 안 PN→원본 KC 시냅스의 시냅스 전 위치",
            "kc_calyx": "KC, 본 칼릭스 CA(R) 안 원본 PN→KC 시냅스의 시냅스 후 위치",
            "kc_lobe": "KC, 꽃자루+15구획 안 원본 KC→MBON 시냅스의 시냅스 전 위치 (칼릭스와 구분해 저장)",
            "mbon_lobe": "MBON, 꽃자루+15구획 안 원본 KC→MBON 시냅스의 시냅스 후 위치",
        },
        "point_definition": "뉴런별 3차원 고유 위치(복셀). 한 시냅스 전 위치가 여러 짝을 가져도 한 점. 정면 투영은 앞뒤 축을 버리므로 2차원에서 겹치는 점이 있을 수 있다",
        "neuron_order": "PN, KC, MBON 모두 회로 원본(data/circuit/raw/mb_raw_v1.npz) 순서 = bodyId 오름차순",
        "original_synapse_counts": "{종류}_n_orig 배열(뉴런별). 요약은 stats",
        "stats": nums,
        "projection": ctx_meta["projection"],
        "sources": {
            "anat_pn_kc__syn.parquet": sha256(f_pk), "anat_kc_mbon__syn.parquet(시냅스 후 위치 대조)": sha256(f_km),
            "kc_mbon_pre_post.parquet(새로 받음, 시냅스 전·후)": sha256(km_path),
            "circuit_raw": sha256(ROOT / cfg["circuit_raw"]),
        },
        "kc_mbon_post_check": "새로 받은 표의 시냅스 후 위치가 기존 캐시와 행 단위로 같음",
        "attribution": "hemibrain v1.2.1, Janelia FlyEM. Scheffer et al. 2020, eLife 9:e57443. CC BY 4.0 (관리자 확인 후 확정).",
        "honesty_note": "화면의 빛은 모델의 활동을 실제 해부 위치에 표시한 것이며 실제 초파리의 측정이 아니다 (TRD 19.13).",
        "config_hash": cfg.hash(),
    }
    opath.with_name(opath.name[:-4] + ".meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(nums, ensure_ascii=False, indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--env", type=Path, default=Path.home() / "dev/fly-pacman/.env")
    ap.add_argument("--read-cache", type=Path, default=Path.home() / "dev/fly-pacman/runs/extract/cache")
    a = ap.parse_args(argv)
    ec.ENV_PATH = a.env
    scrub = ec.Scrubber(ec.read_token())
    try:
        run(a.config, a.env, a.read_cache)
        return 0
    except SystemExit as e:
        print(scrub(e), file=sys.stderr)
        return 1
    except Exception as e:
        print(f"실패: {type(e).__name__}: {scrub(e)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
