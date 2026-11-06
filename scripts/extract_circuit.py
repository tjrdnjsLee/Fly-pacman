"""회로 추출 스크립트 (TRD 10.1, M1 뇌 작업 지시서 ① 3 · 4절).

단계:
  --stage inventory : 목록 조사 (토큰 필요). runs/extract/inventory_report.md 를 만든다.
  --stage extract   : 원본 추출 (토큰 필요, 캐시가 있으면 캐시만 읽음).
                      --config configs/circuit_extract_v1.yaml
                      data/circuit/raw/mb_raw_v1.npz 와 .meta.json 을 만든다.

실행: uv run --extra extract python scripts/extract_circuit.py --stage ...
토큰은 저장소 루트 .env 의 NEUPRINT_TOKEN 으로만 읽고, 어떤 출력 · 파일에도 남기지 않는다.

서버 조회 결과는 runs/extract/cache/ 에 parquet로 저장하고, 있으면 다시 받지 않는다
(--refresh 로만 다시 받음). runs/ 는 저장소에 올리지 않는다.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"
OUT_DIR = ROOT / "runs" / "extract"
CACHE_DIR = OUT_DIR / "cache"

SERVER = "neuprint.janelia.org"
DATASET = "hemibrain:v1.2.1"

# --- ROI 이름 (2026-11-06 서버에서 확인한 이름. 보고서 1절에 근거를 남긴다) ---
ROI_MAIN_CALYX = "CA(R)"
ROI_ACC_CALYX = ("dACA(R)", "lACA(R)", "vACA(R)")
ROI_PED = "PED(R)"
ROI_COMPARTMENTS = (
    "a1(R)", "a2(R)", "a3(R)",
    "a'1(R)", "a'2(R)", "a'3(R)",
    "b1(R)", "b2(R)",
    "b'1(R)", "b'2(R)",
    "g1(R)", "g2(R)", "g3(R)", "g4(R)", "g5(R)",
)
ROI_LOBES = ("aL(R)", "a'L(R)", "bL(R)", "b'L(R)", "gL(R)")
ROI_LH = "LH(R)"

# --- 종류 이름 패턴 (neuPrint 정규식은 전체 일치) ---
TYPE_RE_ALL = r".*PN.*|KC.*|MBON.*|PAM.*|PPL1.*|APL|DPM"

# 조사용 문턱 (지시서 3절 7번이 정한 값). 회로 선택용 문턱이 아니다.
INVENTORY_THRESHOLDS = (1, 3, 5)
PHEROMONE_GLOMS = ("DA1", "VA1v", "VA1d", "DL3")
QUANTILES = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)

KC_CLASS_ORDER = ("gamma", "apbp", "ab")
KC_CLASS_LABEL = {"gamma": "γ", "apbp": "α′β′", "ab": "αβ"}

PN_CAT_LABEL = {
    "uni": "단일 사구체 (후각 후보)",
    "thermo_hygro": "온습도 (VP, 단일)",
    "multi_vp": "다사구체 + VP 포함",
    "multi": "다사구체",
    "z": "Z_ (SEZ 관련, 판단 필요)",
    "non_al": "더듬이엽 밖 PN (WEDPN, LPN)",
}


# ---------------------------------------------------------------- 토큰 · 접속

def read_token() -> str:
    if not ENV_PATH.exists():
        raise SystemExit(f"{ENV_PATH} 가 없습니다.")
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("NEUPRINT_TOKEN="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


class Scrubber:
    """출력할 문자열에서 토큰을 지운다."""

    def __init__(self, token: str):
        self.token = token

    def __call__(self, text: object) -> str:
        s = str(text)
        return s.replace(self.token, "<TOKEN>") if self.token else s


def connect(token: str):
    from neuprint import Client

    client = Client(SERVER, dataset=DATASET, token=token)
    datasets = client.fetch_datasets()
    names = list(datasets.keys()) if isinstance(datasets, dict) else list(datasets)
    if DATASET not in names:
        raise SystemExit(f"서버에 {DATASET} 가 없습니다. 있는 것: {names}")
    return client


def neuprint_version() -> str:
    from importlib.metadata import version

    try:
        return version("neuprint-python")
    except Exception:
        return "unknown"


# ---------------------------------------------------------------- 캐시

class Cache:
    def __init__(self, refresh: bool):
        self.refresh = refresh
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.meta_path = CACHE_DIR / "meta.json"
        self.meta = (
            json.loads(self.meta_path.read_text(encoding="utf-8"))
            if self.meta_path.exists() else {}
        )

    def get(self, name: str, describe: str, fetch) -> dict[str, pd.DataFrame]:
        entry = self.meta.get(name)
        if entry and not self.refresh:
            paths = {k: CACHE_DIR / f"{name}__{k}.parquet" for k in entry["tables"]}
            if all(p.exists() for p in paths.values()):
                print(f"[캐시] {name}")
                return {k: pd.read_parquet(p) for k, p in paths.items()}
        print(f"[조회] {name} ...", flush=True)
        tables = fetch()
        for k, df in tables.items():
            df.to_parquet(CACHE_DIR / f"{name}__{k}.parquet", index=False)
        self.meta[name] = {
            "query": describe,
            "dataset": DATASET,
            "server": SERVER,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "neuprint_python_version": neuprint_version(),
            "tables": {k: int(len(df)) for k, df in tables.items()},
        }
        self.meta_path.write_text(
            json.dumps(self.meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return tables


# ---------------------------------------------------------------- 조회

def _roi_parents(tree: dict, parent: str | None, out: dict) -> None:
    for name, sub in tree.items():
        clean = name.rstrip("*")
        out[clean] = parent
        if isinstance(sub, dict):
            _roi_parents(sub, clean, out)


def fetch_rois(client) -> dict[str, pd.DataFrame]:
    from neuprint import fetch_all_rois, fetch_primary_rois, fetch_roi_hierarchy

    tree = fetch_roi_hierarchy(include_subprimary=True, mark_primary=False,
                               format="dict", client=client)
    parents: dict[str, str | None] = {}
    _roi_parents(tree, None, parents)
    primary = set(fetch_primary_rois(client=client))
    rows = [{"roi": r, "parent": parents.get(r), "primary": r in primary}
            for r in fetch_all_rois(client=client)]
    return {"rois": pd.DataFrame(rows)}


def fetch_neuron_table(client) -> dict[str, pd.DataFrame]:
    from neuprint import NeuronCriteria as NC, fetch_neurons

    neurons, roi_counts = fetch_neurons(NC(type=TYPE_RE_ALL, regex=True), client=client)
    keep = [c for c in ("bodyId", "type", "instance", "pre", "post", "status",
                        "cropped", "somaLocation") if c in neurons.columns]
    neurons = neurons[keep].copy()
    if "somaLocation" in neurons.columns:
        neurons["somaLocation"] = neurons["somaLocation"].astype(str)
    return {"neurons": neurons, "roi_counts": roi_counts[["bodyId", "roi", "pre", "post"]]}


def fetch_kc_inputs(client) -> dict[str, pd.DataFrame]:
    """KC가 칼릭스(본 · 부속)에서 받는 모든 입력 (주석 불명 조각 포함)."""
    from neuprint import NeuronCriteria as NC, fetch_adjacencies

    rois = [ROI_MAIN_CALYX, *ROI_ACC_CALYX]
    neurons, conn = fetch_adjacencies(
        None, NC(type="KC.*", regex=True), rois=rois, include_nonprimary=True,
        weight_props=["weight"], properties=["type", "instance"], client=client,
    )
    conn = conn[conn["roi"].isin(rois)][["bodyId_pre", "bodyId_post", "roi", "weight"]]
    return {"neurons": neurons[["bodyId", "type", "instance"]], "conn": conn}


def fetch_kc_to_mbon(client) -> dict[str, pd.DataFrame]:
    from neuprint import NeuronCriteria as NC, fetch_adjacencies

    rois = [ROI_PED, *ROI_COMPARTMENTS, *ROI_LOBES]
    neurons, conn = fetch_adjacencies(
        NC(type="KC.*", regex=True), NC(type="MBON.*", regex=True), rois=rois,
        include_nonprimary=True, weight_props=["weight"],
        properties=["type", "instance"], client=client,
    )
    conn = conn[conn["roi"].isin(rois)][["bodyId_pre", "bodyId_post", "roi", "weight"]]
    return {"neurons": neurons[["bodyId", "type", "instance"]], "conn": conn}


# ---------------------------------------------------------------- 분류

def glom_of(t: str) -> str:
    return t.split("_", 1)[0] if "_" in t else t


def lineage_of(t: str) -> str:
    if t.startswith("WEDPN") or t == "LPN" or "_" not in t:
        return "-"
    m = re.match(r"([a-z0-9]*?PN)", t.split("_", 1)[1])
    return m.group(1) if m else "?"


def pn_category(t: str) -> str:
    if t.startswith("WEDPN") or t == "LPN":
        return "non_al"
    if t.startswith(("M_", "MZ_")):
        return "multi"
    if t.startswith("Z_"):
        return "z"
    g = glom_of(t)
    if "+" in g:
        return "multi_vp" if g.startswith("VP") else "multi"
    if g.startswith("VP"):
        return "thermo_hygro"
    return "uni"


def kc_class(t: str) -> str:
    if t.startswith("KCg"):
        return "gamma"
    if t.startswith("KCa'b'"):
        return "apbp"
    if t.startswith("KCab"):
        return "ab"
    return "?"


def side_of(instance: object) -> str:
    s = str(instance)
    if s.endswith("_R") or "_R_" in s or s.endswith("(R)"):
        return "R"
    if s.endswith("_L") or "_L_" in s or s.endswith("(L)"):
        return "L"
    return "?"


# ---------------------------------------------------------------- 표 출력

def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else floatfmt.format(v)
        return str(v)

    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)


def qrow(x: pd.Series) -> dict:
    x = pd.Series(x, dtype=float).dropna()
    if len(x) == 0:
        return {f"q{int(q * 100)}": np.nan for q in QUANTILES}
    return {f"q{int(q * 100)}": float(x.quantile(q)) for q in QUANTILES}


def write_csv(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUT_DIR / name, index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------- 단계 1

def load_data(client_factory, refresh: bool) -> dict:
    """캐시(없으면 서버)에서 표를 읽고 PN · KC 분류와 KC 칼릭스 입력 출처를 붙인다."""
    cache = Cache(refresh)
    client = None

    def lazy(fn):
        def run():
            nonlocal client
            if client is None:
                client = client_factory()
            return fn(client)
        return run

    rois = cache.get("rois", "fetch_all_rois + fetch_roi_hierarchy + fetch_primary_rois",
                     lazy(fetch_rois))["rois"]
    nt = cache.get("neurons", f"fetch_neurons(type =~ '{TYPE_RE_ALL}')",
                   lazy(fetch_neuron_table))
    kin = cache.get("kc_inputs",
                    f"fetch_adjacencies(None -> KC.*, rois={[ROI_MAIN_CALYX, *ROI_ACC_CALYX]}, "
                    "include_nonprimary=True)", lazy(fetch_kc_inputs))
    k2m = cache.get("kc_to_mbon",
                    f"fetch_adjacencies(KC.* -> MBON.*, rois=PED+15구획+5엽, "
                    "include_nonprimary=True)", lazy(fetch_kc_to_mbon))

    neurons = nt["neurons"].copy()
    neurons["type"] = neurons["type"].astype(str)
    neurons["side"] = neurons["instance"].map(side_of)
    roi_counts = nt["roi_counts"]

    is_pn = neurons["type"].str.contains("PN")
    pn = neurons[is_pn].copy()
    pn["glom"] = pn["type"].map(glom_of)
    pn["lineage"] = pn["type"].map(lineage_of)
    pn["category"] = pn["type"].map(pn_category)
    kc = neurons[neurons["type"].str.startswith("KC")].copy()
    kc["kc_class"] = kc["type"].map(kc_class)
    mbon = neurons[neurons["type"].str.startswith("MBON")].copy()
    dan = neurons[neurons["type"].str.match(r"PAM|PPL1")].copy()

    pn_cat = pn.set_index("bodyId")["category"]
    kc_cls = kc.set_index("bodyId")["kc_class"]
    kc_typ = kc.set_index("bodyId")["type"]

    # 입력 출처 분류 (KC 칼릭스 입력)
    up = kin["neurons"].copy()
    up_type = up.set_index("bodyId")["type"]
    conn = kin["conn"].copy()
    conn["pre_type"] = conn["bodyId_pre"].map(up_type)

    def src_of(row_pre, row_type):
        if row_pre in pn_cat.index:
            return pn_cat[row_pre]
        if row_type is None or (isinstance(row_type, float) and np.isnan(row_type)) \
                or str(row_type) in ("", "None", "nan"):
            return "unannotated"
        t = str(row_type)
        if "PN" in t:   # PN 이름인데 neurons 표에 없는 경우 (없어야 정상)
            return "pn_other"
        if t.startswith("KC"):
            return "kc"
        if t in ("APL", "DPM"):
            return t
        return "other_annotated"

    src_cache = {}
    srcs = []
    for b, t in zip(conn["bodyId_pre"].to_numpy(), conn["pre_type"].to_numpy()):
        key = (b, t if isinstance(t, str) else None)
        if key not in src_cache:
            src_cache[key] = src_of(b, key[1])
        srcs.append(src_cache[key])
    conn["src"] = srcs

    ca = conn[conn["roi"] == ROI_MAIN_CALYX]
    pn_cats = list(PN_CAT_LABEL)
    pn_kc_ca = ca[ca["src"].isin(pn_cats) & ca["bodyId_post"].isin(kc["bodyId"])]
    return dict(rois=rois, roi_counts=roi_counts, neurons=neurons, pn=pn, kc=kc, mbon=mbon, dan=dan, pn_cat=pn_cat, kc_cls=kc_cls, kc_typ=kc_typ, kin=kin, k2m=k2m, conn=conn, ca=ca, pn_cats=pn_cats, pn_kc_ca=pn_kc_ca, cache=cache)


def stage_inventory(client_factory, refresh: bool) -> Path:
    d = load_data(client_factory, refresh)
    rois = d["rois"]
    roi_counts = d["roi_counts"]
    neurons = d["neurons"]
    pn = d["pn"]
    kc = d["kc"]
    mbon = d["mbon"]
    dan = d["dan"]
    pn_cat = d["pn_cat"]
    kc_cls = d["kc_cls"]
    kc_typ = d["kc_typ"]
    kin = d["kin"]
    k2m = d["k2m"]
    conn = d["conn"]
    ca = d["ca"]
    pn_cats = d["pn_cats"]
    pn_kc_ca = d["pn_kc_ca"]

    report: list[str] = []
    w = report.append
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    w("# 단계 1: 목록 조사 보고서 (M1 뇌 작업 지시서 ① 3절)\n")
    w(f"- 데이터셋: `{DATASET}` (서버 `{SERVER}`), neuprint-python {neuprint_version()}")
    w(f"- 보고서 생성: {now} (UTC). 조회 시각은 `runs/extract/cache/meta.json`.")
    w("- 시냅스 수는 neuPrint `weight` (ROI별 값). 한 연결의 시냅스 수 = 해당 ROI 안의 값. "
      "서로 겹치는 ROI(엽 ⊃ 구획)는 더하지 않았다.")
    w("- **이 보고서는 문턱값 · 제외 목록을 정하지 않는다.** 비율은 아래 '잠정 정의'로 계산했다.\n")

    # ---- 1. ROI
    w("## 1. ROI 이름 (오른쪽)\n")
    want = [ROI_MAIN_CALYX, *ROI_ACC_CALYX, ROI_PED, *ROI_COMPARTMENTS, *ROI_LOBES, ROI_LH,
            "MB(R)", "MB(+ACA)(R)", "CAN(R)"]
    rr = rois.set_index("roi")
    missing = [r for r in want if r not in rr.index]
    rows = []
    role = {ROI_MAIN_CALYX: "본 칼릭스", ROI_PED: "꽃자루", ROI_LH: "외측뿔 (모델 항, 뉴런 아님)",
            "MB(R)": "버섯체 (부속 칼릭스 제외)", "MB(+ACA)(R)": "버섯체 + 부속 칼릭스",
            "CAN(R)": "칸틀 (칼릭스 아님, 참고)"}
    role.update({r: "부속 칼릭스" for r in ROI_ACC_CALYX})
    role.update({r: "엽 구획" for r in ROI_COMPARTMENTS})
    role.update({r: "엽 (구획의 상위, 검증용)" for r in ROI_LOBES})
    for r in want:
        if r in rr.index:
            rows.append({"ROI": f"`{r}`", "역할": role.get(r, ""),
                         "상위": f"`{rr.loc[r, 'parent']}`", "primary": bool(rr.loc[r, "primary"])})
    w(md_table(pd.DataFrame(rows)))
    w("")
    w(f"- 엽 구획 {len(ROI_COMPARTMENTS)}개 (지시서의 15개와 일치: α1–3, α′1–3, β1–2, β′1–2, γ1–5)."
      if len(ROI_COMPARTMENTS) == 15 else "- **엽 구획 수가 15가 아니다.**")
    w(f"- 서버에 없는 이름: {missing if missing else '없음'}")
    w("- 부속 칼릭스는 하나가 아니라 `dACA(R)`, `lACA(R)`, `vACA(R)` 세 ROI이고, `MB(R)` 밖 "
      "(`MB(+ACA)(R)` 아래)에 있다. 본 칼릭스 `CA(R)`과 겹치지 않는다.")
    w("- 엽 구획은 primary ROI가 아니다 (상위 엽이 primary). 그래서 조회에 "
      "`include_nonprimary=True`를 썼다.")
    # 구획 합 = 엽 합 검증
    k2 = k2m["conn"]
    tot_comp = int(k2[k2["roi"].isin(ROI_COMPARTMENTS)]["weight"].sum())
    tot_lobe = int(k2[k2["roi"].isin(ROI_LOBES)]["weight"].sum())
    w(f"- 검증: KC→MBON 시냅스의 15구획 합 {tot_comp:,} / 5엽 합 {tot_lobe:,} "
      f"= {tot_comp / max(tot_lobe, 1):.4f} (1에 가까우면 구획이 엽을 거의 덮음).\n")

    # ---- 2. PN
    w("## 2. PN 후보\n")
    w("종류 이름에 `PN`이 들어간 뉴런 전체. 사구체 = 종류 이름의 첫 `_` 앞, 계열 = `_` 뒤의 "
      "`…PN`까지 (예: `adPN`, `lPN`, `vPN`, `l2PN`).\n")
    w("**분류 (자동, 잠정).** 빼는 것은 사람이 정한다.\n")
    pn_out = pn_kc_ca.groupby("bodyId_pre").agg(
        ca_kc_syn=("weight", "sum"), ca_kc_n=("bodyId_post", "nunique"))
    pn = pn.merge(pn_out, left_on="bodyId", right_index=True, how="left")
    pn[["ca_kc_syn", "ca_kc_n"]] = pn[["ca_kc_syn", "ca_kc_n"]].fillna(0).astype(int)
    lh_post = roi_counts[roi_counts["roi"] == ROI_LH].set_index("bodyId")["pre"]
    pn["lh_pre"] = pn["bodyId"].map(lh_post).fillna(0).astype(int)
    rows = []
    for cat in pn_cats:
        s = pn[pn["category"] == cat]
        if len(s) == 0:
            continue
        n_glom = "-" if cat in ("multi", "z", "non_al") else s["glom"].nunique()
        rows.append({"분류": PN_CAT_LABEL[cat], "종류 수": s["type"].nunique(),
                     "사구체 수": n_glom, "뉴런 수": len(s),
                     "칼릭스→KC 시냅스 합": int(s["ca_kc_syn"].sum()),
                     "KC 시냅스 0인 뉴런": int((s["ca_kc_syn"] == 0).sum())})
    w(md_table(pd.DataFrame(rows)))
    w("")
    w("- 분류 규칙: `WEDPN*`, `LPN` → 더듬이엽 밖. `M_`, `MZ_` 접두 또는 사구체 이름에 `+` → 다사구체 "
      "(VP가 들어 있으면 '다사구체 + VP'). `Z_` 접두 → 따로 표시. 사구체가 `VP`로 시작 → 온습도. "
      "나머지 → 단일 사구체 후각 후보.")
    w("- 다음은 규칙상 '단일 사구체'지만 **사람이 확인할 것**: `V` (CO₂ 사구체), `D`. "
      "그리고 `vPN` 계열(문헌상 대부분 GABA성 억제 PN으로 알려짐, **확인 필요**) — 지시서 4.1은 PN을 "
      "흥분성으로 가정한다.")
    w("")
    w("**계열별 (단일 사구체 후각 후보만)**\n")
    u = pn[pn["category"] == "uni"]
    rows = []
    for lin, s in u.groupby("lineage"):
        rows.append({"계열": lin, "사구체 수": s["glom"].nunique(), "뉴런 수": len(s),
                     "칼릭스→KC 시냅스 합": int(s["ca_kc_syn"].sum()),
                     "KC 시냅스 0인 뉴런": int((s["ca_kc_syn"] == 0).sum()),
                     "뉴런당 중앙값": float(s["ca_kc_syn"].median())})
    w(md_table(pd.DataFrame(rows), "{:.1f}"))
    w("")
    w("**뉴런당 칼릭스→KC 시냅스 수 분포 (분류별)**\n")
    rows = []
    for cat in pn_cats:
        s = pn[pn["category"] == cat]["ca_kc_syn"]
        if len(s):
            rows.append({"분류": PN_CAT_LABEL[cat], "n": len(s), **qrow(s)})
    w(md_table(pd.DataFrame(rows), "{:.0f}"))
    w("")
    w("**칼릭스→KC 시냅스가 적은 단일 사구체 후각 후보 PN (θ_PN 참고, 하위 25개)**\n")
    low = u.sort_values("ca_kc_syn").head(25)[
        ["bodyId", "type", "lineage", "side", "ca_kc_syn", "ca_kc_n", "lh_pre"]]
    w(md_table(low))
    w("")
    w("전체 PN 표: `runs/extract/pn_candidates.csv` (뉴런별), `runs/extract/pn_glomeruli.csv` "
      "(사구체별).\n")
    pn_sorted = pn.sort_values("bodyId")
    write_csv(pn_sorted[["bodyId", "type", "instance", "side", "glom", "lineage", "category",
                         "ca_kc_syn", "ca_kc_n", "lh_pre", "pre", "post"]], "pn_candidates.csv")
    gl = pn.groupby(["glom", "category"]).agg(
        n_neurons=("bodyId", "size"), lineages=("lineage", lambda x: ",".join(sorted(set(x)))),
        ca_kc_syn=("ca_kc_syn", "sum"), min_neuron_syn=("ca_kc_syn", "min")).reset_index()
    write_csv(gl.sort_values("glom"), "pn_glomeruli.csv")

    # ---- 3. 페로몬
    w("## 3. 페로몬 후보 사구체\n")
    w("도달 KC 수 = 그 사구체의 PN 중 하나라도 본 칼릭스에서 연결당 시냅스 ≥ 문턱으로 닿는 KC 수 "
      "(연결 = PN 뉴런 하나 → KC 하나).\n")
    rows = []
    reach = {}
    for g in PHEROMONE_GLOMS:
        s = pn[pn["glom"] == g]
        e = pn_kc_ca[pn_kc_ca["bodyId_pre"].isin(s["bodyId"])]
        row = {"사구체": g, "뉴런 수": len(s),
               "계열 (뉴런 수)": ", ".join(f"{k} ({v})" for k, v in s["lineage"].value_counts().sort_index().items()),
               "칼릭스 KC 시냅스": int(e["weight"].sum())}
        for th in INVENTORY_THRESHOLDS:
            n = e[e["weight"] >= th]["bodyId_post"].nunique()
            row[f"도달 KC (≥{th})"] = n
            reach.setdefault(th, {})[g] = n
        rows.append(row)
    w(md_table(pd.DataFrame(rows)))
    w("")
    for th in INVENTORY_THRESHOLDS:
        vals = reach[th]
        lo = min(vals.values())
        ratio = (max(vals.values()) / lo) if lo > 0 else float("inf")
        w(f"- 문턱 {th}: 도달 KC 수 max/min = {ratio:.2f}")
    w("- 참고: 지시서 6.4의 전환 조건 2(max/min > 1.5)는 **확정된 문턱 · 선택 KC 기준**으로 단계 4에서 "
      "판정한다. 위 값은 후보 KC 전체 · 제외 목록 미적용 기준의 미리보기다. vPN을 빼면 값이 바뀐다.\n")
    rows = []
    for g in PHEROMONE_GLOMS:
        for lin, s in pn[pn["glom"] == g].groupby("lineage"):
            e = pn_kc_ca[pn_kc_ca["bodyId_pre"].isin(s["bodyId"])]
            rows.append({"사구체": g, "계열": lin, "뉴런 수": len(s),
                         "칼릭스 KC 시냅스": int(e["weight"].sum()),
                         "도달 KC (≥3)": e[e["weight"] >= 3]["bodyId_post"].nunique()})
    w("**계열별 나눔**\n")
    w(md_table(pd.DataFrame(rows)))
    w("")

    # ---- 4. KC
    w("## 4. KC 후보\n")
    rows = []
    for t, s in kc.groupby("type"):
        rows.append({"종류": t, "KC 종류": KC_CLASS_LABEL.get(kc_class(t), "?"), "뉴런 수": len(s),
                     "오른쪽(R)": int((s["side"] == "R").sum()),
                     "그 외": int((s["side"] != "R").sum())})
    w(md_table(pd.DataFrame(rows)))
    w("")
    rows = [{"KC 종류": KC_CLASS_LABEL[c], "종류": ", ".join(sorted(kc[kc["kc_class"] == c]["type"].unique())),
             "뉴런 수": int((kc["kc_class"] == c).sum())} for c in KC_CLASS_ORDER]
    w("**KC 종류 분류표** (종류 이름 접두사 `KCg` → γ, `KCa'b'` → α′β′, `KCab` → αβ)\n")
    w(md_table(pd.DataFrame(rows)))
    unk = kc[kc["kc_class"] == "?"]
    w(f"\n- 분류 못 한 KC 종류: {sorted(unk['type'].unique()) if len(unk) else '없음'}\n")

    # KC별 본 칼릭스 입력 분해
    kc_ca = ca[ca["bodyId_post"].isin(kc["bodyId"])]
    piv = kc_ca.pivot_table(index="bodyId_post", columns="src", values="weight",
                            aggfunc="sum", fill_value=0)
    for col in [*pn_cats, "unannotated", "APL", "DPM", "kc", "other_annotated", "pn_other"]:
        if col not in piv.columns:
            piv[col] = 0
    piv = piv.reindex(kc["bodyId"], fill_value=0)
    piv["pn_all"] = piv[pn_cats].sum(axis=1)
    piv["pn_al"] = piv["pn_all"] - piv["non_al"]
    piv["total"] = piv[[c for c in piv.columns if c not in ("pn_all", "pn_al")]].sum(axis=1)
    piv["frac_A"] = np.where(piv["pn_all"] > 0, piv["uni"] / piv["pn_all"].where(piv["pn_all"] > 0), np.nan)
    piv["frac_B"] = np.where(piv["pn_al"] > 0, piv["uni"] / piv["pn_al"].where(piv["pn_al"] > 0), np.nan)
    piv["type"] = piv.index.map(kc_typ)
    piv["kc_class"] = piv.index.map(kc_cls)
    acc = conn[conn["roi"].isin(ROI_ACC_CALYX) & conn["bodyId_post"].isin(kc["bodyId"])]
    piv["acc_calyx_in"] = acc.groupby("bodyId_post")["weight"].sum().reindex(piv.index).fillna(0).astype(int)
    piv["acc_calyx_pn"] = acc[acc["src"].isin(pn_cats)].groupby("bodyId_post")["weight"].sum() \
        .reindex(piv.index).fillna(0).astype(int)

    w("**본 칼릭스 `CA(R)` 입력의 출처 (KC 전체 합)**\n")
    tot = piv[[*pn_cats, "APL", "DPM", "kc", "other_annotated", "pn_other", "unannotated"]].sum()
    label = {**PN_CAT_LABEL, "APL": "APL", "DPM": "DPM", "kc": "KC (KC→KC)",
             "other_annotated": "그 밖의 주석 있는 뉴런", "pn_other": "PN 이름이나 PN 표에 없음",
             "unannotated": "주석 불명 (type 없음)"}
    gsum = tot.sum()
    rows = [{"출처": label[k], "시냅스": int(v), "비율": v / gsum} for k, v in tot.items()]
    w(md_table(pd.DataFrame(rows)))
    w("")
    oth = kc_ca[kc_ca["src"] == "other_annotated"].groupby("pre_type")["weight"].sum() \
        .sort_values(ascending=False).head(10)
    w("'그 밖의 주석 있는 뉴런' 상위 10종류: " +
      ", ".join(f"`{k}` {int(v):,}" for k, v in oth.items()) + "\n")
    w("**후각 비율의 잠정 정의.** 분자 = 단일 사구체 후각 후보 PN 입력 (V, D, vPN 포함 — 아직 제외 목록 없음).")
    w("- 정의 A (지시서 4.1 문구 그대로): 분모 = 이름에 PN이 들어간 모든 뉴런의 입력 (WEDPN · LPN 포함).")
    w("- 정의 B: 분모에서 더듬이엽 밖 PN (WEDPN, LPN)을 뺌.")
    w("- 두 정의 모두 APL, DPM, KC, 주석 불명 입력은 분모에 넣지 않는다. "
      "PN 입력이 0인 KC는 비율을 정의하지 않고 따로 센다.\n")
    rows = []
    for t, s in piv.groupby("type"):
        rows.append({"종류": t, "n": len(s), "PN 입력 0": int((s["pn_all"] == 0).sum()),
                     "PN 입력 중앙값": float(s["pn_all"].median()),
                     "주석 불명 중앙값": float(s["unannotated"].median()),
                     "부속 칼릭스 입력 중앙값": float(s["acc_calyx_in"].median()),
                     **{k: v for k, v in qrow(s["frac_A"]).items()}})
    w("**KC 종류별 후각 비율 분포 (정의 A, q = 분위수)**\n")
    w(md_table(pd.DataFrame(rows), "{:.3f}"))
    w("")
    rows = []
    for t, s in piv.groupby("type"):
        rows.append({"종류": t, **qrow(s["frac_B"])})
    w("**KC 종류별 후각 비율 분포 (정의 B)**\n")
    w(md_table(pd.DataFrame(rows), "{:.3f}"))
    same = bool(np.allclose(piv["frac_A"].fillna(-1), piv["frac_B"].fillna(-1)))
    w(f"\n- 정의 A와 B가 모든 KC에서 같은가: {same}"
      + (" (WEDPN · LPN의 본 칼릭스 → KC 시냅스가 0이라 차이가 없다)" if same else ""))
    w("")
    rows = []
    for c in KC_CLASS_ORDER:
        s = piv[piv["kc_class"] == c]
        row = {"KC 종류": KC_CLASS_LABEL[c], "n": len(s), "PN 입력 0": int((s["pn_all"] == 0).sum())}
        for cut in (0.5, 0.7, 0.8, 0.9, 0.95):
            row[f"A ≥ {cut}"] = int((s["frac_A"] >= cut).sum())
        rows.append(row)
    w("**θ_KC 후보별 남는 KC 수 (정의 A)**\n")
    w(md_table(pd.DataFrame(rows)))
    w("")
    rows = []
    for t, s in piv.groupby("type"):
        rows.append({"종류": t, **{f"PN 입력 {lab}": int(v) for lab, v in
                     (("< 5", (s["pn_all"] < 5).sum()), ("< 20", (s["pn_all"] < 20).sum()))},
                     "온습도+VP다사구체 비율 중앙값": float(((s["thermo_hygro"] + s["multi_vp"]) /
                                                     s["pn_all"].where(s["pn_all"] > 0)).median()),
                     "다사구체 비율 중앙값": float((s["multi"] / s["pn_all"].where(s["pn_all"] > 0)).median())})
    w("**PN 입력량이 아주 적은 KC (비율만으로 통과할 위험, 지시서 6.5)**\n")
    w(md_table(pd.DataFrame(rows), "{:.3f}"))
    w("")
    w("KC별 표: `runs/extract/kc_candidates.csv`.\n")
    kc_out = piv.reset_index().rename(columns={"bodyId_post": "bodyId"})
    kc_out["instance"] = kc_out["bodyId"].map(kc.set_index("bodyId")["instance"])
    write_csv(kc_out[["bodyId", "type", "instance", "kc_class", "total", "pn_all", "pn_al", *pn_cats,
                      "APL", "DPM", "kc", "other_annotated", "unannotated", "frac_A", "frac_B",
                      "acc_calyx_in", "acc_calyx_pn"]].sort_values("bodyId"), "kc_candidates.csv")

    # ---- 5. MBON
    w("## 5. MBON 후보\n")
    mb_rois = [ROI_PED, *ROI_COMPARTMENTS]
    e = k2[k2["roi"].isin(mb_rois)].copy()
    e["kc_class"] = e["bodyId_pre"].map(kc_cls)
    per = e.groupby("bodyId_post")["weight"].sum()
    bycls = e.pivot_table(index="bodyId_post", columns="kc_class", values="weight",
                          aggfunc="sum", fill_value=0)
    byroi = e.pivot_table(index="bodyId_post", columns="roi", values="weight",
                          aggfunc="sum", fill_value=0)
    mb = mbon.set_index("bodyId")
    mb["kc_syn"] = per.reindex(mb.index).fillna(0).astype(int)
    for c in KC_CLASS_ORDER:
        mb[f"frac_{c}"] = (bycls[c] if c in bycls.columns else pd.Series(dtype=float)) \
            .reindex(mb.index).fillna(0) / mb["kc_syn"].where(mb["kc_syn"] > 0)
    mb["kc_n"] = e.groupby("bodyId_post")["bodyId_pre"].nunique().reindex(mb.index).fillna(0).astype(int)
    mb["main_roi"] = byroi.idxmax(axis=1).reindex(mb.index)
    mb["rois"] = byroi.apply(lambda r: ",".join(f"{k}:{int(v)}" for k, v in r[r > 0]
                                                .sort_values(ascending=False).items()), axis=1) \
        .reindex(mb.index).fillna("")
    mb = mb.reset_index().sort_values(["kc_syn"], ascending=False)
    w(f"- MBON 뉴런 {len(mb)}개 ({mb['type'].nunique()}종류). 오른쪽(R) {int((mb['side'] == 'R').sum())}, "
      f"그 외 {int((mb['side'] != 'R').sum())}. 기존 캐시(fly-connectome, 2026-10-25)는 69개.")
    w(f"- KC 입력이 0인 MBON: {int((mb['kc_syn'] == 0).sum())}개.")
    w("- `MBON15-like`, `MBON17-like`도 이름 규칙상 포함했다 (판단 필요).")
    w("- 지시서 4.1의 5번대로 KC→MBON은 꽃자루 + 15구획만 셌다. 칼릭스 MBON인 `MBON22(calyx)`의 "
      "칼릭스 입력은 이 수에 들어가지 않는다.\n")
    cols = ["bodyId", "type", "instance", "kc_syn", "kc_n", "frac_gamma", "frac_apbp", "frac_ab", "main_roi"]
    show = mb[cols].rename(columns={"kc_syn": "KC 시냅스", "kc_n": "KC 수", "frac_gamma": "γ",
                                    "frac_apbp": "α′β′", "frac_ab": "αβ", "main_roi": "주 구획"})
    w(md_table(show, "{:.2f}"))
    w("")
    w("**θ_MBON 후보별 남는 MBON 수 (후보 KC 전체 기준)**\n")
    w(md_table(pd.DataFrame([{f"≥ {c}": int((mb["kc_syn"] >= c).sum())
                              for c in (1, 10, 50, 100, 200, 500, 1000)}])))
    w("\nMBON 표 (구획별 시냅스 포함): `runs/extract/mbon_candidates.csv`.\n")
    write_csv(mb.sort_values("bodyId")[[*cols, "rois", "pre", "post"]]
              , "mbon_candidates.csv")

    # ---- 6. DAN
    w("## 6. DAN 후보 (해부 자료 전용)\n")
    rc = roi_counts[roi_counts["roi"].isin(ROI_COMPARTMENTS) & roi_counts["bodyId"].isin(dan["bodyId"])]
    dpiv = rc.pivot_table(index="bodyId", columns="roi", values="pre", aggfunc="sum", fill_value=0)
    dpiv = dpiv.reindex(dan["bodyId"], fill_value=0)
    d = dan.set_index("bodyId")
    d["family"] = d["type"].map(lambda t: "PAM" if t.startswith("PAM") else "PPL1")
    d["comp_pre"] = dpiv.sum(axis=1)
    d["main_comp"] = dpiv.idxmax(axis=1).where(d["comp_pre"] > 0)
    d["main_share"] = dpiv.max(axis=1) / d["comp_pre"].where(d["comp_pre"] > 0)
    rows = []
    for r in ROI_COMPARTMENTS:
        s = d[d["main_comp"] == r]
        rows.append({"구획": r, "PAM (주 구획)": int((s["family"] == "PAM").sum()),
                     "PPL1 (주 구획)": int((s["family"] == "PPL1").sum()),
                     "시냅스 전 위치가 하나라도 있는 DAN": int((dpiv[r] > 0).sum()) if r in dpiv.columns else 0,
                     "종류": ", ".join(sorted(s["type"].unique()))})
    w("주 구획 = 15구획 중 시냅스 전 위치(pre)가 가장 많은 구획.\n")
    w(md_table(pd.DataFrame(rows)))
    w("")
    w(f"- DAN 뉴런 {len(d)}개 (PAM {int((d['family'] == 'PAM').sum())}, "
      f"PPL1 {int((d['family'] == 'PPL1').sum())}). 오른쪽(R) {int((d['side'] == 'R').sum())}.")
    w(f"- 구획에 시냅스 전 위치가 없는 DAN: {int((d['comp_pre'] == 0).sum())}개.")
    w(f"- 주 구획 집중도(주 구획 pre / 15구획 pre) 중앙값 {d['main_share'].median():.2f}, "
      f"0.5 미만 {int((d['main_share'] < 0.5).sum())}개.")
    w("\nDAN 표: `runs/extract/dan_candidates.csv`.\n")
    d2 = d.reset_index()
    write_csv(d2.sort_values("bodyId")[["bodyId", "type", "instance", "side", "family", "main_comp",
                                        "main_share", "comp_pre"]], "dan_candidates.csv")

    # ---- 7. 연결 문턱
    w("## 7. 연결 문턱 후보\n")
    w("연결 = 뉴런 쌍 하나. PN→KC는 본 칼릭스 `CA(R)` 안의 시냅스 수, KC→MBON은 꽃자루 + 15구획 합.\n")

    def thr_rows(name, wts: pd.Series):
        r = {"연결": name, "연결 수": len(wts), "시냅스 합": int(wts.sum())}
        for th in INVENTORY_THRESHOLDS:
            keep = wts >= th
            r[f"≥{th} 연결 비율"] = float(keep.mean())
            r[f"≥{th} 시냅스 비율"] = float(wts[keep].sum() / wts.sum())
        return r

    pk_all = pn_kc_ca.groupby(["bodyId_pre", "bodyId_post"])["weight"].sum()
    pk_uni = pn_kc_ca[pn_kc_ca["src"] == "uni"].groupby(["bodyId_pre", "bodyId_post"])["weight"].sum()
    km = e.groupby(["bodyId_pre", "bodyId_post"])["weight"].sum()
    w(md_table(pd.DataFrame([thr_rows("PN→KC (PN 전체)", pk_all),
                             thr_rows("PN→KC (단일 사구체 후각 후보)", pk_uni),
                             thr_rows("KC→MBON", km)])))
    w("")
    bins = [0, 1, 2, 3, 4, 5, 10, 20, 50, np.inf]
    labels = ["1", "2", "3", "4", "5", "6–10", "11–20", "21–50", ">50"]
    rows = []
    for name, wts in (("PN→KC (PN 전체)", pk_all), ("KC→MBON", km)):
        h = pd.cut(wts, bins=bins, labels=labels).value_counts().reindex(labels).fillna(0).astype(int)
        rows.append({"연결": name, **h.to_dict()})
    w("**연결당 시냅스 수 히스토그램 (연결 수)**\n")
    w(md_table(pd.DataFrame(rows)))
    w("")
    w("## 사람이 정할 것 (지시서 3절 표)\n")
    w("- s_min, θ_PN, θ_KC, θ_MBON, 제외 목록. 이 보고서는 어느 것도 정하지 않았다.")
    w("- 정할 때 함께 정해야 하는 해석 문제 (지시서에 없는 것):")
    w("  1. 후각 비율 분모에 WEDPN · LPN을 넣는가 (정의 A/B).")
    w("  2. `vPN` 계열(억제성 추정)을 후보에서 빼는가. 페로몬 사구체 DA1, VA1v, VA1d에 vPN이 있다.")
    w("  3. `V`(CO₂), `D`, `Z_` 계열을 후각 후보로 보는가.")
    w("  4. KC 후보에 KCg-d, KCab-p처럼 본 칼릭스 PN 입력이 적은 종류를 넣는가 (θ_KC로 자동으로 걸러지는지 위 표 참고).")
    w("  5. KC · PN의 왼쪽(_L) 뉴런 처리. 연결은 오른쪽 ROI 안에서만 셌다.")
    w("  6. `MBON15-like`, `MBON17-like` 포함 여부.")
    w("")

    path = OUT_DIR / "inventory_report.md"
    path.write_text("\n".join(report) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------- 단계 2

def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_config_rois(cfg) -> None:
    """설정의 ROI 이름이 이 스크립트가 조회한 이름과 같은지 확인한다 (캐시와 어긋나지 않게)."""
    r = cfg["rois"]
    pairs = [(r["main_calyx"], ROI_MAIN_CALYX), (tuple(r["accessory_calyx"]), ROI_ACC_CALYX),
             (r["peduncle"], ROI_PED), (tuple(r["compartments"]), ROI_COMPARTMENTS),
             (tuple(r["lobes"]), ROI_LOBES), (r["lateral_horn"], ROI_LH)]
    for got, want in pairs:
        if got != want:
            raise SystemExit(f"설정 ROI {got} 가 조회 ROI {want} 와 다르다.")
    if cfg["dataset"] != DATASET or cfg["server"] != SERVER:
        raise SystemExit("설정의 데이터셋 · 서버가 스크립트와 다르다.")


def select_circuit(d: dict, cfg) -> dict:
    """지시서 4.1: 주석으로 정의, 연결로 선별. 한 번 통과, 반복 제거 없음."""
    th = cfg["thresholds"]
    ex = cfg["exclude"]
    pn, kc, mbon = d["pn"], d["kc"], d["mbon"]
    ca = d["ca"]
    k2 = d["k2m"]["conn"]

    # 1. 주석으로 후보군
    cand_pn = pn[(pn["category"] == "uni") & ~pn["lineage"].isin(list(ex["pn_lineages"]))
                 & ~pn["category"].isin(list(ex["pn_categories"]))]
    cand_kc = kc
    cand_mbon = mbon
    excluded = pn[~pn["bodyId"].isin(cand_pn["bodyId"])]

    # 2. PN 선별: 본 칼릭스에서 후보 KC로 가는 시냅스 합 ≥ θ_PN
    ca_kc = ca[ca["bodyId_post"].isin(cand_kc["bodyId"])]
    pn_out = ca_kc[ca_kc["bodyId_pre"].isin(cand_pn["bodyId"])].groupby("bodyId_pre")["weight"].sum()
    pn_out = pn_out.reindex(cand_pn["bodyId"]).fillna(0)
    sel_pn_ids = np.sort(pn_out[pn_out >= th["theta_pn"]].index.to_numpy())

    # 3. KC 선별. 분모 = 본 칼릭스의 PN 계열(이름에 PN) 입력 전체 (배제한 PN 포함, 정의 A)
    pn_family = ca_kc[ca_kc["bodyId_pre"].isin(pn["bodyId"])]
    den = pn_family.groupby("bodyId_post")["weight"].sum().reindex(cand_kc["bodyId"]).fillna(0)
    num = pn_family[pn_family["bodyId_pre"].isin(sel_pn_ids)].groupby("bodyId_post")["weight"].sum() \
        .reindex(cand_kc["bodyId"]).fillna(0)
    frac = (num / den.where(den > 0))
    kc_keep = (den > 0) & (frac >= th["theta_kc"]) & (num >= th["kc_min_selected_pn_input"])
    sel_kc_ids = np.sort(kc_keep[kc_keep].index.to_numpy())

    # 4. MBON 선별: 선택 KC에서 받는 시냅스 (꽃자루 + 15구획) ≥ θ_MBON
    mb_rois = [ROI_PED, *ROI_COMPARTMENTS]
    e = k2[k2["roi"].isin(mb_rois)]
    e_sel = e[e["bodyId_pre"].isin(sel_kc_ids)]
    m_in = e_sel.groupby("bodyId_post")["weight"].sum().reindex(cand_mbon["bodyId"]).fillna(0)
    sel_mbon_ids = np.sort(m_in[m_in >= th["theta_mbon"]].index.to_numpy())

    # 5. 시냅스 계수: PN→KC 본 칼릭스, KC→MBON 꽃자루 + 15구획
    pk = ca_kc[ca_kc["bodyId_pre"].isin(sel_pn_ids) & ca_kc["bodyId_post"].isin(sel_kc_ids)] \
        .groupby(["bodyId_post", "bodyId_pre"])["weight"].sum()
    km = e_sel[e_sel["bodyId_post"].isin(sel_mbon_ids)].groupby(["bodyId_pre", "bodyId_post"])["weight"].sum()
    kc_idx = {b: i for i, b in enumerate(sel_kc_ids)}
    pn_idx = {b: i for i, b in enumerate(sel_pn_ids)}
    mb_idx = {b: i for i, b in enumerate(sel_mbon_ids)}
    J_raw = np.zeros((len(sel_kc_ids), len(sel_pn_ids)), dtype=np.int64)
    for (k, p_), wgt in pk.items():
        J_raw[kc_idx[k], pn_idx[p_]] = wgt
    W_raw = np.zeros((len(sel_kc_ids), len(sel_mbon_ids)), dtype=np.int64)
    for (k, m), wgt in km.items():
        W_raw[kc_idx[k], mb_idx[m]] = wgt

    # 6. s_min 미만 연결은 0
    s_min = th["s_min"]
    J = np.where(J_raw >= s_min, J_raw, 0).astype(np.int32)
    W0 = np.where(W_raw >= s_min, W_raw, 0).astype(np.int32)

    # MBON별 KC 종류 입력 비율: 모든 후보 KC 기준, 문턱 적용 전 (전체 보존)
    kc_cls = d["kc_cls"]
    prof = np.zeros((len(sel_mbon_ids), 3), dtype=np.float64)
    ee = e[e["bodyId_post"].isin(sel_mbon_ids)].copy()
    ee["cls"] = ee["bodyId_pre"].map(kc_cls)
    tab = ee.groupby(["bodyId_post", "cls"])["weight"].sum()
    for (m, c), wgt in tab.items():
        prof[mb_idx[m], KC_CLASS_ORDER.index(c)] = wgt
    prof = prof / prof.sum(axis=1, keepdims=True)

    pn_t = pn.set_index("bodyId")
    kc_t = kc.set_index("bodyId")
    mb_t = mbon.set_index("bodyId")
    pn_glom = pn_t.loc[sel_pn_ids, "glom"].to_numpy().astype(str)

    # 주석 불명 입력 (본 칼릭스, KC가 받는 것)
    unann = ca_kc[ca_kc["src"] == "unannotated"]
    stats = {
        "candidates": {"pn": int(len(cand_pn)), "kc": int(len(cand_kc)), "mbon": int(len(cand_mbon)),
                       "pn_excluded_by_annotation": int(len(excluded))},
        "selected": {"pn": int(len(sel_pn_ids)), "kc": int(len(sel_kc_ids)),
                     "mbon": int(len(sel_mbon_ids)), "glomeruli": int(len(np.unique(pn_glom)))},
        "kc_fail": {"no_pn_family_input": int((den == 0).sum()),
                    "ratio_below_theta_kc": int(((den > 0) & (frac < th["theta_kc"])).sum()),
                    "selected_input_below_min": int(((den > 0) & (frac >= th["theta_kc"])
                                                     & (num < th["kc_min_selected_pn_input"])).sum())},
        "unannotated_input_main_calyx": {
            "to_all_candidate_kc": int(unann["weight"].sum()),
            "to_selected_kc": int(unann[unann["bodyId_post"].isin(sel_kc_ids)]["weight"].sum()),
            "pn_family_to_selected_kc": int(pn_family[pn_family["bodyId_post"].isin(sel_kc_ids)]["weight"].sum()),
        },
        "s_min_effect": {
            "pn_kc_connections_before": int((J_raw > 0).sum()), "pn_kc_connections_after": int((J > 0).sum()),
            "pn_kc_synapses_before": int(J_raw.sum()), "pn_kc_synapses_after": int(J.sum()),
            "kc_mbon_connections_before": int((W_raw > 0).sum()), "kc_mbon_connections_after": int((W0 > 0).sum()),
            "kc_mbon_synapses_before": int(W_raw.sum()), "kc_mbon_synapses_after": int(W0.sum()),
        },
        "after_s_min": {
            "pn_without_kc": int(((J > 0).sum(axis=0) == 0).sum()),
            "kc_without_pn": int(((J > 0).sum(axis=1) == 0).sum()),
            "kc_without_mbon": int(((W0 > 0).sum(axis=1) == 0).sum()),
            "mbon_without_kc": int(((W0 > 0).sum(axis=0) == 0).sum()),
        },
    }
    arrays = {
        "pn_body": sel_pn_ids.astype(np.int64),
        "pn_type": pn_t.loc[sel_pn_ids, "type"].to_numpy().astype(str),
        "pn_glom": pn_glom,
        "kc_body": sel_kc_ids.astype(np.int64),
        "kc_type": kc_t.loc[sel_kc_ids, "type"].to_numpy().astype(str),
        "kc_class": kc_t.loc[sel_kc_ids, "kc_class"].to_numpy().astype(str),
        "mbon_body": sel_mbon_ids.astype(np.int64),
        "mbon_type": mb_t.loc[sel_mbon_ids, "type"].to_numpy().astype(str),
        "J_pn": J,
        "W0": W0,
        "mbon_class_profile": prof,
        "glom_names": np.unique(pn_glom).astype(str),
    }
    tables = {"pn_out": pn_out, "kc_num": num, "kc_den": den, "kc_frac": frac, "mbon_in": m_in,
              "excluded": excluded, "cand_pn": cand_pn}
    return {"arrays": arrays, "stats": stats, "tables": tables}


def write_snapshot(d: dict, snap_dir: Path) -> dict[str, str]:
    """원본 조회 결과(인접 표)를 정렬된 CSV로 저장하고 SHA-256을 돌려준다."""
    snap_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    tables = {
        "neurons.csv": d["neurons"].drop(columns=["side"]).sort_values("bodyId"),
        "neuron_roi_counts.csv": d["roi_counts"].sort_values(["bodyId", "roi"]),
        "kc_inputs_conn.csv": d["kin"]["conn"].sort_values(["bodyId_pre", "bodyId_post", "roi"]),
        "kc_inputs_neurons.csv": d["kin"]["neurons"].sort_values("bodyId"),
        "kc_to_mbon_conn.csv": d["k2m"]["conn"].sort_values(["bodyId_pre", "bodyId_post", "roi"]),
        "rois.csv": d["rois"].sort_values("roi"),
    }
    for name, df in tables.items():
        path = snap_dir / name
        df.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")
        out[name] = sha256_file(path)
    return out


def stage_extract(client_factory, refresh: bool, cfg_path: Path) -> Path:
    from flypac.core.config import load_config

    cfg = load_config(cfg_path)
    check_config_rois(cfg)
    d = load_data(client_factory, refresh)
    res = select_circuit(d, cfg)
    arrays, stats = res["arrays"], res["stats"]

    raw_path = ROOT / cfg["output"]["raw"]
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(raw_path, **arrays)
    snap = write_snapshot(d, ROOT / cfg["output"]["snapshot_dir"])

    th = cfg["thresholds"]
    kc_types = pd.Series(arrays["kc_type"]).value_counts().sort_index()
    meta = {
        "format": "mb_raw_v1",
        "dataset": DATASET,
        "server": SERVER,
        "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fetched_at": {k: v["fetched_at"] for k, v in d["cache"].meta.items()},
        "neuprint_python_version": neuprint_version(),
        "rois": {"main_calyx": ROI_MAIN_CALYX, "accessory_calyx": list(ROI_ACC_CALYX),
                 "peduncle": ROI_PED, "compartments": list(ROI_COMPARTMENTS),
                 "lobes_for_check": list(ROI_LOBES), "lateral_horn": ROI_LH},
        "synapse_count_rois": {"J_pn": [ROI_MAIN_CALYX], "W0": [ROI_PED, *ROI_COMPARTMENTS]},
        "thresholds": dict(th),
        "exclude": {"pn_lineages": list(cfg["exclude"]["pn_lineages"]),
                    "pn_categories": list(cfg["exclude"]["pn_categories"]),
                    "category_rules": {
                        "non_al": "type이 WEDPN으로 시작하거나 LPN",
                        "multi": "type이 M_ 또는 MZ_로 시작하거나 사구체 이름에 + (VP 없음)",
                        "multi_vp": "사구체 이름에 +, VP로 시작",
                        "z": "type이 Z_로 시작",
                        "thermo_hygro": "사구체가 VP로 시작 (단일)",
                        "uni": "그 밖의 단일 사구체 (V, D 포함)"}},
        "kc_ratio_definition": "본 칼릭스에서 선택 PN 입력 / 본 칼릭스에서 이름에 PN이 들어간 모든 뉴런의 입력 "
                               "(배제한 PN 포함, APL · DPM · KC · 주석 불명 제외)",
        "mbon_class_profile": {"columns": ["gamma", "apbp", "ab"],
                               "basis": "모든 후보 KC에서 받는 시냅스 (꽃자루 + 15구획), 문턱 적용 전"},
        "kc_class_labels": {"gamma": "γ (KCg-*)", "apbp": "α′β′ (KCa'b'-*)", "ab": "αβ (KCab-*)"},
        "counts": stats,
        "kc_types_selected": {k: int(v) for k, v in kc_types.items()},
        "sort_rule": "모든 뉴런 배열은 bodyId 오름차순. 이 순서가 회로 순서다. glom_names는 문자열 정렬.",
        "snapshot_sha256": snap,
        "snapshot_dir": cfg["output"]["snapshot_dir"] + " (저장소에 올리지 않음)",
        "raw_sha256": sha256_file(raw_path),
        "attribution": "hemibrain v1.2.1, Janelia FlyEM. Scheffer et al. 2020, eLife 9:e57443, "
                       "\"A connectome and analysis of the adult Drosophila central brain\". CC BY 4.0 "
                       "(관리자 확인 후 확정).",
        "config": str(cfg_path.relative_to(ROOT) if cfg_path.is_absolute() else cfg_path),
        "config_hash": cfg.hash(),
    }
    meta_path = raw_path.with_name(raw_path.name[:-4] + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # 보고서 (저장소에 올리지 않음)
    t = res["tables"]
    lines = ["# 단계 2: 원본 추출 보고서\n",
             f"- 설정: `{meta['config']}` (해시 {meta['config_hash']})",
             f"- 출력: `{cfg['output']['raw']}` (SHA-256 {meta['raw_sha256'][:16]}…)\n",
             "## 단계별 남은 뉴런 수\n",
             "| 단계 | PN | KC | MBON |", "|---|---|---|---|",
             f"| 이름 규칙 전체 | {len(d['pn'])} | {len(d['kc'])} | {len(d['mbon'])} |",
             f"| 1. 주석 후보 (제외 목록 적용) | {stats['candidates']['pn']} | {stats['candidates']['kc']} | {stats['candidates']['mbon']} |",
             f"| 2. PN ≥ θ_PN={th['theta_pn']} | {stats['selected']['pn']} | | |",
             f"| 3. KC 비율 ≥ {th['theta_kc']} · 입력 ≥ {th['kc_min_selected_pn_input']} | | {stats['selected']['kc']} | |",
             f"| 4. MBON ≥ θ_MBON={th['theta_mbon']} | | | {stats['selected']['mbon']} |",
             "", f"- 선택 PN의 사구체 수: {stats['selected']['glomeruli']}",
             f"- KC 탈락 이유: PN 계열 입력 0 {stats['kc_fail']['no_pn_family_input']}, "
             f"비율 미달 {stats['kc_fail']['ratio_below_theta_kc']}, "
             f"비율은 통과했으나 입력 < {th['kc_min_selected_pn_input']} {stats['kc_fail']['selected_input_below_min']}",
             "", "## 지정 KC 종류의 남은 수\n", "| 종류 | 후보 | 남음 |", "|---|---|---|"]
    all_types = d["kc"]["type"].value_counts()
    for typ in ["KCg-d", "KCab-p", "KCg-s1", "KCg-s2", "KCg-s3", "KCg-s4"]:
        lines.append(f"| {typ} | {int(all_types.get(typ, 0))} | {int(kc_types.get(typ, 0))} |")
    lines += ["", "## KC 종류별 남은 수 (전체)\n", "| 종류 | 후보 | 남음 |", "|---|---|---|"]
    for typ, n in all_types.sort_index().items():
        lines.append(f"| {typ} | {int(n)} | {int(kc_types.get(typ, 0))} |")
    sel_pn = set(arrays["pn_body"].tolist())
    cp = t["cand_pn"].copy()
    cp["out"] = cp["bodyId"].map(t["pn_out"])
    failed = cp[~cp["bodyId"].isin(sel_pn)].sort_values("out")
    lines += ["", f"## θ_PN에서 탈락한 후보 PN ({len(failed)}개)\n", "| bodyId | type | side | 후보 KC 시냅스 |", "|---|---|---|---|"]
    for r in failed.itertuples():
        lines.append(f"| {r.bodyId} | {r.type} | {r.side} | {int(r.out)} |")
    lines += ["", "## 페로몬 후보 사구체의 선택 결과\n", "| 사구체 | 선택 PN | 선택 PN 종류 |", "|---|---|---|"]
    for g in cfg["pheromone_glomeruli"]:
        m = arrays["pn_glom"] == g
        lines.append(f"| {g} | {int(m.sum())} | {', '.join(sorted(set(arrays['pn_type'][m])))} |")
    lines += ["", "## s_min 적용 효과와 남은 고립 뉴런\n", "```", json.dumps(stats["s_min_effect"], indent=2),
              json.dumps(stats["after_s_min"], indent=2), "```",
              "", "## 주석 불명 입력 (본 칼릭스, KC가 받는 것)\n", "```",
              json.dumps(stats["unannotated_input_main_calyx"], indent=2), "```"]
    mbin = t["mbon_in"]
    mt = d["mbon"].set_index("bodyId")
    lines += ["", "## MBON 선별 (선택 KC에서 받는 시냅스)\n", "| bodyId | type | instance | 시냅스 | 선택 |", "|---|---|---|---|---|"]
    for b, v in mbin.sort_values(ascending=False).items():
        lines.append(f"| {b} | {mt.loc[b, 'type']} | {mt.loc[b, 'instance']} | {int(v)} | "
                     f"{'예' if v >= th['theta_mbon'] else ''} |")
    rep = OUT_DIR / "extract_report.md"
    rep.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(stats["selected"], ensure_ascii=False))
    return raw_path


# ---------------------------------------------------------------- 진입점

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", required=True, choices=["inventory", "extract"])
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--refresh", action="store_true", help="캐시가 있어도 서버에서 다시 받는다")
    args = ap.parse_args(argv)

    token = read_token()
    scrub = Scrubber(token)
    try:
        OUT_DIR.mkdir(parents=True, exist_ok=True)

        def factory():
            if not token:
                raise SystemExit("NEUPRINT_TOKEN 이 .env 에 없습니다.")
            return connect(token)

        if args.stage == "extract":
            if args.config is None:
                raise SystemExit("--stage extract 에는 --config 가 필요하다.")
            path = stage_extract(factory, args.refresh, args.config)
            print(f"원본: {path.relative_to(ROOT)}, 보고서: runs/extract/extract_report.md")
        else:
            path = stage_inventory(factory, args.refresh)
            print(f"보고서: {path.relative_to(ROOT)}")
        return 0
    except SystemExit as e:
        print(scrub(e), file=sys.stderr)
        return 1
    except Exception as e:  # 토큰이 오류 문구에 섞여도 지운다
        print(f"실패: {type(e).__name__}: {scrub(e)}", file=sys.stderr)
        print(f"토큰 길이: {len(token)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
