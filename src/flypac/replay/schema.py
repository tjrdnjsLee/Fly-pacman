"""재생 로그 형식 (TRD 19.4, 19.13). 판 하나 = npz 파일 하나.

형식 버전 2 = 19.4 기본 필드 + 19.13 뇌 필드(선택 묶음). 형식 변경은 시각화 팀과 합의한다.

모양 표기: T 틱 수, N 걸을 수 있는 칸 수, K 켜진 KC 최대 수, M MBON 수, F = 4 초파리 수.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

FORMAT_VERSION = 2
N_FLIES = 4
N_SENSE = 8  # 냄새 4방향 + 페로몬 4방향 (TRD 19.4)
N_ACTIONS = 4
N_PN = 9  # 냄새 4 + 상태 z 1 + 페로몬 4 (TRD 10.1, 19.13)

META_KEY = "meta"
META_FIELDS = ("format_version", "condition", "seed", "game_no", "git_commit", "config_hash", "map_name")


@dataclass(frozen=True)
class Field:
    name: str
    shape: tuple  # 기호(문자열) 또는 정수
    dtype: str
    group: str  # "base" 필수 | "phero" 체제3만 | "brain" 뇌 필드 묶음(전부 있거나 전부 없음)
    use: str


FIELDS = (
    # --- 19.4 기본 ---
    Field("pac_cell", ("T",), "int16", "base", "팩맨 칸"),
    Field("pac_dir", ("T",), "int8", "base", "팩맨 방향 (시계방향 번호)"),
    Field("pac_acc", ("T",), "float16", "base", "팩맨 속도 누적값 (보간)"),
    Field("fly_cell", ("T", N_FLIES), "int16", "base", "초파리 칸"),
    Field("fly_dir", ("T", N_FLIES), "int8", "base", "초파리 방향"),
    Field("fly_acc", ("T", N_FLIES), "float16", "base", "초파리 속도 누적값 (보간)"),
    Field("fly_z", ("T", N_FLIES), "bool", "base", "상태 z"),
    Field("fly_in_house", ("T", N_FLIES), "bool", "base", "집 안 여부"),
    Field("lives", ("T",), "int8", "base", "목숨"),
    Field("power_ticks", ("T",), "int16", "base", "파워 남은 틱"),
    Field("score", ("T",), "int32", "base", "점수"),
    Field("fruit", ("T",), "int8", "base", "과일 유무"),
    Field("ev_tick", ("E",), "int32", "base", "사건의 틱"),
    Field("ev_kind", ("E",), "<U16", "base", "사건 종류 (core.events.Kind 값)"),
    Field("ev_fly", ("E",), "int8", "base", "사건의 초파리 번호, 없으면 -1"),
    Field("odor_field", ("T", "N"), "float16", "base", "냄새 장"),
    Field("sense", ("T", N_FLIES, N_SENSE), "float16", "base", "감각값: 냄새 4 + 페로몬 4 (자기중심 순서)"),
    Field("action", ("T", N_FLIES), "int8", "base", "결정 (자기중심 행동, 결정 아닌 틱은 -1)"),
    Field("pi", ("T", N_FLIES, N_ACTIONS), "float16", "base", "행동 확률 π"),
    # --- 19.4 체제3만 ---
    Field("phero_field", ("T", "N"), "float16", "phero", "페로몬 장 (체제3만)"),
    # --- 19.13 뇌 필드 (replay.brain 설정으로 켜고 끔). KC 번호는 이 묶음에서 필수 ---
    Field("kc_active", ("T", N_FLIES, "K"), "int16", "brain", "켜진 KC 번호 (모자라는 자리는 -1)"),
    Field("pn_input", ("T", N_FLIES, N_PN), "float16", "brain", "PN 입력, 채널 9묶음"),
    Field("mbon_out", ("T", N_FLIES, "M"), "float16", "brain", "개별 MBON 출력"),
    Field("dan", ("T", N_FLIES), "float16", "brain", "확정 시점의 R_n - R̄, 그 외 틱은 0"),
)

FIELD_BY_NAME = {f.name: f for f in FIELDS}
BRAIN_FIELDS = tuple(f.name for f in FIELDS if f.group == "brain")
BASE_FIELDS = tuple(f.name for f in FIELDS if f.group == "base")
EVENT_FIELDS = ("ev_tick", "ev_kind", "ev_fly")


class SchemaError(ValueError):
    pass


def make_meta(condition: str, seed: int, game_no: int, git_commit: str, config_hash: str, map_name: str) -> str:
    return json.dumps(
        {
            "format_version": FORMAT_VERSION,
            "condition": condition,
            "seed": int(seed),
            "game_no": int(game_no),
            "git_commit": git_commit,
            "config_hash": config_hash,
            "map_name": map_name,
        },
        ensure_ascii=False,
    )


def parse_meta(raw) -> dict:
    meta = json.loads(str(np.asarray(raw)[()]))
    missing = [k for k in META_FIELDS if k not in meta]
    if missing:
        raise SchemaError(f"메타데이터에 없는 항목: {missing}")
    return meta


def validate(arrays: dict) -> dict:
    """arrays(npz에서 읽은 사전 또는 쓸 사전)가 형식을 만족하는지 검사한다. 메타데이터 사전을 돌려준다."""
    if META_KEY not in arrays:
        raise SchemaError("메타데이터가 없다")
    meta = parse_meta(arrays[META_KEY])
    if meta["format_version"] != FORMAT_VERSION:
        raise SchemaError(f"형식 버전 {meta['format_version']} != {FORMAT_VERSION}")

    known = set(FIELD_BY_NAME) | {META_KEY}
    extra = set(arrays) - known
    if extra:
        raise SchemaError(f"알 수 없는 필드: {sorted(extra)}")
    for name in BASE_FIELDS:
        if name not in arrays:
            raise SchemaError(f"필수 필드가 없다: {name}")
    present_brain = [n for n in BRAIN_FIELDS if n in arrays]
    if present_brain and len(present_brain) != len(BRAIN_FIELDS):
        raise SchemaError(f"뇌 필드는 전부 있거나 전부 없어야 한다: 있는 것 {present_brain}")

    dims: dict[str, int] = {}
    for name, arr in arrays.items():
        if name == META_KEY:
            continue
        f = FIELD_BY_NAME[name]
        arr = np.asarray(arr)
        if f.dtype.startswith("<U"):
            if arr.dtype.kind != "U":
                raise SchemaError(f"{name}: 문자열 배열이어야 한다 (현재 {arr.dtype})")
        elif arr.dtype != np.dtype(f.dtype):
            raise SchemaError(f"{name}: 형 {f.dtype} 이어야 한다 (현재 {arr.dtype})")
        if arr.ndim != len(f.shape):
            raise SchemaError(f"{name}: 차원 {len(f.shape)} 이어야 한다 (현재 {arr.ndim})")
        for axis, want in enumerate(f.shape):
            got = arr.shape[axis]
            if isinstance(want, int):
                if got != want:
                    raise SchemaError(f"{name}: 축 {axis} 크기 {want} 이어야 한다 (현재 {got})")
            elif dims.setdefault(want, got) != got:
                raise SchemaError(f"{name}: 기호 {want}={dims[want]} 인데 축 {axis} 크기가 {got}")
    if "E" in dims and any(len(arrays[n]) != dims["E"] for n in EVENT_FIELDS):
        raise SchemaError("사건 필드의 길이가 다르다")
    return meta
