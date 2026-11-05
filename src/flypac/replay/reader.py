"""재생 로그 읽기 + 형식 검사 (TRD 19.4)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from flypac.replay import schema as s


def read_replay(path: Path) -> tuple[dict, dict]:
    """(메타데이터 사전, 배열 사전). 형식이 맞지 않으면 SchemaError."""
    with np.load(Path(path), allow_pickle=False) as z:
        arrays = {name: z[name] for name in z.files}
    meta = s.validate(arrays)
    arrays.pop(s.META_KEY)
    return meta, arrays


def kind_names(ev_kind: np.ndarray) -> list[str]:
    """ev_kind 정수 코드를 종류 이름으로."""
    return [s.CODE_KINDS[int(k)] for k in ev_kind]
