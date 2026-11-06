"""재생 로그 쓰기 (TRD 19.4, 19.5). 현재 판을 메모리에 쌓다가 판 끝에 남길 판만 파일로 저장한다."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from flypac.replay import schema as s

_DT = {f.name: f.dtype for f in s.FIELDS}


class ReplayBuffer:
    """한 판의 틱별 정보를 쌓는다. 틱당 add_frame 한 번, 사건은 add_event."""

    def __init__(self, with_phero: bool = False, with_brain: bool = False):
        self.with_phero, self.with_brain = with_phero, with_brain
        self._cols: dict[str, list] = {}
        self._events: list[tuple[int, int, int]] = []

    def __len__(self) -> int:
        return len(self._cols.get("pac_cell", []))

    def add_frame(self, **fields) -> None:
        """필드 이름은 schema.FIELDS의 이름 (사건과 메타데이터 제외). 모양은 틱 하나 분량.

        값은 **복사해서** 쌓는다. 호출한 쪽이 같은 배열을 제자리에서 고쳐도(냄새 장 갱신,
        조종기의 목숨 초기화 등) 이미 쌓은 프레임은 바뀌지 않는다.
        """
        expected = {f.name for f in s.FIELDS if f.group == "base" and f.name not in s.EVENT_FIELDS}
        if self.with_phero:
            expected.add("phero_field")
        if self.with_brain:
            expected |= set(s.BRAIN_FIELDS)
        if set(fields) != expected:
            raise ValueError(f"필드가 맞지 않는다: 빠진 것 {sorted(expected - set(fields))}, "
                             f"넘친 것 {sorted(set(fields) - expected)}")
        for name, value in fields.items():
            self._cols.setdefault(name, []).append(np.array(value, copy=True))

    def add_event(self, tick: int, kind: str, fly: int) -> None:
        self._events.append((int(tick), s.KIND_CODES[str(kind)], int(fly)))

    def arrays(self, meta_json: str) -> dict:
        """형식에 맞는 배열 사전 (검사는 하지 않는다)."""
        if len(self) == 0:
            raise ValueError("프레임이 하나도 없다")
        out = {name: np.stack(cols).astype(_DT[name]) for name, cols in self._cols.items()}
        ev = np.array(self._events, dtype=np.int64).reshape(-1, 3)
        out["ev_tick"] = ev[:, 0].astype(_DT["ev_tick"])
        out["ev_kind"] = ev[:, 1].astype(_DT["ev_kind"])
        out["ev_fly"] = ev[:, 2].astype(_DT["ev_fly"])
        out[s.META_KEY] = np.array(meta_json)
        return out


def replay_filename(condition: str, seed: int, game_no: int) -> str:
    return f"{condition}_s{seed}_g{game_no}.npz"


def write_replay(path: Path, arrays: dict) -> Path:
    """형식을 검사한 뒤 압축 npz로 저장한다 (pickle 없음)."""
    s.validate(arrays)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path
