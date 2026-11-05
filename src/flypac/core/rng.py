"""난수 (TRD 13.2): (시드, 판 번호, 용도)별로 SeedSequence에서 나눈다."""

from __future__ import annotations

import numpy as np

# 용도 번호는 한 번 정하면 바꾸지 않는다 (재현성). 새 용도는 뒤에 추가한다.
PURPOSES = {
    "engine": 0,
    "brain": 1,
    "pacman": 2,
    "ghost": 3,
    "synth": 4,
    "replay": 5,
}


def make_rng(seed: int, game_no: int, purpose: str) -> np.random.Generator:
    if purpose not in PURPOSES:
        raise KeyError(f"알 수 없는 난수 용도: {purpose}")
    return np.random.default_rng(np.random.SeedSequence([seed, game_no, PURPOSES[purpose]]))
