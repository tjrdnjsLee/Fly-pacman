"""난수 (TRD 13.2): (시드, 판 번호, 용도)별로 SeedSequence에서 나눈다. 게임 칸별 묶음은 GameRngs."""

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


class GameRngs:
    """게임 칸별 난수 생성기 묶음 (TRD 13.2). 칸 b의 생성기 = make_rng(seeds[b], 판 번호[b], 용도).

    한 칸의 난수열은 그 칸의 (시드, 판 번호, 용도)로만 정해진다. 그래서 묶음 크기 B나 칸 순서를 바꿔도
    같은 시드 · 같은 판의 난수열은 같다. 판이 바뀌면 실행기가 renew(mask, game_no)를 부른다.
    조종기(무작위 유령, 다음 단계의 뇌)는 rngs[b]만 쓴다.
    """

    def __init__(self, seeds, purpose: str, game_no=1):
        if purpose not in PURPOSES:
            raise KeyError(f"알 수 없는 난수 용도: {purpose}")
        self.seeds = np.asarray(seeds, dtype=np.int64)
        self.purpose = purpose
        self.game_no = np.broadcast_to(np.asarray(game_no, dtype=np.int64), self.seeds.shape).copy()
        self._gens = [make_rng(int(s), int(g), purpose) for s, g in zip(self.seeds, self.game_no)]

    def __len__(self) -> int:
        return len(self._gens)

    def __getitem__(self, b: int) -> np.random.Generator:
        return self._gens[b]

    def renew(self, mask, game_no) -> None:
        """mask가 참인 칸의 생성기를 새 판 번호로 다시 만든다. game_no: (B,) 또는 스칼라."""
        mask = np.asarray(mask, dtype=bool)
        new_no = np.broadcast_to(np.asarray(game_no, dtype=np.int64), self.seeds.shape)
        for b in np.flatnonzero(mask):
            self.game_no[b] = new_no[b]
            self._gens[b] = make_rng(int(self.seeds[b]), int(self.game_no[b]), self.purpose)

    def state_dict(self) -> dict:
        return {"seeds": self.seeds.copy(), "game_no": self.game_no.copy(),
                "states": [g.bit_generator.state for g in self._gens]}

    def load_state_dict(self, state: dict) -> None:
        if not np.array_equal(state["seeds"], self.seeds):
            raise ValueError("시드 목록이 다르다")
        self.game_no = np.asarray(state["game_no"], dtype=np.int64).copy()
        for g, st in zip(self._gens, state["states"]):
            g.bit_generator.state = st


def game_rngs(seeds, purpose: str, game_no=1) -> GameRngs:
    """칸별 (시드, 판 번호, 용도) 생성기 묶음을 만든다. 공용 진입점."""
    return GameRngs(seeds, purpose, game_no)
