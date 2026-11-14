"""학습 가중치 저장 (DECISIONS 26.1). 파일 하나 = 칸 하나 · 판 하나: `weights/s<시드>_g<판 5자리>.npz`.

저장 시점은 판을 끝낸 틱의 종료 처리 안이다 (`core/tick.py::_end_of_tick`, flush_learning 뒤 · reset_life 전).
TickCore가 판이 끝난 칸마다 `on_game_end(칸, 판 번호, 조종기)`를 부르고, 저장할지(판 번호가 every_games의 배수이거나
마지막 판) 정하는 것도 여기다. 조종기(뇌)는 `slot_state(b)`로 칸 하나의 상태를 내줄 뿐이다.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

from flypac.core.config import Config

_NAME = re.compile(r"^s(\d+)_g(\d{5})\.npz$")


def weight_filename(seed: int, game_no: int) -> str:
    return f"s{int(seed)}_g{int(game_no):05d}.npz"


def parse_weight_filename(name: str) -> tuple[int, int] | None:
    m = _NAME.match(name)
    return (int(m.group(1)), int(m.group(2))) if m else None


class WeightWriter:
    def __init__(self, directory: Path, *, seeds, every_games: int, n_games: int, run_config_hash: str):
        if int(every_games) < 1:
            raise ValueError("weights.every_games는 1 이상이어야 한다")
        self.dir = Path(directory)
        self.seeds = [int(s) for s in seeds]
        self.every_games, self.n_games = int(every_games), int(n_games)
        self.run_config_hash = str(run_config_hash)
        self.files: list[Path] = []

    @classmethod
    def from_config(cls, cfg, recorder) -> "WeightWriter":
        every = int(cfg.weights.every_games)
        # 실행 설정 해시는 재생 로그 메타의 config_hash와 같은 값이다 (기록기가 계산한다)
        run_hash = getattr(recorder, "config_hash", None) or Config(
            {k: v for k, v in cfg.to_dict().items() if k != "weights"}).hash()
        return cls(Path(recorder.run_dir) / "weights", seeds=cfg.seeds, every_games=every,
                   n_games=int(cfg.n_games), run_config_hash=run_hash)

    def should_write(self, game_no: int) -> bool:
        return game_no % self.every_games == 0 or game_no == self.n_games

    def on_game_end(self, b: int, game_no: int, ghosts) -> None:
        if not self.should_write(game_no):
            return
        st = ghosts.slot_state(b)
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / weight_filename(self.seeds[b], game_no)
        tmp = path.with_name(path.name + ".tmp")
        # k_gain은 본능 세기를 배우는 뇌만 낸다 (DECISIONS 38.3). 없으면 파일은 이전과 바이트 단위로 같다
        extra = {"k_gain": st["k_gain"]} if "k_gain" in st else {}
        with open(tmp, "wb") as fh:
            np.savez(fh, dw_masked=st["dw_masked"], rbar=st["rbar"], seed=np.int64(self.seeds[b]),
                     game_no=np.int64(game_no), condition=np.str_(st["condition"]), format=np.int64(st["format"]),
                     circuit_sha256=np.str_(st["circuit_sha256"]), brain_config_hash=np.str_(st["brain_config_hash"]),
                     run_config_hash=np.str_(self.run_config_hash), **extra)
        os.replace(tmp, path)
        self.files.append(path)


def read_weight_file(path) -> dict:
    """가중치 파일을 사전으로 읽는다 (FlyBrain.load_slot_state에 그대로 넘길 수 있다)."""
    with np.load(Path(path), allow_pickle=False) as z:
        extra = {"k_gain": z["k_gain"]} if "k_gain" in z.files else {}   # 본능 세기를 배운 실행의 파일에만 있다
        return {
            "dw_masked": z["dw_masked"], "rbar": z["rbar"], "seed": int(z["seed"]), "game_no": int(z["game_no"]),
            "condition": str(z["condition"]), "format": int(z["format"]),
            "circuit_sha256": str(z["circuit_sha256"]), "brain_config_hash": str(z["brain_config_hash"]),
            "run_config_hash": str(z["run_config_hash"]), **extra,
        }
