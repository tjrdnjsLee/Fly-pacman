"""원작 유령의 공용 설정과 엔진이 소유하는 모드 상태 (TRD 7.3)."""

from dataclasses import dataclass

import numpy as np

from flypac.core.timing import config_ticks
from flypac.core.types import GameView


_B1_KEYS = (
    "force_reversal", "scatter_chase_s", "scatter_targets", "pinky_ahead",
    "inky_ahead", "clyde_chase_distance", "no_up_cells",
)


def b1_settings(cfg):
    section = cfg.engine.get("b1", {})
    missing = [f"engine.b1.{key}" for key in _B1_KEYS if key not in section]
    if missing:
        raise ValueError("B1에 필요한 설정 누락: " + ", ".join(missing))
    if section.force_reversal != "original":
        raise ValueError("조건 b1에는 engine.b1.force_reversal: original이 필요하다")
    return section


@dataclass(frozen=True)
class B1GameView(GameView):
    """원작 조종기가 읽는 엔진의 현재 모드."""

    scatter: bool = True


class B1Motion:
    """일곱 구간 뒤에는 계속 추격한다. 반전 신호는 이동할 때 한 번 적용한다."""

    def __init__(self, cfg, map):
        settings = b1_settings(cfg)
        self.schedule = tuple(config_ticks(cfg, seconds) for seconds in settings.scatter_chase_s)
        if len(self.schedule) != 7 or any(ticks <= 0 for ticks in self.schedule):
            raise ValueError("engine.b1.scatter_chase_s는 양의 틱으로 변환되는 일곱 구간이어야 한다")
        self.no_up = frozenset(int(map.cell_id[tuple(rc)]) for rc in settings.no_up_cells)
        self.reverse = np.zeros(len(map.fly_starts), dtype=bool)
        self.reset_life()

    @property
    def scatter(self) -> bool:
        return self.phase % 2 == 0

    def reset_life(self) -> None:
        self.phase = 0
        self.remaining = self.schedule[0]
        self.reverse.fill(False)

    def queue_reverse(self) -> None:
        self.reverse.fill(True)

    def tick(self) -> None:
        if self.phase == len(self.schedule):
            return
        self.remaining -= 1
        if self.remaining == 0:
            self.phase += 1
            if self.phase < len(self.schedule):
                self.remaining = self.schedule[self.phase]
            self.queue_reverse()
