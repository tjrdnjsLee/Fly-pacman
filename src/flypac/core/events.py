"""사건 규격 (TRD 4.5).

사건 하나 = (종류, 게임, 틱, 초파리 번호, 활성). 초파리가 없는 사건은 fly = -1.
`active`는 엔진이 사건을 만드는 순간의 학습 활성 여부다 (틱 끝의 값이 아니다).
`cause`는 Z_CHANGE의 원인, GAME_OVER의 승자를 담는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

NO_FLY = -1


class Kind(StrEnum):
    CATCH = "CATCH"
    EATEN = "EATEN"
    DOT = "DOT"
    ENERGIZER = "ENERGIZER"
    FRUIT = "FRUIT"
    LIFE_LOST = "LIFE_LOST"
    GAME_OVER = "GAME_OVER"
    POWER_START = "POWER_START"
    POWER_END = "POWER_END"
    Z_CHANGE = "Z_CHANGE"
    ALL_RELEASED = "ALL_RELEASED"


class Winner(StrEnum):
    FLIES = "FLIES"
    PACMAN = "PACMAN"
    CENSORED = "CENSORED"  # 감시 장치 중단


class ZCause(StrEnum):
    POWER_START = "POWER_START"
    POWER_END = "POWER_END"
    REVIVE = "REVIVE"


# 초파리 번호가 필요한 종류 (TRD 4.5 표)
FLY_KINDS = frozenset({Kind.CATCH, Kind.EATEN, Kind.Z_CHANGE, Kind.ALL_RELEASED})


@dataclass(frozen=True)
class Event:
    kind: Kind
    game: int
    tick: int
    fly: int
    active: bool
    cause: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "kind", Kind(self.kind))
        if self.kind in FLY_KINDS:
            if not 0 <= self.fly:
                raise ValueError(f"{self.kind} 사건에는 초파리 번호가 필요하다")
        elif self.fly != NO_FLY:
            raise ValueError(f"{self.kind} 사건은 fly = -1 이어야 한다")
        if self.kind is Kind.GAME_OVER:
            object.__setattr__(self, "cause", Winner(self.cause))
        elif self.kind is Kind.Z_CHANGE:
            object.__setattr__(self, "cause", ZCause(self.cause))
        elif self.cause is not None:
            raise ValueError(f"{self.kind} 사건은 cause를 갖지 않는다")

    @property
    def winner(self) -> Winner | None:
        return self.cause if self.kind is Kind.GAME_OVER else None
