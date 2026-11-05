"""가짜 엔진 (TRD 5.2, M0 지시서 4단계). 자리 파일: 진짜 엔진과 같은 인터페이스를 가진 단순판."""

from __future__ import annotations

from flypac.engine.game import Game, VecGame


class FakeGame(Game):
    """TRD 5.2 `Game`과 같은 인터페이스."""


class FakeVecGame(VecGame):
    """TRD 5.2 `VecGame`과 같은 인터페이스. 묶는 대상만 FakeGame이다."""

    game_cls = FakeGame
