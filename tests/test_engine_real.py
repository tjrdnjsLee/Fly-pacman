"""진짜 엔진 시험 (engine/game.py, 지시서 묶음 A). 항목이 진행될수록 시험을 더한다.

8번 뼈대: __init__, view, reset_game, reset_life, retire.
아직 이동 함수가 없어서, 상태를 바꿀 때는 속성을 직접 고친다. 그래서 이 파일은 다음 속성 이름을 전제한다
(가짜 엔진과 같다): lives, score, tick, dots, energizers, pac_cell, fly_cell, frozen, learning_active.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from flypac.core.config import load_config
from flypac.core.rng import make_rng
from flypac.engine.fake import FakeGame
from flypac.engine.game import Game
from flypac.maps.original import load_original

SMOKE = Path(__file__).resolve().parents[1] / "configs" / "m0_smoke.yaml"


def _make(cls):
    return cls(load_original(), load_config(SMOKE), make_rng(0, 0, "engine"))


@pytest.fixture
def game():
    return _make(Game)


def _dirty(g):
    """한 판이 어느 정도 진행된 상태로 만든다."""
    cell = int(np.flatnonzero(g.dots)[0])
    g.dots[cell] = False
    g.score = 500
    g.lives = 2
    g.tick = 100
    g.pac_cell = int(g.map.fly_starts[0])
    g.fly_cell[1] = int(g.map.pacman_start)
    g.learning_active = True
    g.frozen = True
    return cell


def test_new_game_view_matches_fake_engine(game):
    """새 게임의 상태는 가짜 엔진과 필드까지 같다. 틀린 필드 이름을 알려 준다."""
    want, got = _make(FakeGame).view(), game.view()
    for f in dataclasses.fields(want):
        a, b = getattr(want, f.name), getattr(got, f.name)
        assert np.array_equal(a, b), f"view().{f.name}: 기대 {a}, 실제 {b}"


def test_game_does_not_change_the_map(game):
    """게임이 점을 지워도 지도의 원본 점 배치는 그대로다."""
    before = game.map.dots.copy()
    game.dots[int(np.flatnonzero(game.dots)[0])] = False
    assert np.array_equal(game.map.dots, before), "dots를 복사하지 않고 지도 배열을 그대로 썼다"


def test_view_is_a_copy(game):
    """지시서 2.7: 내보낸 사본은 엔진 상태가 바뀌어도 그대로다."""
    v = game.view()
    cell, dots = v.fly_cell.copy(), v.dots.copy()
    game.fly_cell[0] = int(game.map.pacman_start)
    game.dots[int(np.flatnonzero(game.dots)[0])] = False
    assert np.array_equal(v.fly_cell, cell), "view().fly_cell이 복사본이 아니다"
    assert np.array_equal(v.dots, dots), "view().dots가 복사본이 아니다"


def test_reset_life_restores_positions_and_keeps_progress(game):
    """목숨 초기화: 위치와 동결은 되돌리고, 점수 · 목숨 · 먹은 점은 유지한다."""
    eaten = _dirty(game)
    game.reset_life()
    v = game.view()
    assert v.pac_cell == game.map.pacman_start
    assert np.array_equal(v.fly_cell, game.map.fly_starts)
    assert not v.frozen and not v.learning_active
    assert v.score == 500 and v.lives == 2
    assert not v.dots[eaten], "목숨을 잃어도 먹은 점은 되살아나지 않는다"


def test_reset_game_restores_everything(game):
    """판 초기화: 새 게임과 같은 상태로 돌아간다."""
    _dirty(game)
    game.reset_game()
    want, got = _make(FakeGame).view(), game.view()
    for f in dataclasses.fields(want):
        a, b = getattr(want, f.name), getattr(got, f.name)
        assert np.array_equal(a, b), f"reset_game 뒤 view().{f.name}: 기대 {a}, 실제 {b}"


def test_retire_only_lowers_learning_active(game):
    """은퇴: 학습 활성만 내린다. 동결과 위치는 그대로다 (DECISIONS 3절)."""
    game.learning_active = True
    game.frozen = True
    cell = game.pac_cell
    game.retire()
    assert not game.learning_active
    assert game.frozen and game.pac_cell == cell
