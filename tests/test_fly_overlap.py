"""초파리끼리 겹침과 충돌 검사 순서의 현재 동작을 고정한다 (DECISIONS 25.2). 엔진은 바꾸지 않았다.

장면은 `Game`의 상태를 직접 놓아 만든다: 초파리를 방출됨(`released`)으로 두고 `fly_cell` · `fly_z`를 직접 쓴다.
25.4에서 규칙이 바뀌면 이 시험도 함께 바꾼다.
"""

import numpy as np

from flypac.core import directions as d
from flypac.core.events import Kind
from helpers import cell
from test_engine_movement import new_game

NO_ACTION = np.full(4, -1)


def scene(flies_at, z=()):
    """팩맨은 (1,1)에서 동쪽으로 한 칸 가면 (1,2)에 닿는다. flies_at: {번호: 칸}."""
    game = new_game(pacman_speed=1, fly_speed=1)
    for k, c in flies_at.items():
        game.released[k] = True
        game.fly_cell[k] = c
        game.fly_dir[k] = d.EAST
    for k in z:
        game.fly_z[k] = True
    return game


def test_two_released_flies_can_share_a_cell_and_keep_moving():
    game = scene({0: cell(1, 5), 1: cell(1, 5)})
    game.pac_cell = cell(29, 1)  # 멀리
    assert game.fly_cell[0] == game.fly_cell[1]
    events, moved, _, _ = game.step_flies(NO_ACTION)
    assert moved[0] and moved[1] and not events
    assert game.fly_cell[0] == game.fly_cell[1] == cell(1, 6)  # 둘 다 막히지 않고 같은 칸으로 간다


def test_pacman_entering_cell_with_two_normal_flies_catches_lower_index_only():
    game = scene({0: cell(1, 2), 1: cell(1, 2)})
    events = game.step_pacman(d.EAST)
    assert [(e.kind, e.fly) for e in events if e.fly >= 0] == [(Kind.CATCH, 0)]
    assert not [e for e in events if e.fly == 1]
    assert game.view().lives == 2 and game.frozen


def test_same_cell_normal_j_and_frightened_k_lower_k_is_eaten_then_j_caught():
    game = scene({0: cell(1, 2), 1: cell(1, 2)}, z=[0])  # k = 0 < j = 1
    events = game.step_pacman(d.EAST)
    kinds = [(e.kind, e.fly) for e in events if e.kind in (Kind.EATEN, Kind.CATCH)]
    assert kinds == [(Kind.EATEN, 0), (Kind.CATCH, 1)]
    assert game.view().lives == 2


def test_same_cell_normal_j_and_frightened_k_higher_k_is_not_eaten():
    game = scene({0: cell(1, 2), 1: cell(1, 2)}, z=[1])  # k = 1 > j = 0
    events = game.step_pacman(d.EAST)
    kinds = [(e.kind, e.fly) for e in events if e.kind in (Kind.EATEN, Kind.CATCH)]
    assert kinds == [(Kind.CATCH, 0)]
    assert game.fly_cell[1] == cell(1, 2) and game.fly_z[1]  # 먹히지 않고 제자리
