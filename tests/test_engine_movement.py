"""진짜 엔진의 단계 2–5. 고정 장면에서 이동과 사건을 검사한다."""

import numpy as np
import pytest

from flypac.core import directions as d
from flypac.core.config import Config, load_config
from flypac.core.rng import make_rng
from flypac.engine.game import Game
from helpers import MAP, cell
from test_engine_real import SMOKE


def new_game(**engine):
    cfg = load_config(SMOKE).to_dict()
    cfg['engine'].update(engine)
    game = Game(MAP, Config(cfg), make_rng(9, 0, 'engine'))
    game.reset_game(make_rng(9, 1, 'engine'))
    # 다른 곳에 점을 남겨 이동 시험 도중 클리어를 막는다.
    game.dots[:] = False
    game.energizers[:] = False
    game.dots[cell(29, 1)] = True
    game.pac_cell = cell(1, 1)
    game.pac_dir = d.EAST
    return game


@pytest.mark.parametrize('speed', [0.0, 0.4, 0.8, 1.0])
def test_pacman_average_speed_through_step(speed):
    game = new_game(pacman_speed=speed)
    moves = 0
    for _ in range(1000):
        before = game.pac_cell
        direction = d.EAST if before == cell(1, 1) else d.WEST
        game.step_pacman(direction)
        moves += game.pac_cell != before
    assert abs(moves - 1000 * speed) <= 1


def test_pacman_can_reverse_and_wrap_tunnel():
    game = new_game(pacman_speed=1)
    game.pac_cell, game.pac_dir = cell(14, 0), d.EAST
    game.step_pacman(d.WEST)
    assert game.pac_cell == cell(14, 27) and game.pac_dir == d.WEST
    game.step_pacman(d.EAST)
    assert game.pac_cell == cell(14, 0)


def test_buffered_turn_is_applied_when_open():
    game = new_game(pacman_speed=1)
    game.pac_cell = cell(1, 5)
    game.step_pacman(d.SOUTH)  # 여기서는 막혔지만 다음 갈림길에서는 열린다.
    assert game.pac_cell == cell(1, 6) and game.pac_dir == d.EAST
    game.step_pacman(-1)  # 새 입력 없이 이전 입력 유지
    assert game.pac_cell == cell(2, 6) and game.pac_dir == d.SOUTH


def test_wall_and_house_door_block_pacman():
    game = new_game(pacman_speed=1)
    game.pac_dir = d.NORTH
    game.step_pacman(d.NORTH)
    assert game.pac_cell == cell(1, 1)
    game.pac_cell, game.pac_dir = cell(11, 13), d.SOUTH
    game.step_pacman(d.SOUTH)
    assert game.pac_cell == cell(11, 13)


def test_frozen_pacman_keeps_position_direction_and_accumulator():
    game = new_game(pacman_speed=1)
    game.frozen = True
    before = (game.pac_cell, game.pac_dir, game.pac_acc)
    assert game.step_pacman(d.SOUTH) == []
    assert (game.pac_cell, game.pac_dir, game.pac_acc) == before
