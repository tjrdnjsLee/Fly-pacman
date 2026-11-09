"""명시적 엔진 설정과 방출·속도·행동 경계의 회귀 시험."""

import numpy as np
import pytest

from flypac.core import directions as d
from flypac.core.config import Config, load_config
from flypac.core.rng import make_rng
from flypac.engine.game import Game
from helpers import MAP, cell
from test_engine_house import eat_dots, fly_step, house_game
from test_engine_movement import new_game
from test_engine_real import SMOKE


def test_real_engine_reports_all_missing_settings_together():
    cfg = load_config(SMOKE).to_dict()
    missing = ("house_personal_dots", "fly_tunnel_speed", "pacman_dot_speed", "pacman_normal_speed_cap")
    for key in missing:
        del cfg["engine"][key]
    del cfg["time"]["v_max_tiles_per_s"]
    with pytest.raises(ValueError) as exc:
        Game(MAP, Config(cfg), make_rng(9, 1, "engine"))
    assert all(f"engine.{key}" in str(exc.value) for key in missing)
    assert "time.v_max_tiles_per_s" in str(exc.value)


@pytest.mark.parametrize("key,value", [
    ("house_personal_dots", [0, 0, 30]),
    ("house_global_dots", [7, 17]),
    ("score_ghosts", [200, 400]),
    ("elroy_dots_left", [20]),
    ("elroy_speeds", [0.8]),
    ("fruit_dots_eaten", [70]),
])
def test_wrong_array_length_is_rejected_at_construction(key, value):
    with pytest.raises(ValueError, match=rf"engine\.{key}"):
        new_game(**{key: value})


@pytest.mark.parametrize("tile,frightened", [((14, 0), False), ((1, 6), True), ((14, 0), True)])
def test_slow_fly_does_not_accelerate_in_special_state(tile, frightened):
    game = new_game(fly_speed=.3)
    game.fly_cell[0], game.fly_dir[0] = cell(*tile), d.EAST
    game.fly_z[0] = frightened
    assert not fly_step(game)[1][0]
    assert game.fly_acc[0] == pytest.approx(.3)


def test_personal_count_survives_death_and_resumes_at_global_32():
    game = house_game()
    for _ in range(3):
        fly_step(game)
    eat_dots(game, 30)
    for _ in range(5):
        fly_step(game)
    eat_dots(game, 59)
    before = game.house_personal_count.copy()
    assert before.tolist() == [0, 0, 30, 59]

    game.lives -= 1
    game.reset_life()
    assert np.array_equal(game.house_personal_count, before)
    assert game.use_global_house_count and game.house_global_count == 0
    eat_dots(game, 31)
    for _ in range(6):
        fly_step(game)
    assert game.use_global_house_count and not game.exiting[3]
    eat_dots(game, 1)
    assert not game.use_global_house_count and game.house_global_count == 0
    assert np.array_equal(game.house_personal_count, before)
    assert not fly_step(game)[1][3]  # 32개째는 전역 카운터 전환에만 쓰인다.
    eat_dots(game, 1)
    assert fly_step(game)[1][3]  # 보존한 59에 개인 점 하나를 더한다.


def test_global_count_does_not_switch_at_32_when_clyde_is_outside():
    game = house_game()
    game.lives -= 1
    game.reset_life()
    game.fly_cell[3] = cell(11, 13)
    game.released[3] = True
    eat_dots(game, 33)
    assert game.use_global_house_count and game.house_global_count == 33


@pytest.mark.parametrize("action", [-1, 4])
def test_invalid_pacman_action_raises_without_advancing_accumulator(action):
    game = new_game()
    with pytest.raises(ValueError, match="불가능한 행동"):
        game.step_pacman(action)
    assert game.pac_acc == 0 and game.pac_cell == cell(1, 1)


def test_pacman_can_reverse_in_an_ordinary_corridor():
    game = new_game(pacman_speed=1)
    game.pac_cell = cell(1, 2)
    game.step_pacman(d.EAST)
    assert game.pac_cell == cell(1, 3)
    game.step_pacman(d.WEST)
    assert game.pac_cell == cell(1, 2) and game.pac_dir == d.WEST
