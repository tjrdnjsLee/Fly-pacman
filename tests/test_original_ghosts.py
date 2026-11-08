"""B1 목표 칸, 모드 반전, 학습·실험 동등성 (torch 불필요)."""

import copy
from pathlib import Path

import numpy as np
import pytest

from flypac.core import directions as d
from flypac.core.config import Config, load_config
from flypac.core.events import Kind
from flypac.core.rng import game_rngs, make_rng
from flypac.core.runner import build_runner
from flypac.core.types import FlyObs
from flypac.engine.game import Game
from flypac.ghosts.original import OriginalGhosts
from flypac.pacman.preflight import require_ready
from flypac.pacman.train_env import PacmanTrainEnv
from helpers import MAP, Scripted, cell, make_cfg
from test_engine_movement import new_game

TRAIN = Path(__file__).resolve().parents[1] / "configs" / "pacman_train_v1.yaml"
B1 = {
    "force_reversal": "original",
    "scatter_chase_s": [7, 20, 7, 20, 5, 20, 5],
    "scatter_targets": [[-3, 25], [-3, 2], [31, 27], [31, 0]],
    "pinky_ahead": 4,
    "inky_ahead": 2,
    "clyde_chase_distance": 8,
    "no_up_cells": [[11, 12], [11, 15], [23, 12], [23, 15]],
}


def original_cfg(**opts):
    return make_cfg(condition="b1", engine={"kind": "real", "b1": {**B1, **opts}})


def controller(cfg=None):
    return OriginalGhosts(cfg or original_cfg(), 1, game_rngs([1], "ghost"))


def target_view(*, heading=d.EAST, clyde=(29, 1)):
    game = new_game()
    game.configure_condition(original_cfg(), "b1")
    game.pac_cell, game.pac_dir = cell(5, 6), heading
    game.fly_cell[0], game.fly_cell[3] = cell(1, 6), cell(*clyde)
    game.b1.phase = 1
    return game.view()


@pytest.mark.parametrize("fly,heading,clyde,target", [
    (0, d.EAST, (29, 1), (5, 6)),
    (1, d.EAST, (29, 1), (5, 10)),
    (1, d.NORTH, (29, 1), (1, 2)),
    (2, d.EAST, (29, 1), (9, 10)),
    (2, d.NORTH, (29, 1), (5, 2)),
    (3, d.EAST, (29, 1), (5, 6)),
    (3, d.EAST, (5, 7), (31, 0)),
])
def test_arcade_chase_targets(fly, heading, clyde, target):
    assert controller().target_tile(target_view(heading=heading, clyde=clyde), fly) == target


def test_scatter_corner_targets_except_elroy_blinky():
    cfg = original_cfg()
    game = new_game()
    game.configure_condition(cfg, "b1")
    game.pac_cell = cell(5, 6)
    ghosts = controller(cfg)
    assert [ghosts.target_tile(game.view(), k) for k in range(4)] == list(map(tuple, B1["scatter_targets"]))
    game.elroy_stage = 1
    assert ghosts.target_tile(game.view(), 0) == (5, 6)


def test_propose_uses_euclidean_distance_then_arcade_tie_order():
    game = new_game()
    game.configure_condition(original_cfg(), "b1")
    game.b1.phase = 1
    game.fly_cell[0], game.fly_dir[0] = cell(5, 6), d.EAST
    legal, decide = game.decision_info()
    fields = np.zeros((1, 4, 4))
    obs = FlyObs(fields, fields, game.fly_z[None], legal[None], decide[None])
    ghosts = controller()
    game.pac_cell = cell(4, 6)
    assert ghosts.propose(obs, np.array([True]), [game.view()])[0, 0] == d.LEFT
    game.pac_cell = cell(5, 7)
    assert ghosts.propose(obs, np.array([True]), [game.view()])[0, 0] == d.FORWARD
    # 목표와 두 후보 사이의 거리가 같은 경우도 북쪽 우선.
    game.pac_cell = cell(5, 6)
    assert ghosts.propose(obs, np.array([True]), [game.view()])[0, 0] == d.LEFT


def test_propose_is_read_only_even_when_frightened():
    cfg = original_cfg()
    game = new_game()
    game.configure_condition(cfg, "b1")
    game.fly_cell[0], game.fly_dir[0] = cell(5, 6), d.EAST
    game.fly_z[0], game.fly_acc[0] = True, .5
    legal, decide = game.decision_info()
    fields = np.zeros((1, 4, 4))
    obs = FlyObs(fields, fields, game.fly_z[None], legal[None], decide[None])
    ghosts = controller(cfg)
    before = copy.deepcopy(ghosts.rngs[0].bit_generator.state)
    first = ghosts.propose(obs, np.array([True]), [game.view()])
    assert np.array_equal(first, ghosts.propose(obs, np.array([True]), [game.view()]))
    assert legal[0, first[0, 0]] and first[0, 0] >= 0
    assert ghosts.rngs[0].bit_generator.state == before
    ghosts.commit(np.zeros((1, 4), bool), np.ones((1, 4), bool))
    assert ghosts.rngs[0].bit_generator.state == before
    ghosts.commit(np.array([[True, False, False, False]]), np.zeros((1, 4), bool))
    assert ghosts.rngs[0].bit_generator.state != before


def test_frightened_tries_clockwise_from_blocked_direction():
    cfg = original_cfg()
    game = new_game()
    game.configure_condition(cfg, "b1")
    game.fly_cell[0], game.fly_dir[0] = cell(5, 6), d.EAST
    game.fly_z[0], game.fly_acc[0] = True, .5
    legal, decide = game.decision_info()
    fields = np.zeros((1, 4, 4))
    obs = FlyObs(fields, fields, game.fly_z[None], legal[None], decide[None])
    ghosts = OriginalGhosts(cfg, 1, game_rngs([0], "ghost"))
    # 시드 0의 첫 표본은 서쪽이다. 반전으로 막혀 시계방향의 북쪽을 고른다.
    assert ghosts.propose(obs, np.array([True]), [game.view()])[0, 0] == d.LEFT


def test_schedule_converts_to_approved_ticks_and_pauses_through_power():
    cfg = original_cfg()
    game = new_game()
    game.configure_condition(cfg, "b1")
    assert game.b1.schedule == (66, 189, 66, 189, 47, 189, 47)
    for _ in range(65):
        game.tick_timers()
    game.power_ticks = 2
    game.tick_timers()
    game.tick_timers()
    assert game.view().scatter and not game.b1.reverse.any()
    game.tick_timers()
    assert not game.view().scatter and game.b1.reverse.all()


def test_power_start_forces_turn_only_in_b1_and_does_not_commit():
    cfg = original_cfg()
    for condition in ("regime1", "b1"):
        game = new_game(fly_frightened_speed=1)
        game.configure_condition(cfg, condition)
        game.fly_cell[0], game.fly_dir[0] = cell(1, 3), d.EAST
        game.energizers[cell(1, 2)] = True
        game.step_pacman(d.EAST)
        # 팩맨을 옮겨서 강제 반전 자체를 관찰한다.
        game.pac_cell = cell(1, 1)
        _, moved, committed, _ = game.step_flies(np.full(4, -1))
        assert moved[0] and not committed[0]
        assert game.fly_cell[0] == cell(1, 2 if condition == "b1" else 4)


@pytest.mark.parametrize("row,col", [(11, 12), (11, 15), (23, 12), (23, 15)])
def test_b1_no_up_tiles_allow_north_only_while_frightened(row, col):
    game = new_game()
    game.configure_condition(original_cfg(), "b1")
    game.fly_cell[0], game.fly_dir[0] = cell(row, col), d.EAST
    assert not game.decision_info()[0][0, d.LEFT]
    game.fly_z[0] = True
    assert game.decision_info()[0][0, d.LEFT]


def test_real_b1_training_and_runner_match_through_forced_reverse():
    train_cfg = load_config(TRAIN).to_dict()
    train_cfg["engine"].update(
        b1={**B1, "scatter_chase_s": [.106, 20, 7, 20, 5, 20, 5]},
        pacman_speed=1, pacman_normal_speed_cap=1, pacman_dot_speed=1,
        fly_speed=1, fly_normal_speed_cap=1,
    )
    env = PacmanTrainEnv(Config(train_cfg), 1, 0, "original", "b1")
    cfg = make_cfg(condition="b1", seeds=env.slot_seeds, ghosts={"kind": "original"},
                   engine=train_cfg["engine"])
    runner = build_runner(cfg)
    runner._reset_all()
    env.core.start()
    for vec in (env.game, runner.game):
        vec.games[0].fly_cell[0] = cell(1, 2)
        vec.games[0].fly_dir[0] = d.EAST
        vec.games[0].pac_cell = cell(29, 1)
    env._observe()
    runner.pacman = Scripted(d.EAST)

    # 동일 행동열로 양쪽의 TickOut을 기록한다. 강제 반전은 두 번째 틱에 발생한다.
    observed = []
    original_finish = env.core.finish

    def record(action):
        out = original_finish(action)
        observed.append(out)
        return out
    env.core.finish = record
    for expected in [(1, 3), (1, 2), (1, 1)]:
        env.step(np.array([d.EAST]))
        views = runner.begin()
        out = runner.finish(runner.pacman.act(views))
        other = observed[-1]
        assert other.events == out.events
        assert [event.kind for event in out.events[0]] == [Kind.DOT]
        assert np.array_equal(other.fly_act, out.fly_act)
        assert np.array_equal(other.moved, out.moved)
        assert np.array_equal(other.committed, out.committed)
        assert np.array_equal(other.now[0].fly_cell, out.now[0].fly_cell)
        assert out.now[0].fly_cell[0] == cell(*expected)
    assert observed[1].moved[0, 0] and not observed[1].committed[0, 0]
    assert observed[1].now[0].fly_dir[0] == d.WEST


def test_production_training_preflight_supports_b1():
    require_ready(load_config(TRAIN))
