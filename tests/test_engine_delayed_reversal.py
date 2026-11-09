"""진짜 엔진의 초파리 반전 기회 (DECISIONS 4.1, 7~11행)."""

import numpy as np
import pytest

from flypac.core import directions as d
from flypac.core.config import Config
from flypac.core.events import Kind, ZCause
from flypac.core.rng import make_rng
from flypac.core.runner import build_runner
from flypac.pacman.train_env import PacmanTrainEnv
from helpers import Scripted, cell, make_cfg
from test_engine_movement import new_game
from test_original_ghosts import original_cfg
from test_pacman_obs import train_cfg


NO_ACTIONS = np.full(4, -1)


def actions(first):
    return np.array([first, -1, -1, -1])


def arrange_power(game, fly=(1, 4), *, heading=d.EAST, accumulator=0):
    game.pac_cell, game.pac_dir = cell(1, 1), d.EAST
    game.pac_acc = 1 - game.v_pac
    pellet = cell(1, 2)
    game.dots[pellet] = False
    game.energizers[pellet] = True
    game.fly_cell[0], game.fly_dir[0], game.fly_acc[0] = cell(*fly), heading, accumulator


def power_game(**overrides):
    game = new_game(**({"elroy_dots_left": [0, 0], "pacman_power_speed": 1,
                        "pacman_power_dot_speed": 1} | overrides))
    arrange_power(game)
    return game


@pytest.mark.parametrize("fly,heading,accumulator,open_actions,choice", [
    ((1, 4), d.EAST, 0, {d.FORWARD, d.BACK}, d.BACK),
    ((1, 6), d.EAST, 0.5, {d.FORWARD, d.RIGHT, d.BACK}, d.BACK),
    ((5, 6), d.EAST, 0.5, {d.LEFT, d.FORWARD, d.RIGHT, d.BACK}, d.BACK),
    ((11, 13), d.EAST, 0.5, {d.FORWARD, d.BACK}, d.BACK),
    ((11, 13), d.NORTH, 0.5, {d.LEFT, d.RIGHT}, d.LEFT),  # 출구에서 집으로 돌아가는 뒤쪽은 제외
])
def test_power_start_opens_full_legal_mask_once(fly, heading, accumulator, open_actions, choice):
    game = power_game()
    arrange_power(game, fly, heading=heading, accumulator=accumulator)
    events = game.step_pacman(d.EAST)
    assert any(e.kind is Kind.POWER_START and e.tick == 0 for e in events)
    legal, decide = game.decision_info()
    assert decide.tolist() == [True, False, False, False]
    assert set(np.flatnonzero(legal[0])) == open_actions and not legal[1:].any()
    assert game.decision_info()[1][0]  # 조회는 기회를 소비하지 않는다.
    _, moved, committed, _ = game.step_flies(actions(choice))
    assert committed.tolist() == [True, False, False, False]
    assert moved[0] == (accumulator == 0.5)
    assert not game.decision_info()[0][0, d.BACK]


@pytest.mark.parametrize("choice,heading,target", [(d.BACK, d.WEST, (1, 3)), (d.FORWARD, d.EAST, (1, 5))])
def test_stores_absolute_direction_then_applies_without_recommit(choice, heading, target):
    game = power_game(fly_speed=0.25)
    game.step_pacman(d.EAST)
    game.learning_active = True
    _, moved, committed, active = game.step_flies(actions(choice))
    assert not moved[0] and committed[0] and active[0]
    assert game.pending_fly_dir[0] == heading
    assert (game.fly_cell[0], game.fly_dir[0]) == (cell(1, 4), d.EAST)
    for _ in range(2):
        assert not game.decision_info()[1][0]
        _, moved, committed, _ = game.step_flies(NO_ACTIONS)
        assert not moved[0] and not committed[0]
    assert not game.decision_info()[1][0]  # 적용 틱도 새 결정을 묻지 않는다.
    _, moved, committed, _ = game.step_flies(NO_ACTIONS)
    assert moved[0] and not committed[0] and game.pending_fly_dir[0] == -1
    assert (game.fly_cell[0], game.fly_dir[0]) == (cell(*target), heading)


def test_power_end_offers_on_following_tick_to_changed_outside_flies():
    game = power_game(fly_speed=0.5, power_s=0.11)
    game.fly_acc[0] = 0.5
    game.step_pacman(d.EAST)
    _, moved, committed, _ = game.step_flies(actions(d.FORWARD))
    assert moved[0] and committed[0] and game.view().tick == 0
    assert not game.decision_info()[1][0]
    # 부활한 초파리는 집 밖에 있더라도 종료 시 z가 그대로다.
    game.fly_cell[1], game.fly_dir[1] = cell(1, 4), d.EAST
    game.released[1], game.fly_z[1] = True, False
    events = game.tick_timers()
    assert events[0].kind is Kind.POWER_END and events[0].tick == 0
    assert game.view().tick == 1 and not game.view().fly_z.any()
    legal, decide = game.decision_info()
    assert decide.tolist() == [True, False, False, False]
    assert legal[0, d.BACK]
    _, moved, committed, _ = game.step_flies(actions(d.BACK))
    assert not moved[0] and committed[0] and game.pending_fly_dir[0] == d.WEST


def test_two_power_transitions_before_decision_leave_observed_z_unchanged():
    game = power_game(fly_speed=0.5, power_s=0.11)
    game.fly_acc[0] = 0.5
    game.step_pacman(d.EAST)
    assert game.step_flies(actions(d.FORWARD))[2][0]
    assert game.tick_timers()[0].kind is Kind.POWER_END
    game.energizers[cell(1, 3)] = True
    assert any(e.kind is Kind.POWER_START for e in game.step_pacman(d.EAST))
    legal, decide = game.decision_info()
    assert game.view().fly_z[0] and not decide[0] and not legal[0, d.BACK]


def test_power_end_during_reservation_keeps_the_previous_choice():
    game = power_game(fly_speed=0.25, power_s=0.11)
    game.step_pacman(d.EAST)
    assert game.step_flies(actions(d.BACK))[2][0]
    assert game.tick_timers()[0].kind is Kind.POWER_END
    for _ in range(2):
        assert not game.decision_info()[1][0]
        _, moved, committed, _ = game.step_flies(NO_ACTIONS)
        assert not moved[0] and not committed[0]
    assert not game.decision_info()[1][0]
    _, moved, committed, _ = game.step_flies(NO_ACTIONS)
    assert moved[0] and not committed[0] and game.fly_dir[0] == d.WEST


@pytest.mark.parametrize("stored", (False, True))
def test_eaten_fly_loses_opportunity_or_reservation_without_offer_on_revive(stored):
    game = power_game(fly_speed=0.25)
    if stored:
        game.step_pacman(d.EAST)
        assert game.step_flies(actions(d.BACK))[2][0]
        game.pac_cell = cell(1, 3)
    else:
        game.fly_cell[0] = cell(1, 2)
    events = game.step_pacman(d.EAST)
    assert any(e.kind is Kind.EATEN and e.fly == 0 for e in events)
    assert any(e.cause is ZCause.REVIVE and e.fly == 0 for e in events)
    assert game.pending_fly_dir[0] == -1 and not game.view().fly_z[0]
    legal, decide = game.decision_info()
    assert not decide[0] and not legal[0, d.BACK]


def test_life_loss_discards_reservation_and_preserves_previous_commit():
    game = power_game(fly_speed=0.25)
    game.step_pacman(d.EAST)
    _, moved, committed, _ = game.step_flies(actions(d.BACK))
    assert committed[0] and not moved[0]
    game.fly_cell[1], game.fly_z[1], game.released[1] = cell(1, 3), False, True
    assert any(e.kind is Kind.LIFE_LOST for e in game.step_pacman(d.EAST))
    assert game.frozen and np.all(game.pending_fly_dir == -1)
    assert committed[0]
    game.reset_life()
    assert not game.decision_info()[1].any()


def test_new_game_discards_reservation():
    game = power_game(fly_speed=0.25)
    game.step_pacman(d.EAST)
    assert game.step_flies(actions(d.BACK))[2][0]
    assert game.pending_fly_dir[0] == d.WEST
    game.reset_game(make_rng(9, 2, "engine"))
    assert np.all(game.pending_fly_dir == -1)
    assert not game.decision_info()[1].any()


def test_b1_power_start_only_forces_one_turn_without_asking():
    game = power_game(fly_frightened_speed=1)
    game.configure_condition(original_cfg(), "b1")
    arrange_power(game, (1, 6))
    assert any(e.kind is Kind.POWER_START for e in game.step_pacman(d.EAST))
    legal, decide = game.decision_info()
    assert not decide[0] and not legal[0, d.BACK]
    _, moved, committed, _ = game.step_flies(NO_ACTIONS)
    assert moved[0] and not committed[0]
    assert (game.fly_cell[0], game.fly_dir[0]) == (cell(1, 5), d.WEST)
    assert not game.decision_info()[0][0, d.BACK]
    _, moved, committed, _ = game.step_flies(NO_ACTIONS)
    assert moved[0] and not committed[0]
    assert (game.fly_cell[0], game.fly_dir[0]) == (cell(1, 4), d.WEST)


def test_training_environment_and_runner_match_through_both_reversal_opportunities(monkeypatch):
    data = train_cfg().to_dict()
    data["engine"].update(kind="real", fly_speed=0.25, power_s=0.6)
    env = PacmanTrainEnv(Config(data), 1, 3, ghosts="random", condition="regime1")
    runner = build_runner(make_cfg(seeds=env.slot_seeds, engine=data["engine"]))
    env.core.start()
    env.observer.reset(np.ones(1, dtype=bool))
    arrange_power(env.game.games[0], (5, 6))
    env._observe()
    runner._reset_all()
    arrange_power(runner.game.games[0], (5, 6))
    runner.pacman = Scripted(d.EAST)

    def observe(core):
        steps = []
        propose, finish = core.ghosts.propose, core.finish

        def log_obs(obs, alive, views=None):
            steps.append((obs.legal.copy(), obs.decide.copy(), obs.z.copy()))
            return propose(obs, alive, views)

        def log_out(act):
            out = finish(act)
            steps.append(out)
            return out

        monkeypatch.setattr(core.ghosts, "propose", log_obs)
        monkeypatch.setattr(core, "finish", log_out)
        return steps

    env_steps, runner_steps = observe(env.core), observe(runner)
    fly_trace, power_end_ticks = [], []
    for tick in range(8):
        env.step(np.array([d.EAST]))
        runner.tick()
        a_obs, a = env_steps[-2:]
        b_obs, b = runner_steps[-2:]
        for left, right in zip(a_obs, b_obs):
            np.testing.assert_array_equal(left, right)
        assert a.events == b.events and a.now[0].tick == b.now[0].tick == tick + 1
        power_end_ticks.extend(e.tick for e in a.events[0] if e.kind is Kind.POWER_END)
        for name in ("fly_act", "moved", "committed"):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
        for name in ("fly_cell", "fly_dir"):
            np.testing.assert_array_equal(getattr(a.now[0], name), getattr(b.now[0], name))
        legal, decide, z = a_obs
        fly_trace.append((bool(decide[0, 0]), bool(legal[0, 0, d.BACK]), bool(z[0, 0]),
                          bool(a.moved[0, 0]), bool(a.committed[0, 0])))
    assert fly_trace[0] == (True, True, True, False, True)   # 시작 틱에 저장
    assert fly_trace[3] == (False, False, True, True, False)  # 예약 적용
    assert power_end_ticks == [5]
    assert fly_trace[6] == (True, True, False, False, True)  # 종료 다음 틱에 저장
    assert fly_trace[7] == (False, False, False, True, False)
