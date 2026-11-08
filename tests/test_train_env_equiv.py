"""실행기와 학습 환경의 동등성 · 관측 일치 · 조건 이름 (torch 없이 돈다).

같은 시드 · 같은 팩맨 행동열이면 실행기(Runner.tick)와 학습 환경(PacmanTrainEnv.step)이 같은 사건열 · 틱 번호 ·
유령 제안 행동 · 유령 위치를 낸다. 시나리오: 칸 0만 목숨 1개로 시작해 첫 틱에 잡혀 판이 끝나고, 칸 1은 같은 틱에
잡히지만 목숨이 남아 판이 계속되며, 칸 2는 아무 일도 없다.
"""

from __future__ import annotations

import numpy as np
import pytest

from flypac.core import directions as d
from flypac.core.config import Config
from flypac.core.events import Kind, Winner
from flypac.core.runner import build_runner
from flypac.ghosts.random_ghost import RandomGhost
from flypac.pacman.base import PacmanController
from flypac.pacman.obs import PacmanObserver, engine_constants
from flypac.pacman.train_env import GHOST_FACTORIES, PacmanTrainEnv, check_condition, make_ghosts
from helpers import MAP, make_cfg
from test_pacman_obs import train_cfg

B, T, SEED = 3, 70, 3


def adjacent_to_fly0():
    """(팩맨을 놓을 칸, 그 칸에서 초파리 0번 칸으로 가는 팩맨 행동)."""
    fly0 = int(MAP.fly_starts[0])
    for dd in range(4):
        n = int(MAP.nbr[fly0][dd])
        back = int(d.opposite(dd))
        if n >= 0 and MAP.move_mask["pacman"][n][back]:
            return n, back
    raise AssertionError("초파리 0번 옆에 팩맨이 올 수 있는 칸이 없다")


def arrange(vec) -> int:
    """칸 0 · 1의 팩맨을 초파리 0번 옆에 놓고, 칸 0만 목숨을 1개로 한다. 첫 틱에 팩맨이 걸어가면 잡힌다."""
    cell, act = adjacent_to_fly0()
    for b in (0, 1):
        vec.games[b].pac_cell = cell
    vec.games[0].lives = 1
    return act


def action_table(first: int) -> np.ndarray:
    acts = np.random.default_rng(7).integers(0, 4, size=(T, B))
    acts[0, :2] = first
    return acts


class ScriptedTable(PacmanController):
    """틱 번호(호출 횟수)로 행동을 고르는 팩맨. 상태와 무관하므로 두 실행에 같은 행동열을 준다."""

    def __init__(self, acts):
        self.acts, self.t, self.resets = acts, 0, []

    def act(self, views):
        a = self.acts[self.t]
        self.t += 1
        return a.astype(np.int64)

    def reset(self, mask):
        self.resets.append(np.asarray(mask).copy())


def start_env(env: PacmanTrainEnv) -> dict:
    """env.reset과 같은 순서로 시작하되 첫 관측 전에 시나리오를 놓는다."""
    env.core.start()
    env.observer.reset(np.ones(env.B, dtype=bool))
    env.first_action = arrange(env.game)
    return env._observe()


def make_pair():
    env = PacmanTrainEnv(train_cfg(), B, SEED, ghosts="random", condition="regime1")
    cfg = make_cfg(seeds=env.slot_seeds, n_games=1000, run={"max_loop_ticks": 10**6})
    runner = build_runner(cfg)
    return env, runner


def spy(obj, outs):
    orig = obj.finish
    obj.finish = lambda a: outs.append(orig(a)) or outs[-1]


def run_both(acts):
    env, runner = make_pair()
    env_outs, run_outs = [], []
    spy(env.core, env_outs)
    spy(runner, run_outs)
    start_env(env)
    runner._reset_all()
    arrange(runner.game)
    runner.pacman = ScriptedTable(acts)
    obs_env = []
    for t in range(T):
        obs, *_ = env.step(acts[t])
        obs_env.append(obs)
        runner.tick()
    return env, runner, env_outs, run_outs, obs_env


def test_train_env_matches_runner_events_ticks_and_ghosts():
    first = adjacent_to_fly0()[1]
    env, runner, eo, ro, _ = run_both(action_table(first))
    assert len(eo) == len(ro) == T
    for t, (a, b) in enumerate(zip(eo, ro)):
        assert a.events == b.events, f"틱 {t}: 사건열이 다르다"  # 종류 · 칸 · 틱 번호 · 초파리 · 활성 · 원인까지
        assert np.array_equal(a.fly_act, b.fly_act), f"틱 {t}: 유령 제안 행동"
        assert np.array_equal(a.moved, b.moved) and np.array_equal(a.committed, b.committed)
        assert np.array_equal(a.ends, b.ends) and np.array_equal(a.over, b.over) and np.array_equal(a.started, b.started)
        for k in range(B):
            assert np.array_equal(a.now[k].fly_cell, b.now[k].fly_cell), f"틱 {t} 칸 {k}: 유령 위치"
            assert a.now[k].pac_cell == b.now[k].pac_cell and a.now[k].tick == b.now[k].tick
            assert a.now[k].lives == b.now[k].lives and a.now[k].score == b.now[k].score
    assert np.array_equal(env.core.tick_no, runner.tick_no) and np.array_equal(env.core.game_no, runner.game_no)

    # 시나리오가 실제로 일어났다: 칸 0만 판이 끝나고, 칸 1은 목숨만 잃고, 칸 2는 아무 일도 없다
    ev0 = eo[0].events
    assert [e.kind for e in ev0[0]] == [Kind.CATCH, Kind.LIFE_LOST, Kind.GAME_OVER] and ev0[0][2].winner is Winner.FLIES
    assert [e.kind for e in ev0[1]] == [Kind.CATCH, Kind.LIFE_LOST]
    assert not any(e.kind in (Kind.LIFE_LOST, Kind.GAME_OVER) for e in ev0[2])
    assert eo[0].over.tolist() == [True, False, False] and eo[0].ends.tolist() == [True, True, False]
    assert env.core.game_no[0] == 2 and env.core.game_no[1] == 1  # 한 칸만 새 판
    # 틱 번호 = 재생 프레임 번호: 사건이 난 틱 0, 새 판은 0부터 다시
    assert all(e.tick == 0 for e in ev0[0])
    assert eo[1].now[0].tick == 1 and eo[1].now[1].tick == 2


def test_train_env_and_runner_use_same_rng_purposes_and_game_numbers():
    env, runner, *_ = run_both(action_table(adjacent_to_fly0()[1]))
    for name, a, b in (("engine", env.core.engine_rngs, runner.engine_rngs),
                       ("ghost", env.ghosts.rngs, runner.ghosts.rngs)):
        assert a.purpose == b.purpose == name
        assert np.array_equal(a.seeds, b.seeds) and np.array_equal(a.game_no, b.game_no)
        sa, sb = a.state_dict(), b.state_dict()
        assert sa["states"] == sb["states"], f"{name} 난수 상태가 다르다"
    assert env.ghosts.rngs.game_no.tolist() == [2, 1, 1]  # 판이 바뀐 칸만 갱신


def test_observation_matches_the_runner_side_observer_every_tick():
    """학습 환경의 관측(격자 전체 · 스칼라 · 마스크)이 실행기 쪽(PPOPacman과 같은 PacmanObserver 사용 순서)과
    여러 틱, 한 칸만 초기화되는 경우(목숨 · 판)에서 같다."""
    acts = action_table(adjacent_to_fly0()[1])
    env, runner = make_pair()
    consts = engine_constants(train_cfg())

    class ObsLogger(ScriptedTable):
        """PPOPacman.act과 같은 순서로 관측한다 (observe → legal). reset은 프레임 이력만 지운다."""

        def __init__(self, acts):
            super().__init__(acts)
            self.observer = PacmanObserver(MAP, 4, True, consts)
            self.log = []

        def act(self, views):
            obs = self.observer.observe(views)
            self.log.append((obs["grid"], obs["scalars"], self.observer.legal(views), self.observer.engine_legal(views)))
            return super().act(views)

        def reset(self, mask):
            super().reset(mask)
            self.observer.reset(mask)

    pac = ObsLogger(np.vstack([acts, acts[-1:]]))  # 마지막 틱 뒤의 관측까지 얻으려고 한 줄 더
    obs0 = start_env(env)
    runner._reset_all()
    arrange(runner.game)
    runner.pacman = pac
    env_log = [(obs0["grid"], obs0["scalars"], env.legal_masks(), env.observer.engine_legal(env._views))]
    for t in range(T):
        obs, *_ = env.step(acts[t])
        env_log.append((obs["grid"], obs["scalars"], env.legal_masks(), env.observer.engine_legal(env._views)))
        runner.tick()
    runner.pacman.act(runner.begin())  # 마지막 틱 뒤의 관측까지 같은 방법으로 얻는다
    assert len(pac.log) == len(env_log) == T + 1
    for t, (e, r) in enumerate(zip(env_log, pac.log)):
        for name, x, y in zip(("격자", "스칼라", "탐욕 마스크", "엔진 마스크"), e, r):
            assert np.array_equal(x, y), f"틱 {t}: {name}이 다르다"
    # 초기화된 칸만 프레임 이력이 다시 채워졌다: 틱 1(칸 0 · 1이 첫 틱에 잡힘)에서 칸 0 · 1은 4칸이 모두 같다
    grid1 = env_log[1][0].reshape(B, 4, 8, 31, 28)
    for b in (0, 1):
        assert all(np.array_equal(grid1[b, 0], grid1[b, i]) for i in range(1, 4)), f"칸 {b}"
    assert not all(np.array_equal(grid1[2, 0], grid1[2, i]) for i in range(1, 4))  # 칸 2는 그대로 이어진다
    # 스칼라: 칸 1은 목숨 2/3, 칸 0은 새 판이라 3/3 (목숨 분모 = 엔진 설정의 초기 목숨)
    assert env_log[1][1][:, 1].tolist() == pytest.approx([1.0, 2 / 3, 1.0])


def test_state_roundtrip_with_one_slot_reset():
    """한 칸만 판이 바뀐 시점에 저장해도 복원 뒤 같은 궤적이 나온다."""
    acts = action_table(adjacent_to_fly0()[1])
    env = PacmanTrainEnv(train_cfg(), B, SEED, ghosts="random", condition="regime1")
    start_env(env)
    for t in range(5):
        env.step(acts[t])
    blob = env.state_bytes()
    first = [env.step(acts[t])[0]["grid"].copy() for t in range(5, 40)]
    env2 = PacmanTrainEnv(train_cfg(), B, 99, ghosts="random", condition="regime1")
    env2.load_state_bytes(blob)
    second = [env2.step(acts[t])[0]["grid"].copy() for t in range(5, 40)]
    assert all(np.array_equal(x, y) for x, y in zip(first, second))


# ---- 조건 이름 (원칙 2) ----
class ViewsGhost(RandomGhost):
    """원작 유령 대역: views를 쓴다."""

    uses_views = True

    def __init__(self, cfg, B, rngs):
        super().__init__(cfg, B, rngs)
        self.seen_views = []

    def propose(self, obs, alive, views=None):
        self.seen_views.append(views)
        return super().propose(obs, alive, views)


def test_b1_condition_passes_views(monkeypatch):
    monkeypatch.setitem(GHOST_FACTORIES, "original", lambda cfg, B, seeds: ViewsGhost(cfg, B, __import__(
        "flypac.core.rng", fromlist=["game_rngs"]).game_rngs(seeds, "ghost")))
    env = PacmanTrainEnv(train_cfg(), 2, 0, ghosts="original", condition="b1")
    env.reset()
    env.step(np.array([d.WEST, d.WEST]))
    assert env.ghosts.seen_views and all(v is not None and len(v) == 2 for v in env.ghosts.seen_views)
    # 무작위 유령은 어떤 조건에서도 views를 받지 않는다
    env2 = PacmanTrainEnv(train_cfg(), 2, 0, ghosts="random", condition="regime1")
    assert env2.core.pass_views is False


def test_condition_name_is_not_inferred(monkeypatch):
    monkeypatch.setitem(GHOST_FACTORIES, "original", lambda cfg, B, seeds: ViewsGhost(cfg, B, __import__(
        "flypac.core.rng", fromlist=["game_rngs"]).game_rngs(seeds, "ghost")))
    with pytest.raises(ValueError, match="b1"):  # views 유령에 b1이 아닌 조건
        PacmanTrainEnv(train_cfg(), 1, 0, ghosts="original", condition="regime1")
    with pytest.raises(ValueError, match="b1"):  # views를 안 쓰는 유령에 b1
        PacmanTrainEnv(train_cfg(), 1, 0, ghosts="random", condition="b1")
    env = PacmanTrainEnv(train_cfg(), 1, 0, ghosts="random", condition="regime1")
    with pytest.raises(ValueError, match="b1"):
        env.set_ghosts("original", "regime1")
    env.set_ghosts("original", "b1")
    assert env.core.condition == "b1" and env.core.pass_views is True
    check_condition(make_ghosts("random", train_cfg(), 1, [0]), "regime2")  # 조건 이름은 b1만 특별하다


def test_runner_ppo_branch_needs_settings():
    with pytest.raises(ValueError, match="checkpoint"):
        build_runner(make_cfg(pacman={"kind": "ppo"}))
    with pytest.raises(NotImplementedError):
        build_runner(make_cfg(pacman={"kind": "nothing"}))
    assert isinstance(Config({}), Config)
