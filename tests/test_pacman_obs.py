"""PPO 팩맨 관측 · 보상 · 학습 환경 (torch 없이 돈다)."""

from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest

from flypac.core import directions as d
from flypac.core.config import Config
from flypac.core.events import Event, Kind, Winner
from flypac.maps.original import WALL
from flypac.core.timing import config_ticks, seconds_to_ticks
from flypac.pacman.obs import (NUM_CHANNELS, NUM_SCALARS, FrameStack, ObsEncoder, dangerous_flies, edible_flies,
                               engine_constants, legal_mask)
from flypac.pacman.rewards import RewardFn
from flypac.pacman.train_env import PacmanTrainEnv
from helpers import BASE, MAP

TRAIN = {
    "seed": 0, "num_envs": 3, "device": "cpu", "run_dir": "unused",
    "obs": {"frame_stack": 4, "mask_reverse": True},
    "curriculum": [{"from_update": 0, "ghosts": "random", "condition": "regime1"}],
    "entropy_boost_updates": 300,
    "rewards": {"eat_pellet": 1.0, "eat_power_pellet": 2.0, "eat_ghost": [5.0, 10.0, 15.0, 20.0],
                "eat_fruit": 3.0, "clear_level": 50.0, "death": -10.0, "game_over": -25.0,
                "ghost_proximity": -0.3, "ghost_proximity_dist": 2, "time_step": -0.01},
}


def train_cfg(**over) -> Config:
    data = copy.deepcopy(BASE)
    data["pacman_train"] = copy.deepcopy(TRAIN)
    data["pacman_train"].update(over)
    return Config(data)


def start_view():
    env = PacmanTrainEnv(train_cfg(), 1, 0, ghosts="random", condition="regime1")
    env.reset()
    return env.game.views()[0]


def test_grid_channels():
    enc = ObsEncoder(MAP, 57, 3)
    v = start_view()
    v = replace(v, fly_z=np.array([False, True, False, False]),
                fly_in_house=np.array([False, False, True, True]), fruit=True)
    grid, scalars = enc.encode([v])
    assert grid.shape == (1, NUM_CHANNELS, 31, 28) and grid.dtype == np.float32
    assert scalars.shape == (1, NUM_SCALARS)
    g = grid[0]
    assert np.array_equal(g[0], (MAP.grid == WALL).astype(np.float32))
    r, c = MAP.cell_rc[v.pac_cell]
    assert g[1].sum() == 1 and g[1, r, c] == 1
    assert g[2].sum() == MAP.dots.sum() and g[3].sum() == MAP.energizers.sum()
    # 0번은 집 밖 z=0 → 위험, 1번은 집 밖 z=1 → 먹을 수 있음, 2·3번은 집 안 → 어디에도 없음
    r0, c0 = MAP.cell_rc[v.fly_cell[0]]
    r1, c1 = MAP.cell_rc[v.fly_cell[1]]
    assert g[4].sum() == 1 and g[4, r0, c0] == 1
    assert g[5].sum() == 1 and g[5, r1, c1] == 1
    assert g[6].sum() == len(MAP.house_cells) + len(MAP.door_cells)
    fr, fc = MAP.cell_rc[MAP.fruit_cell]
    assert g[7].sum() == 1 and g[7, fr, fc] == 1


def test_scalars():
    enc = ObsEncoder(MAP, 57, 3)
    v = start_view()
    dots = v.dots.copy()
    dots[np.flatnonzero(dots)[:10]] = False
    v = replace(v, power=True, power_ticks=19, lives=2, ghost_streak=2, dots=dots, pac_dir=d.EAST)
    _, s = enc.encode([v])
    total = MAP.dots.sum() + MAP.energizers.sum()
    assert s[0] == pytest.approx([19 / 57, 2 / 3, 0.5, 10 / total, 3 / 3])  # 동 = pacman-ai 우(3)
    _, s = enc.encode([replace(v, power=False, power_ticks=19)])
    assert s[0, 0] == 0.0


def test_frame_stack_fill_and_shift():
    fs = FrameStack(2, 4, (1, 1, 1))
    f = lambda x: np.array(x, dtype=np.float32).reshape(2, 1, 1, 1)  # noqa: E731
    out = fs.push(f([1, 10]))
    assert out[:, :, 0, 0].tolist() == [[1] * 4, [10] * 4]  # 처음: 4칸 모두 첫 프레임
    out = fs.push(f([2, 20]))
    assert out[:, :, 0, 0].tolist() == [[1, 1, 1, 2], [10, 10, 10, 20]]  # 오래된 것이 앞
    fs.reset(np.array([False, True]))
    out = fs.push(f([3, 30]))
    assert out[:, :, 0, 0].tolist() == [[1, 1, 2, 3], [30] * 4]  # 지운 칸만 다시 채운다


def test_legal_mask_reverse():
    v = start_view()
    plain = MAP.move_mask["pacman"][v.pac_cell]
    m = legal_mask(MAP, [v], mask_reverse=False)[0]
    assert np.array_equal(m, plain)
    m = legal_mask(MAP, [v], mask_reverse=True)[0]
    back = int(d.opposite(v.pac_dir))
    assert plain[back] and not m[back] and m.sum() == plain.sum() - 1


def ev(kind, fly=-1, cause=None):
    return Event(kind, 0, 0, fly, False, cause)


def far_view(**over):
    """모든 초파리가 집 안: 근접 항이 없는 배치."""
    return replace(start_view(), fly_in_house=np.ones(4, dtype=bool), **over)


def near_view(n_danger=2):
    """위험한 유령 n마리가 팩맨과 맨해튼 거리 2 이내. 나머지는 z = 1이거나 집 안."""
    v = start_view()
    pac = v.pac_cell
    near = int(MAP.nbr[pac][MAP.nbr[pac] >= 0][0])
    cells = np.array([near, pac, pac, near])
    in_house = np.array([False, False, True, False])
    z = np.array([False, False, False, True])
    out = replace(v, fly_cell=cells, fly_in_house=in_house, fly_z=z)
    assert dangerous_flies(out).sum() == 2  # 0번 · 1번만 위험 (2번 집 안, 3번 z = 1)
    return out


def test_rewards_fixed_cases():
    """고정 사례: 이동 / 정지, 사망, 클리어, 먹기. 시간 · 근접 항은 팩맨이 움직인 틱에만 (DECISIONS 14.1절)."""
    fn = RewardFn(TRAIN["rewards"], MAP.cell_rc)
    v, near = far_view(), near_view()
    # 이동 없는 틱: 시간 · 근접 항이 없다. 가까운 위험한 유령이 있어도 0
    assert fn([], v, near, moved=False) == 0.0
    # 이동한 틱: 시간 항 + 근접 항(위험한 유령 2마리)
    assert fn([], v, near, moved=True) == pytest.approx(-0.01 - 0.3 * 2)
    assert fn([], v, v, moved=True) == pytest.approx(-0.01)
    # 사건 항은 움직임과 무관하다
    assert fn([ev(Kind.DOT)], v, v, moved=False) == pytest.approx(1.0)
    assert fn([ev(Kind.DOT)], v, v, moved=True) == pytest.approx(1.0 - 0.01)
    assert fn([ev(Kind.ENERGIZER)], v, v, moved=True) == pytest.approx(2.0 - 0.01)
    assert fn([ev(Kind.FRUIT)], v, v, moved=True) == pytest.approx(3.0 - 0.01)
    # 사망(판 계속), 정지 틱(유령이 팩맨에게 걸어옴): death + 근접 항(초기화 뒤 배치), 시간 항 없음
    die = [ev(Kind.CATCH, 0), ev(Kind.LIFE_LOST)]
    assert fn(die, v, near, moved=False) == pytest.approx(-10 - 0.3 * 2)
    # 사망, 이동한 틱: death + 시간 + 근접
    assert fn(die, v, near, moved=True) == pytest.approx(-10 - 0.01 - 0.3 * 2)
    assert fn(die, v, v, moved=True) == pytest.approx(-10 - 0.01)
    # 판 종료(승자 FLIES): death + game_over + 근접(잡힌 순간 배치)
    over = die + [ev(Kind.GAME_OVER, cause=Winner.FLIES)]
    assert fn(over, v, near, moved=True) == pytest.approx(-10 - 25 - 0.01 - 0.3 * 2)
    # 클리어: 이동한 틱의 사건 + 시간 항. 근접 항은 now 배치 기준
    win = [ev(Kind.DOT), ev(Kind.GAME_OVER, cause=Winner.PACMAN)]
    assert fn(win, v, v, moved=True) == pytest.approx(1 + 50 - 0.01)
    assert fn(win, v, near, moved=True) == pytest.approx(1 + 50 - 0.01 - 0.6)
    # 중단: 보상 사건 없음
    cens = [ev(Kind.GAME_OVER, cause=Winner.CENSORED)]
    assert fn(cens, v, v, moved=False) == 0.0
    assert fn(cens, v, v, moved=True) == pytest.approx(-0.01)


def test_rewards_eaten_order_from_engine_streak():
    """먹힘 순번은 엔진 상태에서: k번째 EATEN의 순번 = ghost_streak − EATEN 수 + k."""
    fn = RewardFn(TRAIN["rewards"], MAP.cell_rc)
    v = far_view()
    one = [ev(Kind.EATEN, 0)]
    two = [ev(Kind.EATEN, 0), ev(Kind.EATEN, 1)]
    # 한 틱에 두 마리: 이 틱 전에 0마리 → 순번 0, 1
    assert fn(two, replace(v, ghost_streak=2), v, moved=True) == pytest.approx(5 + 10 - 0.01)
    # 이미 1마리를 먹은 파워에서 두 마리 더: 순번 1, 2
    assert fn(two, replace(v, ghost_streak=3), v, moved=True) == pytest.approx(10 + 15 - 0.01)
    # 한 마리: 첫 번째
    assert fn(one, replace(v, ghost_streak=1), v, moved=True) == pytest.approx(5 - 0.01)
    # 마지막 값에서 멈춘다
    assert fn(one, replace(v, ghost_streak=7), v, moved=True) == pytest.approx(20 - 0.01)
    # ghost_streak < 이번 틱 EATEN 수: 엔진 계약 위반 → 멈춘다
    with pytest.raises(RuntimeError, match="ghost_streak"):
        fn(two, replace(v, ghost_streak=1), v, moved=True)


def test_dangerous_flies_is_shared_by_obs_and_reward():
    v = replace(start_view(), fly_z=np.array([False, True, False, False]),
                fly_in_house=np.array([False, False, True, True]))
    assert dangerous_flies(v).tolist() == [True, False, False, False]
    assert edible_flies(v).tolist() == [False, True, False, False]
    fn = RewardFn(TRAIN["rewards"], MAP.cell_rc)
    pac = v.pac_cell
    # 보상의 근접 항이 세는 유령 = dangerous_flies와 같은 판정 (z = 1 · 집 안은 세지 않는다)
    w = replace(v, fly_cell=np.full(4, pac))
    assert fn.proximity_count(w) == int(dangerous_flies(w).sum()) == 1
    grid, _ = ObsEncoder(MAP, 57, 3).encode([w])
    assert grid[0, 4].sum() == 1 and grid[0, 5].sum() == 1  # 채널 4 · 5도 같은 분류


def test_engine_constants_and_ticks():
    assert seconds_to_ticks(6, 9.47) == 57  # round(56.82)
    cfg = train_cfg()
    assert config_ticks(cfg, 6) == 57
    assert engine_constants(cfg) == {"power_ticks_max": 57, "lives": 3}
    data = cfg.to_dict()
    data["engine"]["power_s"] = 3
    data["engine"]["lives"] = 5
    assert engine_constants(Config(data)) == {"power_ticks_max": round(3 * 9.47), "lives": 5}
    del data["engine"]["power_s"]
    with pytest.raises(ValueError, match="power_s"):
        engine_constants(Config(data))


def test_env_constants_come_from_engine_config():
    env = PacmanTrainEnv(train_cfg(), 1, 0, ghosts="random", condition="regime1")
    assert env.constants == {"power_ticks_max": 57, "lives": 3}
    assert env.observer.encoder.power_ticks_max == 57 and env.observer.encoder.lives_max == 3


def test_idle_ticks_get_no_time_penalty():
    """팩맨 속도 0.5 → 두 틱에 한 번만 움직인다. 정지 틱의 보상은 0, 이동 틱은 시간 항."""
    data = train_cfg().to_dict()
    data["engine"]["pacman_speed"] = 0.5
    env = PacmanTrainEnv(Config(data), 1, 0, ghosts="random", condition="regime1")
    env.reset()
    seen = {True: 0, False: 0}
    for _ in range(8):
        act = np.array([d.WEST])
        _, r, _, info = env.step(act)
        m = bool(info["moved"][0])
        seen[m] += 1
        assert not info["ends"][0]
        if m:  # 이동 틱: 시간 항 (점을 먹었으면 +1이 더해진다)
            assert r[0] == pytest.approx(-0.01) or r[0] == pytest.approx(0.99)
        else:  # 정지 틱: 아무 항도 없다
            assert r[0] == 0.0
    assert seen[True] > 0 and seen[False] > 0


def test_env_runs_and_resets():
    env = PacmanTrainEnv(train_cfg(), 3, 0, ghosts="random", condition="regime1")
    obs = env.reset()
    assert obs["grid"].shape == (3, 32, 31, 28) and obs["scalars"].shape == (3, 5)
    rng = np.random.default_rng(0)
    dones_seen = 0
    outs = []
    orig = env.core.finish
    env.core.finish = lambda a: outs.append(orig(a)) or outs[-1]
    for _ in range(3000):
        legal = env.legal_masks()
        acts = np.array([rng.choice(np.flatnonzero(m)) for m in legal])
        obs, r, done, info = env.step(acts)
        for b, v in enumerate(outs[-1].now):  # 엔진 계약: 매 틱 ghost_streak − 이번 틱 EATEN 수 ≥ 0
            assert v.ghost_streak - sum(e.kind is Kind.EATEN for e in outs[-1].events[b]) >= 0
        assert r.shape == (3,) and done.shape == (3,)
        dones_seen += int(done.sum())
        assert len(info["finished"]) == int(done.sum())
        for g in info["finished"]:
            assert g["winner"] in ("FLIES", "PACMAN", "CENSORED")
        if done.any():  # 새 판의 첫 관측: 프레임 4칸이 모두 같다
            b = int(np.flatnonzero(done)[0])
            fr = obs["grid"][b].reshape(4, NUM_CHANNELS, 31, 28)
            assert all(np.array_equal(fr[0], fr[i]) for i in range(1, 4))
    assert dones_seen > 0


def test_env_state_roundtrip():
    """저장한 환경 상태로 되돌리면 같은 행동열에 같은 궤적이 나온다 (체크포인트 재개의 바탕)."""
    env = PacmanTrainEnv(train_cfg(), 2, 3, ghosts="random", condition="regime1")
    env.reset()
    for _ in range(50):
        env.step(np.array([d.WEST, d.EAST]))
    blob = env.state_bytes()
    acts = [np.array([a % 4, (a + 1) % 4]) for a in range(200)]

    def run(e):
        out = []
        for a in acts:
            o, r, dn, _ = e.step(a)
            out.append((o["grid"].sum(), r.copy(), dn.copy()))
        return out

    first = run(env)
    env2 = PacmanTrainEnv(train_cfg(), 2, 99, ghosts="random", condition="regime1")
    env2.load_state_bytes(blob)
    second = run(env2)
    for (g1, r1, d1), (g2, r2, d2) in zip(first, second):
        assert g1 == g2 and np.array_equal(r1, r2) and np.array_equal(d1, d2)


def test_grid_is_binary():
    """PPO 버퍼는 격자를 uint8로 담는다: 격자 값은 0 · 1뿐이어야 한다."""
    env = PacmanTrainEnv(train_cfg(), 2, 0, ghosts="random", condition="regime1")
    obs = env.reset()
    for a in range(300):
        obs, *_ = env.step(np.array([a % 4, (a // 7) % 4]))
        assert set(np.unique(obs["grid"]).tolist()) <= {0.0, 1.0}
