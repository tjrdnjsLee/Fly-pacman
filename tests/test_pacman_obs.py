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
from flypac.pacman.obs import NUM_CHANNELS, NUM_SCALARS, FrameStack, ObsEncoder, legal_mask
from flypac.pacman.rewards import RewardFn
from flypac.pacman.train_env import PacmanTrainEnv
from helpers import BASE, MAP

TRAIN = {
    "seed": 0, "num_envs": 3, "device": "cpu", "run_dir": "unused",
    "obs": {"frame_stack": 4, "power_ticks_max": 57, "mask_reverse": True},
    "curriculum": [{"from_update": 0, "ghosts": "random"}],
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
    env = PacmanTrainEnv(train_cfg(), 1, 0, ghosts="random")
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


def test_rewards():
    fn = RewardFn(TRAIN["rewards"], MAP.cell_rc)
    v = replace(start_view(), fly_in_house=np.ones(4, dtype=bool))  # 근접 항 없음
    ev = lambda k, fly=-1, cause=None: Event(k, 0, 0, fly, False, cause)  # noqa: E731
    r, s = fn([ev(Kind.DOT)], v, 0)
    assert r == pytest.approx(1.0 - 0.01) and s == 0
    r, s = fn([ev(Kind.ENERGIZER)], v, 3)
    assert r == pytest.approx(2.0 - 0.01) and s == 0  # 새 파워: 연속 수를 지운다
    r, s = fn([ev(Kind.EATEN, 0), ev(Kind.EATEN, 1)], v, 0)
    assert r == pytest.approx(5 + 10 - 0.01) and s == 2
    r, s = fn([ev(Kind.EATEN, 0)], v, 7)
    assert r == pytest.approx(20 - 0.01)  # 마지막 값에서 멈춘다
    r, _ = fn([ev(Kind.CATCH, 0), ev(Kind.LIFE_LOST), ev(Kind.GAME_OVER, cause=Winner.FLIES)], v, 0)
    assert r == pytest.approx(-10 - 25 - 0.01)
    r, _ = fn([ev(Kind.DOT), ev(Kind.GAME_OVER, cause=Winner.PACMAN)], v, 0)
    assert r == pytest.approx(1 + 50 - 0.01)
    r, _ = fn([ev(Kind.GAME_OVER, cause=Winner.CENSORED)], v, 0)
    assert r == pytest.approx(-0.01)


def test_proximity_reward():
    fn = RewardFn(TRAIN["rewards"], MAP.cell_rc)
    v = start_view()
    pac = v.pac_cell
    near = int(MAP.nbr[pac][MAP.nbr[pac] >= 0][0])  # 1칸 옆
    cells = np.array([near, near, pac, pac])
    v2 = replace(v, fly_cell=cells, fly_in_house=np.array([False, False, False, True]),
                 fly_z=np.array([False, True, False, False]))
    r, _ = fn([], v2, 0)
    assert r == pytest.approx(-0.01 - 0.3 * 2)  # 0번(위험, 1칸), 2번(위험, 0칸). 1번은 z=1, 3번은 집 안


def test_env_runs_and_resets():
    env = PacmanTrainEnv(train_cfg(), 3, 0, ghosts="random")
    obs = env.reset()
    assert obs["grid"].shape == (3, 32, 31, 28) and obs["scalars"].shape == (3, 5)
    rng = np.random.default_rng(0)
    dones_seen = 0
    for _ in range(3000):
        legal = env.legal_masks()
        acts = np.array([rng.choice(np.flatnonzero(m)) for m in legal])
        obs, r, done, info = env.step(acts)
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
    env = PacmanTrainEnv(train_cfg(), 2, 3, ghosts="random")
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
    env2 = PacmanTrainEnv(train_cfg(), 2, 99, ghosts="random")
    env2.load_state_bytes(blob)
    second = run(env2)
    for (g1, r1, d1), (g2, r2, d2) in zip(first, second):
        assert g1 == g2 and np.array_equal(r1, r2) and np.array_equal(d1, d2)


def test_grid_is_binary():
    """PPO 버퍼는 격자를 uint8로 담는다: 격자 값은 0 · 1뿐이어야 한다."""
    env = PacmanTrainEnv(train_cfg(), 2, 0, ghosts="random")
    obs = env.reset()
    for a in range(300):
        obs, *_ = env.step(np.array([a % 4, (a // 7) % 4]))
        assert set(np.unique(obs["grid"]).tolist()) <= {0.0, 1.0}
