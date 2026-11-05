import numpy as np
import pytest
from helpers import MAP, Scripted, SpyGhost, SpyRecorder, cell, make_cfg, place

from flypac.core import directions as d
from flypac.core.config import load_config
from flypac.core.events import Kind
from flypac.core.rng import make_rng
from flypac.core.runner import build_runner
from pathlib import Path


def runner_with(recorder=None, pacman=None, ghost_cls=SpyGhost, **over):
    cfg = make_cfg(**over)
    rec = recorder or SpyRecorder()
    r = build_runner(cfg, rec)
    if pacman is not None:
        r.pacman = pacman
    if ghost_cls is not None:
        r.ghosts = ghost_cls(cfg, r.B, make_rng(0, 0, "ghost"))
    r._reset_all()
    return r, rec


def g0(r):
    return r.game.games[0]


def test_full_game_runs_to_the_end():
    r = build_runner(make_cfg(seeds=[0, 1], n_games=2))
    out = r.run()
    assert len(out["games"]) == 4 and r.finished.all()
    assert {g["winner"] for g in out["games"]} <= {"FLIES", "PACMAN", "CENSORED"}


def test_infinite_loop_guard():
    r = build_runner(make_cfg(run={"max_loop_ticks": 5}))
    with pytest.raises(RuntimeError):
        r.run()


def test_same_seed_same_result():
    def go():
        r = build_runner(make_cfg(seeds=[3, 4], n_games=2))
        out = r.run()
        return out, r.ghosts.reward_total.copy(), r.odor.c.copy()

    a, b = go(), go()
    assert a[0] == b[0] and np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2])


def test_life_lost_in_pacman_phase_before_departure():
    # 팩맨 이동 단계(3번)에서 포획: 그 틱에 초파리는 움직이지 않고, 출발 전이라 보상 0
    r, rec = runner_with(pacman=Scripted(d.EAST))
    place(g0(r), pac=cell(11, 12))
    r.tick()
    catch = [e for c in rec.of("life_end") for e in c[3] if e.kind is Kind.CATCH][0]
    assert catch.active is False
    log = r.ghosts.log
    commit = [x for x in log if x[0] == "commit"][0]
    assert not commit[1].any()  # executed 전부 거짓
    assert [x for x in log if x[0] == "propose"][0][1].tolist() == [False]  # 동결 게임
    assert not r.ghosts.reward_total.any()  # 체제 보상 0
    assert len(rec.of("life_end")) == 1 and not rec.of("game_end")
    assert [x[0] for x in log].count("flush") == 1
    # 종료 상태는 초기화 전 (목숨 2, 동결)을 기록했고, 이후 목숨만 초기화됨
    view = rec.of("life_end")[0][2]
    assert view.lives == 2 and view.frozen
    now = r.game.views()[0]
    assert not now.frozen and now.lives == 2 and now.pac_cell == MAP.pacman_start


def test_last_life_two_events_one_processing():
    r, rec = runner_with(pacman=Scripted(d.EAST))
    g0(r).lives = 1
    place(g0(r), pac=cell(11, 12), released=True, active=True)
    r.tick()
    evs = rec.of("life_end")[0][3]
    assert {e.kind for e in evs} >= {Kind.LIFE_LOST, Kind.GAME_OVER}
    assert len(rec.of("life_end")) == 1 and len(rec.of("game_end")) == 1 and len(rec.of("replay_end")) == 1
    assert [x[0] for x in r.ghosts.log].count("flush") == 1  # 학습 1회
    flush_mask = [x for x in r.ghosts.log if x[0] == "flush"][0][1]
    assert flush_mask.tolist() == [True]
    assert rec.of("game_end")[0][2].lives == 0  # 초기화 전 상태
    assert r.game.views()[0].lives == 3 and r.game_no[0] == 2  # 판 초기화 1회
    assert r.results[0]["winner"] == "FLIES"


def test_all_dots_eaten_game_over_without_life_lost():
    r, rec = runner_with(pacman=Scripted(d.EAST))
    g = g0(r)
    place(g, pac=cell(5, 3))
    g.dots[:] = False
    g.energizers[:] = False
    g.dots[cell(5, 4)] = True
    r.tick()
    kinds = [e.kind for e in rec.of("life_end")[0][3]]
    assert Kind.LIFE_LOST not in kinds and Kind.GAME_OVER in kinds
    assert len(rec.of("game_end")) == 1 and r.results[0]["winner"] == "PACMAN"
    assert [x[0] for x in r.ghosts.log].count("flush") == 1


def test_last_dot_and_collision_same_tick_pacman_wins():
    r, rec = runner_with(pacman=Scripted(d.EAST))
    g = g0(r)
    place(g, pac=cell(5, 3), released=True, active=True, flies={0: cell(5, 4)})
    g.dots[:] = False
    g.energizers[:] = False
    g.dots[cell(5, 4)] = True
    r.tick()
    assert r.results[0]["winner"] == "PACMAN"
    assert Kind.CATCH not in [e.kind for e in rec.of("life_end")[0][3]]


def test_fly_phase_catch_blocks_later_flies_and_rewards_by_regime():
    r, rec = runner_with(pacman=Scripted(d.EAST))
    g = g0(r)
    place(g, pac=cell(5, 10), released=True, active=True,
          flies={0: cell(5, 1), 1: cell(5, 20), 2: cell(5, 22), 3: cell(5, 24)},
          fly_dirs={0: d.EAST, 1: d.WEST, 2: d.WEST, 3: d.WEST})
    g.pac_acc = -10.0
    # 1번(0부터) 초파리를 팩맨 바로 옆에 둔다
    g.fly_cell[1], g.fly_dir[1] = cell(5, 9), d.EAST
    r.ghosts.propose = lambda obs, alive, views=None: np.array([[-1, d.FORWARD, -1, -1]])
    r.tick()
    commit = [x for x in r.ghosts.log if x[0] == "commit"][0]
    assert commit[1][0].tolist() == [True, True, False, False]
    # 체제1: 잡은 놈 +1, 나머지 -p, 팀 합 1 - 3p
    R = r.ghosts.reward_total[0]
    assert R[1] == 1.0 and np.allclose(np.delete(R, 1), -0.25) and np.isclose(R.sum(), 1 - 3 * 0.25)


def test_regime2_team_sum_is_one():
    r, rec = runner_with(pacman=Scripted(d.EAST), condition="regime2")
    g = g0(r)
    g.lives = 3
    place(g, pac=cell(11, 12), released=True, active=True)
    r.tick()
    assert np.isclose(r.ghosts.reward_total[0].sum(), 1.0)


def test_on_release_is_game_level_and_views_withheld():
    r, rec = runner_with(pacman=Scripted(d.WEST), engine={"release_ticks": [0, 1, 2, 3], "pacman_speed": 0.0})
    for _ in range(5):
        r.tick()
    rel = [x for x in r.ghosts.log if x[0] == "on_release"]
    assert len(rel) == 1 and rel[0][1].shape == (1,) and rel[0][1].tolist() == [True]
    assert all(x[2] is None for x in r.ghosts.log if x[0] == "propose")  # 원칙 2


def test_views_only_to_controllers_that_use_them():
    class B1Spy(SpyGhost):
        uses_views = True

    r, rec = runner_with(ghost_cls=B1Spy)
    r.tick()
    got = [x for x in r.ghosts.log if x[0] == "propose"][0][2]
    assert got is not None and len(got) == 1


def test_views_refreshed_after_pacman_move():
    r, rec = runner_with(pacman=Scripted(d.EAST))
    place(g0(r), pac=cell(5, 3))
    r.tick()
    assert g0(r).pac_cell == cell(5, 4)
    frames = rec.of("events")
    assert frames[0][3].pac_cell == cell(5, 4)  # 기록에 넘긴 views가 이동 뒤 사본


def test_odor_not_stepped_for_frozen_game_and_reset_on_end():
    r, rec = runner_with(pacman=Scripted(d.EAST))
    place(g0(r), pac=cell(11, 12))
    r.tick()  # 포획 → 목숨 초기화 → 냄새 장 초기화
    assert r.odor.total()[0] == 0.0
    r.tick()
    assert r.odor.total()[0] > 0.0


def test_smoke_config_file_runs():
    cfg = load_config(Path(__file__).parent.parent / "configs" / "m0_smoke.yaml")
    out = build_runner(cfg).run()
    assert len(out["games"]) == len(cfg.seeds) * cfg.n_games
