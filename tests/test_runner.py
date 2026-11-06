import numpy as np
import pytest
from helpers import MAP, Scripted, SpyGhost, SpyRecorder, cell, make_cfg, place, share_journal

from flypac.core import directions as d
from flypac.core.config import load_config
from flypac.core.events import Event, Kind
from flypac.core.rng import game_rngs
from flypac.core.runner import build_runner
from pathlib import Path


def runner_with(recorder=None, pacman=None, ghost_cls=SpyGhost, **over):
    cfg = make_cfg(**over)
    rec = recorder or SpyRecorder()
    r = build_runner(cfg, rec)
    if pacman is not None:
        r.pacman = pacman
    if ghost_cls is not None:
        r.ghosts = ghost_cls(cfg, r.B, game_rngs(cfg.seeds, "ghost"))
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
    assert not commit[1].any()  # committed 전부 거짓
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
    # 0번은 복도 이동(-1)이라 확정 아님, 1번은 결정한 뒤 이동한 칸에서 포획: 확정 (DECISIONS 4절의 표)
    assert commit[1][0].tolist() == [False, True, False, False]
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


class B1Spy(SpyGhost):
    uses_views = True


def test_views_given_only_under_condition_b1():
    r, rec = runner_with(condition="b1", ghost_cls=B1Spy)
    r.tick()
    got = [x for x in r.ghosts.log if x[0] == "propose"][0][2]
    assert got is not None and len(got) == 1


def test_views_user_is_an_error_at_start_outside_b1():
    # uses_views가 원칙 2를 우회하지 못한다: regime1에서 views를 쓰는 조종기는 시작 때 오류
    with pytest.raises(ValueError, match="uses_views"):
        runner_with(condition="regime1", ghost_cls=B1Spy)
    r = build_runner(make_cfg(condition="regime1"))
    r.ghosts = B1Spy(r.cfg, r.B, game_rngs(r.cfg.seeds, "ghost"))
    with pytest.raises(ValueError):
        r.run()


def test_b1_with_a_controller_that_ignores_views_is_an_error_at_start():
    with pytest.raises(ValueError, match="uses_views"):
        runner_with(condition="b1", ghost_cls=SpyGhost)
    with pytest.raises(ValueError):
        build_runner(make_cfg(condition="b1")).run()  # 무작위 초파리는 views를 쓰지 않는다


@pytest.mark.parametrize("condition", ["regime1", "regime2", "regime3", "b0", "b0p"])
def test_views_withheld_under_every_non_b1_condition(condition):
    r, rec = runner_with(condition=condition)
    r.tick()
    assert all(x[2] is None for x in r.ghosts.log if x[0] == "propose")


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


def test_on_release_only_for_the_game_that_released_all_flies():
    # B=2: 게임 0은 이미 모두 나와 있고, 게임 1만 이번 틱에 ALL_RELEASED가 난다
    r, rec = runner_with(seeds=[0, 1], pacman=Scripted(d.WEST),
                         engine={"release_ticks": [0, 0, 0, 0], "pacman_speed": 0.0})
    place(g0(r), released=True, active=True)
    r.tick()
    rel = [x for x in r.ghosts.log if x[0] == "on_release"]
    assert len(rel) == 1 and rel[0][1].tolist() == [False, True]
    assert r.ghosts.n_release.tolist() == [0, 1]


def test_on_release_called_even_though_all_released_event_is_inactive():
    r, rec = runner_with(pacman=Scripted(d.WEST), engine={"release_ticks": [0, 0, 0, 0], "pacman_speed": 0.0})
    r.tick()
    ev = [e for c in rec.of("events") for e in c[2] if e.kind is Kind.ALL_RELEASED]
    assert len(ev) == 1 and ev[0].active is False
    assert [x[0] for x in r.ghosts.log].count("on_release") == 1


class TraceGhost(SpyGhost):
    """흔적 e를 흉내 내는 가짜 뇌. on_release가 e를 0으로 만들고, commit은 실행된 활성 결정만 더한다."""

    def __init__(self, cfg, B, rng):
        super().__init__(cfg, B, rng)
        self.e = np.zeros((B, 4), dtype=int)

    def on_release(self, mask):
        self.e[mask] = 0
        super().on_release(mask)

    def commit(self, committed, active):
        self.e += committed & active
        super().commit(committed, active)


def test_first_active_decisions_after_all_released_survive_on_release():
    # 0번이 마지막에 나온다: 같은 틱에 1~3번이 ALL_RELEASED 뒤 활성 상태로 움직인다
    r, rec = runner_with(ghost_cls=TraceGhost, pacman=Scripted(d.WEST),
                         engine={"release_ticks": [1, 0, 0, 0], "pacman_speed": 0.0})
    r.tick()  # 틱 0: 1~3번이 나온다. 아직 비활성
    assert not r.ghosts.e.any()
    start = len(r.ghosts.log)
    r.tick()  # 틱 1: 0번이 나오며 ALL_RELEASED, 1~3번은 활성으로 이동
    names = [x[0] for x in r.ghosts.log[start:]]
    assert names.index("on_release") < names.index("commit")  # on_release → commit
    commit = [x for x in r.ghosts.log[start:] if x[0] == "commit"][0]
    assert commit[1][0].tolist() == [False, True, True, True]  # 0번의 출구 통과는 확정이 아니다
    assert commit[2][0].tolist() == [False, True, True, True]  # 0번의 출구 통과는 활성 이전
    assert r.ghosts.e[0].tolist() == [0, 1, 1, 1]  # 흔적이 지워지지 않았다


def test_on_eaten_runs_after_commit():
    r, rec = runner_with(pacman=Scripted(d.WEST), engine={"release_ticks": [0, 0, 0, 0], "pacman_speed": 0.0})
    eaten_calls = []
    r.ghosts.on_eaten = lambda mask: eaten_calls.append([x[0] for x in r.ghosts.log])
    g = g0(r)
    orig = g.step_flies

    def with_eaten(actions):
        ev, mv, cm, ac = orig(actions)
        return ev + [Event(Kind.EATEN, g.index, g.tick, 2, ac[2])], mv, cm, ac

    g.step_flies = with_eaten
    r.tick()
    assert len(eaten_calls) == 1
    assert eaten_calls[0][-2:] == ["on_release", "commit"]


@pytest.mark.parametrize("lives", [1, 3])
def test_end_of_tick_order_flush_record_reset(lives):
    r, rec = runner_with(pacman=Scripted(d.EAST))
    journal = share_journal(r, rec)
    g = g0(r)
    g.lives = lives
    place(g, pac=cell(11, 12), released=True, active=True, flies={0: cell(11, 13)})
    r.ghosts.probe = lambda: g.learning_active
    del journal[:]  # 실행 시작 때의 초기화 호출은 뺀다 (같은 목록 객체를 유지한다)
    r.tick()
    watched = ("flush", "life_end", "game_end", "replay_end", "reset_life", "game_reset_life", "game_reset_game")
    names = [c[0] for c in journal if c[0] in watched]
    if lives == 1:  # 마지막 목숨: 목숨 기록 + 판 기록, 판 초기화 1회
        assert names == ["flush", "life_end", "game_end", "replay_end", "reset_life", "game_reset_game"]
    else:
        assert names == ["flush", "life_end", "reset_life", "game_reset_life"]
    # 마지막 학습과 종료 기록은 학습 활성이 켜진 채로, 해제는 초기화 단계에서
    flush = [c for c in journal if c[0] == "flush"][0]
    assert flush[2] is True
    assert [c for c in journal if c[0] == "life_end"][0][2].learning_active is True
    if lives == 1:
        assert [c for c in journal if c[0] == "game_end"][0][2].learning_active is True
    assert g.learning_active is False


def test_finished_slot_is_retired_not_restarted():
    r, rec = runner_with(pacman=Scripted(d.EAST), n_games=1)
    journal = share_journal(r, rec)
    g = g0(r)
    g.lives = 1
    place(g, pac=cell(11, 12), released=True, active=True, flies={0: cell(11, 13)})
    del journal[:]
    r.tick()
    assert r.finished[0] and r.game_no[0] == 1
    assert [c[0] for c in journal].count("game_retire") == 1
    assert not [c for c in journal if c[0] == "game_reset_game"]  # 새 판은 시작하지 않는다
    v = r.game.views()[0]
    assert v.learning_active is False and v.frozen and v.lives == 0


def test_replay_frame_gets_fields_of_step_5():
    # 한 프레임의 views, obs, 냄새 장 · 페로몬 장이 같은 순간(5번)이다. 10번의 갱신이 새어 들어가면 안 된다
    r, rec = runner_with(pacman=Scripted(d.WEST), condition="regime3")
    r.tick()
    odor_before, phero_before = r.odor.c.copy(), r.phero.c.copy()
    assert odor_before.any()
    r.tick()
    _, _, frame_odor, frame_phero = rec.of("frame")[-1]
    assert np.array_equal(frame_odor, odor_before) and np.array_equal(frame_phero, phero_before)
    assert not np.array_equal(frame_odor, r.odor.c)  # 이 틱의 10번 갱신은 반영되지 않았다
    assert not np.shares_memory(frame_odor, r.odor.c) and not np.shares_memory(frame_phero, r.phero.c)


@pytest.mark.parametrize("condition", ["b0", "b0p"])
def test_baseline_conditions_run_with_zero_reward(condition):
    r = build_runner(make_cfg(condition=condition, seeds=[0, 1], n_games=1))
    out = r.run()
    assert len(out["games"]) == 2 and not r.ghosts.reward_total.any()
