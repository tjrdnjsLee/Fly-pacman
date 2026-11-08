"""재생 기록이 배열 참조가 아니라 복사본을 쌓는지 (점검 F01)."""

from __future__ import annotations

import numpy as np
from helpers import MAP, make_cfg

from flypac.core import directions as d
from flypac.ghosts.random_ghost import RandomGhost
from flypac.core.runner import build_runner
from flypac.recording.recorder import Recorder
from flypac.replay.reader import read_replay
from flypac.replay.synth import make_synthetic_arrays
from flypac.replay.writer import ReplayBuffer


def _frame(odor, action):
    F = 4
    return dict(pac_cell=0, pac_dir=0, pac_acc=0.0, fly_cell=np.zeros(F, int), fly_dir=np.zeros(F, int),
                fly_acc=np.zeros(F), fly_z=np.zeros(F, bool), fly_in_house=np.zeros(F, bool), lives=3,
                power_ticks=0, score=0, fruit=0, odor_field=odor, sense=np.zeros((F, 4, 2)),
                action=action, pi=np.zeros((F, 4)), committed=np.zeros(F, bool), moved=np.zeros(F, bool))


def test_buffer_keeps_values_when_source_changes_in_place():
    odor = np.zeros(320)
    action = np.array([1, -1, 2, -1])
    buf = ReplayBuffer()
    buf.add_frame(**_frame(odor, action))
    odor += 5.0          # 냄새 장이 제자리에서 갱신된다
    action[:] = -1       # 조종기가 목숨 초기화로 제안 배열을 고친다
    buf.add_frame(**_frame(odor, action))
    cols = buf._cols
    assert cols["odor_field"][0].sum() == 0.0 and cols["odor_field"][1].sum() == 5.0 * 320
    assert list(cols["action"][0]) == [1, -1, 2, -1]


def test_synthetic_first_frame_has_zero_odor_mass():
    arr = make_synthetic_arrays(T=50, seed=0)
    mass = arr["odor_field"].astype(np.float64).sum(axis=1)
    assert mass[0] == 0.0          # 첫 프레임은 냄새 장 초기화 직후
    assert mass[-1] > 0.0 and np.all(np.diff(mass[:10]) > 0)   # 과거 프레임이 마지막 값으로 덮이지 않음


def test_random_ghost_reset_life_does_not_modify_returned_proposal():
    cfg = make_cfg(seeds=[0], n_games=1)
    r = build_runner(cfg)
    r._reset_all()
    for _ in range(400):           # 결정이 하나라도 나올 때까지
        r.game.views()
        info = r.game.decision_info()
        from flypac.senses.sense import sense
        obs = sense(r.odor.c, np.zeros_like(r.odor.c), r.game.views(), info, 0.01, nbr=r.map.nbr)
        out = r.ghosts.propose(obs, np.ones(1, bool))
        if (out >= 0).any():
            break
        r.tick()
    kept = out.copy()
    assert (kept >= 0).any()
    r.ghosts.reset_life(np.ones(1, bool))
    assert np.array_equal(out, kept)
    assert (r.ghosts.proposed == -1).all()


class FirstLegal(RandomGhost):
    """결정할 때 가장 작은 가능한 자기중심 행동을 고르는 조종기 (장면 고정용)."""

    def propose(self, obs, alive, views=None):
        out = np.full((self.B, self.F), -1, dtype=np.int64)
        for b in range(self.B):
            for k in range(self.F):
                if alive[b] and obs.decide[b, k]:
                    out[b, k] = int(np.flatnonzero(obs.legal[b, k])[0])
        self.proposed = out
        return out


def test_capture_tick_action_is_preserved_in_replay(tmp_path):
    """포획 틱에 결정한 행동이 목숨 초기화 뒤에도 재생 로그에 그대로 남는다.

    목숨이 끝나도 판은 이어지는 경우라야 프레임이 버퍼에 남은 채 초기화를 겪는다 (목숨 2, 판은 감시 장치로 끝냄).
    """
    cfg = make_cfg(seeds=[0], n_games=1, replay={"keep_games": [1]},
                   engine={"lives": 2, "pacman_speed": 0.0, "watchdog_ticks": 40})
    rec = Recorder(tmp_path / "run", cfg)
    r = build_runner(cfg, rec)
    r.ghosts = FirstLegal(cfg, r.B, r.ghosts.rngs)
    r._reset_all()  # 판 1 시작(reset_game(rng)) 뒤에 장면을 놓는다
    g = r.game.games[0]
    g.released[:] = [True, False, False, False]
    found = False
    for J in range(MAP.nbr.shape[0]):                 # 엔진이 결정을 요구하는 갈림길 (칸, 방향)을 찾는다
        for h in range(4):
            g.fly_cell[0], g.fly_dir[0] = J, h
            legal, decide = r.game.decision_info()
            if decide[0, 0]:
                found = True
                break
        if found:
            break
    assert found
    ego = int(np.flatnonzero(legal[0, 0])[0])
    g.pac_cell = int(MAP.nbr[J, int(d.ego_to_abs(h, ego))])
    while not r.finished.all():
        r.tick()
    rec.close()
    meta, arr = read_replay(tmp_path / "run" / "replay" / "regime1_s0_g1.npz")
    assert 0 in arr["ev_kind"].tolist()                # CATCH (코드 0)
    assert arr["ev_tick"][arr["ev_kind"] == 0][0] == 0  # 첫 틱에 포획 (판은 계속)
    assert len(arr["action"]) > 1
    assert arr["action"][0, 0] == ego                 # 목숨 초기화(-1)로 덮이지 않았다
