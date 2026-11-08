"""엔진 · 팩맨 난수의 판별 갱신과 공용 틱 처리 (DECISIONS 3절 · 6.1절 · 11절, 관리자 결정 2026-11-08).

- 판을 시작할 때마다(판 1 포함) 엔진은 reset_game(rng)로 make_rng(시드, 판 번호, "engine")을 받는다.
- 그래서 같은 시드 · 같은 판의 엔진 난수열은 묶음 크기 · 칸 순서와 무관하다.
- 팩맨 조종기의 칸별 난수(GameRngs "pacman")도 판마다 다시 만든다.
- 한 틱은 begin → finish 순서로만 돈다. 실행기와 학습 환경은 같은 TickCore를 쓴다.
"""

from __future__ import annotations

import numpy as np
import pytest
from helpers import MAP, make_cfg

from flypac.core.config import Config
from flypac.core.rng import game_rngs, make_rng
from flypac.core.runner import build_runner
from flypac.core.tick import TickCore
from flypac.engine.fake import FakeGame, FakeVecGame
from flypac.ghosts.random_ghost import RandomGhost
from flypac.pacman.rule_based import RuleBasedPacman


class ProbeGame(FakeGame):
    """가짜 엔진은 난수를 쓰지 않으므로, 매 틱 엔진 난수에서 하나를 뽑아 남기는 대역."""

    def tick_timers(self):
        self.last_draw = float(self.rng.random())
        return super().tick_timers()


def _engine_streams(monkeypatch, seeds, n_games=2):
    """(시드, 판 번호)별 엔진 난수열 (매 틱 한 개)."""
    monkeypatch.setattr(FakeVecGame, "game_cls", ProbeGame)
    r = build_runner(make_cfg(seeds=list(seeds), n_games=n_games))
    streams: dict[tuple[int, int], list[float]] = {}
    tick = r.tick

    def spy():
        game_no, running = r.game_no.copy(), ~r.finished
        tick()
        for b, s in enumerate(seeds):
            if running[b]:
                streams.setdefault((int(s), int(game_no[b])), []).append(r.game.games[b].last_draw)

    r.tick = spy
    r.run()
    return streams


def test_engine_rng_stream_same_regardless_of_batch_size_and_order(monkeypatch):
    three = _engine_streams(monkeypatch, [5, 6, 7])
    two = _engine_streams(monkeypatch, [7, 5])
    assert {k[1] for k in three} == {1, 2}  # 판 경계를 지나야 시험이 의미 있다
    for key in two:
        assert np.array_equal(two[key], three[key]), key
    # 각 판의 난수열은 (시드, 판 번호, "engine")에서 새로 시작한다 (앞 판에서 이어지지 않는다)
    for (s, n), xs in three.items():
        assert np.array_equal(xs, make_rng(s, n, "engine").random(len(xs))), (s, n)


@pytest.mark.parametrize("kind", ["fake", "real"])
def test_every_game_start_gives_engine_and_pacman_fresh_rngs(kind):
    """엔진 종류와 무관하게: 판 시작마다 reset_game이 받은 생성기 = (시드, 판 번호, 용도). real은 엔진이 생기면 돈다."""
    d = make_cfg(seeds=[3, 4], n_games=3).to_dict()
    d["engine"] = {**d["engine"], "kind": kind}
    try:
        r = build_runner(Config(d))
    except NotImplementedError:
        pytest.skip(f"engine.kind={kind} 엔진이 아직 없다")
    seeds = [3, 4]
    starts: list[tuple[int, int]] = []
    reset_game = r.game.reset_game

    def spy(mask, rngs=None):
        assert rngs is not None, "reset_game에 엔진 난수를 주지 않았다"
        for b in np.flatnonzero(mask):
            n = int(r.game_no[b])
            starts.append((seeds[b], n))
            assert rngs[b].bit_generator.state == make_rng(seeds[b], n, "engine").bit_generator.state
            assert r.pacman.rngs[b].bit_generator.state == make_rng(seeds[b], n, "pacman").bit_generator.state
        return reset_game(mask, rngs)

    r.game.reset_game = spy
    r.run()
    assert sorted(starts) == [(s, n) for s in seeds for n in (1, 2, 3)]


def test_finish_without_begin_and_double_begin_are_errors():
    r = build_runner(make_cfg())
    r._reset_all()
    with pytest.raises(RuntimeError, match="begin 없이"):
        r.finish(np.zeros(r.B, dtype=np.int64))
    views = r.begin()
    with pytest.raises(RuntimeError, match="두 번"):
        r.begin()
    r.finish(r.pacman.act(views))
    with pytest.raises(RuntimeError, match="begin 없이"):
        r.finish(np.zeros(r.B, dtype=np.int64))


def test_training_style_core_matches_runner_without_fields_and_records():
    """냄새 장 · 기록 · 체제 보상을 끈 TickCore(학습 환경 방식: 행동은 밖에서)가 실행기와 같은 판을 만든다.

    무작위 유령은 냄새를 쓰지 않으므로 장을 꺼도 궤적이 같아야 한다. n_games=None이면 판이 끝없이 이어진다.
    """
    seeds = [0, 1]
    cfg = make_cfg(seeds=seeds, n_games=2)
    want = build_runner(cfg).run()["games"]

    engine_rngs = game_rngs(seeds, "engine")
    game = FakeVecGame(len(seeds), MAP, cfg, [engine_rngs[b] for b in range(len(seeds))])
    ghosts = RandomGhost(cfg, len(seeds), game_rngs(seeds, "ghost"))
    policy = RuleBasedPacman(MAP, cfg, game_rngs(seeds, "pacman"))  # 학습 환경의 PPO 자리
    core = TickCore(MAP, game, engine_rngs, ghosts, condition="regime1")  # pacman=None, 장 · 기록 · 체제 없음
    core.start()
    got, started = [], 0
    while len(core.results) < 6:
        views = core.begin()
        out = core.finish(policy.act(views))
        policy.reset(out.ends)  # 프레임 이력 같은 것은 밖에서 지운다
        started += int(out.started.sum())
        for b in np.flatnonzero(out.started):
            assert game.games[b].view().tick == 0 and core.tick_no[b] == 0
    got = [g for g in core.results if g["game_no"] <= 2]
    assert sorted(got, key=lambda g: (g["slot"], g["game_no"])) == \
        sorted(want, key=lambda g: (g["slot"], g["game_no"]))
    assert started == 6 and not core.finished.any()  # 판 3, 4도 시작했다 (끝이 없다)
