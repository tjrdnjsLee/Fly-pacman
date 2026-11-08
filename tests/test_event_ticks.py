"""불변식: 사건의 틱 번호 = 그 사건이 기록된 재생 프레임 번호 (DECISIONS 8절, 관리자 결정 2026-11-08).

틱 번호의 주인은 실행기 하나다. 실행기는 동결된 틱을 포함해 매 틱 번호를 올리고, 새 판에서 0으로 돌린다.
사건의 틱 번호는 실행기의 번호를 쓴다.

엔진에 묶이지 않는다: 연기 시험 설정을 끝까지 돌리며 기록기 자리에서 본다. 기록기는 칸마다 현재 판의 프레임을
틱당 하나 쌓으므로(on_events → replay_frame 순서), on_events 때 이미 쌓인 프레임 수가 이번 틱의 프레임 번호다.
engine.kind별로 돌고, 진짜 엔진이 아직 없으면(NotImplementedError) 건너뛴다 (불변식 시험과 같은 방식).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flypac.core.config import Config, load_config
from flypac.core.events import Kind
from flypac.core.runner import NullRecorder, build_runner

SMOKE = Path(__file__).resolve().parents[1] / "configs" / "m0_smoke.yaml"


class FrameSpy(NullRecorder):
    """칸별 현재 판의 프레임 수를 세고, 사건마다 (틱 번호, 프레임 번호)를 남긴다."""

    def __init__(self, B: int):
        self.frames = [0] * B
        self.pairs: list[tuple[int, int, int, str]] = []   # (칸, 사건 틱, 프레임 번호, 종류)
        self.game_lengths: list[tuple[int, int, int]] = []  # (칸, 프레임 수, 마지막 사건 틱)
        self.last_tick = [-1] * B

    def on_events(self, game, events, view):
        for e in events:
            self.pairs.append((game, int(e.tick), self.frames[game], str(e.kind)))
            self.last_tick[game] = int(e.tick)

    def replay_frame(self, game, views, obs, odor, phero, brain):
        self.frames[game] += 1

    def replay_end(self, game, keep):
        self.game_lengths.append((game, self.frames[game], self.last_tick[game]))
        self.frames[game] = 0
        self.last_tick[game] = -1


def _cfg(kind: str) -> Config:
    d = load_config(SMOKE).to_dict()
    d["engine"] = {**d["engine"], "kind": kind}
    return Config(d)


@pytest.mark.parametrize("kind", ["fake", "real"])
def test_event_tick_equals_replay_frame_index(kind):
    cfg = _cfg(kind)
    spy = FrameSpy(len(cfg.seeds))
    try:
        r = build_runner(cfg, spy)
    except NotImplementedError:
        pytest.skip(f"engine.kind={kind} 엔진이 아직 없다")
    r.run()
    bad = [p for p in spy.pairs if p[1] != p[2]]
    assert not bad, f"사건 틱 ≠ 프레임 번호 (칸, 사건 틱, 프레임, 종류) 처음 몇 개: {bad[:5]}"
    # 판을 끝낸 사건(GAME_OVER)은 그 판의 마지막 프레임에 있다
    assert all(last == n - 1 for _, n, last in spy.game_lengths), spy.game_lengths
    # 시험이 의미 있으려면 판 도중에 목숨을 잃고(동결 틱) 그 뒤에도 사건이 있어야 한다
    kinds_after_life_lost = [
        p for i, p in enumerate(spy.pairs)
        if any(q[0] == p[0] and q[3] == Kind.LIFE_LOST and q[1] < p[1] for q in spy.pairs[:i])
    ]
    assert kinds_after_life_lost, "판 도중 목숨을 잃은 뒤의 사건이 없다: 설정을 바꿔야 한다"


def test_runner_stops_engine_that_skips_frozen_ticks(monkeypatch):
    """엔진의 틱 번호가 실행기와 어긋나면(동결 틱을 세지 않는 옛 동작) 실행기가 멈춘다."""
    from flypac.engine.fake import FakeGame

    orig = FakeGame.tick_timers

    def old_tick_timers(self):
        if self.frozen:
            return []  # 고치기 전 동작: 동결 틱에 번호를 올리지 않음
        return orig(self)

    monkeypatch.setattr(FakeGame, "tick_timers", old_tick_timers)
    with pytest.raises(RuntimeError, match="엔진 틱"):
        build_runner(_cfg("fake")).run()
