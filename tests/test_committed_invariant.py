"""불변식: committed이면 그 틱에 결정을 물었고(decide 참) 행동 ≥ 0이다 (DECISIONS 4.1절 표, 점검 F04).

엔진에 묶이지 않는다: 연기 시험 설정으로 실행기를 끝까지 돌리며, 실행기 6번(propose)이 받은 관측 ·
낸 행동과 7번(step_flies)이 돌려준 moved · committed를 매 틱 대조한다. 엔진 내부 상태는 보지 않는다.
engine.kind별로 같은 시험을 돌린다. 진짜 엔진이 아직 없으면(NotImplementedError) 건너뛴다.

4.1절 표의 행별 조건 (b, k, 틱 t; decide · z는 t의 5번 관측, 행동은 t의 6번 제안):
- committed ⇒ decide 참 그리고 행동 ≥ 0                        (행 1, 2, 7, 11. 모든 확정의 필요조건)
- committed 그리고 moved ⇒ 위와 같다                            (행 1, 2, 11: 이동하며 확정)
- committed 그리고 moved 아님 ⇒ 지연 반전을 **저장하는 틱**(행 7):
  decide 참, 행동 ≥ 0, 그리고 결정 지점 ②여야 하므로 그 초파리의 z가 직전 틱 관측과 다르다
  (z는 같은 목숨 안에서 비교한다. 목숨이 바뀐 첫 틱은 직전 값이 없다)
- 행동 -1 ⇒ committed 거짓                                       (행 3, 4, 8, 9: 복도 · 집 출구 · 지연 반전 적용 · 대기)
- 동결된 게임(alive 거짓) ⇒ moved · committed 모두 거짓          (행 6)
- 결정 시점 ① (M1 뇌 ② 지시서 2.3, DECISIONS 12.2): 반전 기회가 아닌(z가 직전 틱과 같은) 결정 틱에서 decide이면
  그 초파리의 이동 차례까지 게임이 끝나지 않았을 때 그 틱에 moved. 앞 초파리의 포획(CATCH)으로 이동 단계
  도중에 끝난 게임은 그 초파리보다 뒤 번호만 예외다. 즉 decide는 '이번 틱에 이동한다'를 함의한다.
조종기는 무작위 초파리와 초파리 뇌 둘 다로 돌린다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from flypac.core.config import Config, load_config
from flypac.core.events import Kind
from flypac.core.runner import build_runner

SMOKE = Path(__file__).resolve().parents[1] / "configs" / "m0_smoke.yaml"


def _cfg(kind: str, fly_speed: float | None = None, ghosts: str = "random") -> Config:
    d = load_config(SMOKE).to_dict()
    d["engine"] = {**d["engine"], "kind": kind}
    if ghosts == "flybrain":
        d["ghosts"] = {"kind": "flybrain", "config": "configs/brain_v1.yaml"}
    if fly_speed is not None:
        d["engine"]["fly_speed"] = fly_speed
    return Config(d)


def _runner_or_skip(cfg):
    try:
        return build_runner(cfg)
    except NotImplementedError:
        pytest.skip(f"engine.kind={cfg.engine.kind} 엔진이 아직 없다")


@pytest.mark.parametrize("ghosts", ["random", "flybrain"])
@pytest.mark.parametrize("kind", ["fake", "real"])
@pytest.mark.parametrize("fly_speed", [None, 0.75, 0.6])  # 연기 설정 그대로, 그리고 속도 부족 틱이 생기는 값
def test_committed_implies_decide_and_action_every_tick(kind, fly_speed, ghosts):
    r = _runner_or_skip(_cfg(kind, fly_speed, ghosts))
    seen = {"tick": None}
    stats = {"committed": 0, "moved_not_committed": 0, "ticks": 0, "decide_point1": 0, "ended_before_turn": 0}
    prev_z = {}          # 칸 b → 직전 틱 관측의 z (같은 목숨 안에서만)
    life_key = {}        # 칸 b → (판 번호, 목숨 수): 목숨이 바뀌면 직전 z를 버린다

    propose = r.ghosts.propose

    def spy_propose(obs, alive, views=None):
        act = propose(obs, alive, views)
        seen["tick"] = (obs.decide.copy(), obs.z.copy(), np.asarray(alive).copy(), np.asarray(act).copy())
        return act

    step_flies = r.game.step_flies

    def spy_step_flies(actions):
        events, moved, committed, active = step_flies(actions)
        decide, z, alive, act = seen["tick"]
        assert np.array_equal(np.asarray(actions), act), "실행기가 제안과 다른 행동을 넘겼다"
        moved, committed = np.asarray(moved, bool), np.asarray(committed, bool)
        views = r.game.views()
        for b in range(r.B):
            key = (int(r.game_no[b]), int(views[b].lives))
            z_before = prev_z.get(b) if life_key.get(b) == key else None
            # 이동 단계 도중 포획으로 끝났으면, 잡은 초파리보다 뒤 번호는 차례가 오지 않았다
            catchers = [e.fly for e in events[b] if e.kind is Kind.CATCH]
            last_turn = min(catchers) if catchers else r.F - 1
            for k in range(r.F):
                where = f"틱 {stats['ticks']}, 칸 {b}, 초파리 {k}"
                if not alive[b]:
                    assert not moved[b, k] and not committed[b, k], f"{where}: 동결 게임인데 이동 · 확정"
                if act[b, k] < 0:
                    assert not committed[b, k], f"{where}: 행동 -1인데 확정"
                reversal = z_before is not None and z[b, k] != z_before[k]
                if alive[b] and decide[b, k] and not reversal:  # 결정 시점 ①
                    if k <= last_turn:
                        assert moved[b, k], f"{where}: 결정 시점 ①인데 이번 틱에 이동하지 않았다"
                        stats["decide_point1"] += 1
                    else:
                        stats["ended_before_turn"] += 1
                if committed[b, k]:
                    assert decide[b, k], f"{where}: decide 거짓인데 확정"
                    assert act[b, k] >= 0, f"{where}: 행동 {act[b, k]}인데 확정"
                    stats["committed"] += 1
                    if not moved[b, k]:  # 지연 반전 저장 틱 (표 7)
                        assert z_before is not None and z[b, k] != z_before[k], \
                            f"{where}: 움직이지 않고 확정했는데 z가 바뀐 다음 틱(결정 지점 ②)이 아니다"
                elif moved[b, k]:
                    stats["moved_not_committed"] += 1
            prev_z[b], life_key[b] = z[b].copy(), key
        stats["ticks"] += 1
        return events, moved, committed, active

    r.game.step_flies = spy_step_flies
    r.ghosts.propose = spy_propose
    r.run()
    # 시험이 의미 있으려면 확정도, 확정 없는 이동(복도 · 집 출구)도 있어야 한다
    assert stats["committed"] > 0 and stats["moved_not_committed"] > 0 and stats["decide_point1"] > 0, stats
