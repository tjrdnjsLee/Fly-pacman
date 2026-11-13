"""재생 로그의 뇌 필드 (TRD 19.13 로그 추가 필드, DECISIONS 16절)."""

from __future__ import annotations

import numpy as np
import pytest
from helpers import make_cfg
from test_flybrain import CFG, F, brain, obs, random_obs  # noqa: F401  (circuit 고정물은 아래에서 다시 만든다)

from flypac.brain.flybrain import FlyBrain, pn_input
from flypac.core.runner import build_runner
from flypac.recording.recorder import Recorder
from flypac.replay import schema as s
from flypac.replay.reader import read_replay


@pytest.fixture(scope="module")
def circuit():
    return FlyBrain(CFG, 1, [0]).circuit


def _expected_shape(field: s.Field, B: int, k: int, m: int) -> tuple:
    """schema의 모양에서 시간 축 T를 B로 바꾼다 (한 틱, 게임 B개)."""
    dims = {"K": k, "M": m}
    return (B,) + tuple(dims[x] if isinstance(x, str) else x for x in field.shape[1:])


def test_replay_info_matches_schema_and_values(circuit):
    B = 2
    b = brain(circuit, B=B)
    rng = np.random.default_rng(3)
    o = random_obs(rng, B)
    a = b.propose(o, np.ones(B, bool))
    w_at_propose = b.w.copy()
    kv = b.last_kc.copy()
    m_at_propose = b.last_m.copy()
    b.commit((a >= 0), np.ones((B, F), bool))
    info = b.replay_info()

    assert set(info) == set(s.BRAIN_FIELDS) | {"brain_on"}
    for name in (*s.BRAIN_FIELDS, "brain_on"):
        f = s.FIELD_BY_NAME[name]
        got = np.asarray(info[name])
        assert got.shape == _expected_shape(f, B, b.k, circuit.n_mbon), name
        assert got.astype(f.dtype).shape == got.shape
    assert info["kc_active"].dtype == np.int16 and info["brain_on"].all()

    # PN 입력: α_c를 곱하기 전의 9채널
    assert np.allclose(info["pn_input"], pn_input(o.odor, o.z, o.phero))
    # 켜진 KC 번호: last_kc와 같은 집합, 모자라는 자리는 -1, 중복 없음
    for bi in range(B):
        for f in range(F):
            ids = info["kc_active"][bi, f]
            on = ids[ids >= 0]
            assert (ids[len(on):] == -1).all() and np.array_equal(np.sort(on), np.flatnonzero(kv[bi, f]))
    # MBON 출력: 제안 시점의 w로 Σ_i w_ij k_i. 기준선(w0)을 빼기 전의 값이다
    want = np.einsum("bfi,bfij->bfj", kv.astype(float), w_at_propose)
    assert np.allclose(info["mbon_out"], want)
    base = np.einsum("bfi,ij->bfj", kv.astype(float), circuit.w0)
    assert np.allclose((info["mbon_out"] - base) @ b._bundle_mat, m_at_propose)  # 기준선을 빼면 뇌가 쓴 m
    h16 = info["mbon_out"].astype(np.float16)
    assert np.isfinite(h16).all(), "float16 범위를 넘는 MBON 활동"


def test_replay_info_marks_dead_rows(circuit):
    b = brain(circuit, B=2)
    o = random_obs(np.random.default_rng(1), 2)
    b.propose(o, np.array([True, False]))
    info = b.replay_info()
    assert info["brain_on"][0].all() and not info["brain_on"][1].any()
    assert (info["kc_active"][1] == -1).all() and not info["pn_input"][1].any() and not info["mbon_out"][1].any()


def _run_ticks(b: FlyBrain, ticks: int, reward: float, active_p: float, seed: int):
    rng = np.random.default_rng(seed)
    B = b.B
    seen = {"committed": [], "active": [], "dan": []}
    for _ in range(ticks):
        b.give_reward(np.full((B, F), reward))  # 이번 틱의 보상이 먼저 쌓인 상태에서 확정한다
        o = random_obs(rng, B, p_decide=0.6)
        a = b.propose(o, np.ones(B, bool))
        committed = (a >= 0) & (rng.random((B, F)) < 0.7)
        active = rng.random((B, F)) < active_p
        b.commit(committed, active)
        info = b.replay_info()
        seen["committed"].append(committed)
        seen["active"].append(active)
        seen["dan"].append(info["dan"])
    return {k: np.array(v) for k, v in seen.items()}


def test_dan_nonzero_exactly_on_committed_learning_ticks(circuit):
    """DAN이 0이 아닌 틱 = 학습이 일어난 확정 틱 (보상이 매 틱 쌓여 R_n - R̄ ≠ 0일 때 정확히 같다)."""
    b = brain(circuit, B=3)
    seen = _run_ticks(b, ticks=40, reward=1.0, active_p=0.8, seed=5)
    assert seen["committed"].any() and (seen["committed"] & ~seen["active"]).any()
    assert np.array_equal(seen["dan"] != 0, seen["committed"] & seen["active"])


def test_dan_value_is_reward_minus_baseline_before_update(circuit):
    b = brain(circuit, B=1)
    b.rbar[:] = 0.25
    b.give_reward(np.full((1, F), 1.0))
    o = random_obs(np.random.default_rng(0), 1, p_decide=1.0)
    a = b.propose(o, np.ones(1, bool))
    b.commit(a >= 0, np.ones((1, F), bool))
    dan = b.replay_info()["dan"][0]
    assert (a[0] >= 0).any()
    assert np.allclose(dan[a[0] >= 0], 1.0 - 0.25)  # 학습 직전의 R_n - R̄ (R̄는 이 확정에서 갱신된다)
    assert np.allclose(b.rbar[0][a[0] >= 0], 0.99 * 0.25 + 0.01 * 1.0)
    assert not dan[a[0] < 0].any()
    # 다음 틱에는 확정이 없으므로 0으로 돌아간다
    b.propose(random_obs(np.random.default_rng(1), 1, p_decide=0.0), np.ones(1, bool))
    b.commit(np.zeros((1, F), bool), np.ones((1, F), bool))
    assert not b.replay_info()["dan"].any()


def test_dan_never_nonzero_off_commit_even_with_noisy_rewards(circuit):
    b = brain(circuit, B=2)
    seen = _run_ticks(b, ticks=30, reward=0.0, active_p=1.0, seed=2)
    assert not (seen["dan"] != 0).any()  # 보상이 없으면 R_n - R̄ = 0
    b2 = brain(circuit, B=2)
    seen = _run_ticks(b2, ticks=30, reward=-0.5, active_p=1.0, seed=2)
    assert ((seen["dan"] != 0) & ~seen["committed"]).sum() == 0  # 확정이 아닌 틱에는 항상 0


# ---------------------------------------------------------------- 실행기 · 기록기를 거친 로그

def _record(tmp_path, ghosts, replay_brain, ticks_games=1):
    over = {"replay": {"keep_games": [1], "brain": replay_brain}, "seeds": [0], "n_games": ticks_games}
    if ghosts == "flybrain":
        over["ghosts"] = {"kind": "flybrain", "config": "configs/brain_v1.yaml"}
    cfg = make_cfg(**over)
    rec = Recorder(tmp_path / "run", cfg)
    build_runner(cfg, rec).run()
    assert rec.replay_files
    return read_replay(rec.replay_files[0])


def test_recorded_brain_log_has_format_fields(tmp_path):
    meta, a = _record(tmp_path, "flybrain", True)
    assert meta["format_version"] == s.FORMAT_VERSION == 5
    T = len(a["pac_cell"])
    for name in (*s.BRAIN_FIELDS, "brain_on"):
        assert a[name].dtype == np.dtype(s.FIELD_BY_NAME[name].dtype), name
        assert a[name].shape[:2] == (T, 4) or a[name].shape[:1] == (T,)
    assert a["kc_active"].shape[2] == FlyBrain(CFG, 1, [0]).k and a["pn_input"].shape == (T, 4, 9)
    assert a["brain_on"].any() and (a["kc_active"] >= -1).all()
    # DAN 표시는 확정 틱에만 나온다
    assert not ((a["dan"] != 0) & ~a["committed"]).any()
    # 뇌가 계산하지 않은 자리(brain_on 거짓)의 값은 비어 있다
    off = ~a["brain_on"]
    assert (a["kc_active"][off] == -1).all() and not a["mbon_out"][off].any() and not a["dan"][off].any()


@pytest.mark.parametrize("ghosts,replay_brain", [("random", True), ("flybrain", False)])
def test_log_without_brain_has_no_brain_fields(tmp_path, ghosts, replay_brain):
    """뇌가 아닌 조종기(무작위 유령, B1도 같은 경로) 또는 replay.brain이 꺼진 로그: 뇌 필드 묶음이 없다."""
    meta, a = _record(tmp_path, ghosts, replay_brain)
    assert meta["format_version"] == 5
    assert not any(n in a for n in s.BRAIN_FIELDS)
    assert "brain_on" in a and not a["brain_on"].any()
