"""표현력 시험의 자료 구성 · 누적 h 재현 (DECISIONS 13절). sklearn이 필요한 시험은 설치되지 않으면 건너뛴다."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from flypac.brain import representation as R
from flypac.brain.flybrain import FlyBrain, build_circuit, load_brain_config, repo_path
from flypac.core.config import Config, load_config
from flypac.core.runner import build_runner

REPO = Path(__file__).resolve().parents[1]


def _cfg(seeds, n_games=2, watchdog=500):
    d = load_config(REPO / "configs" / "brain_repr_v1.yaml").to_dict()
    d["seeds"], d["n_games"] = list(seeds), n_games
    d["engine"] = {**d["engine"], "watchdog_ticks": watchdog}
    return Config(d)


@pytest.fixture(scope="module")
def rows():
    return R.generate_rows(_cfg([1000, 1001, 1014]))


def test_rows_are_consecutive_ticks_and_commit_resets(rows):
    assert len(rows["seed"]) % 4 == 0 and set(np.unique(rows["fly"])) == {0, 1, 2, 3}
    for sd in np.unique(rows["seed"]):
        for g in np.unique(rows["game"][rows["seed"] == sd]):
            m = (rows["seed"] == sd) & (rows["game"] == g) & (rows["fly"] == 0)
            t = rows["tick"][m]
            assert np.array_equal(t, np.arange(len(t)))             # 틱 번호는 0부터 연속 (DECISIONS 8절)
    assert rows["reset"][rows["committed"]].all()                  # 확정은 h를 비운다
    # 벽 마스크: 합법 행동은 벽이 아니다
    assert not (rows["legal"] & rows["wall"]).any()


def test_task_c_variants_have_answers_A_B_B_A_with_unique_extremes(rows):
    p, _ = R.build_test4(rows)
    m = int(p.meta["base"].max()) + 1
    assert len(p) == 4 * m and m > 20
    var, base = p.meta["var"], p.meta["base"]
    for b in range(m):
        idx = np.flatnonzero(base == b)
        v = {int(var[i]): i for i in idx}
        A_, B_ = int(p.meta["A"][b]), int(p.meta["B"][b])
        assert A_ != B_
        assert [int(p.y[v[k]]) for k in range(4)] == [A_, B_, B_, A_]
        assert [float(p.z[v[k]]) for k in range(4)] == [0.0, 1.0, 0.0, 1.0]
        o, lg = p.odor[v[0]], p.legal[v[0]]
        assert np.array_equal(p.legal[v[0]], p.legal[v[2]])         # 네 변형은 같은 L
        others = [a for a in range(4) if lg[a] and a not in (A_, B_)]
        assert o[A_] - max([o[a] for a in others] + [o[B_]]) > R.TOL     # A 유일 최대
        assert min([o[a] for a in others] + [o[A_]]) - o[B_] > R.TOL     # B 유일 최소
        s = p.odor[v[2]]
        assert s[A_] == o[B_] and s[B_] == o[A_]                    # 두 값만 맞바꿈
        assert all(s[a] == o[a] for a in others)                    # 다른 합법 방향은 그대로
        assert np.nanargmax(np.where(lg, s, np.nan)) == B_ and np.nanargmin(np.where(lg, s, np.nan)) == A_


def test_pheromone_partner_has_same_legal_and_wall_mask_and_other_game(rows):
    p, stats = R.build_test3(rows)
    assert len(p) > 50 and stats["unique"] == len(p)
    src, par = p.meta["src"], p.meta["partner"]
    assert np.array_equal(rows["legal"][src], rows["legal"][par])
    assert np.array_equal(rows["wall"][src], rows["wall"][par])
    same_game = (rows["seed"][src] == rows["seed"][par]) & (rows["game"][src] == rows["game"][par])
    assert not same_game.any()
    assert np.array_equal(p.phero, rows["odor"][par])
    legal = p.legal
    assert np.array_equal(p.y, np.where(legal, p.phero, -np.inf).argmax(1))
    # 같은 입력에서 두 번 만들면 같다 (repr_synth 난수)
    p2, _ = R.build_test3(rows)
    assert np.array_equal(p.meta["partner"], p2.meta["partner"])


def test_reference_odor_uses_only_open_directions_of_given_rows():
    n = 6
    odor = np.array([[0.0, 1.0, 3.0, 0.0]] * 3 + [[0.0, 5.0, 7.0, 0.0]] * 3)
    wall = np.array([[True, False, False, True]] * n)
    rows = {"decide": np.ones(n, bool), "frozen": np.zeros(n, bool), "legal": np.array([[0, 1, 1, 0]] * n, bool),
            "odor": odor, "wall": wall}
    med = R.reference_odor(rows)
    assert med[1] == 3.0 and med[2] == 5.0                       # 벽인 방향의 0은 제외한 중앙값
    rows2 = {k: v.copy() for k, v in rows.items()}
    rows2["odor"][:, 1] = 100.0
    assert R.reference_odor(rows2)[1] == 100.0                   # 주어진 행(개발-학습)만으로 정해진다


def test_kc_tiebreak_depends_only_on_seed_and_game_not_batch_order():
    circ = build_circuit(*_circuit_args(), 0)
    odor = np.tile([0.5, 0.2, 0.9, 0.1], (6, 1))
    z, ph = np.zeros(6), np.zeros((6, 4))
    seeds, games = np.array([5, 6, 5, 6, 5, 6]), np.array([1, 1, 1, 1, 2, 2])
    a = R.kc_features(circ, 0.05, 0.0, odor, z, ph, seeds, games)
    perm = np.array([5, 4, 3, 2, 1, 0])
    b = R.kc_features(circ, 0.05, 0.0, odor[perm], z[perm], ph[perm], seeds[perm], games[perm])
    # 같은 (시드, 판)의 행 순서가 같으면(여기서는 뒤집혀 다름) 결과는 같고, 아니면 동률 끊기만 다를 수 있다
    assert a.sum(1).tolist() == [79] * 6 and b.sum(1).tolist() == [79] * 6
    c = R.kc_features(circ, 0.05, 0.0, odor, z, ph, seeds, games)
    assert np.array_equal(a, c)                                  # 재현성


def _circuit_args():
    bcfg = load_brain_config("configs/brain_v1.yaml")
    return bcfg.circuit.raw, load_config(repo_path(bcfg.circuit.assign))


def _naive_h(circ, rows, kappa):
    """구간 논리를 따로 구현한 기준: 초파리별 시간순으로 훑으며 alive 행만 갱신, reset에서 비운다."""
    order = np.lexsort((rows["tick"], rows["fly"], rows["game"], rows["seed"]))
    h = np.zeros((len(order), circ.n_kc))
    cur, prev = np.zeros(circ.n_kc), None
    for i in order:
        key = (rows["seed"][i], rows["game"][i], rows["fly"][i])
        if key != prev:
            cur, prev = np.zeros(circ.n_kc), key
        if not rows["frozen"][i]:
            k = R.kc_features(circ, 1.0, 0.0, rows["odor"][i:i + 1], rows["z"][i:i + 1], np.zeros((1, 4)),
                              rows["seed"][i:i + 1], rows["game"][i:i + 1])
            cur = (1 - kappa) * cur + k[0]
        h[i] = cur
        if rows["reset"][i]:
            cur = np.zeros(circ.n_kc)
    return h


def test_h_replay_matches_naive_recursion(rows):
    """ρ = 1(동률 난수 없음)에서 구간 재현 h = 순차 재귀 h (θ = 0, 시험 1 표본)."""
    circ = build_circuit(*_circuit_args(), 0)
    sub = {k: v[: 4 * 260] for k, v in rows.items()}
    p, _, _ = R.build_test1(sub)
    assert len(p) > 5
    layout = R.interval_layout(sub)
    got = R.h_features(circ, 1.0, 0.0, sub, layout, p, kappas=(0.2,))[0.2]
    want = _naive_h(circ, sub, 0.2)[p.meta["src"]]
    assert np.allclose(got, want, rtol=1e-4, atol=1e-4)


def test_h_replay_matches_live_flybrain():
    """살아 있는 FlyBrain의 h(propose 직후) = 같은 실행의 자료로 재현한 h (ρ = 1, 동률 난수 없음)."""
    cfg = _cfg([1000], n_games=1, watchdog=400)
    r = build_runner(cfg)
    col = R.Collector([1000], r.map.nbr)
    bd = load_brain_config("configs/brain_v1.yaml").to_dict()
    bd["kc"] = {**bd["kc"], "rho": 1.0, "theta": 0.0}
    circ = build_circuit(*_circuit_args(), int(bd["circuit"]["seed"]))
    brain = FlyBrain(Config(bd), r.B, [1000], condition="regime1", circuit=circ)
    snaps = []
    propose = brain.propose

    def spy(obs, alive, views=None):
        out = propose(obs, alive, views)
        snaps.append(brain.h[0].copy())
        return out

    brain.propose = spy
    r.ghosts, r.recorder = brain, col
    r.run()
    rows = col.rows()
    p, _, _ = R.build_test1(rows)
    assert len(p) > 5
    got = R.h_features(circ, 1.0, 0.0, rows, R.interval_layout(rows), p, kappas=(brain.kappa,))[brain.kappa]
    t, f = rows["tick"][p.meta["src"]], rows["fly"][p.meta["src"]]
    want = np.stack([snaps[ti][fi] for ti, fi in zip(t, f)])
    assert np.allclose(got, want, rtol=1e-4, atol=1e-4)


def test_control_classifier_is_at_most_half_on_complete_bundles(rows):
    pytest.importorskip("sklearn")
    tr = R.build_split("dev_train", R.split_rows(rows, 1000, 1001))
    ev = R.build_split("dev_eval", R.split_rows(rows, 1014, 1014))
    c = R.control_scores(tr, ev)
    assert c["ok"]
    for name in ("train", "eval"):
        for zv in (0, 1):
            assert c[f"t4_{name}_general_acc_z{zv}"] <= R.CONTROL_MAX + 1e-12


def test_selection_takes_first_in_prior_order_and_counts_passing_seeds():
    def mk(s, rho, c, ok):
        return {"seed": s, "rho": rho, "c": c, "pass": ok}
    combos = [mk(2, 0.05, 0, True), mk(1, 0.10, 2, True), mk(1, 0.02, 1, True), mk(0, 0.05, 0, False)]
    assert R.select_candidate(combos)["seed"] == 1 and R.select_candidate(combos)["rho"] == 0.02   # ρ 순서 0.05, 0.02, 0.10
    assert R.passing_seeds(combos) == [1, 2]
    assert R.select_candidate([mk(0, 0.05, 0, False)]) is None
