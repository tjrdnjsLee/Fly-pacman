"""회로 v2 · 시각 · 냄새 퍼텐셜 보상 (DECISIONS 42). 토큰 · 실제 원본 없이 합성 자료로 돈다."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

from flypac.brain import assign as A
from flypac.brain.flybrain import Circuit, FlyBrain, build_circuit, kc_drive, pn_input
from flypac.core import directions as d
from flypac.core.config import Config, load_config
from flypac.core.shaping import OdorShaping
from flypac.core.types import FlyObs
from flypac.maps.original import load_original
from flypac.senses.sense import see

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


# ---------------------------------------------------------------- 추출 (select_circuit_v2)

def _synthetic_d():
    """후각 PN 2개(서로 다른 사구체), 시각 후보 4종류, KC 6개, MBON 2개. pandas가 없으면(기본 설치) 건너뛴다."""
    pd = pytest.importorskip("pandas")
    import extract_circuit as ec
    kc = pd.DataFrame({"bodyId": [101, 102, 103, 104, 105, 106],
                       "type": ["KCg-m", "KCg-m", "KCab-p", "KCg-d", "KCab-p", "KCg-m"]})
    kc["kc_class"] = kc["type"].map(ec.kc_class)
    pn = pd.DataFrame({"bodyId": [1, 2, 3], "type": ["DA1_lPN", "VA1v_adPN", "VP2_adPN"]})
    pn["glom"] = pn["type"].map(ec.glom_of)
    pn["lineage"] = pn["type"].map(ec.lineage_of)
    pn["category"] = pn["type"].map(ec.pn_category)
    mbon = pd.DataFrame({"bodyId": [201, 202], "type": ["MBON01", "MBON07"], "instance": ["MBON01_R", "MBON07_R"]})
    rows = []
    # 후각 KC 101, 102: 본 칼릭스에서 PN 1, 2로부터 큰 입력
    for k in (101, 102, 106):
        rows += [(1, k, "CA(R)", 40), (2, k, "CA(R)", 40)]
    # 시각 후보: SLP360(300, dACA), aMe12(301, vACA), OA-VPM3(302, 신경조절), VP2_adPN(3, PN) — lACA는 세지 않음
    for k in (103, 105):
        rows += [(300, k, "dACA(R)", 30), (302, k, "dACA(R)", 30), (3, k, "dACA(R)", 30), (2, k, "CA(R)", 4)]
    rows += [(301, 104, "vACA(R)", 25), (301, 103, "vACA(R)", 2)]          # 2는 s_min 미만
    rows += [(300, 106, "lACA(R)", 50)]                                    # lACA: 세지 않는다
    rows += [(400, 104, "vACA(R)", 30)]                                    # 주석 없음
    conn = pd.DataFrame(rows, columns=["bodyId_pre", "bodyId_post", "roi", "weight"])
    typ = {1: "DA1_lPN", 2: "VA1v_adPN", 3: "VP2_adPN", 300: "SLP360", 301: "aMe12", 302: "OA-VPM3", 400: None}
    conn["pre_type"] = conn["bodyId_pre"].map(typ)
    pn_cat = pn.set_index("bodyId")["category"]

    def src(b, t):
        if b in pn_cat.index:
            return pn_cat[b]
        if t is None or (isinstance(t, float) and np.isnan(t)):
            return "unannotated"
        return "other_annotated"
    conn["src"] = [src(b, t) for b, t in zip(conn["bodyId_pre"], conn["pre_type"])]
    k2m = pd.DataFrame([(k, m, "g2(R)", 60) for k in (101, 102, 103, 104, 105, 106) for m in (201, 202)],
                       columns=["bodyId_pre", "bodyId_post", "roi", "weight"])
    kin_neurons = pd.DataFrame({"bodyId": list(typ), "type": list(typ.values())})
    return dict(pn=pn, kc=kc, mbon=mbon, conn=conn, ca=conn[conn["roi"] == "CA(R)"], k2m={"conn": k2m},
                kin={"neurons": kin_neurons}, kc_cls=kc.set_index("bodyId")["kc_class"])


def _cfg(theta_v=20):
    c = load_config(ROOT / "configs" / "circuit_extract_v2.yaml").to_dict()
    c["thresholds"].update(theta_pn=50, theta_mbon=100)
    c["vision"]["theta_v"] = theta_v
    return Config(c)


def test_select_v2_keeps_olfactory_and_adds_visual():
    d0 = _synthetic_d()
    import extract_circuit as ec
    v1 = ec.select_circuit(d0, _cfg())
    res = ec.select_circuit_v2(d0, _cfg())
    a = res["arrays"]
    assert list(a["pn_body"]) == list(v1["arrays"]["pn_body"]) == [1, 2]       # 후각 PN 그대로
    assert list(a["vis_body"]) == [300, 301]                                   # OA-VPM3 · VP2_adPN · 주석 없음 제외
    assert set(v1["arrays"]["kc_body"]) <= set(a["kc_body"])                   # 후각 KC 그대로 포함
    mod = dict(zip(a["kc_body"].tolist(), a["kc_modality"].tolist()))
    assert mod[101] == mod[102] == mod[106] == "olf"
    assert mod[103] == mod[105] == mod[104] == "vis"
    assert a["J_vis"].shape == (len(a["kc_body"]), 2) and a["J_pn"].shape == (len(a["kc_body"]), 2)
    i106 = list(a["kc_body"]).index(106)
    assert a["J_vis"][i106].sum() == 0                                         # lACA는 세지 않는다
    i103 = list(a["kc_body"]).index(103)
    assert a["J_vis"][i103, 1] == 0                                            # s_min 미만 연결은 0


def test_select_v2_theta_v_drops_weak_visual_neuron():
    d0 = _synthetic_d()
    import extract_circuit as ec
    a = ec.select_circuit_v2(d0, _cfg(theta_v=40))["arrays"]
    assert list(a["vis_body"]) == [300]                                        # aMe12 합 25 < 40
    assert 104 not in a["kc_body"]


# ---------------------------------------------------------------- 채널 · 뇌

def _raw_v2(tmp_path, n_kc=40, n_pn=12, n_vis=9, n_mbon=8, seed=3):
    rng = np.random.default_rng(seed)
    gl = np.array([f"G{i}" for i in range(n_pn)])
    arr = dict(
        pn_body=np.arange(n_pn), pn_type=gl, pn_glom=gl, glom_names=np.sort(gl),
        kc_body=np.arange(n_kc), kc_type=np.array(["KCg-m"] * n_kc), kc_class=np.array(["gamma", "ab", "apbp", "ab"] * (n_kc // 4)),
        mbon_body=np.arange(n_mbon), mbon_type=np.array([f"M{i}" for i in range(n_mbon)]),
        J_pn=(rng.integers(3, 20, (n_kc, n_pn)) * (rng.random((n_kc, n_pn)) < 0.3)).astype(np.int32),
        W0=(rng.integers(3, 20, (n_kc, n_mbon)) * (rng.random((n_kc, n_mbon)) < 0.5)).astype(np.int32),
        mbon_class_profile=np.full((n_mbon, 3), 1 / 3), mbon_class_profile_candidates=np.full((n_mbon, 3), 1 / 3),
        vis_body=np.arange(1000, 1000 + n_vis), vis_type=np.array(["SLP360"] * n_vis),
        J_vis=(rng.integers(3, 20, (n_kc, n_vis)) * (rng.random((n_kc, n_vis)) < 0.4)).astype(np.int32),
        kc_modality=np.array(["olf"] * n_kc),
    )
    p = tmp_path / "raw_v2.npz"
    np.savez_compressed(p, **arr)
    return p


def _assign_cfg():
    return load_config(ROOT / "configs" / "circuit_assign_v2.yaml")


def test_vis_channels_balanced_and_independent_of_odor_assignment():
    lab = A.assign_vis_channels(9, 0)
    assert set(lab) == set(A.VISION_CHANNELS)
    assert np.bincount(lab - 9).max() - np.bincount(lab - 9).min() <= 1
    assert (A.assign_vis_channels(9, 0) == lab).all()                         # 같은 시드 → 같은 결과
    g = np.array([f"G{i}" for i in range(12)])
    before = A.assign_pn_channels(g, 0, "G", [])
    A.assign_vis_channels(9, 0)
    assert (A.assign_pn_channels(g, 0, "G", []) == before).all()              # 후각 배정은 그대로


def test_build_circuit_v2_has_13_channels_and_first_9_equal_v1(tmp_path):
    p = _raw_v2(tmp_path)
    c2 = build_circuit(p, _assign_cfg(), 0)
    assert c2.J.shape[1] == A.N_CHANNELS_V2 and c2.has_vision and c2.vis_channel.shape == (9,)
    raw = A.load_raw(p)
    pch = A.pn_channels(raw.pn_glom, raw.glom_names, A.assign_pn_channels(raw.glom_names, 0, "G", []))
    assert np.array_equal(c2.J[:, :9], A.channel_matrix(raw.J_pn, pch))
    for ch in A.VISION_CHANNELS:
        assert c2.J[:, ch].sum() == raw.J_vis[:, c2.vis_channel == ch].sum()


def test_kc_drive_pads_missing_vision_with_zero(tmp_path):
    c2 = build_circuit(_raw_v2(tmp_path), _assign_cfg(), 0)
    rng = np.random.default_rng(0)
    odor, z, ph = rng.random((5, 4)), rng.random(5) < 0.5, rng.random((5, 4))
    u9 = kc_drive(pn_input(odor, z, ph), c2)
    u13 = kc_drive(pn_input(odor, z, ph, np.zeros((5, 4))), c2)
    assert np.allclose(u9, u13)
    vis = np.zeros((5, 4)); vis[:, 1] = 1.0
    assert not np.allclose(kc_drive(pn_input(odor, z, ph, vis), c2), u9)


def test_brain_forward_uses_obs_vision(tmp_path):
    c2 = build_circuit(_raw_v2(tmp_path), _assign_cfg(), 0)
    cfg = load_config(ROOT / "configs" / "brain_v1.yaml").to_dict()
    cfg["kc"] = {"rho": 0.25, "theta": 0.0}
    b = FlyBrain(Config(cfg), 1, [0], circuit=c2)
    assert len(b.channel_names) == 13
    F = 4
    base = dict(odor=np.zeros((1, F, 4)), phero=np.zeros((1, F, 4)), z=np.zeros((1, F), bool),
                legal=np.ones((1, F, 4), bool), decide=np.zeros((1, F), bool))
    b.propose(FlyObs(**base), np.ones(1, bool)); b.commit(np.zeros((1, F), bool), np.zeros((1, F), bool))
    assert b.last_kc.sum() == 0                                                # 무입력
    vis = np.zeros((1, F, 4)); vis[0, :, 1] = 1.0
    b.propose(FlyObs(**base, vision=vis), np.ones(1, bool))
    assert b.last_kc.sum() > 0                                                 # 시각만으로 KC가 켜진다
    assert b.replay_info()["pn_input"].shape == (1, F, 13)


# ---------------------------------------------------------------- 시각 (see)

def _corridor():
    """원작 맵에서 동쪽으로 R칸 이상 곧게 이어진 칸 하나."""
    m = load_original()
    nbr = m.nbr
    for c in range(nbr.shape[0]):
        x, ok = c, True
        for _ in range(4):
            x = nbr[x, d.EAST]
            if x < 0:
                ok = False
                break
        if ok:
            return nbr, c
    raise AssertionError


def test_see_straight_line_within_R_and_walls_block():
    nbr, c = _corridor()
    c1, c3, c4 = nbr[c, d.EAST], nbr[nbr[nbr[c, d.EAST], d.EAST], d.EAST], None
    c4 = nbr[c3, d.EAST]
    cells = np.array([[c, c, c, c]]); head = np.full((1, 4), d.EAST)
    out = see(cells, head, np.array([c3]), nbr, R=3, s=2.0)
    assert np.allclose(out[0, :, 1], 2.0) and np.allclose(out[0, :, [0, 2, 3]], 0.0)    # 앞 3칸: 보임
    assert np.allclose(see(cells, head, np.array([c4]), nbr, R=3, s=2.0), 0.0)        # 4칸: 안 보임
    west = np.full((1, 4), d.WEST)
    assert np.allclose(see(cells, west, np.array([c1]), nbr, R=3, s=1.0)[0, :, 3], 1.0)   # 뒤도 본다 (겹눈)
    assert np.allclose(see(cells, head, np.array([c]), nbr, R=3, s=1.0), 0.0)         # 같은 칸은 0
    assert np.allclose(see(cells, head, np.array([c3]), nbr, R=0, s=1.0), 0.0)        # R = 0: 끔


# ---------------------------------------------------------------- 냄새 퍼텐셜 보상

class _V:
    def __init__(self, cells, z):
        self.fly_cell, self.fly_z = np.asarray(cells), np.asarray(z)


def test_shaping_telescopes_and_resets():
    sh = OdorShaping(1, 4, lam=2.0, c0=0.01)
    c = np.zeros((1, 10))
    run = np.ones(1, bool)
    seq = [0.0, 0.1, 0.5, 0.2]
    tot = np.zeros(4)
    for i, val in enumerate(seq):
        c[0, 3] = val
        r = sh.step(c, [_V([3, 3, 3, 3], [0, 0, 0, 0])], run)
        if i == 0:
            assert np.allclose(r, 0.0)                                         # 첫 틱: 기준점만 잡는다
        tot += r[0]
    phi = lambda v: np.log1p(v / 0.01)  # noqa: E731
    assert np.allclose(tot, 2.0 * (phi(seq[-1]) - phi(seq[0])))               # 망원
    c[0, 3] = 0.9
    sh.eaten(np.array([[True, False, False, False]]))
    r = sh.step(c, [_V([3, 3, 3, 3], [0, 0, 0, 0])], run)
    assert r[0, 0] == 0.0 and r[0, 1] > 0                                     # 먹힌 초파리는 그 틱 0
    r = sh.step(c, [_V([3, 3, 3, 3], [1, 0, 0, 0])], run)
    assert r[0, 0] == 0.0                                                      # z가 바뀐 틱은 0
    c[0, 3] = 0.5
    r = sh.step(c, [_V([3, 3, 3, 3], [1, 0, 0, 0])], run)
    assert r[0, 0] > 0 and r[0, 1] < 0                                         # 겁먹으면 멀어질 때(옅어질 때) +
    sh.reset(np.array([True]))
    assert np.allclose(sh.step(c, [_V([3, 3, 3, 3], [1, 0, 0, 0])], run), 0.0)
    assert np.allclose(sh.step(c, [_V([3, 3, 3, 3], [1, 0, 0, 0])], np.array([False])), 0.0)


# ---------------------------------------------------------------- 실행기 (v1 뇌 + 시각 + 냄새 보상, 끝까지 돈다)

def test_runner_with_vision_and_shaping_smoke(tmp_path):
    from flypac.core.runner import build_runner
    from flypac.recording.recorder import Recorder
    base = load_config(ROOT / "configs" / "final_eval_v1.yaml").to_dict()
    del base["final_eval"]
    base.update(seeds=[0], n_games=1, condition="regime1", evaluate=False, experiment="vision_smoke",
                ghosts={"kind": "flybrain", "config": "configs/brain_v1.yaml"})
    base["pacman"] = {"kind": "rule_based"}; base["rule_based"] = {"danger_dist": 6, "chase_dist": 8}
    base["run"] = {"max_loop_ticks": 20001}
    base["replay"] = {"keep_games": [], "brain": False}
    base["vision"] = {"R": 3, "s": 1.0}
    base["reward"] = {"shaping": {"lam": 1.0}}
    cfg = Config(base)
    r = build_runner(cfg, Recorder(tmp_path / "run", cfg))
    assert r.vision_R == 3 and r.shaping is not None and r.shaping.lam == 1.0
    seen = []
    orig = r.ghosts.propose

    def spy(obs, alive, views=None):
        seen.append(obs.vision is not None and obs.vision.shape == obs.odor.shape)
        return orig(obs, alive, views)
    r.ghosts.propose = spy
    r.run()
    assert seen and all(seen)
    assert r.ghosts.dw.any()


def test_defaults_are_v1(tmp_path):
    from flypac.core.runner import shaping_lam, vision_R
    cfg = load_config(ROOT / "configs" / "main_v1.yaml")
    assert vision_R(cfg) == 0 and shaping_lam(cfg) == 0.0
