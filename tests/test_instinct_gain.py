"""본능 세기의 학습 시험 (DECISIONS 38.3 · 38.4). torch · 체크포인트는 쓰지 않는다. 시드는 0 ~ 99, 임시 폴더.

- 기본값 불변: `instinct_gain` 블록이 없는 뇌는 이 기능을 넣기 전(main 9bc941b)과 바이트 단위로 같은 파일을 낸다.
- 켜진 뇌: 점수의 수치 미분, 학습 방향, b0 · 평가 모드, 시냅스 동결, 범위, 저장 · 불러오기, "k = 1 · η_k = 0 = 기존 B0".
- 설정: 새 설정 셋은 허용한 키에서만 다르고, 새 시드 범위는 다른 설정에 없고, 기존 세 설정의 해시가 그대로다.
- `scripts/k_curve.py`의 계산.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from flypac.brain.flybrain import FlyBrain, load_brain_config
from flypac.core.config import Config, load_config
from flypac.core.runner import build_runner
from flypac.recording.recorder import Recorder
from flypac.recording.weights import read_weight_file, weight_filename

ROOT = Path(__file__).resolve().parents[1]
BRAIN, BRAIN_K = "configs/brain_v1.yaml", "configs/brain_k_v1.yaml"
CFG, CFG_K = load_brain_config(BRAIN), load_brain_config(BRAIN_K)
F = 4
CSVS = ("games.csv", "lives.csv", "flies.csv")

# 이 PR 전의 main(9bc941b) 코드로 `_run_cfg()`(뇌 brain_v1, 체제 1, 시드 0 · 1, 3판, 판마다 가중치 저장)를 돌려 잰 sha256.
# 실제 실험과 같이 별도 프로세스 · 수치 라이브러리 스레드 1(`THREAD_ENV`)에서 쟀다. CSV는 스레드 수 · 장치가 달라도 같았고,
# 가중치 파일(dw의 마지막 비트)은 스레드 수 · CPU · OS에 따라 달랐다 (DECISIONS 22 · 27.5와 같은 성질).
BASELINE_SHA256 = {
    "games.csv": "5b39c07bdead38f55df320d2899d60b13cd6dd0912bdd2996f5552808e4123c3",
    "lives.csv": "a735a6524a73787c1107402dd7d2c53a8b92c773f29aeb2c9b7e3b10792f5b8d",
    "flies.csv": "13d57c53efec4c742af9adfa17617fbf3c20dcb287911738b3a6077c350a9912",
    "weights/s0_g00001.npz": "8f45355f52381532854246dd9369834b07ee392af69ea3acc3634a3a326b5a91",
    "weights/s0_g00002.npz": "0443c1835a5c0228c2c80e37b863b8dbff8f7459cbe3d60ff53fc9b12d8fadbf",
    "weights/s0_g00003.npz": "3cb52c19b88ee6a39dd12fd97a5d8de5e3833fa630a9d4d9eac810c15d3b8c3c",
    "weights/s1_g00001.npz": "cb7678e959baf205d3fcd8714ca652e3d19e8fd9bd660f5d67e1ae11adc0a8c1",
    "weights/s1_g00002.npz": "e2ac7bec5bcf183806a5a930234bd45a5d34c14b378ea8ff94f460e8ac83baf6",
    "weights/s1_g00003.npz": "886d3aeec91911501030d163fa4dc7a7f4238c2e8a9ae1b61ebcf7e2f91ac679",
}
# 기존 설정의 Config.hash() (이 PR 전의 main에서 잰 값)
OLD_CONFIG_HASH = {"brain_v1": "1e3c82209c1a4337", "main_v1": "d488a2fdafc009da", "final_eval_v1": "6a0948dacb984e86"}


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _run_cfg(brain=BRAIN, condition="regime1", **over) -> Config:
    """진짜 엔진(integ_v1의 블록) + 규칙 팩맨, 시드 0 · 1, 3판, 판마다 가중치 저장."""
    d = load_config(ROOT / "configs" / "integ_v1.yaml").to_dict()
    d.update(experiment="kgain_default", condition=condition, seeds=[0, 1], n_games=3,
             pacman={"kind": "rule_based"}, rule_based={"danger_dist": 6, "chase_dist": 8},
             ghosts={"kind": "flybrain", "config": brain}, replay={"keep_games": [], "brain": False},
             weights={"every_games": 1})
    d.update(over)
    return Config(d)


def _run(tmp_path, name, cfg):
    runner = build_runner(cfg, Recorder(tmp_path / name, cfg))
    runner.run()
    return tmp_path / name, runner


@pytest.fixture(scope="module")
def circuit():
    return FlyBrain(CFG, 1, [0]).circuit


def kbrain(circuit, B=1, condition="regime1", evaluate=False, **gain) -> FlyBrain:
    d = CFG_K.to_dict()
    d["instinct_gain"] = {**d["instinct_gain"], **gain}
    return FlyBrain(Config(d), B, list(range(B)), condition=condition, circuit=circuit, evaluate=evaluate)


def obs(B, odor, legal=None, decide=True):
    from flypac.core.types import FlyObs

    return FlyObs(odor=np.asarray(odor, float), phero=np.zeros((B, F, 4)), z=np.zeros((B, F), bool),
                  legal=np.ones((B, F, 4), bool) if legal is None else np.asarray(legal, bool),
                  decide=np.full((B, F), bool(decide)))


def tick(b: FlyBrain, o, active=True, R=None):
    a = b.propose(o, np.ones(b.B, bool))
    b.commit(a >= 0, np.full((b.B, F), bool(active)))
    if R is not None:
        b.give_reward(R)
    return a


# ---------------------------------------------------------------- 기본값 불변

OLD_WEIGHT_KEYS = ["dw_masked", "rbar", "seed", "game_no", "condition", "format", "circuit_sha256",
                   "brain_config_hash", "run_config_hash"]   # 가중치 파일의 항목과 순서 (이 PR 전)
THREAD_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}   # run_parallel.child_env와 같다
# 기준 장치(BASELINE_SHA256을 잰 곳)의 부동소수점 결과 지문: 같은 환경(THREAD_ENV)에서 잰 `_numeric_probe()`의 값
BASELINE_PROBE = "badc818dc5b5c829da1703cd4e5584412235c0226365f5aa03bb707543c90f49"


def _numeric_probe(circuit) -> str:
    """이 장치의 부동소수점 결과 지문: 뇌가 쓰는 연산(J p 행렬곱, exp, einsum, 외적)을 정해진 입력에 한 결과의 sha256."""
    from flypac.brain.flybrain import kc_drive

    rng = np.random.default_rng(0)
    p = rng.random((64, 9)) * 3
    u = kc_drive(p, circuit)
    w = rng.random((64, circuit.n_kc, 8))
    h = hashlib.sha256()
    for a in (u, np.exp(np.linspace(-9.0, 9.0, 4001)), np.exp(4.0 * rng.random(4001) - 2.0),
              np.einsum("bi,bij->bj", u, w), u @ rng.random((circuit.n_kc, 4)), np.outer(u[0], rng.random(47)) * 1.129):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


@pytest.fixture(scope="module")
def default_run(tmp_path_factory):
    """`_run_cfg()`를 별도 프로세스 · 스레드 1에서 돌린 결과 폴더와 그 프로세스의 지문 (이 파일의 `__main__`)."""
    out = tmp_path_factory.mktemp("default") / "r"
    done = subprocess.run([sys.executable, str(Path(__file__).resolve()), str(out)], env={**os.environ, **THREAD_ENV},
                          capture_output=True, text=True, encoding="utf-8")
    assert done.returncode == 0, done.stderr
    return out, (out / "probe.txt").read_text(encoding="utf-8").strip()


def test_default_brain_has_no_gain_state(circuit):
    assert "instinct_gain" not in CFG
    g = FlyBrain(CFG, 1, [0], circuit=circuit)
    assert not g.gain_on and not any(hasattr(g, name) for name in ("k_gain", "e_k", "xo", "eta_k"))
    assert "k_gain" not in g.slot_state(0) and "k_gain" not in g.state_dict()


def test_default_brain_csvs_and_weight_layout_are_unchanged(default_run, circuit):
    """블록이 없는 뇌: CSV는 main과 바이트 단위로 같고, 가중치 파일은 이름 · 항목 · 순서 · 꼬리표가 그대로다 (장치 무관)."""
    d, _ = default_run
    assert {name: _sha(d / name) for name in CSVS} == {name: BASELINE_SHA256[name] for name in CSVS}
    files = sorted((d / "weights").glob("*.npz"))
    assert [f"weights/{p.name}" for p in files] == sorted(k for k in BASELINE_SHA256 if k.startswith("weights/"))
    for p in files:
        with np.load(p, allow_pickle=False) as z:
            assert z.files == OLD_WEIGHT_KEYS, p.name
            assert z["dw_masked"].dtype == np.float64 and z["dw_masked"].shape == (F, int(circuit.M.sum()))
            assert z["dw_masked"].any()                                   # 학습이 실제로 일어난 실행이다
            assert (int(z["format"]), str(z["condition"]), str(z["brain_config_hash"])) == (1, "regime1", CFG.hash())


def test_default_brain_weight_files_are_byte_identical_to_main(default_run):
    """가중치 파일의 sha256이 main과 같다. 기준 장치와 부동소수점 결과가 같은 장치에서만 성립하므로 다른 장치는 건너뛴다."""
    d, probe = default_run
    if sys.platform != "linux" or probe != BASELINE_PROBE:
        pytest.skip("이 장치는 기준 sha256을 잰 장치와 부동소수점 결과(또는 OS)가 다르다: 가중치 파일의 바이트 비교는 기준 장치에서만 한다")
    got = {f"weights/{p.name}": _sha(p) for p in sorted((d / "weights").glob("*.npz"))}
    assert got == {k: v for k, v in BASELINE_SHA256.items() if k.startswith("weights/")}


def test_block_with_learn_false_behaves_like_no_block(tmp_path, circuit):
    b = kbrain(circuit, learn=False)
    assert not b.gain_on and b.eta == float(CFG.learn.eta) and not hasattr(b, "k_gain")
    rng = np.random.default_rng(1)
    ref = FlyBrain(CFG, 1, [0], circuit=circuit)
    for _ in range(40):
        o = obs(1, rng.random((1, F, 4)) * 3)
        R = rng.normal(0, 1, (1, F))
        assert np.array_equal(tick(b, o, R=R), tick(ref, o, R=R))
    assert ref.dw.any() and np.array_equal(b.dw, ref.dw) and np.array_equal(b.x, ref.x)


# ---------------------------------------------------------------- 켜진 뇌: 식

def test_score_matches_finite_difference_of_log_pi_in_k(circuit):
    """s = β g (ō_a − Σ_b π_b ō_b) = ∂ ln π_a / ∂k (합법 방향 안, ō = xo)."""
    rng = np.random.default_rng(2)
    legal = rng.random((1, F, 4)) < 0.6
    legal[0, :, 1] = True
    legal[0, 0] = True                         # 초파리 0은 네 방향 모두
    odors = [rng.random((1, F, 4)) * 3 for _ in range(3)]
    k0 = rng.uniform(0.5, 4.0, (1, F))

    def run(k):
        b = kbrain(circuit)
        b.k_gain = k.copy()
        for od in odors[:-1]:                  # 누적만 (결정 없음)
            b.propose(obs(1, od, legal, decide=False), np.ones(1, bool))
            b.commit(np.zeros((1, F), bool), np.ones((1, F), bool))
        a = b.propose(obs(1, odors[-1], legal), np.ones(1, bool))
        xo = b.xo.copy()
        b.commit(a >= 0, np.ones((1, F), bool))
        return b, a, xo

    b, a, xo = run(k0)
    assert not b.xo.any() and not b.x.any()    # 확정에서 xo도 0
    kappa = b.kappa
    assert np.allclose(xo, (1 - kappa) ** 2 * odors[0] + (1 - kappa) * odors[1] + odors[2])
    eps = 1e-5
    for f in range(F):
        def log_pi(k):
            z = np.where(legal[0, f], b.beta * k * b.g * xo[0, f], -np.inf)
            z = z - z[legal[0, f]].max()
            return z[a[0, f]] - np.log(np.exp(z).sum())

        fd = (log_pi(k0[0, f] + eps) - log_pi(k0[0, f] - eps)) / (2 * eps)
        assert b.e_k[0, f] == pytest.approx(fd, abs=1e-8) and abs(fd) > 0
        # 뇌가 낸 π와 같은 식이다
        assert np.allclose(np.exp(log_pi(k0[0, f])), b.last_pi[0, f, a[0, f]], atol=1e-12)


def test_update_order_matches_synaptic_rule(circuit):
    """확정 1번에서 k ← k + η_k (R_n − R̄) e_k (그 전까지의 흔적), R̄ 갱신은 한 번, 그다음 e_k ← γ e_k + s."""
    b = kbrain(circuit)
    rng = np.random.default_rng(3)
    tick(b, obs(1, rng.random((1, F, 4)) * 3), R=np.full((1, F), 0.7))
    e1, k1 = b.e_k.copy(), b.k_gain.copy()
    assert e1.all() and (k1 == b.k_init).all() and (b.rbar == 0).all()   # 첫 확정: R_n = 0, e_k = 0
    tick(b, obs(1, rng.random((1, F, 4)) * 3))
    assert np.allclose(b.k_gain, k1 + b.eta_k * 0.7 * e1)
    assert np.allclose(b.rbar, b.alpha_rbar * 0.7)
    s2 = b.e_k - b.gamma * e1
    assert s2.all()
    b.give_reward(np.full((1, F), -1.0))
    k2, e2, rbar = b.k_gain.copy(), b.e_k.copy(), b.rbar.copy()
    b.flush_learning(np.ones(1, bool))                                   # flush도 같은 갱신
    assert np.allclose(b.k_gain, k2 + b.eta_k * (-1.0 - rbar) * e2)
    assert np.allclose(b.rbar, (1 - b.alpha_rbar) * rbar + b.alpha_rbar * -1.0) and np.array_equal(b.e_k, e2)


def _odor_task(circuit, sign, steps=400, **kw):
    """냄새가 가장 센 쪽을 고르면 sign, 아니면 −sign."""
    b = kbrain(circuit, **kw)
    rng = np.random.default_rng(4)
    for _ in range(steps):
        odor = rng.random((1, F, 4)) * 5
        a = tick(b, obs(1, odor))
        b.give_reward(np.where(a == odor.argmax(axis=-1), sign, -sign).astype(float))
    return b


def test_k_grows_when_following_odor_is_rewarded_and_shrinks_when_reversed(circuit):
    up, down = _odor_task(circuit, +1.0), _odor_task(circuit, -1.0)
    assert (up.k_gain > up.k_init + 0.5).all(), up.k_gain
    assert (down.k_gain < down.k_init).all() and down.k_gain.mean() < down.k_init - 0.5, down.k_gain
    assert not up.dw.any() and not down.dw.any()                         # 시냅스는 그대로 (freeze_synapses)


@pytest.mark.parametrize("kw", [{"condition": "b0"}, {"evaluate": True}])
def test_k_does_not_change_in_b0_or_evaluate(circuit, kw):
    b = kbrain(circuit, **kw)
    assert b.eta_k == 0.0 and b.eta == 0.0
    b.k_gain[:] = b.k_init if kw.get("condition") == "b0" else 3.0
    before = b.k_gain.copy()
    rng = np.random.default_rng(5)
    for _ in range(60):
        tick(b, obs(1, rng.random((1, F, 4)) * 40), R=rng.normal(0, 1, (1, F)))
    b.flush_learning(np.ones(1, bool))
    assert np.array_equal(b.k_gain, before) and b.e_k.any()


def test_freeze_synapses_keeps_dw_zero_and_unfrozen_learns_both(circuit):
    rng = np.random.default_rng(6)
    frozen, free = kbrain(circuit), kbrain(circuit, freeze_synapses=False)
    assert frozen.eta == 0.0 and free.eta == float(CFG_K.learn.eta)
    for _ in range(80):
        o, R = obs(1, rng.random((1, F, 4)) * 40), rng.normal(0, 1, (1, F))
        tick(frozen, o, R=R)
        tick(free, o, R=R)
    assert not frozen.dw.any() and not frozen.x.any() and (frozen.k_gain != frozen.k_init).all()
    assert free.dw.any() and (free.k_gain != free.k_init).all()


def test_k_is_clipped_to_range(circuit):
    for reward, kw, want in ((+100.0, {"k_max": 1.5}, 1.5), (-100.0, {"k_min": 0.5}, 0.5)):
        b = kbrain(circuit, **kw)
        tick(b, obs(1, np.random.default_rng(8).random((1, F, 4)) * 5))   # 활성 확정 한 번 (flush가 학습하게)
        b.e_k[:] = 1.0                                                     # 흔적을 정해 두고 큰 보상을 준다
        b.give_reward(np.full((1, F), reward))
        b.flush_learning(np.ones(1, bool))
        assert (b.k_gain == want).all(), b.k_gain
    with pytest.raises(ValueError, match="k_min"):
        kbrain(circuit, k_init=60.0)


def test_resets_follow_x_and_e(circuit):
    """xo는 x가 0이 되는 곳(확정, on_eaten, reset_life), e_k는 e가 0이 되는 곳(on_release, reset_life). k는 유지."""
    b = kbrain(circuit, B=2)
    rng = np.random.default_rng(7)
    tick(b, obs(2, rng.random((2, F, 4)) * 3), R=np.ones((2, F)))
    tick(b, obs(2, rng.random((2, F, 4)) * 3))
    b.propose(obs(2, rng.random((2, F, 4)) * 3, decide=False), np.ones(2, bool))
    b.commit(np.zeros((2, F), bool), np.ones((2, F), bool))
    k = b.k_gain.copy()
    assert b.xo.all() and b.e_k.all() and (k != b.k_init).all()
    eaten = np.zeros((2, F), bool)
    eaten[1, 2] = True
    b.on_eaten(eaten)
    assert not b.xo[1, 2].any() and b.xo[~eaten].all() and b.e_k.all()
    b.on_release(np.array([True, False]))
    assert not b.e_k[0].any() and b.e_k[1].all() and b.xo[0].all()
    b.reset_life(np.array([False, True]))
    assert not b.xo[1].any() and not b.e_k[1].any() and b.xo[0].all()
    assert np.array_equal(b.k_gain, k)


# ---------------------------------------------------------------- 저장 · 불러오기

def test_weight_file_roundtrip_with_k_gain(tmp_path):
    d, runner = _run(tmp_path, "r", _run_cfg(brain=BRAIN_K))
    g = runner.ghosts
    assert g.gain_on and not g.dw.any() and (g.k_gain != g.k_init).any()
    f = read_weight_file(d / "weights" / weight_filename(1, 3))
    assert f["k_gain"].shape == (F,) and f["k_gain"].dtype == np.float64 and np.array_equal(f["k_gain"], g.k_gain[1])
    fresh = FlyBrain(CFG_K, 2, [0, 1], condition="regime1", evaluate=True)
    fresh.load_slot_state(1, f)
    assert np.array_equal(fresh.k_gain[1], g.k_gain[1]) and (fresh.k_gain[0] == fresh.k_init).all()
    assert np.array_equal(fresh.rbar[1], g.rbar[1])
    # 켜진 뇌에 k_gain 없는 파일 → 오류. 꺼진 뇌에 k_gain 있는 상태 → 오류
    no_k = {k: v for k, v in f.items() if k != "k_gain"}
    with pytest.raises(ValueError, match="k_gain"):
        fresh.load_slot_state(0, no_k)
    off = FlyBrain(CFG, 1, [0], condition="regime1")
    with pytest.raises(ValueError, match="k_gain"):
        off.load_slot_state(0, {**off.slot_state(0), "k_gain": np.ones(F)})
    for bad in (np.ones(F - 1), np.full(F, fresh.k_max + 1.0), np.full(F, np.nan)):
        with pytest.raises(ValueError, match="k_gain"):
            fresh.load_slot_state(0, {**f, "k_gain": bad})


def test_state_dict_roundtrip_and_missing_k_gain(circuit):
    a = _odor_task(circuit, +1.0, steps=50)
    st = a.state_dict()
    assert np.array_equal(st["k_gain"], a.k_gain) and not np.shares_memory(st["k_gain"], a.k_gain)
    b = kbrain(circuit)
    b.load_state_dict(st)
    assert np.array_equal(b.k_gain, a.k_gain) and not b.e_k.any() and not b.xo.any()
    with pytest.raises(ValueError, match="k_gain"):
        kbrain(circuit).load_state_dict({k: v for k, v in st.items() if k != "k_gain"})
    with pytest.raises(ValueError, match="B0"):          # 학습된 k를 불러온 정책은 B0가 아니다
        kbrain(circuit, condition="b0").load_state_dict({**st, "condition": "b0"})


# ---------------------------------------------------------------- k = 1, η_k = 0인 켜진 뇌 = 기존 B0

def test_gain_brain_with_k1_and_no_learning_plays_same_games_as_b0(tmp_path):
    old, r_old = _run(tmp_path, "old", _run_cfg(brain=BRAIN, condition="b0"))
    new, r_new = _run(tmp_path, "new", _run_cfg(brain=BRAIN_K, condition="b0"))
    g = r_new.ghosts
    assert g.gain_on and g.eta_k == 0.0 and (g.k_gain == 1.0).all() and not r_old.ghosts.gain_on
    assert (old / "games.csv").read_bytes() == (new / "games.csv").read_bytes()


# ---------------------------------------------------------------- 설정

def load(name):
    return yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text(encoding="utf-8"))


def diff_keys(a: dict, b: dict) -> set:
    return {k for k in set(a) | set(b) if a.get(k) != b.get(k)}


def test_old_config_hashes_unchanged():
    for name, want in OLD_CONFIG_HASH.items():
        assert load_config(ROOT / "configs" / f"{name}.yaml").hash() == want, name


def test_brain_k_is_brain_v1_plus_block():
    old, new = load("brain_v1"), load("brain_k_v1")
    assert diff_keys(old, new) == {"instinct_gain"} and "instinct_gain" not in old
    assert new["instinct_gain"] == {"learn": True, "eta_k": 0.28, "k_init": 1.0, "k_min": 0.0, "k_max": 50.0,
                                    "freeze_synapses": True}
    assert Config(new["learn"]).hash() == Config(old["learn"]).hash()     # 회로 meta와 대조되는 해시


def test_kdev_differs_from_main_only_in_allowed_keys():
    main, dev = load("main_v1"), load("kdev_v1")
    assert diff_keys(main, dev) == {"experiment", "seeds", "ghosts", "weights", "replay"}
    assert dev["experiment"] == "kdev_v1" and dev["seeds"] == list(range(7500, 7510)) and dev["n_games"] == 3000
    assert dev["ghosts"] == {"kind": "flybrain", "config": BRAIN_K} and dev["weights"] == {"every_games": 50}
    assert diff_keys(main["replay"], dev["replay"]) == {"keep_games"}
    assert dev["replay"]["keep_games"] == [1, 1000, 2000, 3000]


def test_kdev_eval_differs_from_final_eval_only_in_allowed_keys():
    fin, ev = load("final_eval_v1"), load("kdev_eval_v1")
    assert diff_keys(fin, ev) == {"experiment", "seeds", "final_eval"}
    assert diff_keys(fin["final_eval"], ev["final_eval"]) == {"train_seeds", "conditions", "ghosts"}
    assert diff_keys(fin["final_eval"]["ghosts"], ev["final_eval"]["ghosts"]) == {"flybrain"}
    assert ev["experiment"] == "kdev_eval_v1" and ev["seeds"] == list(range(9300, 9310))
    fe = ev["final_eval"]
    assert fe["train_seeds"] == list(range(7500, 7510)) and fe["conditions"] == ["regime1", "regime2", "b0"]
    assert fe["ghosts"]["flybrain"] == {"kind": "flybrain", "config": BRAIN_K}
    dev = load("kdev_v1")
    assert fe["train_seeds"] == dev["seeds"] and fe["train_n_games"] == dev["n_games"]
    for k in ("engine", "time", "odor", "phero", "pacman"):               # 학습과 같아야 하는 블록 (28.4)
        assert ev[k] == dev[k], k


def test_new_seed_ranges_are_used_nowhere_else():
    """개발 시드(7500대 · 9300대)는 kdev 설정에만 있다. 확인 실행의 시드(8100대 · 9400대, 38.5)는 아직 어디에도 없다."""
    dev, confirm = set(range(7500, 7510)) | set(range(9300, 9310)), set(range(8100, 8110)) | set(range(9400, 9410))
    for path in sorted((ROOT / "configs").glob("*.yaml")):
        d = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        used = list(d.get("seeds") or []) + list((d.get("final_eval") or {}).get("train_seeds") or [])
        used = {s for s in used if isinstance(s, int)}
        assert not confirm & used, path.name
        if path.stem not in ("kdev_v1", "kdev_eval_v1"):
            assert not dev & used, path.name


# ---------------------------------------------------------------- scripts/k_curve.py

def _k_curve():
    name = "k_curve"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _fake_bundle(root: Path, table: dict, extra_conditions=("b0",)) -> Path:
    """table: {조건: {시드: {판: k_gain (4,)}}}. k_gain만 든 가중치 파일을 쓴다 (다른 것은 읽지 않아야 한다)."""
    for cond in (*table, *extra_conditions):
        d = root / f"{cond}_p1"
        d.mkdir(parents=True)
        (d / "config.yaml").write_text(yaml.safe_dump({"condition": cond}), encoding="utf-8")
        for seed, games in table.get(cond, {}).items():
            (d / "weights").mkdir(exist_ok=True)
            for game, k in games.items():
                np.savez(d / "weights" / weight_filename(seed, game), k_gain=np.asarray(k, float))
    return root


def test_k_curve_means_sd_and_last_game_ci(tmp_path, capsys):
    kc = _k_curve()
    table = {"regime1": {0: {50: [1, 1, 1, 1], 100: [1, 2, 3, 2]}, 1: {50: [2, 2, 2, 2], 100: [3, 3, 3, 3]},
                         2: {50: [3, 3, 3, 3], 100: [4, 4, 4, 4]}}}
    bundle = _fake_bundle(tmp_path / "b", table)
    assert kc.main(["k_curve.py", str(bundle)]) == 0
    res = kc.build(bundle)
    cv = res["conditions"]["regime1"]
    assert res["conditions"]["b0"] is None and cv["seeds"] == [0, 1, 2] and cv["games"] == [50, 100]
    assert [r["mean"] for r in cv["rows"]] == pytest.approx([2.0, 3.0])
    assert [r["sd"] for r in cv["rows"]] == pytest.approx([1.0, 1.0])
    ci = cv["last_minus_ref"]                      # k̄_s − 1 = 1, 2, 3: 평균 2, sd 1, t(0.975, 2) = 4.3027
    assert (ci["n"], ci["df"]) == (3, 2) and ci["mean"] == pytest.approx(2.0) and ci["sd"] == pytest.approx(1.0)
    half = 4.302652729911275 / 3 ** 0.5
    assert ci["lo"] == pytest.approx(2.0 - half) and ci["hi"] == pytest.approx(2.0 + half) and not ci["excludes_zero"]
    assert (bundle / "k_curve.json").exists() and "regime1" in (bundle / "k_curve.md").read_text(encoding="utf-8")
    assert "가중치 파일 없음" in capsys.readouterr().out


def test_k_curve_stops_on_missing_k_gain_or_uneven_games(tmp_path, capsys):
    kc = _k_curve()
    uneven = _fake_bundle(tmp_path / "u", {"regime1": {0: {50: [1] * 4, 100: [1] * 4}, 1: {50: [1] * 4}}})
    assert kc.main(["k_curve.py", str(uneven)]) == 1 and "저장 판" in capsys.readouterr().err
    old = _fake_bundle(tmp_path / "o", {"regime1": {0: {50: [1] * 4}, 1: {50: [1] * 4}}})
    np.savez(old / "regime1_p1" / "weights" / weight_filename(0, 50), rbar=np.zeros(4))
    assert kc.main(["k_curve.py", str(old)]) == 1 and "k_gain" in capsys.readouterr().err


def test_k_curve_reads_real_weight_files(tmp_path):
    """실행기가 쓴 묶음 모양(조건 폴더 / weights)에서 끝까지 돈다. games.csv를 지워도 된다."""
    kc = _k_curve()
    for cond in ("regime1", "b0"):
        d, _ = _run(tmp_path / "bundle", f"{cond}_p1", _run_cfg(brain=BRAIN_K, condition=cond))
        (d / "games.csv").unlink()
    res = kc.build(tmp_path / "bundle")
    cv = res["conditions"]["regime1"]
    assert res["conditions"]["b0"] is None and cv["seeds"] == [0, 1] and cv["games"] == [1, 2, 3]
    f = read_weight_file(tmp_path / "bundle" / "regime1_p1" / "weights" / weight_filename(0, 3))
    assert cv["last_k_by_seed"][0] == pytest.approx(float(f["k_gain"].mean()))


if __name__ == "__main__":   # default_run이 별도 프로세스로 부른다: 결과 폴더와 지문을 쓴다
    _out = Path(sys.argv[1])
    _cfg = _run_cfg()
    _runner = build_runner(_cfg, Recorder(_out, _cfg))
    _runner.run()
    (_out / "probe.txt").write_text(_numeric_probe(_runner.ghosts.circuit) + "\n", encoding="utf-8")
