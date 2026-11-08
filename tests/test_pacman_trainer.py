"""PPO 팩맨 학습기: 체크포인트 · 재개 · 설정 대조 (지시서 4.2). torch가 없으면 건너뛴다."""

from __future__ import annotations

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from flypac.core.config import Config  # noqa: E402
from flypac.pacman.obs import OBS_VERSION  # noqa: E402
from flypac.pacman.trainer import Trainer  # noqa: E402
from test_pacman_obs import BASE, TRAIN  # noqa: E402

SMALL = {
    "num_envs": 3,
    "curriculum": [{"from_update": 0, "ghosts": "random"}, {"from_update": 2, "ghosts": "random"}],
    "entropy_boost_updates": 2,
    "network": {"cnn_channels": [4, 8], "cnn_kernels": [3, 3], "cnn_strides": [2, 2],
                "shared_hidden": 16, "head_hidden": 8},
    "rnd": {"enabled": True, "hidden_size": 8, "output_size": 4, "learning_rate": 1e-3,
            "intrinsic_coef": 0.1, "update_proportion": 0.25, "min_samples": 4, "clip": 5.0},
    "ppo": {"rollout_steps": 8, "num_epochs": 2, "minibatch_size": 8, "learning_rate": 2.5e-4,
            "adam_eps": 1e-5, "lr_anneal": True, "gamma": 0.99, "gae_lambda": 0.95, "clip_epsilon": 0.2,
            "value_loss_coef": 0.5, "entropy_coef_start": 0.15, "entropy_coef_end": 0.01,
            "entropy_anneal_fraction": 0.7, "max_grad_norm": 0.5, "reward_norm_clip": 10.0},
    "training": {"total_updates": 6, "log_every": 1, "eval_every": 3, "eval_games": 2,
                 "eval_max_ticks": 50, "checkpoint_every": 2, "keep_checkpoints": 2},
}


def small_cfg(**over) -> Config:
    data = copy.deepcopy(BASE)
    p = copy.deepcopy(TRAIN)
    p.update(copy.deepcopy(SMALL))
    for k, v in over.items():
        if isinstance(v, dict):
            p[k].update(v)
        else:
            p[k] = v
    data["pacman_train"] = p
    return Config(data)


def params(tr: Trainer) -> list[np.ndarray]:
    return [t.detach().cpu().numpy().copy() for t in tr.network.state_dict().values()]


def test_trains_and_logs(tmp_path):
    tr = Trainer(small_cfg(), run_dir=tmp_path, device="cpu", quiet=True)
    tr.train()
    assert tr.update == 6
    log = (tmp_path / "train_log.csv").read_text(encoding="utf-8-sig").splitlines()
    assert log[0].startswith("update,phase,steps_per_sec") and len(log) == 7
    assert "커리큘럼 단계 1" in (tmp_path / "train_log.txt").read_text(encoding="utf-8")
    assert (tmp_path / "eval_log.csv").exists()
    ck = sorted(p.name for p in (tmp_path / "checkpoints").glob("update_*.pt"))
    assert ck == ["update_000004.pt", "update_000006.pt"]  # 최근 2개만
    assert (tmp_path / "checkpoints" / "latest.pt").exists()


def test_checkpoint_contents(tmp_path):
    tr = Trainer(small_cfg(), run_dir=tmp_path, device="cpu", quiet=True)
    tr.train(total_updates=2)
    s = torch.load(tmp_path / "checkpoints" / "latest.pt", map_location="cpu", weights_only=False)
    for key in ("model", "optimizer", "update", "phase", "rng_numpy", "rng_torch", "env", "rnd",
                "reward_norm", "obs_version", "config", "config_hash", "commit"):
        assert key in s, key
    assert s["update"] == 2 and s["obs_version"] == OBS_VERSION


def test_resume_matches_uninterrupted(tmp_path):
    """짧은 학습 → 저장 → 재개 → 몇 갱신 더 = 끊지 않고 같은 수만큼 (같은 장치 · 같은 시드)."""
    a = Trainer(small_cfg(), run_dir=tmp_path / "a", device="cpu", quiet=True)
    a.train(total_updates=6)

    b = Trainer(small_cfg(), run_dir=tmp_path / "b", device="cpu", quiet=True)
    b.train(total_updates=6, stop_after=3)  # 커리큘럼 전환(2)을 지난 뒤 끊는다
    b.save()
    del b
    c = Trainer(small_cfg(), run_dir=tmp_path / "b", device="cpu", quiet=True)
    c.load(tmp_path / "b")
    assert c.update == 3 and c.phase == 1
    c.train(total_updates=6)
    assert c.update == 6
    for x, y in zip(params(a), params(c)):
        assert np.array_equal(x, y)
    assert a.reward_norm.state_dict() == c.reward_norm.state_dict()


def test_resume_rejects_changed_learning_config(tmp_path):
    a = Trainer(small_cfg(), run_dir=tmp_path, device="cpu", quiet=True)
    a.train(total_updates=2)
    b = Trainer(small_cfg(ppo={"learning_rate": 1e-3}), run_dir=tmp_path, device="cpu", quiet=True)
    with pytest.raises(ValueError, match="학습 설정"):
        b.load(tmp_path)
    # 갱신 수 · 기록 주기는 바꿔도 된다
    c = Trainer(small_cfg(training={"total_updates": 10, "log_every": 5}), run_dir=tmp_path, device="cpu",
                quiet=True)
    c.load(tmp_path)
    assert c.update == 2


def test_stop_flag_saves_and_exits(tmp_path):
    """Ctrl+C를 받은 것과 같은 상태: 이번 갱신을 마치고 저장한 뒤 끝낸다."""
    tr = Trainer(small_cfg(), run_dir=tmp_path, device="cpu", quiet=True)
    tr._stop = True
    tr.train(total_updates=6)
    assert tr.update == 1
    s = torch.load(tmp_path / "checkpoints" / "latest.pt", map_location="cpu", weights_only=False)
    assert s["update"] == 1
    assert "Ctrl+C" in (tmp_path / "train_log.txt").read_text(encoding="utf-8")


def test_cli_start_refuse_and_resume(tmp_path):
    import importlib.util
    from pathlib import Path

    import yaml

    from flypac.core.config import load_config

    script = Path(__file__).resolve().parents[1] / "scripts" / "train_pacman.py"
    spec = importlib.util.spec_from_file_location("train_pacman", script)
    train_pacman = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_pacman)

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(small_cfg().to_dict(), allow_unicode=True), encoding="utf-8")
    run = tmp_path / "run"
    args = ["--config", str(cfg_path), "--run-dir", str(run), "--device", "cpu"]
    assert train_pacman.main(args + ["--updates", "2"]) == 0
    assert train_pacman.main(args + ["--updates", "3"]) == 2  # 이미 있으면 덮어쓰지 않는다
    assert train_pacman.main(args + ["--updates", "3", "--resume", str(run)]) == 0
    s = torch.load(run / "checkpoints" / "latest.pt", map_location="cpu", weights_only=False)
    assert s["update"] == 3
    assert load_config(cfg_path).pacman_train.num_envs == 3


def test_ppo_pacman_controller(tmp_path):
    from flypac.core import directions as d
    from flypac.pacman.ppo import PPOPacman
    from flypac.pacman.train_env import PacmanTrainEnv

    tr = Trainer(small_cfg(), run_dir=tmp_path, device="cpu", quiet=True)
    tr.train(total_updates=2)
    ck = tmp_path / "checkpoints" / "latest.pt"
    env = PacmanTrainEnv(small_cfg(), 2, 0, ghosts="random")
    env.reset()
    views = env.game.views()

    pac = PPOPacman(str(ck), "cpu", 0.0)
    pac.reset(np.ones(2, dtype=bool))
    a = pac.act(views)
    assert a.shape == (2,) and a.dtype == np.int64
    legal = env.legal_masks()
    assert all(legal[b, a[b]] for b in range(2))
    # 탐욕 행동은 학습기 신경망의 argmax와 같다 (같은 관측 코드)
    obs = env._observe()
    with torch.no_grad():
        logits, _ = tr.network(torch.as_tensor(obs["grid"]).float(), torch.as_tensor(obs["scalars"]),
                               torch.as_tensor(legal))
    assert np.array_equal(a, logits.argmax(-1).numpy())

    rnd = PPOPacman(str(ck), "cpu", 1.0, rng=np.random.default_rng(0))
    picks = {int(rnd.act(views)[0]) for _ in range(50)}
    assert picks == set(np.flatnonzero(legal[0]).tolist())  # ε=1: 가능한 방향 중 균등
    assert all(0 <= x <= d.WEST for x in picks)
