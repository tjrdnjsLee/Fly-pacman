"""PPO 팩맨 학습기: 체크포인트 · 재개 · 설정 대조 (지시서 4.2). torch가 없으면 건너뛴다."""

from __future__ import annotations

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from flypac.core.config import Config  # noqa: E402
from flypac.pacman.obs import OBS_VERSION  # noqa: E402
from flypac.pacman.train_env import PacmanTrainEnv  # noqa: E402
from flypac.pacman.trainer import CKPT_FORMAT, Trainer  # noqa: E402
from test_pacman_obs import BASE, TRAIN  # noqa: E402

SMALL = {
    "num_envs": 3,
    "curriculum": [{"from_update": 0, "ghosts": "random", "condition": "regime1"},
                   {"from_update": 2, "ghosts": "random", "condition": "regime1"}],
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
                 "eval_max_ticks": 50, "checkpoint_every": 2, "keep_checkpoints": 2,
                 "eval_opponent": {"ghosts": "random", "condition": "regime1"}},
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


def load_ckpt(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def csv_rows(path):
    import csv
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def test_trains_and_logs(tmp_path):
    tr = Trainer(small_cfg(), run_dir=tmp_path, device="cpu", quiet=True)
    tr.train()
    assert tr.update == 6
    log = (tmp_path / "train_log.csv").read_text(encoding="utf-8-sig").splitlines()
    assert log[0].startswith("update,phase,steps_per_sec") and len(log) == 7
    assert "커리큘럼 단계 1" in (tmp_path / "train_log.txt").read_text(encoding="utf-8")
    ev = csv_rows(tmp_path / "eval_log.csv")
    assert list(ev[0]) == ["update", "ghosts", "games", "censored", "unfinished", "win_rate", "mean_score"]
    ck = sorted(p.name for p in (tmp_path / "checkpoints").glob("update_*.pt"))
    assert ck == ["update_000004.pt", "update_000006.pt"]  # 최근 2개만
    assert (tmp_path / "checkpoints" / "latest.pt").exists()


def test_checkpoint_contents(tmp_path):
    tr = Trainer(small_cfg(), run_dir=tmp_path, device="cpu", quiet=True)
    tr.train(stop_at=2)
    s = load_ckpt(tmp_path / "checkpoints" / "latest.pt")
    for key in ("model", "optimizer", "update", "phase", "rng_numpy", "rng_torch", "env", "rnd", "reward_norm",
                "obs_version", "config", "config_hash", "commit", "format", "engine_constants", "total_updates"):
        assert key in s, key
    assert s["update"] == 2 and s["obs_version"] == OBS_VERSION and s["format"] == CKPT_FORMAT
    assert s["engine_constants"] == {"power_ticks_max": 57, "lives": 3}  # 엔진 설정에서: round(6 × 9.47), 초기 목숨
    assert s["total_updates"] == 6  # 이번 실행을 멈춘 지점(2)이 아니라 일정의 총 길이


def test_resume_matches_uninterrupted(tmp_path):
    """중간(stop_at)에서 멈추고 재개해도 끊지 않은 학습과 같다: 가중치 · 보상 정규화 · 학습률 · 엔트로피 일정까지."""
    a = Trainer(small_cfg(), run_dir=tmp_path / "a", device="cpu", quiet=True)
    a.train()

    b = Trainer(small_cfg(), run_dir=tmp_path / "b", device="cpu", quiet=True)
    b.train(stop_at=3)  # 커리큘럼 전환(2)을 지난 뒤 멈춘다. 저장은 train이 한다
    assert b.update == 3 and load_ckpt(tmp_path / "b" / "checkpoints" / "latest.pt")["update"] == 3
    del b
    c = Trainer(small_cfg(), run_dir=tmp_path / "b", device="cpu", quiet=True)
    c.load(tmp_path / "b")
    assert c.update == 3 and c.phase == 1
    c.train()
    assert c.update == 6
    for x, y in zip(params(a), params(c)):
        assert np.array_equal(x, y)
    assert a.reward_norm.state_dict() == c.reward_norm.state_dict()
    # 학습률 · 엔트로피 계수가 갱신마다 같다 (멈춘 지점이 일정을 바꾸지 않는다)
    ra, rb = csv_rows(tmp_path / "a" / "train_log.csv"), csv_rows(tmp_path / "b" / "train_log.csv")
    assert [(r["update"], r["lr"], r["entropy_coef"]) for r in ra] == [(r["update"], r["lr"], r["entropy_coef"]) for r in rb]
    lrs = [float(r["lr"]) for r in ra]
    assert lrs[0] == pytest.approx(2.5e-4) and lrs[3] == pytest.approx(2.5e-4 * (1 - 3 / 6))  # 총 길이 6 기준


def test_stop_at_never_changes_schedule(tmp_path):
    tr = Trainer(small_cfg(), run_dir=tmp_path, device="cpu", quiet=True)
    tr.train(stop_at=1)
    assert tr.ppo.optimizer.param_groups[0]["lr"] == pytest.approx(2.5e-4)  # 총 1짜리 일정이었다면 0이 된다
    with pytest.raises(ValueError, match="stop_at"):
        tr.train(stop_at=7)  # 일정의 총 길이를 넘을 수 없다


def test_resume_rejects_changed_learning_config(tmp_path):
    a = Trainer(small_cfg(), run_dir=tmp_path, device="cpu", quiet=True)
    a.train(stop_at=2)
    b = Trainer(small_cfg(ppo={"learning_rate": 1e-3}), run_dir=tmp_path, device="cpu", quiet=True)
    with pytest.raises(ValueError, match="학습 설정"):
        b.load(tmp_path)
    # 총 갱신 수는 일정을 정하므로 바꿀 수 없다
    t = Trainer(small_cfg(training={"total_updates": 10}), run_dir=tmp_path, device="cpu", quiet=True)
    with pytest.raises(ValueError, match="총 갱신 수"):
        t.load(tmp_path)
    # 기록 주기는 바꿔도 된다
    c = Trainer(small_cfg(training={"log_every": 5}), run_dir=tmp_path, device="cpu", quiet=True)
    c.load(tmp_path)
    assert c.update == 2


def test_load_checks_format_commit_and_engine_constants(tmp_path):
    a = Trainer(small_cfg(), run_dir=tmp_path, device="cpu", quiet=True)
    a.train(stop_at=2)
    path = tmp_path / "checkpoints" / "latest.pt"
    s = load_ckpt(path)

    def tampered(name, **over):
        torch.save({**s, **over}, tmp_path / name)
        return tmp_path / name

    t = Trainer(small_cfg(), run_dir=tmp_path / "t", device="cpu", quiet=True)
    with pytest.raises(ValueError, match="저장 형식"):
        t.load(tampered("f1.pt", format=1))
    with pytest.raises(ValueError, match="엔진 숫자"):
        t.load(tampered("c.pt", engine_constants={"power_ticks_max": 40, "lives": 3}))
    # 커밋이 다르면 경고만: 재개는 된다
    t.load(tampered("k.pt", commit="0000000+other"))
    assert t.update == 2
    assert "경고" in (tmp_path / "t" / "train_log.txt").read_text(encoding="utf-8")


def test_evaluation_excludes_censored_from_denominator_and_best(tmp_path, monkeypatch):
    """CENSORED 판은 승률 분모 · 평균 점수 · best 선택에서 빼고 따로 센다."""
    import flypac.pacman.trainer as trainer_mod

    scripted = [
        [{"slot": 0, "score": 100, "winner": "PACMAN", "return": 0.0, "length": 1},
         {"slot": 2, "score": 999, "winner": "CENSORED", "return": 0.0, "length": 1}],
        [{"slot": 1, "score": 50, "winner": "FLIES", "return": 0.0, "length": 1}],
    ]

    class FakeEnv:
        def __init__(self, *a, **k):
            self.i = 0
            self.constants = {"power_ticks_max": 57, "lives": 3}

        def reset(self):
            return {"grid": np.zeros((4, 32, 31, 28), dtype=np.uint8), "scalars": np.zeros((4, 5), dtype=np.float32)}

        def legal_masks(self):
            return np.ones((4, 4), dtype=bool)

        def step(self, a):
            fin = scripted[self.i] if self.i < len(scripted) else []
            self.i += 1
            return self.reset(), np.zeros(4), np.zeros(4, dtype=bool), {"finished": fin}

    tr = Trainer(small_cfg(), run_dir=tmp_path, device="cpu", quiet=True)
    monkeypatch.setattr(trainer_mod, "PacmanTrainEnv", FakeEnv)
    res = tr.evaluate(4, "random", "regime1", 0, 5)
    assert (res["games"], res["censored"], res["unfinished"]) == (2, 1, 1)
    assert res["win_rate"] == pytest.approx(0.5) and res["mean_score"] == pytest.approx(75.0)  # 999는 평균에 없다
    # 모두 중단이면 계산할 판이 없다: best로 뽑지 않는다
    monkeypatch.setattr(tr, "evaluate", lambda *a, **k: {"games": 0, "censored": 4, "unfinished": 0,
                                                       "win_rate": float("nan"), "mean_score": float("nan")})
    assert tr._evaluate(10) is False and tr.best is None
    monkeypatch.setattr(tr, "evaluate", lambda *a, **k: {"games": 2, "censored": 1, "unfinished": 0,
                                                       "win_rate": 0.5, "mean_score": 75.0})
    assert tr._evaluate(20) is True and tr.best == (0.5, 75.0)
    last = csv_rows(tmp_path / "eval_log.csv")[-1]
    assert (last["games"], last["censored"]) == ("2", "1")


def test_stop_flag_saves_and_exits(tmp_path):
    """Ctrl+C를 받은 것과 같은 상태: 이번 갱신을 마치고 저장한 뒤 끝낸다."""
    tr = Trainer(small_cfg(), run_dir=tmp_path, device="cpu", quiet=True)
    tr._stop = True
    tr.train()
    assert tr.update == 1
    s = load_ckpt(tmp_path / "checkpoints" / "latest.pt")
    assert s["update"] == 1
    assert "Ctrl+C" in (tmp_path / "train_log.txt").read_text(encoding="utf-8")


def load_script(name):
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name[:-3], script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_start_refuse_and_resume(tmp_path):
    import yaml

    from flypac.core.config import load_config

    train_pacman = load_script("train_pacman.py")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(small_cfg().to_dict(), allow_unicode=True), encoding="utf-8")
    run = tmp_path / "run"
    args = ["--config", str(cfg_path), "--run-dir", str(run), "--device", "cpu"]
    assert train_pacman.main(args + ["--stop-at", "2"]) == 0
    assert train_pacman.main(args + ["--stop-at", "3"]) == 2  # 이미 있으면 덮어쓰지 않는다
    assert train_pacman.main(args + ["--stop-at", "3", "--resume", str(run)]) == 0
    s = load_ckpt(run / "checkpoints" / "latest.pt")
    assert s["update"] == 3 and s["total_updates"] == 6
    assert train_pacman.main(args + ["--resume", str(run)]) == 0  # 멈출 지점 없이: 일정 끝까지
    assert load_ckpt(run / "checkpoints" / "latest.pt")["update"] == 6
    assert load_config(cfg_path).pacman_train.num_envs == 3


def test_cli_stops_before_training_when_a_part_is_missing(tmp_path, capsys):
    import yaml

    train_pacman = load_script("train_pacman.py")
    data = small_cfg().to_dict()
    data["engine"]["kind"] = "bogus"
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    code = train_pacman.main(["--config", str(cfg_path), "--run-dir", str(tmp_path / "r"), "--device", "cpu"])
    assert code == 3 and "시작 검사 실패" in capsys.readouterr().err
    assert not (tmp_path / "r" / "checkpoints").exists()  # 학습을 시작하지 않았다


def trained_checkpoint(tmp_path, updates=2):
    tr = Trainer(small_cfg(), run_dir=tmp_path, device="cpu", quiet=True)
    tr.train(stop_at=updates)
    return tr, tmp_path / "checkpoints" / "latest.pt"


def test_ppo_pacman_controller(tmp_path):
    from flypac.core import directions as d
    from flypac.core.rng import game_rngs
    from flypac.maps.original import load_original
    from flypac.pacman.ppo import PPOPacman

    tr, ck = trained_checkpoint(tmp_path)
    cfg = small_cfg()
    amap = load_original()
    env = PacmanTrainEnv(cfg, 2, 0, ghosts="random", condition="regime1")
    env.reset()
    views = env.game.views()

    pac = PPOPacman(str(ck), "cpu", 0.0, game_rngs([0, 1], "pacman"), cfg)
    pac.reset(np.ones(2, dtype=bool))
    # 여러 틱, 목숨 · 판 종료 때 reset: 탐욕 행동은 학습기 신경망의 argmax와 같다 (같은 관측 코드)
    rng = np.random.default_rng(0)
    n_ends = 0
    for t in range(400):
        legal = env.legal_masks()
        a = pac.act(env._views)
        assert a.shape == (2,) and a.dtype == np.int64 and all(legal[b, a[b]] for b in range(2))  # 반전 금지 마스크 안
        env_obs = {"grid": pac.observer.current()["grid"], "scalars": pac.observer.encoder.encode(env._views)[1]}
        with torch.no_grad():
            logits, _ = tr.network(torch.as_tensor(env_obs["grid"]).float(), torch.as_tensor(env_obs["scalars"]),
                                   torch.as_tensor(legal))
        assert np.array_equal(a, logits.argmax(-1).numpy()), f"틱 {t}"
        _, _, _, info = env.step(rng.integers(0, 4, size=2))  # 환경은 무작위로 굴려 목숨 · 판 종료를 만든다
        n_ends += int(info["ends"].sum())
        pac.reset(info["ends"])
    assert n_ends > 0, "목숨을 잃는 경우가 시험에 들어 있어야 한다"

    # ε = 1: 엔진이 허용하는 방향 전체(반전 포함) 중 균등하게 고른다. 탐욕 마스크는 반전을 막는다
    env.reset()
    views = env._views
    legal = env.legal_masks()
    free = amap.move_mask["pacman"][views[0].pac_cell]
    back = int(d.opposite(views[0].pac_dir))
    assert free[back] and not legal[0, back]
    rnd = PPOPacman(str(ck), "cpu", 1.0, game_rngs([0, 1], "pacman"), cfg)
    picks = {int(rnd.act(views)[0]) for _ in range(80)}
    assert picks == set(np.flatnonzero(free).tolist()) and back in picks


def test_ppo_pacman_checks_header_and_engine_numbers(tmp_path):
    from flypac.core.rng import game_rngs
    from flypac.pacman.ppo import PPOPacman

    _, ck = trained_checkpoint(tmp_path)
    s = load_ckpt(ck)
    rngs = lambda: game_rngs([0], "pacman")  # noqa: E731
    cfg = small_cfg()
    bad_fmt = tmp_path / "fmt.pt"
    torch.save({**s, "format": 1}, bad_fmt)
    with pytest.raises(ValueError, match="저장 형식"):
        PPOPacman(str(bad_fmt), "cpu", 0.0, rngs(), cfg)
    # 실험 설정의 엔진 숫자가 다르면 멈춘다 (파워 분모 · 초기 목숨)
    for key, val in (("power_s", 3), ("lives", 5)):
        data = cfg.to_dict()
        data["engine"][key] = val
        with pytest.raises(ValueError, match="엔진 숫자"):
            PPOPacman(str(ck), "cpu", 0.0, rngs(), Config(data))
    # 커밋이 다르면 경고만
    other = tmp_path / "commit.pt"
    torch.save({**s, "commit": "0000000+other"}, other)
    with pytest.warns(UserWarning, match="커밋"):
        PPOPacman(str(other), "cpu", 0.0, rngs(), cfg)


def test_run_experiment_command_runs_one_game_with_ppo_checkpoint(tmp_path):
    """실제 실행 명령(scripts/run_experiment.py)으로 체크포인트를 읽어 한 판을 끝까지 돈다 (가짜 엔진)."""
    import subprocess
    import sys
    from pathlib import Path

    import yaml

    _, ck = trained_checkpoint(tmp_path / "train")
    root = Path(__file__).resolve().parents[1]
    run_cfg = load_yaml(root / "configs" / "m0_smoke.yaml")
    run_cfg.update(experiment="ppo_smoke", seeds=[0, 1], n_games=1,
                   pacman={"kind": "ppo", "checkpoint": str(ck), "device": "cpu", "eps": 0.1})
    run_cfg["engine"].update(lives=3, watchdog_ticks=400)
    cfg_path = tmp_path / "ppo_run.yaml"
    cfg_path.write_text(yaml.safe_dump(run_cfg, allow_unicode=True), encoding="utf-8")
    cmd = [sys.executable, str(root / "scripts" / "run_experiment.py"), str(cfg_path)]
    proc = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True, encoding="utf-8", timeout=300)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "끝:" in proc.stdout and "2판" in proc.stdout  # 칸 2개 × 1판
    games = list((tmp_path / "runs").glob("*_ppo_smoke/games.csv"))
    assert len(games) == 1
    rows = games[0].read_text(encoding="utf-8-sig").splitlines()
    assert len(rows) == 3  # 머리글 + 2판

    # 체크포인트의 엔진 숫자와 다른 실험 설정은 실행 명령이 시작 때 거부한다
    run_cfg["engine"]["power_s"] = 3
    cfg_path.write_text(yaml.safe_dump(run_cfg, allow_unicode=True), encoding="utf-8")
    proc = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True, encoding="utf-8", timeout=300)
    assert proc.returncode != 0 and "엔진 숫자" in proc.stderr


def load_yaml(path):
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8"))
