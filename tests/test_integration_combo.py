"""조합 시험: 진짜 엔진 + PPO 팩맨 + 초파리 뇌 + 기록기가 한 판 끝까지 도는가 (configs/integ_v1.yaml).

학습곡선이나 성능은 보지 않는다. torch가 없으면 건너뛴다 (CI에 torch를 넣는 일은 다음 PR).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pathlib import Path  # noqa: E402

from flypac.core.config import Config, load_config  # noqa: E402
from flypac.core.runner import build_runner  # noqa: E402
from flypac.recording.recorder import Recorder  # noqa: E402
from flypac.replay.reader import read_replay  # noqa: E402
from test_pacman_trainer import small_cfg, trained_checkpoint  # noqa: E402

INTEG = Path(__file__).resolve().parents[1] / "configs" / "integ_v1.yaml"


def _cfg(condition, checkpoint, tmp_engine):
    d = load_config(INTEG).to_dict()
    d["condition"] = condition
    d["seeds"] = [0]
    d["n_games"] = 1
    d["engine"] = {**d["engine"], **tmp_engine}
    d["pacman"] = {"kind": "ppo", "checkpoint": str(checkpoint), "device": "cpu", "eps": 0.0}
    d["replay"] = {"keep_games": [1], "brain": True}
    return Config(d)


@pytest.mark.parametrize("condition", ["regime1", "regime2", "b0"])
def test_real_engine_ppo_flybrain_recorder_runs_one_game(tmp_path, condition):
    tr, ck = trained_checkpoint(tmp_path / "ppo")
    train_engine = small_cfg().engine.to_dict()
    # 체크포인트와 실험의 엔진 숫자(파워 분모 · 목숨)를 맞추고, 짧은 판 제한만 건다
    cfg = _cfg(condition, ck, {k: train_engine[k] for k in ("power_s", "lives") if k in train_engine})
    cfg = Config({**cfg.to_dict(), "engine": {**cfg.engine.to_dict(), "watchdog_ticks": 300}})
    run_dir = tmp_path / "run"
    summary = build_runner(cfg, Recorder(run_dir, cfg)).run()

    assert len(summary["games"]) == 1 and summary["ticks"] > 0
    assert summary["games"][0]["winner"] in ("FLIES", "PACMAN", "CENSORED")
    for name in ("config.yaml", "games.csv", "lives.csv", "flies.csv"):
        assert (run_dir / name).exists(), name
    saved = load_config(run_dir / "config.yaml")
    if condition != "b0":
        assert saved.reward.p == pytest.approx(1 / 3) and saved.reward.q == 1.0
    replays = sorted((run_dir / "replay").glob("*.npz"))
    assert len(replays) == 1
    _, log = read_replay(replays[0])
    assert np.asarray(log["brain_on"]).any()  # 뇌 필드가 실제로 계산되어 기록됨
