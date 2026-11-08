"""PPO 팩맨 속도 실측 (지시서 4.4, TRD 12.3).

    uv run --extra pacman python scripts/bench_pacman.py --config configs/pacman_train_v1.yaml

1. 환경 단독: 무작위 행동, B개 게임, 초당 스텝 (엔진 + 상대 유령 + 관측 만들기).
2. 전체 학습 처리량: PPO 갱신 포함 몇 번의 갱신, 초당 스텝 (첫 갱신은 준비 시간이라 뺀다).
3. 예상 시간: total_updates × num_envs × rollout_steps 스텝.
결과는 화면과 --out(기본 <임시 폴더>/bench_pacman.json)에 남긴다. 측정 장치(CPU · GPU)도 적는다.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from flypac.core.config import load_config


def bench_env(cfg, B: int, steps: int, ghosts: str) -> float:
    from flypac.pacman.train_env import PacmanTrainEnv
    env = PacmanTrainEnv(cfg, B, 0, ghosts=ghosts)
    env.reset()
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    for _ in range(steps):
        legal = env.legal_masks()
        acts = np.array([rng.choice(np.flatnonzero(m)) for m in legal])
        env.step(acts)
    return B * steps / (time.perf_counter() - t0)


def device_info(device) -> dict:
    import torch
    info = {"cpu": platform.processor() or platform.machine(), "python": platform.python_version(),
            "torch": torch.__version__, "device": str(device), "threads": torch.get_num_threads()}
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device)
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="PPO 팩맨 속도 실측")
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--env-steps", type=int, default=200, help="환경 단독 측정 스텝 수")
    ap.add_argument("--updates", type=int, default=4, help="전체 처리량 측정 갱신 수 (첫 갱신 제외)")
    ap.add_argument("--ghosts", default=None, help="상대 유령 (기본: 커리큘럼 마지막 단계)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    from flypac.pacman.trainer import Trainer

    cfg = load_config(args.config)
    p = cfg.pacman_train
    B, T = int(p.num_envs), int(p.ppo.rollout_steps)
    total_updates = int(p.training.total_updates)
    ghosts = args.ghosts or str(p.curriculum[-1]["ghosts"])

    env_sps = bench_env(cfg, B, args.env_steps, ghosts)
    print(f"환경 단독 ({ghosts} 유령, 게임 {B}개): 초당 {env_sps:,.0f} 스텝", flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        tr = Trainer(cfg, run_dir=tmp, device=args.device, quiet=True)
        if ghosts != tr.phases[0][1]:
            tr.phases = [(0, ghosts)]
            tr.env.set_ghosts(ghosts)
        tr.train(total_updates=total_updates, stop_after=1)  # 준비 (첫 갱신)
        t0 = time.perf_counter()
        tr.train(total_updates=total_updates, stop_after=args.updates)
        dt = time.perf_counter() - t0
        info = device_info(tr.device)
    train_sps = args.updates * B * T / dt
    total_steps = total_updates * B * T
    hours = total_steps / train_sps / 3600
    print(f"전체 학습 처리량 (갱신 {args.updates}회, 장치 {info['device']}): 초당 {train_sps:,.0f} 스텝, "
          f"갱신당 {dt / args.updates:.2f}초", flush=True)
    print(f"예상 시간: {total_updates} × {B} × {T} = {total_steps:,} 스텝 → 약 {hours:.1f}시간"
          + ("  (36시간 초과: 갱신 횟수를 줄이는 안을 보고한다, TRD 12.3)" if hours > 36 else ""))
    result = {"config": args.config, "engine_kind": str(cfg.engine.kind), "ghosts": ghosts, "num_envs": B,
              "rollout_steps": T, "env_steps_per_sec": env_sps, "train_steps_per_sec": train_sps,
              "sec_per_update": dt / args.updates, "total_updates": total_updates, "total_steps": total_steps,
              "estimated_hours": hours, "machine": info, "measured_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    out = Path(args.out) if args.out else Path(tempfile.gettempdir()) / "bench_pacman.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"결과 파일: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
