"""PPO 팩맨 학습 (TRD 12.2). 명령 하나로 시작 · 재개한다. 대화형 입력은 없다.

    uv run --extra pacman python scripts/train_pacman.py --config configs/pacman_train_v1.yaml
    uv run --extra pacman python scripts/train_pacman.py --config configs/pacman_train_v1.yaml --resume <run_dir 또는 .pt>

중단: Ctrl+C 한 번 → 이번 갱신을 마치고 checkpoints/latest.pt에 저장하고 끝낸다.
진행 상황은 화면과 <run_dir>/train_log.txt, train_log.csv에 남는다. 평가는 eval_log.csv.
"""

from __future__ import annotations

import argparse
import sys

from flypac.core.config import load_config


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="PPO 팩맨 학습")
    ap.add_argument("--config", required=True, help="설정 YAML (예: configs/pacman_train_v1.yaml)")
    ap.add_argument("--resume", default=None, help="이어서 학습할 체크포인트 (.pt) 또는 run_dir")
    ap.add_argument("--run-dir", default=None, help="결과 폴더 (설정의 pacman_train.run_dir 대신)")
    ap.add_argument("--device", default=None, help="auto | cpu | cuda (설정 대신)")
    ap.add_argument("--updates", type=int, default=None, help="총 갱신 수 (설정의 total_updates 대신)")
    args = ap.parse_args(argv)

    from flypac.pacman.trainer import Trainer

    cfg = load_config(args.config)
    tr = Trainer(cfg, run_dir=args.run_dir, device=args.device)
    if args.resume:
        tr.load(args.resume)
    elif (tr.ckpt_dir / "latest.pt").exists():
        print(f"이미 체크포인트가 있습니다: {tr.ckpt_dir / 'latest.pt'}\n"
              f"이어서 하려면 --resume {tr.run_dir} 를, 새로 하려면 다른 --run-dir 를 주세요.", file=sys.stderr)
        return 2
    total = args.updates if args.updates is not None else int(cfg.pacman_train.training.total_updates)
    tr._say(f"시작: 설정 {args.config}, 결과 {tr.run_dir}, 장치 {tr.device}, 게임 {tr.N}개 × {tr.T}스텝, "
            f"갱신 {tr.update} → {total}")
    tr.train(total_updates=total)
    tr._say("끝")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
