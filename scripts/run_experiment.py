"""실험 실행: uv run python scripts/run_experiment.py configs/m0_smoke.yaml"""

from __future__ import annotations

import sys
from pathlib import Path

from flypac.core.config import load_config
from flypac.core.runner import build_runner
from flypac.recording.recorder import Recorder, make_run_dir


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("사용법: run_experiment.py <설정.yaml>", file=sys.stderr)
        return 2
    cfg = load_config(Path(argv[1]))
    run_dir = make_run_dir(cfg, Path("runs"))
    summary = build_runner(cfg, Recorder(run_dir, cfg)).run()
    print(f"결과 폴더: {run_dir}")
    wins = [g["winner"] for g in summary["games"]]
    print(f"끝: {summary['ticks']}틱, {len(wins)}판 (초파리 {wins.count('FLIES')}, 팩맨 {wins.count('PACMAN')}, "
          f"중단 {wins.count('CENSORED')})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
