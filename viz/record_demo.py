"""Record game 1 from one seed with the real engine and the B1 controller."""

from __future__ import annotations

import argparse
from pathlib import Path

from flypac.core.config import Config, load_config
from flypac.core.runner import build_runner
from flypac.recording.recorder import Recorder, make_run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parents[1] / "configs" / "m0_smoke.yaml")
    parser.add_argument("--seed", type=int, help="Defaults to the first seed in the base config")
    parser.add_argument("--out-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()

    values = load_config(args.config).to_dict()
    seed = values["seeds"][0] if args.seed is None else args.seed
    values.update(experiment="viz_b1_demo", condition="b1", seeds=[seed], n_games=1,
                  pacman={"kind": "rule_based"}, ghosts={"kind": "original"},
                  replay={"keep_games": [1], "brain": False})
    values["engine"]["kind"] = "real"
    cfg = Config(values)
    recorder = Recorder(make_run_dir(cfg, args.out_dir), cfg)
    summary = build_runner(cfg, recorder).run()
    print(f"Game 1, seed {seed}, {summary['ticks']} ticks, {summary['games'][0]['winner']}")
    print(recorder.replay_files[0].resolve())


if __name__ == "__main__":
    main()
