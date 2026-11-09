"""Render either or both synchronized views from one replay file."""

from __future__ import annotations

import argparse
from pathlib import Path

from flypac.core.config import load_config

from .replay import Replay


def main() -> None:
    defaults = load_config(Path(__file__).with_name("settings.yaml"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--view", choices=("game", "raw", "both"), default="both")
    parser.add_argument("--out-dir", type=Path, help="Defaults to the replay's directory")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds")
    parser.add_argument("--duration", type=float, help="Maximum duration in seconds")
    parser.add_argument("--tick-rate", type=float, default=defaults.tick_rate, help="Original time.v_max_tiles_per_s")
    parser.add_argument("--c0", type=float, default=defaults.odor_c0, help="Original odor.c0 for the raw view")
    parser.add_argument("--no-observations", action="store_true", help="Hide sensory and decision overlays")
    args = parser.parse_args()

    try:
        from .game import GameRenderer
        from .raw import RawRenderer
        from .video import frame_ticks, write_mp4
    except ModuleNotFoundError as exc:
        if exc.name in ("pygame", "imageio_ffmpeg"):
            parser.error("Rendering requires pygame and imageio-ffmpeg from the viz extra")
        raise

    try:
        replay = Replay(args.replay)
        ticks = frame_ticks(replay.ticks, args.tick_rate, args.start, args.duration)
        views = ("game", "raw") if args.view == "both" else (args.view,)
        out_dir = args.out_dir or args.replay.parent
        for view in views:
            if view == "game":
                renderer = GameRenderer(replay)
            else:
                renderer = RawRenderer(replay, c0=args.c0, observations=not args.no_observations)
            path = out_dir / f"{args.replay.stem}_{view}.mp4"
            write_mp4(renderer, ticks, path)
            print(path.resolve())
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
