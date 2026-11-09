"""Render synchronized game, raw, or brain views from one replay file."""

from __future__ import annotations

import argparse
from pathlib import Path

from flypac.core.config import load_config
from flypac.replay.schema import N_FLIES

from .replay import Replay


def main() -> None:
    defaults = load_config(Path(__file__).with_name("settings.yaml"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--view", choices=("game", "raw", "both", "brain", "side-by-side", "brain-both"), default="both")
    parser.add_argument("--out-dir", type=Path, help="Defaults to the replay's directory")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds")
    parser.add_argument("--duration", type=float, help="Maximum duration in seconds")
    parser.add_argument("--tick-rate", type=float, default=defaults.tick_rate, help="Original time.v_max_tiles_per_s")
    parser.add_argument("--c0", type=float, default=defaults.odor_c0, help="Original odor.c0 for the raw view")
    parser.add_argument("--no-observations", action="store_true", help="Hide sensory and decision overlays")
    parser.add_argument("--fly", type=int, choices=range(1, N_FLIES + 1), default=1, help="Fly to show in the brain views")
    parser.add_argument("--brain-config", type=Path, help="Original brain configuration for anatomical mapping")
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
        view_groups = {"both": ("game", "raw"), "brain-both": ("brain", "side-by-side")}
        views = view_groups.get(args.view, (args.view,))
        if args.view in ("brain", "side-by-side", "brain-both"):
            from .anatomy import DEFAULT_BRAIN_CONFIG, Anatomy
            from .brain import BrainPairRenderer, BrainRenderer

            anatomy = Anatomy(brain_config=args.brain_config or DEFAULT_BRAIN_CONFIG)
            brain = BrainRenderer(replay, anatomy, fly=args.fly, tick_rate=args.tick_rate,
                                  decay_s=defaults.brain_kc_decay_s)
        out_dir = args.out_dir or args.replay.parent
        for view in views:
            if view == "game":
                renderer = GameRenderer(replay)
            elif view == "raw":
                renderer = RawRenderer(replay, c0=args.c0, observations=not args.no_observations)
            else:
                renderer = brain if view == "brain" else BrainPairRenderer(brain)
            path = out_dir / f"{args.replay.stem}_{view}.mp4"
            write_mp4(renderer, ticks, path)
            print(path.resolve())
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
