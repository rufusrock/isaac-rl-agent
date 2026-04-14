from __future__ import annotations

import argparse

from binding_rl_agent.recording import RolloutConfig, record_keyboard_rollout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Record an Isaac rollout by logging live observations and "
            "your current keyboard action state. Frames are saved individually "
            "(no stacking) so preprocessing can be applied at training time."
        )
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional substring to match the game window title.",
    )
    parser.add_argument(
        "--output-dir",
        default="rollouts",
        help="Directory where rollout runs will be saved.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=256,
        help="Capture width in pixels.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=256,
        help="Capture height in pixels.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Capture rate while logging the rollout.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Number of frames to record.",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=3.0,
        help="Seconds to wait so you can refocus the Isaac window.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        default=False,
        help="Record grayscale instead of RGB.",
    )
    parser.add_argument(
        "--multichannel",
        action="store_true",
        default=False,
        help="Record equalized+bilateral grayscale (2ch) instead of RGB.",
    )
    parser.add_argument(
        "--no-nav-hints",
        action="store_true",
        default=False,
        help="Disable nav hint recording (skip UDP telemetry).",
    )
    parser.add_argument(
        "--telemetry-port",
        type=int,
        default=8123,
        help="UDP port for Isaac telemetry (nav hints).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    color = not args.no_color and not args.multichannel
    run_dir = record_keyboard_rollout(
        output_dir=args.output_dir,
        config=RolloutConfig(
            width=args.width,
            height=args.height,
            color=color,
            multichannel=args.multichannel,
            grayscale=not color and not args.multichannel,
            fps=args.fps,
            num_steps=args.steps,
            warmup_seconds=args.warmup,
            title_substring=args.title,
            record_nav_hints=not args.no_nav_hints,
            telemetry_port=args.telemetry_port,
        ),
    )
    print(f"Saved rollout to: {run_dir}")


if __name__ == "__main__":
    main()
