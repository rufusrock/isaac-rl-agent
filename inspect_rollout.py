from __future__ import annotations

import argparse

from binding_rl_agent.inspection import find_latest_run, inspect_rollout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a recorded Isaac rollout and generate visual previews."
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Specific rollout directory to inspect. Defaults to the latest run.",
    )
    parser.add_argument(
        "--rollouts-dir",
        default="rollouts",
        help="Parent directory containing rollout runs.",
    )
    parser.add_argument(
        "--max-preview-frames",
        type=int,
        default=16,
        help="Number of frames to include in the contact sheet.",
    )
    parser.add_argument(
        "--gif-frames",
        type=int,
        default=24,
        help="Number of frames to include in the animated preview.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir or find_latest_run(args.rollouts_dir)
    summary = inspect_rollout(
        run_dir=run_dir,
        max_preview_frames=args.max_preview_frames,
        gif_frames=args.gif_frames,
    )

    print(f"run_dir: {summary.run_dir}")
    print(f"num_observations: {summary.num_observations}")
    print(f"num_actions: {summary.num_actions}")
    print(f"observation_shape: {summary.observation_shape}")
    print(f"action_counts: {summary.action_counts}")
    for name, path in summary.outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
