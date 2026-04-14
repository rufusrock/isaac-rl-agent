from __future__ import annotations

import argparse
import json
from pathlib import Path

from binding_rl_agent.rl_diagnostics import summarize_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a saved RL run.")
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Path to an RL run directory. Defaults to the latest run under rl_runs.",
    )
    parser.add_argument(
        "--output-json",
        action="store_true",
        help="Print the full summary as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else find_latest_run_dir()
    summary = summarize_run(run_dir / "metrics.jsonl")
    latest = summary["latest"]

    if args.output_json:
        print(json.dumps(summary, indent=2))
        return

    print(f"Run: {run_dir}")
    print(f"Updates: {summary['updates']}")
    print(
        f"Latest update #{latest['update']}: reward_sum={float(latest.get('sum_reward', 0.0)):.3f} "
        f"rooms+={int(latest.get('rooms_explored_gained', 0))} "
        f"kills+={int(latest.get('kills_gained', 0))} "
        f"deaths={int(latest.get('deaths', 0))} "
        f"timeouts={int(latest.get('timeouts', 0))}"
    )
    print(
        f"Behavior: exploration_fraction={float(latest.get('exploration_fraction', 0.0)):.3f} "
        f"max_stagnant_steps={int(latest.get('max_stagnant_steps', 0))} "
        f"max_room_steps={int(latest.get('max_room_steps', 0))}"
    )
    print(
        f"Best reward update: #{summary['best_sum_reward_update']} "
        f"({summary['best_sum_reward']:.3f})"
    )
    print(
        f"Most exploratory update: #{summary['best_exploration_update']} "
        f"({summary['best_exploration_fraction']:.3f})"
    )
    best_checkpoint = run_dir / "best_actor_critic.pt"
    if best_checkpoint.exists():
        print(f"Best checkpoint: {best_checkpoint}")
    print(f"Diagnostics: {run_dir / 'diagnostics'}")


def find_latest_run_dir(root: str | Path = "rl_runs") -> Path:
    root_path = Path(root)
    candidates = sorted(
        (
            path
            for path in root_path.iterdir()
            if path.is_dir() and (path.name.startswith("a2c_") or path.name.startswith("ppo_"))
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No RL runs found under {root_path}")
    return candidates[0]


if __name__ == "__main__":
    main()
