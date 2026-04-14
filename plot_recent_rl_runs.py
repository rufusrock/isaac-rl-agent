from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class RunSeries:
    name: str
    updates: np.ndarray
    exploration: np.ndarray
    kills: np.ndarray
    rooms: np.ndarray
    reward: np.ndarray
    stagnant: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot exploration and kill-rate trends for the most recent RL runs."
    )
    parser.add_argument(
        "--runs-root",
        default="rl_runs",
        help="Directory containing saved RL training runs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="How many recent runs to include.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/recent_rl_run_trends.png",
        help="Where to save the plot.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=25,
        help="Rolling window in updates for smoothing kills/update.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = load_recent_runs(args.runs_root, args.limit)
    if not runs:
        raise FileNotFoundError(f"No RL runs found under {args.runs_root}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = build_plot(runs, rolling_window=args.rolling_window)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved plot to: {output_path}")

    print("\nRecent run summary:")
    for run in runs:
        last_window = min(100, len(run.updates))
        kills_last = float(run.kills[-last_window:].mean())
        rooms_last = float(run.rooms[-last_window:].mean())
        explore_last = float(run.exploration[-last_window:].mean())
        reward_last = float(run.reward[-last_window:].mean())
        stagnant_last = float(run.stagnant[-last_window:].mean())
        print(
            f"{run.name}: updates={len(run.updates)} "
            f"last{last_window}_avg_reward={reward_last:.3f} "
            f"last{last_window}_avg_explore={explore_last:.3f} "
            f"last{last_window}_avg_rooms_per_update={rooms_last:.3f} "
            f"last{last_window}_avg_kills_per_update={kills_last:.3f} "
            f"last{last_window}_avg_stagnant={stagnant_last:.1f}"
        )


def load_recent_runs(root: str | Path, limit: int) -> list[RunSeries]:
    root_path = Path(root)
    run_dirs = sorted(
        (
            path
            for path in root_path.iterdir()
            if path.is_dir() and (path.name.startswith("ppo_") or path.name.startswith("a2c_"))
        ),
        key=lambda path: path.name,
        reverse=True,
    )[:limit]
    return [load_run(path) for path in reversed(run_dirs)]


def load_run(run_dir: Path) -> RunSeries:
    metrics_path = run_dir / "metrics.jsonl"
    rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"No metrics found in {metrics_path}")

    return RunSeries(
        name=run_dir.name,
        updates=np.asarray([int(row["update"]) for row in rows], dtype=np.int32),
        exploration=np.asarray(
            [float(row.get("exploration_fraction", 0.0)) for row in rows],
            dtype=np.float32,
        ),
        kills=np.asarray(
            [float(row.get("kills_gained", 0.0)) for row in rows],
            dtype=np.float32,
        ),
        rooms=np.asarray(
            [float(row.get("rooms_explored_gained", 0.0)) for row in rows],
            dtype=np.float32,
        ),
        reward=np.asarray(
            [float(row.get("sum_reward", 0.0)) for row in rows],
            dtype=np.float32,
        ),
        stagnant=np.asarray(
            [float(row.get("mean_stagnant_steps", 0.0)) for row in rows],
            dtype=np.float32,
        ),
    )


def build_plot(runs: list[RunSeries], rolling_window: int) -> plt.Figure:
    progress = np.linspace(0.0, 1.0, num=101)
    cmap = plt.get_cmap("tab10")

    fig, axes = plt.subplots(3, 2, figsize=(16, 14), sharex=True, constrained_layout=True)
    ax_reward, ax_explore = axes[0]
    ax_rooms, ax_kills = axes[1]
    ax_stagnant, ax_legend = axes[2]
    fig.patch.set_facecolor("#faf7f0")
    ax_legend.axis("off")

    for index, run in enumerate(runs):
        color = cmap(index % 10)
        run_progress = np.linspace(0.0, 1.0, num=len(run.updates))
        smooth_rooms = rolling_mean(run.rooms, rolling_window)
        smooth_kills = rolling_mean(run.kills, rolling_window)
        smooth_reward = rolling_mean(run.reward, rolling_window)
        smooth_stagnant = rolling_mean(run.stagnant, rolling_window)

        ax_reward.plot(
            progress * 100.0,
            np.interp(progress, run_progress, smooth_reward),
            color=color,
            linewidth=2.0,
            alpha=0.9,
            label=run.name,
        )

        ax_explore.plot(
            progress * 100.0,
            np.interp(progress, run_progress, run.exploration),
            color=color,
            linewidth=2.0,
            alpha=0.9,
            label=run.name,
        )
        ax_rooms.plot(
            progress * 100.0,
            np.interp(progress, run_progress, smooth_rooms),
            color=color,
            linewidth=2.0,
            alpha=0.9,
            label=run.name,
        )
        ax_kills.plot(
            progress * 100.0,
            np.interp(progress, run_progress, smooth_kills),
            color=color,
            linewidth=2.0,
            alpha=0.9,
            label=run.name,
        )
        ax_stagnant.plot(
            progress * 100.0,
            np.interp(progress, run_progress, smooth_stagnant),
            color=color,
            linewidth=2.0,
            alpha=0.9,
            label=run.name,
        )

    style_axis(ax_reward, "Reward per Update", f"Reward by Training Progress ({rolling_window}-update mean)")
    style_axis(ax_explore, "Exploration Fraction", "Exploration by Training Progress")
    style_axis(ax_rooms, "Rooms per Update", f"Room Progress by Training Progress ({rolling_window}-update mean)")
    style_axis(ax_kills, "Kills per Update", f"Kill Rate by Training Progress ({rolling_window}-update mean)")
    style_axis(ax_stagnant, "Mean Stagnant Steps", f"Stagnation by Training Progress ({rolling_window}-update mean)")
    ax_stagnant.set_xlabel("Training Progress (%)")
    ax_stagnant.set_xlabel("Training Progress (%)")
    ax_kills.set_xlabel("Training Progress (%)")
    ax_explore.set_ylim(-0.02, 1.02)
    ax_reward.axhline(0.0, color="#444444", linewidth=1.0, alpha=0.35)
    ax_rooms.set_ylim(bottom=-0.02)
    ax_kills.set_ylim(bottom=-0.05)
    ax_stagnant.set_ylim(bottom=-5.0)

    handles, labels = ax_reward.get_legend_handles_labels()
    ax_legend.legend(handles, labels, loc="center left", frameon=False)
    fig.suptitle(
        "Recent RL Runs: Reward, Exploration, Room Progress, Kill Rate, and Stagnation",
        fontsize=16,
        fontweight="bold",
    )
    return fig


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) <= 1:
        return values.copy()
    kernel = np.ones(window, dtype=np.float32)
    numer = np.convolve(values, kernel, mode="same")
    denom = np.convolve(np.ones_like(values), kernel, mode="same")
    return numer / np.maximum(denom, 1.0)


def style_axis(ax: plt.Axes, ylabel: str, title: str) -> None:
    ax.set_facecolor("#fffdf8")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_alpha(0.2)


if __name__ == "__main__":
    main()
