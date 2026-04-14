from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from binding_rl_agent.dataset import BOMB_NAMES, MOVEMENT_NAMES, SHOOTING_NAMES
from binding_rl_agent.env import IsaacAction


def build_action_counts(actions: list[IsaacAction]) -> dict[str, dict[str, int]]:
    movement = Counter(MOVEMENT_NAMES[action.movement] for action in actions)
    shooting = Counter(SHOOTING_NAMES[action.shooting] for action in actions)
    bomb = Counter(BOMB_NAMES[action.bomb] for action in actions)
    combos = Counter(_format_action_label(action) for action in actions)
    return {
        "movement": dict(sorted(movement.items())),
        "shooting": dict(sorted(shooting.items())),
        "bomb": dict(sorted(bomb.items())),
        "combined": dict(sorted(combos.items())),
    }


def build_rollout_diagnostics(
    rewards: np.ndarray,
    dones: np.ndarray,
    actions: list[IsaacAction],
    infos: list[dict[str, object]],
) -> dict[str, object]:
    action_counts = build_action_counts(actions)

    def _sum_int(name: str) -> int:
        return int(sum(int(info.get(name, 0)) for info in infos))

    def _sum_float(name: str) -> float:
        return float(sum(float(info.get(name, 0.0)) for info in infos))

    stagnant_steps = np.asarray([int(info.get("stagnant_steps", 0)) for info in infos], dtype=np.int32)
    room_steps = np.asarray([int(info.get("room_steps", 0)) for info in infos], dtype=np.int32)
    recent_progress_steps = np.asarray(
        [int(info.get("recent_progress_steps", 0)) for info in infos],
        dtype=np.int32,
    )
    exploratory_steps = np.asarray(
        [bool(info.get("exploratory_mode", False)) for info in infos],
        dtype=np.bool_,
    )
    episode_summaries = _build_episode_summaries(rewards, dones, infos)

    return {
        "action_counts": action_counts,
        "events": {
            "rooms_explored_gained": _sum_int("rooms_explored_gained"),
            "room_transitions": _sum_int("room_transition"),
            "room_clears": _sum_int("room_clear_candidate"),
            "kills_gained": _sum_int("kills_gained"),
            "damage_taken": _sum_float("damage_taken"),
            "coins_gained": _sum_int("coins_gained"),
            "keys_gained": _sum_int("keys_gained"),
            "collectibles_gained": _sum_int("collectibles_gained"),
            "deaths": _sum_int("death_candidate"),
            "timeouts": _sum_int("timeout_done"),
            "stagnation_penalties": int(
                sum(bool(info.get("stagnation_penalty_applied", 0.0)) for info in infos)
            ),
        },
        "behavior": {
            "mean_stagnant_steps": float(stagnant_steps.mean()) if len(stagnant_steps) else 0.0,
            "max_stagnant_steps": int(stagnant_steps.max()) if len(stagnant_steps) else 0,
            "mean_room_steps": float(room_steps.mean()) if len(room_steps) else 0.0,
            "max_room_steps": int(room_steps.max()) if len(room_steps) else 0,
            "mean_recent_progress_steps": (
                float(recent_progress_steps.mean()) if len(recent_progress_steps) else 0.0
            ),
            "exploration_fraction": float(exploratory_steps.mean()) if len(exploratory_steps) else 0.0,
        },
        "episodes": episode_summaries,
    }


def save_rollout_diagnostics(
    diagnostics_dir: Path,
    update_idx: int,
    rollout: dict[str, object],
    summary: dict[str, object],
    sample_frames: int = 16,
) -> None:
    update_dir = diagnostics_dir / f"update_{update_idx:04d}"
    update_dir.mkdir(parents=True, exist_ok=True)

    observations = np.asarray(rollout["observations"], dtype=np.uint8)
    rewards = np.asarray(rollout["rewards"], dtype=np.float32)
    dones = np.asarray(rollout["dones"], dtype=np.float32)
    actions = list(rollout["actions"])
    infos = list(rollout["infos"])

    sample_indices = _sample_indices(len(actions), sample_frames)
    sampled_observations = observations[sample_indices]
    sampled_rewards = rewards[sample_indices]
    sampled_dones = dones[sample_indices]
    sampled_movement = np.asarray([actions[idx].movement for idx in sample_indices], dtype=np.int64)
    sampled_shooting = np.asarray([actions[idx].shooting for idx in sample_indices], dtype=np.int64)
    sampled_bomb = np.asarray([actions[idx].bomb for idx in sample_indices], dtype=np.int64)
    sampled_stagnant_steps = np.asarray(
        [int(infos[idx].get("stagnant_steps", 0)) for idx in sample_indices],
        dtype=np.int32,
    )
    sampled_room_steps = np.asarray(
        [int(infos[idx].get("room_steps", 0)) for idx in sample_indices],
        dtype=np.int32,
    )
    sampled_exploratory = np.asarray(
        [bool(infos[idx].get("exploratory_mode", False)) for idx in sample_indices],
        dtype=np.bool_,
    )

    np.savez_compressed(
        update_dir / "rollout_sample.npz",
        sample_indices=sample_indices,
        observations=sampled_observations,
        rewards=sampled_rewards,
        dones=sampled_dones,
        movement_actions=sampled_movement,
        shooting_actions=sampled_shooting,
        bomb_actions=sampled_bomb,
        stagnant_steps=sampled_stagnant_steps,
        room_steps=sampled_room_steps,
        exploratory_mode=sampled_exploratory,
    )

    contact_sheet = _build_contact_sheet(
        observations=sampled_observations,
        rewards=sampled_rewards,
        dones=sampled_dones,
        movement_actions=sampled_movement,
        shooting_actions=sampled_shooting,
        bomb_actions=sampled_bomb,
        stagnant_steps=sampled_stagnant_steps,
        room_steps=sampled_room_steps,
        exploratory_mode=sampled_exploratory,
        sample_indices=sample_indices,
    )
    contact_sheet.save(update_dir / "contact_sheet.png")

    payload = {
        "update": int(update_idx),
        "summary": _sanitize_for_json(summary),
    }
    (update_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def summarize_run(metrics_path: Path) -> dict[str, object]:
    summaries = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not summaries:
        raise ValueError(f"No summaries found in {metrics_path}")

    latest = summaries[-1]
    best_reward = max(summaries, key=lambda entry: float(entry.get("sum_reward", 0.0)))
    best_explore = max(
        summaries,
        key=lambda entry: float(entry.get("exploration_fraction", 0.0)),
    )
    return {
        "updates": len(summaries),
        "latest": latest,
        "best_sum_reward_update": int(best_reward["update"]),
        "best_sum_reward": float(best_reward.get("sum_reward", 0.0)),
        "best_exploration_update": int(best_explore["update"]),
        "best_exploration_fraction": float(best_explore.get("exploration_fraction", 0.0)),
    }


def _build_episode_summaries(
    rewards: np.ndarray,
    dones: np.ndarray,
    infos: list[dict[str, object]],
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    start_idx = 0
    for idx, done in enumerate(dones):
        if not done:
            continue
        episode_infos = infos[start_idx : idx + 1]
        episode_rewards = rewards[start_idx : idx + 1]
        summaries.append(
            {
                "length": int(idx - start_idx + 1),
                "reward_sum": float(np.sum(episode_rewards)),
                "rooms_explored_gained": int(
                    sum(int(info.get("rooms_explored_gained", 0)) for info in episode_infos)
                ),
                "kills_gained": int(sum(int(info.get("kills_gained", 0)) for info in episode_infos)),
                "damage_taken": float(
                    sum(float(info.get("damage_taken", 0.0)) for info in episode_infos)
                ),
                "room_clears": int(
                    sum(int(info.get("room_clear_candidate", 0)) for info in episode_infos)
                ),
                "ended_by_death": bool(episode_infos[-1].get("death_candidate", False)),
                "ended_by_timeout": bool(episode_infos[-1].get("timeout_done", False)),
            }
        )
        start_idx = idx + 1
    return summaries


def _sanitize_for_json(payload: dict[str, object]) -> dict[str, object]:
    return json.loads(json.dumps(payload, default=_json_default))


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _sample_indices(length: int, count: int) -> np.ndarray:
    if length <= 0:
        return np.asarray([], dtype=np.int64)
    if length <= count:
        return np.arange(length, dtype=np.int64)
    return np.linspace(0, length - 1, num=count, dtype=np.int64)


def _build_contact_sheet(
    observations: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    movement_actions: np.ndarray,
    shooting_actions: np.ndarray,
    bomb_actions: np.ndarray,
    stagnant_steps: np.ndarray,
    room_steps: np.ndarray,
    exploratory_mode: np.ndarray,
    sample_indices: np.ndarray,
) -> Image.Image:
    tiles = [
        _annotated_frame(
            observation=observations[idx],
            reward=float(rewards[idx]),
            done=bool(dones[idx]),
            movement_action=int(movement_actions[idx]),
            shooting_action=int(shooting_actions[idx]),
            bomb_action=int(bomb_actions[idx]),
            stagnant_step=int(stagnant_steps[idx]),
            room_step=int(room_steps[idx]),
            exploratory_mode=bool(exploratory_mode[idx]),
            rollout_index=int(sample_indices[idx]),
        )
        for idx in range(len(sample_indices))
    ]
    if not tiles:
        raise ValueError("Expected at least one sampled frame.")

    columns = 4
    tile_width, tile_height = tiles[0].size
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), color=(12, 12, 12))

    for idx, tile in enumerate(tiles):
        x = (idx % columns) * tile_width
        y = (idx // columns) * tile_height
        sheet.paste(tile, (x, y))
    return sheet


def _annotated_frame(
    observation: np.ndarray,
    reward: float,
    done: bool,
    movement_action: int,
    shooting_action: int,
    bomb_action: int,
    stagnant_step: int,
    room_step: int,
    exploratory_mode: bool,
    rollout_index: int,
) -> Image.Image:
    latest_frame = observation[-1]
    if latest_frame.ndim == 2:
        image = Image.fromarray(latest_frame, mode="L").convert("RGB")
    else:
        image = Image.fromarray(latest_frame).convert("RGB")

    overlay_height = 50
    annotated = Image.new("RGB", (image.width, image.height + overlay_height), color=(0, 0, 0))
    annotated.paste(image, (0, 0))

    draw = ImageDraw.Draw(annotated)
    draw.text((4, image.height + 2), f"step={rollout_index:03d} reward={reward:+.3f}", fill=(255, 255, 255))
    draw.text((4, image.height + 14), _format_action_names(movement_action, shooting_action, bomb_action), fill=(145, 220, 255))
    flags = f"stagnant={stagnant_step} room={room_step} {'EXP' if exploratory_mode else 'POL'}"
    if done:
        flags += " DONE"
    draw.text((4, image.height + 26), flags, fill=(255, 220, 120))
    return annotated


def _format_action_names(movement_action: int, shooting_action: int, bomb_action: int) -> str:
    return (
        f"move={MOVEMENT_NAMES[movement_action]} "
        f"shoot={SHOOTING_NAMES[shooting_action]} "
        f"bomb={BOMB_NAMES[bomb_action]}"
    )


def _format_action_label(action: IsaacAction) -> str:
    return _format_action_names(action.movement, action.shooting, action.bomb)
