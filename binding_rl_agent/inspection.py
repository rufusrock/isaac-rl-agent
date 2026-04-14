from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from binding_rl_agent.dataset import (
    BOMB_NAMES,
    MOVEMENT_NAMES,
    SHOOTING_NAMES,
    decode_action_heads,
)


@dataclass(frozen=True)
class InspectionSummary:
    run_dir: Path
    num_observations: int
    num_actions: int
    observation_shape: tuple[int, ...]
    action_counts: dict[str, int]
    outputs: dict[str, Path]


def inspect_rollout(
    run_dir: str | Path,
    output_dir: str | Path | None = None,
    max_preview_frames: int = 16,
    gif_frames: int = 24,
    gif_scale: int = 3,
) -> InspectionSummary:
    run_path = Path(run_dir)
    data_path = run_path / "rollout_data.npz"
    metadata_path = run_path / "metadata.json"

    data = np.load(data_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    observations = data["raw_frames"]
    timestamps = data["timestamps"]
    schema_version = int(metadata.get("action_schema_version", 1))
    movement_actions, shooting_actions, bomb_actions = decode_action_heads(
        data=data,
        schema_version=schema_version,
    )

    inspection_dir = Path(output_dir) if output_dir else run_path / "inspection"
    inspection_dir.mkdir(parents=True, exist_ok=True)

    preview_count = min(max_preview_frames, len(movement_actions))
    gif_count = min(gif_frames, len(movement_actions))

    contact_sheet = _build_contact_sheet(
        observations=observations,
        movement_actions=movement_actions,
        shooting_actions=shooting_actions,
        bomb_actions=bomb_actions,
        timestamps=timestamps,
        preview_count=preview_count,
        columns=4,
    )
    gif = _build_gif(
        observations=observations,
        movement_actions=movement_actions,
        shooting_actions=shooting_actions,
        bomb_actions=bomb_actions,
        timestamps=timestamps,
        frame_count=gif_count,
        scale=gif_scale,
    )

    contact_sheet_path = inspection_dir / "contact_sheet.png"
    gif_path = inspection_dir / "preview.gif"
    summary_path = inspection_dir / "summary.json"

    contact_sheet.save(contact_sheet_path)
    gif[0].save(
        gif_path,
        save_all=True,
        append_images=gif[1:],
        duration=_estimate_frame_duration_ms(timestamps),
        loop=0,
    )

    action_counter = Counter(
        _format_action_label(movement, shooting, bomb)
        for movement, shooting, bomb in zip(
            movement_actions,
            shooting_actions,
            bomb_actions,
            strict=True,
        )
    )
    summary_payload = {
        "run_dir": str(run_path),
        "observation_shape": list(observations.shape[1:]),
        "num_observations": int(observations.shape[0]),
        "num_actions": int(movement_actions.shape[0]),
        "preview_frames": preview_count,
        "gif_frames": gif_count,
        "action_counts": dict(sorted(action_counter.items())),
        "outputs": {
            "contact_sheet": str(contact_sheet_path),
            "preview_gif": str(gif_path),
        },
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    return InspectionSummary(
        run_dir=run_path,
        num_observations=int(observations.shape[0]),
        num_actions=int(movement_actions.shape[0]),
        observation_shape=tuple(int(v) for v in observations.shape[1:]),
        action_counts=dict(sorted(action_counter.items())),
        outputs={
            "contact_sheet": contact_sheet_path,
            "preview_gif": gif_path,
            "summary": summary_path,
        },
    )


def find_latest_run(rollouts_dir: str | Path = "rollouts") -> Path:
    root = Path(rollouts_dir)
    candidates = sorted(
        (path for path in root.iterdir() if path.is_dir() and path.name.startswith("run_")),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No rollout runs found under {root}")
    return candidates[0]


def _build_contact_sheet(
    observations: np.ndarray,
    movement_actions: np.ndarray,
    shooting_actions: np.ndarray,
    bomb_actions: np.ndarray,
    timestamps: np.ndarray,
    preview_count: int,
    columns: int,
) -> Image.Image:
    sample_tiles = [
        _annotated_frame(
            observation=observations[idx + 1],
            movement_action=int(movement_actions[idx]),
            shooting_action=int(shooting_actions[idx]),
            bomb_action=int(bomb_actions[idx]),
            timestamp=float(timestamps[idx + 1]),
            index=idx,
        )
        for idx in range(preview_count)
    ]
    if not sample_tiles:
        raise ValueError("Expected at least one action frame in the rollout.")

    tile_width, tile_height = sample_tiles[0].size
    rows = (len(sample_tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), color=(12, 12, 12))

    for idx, tile in enumerate(sample_tiles):
        x = (idx % columns) * tile_width
        y = (idx // columns) * tile_height
        sheet.paste(tile, (x, y))
    return sheet


def _build_gif(
    observations: np.ndarray,
    movement_actions: np.ndarray,
    shooting_actions: np.ndarray,
    bomb_actions: np.ndarray,
    timestamps: np.ndarray,
    frame_count: int,
    scale: int,
) -> list[Image.Image]:
    frames: list[Image.Image] = []
    for idx in range(frame_count):
        frame = _annotated_frame(
            observation=observations[idx + 1],
            movement_action=int(movement_actions[idx]),
            shooting_action=int(shooting_actions[idx]),
            bomb_action=int(bomb_actions[idx]),
            timestamp=float(timestamps[idx + 1]),
            index=idx,
        )
        if scale != 1:
            frame = frame.resize(
                (frame.width * scale, frame.height * scale),
                resample=Image.Resampling.NEAREST,
            )
        frames.append(frame)
    return frames


def _annotated_frame(
    observation: np.ndarray,
    movement_action: int,
    shooting_action: int,
    bomb_action: int,
    timestamp: float,
    index: int,
) -> Image.Image:
    latest_frame = observation[-1]
    if latest_frame.ndim == 2:
        image = Image.fromarray(latest_frame, mode="L").convert("RGB")
    else:
        image = Image.fromarray(latest_frame).convert("RGB")

    overlay_height = 38
    annotated = Image.new("RGB", (image.width, image.height + overlay_height), color=(0, 0, 0))
    annotated.paste(image, (0, 0))

    draw = ImageDraw.Draw(annotated)
    label = f"#{index:03d} t={timestamp:.2f}"
    action_name = _format_action_label(movement_action, shooting_action, bomb_action)
    draw.text((4, image.height + 2), label, fill=(255, 255, 255))
    draw.text((4, image.height + 14), action_name, fill=(145, 220, 255))
    return annotated


def _format_action_label(
    movement_action: int,
    shooting_action: int,
    bomb_action: int,
) -> str:
    movement_name = MOVEMENT_NAMES[movement_action]
    shooting_name = SHOOTING_NAMES[shooting_action]
    bomb_name = BOMB_NAMES[bomb_action]
    return f"move={movement_name} shoot={shooting_name} bomb={bomb_name}"


def _estimate_frame_duration_ms(timestamps: np.ndarray) -> int:
    if len(timestamps) < 3:
        return 100
    deltas = np.diff(timestamps)
    average_seconds = float(np.mean(deltas))
    return max(40, int(average_seconds * 1000))
