from __future__ import annotations

import ctypes
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from binding_rl_agent.game_state import IsaacUDPGameStateReceiver
from binding_rl_agent.preprocessing import resize_frame, to_grayscale, to_multichannel, to_rgb
from binding_rl_agent.room_graph import RoomGraph
from binding_rl_agent.window_capture import IsaacWindowCapture


VK_CODES: dict[str, int] = {
    "w": 0x57,
    "a": 0x41,
    "s": 0x53,
    "d": 0x44,
    "e": 0x45,
    "up": 0x26,
    "down": 0x28,
    "left": 0x25,
    "right": 0x27,
}


@dataclass(frozen=True)
class RolloutConfig:
    width: int = 256
    height: int = 256
    grayscale: bool = False
    multichannel: bool = False  # equalized gray + bilateral (2 channels/frame)
    color: bool = True          # raw RGB (3 channels/frame); overrides grayscale/multichannel
    fps: int = 20
    num_steps: int = 1000
    warmup_seconds: float = 3.0
    title_substring: str | None = None
    record_nav_hints: bool = True
    telemetry_port: int = 8123


def record_keyboard_rollout(
    output_dir: str | Path,
    config: RolloutConfig | None = None,
) -> Path:
    rollout_config = config or RolloutConfig()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    run_dir = output_path / _build_run_name()
    run_dir.mkdir(parents=True, exist_ok=False)

    capture = IsaacWindowCapture(title_substring=rollout_config.title_substring)

    telemetry: IsaacUDPGameStateReceiver | None = None
    if rollout_config.record_nav_hints:
        try:
            telemetry = IsaacUDPGameStateReceiver(port=rollout_config.telemetry_port)
            print(f"Nav hint telemetry listening on UDP port {rollout_config.telemetry_port}")
        except OSError as exc:
            print(f"[WARN] Could not bind telemetry port {rollout_config.telemetry_port}: {exc}. Nav hints will not be recorded.")
            telemetry = None

    # Capture initial frame (observation before any action)
    raw_frames: list[np.ndarray] = [_preprocess(capture.grab(), rollout_config)]
    movement_actions: list[int] = []
    shooting_actions: list[int] = []
    bomb_actions: list[int] = []
    nav_hints: list[int] = []
    timestamps: list[float] = [time.time()]

    print(f"Preparing to record from '{capture.window.title}'. Focus the Isaac window now.")
    for seconds_left in range(int(rollout_config.warmup_seconds), 0, -1):
        print(f"Starting in {seconds_left}...")
        time.sleep(1.0)

    frame_interval = 1.0 / max(rollout_config.fps, 1)
    next_tick = time.monotonic()

    for step_idx in range(rollout_config.num_steps):
        movement_action, shooting_action, bomb_action = current_action_heads_from_keyboard()

        nav_hint = 0
        if telemetry is not None:
            game_state = telemetry.get_latest()
            if game_state is not None and game_state.floor_rooms:
                graph = RoomGraph(game_state.floor_rooms)
                nav_hint = int(graph.nav_hint(game_state.room_index))

        frame = _preprocess(capture.grab(), rollout_config)

        movement_actions.append(movement_action)
        shooting_actions.append(shooting_action)
        bomb_actions.append(bomb_action)
        nav_hints.append(nav_hint)
        raw_frames.append(frame)
        timestamps.append(time.time())

        print(
            f"step={step_idx:03d} move={movement_action} "
            f"shoot={shooting_action} bomb={bomb_action} nav={nav_hint}"
        )

        next_tick += frame_interval
        sleep_for = next_tick - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)

    frames_array = np.stack(raw_frames, axis=0)           # (N+1, C, H, W) or (N+1, H, W)
    movement_array = np.asarray(movement_actions, dtype=np.int64)
    shooting_array = np.asarray(shooting_actions, dtype=np.int64)
    bomb_array = np.asarray(bomb_actions, dtype=np.int64)
    timestamps_array = np.asarray(timestamps, dtype=np.float64)

    save_kwargs: dict = dict(
        raw_frames=frames_array,
        movement_actions=movement_array,
        shooting_actions=shooting_array,
        bomb_actions=bomb_array,
        timestamps=timestamps_array,
    )
    if telemetry is not None:
        save_kwargs["nav_hints"] = np.asarray(nav_hints, dtype=np.int64)

    np.savez_compressed(run_dir / "rollout_data.npz", **save_kwargs)

    metadata = {
        "run_dir": str(run_dir),
        "window_title": capture.window.title,
        "capture_region": asdict(capture.region),
        "config": asdict(rollout_config),
        "frame_shape": list(frames_array.shape[1:]),
        "num_actions_logged": len(movement_actions),
        "movement_space": {"0": "idle", "1": "up", "2": "down", "3": "left", "4": "right"},
        "shooting_space": {"0": "idle", "1": "up", "2": "down", "3": "left", "4": "right"},
        "bomb_space": {"0": "no_bomb", "1": "bomb"},
        "action_schema_version": 3,
        "nav_hints_recorded": telemetry is not None,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nSaved {len(movement_actions)} steps → {run_dir}")
    return run_dir


def _preprocess(frame_bgr: np.ndarray, config: RolloutConfig) -> np.ndarray:
    resized = resize_frame(frame_bgr, config.width, config.height)
    if config.color:
        return to_rgb(resized)          # (3, H, W)
    if config.multichannel:
        return to_multichannel(resized) # (2, H, W)
    return to_grayscale(resized)        # (H, W)


def current_action_heads_from_keyboard() -> tuple[int, int, int]:
    pressed = {key for key, vk_code in VK_CODES.items() if _is_pressed(vk_code)}

    movement = 0
    if "w" in pressed:
        movement = 1
    elif "s" in pressed:
        movement = 2
    elif "a" in pressed:
        movement = 3
    elif "d" in pressed:
        movement = 4

    shooting = 0
    if "up" in pressed:
        shooting = 1
    elif "down" in pressed:
        shooting = 2
    elif "left" in pressed:
        shooting = 3
    elif "right" in pressed:
        shooting = 4

    bomb = 1 if "e" in pressed else 0
    return movement, shooting, bomb


def _is_pressed(virtual_key: int) -> bool:
    return bool(ctypes.windll.user32.GetAsyncKeyState(virtual_key) & 0x8000)


def _build_run_name() -> str:
    return time.strftime("run_%Y%m%d_%H%M%S")
