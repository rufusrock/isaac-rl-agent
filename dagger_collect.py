"""DAgger collection: model plays the game, human corrects mistakes.

Controls:
  F7  -- toggle between MODEL and HUMAN control mode
  F8  -- arm/disarm (start/pause action execution & recording)
  F9  -- emergency stop
  F10 -- stop recording and save
  q   -- quit preview window

In MODEL mode (default):
  The model's actions are executed.  Press WASD / arrow keys to override
  the model on that head (movement / shooting).  Your input takes priority
  whenever you press a key that the model isn't already simulating.

In HUMAN mode (toggle with F7):
  Only your physical key presses control the game.  The model's prediction
  is still shown in the overlay but not executed.  Use this when the model
  is hopelessly stuck and you need to manually navigate.

Saved data uses the same rollout format as ``record_rollout.py`` so it can
be mixed directly into training data.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from binding_rl_agent.env import (
    BOMB_KEY_MAP,
    IsaacAction,
    IsaacFrameEnv,
    MOVEMENT_KEY_MAP,
    ObservationConfig,
    SHOOTING_KEY_MAP,
)
from binding_rl_agent.game_state import IsaacUDPGameStateReceiver
from binding_rl_agent.inference import (
    find_latest_model,
    frame_size_from_checkpoint,
    load_policy_checkpoint,
    nav_hint_from_room_graph,
    obs_config_from_checkpoint,
    predict_policy,
    prediction_to_action,
)
from binding_rl_agent.input_controller import (
    is_function_key_pressed,
    release_all_agent_keys,
    sync_pressed_keys,
    tap_key,
)
from binding_rl_agent.preprocessing import resize_frame, to_rgb
from binding_rl_agent.room_graph import NavHint, RoomGraph

# ---------------------------------------------------------------------------
# Virtual-key codes for detecting physical key presses
# ---------------------------------------------------------------------------
_MOVEMENT_VK: dict[str, int] = {"w": 0x57, "a": 0x41, "s": 0x53, "d": 0x44}
_SHOOTING_VK: dict[str, int] = {"up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27}
_BOMB_VK: int = 0x45  # 'E'

_KEY_TO_MOVEMENT: dict[str, int] = {"w": 1, "s": 2, "a": 3, "d": 4}
_KEY_TO_SHOOTING: dict[str, int] = {"up": 1, "down": 2, "left": 3, "right": 4}

# Source codes for action_sources array
SRC_MODEL = 0
SRC_HUMAN = 1

# Recording resolution (matches existing rollout format)
RECORD_WIDTH = 256
RECORD_HEIGHT = 256


def _is_pressed(vk: int) -> bool:
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)


def _detect_human_movement(
    model_movement_idx: int,
    synced_keys: set[str],
) -> tuple[int, int]:
    """Detect human movement key presses.

    ``synced_keys`` is the set of movement keys we asked the OS to hold at the
    end of the previous frame (from ``active_agent_keys``).  Comparing against
    that — rather than the *current* model action's keys — avoids false-positive
    human detection when the model changes direction between frames (the keys
    from the previous frame are still physically pressed at detection time).
    """
    pressed = {k for k, vk in _MOVEMENT_VK.items() if _is_pressed(vk)}
    human_only = pressed - synced_keys

    if human_only:
        for key in ("w", "s", "a", "d"):
            if key in human_only:
                return _KEY_TO_MOVEMENT[key], SRC_HUMAN
    return model_movement_idx, SRC_MODEL


def _detect_human_shooting(
    model_shooting_idx: int,
    synced_keys: set[str],
) -> tuple[int, int]:
    """Detect human shooting key presses. See ``_detect_human_movement``."""
    pressed = {k for k, vk in _SHOOTING_VK.items() if _is_pressed(vk)}
    human_only = pressed - synced_keys

    if human_only:
        for key in ("up", "down", "left", "right"):
            if key in human_only:
                return _KEY_TO_SHOOTING[key], SRC_HUMAN
    return model_shooting_idx, SRC_MODEL


def _detect_human_action_full() -> tuple[int, int, int]:
    """In HUMAN mode, read all physical key state directly.

    Returns (movement, shooting, bomb).
    """
    movement = 0
    for key in ("w", "s", "a", "d"):
        if _is_pressed(_MOVEMENT_VK[key]):
            movement = _KEY_TO_MOVEMENT[key]
            break

    shooting = 0
    for key in ("up", "down", "left", "right"):
        if _is_pressed(_SHOOTING_VK[key]):
            shooting = _KEY_TO_SHOOTING[key]
            break

    bomb = 1 if _is_pressed(_BOMB_VK) else 0
    return movement, shooting, bomb


def _capture_for_recording(raw_bgr: np.ndarray) -> np.ndarray:
    """Convert a raw BGR capture to (3, 256, 256) uint8 RGB for recording."""
    resized = resize_frame(raw_bgr, RECORD_WIDTH, RECORD_HEIGHT)
    return to_rgb(resized)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

FPS = 20
WARMUP_SECONDS = 5.0
OUTPUT_DIR = "rollouts"
TELEMETRY_PORT = 8123
PREVIEW_SCALE = 4
BOMB_THRESHOLD = 0.95


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DAgger collection: model plays, human corrects.",
    )
    parser.add_argument("--model-path", default=None,
                        help="Checkpoint path.  Defaults to the latest model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # --- Load model ---
    model_path = args.model_path or str(find_latest_model())
    model, device, checkpoint = load_policy_checkpoint(model_path)
    use_nav_hint = bool(checkpoint.get("use_nav_hint_embedding", False))
    checkpoint_frame_size = frame_size_from_checkpoint(checkpoint)

    obs_kwargs = obs_config_from_checkpoint(checkpoint)
    env = IsaacFrameEnv(
        title_substring=None,
        observation_config=ObservationConfig(
            width=checkpoint_frame_size,
            height=checkpoint_frame_size,
            **obs_kwargs,
        ),
    )

    # --- Nav hint telemetry ---
    telemetry: IsaacUDPGameStateReceiver | None = None
    if use_nav_hint:
        try:
            telemetry = IsaacUDPGameStateReceiver(port=TELEMETRY_PORT)
        except OSError as exc:
            print(f"[WARN] Could not bind telemetry: {exc}")

    # --- Recording buffers ---
    raw_frames: list[np.ndarray] = []
    movement_actions: list[int] = []
    shooting_actions: list[int] = []
    bomb_actions: list[int] = []
    nav_hints: list[int] = []
    action_sources: list[tuple[int, int, int]] = []  # (mv_src, sh_src, bomb_src)
    timestamps: list[float] = []

    # --- State ---
    armed = False
    emergency_stop = False
    human_mode = False  # F7 toggle: True = human controls, False = model controls
    prev_f7 = False
    prev_f8 = False
    prev_f9 = False
    active_agent_keys: set[str] = set()
    previous_bomb = 0
    latest_nav_hint: int | None = None
    total_steps = 0
    human_steps = 0  # steps where at least one head was human-corrected

    # --- Initial observation ---
    observation = env.reset()
    initial_raw = env.last_raw_frame
    if initial_raw is not None:
        raw_frames.append(_capture_for_recording(initial_raw))
        timestamps.append(time.time())

    print(f"Loaded model: {model_path}")
    print(f"Device: {device}")
    print(f"Capturing: {env.capture.window.title}")
    print()
    print("Controls:")
    print("  F7:  toggle MODEL / HUMAN mode")
    print("  F8:  arm/disarm")
    print("  F9:  emergency stop")
    print("  F10: save & quit")
    print("  q:   quit preview")
    print()

    warmup_until = time.monotonic() + WARMUP_SECONDS
    print(f"Warmup: {WARMUP_SECONDS:.0f}s ...")

    frame_delay = 1.0 / FPS
    next_tick = time.monotonic()

    try:
        while True:
            now = time.monotonic()
            warmup_remaining = max(0.0, warmup_until - now)
            in_warmup = warmup_remaining > 0.0

            # --- Safety state (F8/F9) ---
            cur_f8 = is_function_key_pressed("f8")
            cur_f9 = is_function_key_pressed("f9")
            if cur_f9 and not prev_f9:
                emergency_stop = not emergency_stop
                if emergency_stop:
                    armed = False
                    release_all_agent_keys()
                    active_agent_keys = set()
            if cur_f8 and not prev_f8 and not emergency_stop and not in_warmup:
                armed = not armed
                if not armed:
                    release_all_agent_keys()
                    active_agent_keys = set()
            prev_f8, prev_f9 = cur_f8, cur_f9

            # --- F7: toggle model/human mode ---
            cur_f7 = is_function_key_pressed("f7")
            if cur_f7 and not prev_f7:
                human_mode = not human_mode
                if human_mode:
                    # Entering human mode: release model's simulated keys
                    release_all_agent_keys()
                    active_agent_keys = set()
                print(f"  -> {'HUMAN' if human_mode else 'MODEL'} mode")
            prev_f7 = cur_f7

            if is_function_key_pressed("f10"):
                break

            # --- Capture frame ---
            observation = env.step(action=None)
            raw_bgr = env.last_raw_frame

            # --- Nav hint ---
            nav_hint = None
            nav_status = "disabled"
            if use_nav_hint:
                if telemetry is None:
                    nav_status = "no telemetry"
                else:
                    gs = telemetry.get_latest()
                    if gs is not None and gs.floor_rooms:
                        graph = RoomGraph(gs.floor_rooms)
                        nav_hint = nav_hint_from_room_graph(graph, gs.room_index)
                        latest_nav_hint = nav_hint
                        nav_status = _nav_hint_label(nav_hint)
                    else:
                        nav_status = "waiting"

            # --- Model prediction ---
            prediction = predict_policy(
                model, device, observation,
                checkpoint=checkpoint,
                nav_hint=nav_hint,
            )
            model_action = prediction_to_action(
                prediction,
                bomb_threshold=BOMB_THRESHOLD,
            )

            # --- Determine final action ---
            if human_mode:
                # HUMAN mode: read all physical keys directly
                h_mv, h_sh, h_bomb = _detect_human_action_full()
                final_movement = h_mv
                final_shooting = h_sh
                final_bomb = h_bomb
                mv_src = SRC_HUMAN
                sh_src = SRC_HUMAN
                bomb_src = SRC_HUMAN
            else:
                # MODEL mode: model plays, human overrides by pressing different keys.
                # Compare physical state against keys we synced last frame (which is
                # what's still physically pressed at the start of this frame).
                synced_movement = active_agent_keys & set(_MOVEMENT_VK.keys())
                synced_shooting = active_agent_keys & set(_SHOOTING_VK.keys())
                final_movement, mv_src = _detect_human_movement(model_action.movement, synced_movement)
                final_shooting, sh_src = _detect_human_shooting(model_action.shooting, synced_shooting)
                final_bomb = model_action.bomb
                bomb_src = SRC_MODEL
                # Check human bomb press independently
                if _is_pressed(_BOMB_VK):
                    final_bomb = 1
                    bomb_src = SRC_HUMAN

            final_action = IsaacAction(
                movement=final_movement,
                shooting=final_shooting,
                bomb=final_bomb,
            )

            game_has_focus = env.capture.is_foreground()

            # --- Execute action ---
            if armed and not emergency_stop:
                if not game_has_focus:
                    env.capture.focus_window()
                    time.sleep(0.02)
                    game_has_focus = env.capture.is_foreground()

            if armed and not emergency_stop and game_has_focus:
                if human_mode:
                    # In human mode, release all model keys.
                    # Human's physical presses go directly to the game.
                    active_agent_keys = sync_pressed_keys([], active_agent_keys)
                else:
                    # In model mode: only sync keys for heads the model is
                    # driving.  When a head is human-overridden, leave its
                    # keys un-synced — the human's physical press drives the
                    # game directly, avoiding key-cancellation when model and
                    # human disagree.  Releasing the previously-synced model
                    # keys still happens implicitly via sync_pressed_keys.
                    desired: list[str] = []
                    if mv_src == SRC_MODEL:
                        desired += MOVEMENT_KEY_MAP[final_action.movement]
                    if sh_src == SRC_MODEL:
                        desired += SHOOTING_KEY_MAP[final_action.shooting]
                    active_agent_keys = sync_pressed_keys(desired, active_agent_keys)

                # Bomb is a tap, not a hold — no cancellation issue.
                if final_action.bomb == 1 and previous_bomb == 0:
                    for k in BOMB_KEY_MAP[1]:
                        tap_key(k, hold_seconds=0.05)
            else:
                active_agent_keys = sync_pressed_keys([], active_agent_keys)
                release_all_agent_keys()

            previous_bomb = final_action.bomb

            # --- Record ---
            if armed and not emergency_stop and raw_bgr is not None:
                raw_frames.append(_capture_for_recording(raw_bgr))
                timestamps.append(time.time())
                movement_actions.append(final_movement)
                shooting_actions.append(final_shooting)
                bomb_actions.append(final_bomb)
                action_sources.append((mv_src, sh_src, bomb_src))

                if nav_hint is not None:
                    nav_hints.append(nav_hint)
                else:
                    nav_hints.append(0)

                total_steps += 1
                any_human = (mv_src == SRC_HUMAN or sh_src == SRC_HUMAN
                             or bomb_src == SRC_HUMAN)
                if any_human:
                    human_steps += 1

            # --- Preview ---
            preview = cv2.cvtColor(observation[-1], cv2.COLOR_GRAY2BGR)
            preview = cv2.resize(
                preview,
                (preview.shape[1] * PREVIEW_SCALE, preview.shape[0] * PREVIEW_SCALE),
                interpolation=cv2.INTER_NEAREST,
            )
            _draw_overlay(
                preview,
                model_path=model_path,
                prediction=prediction,
                final_action=final_action,
                mv_src=mv_src,
                sh_src=sh_src,
                armed=armed,
                emergency_stop=emergency_stop,
                game_has_focus=game_has_focus,
                human_mode=human_mode,
                warmup_remaining=warmup_remaining,
                total_steps=total_steps,
                human_steps=human_steps,
                nav_status=nav_status if use_nav_hint else "not used",
            )
            cv2.imshow("DAgger Collection", preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            next_tick += frame_delay
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
    finally:
        release_all_agent_keys()
        cv2.destroyAllWindows()

    # --- Save ---
    if total_steps == 0:
        print("No steps recorded. Nothing to save.")
        return

    _save_rollout(
        output_dir=OUTPUT_DIR,
        raw_frames=raw_frames,
        movement_actions=movement_actions,
        shooting_actions=shooting_actions,
        bomb_actions=bomb_actions,
        nav_hints=nav_hints,
        action_sources=action_sources,
        timestamps=timestamps,
        capture_title=env.capture.window.title,
        model_path=model_path,
        total_steps=total_steps,
        human_steps=human_steps,
        fps=FPS,
    )


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def _save_rollout(
    output_dir: str,
    raw_frames: list[np.ndarray],
    movement_actions: list[int],
    shooting_actions: list[int],
    bomb_actions: list[int],
    nav_hints: list[int],
    action_sources: list[tuple[int, int, int]],
    timestamps: list[float],
    capture_title: str,
    model_path: str,
    total_steps: int,
    human_steps: int,
    fps: int,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_name = time.strftime("dagger_%Y%m%d_%H%M%S")
    run_dir = out / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    frames_array = np.stack(raw_frames, axis=0)  # (N+1, 3, H, W) uint8
    np.savez_compressed(
        run_dir / "rollout_data.npz",
        raw_frames=frames_array,
        movement_actions=np.asarray(movement_actions, dtype=np.int64),
        shooting_actions=np.asarray(shooting_actions, dtype=np.int64),
        bomb_actions=np.asarray(bomb_actions, dtype=np.int64),
        nav_hints=np.asarray(nav_hints, dtype=np.int64),
        action_sources=np.asarray(action_sources, dtype=np.int64),
        timestamps=np.asarray(timestamps, dtype=np.float64),
    )

    human_pct = (human_steps / total_steps * 100) if total_steps > 0 else 0.0
    metadata = {
        "run_dir": str(run_dir),
        "window_title": capture_title,
        "config": {
            "width": RECORD_WIDTH,
            "height": RECORD_HEIGHT,
            "fps": fps,
            "color": True,
        },
        "frame_shape": list(frames_array.shape[1:]),
        "num_actions_logged": total_steps,
        "movement_space": {"0": "idle", "1": "up", "2": "down", "3": "left", "4": "right"},
        "shooting_space": {"0": "idle", "1": "up", "2": "down", "3": "left", "4": "right"},
        "bomb_space": {"0": "no_bomb", "1": "bomb"},
        "action_schema_version": 3,
        "nav_hints_recorded": True,
        "dagger": True,
        "model_path": model_path,
        "total_steps": total_steps,
        "human_correction_steps": human_steps,
        "human_correction_pct": round(human_pct, 1),
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8",
    )

    print(f"\nSaved {total_steps} steps -> {run_dir}")
    print(f"Human corrections: {human_steps}/{total_steps} ({human_pct:.1f}%)")
    print(f"Frames shape: {frames_array.shape}")


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------

def _draw_overlay(
    preview: np.ndarray,
    model_path: str,
    prediction,
    final_action: IsaacAction,
    mv_src: int,
    sh_src: int,
    armed: bool,
    emergency_stop: bool,
    game_has_focus: bool,
    human_mode: bool,
    warmup_remaining: float,
    total_steps: int,
    human_steps: int,
    nav_status: str,
) -> None:
    if warmup_remaining > 0:
        status = f"WARMUP {warmup_remaining:.1f}s"
    elif emergency_stop:
        status = "EMERGENCY STOP"
    elif armed:
        status = "ARMED & RECORDING"
    else:
        status = "DISARMED"

    mode_label = "HUMAN" if human_mode else "MODEL"
    mv_tag = "[H]" if mv_src == SRC_HUMAN else "[M]"
    sh_tag = "[H]" if sh_src == SRC_HUMAN else "[M]"
    human_pct = (human_steps / total_steps * 100) if total_steps > 0 else 0.0

    lines = [
        f"DAGGER  mode={mode_label}  status={status}",
        f"game: {'FOCUSED' if game_has_focus else 'NOT FOCUSED'}",
        f"nav: {nav_status}",
        f"model: {Path(model_path).parent.name}",
        (
            f"move: {prediction.movement.label} "
            f"({prediction.movement.confidence:.2f}) "
            f"-> {final_action.movement} {mv_tag}"
        ),
        (
            f"shoot: {prediction.shooting.label} "
            f"({prediction.shooting.confidence:.2f}) "
            f"-> {final_action.shooting} {sh_tag}"
        ),
        f"recorded: {total_steps}  human: {human_steps} ({human_pct:.0f}%)",
        "F7=mode  F8=arm  F9=estop  F10=save+quit",
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    for idx, line in enumerate(lines):
        y = 20 + idx * 22
        if idx == 0:
            color = (80, 255, 80) if human_mode else (255, 200, 80)
        elif idx == 1:
            color = (80, 220, 120) if game_has_focus else (64, 64, 255)
        else:
            color = (245, 245, 245)
        cv2.putText(preview, line, (8, y), font, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(preview, line, (8, y), font, 0.45, color, 1, cv2.LINE_AA)


def _nav_hint_label(nav_hint: int) -> str:
    return {
        int(NavHint.STAY): "STAY",
        int(NavHint.NORTH): "NORTH",
        int(NavHint.SOUTH): "SOUTH",
        int(NavHint.WEST): "WEST",
        int(NavHint.EAST): "EAST",
    }.get(nav_hint, f"?({nav_hint})")


if __name__ == "__main__":
    main()
