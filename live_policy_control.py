from __future__ import annotations

import argparse
import time
from collections import deque

import cv2
import numpy as np

from binding_rl_agent.dataset import MOVEMENT_NAMES, SHOOTING_NAMES
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
from binding_rl_agent.room_graph import NavHint, RoomGraph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a safety-gated live policy controller for Isaac. "
            "Starts disarmed and only sends inputs when armed."
        )
    )
    parser.add_argument("--model-path", default=None, help="Optional checkpoint path.")
    parser.add_argument("--title", default=None, help="Optional game window title substring.")
    parser.add_argument("--width", type=int, default=None, help="Observation width (default: from checkpoint).")
    parser.add_argument("--height", type=int, default=None, help="Observation height (default: from checkpoint).")
    parser.add_argument("--fps", type=int, default=10, help="Control loop FPS.")
    parser.add_argument(
        "--movement-threshold",
        type=float,
        default=0.35,
        help="Confidence threshold below which movement becomes idle.",
    )
    parser.add_argument(
        "--shooting-threshold",
        type=float,
        default=0.35,
        help="Confidence threshold below which shooting becomes idle.",
    )
    parser.add_argument(
        "--bomb-threshold",
        type=float,
        default=0.95,
        help="Confidence threshold required before bomb is pressed.",
    )
    parser.add_argument(
        "--preview-scale",
        type=int,
        default=4,
        help="Scale factor for the preview window.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use argmax actions (sets all thresholds to 0, bomb threshold to 0.5).",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=5.0,
        help="Seconds to wait before arming becomes available (default 5).",
    )
    parser.add_argument(
        "--telemetry-port",
        type=int,
        default=8123,
        help="UDP telemetry port for room-graph nav hints.",
    )
    parser.add_argument(
        "--force-non-idle",
        action="store_true",
        help=(
            "When movement/shooting predicts idle as the top class, override it "
            "with the best non-idle class instead."
        ),
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=1,
        help=(
            "Temporal smoothing window size.  When > 1, averages softmax "
            "probabilities over the last N frames before taking argmax.  "
            "Reduces jitter and helps the model commit to a direction.  "
            "Try 5-10 (at 10 fps, 5 = 0.5s smoothing window)."
        ),
    )
    parser.add_argument(
        "--unstick",
        type=int,
        default=0,
        help=(
            "Stuck detection threshold (number of low-change frames before "
            "triggering).  When the agent has been predicting the same non-idle "
            "movement for N frames and the screen barely changes, suppress that "
            "direction and pick the next best.  Try 8-15 (at 10 fps, 10 = 1s). "
            "0 = disabled."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model_path or str(find_latest_model())
    model, device, checkpoint = load_policy_checkpoint(model_path)
    use_nav_hint_embedding = bool(checkpoint.get("use_nav_hint_embedding", False))
    checkpoint_frame_size = frame_size_from_checkpoint(checkpoint)
    width = args.width or checkpoint_frame_size
    height = args.height or checkpoint_frame_size

    if args.deterministic:
        movement_threshold = 0.0
        shooting_threshold = 0.0
        bomb_threshold = 0.5
    else:
        movement_threshold = args.movement_threshold
        shooting_threshold = args.shooting_threshold
        bomb_threshold = args.bomb_threshold

    obs_kwargs = obs_config_from_checkpoint(checkpoint)
    env = IsaacFrameEnv(
        title_substring=args.title,
        observation_config=ObservationConfig(
            width=width,
            height=height,
            **obs_kwargs,
        ),
    )
    telemetry: IsaacUDPGameStateReceiver | None = None
    if use_nav_hint_embedding:
        try:
            telemetry = IsaacUDPGameStateReceiver(port=args.telemetry_port)
        except OSError as exc:
            print(
                f"[WARN] Could not bind telemetry port {args.telemetry_port}: {exc}. "
                "Nav-hint-enabled model will fall back to STAY hints."
            )

    armed = False
    emergency_stop = False
    previous_f8 = False
    previous_f9 = False
    previous_bomb = 0
    active_agent_keys: set[str] = set()
    latest_nav_hint: int | None = None
    nav_status = "disabled"

    # Temporal smoothing buffers
    smooth_n = max(1, args.smooth)
    mv_prob_buffer: deque[np.ndarray] = deque(maxlen=smooth_n)
    sh_prob_buffer: deque[np.ndarray] = deque(maxlen=smooth_n)

    # Stuck detection state
    unstick_n = max(0, args.unstick)
    prev_frame: np.ndarray | None = None
    low_change_count = 0           # consecutive frames with minimal change
    same_direction_count = 0       # consecutive frames predicting same non-idle direction
    last_movement_idx = -1
    stuck_suppressed: int | None = None  # direction currently being suppressed
    stuck_status = ""

    observation = env.reset()

    print(f"Loaded model: {model_path}")
    print(f"Using device: {device}")
    print(f"Capturing window: {env.capture.window.title}")
    if use_nav_hint_embedding:
        if telemetry is not None:
            print(f"Nav hint telemetry listening on UDP port {args.telemetry_port}")
        else:
            print("Nav hint input unavailable; using fallback STAY hint.")
    print("Controls: F8 arm/disarm, F9 emergency stop, F10 quit, q quit preview")
    print("Tip: when armed, the controller will try to refocus the Isaac window.")

    warmup_until = time.monotonic() + args.warmup
    if args.warmup > 0:
        print(f"Warmup: {args.warmup:.0f}s before arming is available...")

    frame_delay = 1.0 / max(args.fps, 1)
    next_tick = time.monotonic()

    while True:
        now = time.monotonic()
        warmup_remaining = max(0.0, warmup_until - now)
        in_warmup = warmup_remaining > 0.0

        armed, emergency_stop, previous_f8, previous_f9 = _update_safety_state(
            armed=armed,
            emergency_stop=emergency_stop,
            previous_f8=previous_f8,
            previous_f9=previous_f9,
            allow_arm=not in_warmup,
        )
        if is_function_key_pressed("f10"):
            break

        observation = env.step(action=None)
        nav_hint = None
        if use_nav_hint_embedding:
            if telemetry is None:
                nav_status = "fallback STAY (telemetry bind failed)"
            else:
                game_state = telemetry.get_latest()
                if game_state is not None and game_state.floor_rooms:
                    graph = RoomGraph(game_state.floor_rooms)
                    nav_hint = nav_hint_from_room_graph(graph, game_state.room_index)
                    latest_nav_hint = nav_hint
                    nav_status = _nav_hint_label(nav_hint)
                elif game_state is not None:
                    nav_status = "fallback STAY (no room graph)"
                else:
                    nav_status = "fallback STAY (waiting for telemetry)"
        prediction = predict_policy(
            model,
            device,
            observation,
            checkpoint=checkpoint,
            nav_hint=nav_hint,
        )

        # Temporal smoothing: average softmax probs over recent frames
        mv_prob_buffer.append(np.array(prediction.movement.probabilities))
        sh_prob_buffer.append(np.array(prediction.shooting.probabilities))

        if smooth_n > 1:
            smoothed_prediction = _apply_smoothing(
                prediction, mv_prob_buffer, sh_prob_buffer,
            )
        else:
            smoothed_prediction = prediction

        # --- Stuck detection ---
        if unstick_n > 0:
            current_frame = observation[-1]  # last frame in stack (H, W) uint8
            movement_idx = smoothed_prediction.movement.index

            # Track frame similarity
            if prev_frame is not None:
                frame_diff = float(np.mean(np.abs(
                    current_frame.astype(np.float32) - prev_frame.astype(np.float32)
                )))
                is_low_change = frame_diff < 2.0  # nearly identical frames
            else:
                is_low_change = False
                frame_diff = 999.0

            # Track same-direction persistence
            if movement_idx != 0 and movement_idx == last_movement_idx:
                same_direction_count += 1
            else:
                same_direction_count = 0
                stuck_suppressed = None  # clear suppression when direction changes

            if is_low_change:
                low_change_count += 1
            else:
                low_change_count = 0
                stuck_suppressed = None  # clear suppression when frame changes

            # Trigger: same non-idle direction for N frames AND screen not changing
            if (same_direction_count >= unstick_n
                    and low_change_count >= unstick_n
                    and movement_idx != 0):
                stuck_suppressed = movement_idx
                stuck_status = f"STUCK! suppressing {MOVEMENT_NAMES.get(movement_idx, '?')}"
                # Reset counters so we re-evaluate the new direction
                same_direction_count = 0
                low_change_count = 0
            elif stuck_suppressed is not None:
                stuck_status = f"unstick: blocked {MOVEMENT_NAMES.get(stuck_suppressed, '?')}"
            else:
                stuck_status = ""

            last_movement_idx = movement_idx
            prev_frame = current_frame

            # Apply suppression: zero out the stuck direction, re-pick
            if stuck_suppressed is not None:
                smoothed_prediction = _suppress_direction(
                    smoothed_prediction, stuck_suppressed,
                )
        else:
            stuck_status = ""

        selected_action = prediction_to_action(
            smoothed_prediction,
            movement_threshold=movement_threshold,
            shooting_threshold=shooting_threshold,
            bomb_threshold=bomb_threshold,
        )
        if args.force_non_idle:
            selected_action = _force_non_idle_action(selected_action, smoothed_prediction)
        game_has_focus = env.capture.is_foreground()

        if armed and not emergency_stop:
            if not game_has_focus:
                env.capture.focus_window()
                time.sleep(0.02)
                game_has_focus = env.capture.is_foreground()

        if armed and not emergency_stop and game_has_focus:
            desired_keys = (
                MOVEMENT_KEY_MAP[selected_action.movement]
                + SHOOTING_KEY_MAP[selected_action.shooting]
            )
            active_agent_keys = sync_pressed_keys(desired_keys, active_agent_keys)
            if selected_action.bomb == 1 and previous_bomb == 0:
                for bomb_key in BOMB_KEY_MAP[1]:
                    tap_key(bomb_key, hold_seconds=0.05)
        else:
            active_agent_keys = sync_pressed_keys([], active_agent_keys)
            release_all_agent_keys()
        previous_bomb = selected_action.bomb

        preview = cv2.cvtColor(observation[-1], cv2.COLOR_GRAY2BGR)
        if args.preview_scale != 1:
            preview = cv2.resize(
                preview,
                (preview.shape[1] * args.preview_scale, preview.shape[0] * args.preview_scale),
                interpolation=cv2.INTER_NEAREST,
            )

        _draw_overlay(
            preview=preview,
            model_path=model_path,
            prediction=smoothed_prediction,
            selected_action=selected_action,
            armed=armed,
            emergency_stop=emergency_stop,
            game_has_focus=game_has_focus,
            warmup_remaining=warmup_remaining,
            nav_status=nav_status,
            model_uses_nav_hint=use_nav_hint_embedding,
            latest_nav_hint=latest_nav_hint,
            force_non_idle=args.force_non_idle,
            smooth_n=smooth_n,
            raw_prediction=prediction if smooth_n > 1 else None,
            stuck_status=stuck_status,
        )
        cv2.imshow("Isaac Live Policy Control", preview)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        next_tick += frame_delay
        sleep_for = next_tick - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)

    active_agent_keys = sync_pressed_keys([], active_agent_keys)
    release_all_agent_keys()
    cv2.destroyAllWindows()


def _update_safety_state(
    armed: bool,
    emergency_stop: bool,
    previous_f8: bool,
    previous_f9: bool,
    allow_arm: bool = True,
) -> tuple[bool, bool, bool, bool]:
    current_f8 = is_function_key_pressed("f8")
    current_f9 = is_function_key_pressed("f9")

    if current_f9 and not previous_f9:
        emergency_stop = not emergency_stop
        if emergency_stop:
            armed = False
            release_all_agent_keys()

    if current_f8 and not previous_f8 and not emergency_stop and allow_arm:
        armed = not armed
        if not armed:
            release_all_agent_keys()

    return armed, emergency_stop, current_f8, current_f9


def _apply_smoothing(
    prediction,
    mv_buffer: deque[np.ndarray],
    sh_buffer: deque[np.ndarray],
):
    """Average softmax probs over the buffer and return a smoothed prediction."""
    from binding_rl_agent.inference import HeadPrediction, PolicyPrediction

    mv_avg = np.mean(list(mv_buffer), axis=0)
    sh_avg = np.mean(list(sh_buffer), axis=0)

    mv_idx = int(np.argmax(mv_avg))
    sh_idx = int(np.argmax(sh_avg))

    return PolicyPrediction(
        movement=HeadPrediction(
            index=mv_idx,
            label=MOVEMENT_NAMES[mv_idx],
            confidence=float(mv_avg[mv_idx]),
            probabilities=tuple(float(p) for p in mv_avg),
        ),
        shooting=HeadPrediction(
            index=sh_idx,
            label=SHOOTING_NAMES[sh_idx],
            confidence=float(sh_avg[sh_idx]),
            probabilities=tuple(float(p) for p in sh_avg),
        ),
        bomb=prediction.bomb,
        device=prediction.device,
    )


def _suppress_direction(prediction, suppress_idx: int):
    """Zero out a movement direction and re-pick from the remaining probs."""
    from binding_rl_agent.inference import HeadPrediction, PolicyPrediction

    probs = np.array(prediction.movement.probabilities)
    probs[suppress_idx] = 0.0
    total = probs.sum()
    if total > 0:
        probs /= total
    else:
        # All probs zeroed — fall back to idle
        probs[0] = 1.0

    new_idx = int(np.argmax(probs))
    return PolicyPrediction(
        movement=HeadPrediction(
            index=new_idx,
            label=MOVEMENT_NAMES[new_idx],
            confidence=float(probs[new_idx]),
            probabilities=tuple(float(p) for p in probs),
        ),
        shooting=prediction.shooting,
        bomb=prediction.bomb,
        device=prediction.device,
    )


def _draw_overlay(
    preview,
    model_path: str,
    prediction,
    selected_action: IsaacAction,
    armed: bool,
    emergency_stop: bool,
    game_has_focus: bool,
    warmup_remaining: float = 0.0,
    nav_status: str = "disabled",
    model_uses_nav_hint: bool = False,
    latest_nav_hint: int | None = None,
    force_non_idle: bool = False,
    smooth_n: int = 1,
    raw_prediction=None,
    stuck_status: str = "",
) -> None:
    if warmup_remaining > 0.0:
        status = f"WARMUP {warmup_remaining:.1f}s"
    elif emergency_stop:
        status = "EMERGENCY STOP"
    elif armed:
        status = "ARMED"
    else:
        status = "DISARMED"
    focus_label = "FOCUSED" if game_has_focus else "NOT FOCUSED"

    # Build move/shoot lines showing raw -> smoothed when smoothing is active
    if raw_prediction is not None:
        move_line = (
            f"move: {raw_prediction.movement.label}({raw_prediction.movement.confidence:.2f})"
            f" ~{smooth_n}~> {prediction.movement.label}({prediction.movement.confidence:.2f})"
            f" -> {selected_action.movement}"
        )
        shoot_line = (
            f"shoot: {raw_prediction.shooting.label}({raw_prediction.shooting.confidence:.2f})"
            f" ~{smooth_n}~> {prediction.shooting.label}({prediction.shooting.confidence:.2f})"
            f" -> {selected_action.shooting}"
        )
    else:
        move_line = (
            f"move: {prediction.movement.label} ({prediction.movement.confidence:.2f}) "
            f"-> {selected_action.movement}"
        )
        shoot_line = (
            f"shoot: {prediction.shooting.label} ({prediction.shooting.confidence:.2f}) "
            f"-> {selected_action.shooting}"
        )

    overlay_lines = [
        f"model: {model_path}",
        f"status: {status}  smooth={smooth_n}",
        f"game: {focus_label}",
        (
            f"nav: {nav_status}"
            if model_uses_nav_hint
            else "nav: not used by model"
        ),
        f"non-idle override: {'ON' if force_non_idle else 'OFF'}",
        move_line,
        shoot_line,
        (
            f"bomb: {prediction.bomb.label} ({prediction.bomb.confidence:.2f}) "
            f"-> {selected_action.bomb}"
        ),
        "F8: arm/disarm",
        "F9: emergency stop toggle",
        "F10/q: quit",
    ]
    if stuck_status:
        overlay_lines.insert(5, stuck_status)

    color = (64, 64, 255) if emergency_stop else ((80, 220, 120) if armed else ((180, 180, 180) if warmup_remaining > 0.0 else (240, 210, 90)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    stuck_line_idx = 5 if stuck_status else -1
    for idx, line in enumerate(overlay_lines):
        y = 20 + idx * 22
        if idx == 1:
            text_color = color
        elif idx == 2:
            text_color = (80, 220, 120) if game_has_focus else (64, 64, 255)
        elif idx == 3 and model_uses_nav_hint and latest_nav_hint is not None:
            text_color = _nav_hint_color(latest_nav_hint)
        elif idx == stuck_line_idx:
            text_color = (0, 100, 255)  # orange for stuck warning
        else:
            text_color = (245, 245, 245)
        cv2.putText(preview, line, (8, y), font, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(preview, line, (8, y), font, 0.5, text_color, 1, cv2.LINE_AA)


def _nav_hint_label(nav_hint: int) -> str:
    labels = {
        int(NavHint.STAY): "STAY",
        int(NavHint.NORTH): "NORTH",
        int(NavHint.SOUTH): "SOUTH",
        int(NavHint.WEST): "WEST",
        int(NavHint.EAST): "EAST",
    }
    return labels.get(nav_hint, f"UNKNOWN({nav_hint})")


def _nav_hint_color(nav_hint: int) -> tuple[int, int, int]:
    colors = {
        int(NavHint.STAY): (180, 180, 180),
        int(NavHint.NORTH): (255, 210, 80),
        int(NavHint.SOUTH): (80, 180, 255),
        int(NavHint.WEST): (120, 220, 120),
        int(NavHint.EAST): (220, 120, 220),
    }
    return colors.get(nav_hint, (245, 245, 245))


def _force_non_idle_action(selected_action: IsaacAction, prediction) -> IsaacAction:
    movement = selected_action.movement
    shooting = selected_action.shooting

    if prediction.movement.index == 0:
        movement = _best_non_idle_index(prediction.movement.probabilities)
    if prediction.shooting.index == 0:
        shooting = _best_non_idle_index(prediction.shooting.probabilities)

    return IsaacAction(
        movement=movement,
        shooting=shooting,
        bomb=selected_action.bomb,
    )


def _best_non_idle_index(probabilities: tuple[float, ...]) -> int:
    if len(probabilities) <= 1:
        return 0
    best_index = 1
    best_prob = probabilities[1]
    for idx in range(2, len(probabilities)):
        if probabilities[idx] > best_prob:
            best_index = idx
            best_prob = probabilities[idx]
    return best_index


if __name__ == "__main__":
    main()
