from __future__ import annotations

import argparse
import time

import cv2

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

FPS = 20
WARMUP_SECONDS = 5.0
TELEMETRY_PORT = 8123
PREVIEW_SCALE = 4
BOMB_THRESHOLD = 0.95


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a safety-gated live policy controller for Isaac. "
            "Starts disarmed and only sends inputs when armed."
        )
    )
    parser.add_argument("--model-path", default=None,
                        help="Checkpoint path.  Defaults to the latest model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model_path or str(find_latest_model())
    model, device, checkpoint = load_policy_checkpoint(model_path)
    use_nav_hint_embedding = bool(checkpoint.get("use_nav_hint_embedding", False))
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
    telemetry: IsaacUDPGameStateReceiver | None = None
    if use_nav_hint_embedding:
        try:
            telemetry = IsaacUDPGameStateReceiver(port=TELEMETRY_PORT)
        except OSError as exc:
            print(
                f"[WARN] Could not bind telemetry port {TELEMETRY_PORT}: {exc}. "
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

    observation = env.reset()

    print(f"Loaded model: {model_path}")
    print(f"Using device: {device}")
    print(f"Capturing window: {env.capture.window.title}")
    if use_nav_hint_embedding:
        if telemetry is not None:
            print(f"Nav hint telemetry listening on UDP port {TELEMETRY_PORT}")
        else:
            print("Nav hint input unavailable; using fallback STAY hint.")
    print("Controls: F8 arm/disarm, F9 emergency stop, F10 quit, q quit preview")

    warmup_until = time.monotonic() + WARMUP_SECONDS
    print(f"Warmup: {WARMUP_SECONDS:.0f}s before arming is available...")

    frame_delay = 1.0 / FPS
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
            model, device, observation,
            checkpoint=checkpoint,
            nav_hint=nav_hint,
        )
        selected_action = prediction_to_action(prediction, bomb_threshold=BOMB_THRESHOLD)
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
        preview = cv2.resize(
            preview,
            (preview.shape[1] * PREVIEW_SCALE, preview.shape[0] * PREVIEW_SCALE),
            interpolation=cv2.INTER_NEAREST,
        )

        _draw_overlay(
            preview=preview,
            model_path=model_path,
            prediction=prediction,
            selected_action=selected_action,
            armed=armed,
            emergency_stop=emergency_stop,
            game_has_focus=game_has_focus,
            warmup_remaining=warmup_remaining,
            nav_status=nav_status,
            model_uses_nav_hint=use_nav_hint_embedding,
            latest_nav_hint=latest_nav_hint,
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


def _draw_overlay(
    preview,
    model_path: str,
    prediction,
    selected_action: IsaacAction,
    armed: bool,
    emergency_stop: bool,
    game_has_focus: bool,
    warmup_remaining: float,
    nav_status: str,
    model_uses_nav_hint: bool,
    latest_nav_hint: int | None,
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

    overlay_lines = [
        f"model: {model_path}",
        f"status: {status}",
        f"game: {focus_label}",
        f"nav: {nav_status}" if model_uses_nav_hint else "nav: not used by model",
        (
            f"move: {prediction.movement.label} ({prediction.movement.confidence:.2f}) "
            f"-> {selected_action.movement}"
        ),
        (
            f"shoot: {prediction.shooting.label} ({prediction.shooting.confidence:.2f}) "
            f"-> {selected_action.shooting}"
        ),
        (
            f"bomb: {prediction.bomb.label} ({prediction.bomb.confidence:.2f}) "
            f"-> {selected_action.bomb}"
        ),
        "F8: arm/disarm  F9: emergency stop  F10/q: quit",
    ]

    color = (64, 64, 255) if emergency_stop else (
        (80, 220, 120) if armed else (
            (180, 180, 180) if warmup_remaining > 0.0 else (240, 210, 90)
        )
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    for idx, line in enumerate(overlay_lines):
        y = 20 + idx * 22
        if idx == 1:
            text_color = color
        elif idx == 2:
            text_color = (80, 220, 120) if game_has_focus else (64, 64, 255)
        elif idx == 3 and model_uses_nav_hint and latest_nav_hint is not None:
            text_color = _nav_hint_color(latest_nav_hint)
        else:
            text_color = (245, 245, 245)
        cv2.putText(preview, line, (8, y), font, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(preview, line, (8, y), font, 0.5, text_color, 1, cv2.LINE_AA)


def _nav_hint_label(nav_hint: int) -> str:
    return {
        int(NavHint.STAY): "STAY",
        int(NavHint.NORTH): "NORTH",
        int(NavHint.SOUTH): "SOUTH",
        int(NavHint.WEST): "WEST",
        int(NavHint.EAST): "EAST",
    }.get(nav_hint, f"UNKNOWN({nav_hint})")


def _nav_hint_color(nav_hint: int) -> tuple[int, int, int]:
    return {
        int(NavHint.STAY): (180, 180, 180),
        int(NavHint.NORTH): (255, 210, 80),
        int(NavHint.SOUTH): (80, 180, 255),
        int(NavHint.WEST): (120, 220, 120),
        int(NavHint.EAST): (220, 120, 220),
    }.get(nav_hint, (245, 245, 245))


if __name__ == "__main__":
    main()
