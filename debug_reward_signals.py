from __future__ import annotations

import argparse
import time

import cv2

from binding_rl_agent.env import ObservationConfig
from binding_rl_agent.rl_env import IsaacVisualRLEnv
from binding_rl_agent.room_graph import NavHint, RoomGraph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview telemetry-based reward/done signals on the live Isaac feed."
    )
    parser.add_argument("--title", default=None, help="Optional game window title substring.")
    parser.add_argument("--width", type=int, default=128, help="Observation width.")
    parser.add_argument("--height", type=int, default=128, help="Observation height.")
    parser.add_argument("--fps", type=int, default=10, help="Preview FPS.")
    parser.add_argument("--preview-scale", type=int, default=4, help="Preview scale factor.")
    parser.add_argument("--telemetry-port", type=int, default=8123, help="UDP telemetry port.")
    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=5.0,
        help="How long to wait for telemetry before failing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = IsaacVisualRLEnv(
        title_substring=args.title,
        observation_config=ObservationConfig(
            width=args.width,
            height=args.height,
            stack_size=4,
            grayscale=True,
        ),
        telemetry_port=args.telemetry_port,
    )
    print(f"Capturing window: {env.frame_env.capture.window.title}")
    print(f"Waiting for telemetry on UDP port {args.telemetry_port}.")
    print("Press q to quit.")
    observation = env.frame_env.reset()
    _wait_for_telemetry(env, args.warmup_seconds)

    frame_delay = 1.0 / max(args.fps, 1)
    next_tick = time.monotonic()

    while True:
        step = env.step(action=None)
        observation = step.observation
        preview = cv2.cvtColor(observation[-1], cv2.COLOR_GRAY2BGR)
        if args.preview_scale != 1:
            preview = cv2.resize(
                preview,
                (preview.shape[1] * args.preview_scale, preview.shape[0] * args.preview_scale),
                interpolation=cv2.INTER_NEAREST,
            )

        state = step.info["game_state"]
        graph = RoomGraph.from_game_state(state)
        nav   = graph.nav_hint(state.room_index)
        _draw_overlay(preview, step.reward, step.done, step.info, nav, graph)
        cv2.imshow("Isaac Reward Debug", preview)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        next_tick += frame_delay
        sleep_for = next_tick - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)

    cv2.destroyAllWindows()


_NAV_LABEL = {
    NavHint.STAY:  "[STAY]",
    NavHint.NORTH: "[ N  ]",
    NavHint.SOUTH: "[ S  ]",
    NavHint.WEST:  "[ W  ]",
    NavHint.EAST:  "[ E  ]",
}


def _draw_overlay(
    preview,
    reward: float,
    done: bool,
    info: dict[str, object],
    nav: NavHint,
    graph: RoomGraph,
) -> None:
    state = info["game_state"]
    n_rooms    = len(state.floor_rooms)
    n_visited  = sum(1 for r in state.floor_rooms if r.visited > 0)
    n_cleared  = sum(1 for r in state.floor_rooms if r.cleared)
    lines = [
        f"nav: {_NAV_LABEL[nav]}  room_idx={state.room_index}",
        f"floor rooms: {n_rooms} known  {n_visited} visited  {n_cleared} cleared",
        "---",
        f"reward: {reward:.3f}",
        f"done: {done}",
        f"transition: {info['room_transition']}",
        f"rooms_explored+: {info['rooms_explored_gained']}",
        f"room_clear: {info['room_clear_candidate']}",
        f"kills+: {info['kills_gained']}",
        f"damage+: {info['damage_taken']:.2f}",
        f"coins+: {info['coins_gained']}",
        f"keys+: {info['keys_gained']}",
        f"collectibles+: {info['collectibles_gained']}",
        f"soul_hearts_delta: {info['soul_hearts_delta']}",
        f"black_hearts_delta: {info['black_hearts_delta']}",
        f"stagnant_steps: {info['stagnant_steps']}",
        f"room_steps: {info['room_steps']}",
        f"stagnation_penalty: {info['stagnation_penalty_applied']:.3f}",
        f"timeout_penalty: {info['timeout_penalty_applied']:.3f}",
        f"timeout_done: {info['timeout_done']}",
        f"deaths_total: {state.deaths}",
        f"rooms_cleared_total: {state.rooms_cleared}",
        "q: quit",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    for idx, line in enumerate(lines):
        y = 20 + idx * 22
        cv2.putText(preview, line, (8, y), font, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(preview, line, (8, y), font, 0.5, (245, 245, 245), 1, cv2.LINE_AA)


def _wait_for_telemetry(env: IsaacVisualRLEnv, warmup_seconds: float) -> None:
    deadline = time.monotonic() + max(warmup_seconds, 0.0)
    while time.monotonic() < deadline:
        game_state = env.game_state_receiver.get_latest()
        if game_state is not None:
            env.reward_detector.reset()
            env.reward_detector.previous_game_state = game_state
            print("Telemetry connected.")
            return
        time.sleep(0.1)
    raise RuntimeError(
        "No Isaac telemetry received on UDP port "
        f"{env.telemetry_port} within {warmup_seconds:.1f}s."
    )


if __name__ == "__main__":
    main()
