from __future__ import annotations

import argparse
import time

from binding_rl_agent.input_controller import (
    is_function_key_pressed,
    release_all_agent_keys,
    sync_pressed_keys,
    tap_key,
)
from binding_rl_agent.window_capture import IsaacWindowCapture


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Directly test whether synthetic keyboard input reaches Isaac. "
            "This ignores the model and sends a fixed input sequence."
        )
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional substring to match the Isaac window title.",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=1.0,
        help="How long to hold each directional test key.",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=3.0,
        help="Seconds to wait before starting the first test input.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    capture = IsaacWindowCapture(title_substring=args.title)
    active_keys: set[str] = set()

    print(f"Target window: {capture.window.title}")
    print("This script will try a few direct input tests.")
    print("Sequence: hold A, hold D, hold W, hold S, tap Left Arrow.")
    print("Press F10 at any time to stop.")

    for seconds_left in range(int(args.warmup), 0, -1):
        print(f"Starting in {seconds_left}...")
        time.sleep(1.0)

    try:
        _ensure_focus(capture)
        _run_hold_test("a", args.hold_seconds, active_keys)
        _run_hold_test("d", args.hold_seconds, active_keys)
        _run_hold_test("w", args.hold_seconds, active_keys)
        _run_hold_test("s", args.hold_seconds, active_keys)
        _run_tap_test("left")
        print("Input sequence completed.")
    finally:
        active_keys = sync_pressed_keys([], active_keys)
        release_all_agent_keys()
        print("Released all agent keys.")


def _ensure_focus(capture: IsaacWindowCapture) -> None:
    if not capture.is_foreground():
        print("Trying to focus the Isaac window...")
        capture.focus_window()
        time.sleep(0.2)
    print(f"Game focused: {capture.is_foreground()}")


def _run_hold_test(key: str, hold_seconds: float, active_keys: set[str]) -> None:
    print(f"Holding {key.upper()} for {hold_seconds:.1f}s")
    active_keys.clear()
    active_keys.update(sync_pressed_keys([key], active_keys))
    _sleep_with_abort(hold_seconds)
    active_keys.update(sync_pressed_keys([], active_keys))
    time.sleep(0.2)


def _run_tap_test(key: str) -> None:
    print(f"Tapping {key.upper()}")
    tap_key(key, hold_seconds=0.08)
    time.sleep(0.2)


def _sleep_with_abort(seconds: float) -> None:
    end_time = time.monotonic() + seconds
    while time.monotonic() < end_time:
        if is_function_key_pressed("f10"):
            raise KeyboardInterrupt("Stopped by F10")
        time.sleep(0.01)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as error:
        print(str(error))
