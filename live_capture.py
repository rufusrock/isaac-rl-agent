from __future__ import annotations

import argparse
import time

import cv2

from binding_rl_agent.window_capture import IsaacWindowCapture


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview a live capture stream from The Binding of Isaac."
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional substring to match the game window title.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=128,
        help="Preview width after resizing.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=128,
        help="Preview height after resizing.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Target preview FPS.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    capture = IsaacWindowCapture(title_substring=args.title)
    print(f"Capturing window: {capture.window.title}")
    print(f"Region: {capture.region}")
    print("Press 'q' in the preview window to exit.")

    frame_delay = max(1, int(1000 / max(args.fps, 1)))
    last_refresh = time.monotonic()

    while True:
        if time.monotonic() - last_refresh > 1.0:
            capture.refresh_region()
            last_refresh = time.monotonic()

        frame = capture.grab()
        preview = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_AREA)
        cv2.imshow("Isaac Live Capture", preview)
        if cv2.waitKey(frame_delay) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
