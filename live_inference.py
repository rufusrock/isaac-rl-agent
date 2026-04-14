from __future__ import annotations

import argparse
import time

import cv2

from binding_rl_agent.env import IsaacFrameEnv, ObservationConfig
from binding_rl_agent.inference import find_latest_model, load_policy_checkpoint, predict_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview live Isaac policy predictions without sending any inputs."
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional checkpoint path. Defaults to the latest trained model.",
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
        help="Observation width for inference.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=128,
        help="Observation height for inference.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Target preview FPS.",
    )
    parser.add_argument(
        "--preview-scale",
        type=int,
        default=4,
        help="Nearest-neighbor scale factor for the preview window.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model_path or str(find_latest_model())
    model, device, checkpoint = load_policy_checkpoint(model_path)

    env = IsaacFrameEnv(
        title_substring=args.title,
        observation_config=ObservationConfig(
            width=args.width,
            height=args.height,
            stack_size=4,
            grayscale=True,
        ),
    )

    observation = env.reset()
    print(f"Loaded model: {model_path}")
    print(f"Using device: {device}")
    print(f"Capturing window: {env.capture.window.title}")
    print("Press 'q' in the preview window to exit.")

    frame_delay = 1.0 / max(args.fps, 1)
    next_tick = time.monotonic()

    while True:
        observation = env.step(action=None)
        prediction = predict_policy(model, device, observation, checkpoint=checkpoint)

        latest_frame = observation[-1]
        preview = cv2.cvtColor(latest_frame, cv2.COLOR_GRAY2BGR)
        if args.preview_scale != 1:
            preview = cv2.resize(
                preview,
                (preview.shape[1] * args.preview_scale, preview.shape[0] * args.preview_scale),
                interpolation=cv2.INTER_NEAREST,
            )

        _draw_prediction_overlay(preview, model_path, prediction)
        cv2.imshow("Isaac Live Inference", preview)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        next_tick += frame_delay
        sleep_for = next_tick - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)

    cv2.destroyAllWindows()


def _draw_prediction_overlay(frame, model_path: str, prediction) -> None:
    overlay_lines = [
        f"model: {model_path}",
        f"device: {prediction.device}",
        (
            f"move: {prediction.movement.label} "
            f"({prediction.movement.confidence:.2f})"
        ),
        (
            f"shoot: {prediction.shooting.label} "
            f"({prediction.shooting.confidence:.2f})"
        ),
        (
            f"bomb: {prediction.bomb.label} "
            f"({prediction.bomb.confidence:.2f})"
        ),
        "q: quit",
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    for idx, line in enumerate(overlay_lines):
        y = 20 + idx * 22
        cv2.putText(frame, line, (8, y), font, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (8, y), font, 0.5, (245, 245, 245), 1, cv2.LINE_AA)


if __name__ == "__main__":
    main()
