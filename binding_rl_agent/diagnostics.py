from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from binding_rl_agent.env import IsaacFrameEnv, ObservationConfig
from binding_rl_agent.preprocessing import resize_frame, to_grayscale
from binding_rl_agent.window_capture import IsaacWindowCapture


def save_diagnostic_images(
    output_dir: str | Path,
    title_substring: str | None = None,
    width: int = 84,
    height: int = 84,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    capture = IsaacWindowCapture(title_substring=title_substring)
    raw_frame = capture.grab()
    resized_frame = resize_frame(raw_frame, width, height)
    grayscale_frame = to_grayscale(resized_frame)

    env = IsaacFrameEnv(
        title_substring=title_substring,
        observation_config=ObservationConfig(width=width, height=height, stack_size=4),
    )
    stacked = env.reset()
    stacked_preview = _build_stack_preview(stacked)

    outputs = {
        "raw_capture": output_path / "raw_capture.png",
        "resized_capture": output_path / "resized_capture.png",
        "grayscale_capture": output_path / "grayscale_capture.png",
        "stacked_frames_preview": output_path / "stacked_frames_preview.png",
    }

    cv2.imwrite(str(outputs["raw_capture"]), raw_frame)
    cv2.imwrite(str(outputs["resized_capture"]), resized_frame)
    cv2.imwrite(str(outputs["grayscale_capture"]), grayscale_frame)
    cv2.imwrite(str(outputs["stacked_frames_preview"]), stacked_preview)

    return outputs


def _build_stack_preview(stacked_frames: np.ndarray) -> np.ndarray:
    if stacked_frames.ndim != 3:
        raise ValueError(
            "Expected stacked grayscale frames with shape (stack, height, width)."
        )
    tiles = [frame for frame in stacked_frames]
    return np.concatenate(tiles, axis=1)
