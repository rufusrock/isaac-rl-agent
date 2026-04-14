from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time
import warnings

import numpy as np

from binding_rl_agent.input_controller import hold_keys
from binding_rl_agent.preprocessing import (
    LIVE_FRAME_TRANSFORMS,
    resize_frame,
    stack_frames,
    to_grayscale,
    to_multichannel,
    to_rgb,
)
from binding_rl_agent.window_capture import IsaacWindowCapture


@dataclass(frozen=True)
class ObservationConfig:
    width: int = 128
    height: int = 128
    stack_size: int = 4
    grayscale: bool = True
    multichannel: bool = False  # equalized gray + bilateral (2 channels/frame, 2*stack_size total)
    color: bool = False          # raw RGB (3 channels/frame, 3*stack_size total; overrides grayscale/multichannel)
    frame_mode: str = ""         # when set, overrides grayscale/multichannel/color (use LIVE_FRAME_TRANSFORMS key)

    def __post_init__(self) -> None:
        for flag, default in (("grayscale", True), ("multichannel", False), ("color", False)):
            if getattr(self, flag) != default:
                warnings.warn(
                    f"ObservationConfig: the '{flag}' bool flag is deprecated and will be removed in a future version. "
                    f"Use 'frame_mode' instead (e.g. frame_mode='multichannel').",
                    DeprecationWarning,
                    stacklevel=2,
                )


@dataclass(frozen=True)
class IsaacAction:
    movement: int = 0
    shooting: int = 0
    bomb: int = 0


@dataclass(frozen=True)
class StepResult:
    observation: np.ndarray
    reward: float
    done: bool
    info: dict[str, object]


ACTION_MAP: dict[int, list[str]] = {
    0: [],
    1: ["w"],
    2: ["s"],
    3: ["a"],
    4: ["d"],
    5: ["up"],
    6: ["down"],
    7: ["left"],
    8: ["right"],
    9: ["e"],
}

ACTION_NAMES: dict[int, str] = {
    0: "idle",
    1: "move_up",
    2: "move_down",
    3: "move_left",
    4: "move_right",
    5: "shoot_up",
    6: "shoot_down",
    7: "shoot_left",
    8: "shoot_right",
    9: "bomb",
}

MOVEMENT_KEY_MAP: dict[int, list[str]] = {
    0: [],
    1: ["w"],
    2: ["s"],
    3: ["a"],
    4: ["d"],
}

SHOOTING_KEY_MAP: dict[int, list[str]] = {
    0: [],
    1: ["up"],
    2: ["down"],
    3: ["left"],
    4: ["right"],
}

BOMB_KEY_MAP: dict[int, list[str]] = {
    0: [],
    1: ["e"],
}


class IsaacFrameEnv:
    """Tiny observation wrapper to make the capture loop RL-friendly."""

    def __init__(
        self,
        title_substring: str | None = None,
        observation_config: ObservationConfig | None = None,
        action_hold_seconds: float = 0.08,
        post_action_wait_seconds: float = 0.04,
    ) -> None:
        self.capture = IsaacWindowCapture(title_substring=title_substring)
        self.config = observation_config or ObservationConfig()
        self.frames: deque[np.ndarray] = deque(maxlen=self.config.stack_size)
        self.action_hold_seconds = action_hold_seconds
        self.post_action_wait_seconds = post_action_wait_seconds
        self.last_raw_frame: np.ndarray | None = None
        self.last_processed_frame: np.ndarray | None = None

    def reset(self) -> np.ndarray:
        self.frames.clear()
        observation = self._capture_processed_frame()
        for _ in range(self.config.stack_size):
            self.frames.append(observation)
        return stack_frames(list(self.frames))

    def step(self, action: int | IsaacAction | None = None) -> np.ndarray:
        if action is not None:
            self.apply_action(action)
        observation = self._capture_processed_frame()
        self.frames.append(observation)
        return stack_frames(list(self.frames))

    def step_with_info(self, action: IsaacAction | None = None) -> StepResult:
        observation = self.step(action=action)
        return StepResult(
            observation=observation,
            reward=0.0,
            done=False,
            info={
                "movement": action.movement if action else 0,
                "shooting": action.shooting if action else 0,
                "bomb": action.bomb if action else 0,
            },
        )

    def apply_action(self, action: int | IsaacAction) -> None:
        if isinstance(action, IsaacAction):
            keys = keys_for_multihead_action(action)
        else:
            keys = ACTION_MAP.get(action)
            if keys is None:
                raise ValueError(f"Unknown discrete action id: {action}")
        if keys:
            hold_keys(keys, hold_seconds=self.action_hold_seconds)
        else:
            time.sleep(self.action_hold_seconds)
        time.sleep(self.post_action_wait_seconds)

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        resized = resize_frame(frame_bgr, self.config.width, self.config.height)
        if self.config.frame_mode:
            return LIVE_FRAME_TRANSFORMS[self.config.frame_mode](resized)
        if self.config.color:
            return to_rgb(resized)   # (3, H, W)
        if self.config.multichannel:
            return to_multichannel(resized)
        if self.config.grayscale:
            return to_grayscale(resized)
        return resized

    def _capture_processed_frame(self) -> np.ndarray:
        raw_frame = self.capture.grab()
        processed = self._preprocess(raw_frame)
        self.last_raw_frame = raw_frame
        self.last_processed_frame = processed
        return processed


def get_action_name(action: int) -> str:
    try:
        return ACTION_NAMES[action]
    except KeyError as error:
        raise ValueError(f"Unknown discrete action id: {action}") from error


def keys_for_multihead_action(action: IsaacAction) -> list[str]:
    try:
        movement_keys = MOVEMENT_KEY_MAP[action.movement]
        shooting_keys = SHOOTING_KEY_MAP[action.shooting]
        bomb_keys = BOMB_KEY_MAP[action.bomb]
    except KeyError as error:
        raise ValueError(f"Unknown multi-head action component: {action}") from error
    return movement_keys + shooting_keys + bomb_keys
