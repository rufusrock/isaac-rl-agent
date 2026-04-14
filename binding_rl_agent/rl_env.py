from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from binding_rl_agent.env import IsaacAction, IsaacFrameEnv, ObservationConfig
from binding_rl_agent.game_state import IsaacGameState, IsaacUDPGameStateReceiver
from binding_rl_agent.reward_detection import RewardSignal, TelemetryRewardConfig, TelemetryRewardDetector


@dataclass(frozen=True)
class IsaacRLStep:
    observation: np.ndarray
    reward: float
    done: bool
    info: dict[str, object]


@dataclass
class IsaacVisualRLEnv:
    title_substring: str | None = None
    observation_config: ObservationConfig = field(default_factory=ObservationConfig)
    reward_config: TelemetryRewardConfig = field(default_factory=TelemetryRewardConfig)
    action_hold_seconds: float = 0.08
    post_action_wait_seconds: float = 0.04
    telemetry_port: int = 8123

    def __post_init__(self) -> None:
        self.frame_env = IsaacFrameEnv(
            title_substring=self.title_substring,
            observation_config=self.observation_config,
            action_hold_seconds=self.action_hold_seconds,
            post_action_wait_seconds=self.post_action_wait_seconds,
        )
        self.reward_detector = TelemetryRewardDetector(config=self.reward_config)
        self.game_state_receiver = IsaacUDPGameStateReceiver(port=self.telemetry_port)

    def reset(self) -> np.ndarray:
        self.reward_detector.reset()
        observation = self.frame_env.reset()
        self._require_game_state()
        return observation

    def step(self, action: IsaacAction | None = None) -> IsaacRLStep:
        observation = self.frame_env.step(action=action)
        game_state = self._require_game_state()
        reward_signal: RewardSignal = self.reward_detector.update(game_state, action=action)
        info = {
            "movement": action.movement if action else 0,
            "shooting": action.shooting if action else 0,
            "bomb": action.bomb if action else 0,
            **reward_signal.info,
        }
        return IsaacRLStep(
            observation=observation,
            reward=reward_signal.reward,
            done=reward_signal.done,
            info=info,
        )

    def _require_game_state(self) -> IsaacGameState:
        game_state = self.game_state_receiver.get_latest()
        if game_state is None:
            raise RuntimeError(
                "No Isaac telemetry received on UDP port "
                f"{self.telemetry_port}. Start the game-side telemetry sender first."
            )
        return game_state
