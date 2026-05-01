from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from binding_rl_agent.env import IsaacAction, IsaacFrameEnv, ObservationConfig
from binding_rl_agent.game_state import IsaacGameState, IsaacUDPGameStateReceiver
from binding_rl_agent.reward_detection import RewardSignal, TelemetryRewardConfig, TelemetryRewardDetector
from binding_rl_agent.room_graph import RoomGraph


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
    # Match BC's 20 Hz training/recording rate (50 ms per step). The BC was
    # trained on 50 ms intervals and its motion channels assume that cadence;
    # running at 8 Hz fed it observations 2.4x more spread-out than expected.
    action_hold_seconds: float = 0.05
    post_action_wait_seconds: float = 0.0
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
        self.current_nav_hint: int = 0

    def reset(self, telemetry_timeout_seconds: float = 5.0) -> np.ndarray:
        self.reward_detector.reset()
        observation = self.frame_env.reset()
        self._wait_for_game_state(telemetry_timeout_seconds)
        # Let any in-flight post-restart packets arrive, then drain the socket
        # to use the FRESHEST state as the new baseline. Without this, deltas
        # from a still-pending counter increment (e.g. R-restart bumping deaths)
        # leak into the next episode and fire phantom done events.
        time.sleep(1.0)
        drained = self.game_state_receiver.drain()
        game_state = drained if drained is not None else self._require_game_state()
        self.reward_detector.previous_game_state = game_state
        self.current_nav_hint = self._compute_nav_hint(game_state)
        return observation

    def _wait_for_game_state(self, timeout_seconds: float) -> IsaacGameState:
        deadline = time.monotonic() + max(timeout_seconds, 0.0)
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                return self._require_game_state()
            except RuntimeError as exc:
                last_error = exc
                time.sleep(0.1)
        # One last try so the error surfaces with a clean traceback.
        if last_error is not None:
            return self._require_game_state()
        return self._require_game_state()

    def step(self, action: IsaacAction | None = None) -> IsaacRLStep:
        observation = self.frame_env.step(action=action)
        game_state = self._require_game_state()
        nav_hint = self._compute_nav_hint(game_state)
        reward_signal: RewardSignal = self.reward_detector.update(
            game_state, action=action, nav_hint=nav_hint
        )
        info = {
            "movement": action.movement if action else 0,
            "shooting": action.shooting if action else 0,
            "bomb": action.bomb if action else 0,
            "nav_hint": nav_hint,
            **reward_signal.info,
        }
        self.current_nav_hint = nav_hint
        return IsaacRLStep(
            observation=observation,
            reward=reward_signal.reward,
            done=reward_signal.done,
            info=info,
        )

    def _compute_nav_hint(self, game_state: IsaacGameState) -> int:
        if not game_state.floor_rooms:
            return 0
        try:
            return int(RoomGraph.from_game_state(game_state).nav_hint(game_state.room_index))
        except Exception:
            return 0

    def _require_game_state(self) -> IsaacGameState:
        # Drain rather than read one packet — the Lua mod sends at ~60Hz while we
        # step at ~10-15Hz, so without draining we accumulate a 40+ packet/sec
        # backlog and increasingly read stale state. Drain returns the most
        # recent packet, which is what an RL agent should react to.
        game_state = self.game_state_receiver.drain()
        if game_state is None:
            raise RuntimeError(
                "No Isaac telemetry received on UDP port "
                f"{self.telemetry_port}. Start the game-side telemetry sender first."
            )
        return game_state
