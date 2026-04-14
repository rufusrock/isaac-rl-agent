from __future__ import annotations

from dataclasses import dataclass, field

from binding_rl_agent.env import IsaacAction
from binding_rl_agent.game_state import IsaacGameState


@dataclass(frozen=True)
class TelemetryRewardConfig:
    time_penalty: float = -0.001
    transition_reward: float = 0.05
    rooms_explored_reward: float = 0.1
    room_clear_reward: float = 1.0
    damage_penalty: float = -0.2
    kill_reward: float = 0.02
    coin_reward: float = 0.01
    key_reward: float = 0.03
    collectible_reward: float = 0.25
    soul_heart_reward: float = 0.05
    black_heart_reward: float = 0.08
    stagnation_penalty: float = -0.02
    stagnation_window: int = 120
    room_timeout_steps: int = 900
    room_timeout_penalty: float = -0.5
    room_timeout_terminates: bool = True
    death_penalty: float = -1.0
    exploratory_death_penalty: float = -0.4
    recent_progress_window: int = 120


@dataclass(frozen=True)
class RewardSignal:
    reward: float
    done: bool
    info: dict[str, object]


@dataclass
class TelemetryRewardDetector:
    config: TelemetryRewardConfig = field(default_factory=TelemetryRewardConfig)
    previous_game_state: IsaacGameState | None = None
    stagnant_steps: int = 0
    room_steps: int = 0
    recent_progress_steps: int = 0
    transition_counts: dict = field(default_factory=dict)

    def reset(self) -> None:
        self.previous_game_state = None
        self.stagnant_steps = 0
        self.room_steps = 0
        self.recent_progress_steps = 0
        self.transition_counts = {}

    def update(
        self,
        game_state: IsaacGameState,
        action: IsaacAction | None = None,
    ) -> RewardSignal:
        reward = self.config.time_penalty
        previous_game_state = self.previous_game_state
        room_transition = False
        room_clear = False
        damage_taken = 0.0
        kills_gained = 0
        rooms_explored_gained = 0
        coins_gained = 0
        keys_gained = 0
        collectibles_gained = 0
        soul_hearts_delta = 0
        black_hearts_delta = 0
        death_candidate = False
        stagnation_penalty_applied = 0.0
        timeout_penalty_applied = 0.0
        death_penalty_applied = 0.0
        timeout_done = False

        if previous_game_state is not None:
            room_transition = game_state.room_index != previous_game_state.room_index
            room_clear = game_state.rooms_cleared > previous_game_state.rooms_cleared
            damage_taken = max(
                0.0,
                game_state.dmg_taken - previous_game_state.dmg_taken,
            )
            kills_gained = max(
                0,
                game_state.kills - previous_game_state.kills,
            )
            rooms_explored_gained = max(
                0,
                game_state.rooms_explored - previous_game_state.rooms_explored,
            )
            coins_gained = max(
                0,
                game_state.coins - previous_game_state.coins,
            )
            keys_gained = max(
                0,
                game_state.keys - previous_game_state.keys,
            )
            collectibles_gained = max(
                0,
                game_state.collectibles - previous_game_state.collectibles,
            )
            soul_hearts_delta = (
                game_state.soul_hearts - previous_game_state.soul_hearts
            )
            black_hearts_delta = (
                game_state.black_hearts - previous_game_state.black_hearts
            )
            death_candidate = game_state.deaths > previous_game_state.deaths

        if room_transition:
            self.room_steps = 0
        else:
            self.room_steps += 1

        meaningful_progress = any(
            value > 0
            for value in (
                kills_gained,
                rooms_explored_gained,
                coins_gained,
                keys_gained,
                collectibles_gained,
                soul_hearts_delta,
                black_hearts_delta,
            )
        ) or room_clear or room_transition
        if meaningful_progress:
            self.stagnant_steps = 0
            self.recent_progress_steps = 0
        else:
            self.stagnant_steps += 1
            self.recent_progress_steps += 1

        self.previous_game_state = game_state

        if room_transition and previous_game_state is not None:
            room_pair = (previous_game_state.room_index, game_state.room_index)
            count = self.transition_counts.get(room_pair, 0)
            # Decays linearly: full reward on first crossing, goes negative after ~5 repeats
            transition_r = self.config.transition_reward * (1.0 - 0.25 * count)
            transition_r = max(-self.config.transition_reward, transition_r)
            reward += transition_r
            self.transition_counts[room_pair] = count + 1
        if rooms_explored_gained > 0:
            reward += self.config.rooms_explored_reward * rooms_explored_gained
        if room_clear:
            reward += self.config.room_clear_reward
        if damage_taken > 0:
            reward += self.config.damage_penalty * damage_taken
        if kills_gained > 0:
            reward += self.config.kill_reward * kills_gained
        if coins_gained > 0:
            reward += self.config.coin_reward * coins_gained
        if keys_gained > 0:
            reward += self.config.key_reward * keys_gained
        if collectibles_gained > 0:
            reward += self.config.collectible_reward * collectibles_gained
        if soul_hearts_delta > 0:
            reward += self.config.soul_heart_reward * soul_hearts_delta
        if black_hearts_delta > 0:
            reward += self.config.black_heart_reward * black_hearts_delta
        if self.stagnant_steps >= self.config.stagnation_window:
            stagnation_penalty_applied = self.config.stagnation_penalty
            reward += stagnation_penalty_applied
        if self.room_steps >= self.config.room_timeout_steps:
            timeout_penalty_applied = self.config.room_timeout_penalty
            reward += timeout_penalty_applied
            timeout_done = self.config.room_timeout_terminates
        if death_candidate:
            if self.recent_progress_steps <= self.config.recent_progress_window:
                death_penalty_applied = self.config.exploratory_death_penalty
            else:
                death_penalty_applied = self.config.death_penalty
            reward += death_penalty_applied

        return RewardSignal(
            reward=reward,
            done=death_candidate or timeout_done,
            info={
                "room_transition": room_transition,
                "room_clear_candidate": room_clear,
                "death_candidate": death_candidate,
                "timeout_done": timeout_done,
                "room_steps": self.room_steps,
                "stagnant_steps": self.stagnant_steps,
                "stagnation_penalty_applied": stagnation_penalty_applied,
                "timeout_penalty_applied": timeout_penalty_applied,
                "death_penalty_applied": death_penalty_applied,
                "recent_progress_steps": self.recent_progress_steps,
                "damage_taken": damage_taken,
                "kills_gained": kills_gained,
                "rooms_explored_gained": rooms_explored_gained,
                "coins_gained": coins_gained,
                "keys_gained": keys_gained,
                "collectibles_gained": collectibles_gained,
                "soul_hearts_delta": soul_hearts_delta,
                "black_hearts_delta": black_hearts_delta,
                "telemetry_used": True,
                "game_state": game_state,
                "action": action,
            },
        )
