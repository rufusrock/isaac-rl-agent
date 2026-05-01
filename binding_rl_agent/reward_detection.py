from __future__ import annotations

from dataclasses import dataclass, field

from binding_rl_agent.env import IsaacAction
from binding_rl_agent.game_state import IsaacGameState


@dataclass(frozen=True)
class TelemetryRewardConfig:
    time_penalty: float = 0.0
    transition_reward: float = 0.1
    rooms_explored_reward: float = 1.0
    room_clear_reward: float = 2.0
    damage_penalty: float = -0.1
    kill_reward: float = 0.05
    coin_reward: float = 0.01
    key_reward: float = 0.05
    collectible_reward: float = 0.5
    soul_heart_reward: float = 0.1
    black_heart_reward: float = 0.15
    stagnation_penalty: float = 0.0
    stagnation_window: int = 90
    room_timeout_steps: int = 300
    room_timeout_penalty: float = 0.0
    room_timeout_terminates: bool = True
    # STAY-room (combat) idle timeout: terminate the episode if the agent
    # spends this many steps in a STAY room without taking damage or scoring
    # a kill. 2400 steps @ 20 Hz = 2 minutes. Targets the freeze attractor
    # where the agent stands still in a combat room and gets nothing.
    stay_idle_timeout_steps: int = 2400
    stay_idle_timeout_penalty: float = 0.0
    stay_idle_timeout_terminates: bool = True
    # When the previous episode ended at max_steps (no terminal event), the
    # agent is unattended during the PPO update (~1-2s) and may take damage
    # or die from sitting idle in combat. We give the agent a short grace
    # period at the start of the next episode where damage/death penalties
    # are zeroed (the episode still ends naturally on death so the game
    # auto-resets cleanly).
    post_update_grace_steps: int = 30
    death_penalty: float = -0.5
    exploratory_death_penalty: float = -0.25
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
    stay_idle_steps: int = 0
    recent_progress_steps: int = 0
    grace_steps_remaining: int = 0
    transition_counts: dict = field(default_factory=dict)

    def reset(self) -> None:
        self.previous_game_state = None
        self.stagnant_steps = 0
        self.room_steps = 0
        self.stay_idle_steps = 0
        self.recent_progress_steps = 0
        self.grace_steps_remaining = 0
        self.transition_counts = {}

    def update(
        self,
        game_state: IsaacGameState,
        action: IsaacAction | None = None,
        nav_hint: int | None = None,
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
        stay_idle_timeout_penalty_applied = 0.0
        death_penalty_applied = 0.0
        timeout_done = False
        stay_idle_timeout_done = False
        # Grace period: zero out damage and death penalties for the first N
        # steps after the agent resumes from a max_steps PPO-update pause.
        grace_active = self.grace_steps_remaining > 0
        if grace_active:
            self.grace_steps_remaining -= 1

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

        # Only count steps toward the room timeout in navigation rooms (nav_hint != STAY).
        # STAY rooms include uncleared/combat rooms where the agent SHOULD linger.
        # When nav_hint is unavailable, fall back to the legacy "always count" behavior.
        is_navigation_room = nav_hint is not None and nav_hint != 0
        is_stay_room = nav_hint is not None and nav_hint == 0
        if room_transition:
            self.room_steps = 0
        elif nav_hint is None or is_navigation_room:
            self.room_steps += 1
        else:
            self.room_steps = 0

        # STAY-room idle timeout: count steps where the agent is in a STAY
        # (combat) room AND is neither taking damage nor scoring kills. Reset
        # on engagement events or on leaving the STAY room.
        engaged_this_step = (kills_gained > 0) or (damage_taken > 0)
        if room_transition or not is_stay_room or engaged_this_step:
            self.stay_idle_steps = 0
        else:
            self.stay_idle_steps += 1

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
        if damage_taken > 0 and not grace_active:
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
        if self.stay_idle_steps >= self.config.stay_idle_timeout_steps:
            stay_idle_timeout_penalty_applied = self.config.stay_idle_timeout_penalty
            reward += stay_idle_timeout_penalty_applied
            stay_idle_timeout_done = self.config.stay_idle_timeout_terminates
        if death_candidate:
            if grace_active:
                # Death within grace window after max_steps — likely caused by
                # the agent being unattended during the PPO update. Let the
                # episode end (game auto-resets) but apply zero penalty.
                death_penalty_applied = 0.0
            elif self.recent_progress_steps <= self.config.recent_progress_window:
                death_penalty_applied = self.config.exploratory_death_penalty
            else:
                death_penalty_applied = self.config.death_penalty
            reward += death_penalty_applied

        return RewardSignal(
            reward=reward,
            done=death_candidate or timeout_done or stay_idle_timeout_done,
            info={
                "room_transition": room_transition,
                "room_clear_candidate": room_clear,
                "death_candidate": death_candidate,
                "timeout_done": timeout_done,
                "stay_idle_timeout_done": stay_idle_timeout_done,
                "grace_active": grace_active,
                "room_steps": self.room_steps,
                "stay_idle_steps": self.stay_idle_steps,
                "stagnant_steps": self.stagnant_steps,
                "stagnation_penalty_applied": stagnation_penalty_applied,
                "timeout_penalty_applied": timeout_penalty_applied,
                "stay_idle_timeout_penalty_applied": stay_idle_timeout_penalty_applied,
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
                "nav_hint": nav_hint,
            },
        )
