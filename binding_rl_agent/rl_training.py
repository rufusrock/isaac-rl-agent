from __future__ import annotations

from collections import deque
import json
import math
import random
import time
from dataclasses import asdict, dataclass, replace
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn

from binding_rl_agent.dataset import BOMB_NAMES, MOVEMENT_NAMES, SHOOTING_NAMES
from binding_rl_agent.env import IsaacAction, ObservationConfig
from binding_rl_agent.inference import frame_size_from_checkpoint, obs_config_from_checkpoint
from binding_rl_agent.input_controller import hold_keys, release_all_agent_keys, tap_key
from binding_rl_agent.models import IsaacCNNPolicy
from binding_rl_agent.reward_detection import TelemetryRewardConfig
from binding_rl_agent.rl_diagnostics import build_rollout_diagnostics, save_rollout_diagnostics
from binding_rl_agent.rl_env import IsaacVisualRLEnv


@dataclass(frozen=True)
class RLTrainConfig:
    title_substring: str | None = None
    telemetry_port: int = 8123
    warmup_seconds: float = 5.0
    total_updates: int = 100
    rollout_steps: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_epochs: int = 2
    ppo_minibatch_size: int = 32
    ppo_clip_epsilon: float = 0.2
    learning_rate: float = 5.0e-5
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    output_dir: str = "rl_runs"
    pretrained_model_path: str | None = None
    prefer_latest_rl_best: bool = False
    device: str | None = None
    auto_reset: bool = False
    deterministic_actions: bool = False
    adaptive_exploration: bool = True
    exploration_stagnation_threshold: int = 30
    exploration_temperature: float = 1.5
    shooting_exploration_temperature: float = 1.2
    movement_action_repeat: int = 1
    shooting_action_repeat: int = 1
    anti_stuck_direction_penalty: float = 2.0
    anti_stuck_trigger_steps: int = 45
    bc_anchor_coef: float = 0.15
    bc_anchor_movement_weight: float = 1.0
    bc_anchor_shooting_weight: float = 2.0
    bc_anchor_bomb_weight: float = 0.5
    early_stop_patience: int = 20
    early_stop_min_delta: float = 0.0
    best_checkpoint_window_episodes: int = 50
    best_checkpoint_min_episodes: int = 20
    diagnostics_sample_frames: int = 16
    seed: int = 7
    reward_config: TelemetryRewardConfig = TelemetryRewardConfig(
        time_penalty=-0.001,
        transition_reward=0.05,
        rooms_explored_reward=1.0,
        room_clear_reward=2.5,
        damage_penalty=-0.5,
        kill_reward=0.1,
        coin_reward=0.01,
        key_reward=0.05,
        collectible_reward=0.5,
        soul_heart_reward=0.1,
        black_heart_reward=0.15,
        stagnation_penalty=-0.02,
        stagnation_window=60,
        room_timeout_steps=400,
        room_timeout_penalty=-1.5,
        death_penalty=-0.5,
        exploratory_death_penalty=-0.25,
        recent_progress_window=120,
    )


class IsaacActorCritic(nn.Module):
    def __init__(
        self,
        input_channels: int = 4,
        num_movement_actions: int = 5,
        num_shooting_actions: int = 5,
        num_bomb_actions: int = 2,
    ) -> None:
        super().__init__()
        self.policy = IsaacCNNPolicy(
            input_channels=input_channels,
            num_movement_actions=num_movement_actions,
            num_shooting_actions=num_shooting_actions,
            num_bomb_actions=num_bomb_actions,
        )
        # _trunk_linear[0] is the Linear(flat_size, hidden_dim) layer; read out_features
        # rather than hardcoding 512 so any hidden_dim works correctly.
        hidden_dim = self.policy._trunk_linear[0].out_features
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, observations: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        shared_features = self.policy.features(observations)
        flat = self.policy._flatten(shared_features)
        shared_hidden = self.policy._trunk_linear(flat)
        logits = {
            "movement": self.policy.movement_head(shared_hidden),
            "shooting": self.policy.shooting_head(shared_hidden),
            "bomb": self.policy.bomb_head(shared_hidden),
        }
        value = self.value_head(shared_hidden).squeeze(-1)
        return logits, value


def train_actor_critic(config: RLTrainConfig) -> Path:
    _seed_everything(config.seed)
    if config.deterministic_actions:
        print("Ignoring deterministic_actions=True during PPO training; sampling must stay stochastic.")
        config = replace(config, deterministic_actions=False)
    if config.movement_action_repeat != 1 or config.shooting_action_repeat != 1:
        print("Disabling action repeats during PPO training to preserve on-policy updates.")
        config = replace(config, movement_action_repeat=1, shooting_action_repeat=1)

    # Resolve pretrained checkpoint first so we can derive observation config from it.
    resolved_pretrained = _resolve_pretrained_model_path(
        explicit_path=config.pretrained_model_path,
        prefer_latest_rl_best=config.prefer_latest_rl_best,
    )

    # Build ObservationConfig from the BC checkpoint when available, so the RL env
    # uses the same frame_mode, stack_size, and resolution that the policy was
    # trained on.  Fall back to sensible defaults for from-scratch runs.
    if resolved_pretrained:
        _device_for_load = torch.device(
            config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        try:
            _ckpt = torch.load(resolved_pretrained, map_location=_device_for_load, weights_only=True)
        except TypeError:
            _ckpt = torch.load(resolved_pretrained, map_location=_device_for_load)
        obs_kwargs = obs_config_from_checkpoint(_ckpt)
        frame_size = frame_size_from_checkpoint(_ckpt)
        print(
            f"Observation config from checkpoint: "
            f"frame_mode={obs_kwargs.get('frame_mode')!r} "
            f"stack_size={obs_kwargs.get('stack_size')} "
            f"frame_size={frame_size}"
        )
    else:
        obs_kwargs = dict(frame_mode="multichannel", stack_size=4)
        frame_size = 128

    obs_config = ObservationConfig(
        width=frame_size, height=frame_size, **obs_kwargs,
    )

    env = IsaacVisualRLEnv(
        title_substring=config.title_substring,
        observation_config=obs_config,
        reward_config=config.reward_config,
        telemetry_port=config.telemetry_port,
    )
    print(f"Capturing window: {env.frame_env.capture.window.title}")
    observation = env.frame_env.reset()
    _wait_for_telemetry(env, config.warmup_seconds)

    device = torch.device(
        config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = IsaacActorCritic(input_channels=int(observation.shape[0]))
    if resolved_pretrained:
        print(f"Loading pretrained checkpoint: {resolved_pretrained}")
        _load_pretrained_weights(model, resolved_pretrained, device)
    else:
        print("No pretrained checkpoint supplied. RL will start from random weights.")
    reference_policy = deepcopy(model.policy).to(device)
    reference_policy.eval()
    for parameter in reference_policy.parameters():
        parameter.requires_grad_(False)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    run_dir = Path(config.output_dir) / time.strftime("ppo_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = run_dir / "metrics.jsonl"
    checkpoint_path = run_dir / "actor_critic.pt"
    best_checkpoint_path = run_dir / "best_actor_critic.pt"
    diagnostics_dir = run_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    best_window_score = -math.inf
    updates_without_improvement = 0
    recent_episode_window: deque[dict[str, float | bool]] = deque(
        maxlen=config.best_checkpoint_window_episodes
    )

    for update_idx in range(1, config.total_updates + 1):
        model.eval()
        rollout = _collect_rollout(env, model, device, observation, config)
        observation = rollout["next_observation"]

        model.train()
        losses = _update_model(
            model=model,
            reference_policy=reference_policy,
            optimizer=optimizer,
            rollout=rollout,
            device=device,
            config=config,
        )
        diagnostics = build_rollout_diagnostics(
            rewards=np.asarray(rollout["rewards"], dtype=np.float32),
            dones=np.asarray(rollout["dones"], dtype=np.float32),
            actions=list(rollout["actions"]),
            infos=list(rollout["infos"]),
        )
        for episode in diagnostics["episodes"]:
            recent_episode_window.append(
                {
                    "exited_first_room": bool(int(episode.get("rooms_explored_gained", 0)) > 0),
                    "cleared_room": bool(int(episode.get("room_clears", 0)) > 0),
                    "rooms_explored_gained": float(episode.get("rooms_explored_gained", 0)),
                    "reward_sum": float(episode.get("reward_sum", 0.0)),
                }
            )
        recent_window_metrics = _summarize_recent_episode_window(recent_episode_window)
        recent_window_score = _score_recent_episode_window(recent_window_metrics)
        summary = {
            "update": update_idx,
            "mean_reward": float(np.mean(rollout["rewards"])),
            "sum_reward": float(np.sum(rollout["rewards"])),
            "episodes_done": int(np.sum(rollout["dones"])),
            **diagnostics["events"],
            **diagnostics["behavior"],
            "episode_summaries": diagnostics["episodes"],
            "action_counts": diagnostics["action_counts"],
            "recent_window_episodes": int(recent_window_metrics["episodes"]),
            "recent_window_exit_rate": float(recent_window_metrics["exit_rate"]),
            "recent_window_clear_rate": float(recent_window_metrics["clear_rate"]),
            "recent_window_avg_rooms": float(recent_window_metrics["avg_rooms"]),
            "recent_window_avg_reward": float(recent_window_metrics["avg_reward"]),
            **losses,
        }
        print(
            f"update={update_idx:04d} reward_mean={summary['mean_reward']:.4f} "
            f"reward_sum={summary['sum_reward']:.4f} value_loss={summary['value_loss']:.4f} "
            f"policy_loss={summary['policy_loss']:.4f} entropy={summary['entropy']:.4f} "
            f"rooms+={summary['rooms_explored_gained']} kills+={summary['kills_gained']} "
            f"window_exit={summary['recent_window_exit_rate']:.3f} "
            f"stagnant_max={summary['max_stagnant_steps']}"
        )
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary) + "\n")
        save_rollout_diagnostics(
            diagnostics_dir=diagnostics_dir,
            update_idx=update_idx,
            rollout=rollout,
            summary=summary,
            sample_frames=config.diagnostics_sample_frames,
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": asdict(config),
                "movement_names": MOVEMENT_NAMES,
                "shooting_names": SHOOTING_NAMES,
                "bomb_names": BOMB_NAMES,
            },
            checkpoint_path,
        )

        if recent_window_metrics["episodes"] >= config.best_checkpoint_min_episodes:
            improvement = recent_window_score - best_window_score
            if improvement > config.early_stop_min_delta:
                best_window_score = recent_window_score
                updates_without_improvement = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": asdict(config),
                        "movement_names": MOVEMENT_NAMES,
                        "shooting_names": SHOOTING_NAMES,
                        "bomb_names": BOMB_NAMES,
                        "best_update": update_idx,
                        "best_window_score": best_window_score,
                        "best_window_metrics": recent_window_metrics,
                    },
                    best_checkpoint_path,
                )
                print(
                    f"new_best update={update_idx:04d} "
                    f"exit_rate={recent_window_metrics['exit_rate']:.3f} "
                    f"clear_rate={recent_window_metrics['clear_rate']:.3f} "
                    f"avg_rooms={recent_window_metrics['avg_rooms']:.3f} "
                    f"checkpoint={best_checkpoint_path.name}"
                )
            else:
                updates_without_improvement += 1
            if updates_without_improvement >= config.early_stop_patience:
                print(
                    f"early_stop update={update_idx:04d} "
                    f"best_window_score={best_window_score:.4f} "
                    f"patience={config.early_stop_patience}"
                )
                break
        else:
            updates_without_improvement = 0

    release_all_agent_keys()
    return run_dir


def _wait_for_telemetry(env: IsaacVisualRLEnv, warmup_seconds: float) -> None:
    print(
        f"Waiting for telemetry on UDP port {env.telemetry_port} "
        f"for up to {warmup_seconds:.1f}s."
    )
    deadline = time.monotonic() + max(warmup_seconds, 0.0)
    while time.monotonic() < deadline:
        game_state = env.game_state_receiver.get_latest()
        if game_state is not None:
            env.reward_detector.reset()
            env.reward_detector.previous_game_state = game_state
            print("Telemetry connected. Starting RL.")
            return
        time.sleep(0.1)
    raise RuntimeError(
        "No Isaac telemetry received on UDP port "
        f"{env.telemetry_port} within {warmup_seconds:.1f}s."
    )


def _collect_rollout(
    env: IsaacVisualRLEnv,
    model: IsaacActorCritic,
    device: torch.device,
    observation: np.ndarray,
    config: RLTrainConfig,
) -> dict[str, object]:
    observations: list[np.ndarray] = []
    actions: list[IsaacAction] = []
    rewards: list[float] = []
    dones: list[bool] = []
    log_probs: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    infos: list[dict[str, object]] = []
    stagnant_steps_before_action: list[int] = []
    last_movement_before_action: list[int] = []
    movement_temperatures: list[float] = []
    shooting_temperatures: list[float] = []
    bomb_temperatures: list[float] = []

    current_observation = observation
    current_stagnant_steps = 0
    repeated_movement: int | None = None
    repeated_shooting: int | None = None
    movement_repeat_steps_remaining = 0
    shooting_repeat_steps_remaining = 0
    last_applied_movement = 0
    for _ in range(config.rollout_steps):
        observation_tensor = (
            torch.from_numpy(current_observation).float().unsqueeze(0).to(device) / 255.0
        )
        logits, value = model(observation_tensor)
        exploratory_mode = (
            config.adaptive_exploration
            and current_stagnant_steps >= config.exploration_stagnation_threshold
        )
        movement_temperature = config.exploration_temperature if exploratory_mode else 1.0
        shooting_temperature = (
            config.shooting_exploration_temperature if exploratory_mode else 1.0
        )
        bomb_temperature = 1.0
        previous_movement = last_applied_movement
        adjusted_logits = _apply_anti_stuck_bias(
            logits=logits,
            stagnant_steps=current_stagnant_steps,
            last_movement=previous_movement,
            trigger_steps=config.anti_stuck_trigger_steps,
            direction_penalty=config.anti_stuck_direction_penalty,
        )
        sampled_action, behavior_policy_log_probs, behavior_policy_entropy = _sample_action(
            sampling_logits=adjusted_logits,
            deterministic=config.deterministic_actions and not exploratory_mode,
            movement_temperature=movement_temperature,
            shooting_temperature=shooting_temperature,
            bomb_temperature=bomb_temperature,
        )
        if repeated_movement is not None and movement_repeat_steps_remaining > 0:
            movement_to_apply = repeated_movement
            movement_repeat_steps_remaining -= 1
            movement_repeated = True
        else:
            movement_to_apply = sampled_action.movement
            repeated_movement = sampled_action.movement
            movement_repeat_steps_remaining = max(config.movement_action_repeat - 1, 0)
            movement_repeated = False

        if repeated_shooting is not None and shooting_repeat_steps_remaining > 0:
            shooting_to_apply = repeated_shooting
            shooting_repeat_steps_remaining -= 1
        else:
            shooting_to_apply = sampled_action.shooting
            repeated_shooting = sampled_action.shooting
            shooting_repeat_steps_remaining = max(config.shooting_action_repeat - 1, 0)

        action_to_apply = IsaacAction(
            movement=movement_to_apply,
            shooting=shooting_to_apply,
            bomb=sampled_action.bomb,
        )
        behavior_policy_log_prob = _action_log_prob(behavior_policy_log_probs, action_to_apply)

        step = env.step(action_to_apply)
        next_observation = step.observation
        last_applied_movement = action_to_apply.movement

        observations.append(current_observation.copy())
        actions.append(action_to_apply)
        rewards.append(step.reward)
        dones.append(step.done)
        log_probs.append(behavior_policy_log_prob)
        values.append(value.squeeze(0))
        entropies.append(behavior_policy_entropy)
        stagnant_steps_before_action.append(current_stagnant_steps)
        last_movement_before_action.append(previous_movement)
        movement_temperatures.append(float(movement_temperature))
        shooting_temperatures.append(float(shooting_temperature))
        bomb_temperatures.append(float(bomb_temperature))
        infos.append(
            {
                **step.info,
                "exploratory_mode": exploratory_mode,
                "movement_repeated": movement_repeated,
                "stagnant_steps_before_action": current_stagnant_steps,
                "last_movement_before_action": previous_movement,
                "movement_temperature": float(movement_temperature),
                "shooting_temperature": float(shooting_temperature),
                "anti_stuck_bias_applied": (
                    current_stagnant_steps >= config.anti_stuck_trigger_steps
                    and previous_movement != 0
                ),
            }
        )

        current_observation = next_observation
        current_stagnant_steps = int(step.info.get("stagnant_steps", 0))
        if step.done:
            if config.auto_reset:
                _auto_reset_game()
            current_observation = env.reset()
            current_stagnant_steps = 0
            repeated_movement = None
            repeated_shooting = None
            movement_repeat_steps_remaining = 0
            shooting_repeat_steps_remaining = 0
            last_applied_movement = 0

    next_obs_tensor = (
        torch.from_numpy(current_observation).float().unsqueeze(0).to(device) / 255.0
    )
    with torch.no_grad():
        _, next_value = model(next_obs_tensor)

    return {
        "observations": np.stack(observations, axis=0),
        "actions": actions,
        "rewards": np.asarray(rewards, dtype=np.float32),
        "dones": np.asarray(dones, dtype=np.float32),
        "log_probs": torch.stack(log_probs),
        "values": torch.stack(values),
        "entropies": torch.stack(entropies),
        "infos": infos,
        "stagnant_steps_before_action": np.asarray(stagnant_steps_before_action, dtype=np.int32),
        "last_movement_before_action": np.asarray(last_movement_before_action, dtype=np.int64),
        "movement_temperatures": np.asarray(movement_temperatures, dtype=np.float32),
        "shooting_temperatures": np.asarray(shooting_temperatures, dtype=np.float32),
        "bomb_temperatures": np.asarray(bomb_temperatures, dtype=np.float32),
        "next_value": next_value.squeeze(0),
        "next_observation": current_observation,
    }


def _update_model(
    model: IsaacActorCritic,
    reference_policy: IsaacCNNPolicy,
    optimizer: torch.optim.Optimizer,
    rollout: dict[str, object],
    device: torch.device,
    config: RLTrainConfig,
) -> dict[str, float]:
    rewards = rollout["rewards"]
    dones = rollout["dones"]
    values = rollout["values"]
    next_value = rollout["next_value"].detach()

    advantages, returns = _compute_gae(
        rewards=rewards,
        dones=dones,
        values=values.detach().cpu().numpy(),
        next_value=float(next_value.cpu().item()),
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
    )
    advantages_tensor = torch.from_numpy(advantages).float().to(device)
    returns_tensor = torch.from_numpy(returns).float().to(device)
    adv_raw_mean = float(advantages_tensor.mean().item())
    adv_raw_std = float(advantages_tensor.std(unbiased=False).item())
    adv_norm_skipped = adv_raw_std < 0.05
    advantages_tensor = _normalize_advantages(advantages_tensor)
    adv_norm_mean = float(advantages_tensor.mean().item())
    adv_norm_std = float(advantages_tensor.std(unbiased=False).item())
    observations_tensor = (
        torch.from_numpy(np.asarray(rollout["observations"], dtype=np.uint8)).float().to(device) / 255.0
    )

    old_log_probs_tensor = rollout["log_probs"].detach().to(device)
    log_prob_mean = float(old_log_probs_tensor.mean().item())
    log_prob_std = float(old_log_probs_tensor.std(unbiased=False).item())

    actions = list(rollout["actions"])
    movement_actions = torch.tensor(
        [action.movement for action in actions],
        dtype=torch.long,
        device=device,
    )
    shooting_actions = torch.tensor(
        [action.shooting for action in actions],
        dtype=torch.long,
        device=device,
    )
    bomb_actions = torch.tensor(
        [action.bomb for action in actions],
        dtype=torch.long,
        device=device,
    )
    stagnant_steps_tensor = torch.from_numpy(rollout["stagnant_steps_before_action"]).to(device=device)
    last_movement_tensor = torch.from_numpy(rollout["last_movement_before_action"]).to(device=device)
    movement_temperature_tensor = torch.from_numpy(rollout["movement_temperatures"]).to(device=device)
    shooting_temperature_tensor = torch.from_numpy(rollout["shooting_temperatures"]).to(device=device)
    bomb_temperature_tensor = torch.from_numpy(rollout["bomb_temperatures"]).to(device=device)

    batch_size = observations_tensor.shape[0]
    minibatch_size = max(1, min(config.ppo_minibatch_size, batch_size))
    policy_loss_values: list[float] = []
    value_loss_values: list[float] = []
    entropy_values: list[float] = []
    bc_anchor_values: list[float] = []
    total_loss_values: list[float] = []

    for _ in range(config.ppo_epochs):
        permutation = torch.randperm(batch_size, device=device)
        for start_idx in range(0, batch_size, minibatch_size):
            batch_indices = permutation[start_idx : start_idx + minibatch_size]
            batch_obs = observations_tensor[batch_indices]
            batch_advantages = advantages_tensor[batch_indices]
            batch_returns = returns_tensor[batch_indices]
            batch_old_log_probs = old_log_probs_tensor[batch_indices]

            logits, value_predictions = model(batch_obs)
            behavior_logits = _apply_behavior_policy_modifiers_to_logits(
                logits=logits,
                stagnant_steps=stagnant_steps_tensor[batch_indices],
                last_movement=last_movement_tensor[batch_indices],
                trigger_steps=config.anti_stuck_trigger_steps,
                direction_penalty=config.anti_stuck_direction_penalty,
                movement_temperature=movement_temperature_tensor[batch_indices],
                shooting_temperature=shooting_temperature_tensor[batch_indices],
                bomb_temperature=bomb_temperature_tensor[batch_indices],
            )
            new_log_probs = _action_log_prob_from_indices(
                logits=behavior_logits,
                movement_actions=movement_actions[batch_indices],
                shooting_actions=shooting_actions[batch_indices],
                bomb_actions=bomb_actions[batch_indices],
            )
            entropy = _true_policy_entropy_from_logits(behavior_logits).mean()
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            unclipped_objective = ratio * batch_advantages
            clipped_ratio = torch.clamp(
                ratio,
                1.0 - config.ppo_clip_epsilon,
                1.0 + config.ppo_clip_epsilon,
            )
            clipped_objective = clipped_ratio * batch_advantages
            policy_loss = -torch.min(unclipped_objective, clipped_objective).mean()
            value_loss = 0.5 * (batch_returns - value_predictions).pow(2).mean()
            bc_anchor_loss = _compute_bc_anchor_loss(
                model=model,
                reference_policy=reference_policy,
                observations=batch_obs,
                movement_weight=config.bc_anchor_movement_weight,
                shooting_weight=config.bc_anchor_shooting_weight,
                bomb_weight=config.bc_anchor_bomb_weight,
            )
            loss = (
                policy_loss
                + config.value_coef * value_loss
                - config.entropy_coef * entropy
                + config.bc_anchor_coef * bc_anchor_loss
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            policy_loss_values.append(float(policy_loss.item()))
            value_loss_values.append(float(value_loss.item()))
            entropy_values.append(float(entropy.item()))
            bc_anchor_values.append(float(bc_anchor_loss.item()))
            total_loss_values.append(float(loss.item()))

    return {
        "policy_loss": float(np.mean(policy_loss_values)) if policy_loss_values else 0.0,
        "value_loss": float(np.mean(value_loss_values)) if value_loss_values else 0.0,
        "entropy": float(np.mean(entropy_values)) if entropy_values else 0.0,
        "bc_anchor_loss": float(np.mean(bc_anchor_values)) if bc_anchor_values else 0.0,
        "total_loss": float(np.mean(total_loss_values)) if total_loss_values else 0.0,
        "adv_raw_mean": adv_raw_mean,
        "adv_raw_std": adv_raw_std,
        "adv_norm_skipped": adv_norm_skipped,
        "adv_norm_mean": adv_norm_mean,
        "adv_norm_std": adv_norm_std,
        "log_prob_mean": log_prob_mean,
        "log_prob_std": log_prob_std,
    }


def _compute_gae(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    next_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_advantage = 0.0
    for step_idx in reversed(range(len(rewards))):
        mask = 1.0 - dones[step_idx]
        next_val = next_value if step_idx == len(rewards) - 1 else values[step_idx + 1]
        delta = rewards[step_idx] + gamma * next_val * mask - values[step_idx]
        last_advantage = delta + gamma * gae_lambda * mask * last_advantage
        advantages[step_idx] = last_advantage
    returns = advantages + values
    return advantages, returns


def _sample_action(
    sampling_logits: dict[str, torch.Tensor],
    deterministic: bool,
    movement_temperature: float,
    shooting_temperature: float,
    bomb_temperature: float,
) -> tuple[
    IsaacAction,
    dict[str, torch.distributions.Categorical],
    torch.Tensor,
]:
    movement_temperature = max(movement_temperature, 1e-3)
    shooting_temperature = max(shooting_temperature, 1e-3)
    bomb_temperature = max(bomb_temperature, 1e-3)
    scaled_logits = {
        "movement": sampling_logits["movement"] / movement_temperature,
        "shooting": sampling_logits["shooting"] / shooting_temperature,
        "bomb": sampling_logits["bomb"] / bomb_temperature,
    }
    movement_dist = torch.distributions.Categorical(logits=scaled_logits["movement"])
    shooting_dist = torch.distributions.Categorical(logits=scaled_logits["shooting"])
    bomb_dist = torch.distributions.Categorical(logits=scaled_logits["bomb"])
    behavior_policy_distributions = {
        "movement": movement_dist,
        "shooting": shooting_dist,
        "bomb": bomb_dist,
    }

    if deterministic:
        movement = int(torch.argmax(scaled_logits["movement"], dim=1).item())
        shooting = int(torch.argmax(scaled_logits["shooting"], dim=1).item())
        bomb = int(torch.argmax(scaled_logits["bomb"], dim=1).item())
    else:
        movement = int(movement_dist.sample().item())
        shooting = int(shooting_dist.sample().item())
        bomb = int(bomb_dist.sample().item())

    behavior_policy_entropy = (
        movement_dist.entropy()
        + shooting_dist.entropy()
        + bomb_dist.entropy()
    )
    return (
        IsaacAction(movement=movement, shooting=shooting, bomb=bomb),
        behavior_policy_distributions,
        behavior_policy_entropy,
    )


def _apply_anti_stuck_bias(
    logits: dict[str, torch.Tensor],
    stagnant_steps: int,
    last_movement: int,
    trigger_steps: int,
    direction_penalty: float,
) -> dict[str, torch.Tensor]:
    if stagnant_steps < trigger_steps or last_movement == 0:
        return logits

    adjusted_logits = dict(logits)
    movement_logits = logits["movement"].clone()
    movement_logits[:, last_movement] -= direction_penalty
    adjusted_logits["movement"] = movement_logits
    return adjusted_logits


def _apply_behavior_policy_modifiers_to_logits(
    logits: dict[str, torch.Tensor],
    stagnant_steps: torch.Tensor,
    last_movement: torch.Tensor,
    trigger_steps: int,
    direction_penalty: float,
    movement_temperature: torch.Tensor,
    shooting_temperature: torch.Tensor,
    bomb_temperature: torch.Tensor,
) -> dict[str, torch.Tensor]:
    adjusted_logits = {
        "movement": logits["movement"].clone(),
        "shooting": logits["shooting"],
        "bomb": logits["bomb"],
    }
    anti_stuck_mask = (stagnant_steps >= trigger_steps) & (last_movement != 0)
    if bool(anti_stuck_mask.any().item()):
        batch_indices = torch.nonzero(anti_stuck_mask, as_tuple=False).squeeze(-1)
        adjusted_logits["movement"][batch_indices, last_movement[batch_indices]] -= direction_penalty

    movement_scale = torch.clamp(movement_temperature.float(), min=1e-3).unsqueeze(1)
    shooting_scale = torch.clamp(shooting_temperature.float(), min=1e-3).unsqueeze(1)
    bomb_scale = torch.clamp(bomb_temperature.float(), min=1e-3).unsqueeze(1)
    adjusted_logits["movement"] = adjusted_logits["movement"] / movement_scale
    adjusted_logits["shooting"] = adjusted_logits["shooting"] / shooting_scale
    adjusted_logits["bomb"] = adjusted_logits["bomb"] / bomb_scale
    return adjusted_logits


def _action_log_prob(
    distributions: dict[str, torch.distributions.Categorical],
    action: IsaacAction,
) -> torch.Tensor:
    device = distributions["movement"].logits.device
    return (
        distributions["movement"].log_prob(torch.tensor(action.movement, device=device))
        + distributions["shooting"].log_prob(torch.tensor(action.shooting, device=device))
        + distributions["bomb"].log_prob(torch.tensor(action.bomb, device=device))
    )


def _action_log_prob_from_indices(
    logits: dict[str, torch.Tensor],
    movement_actions: torch.Tensor,
    shooting_actions: torch.Tensor,
    bomb_actions: torch.Tensor,
) -> torch.Tensor:
    movement_dist = torch.distributions.Categorical(logits=logits["movement"])
    shooting_dist = torch.distributions.Categorical(logits=logits["shooting"])
    bomb_dist = torch.distributions.Categorical(logits=logits["bomb"])
    return (
        movement_dist.log_prob(movement_actions)
        + shooting_dist.log_prob(shooting_actions)
        + bomb_dist.log_prob(bomb_actions)
    )


def _true_policy_entropy_from_logits(logits: dict[str, torch.Tensor]) -> torch.Tensor:
    movement_dist = torch.distributions.Categorical(logits=logits["movement"])
    shooting_dist = torch.distributions.Categorical(logits=logits["shooting"])
    bomb_dist = torch.distributions.Categorical(logits=logits["bomb"])
    return movement_dist.entropy() + shooting_dist.entropy() + bomb_dist.entropy()


def _compute_bc_anchor_loss(
    model: IsaacActorCritic,
    reference_policy: IsaacCNNPolicy,
    observations: torch.Tensor,
    movement_weight: float,
    shooting_weight: float,
    bomb_weight: float,
) -> torch.Tensor:
    current_logits, _ = model(observations)
    with torch.no_grad():
        reference_logits = reference_policy(observations)

    weights = {
        "movement": movement_weight,
        "shooting": shooting_weight,
        "bomb": bomb_weight,
    }
    losses: list[torch.Tensor] = []
    for head_name in ("movement", "shooting", "bomb"):
        reference_probs = torch.softmax(reference_logits[head_name], dim=1)
        current_log_probs = torch.log_softmax(current_logits[head_name], dim=1)
        kl = nn.functional.kl_div(
            current_log_probs,
            reference_probs,
            reduction="batchmean",
        )
        losses.append(weights[head_name] * kl)
    return sum(losses)


def _normalize_advantages(advantages: torch.Tensor) -> torch.Tensor:
    if advantages.numel() <= 1:
        return advantages
    std = advantages.std(unbiased=False)
    if float(std.item()) < 0.05:
        return advantages
    mean = advantages.mean()
    return (advantages - mean) / (std + 1e-8)


def _summarize_recent_episode_window(
    recent_episode_window: deque[dict[str, float | bool]],
) -> dict[str, float]:
    if not recent_episode_window:
        return {
            "episodes": 0.0,
            "exit_rate": 0.0,
            "clear_rate": 0.0,
            "avg_rooms": 0.0,
            "avg_reward": 0.0,
        }

    episodes = float(len(recent_episode_window))
    exit_rate = sum(bool(entry["exited_first_room"]) for entry in recent_episode_window) / episodes
    clear_rate = sum(bool(entry["cleared_room"]) for entry in recent_episode_window) / episodes
    avg_rooms = sum(float(entry["rooms_explored_gained"]) for entry in recent_episode_window) / episodes
    avg_reward = sum(float(entry["reward_sum"]) for entry in recent_episode_window) / episodes
    return {
        "episodes": episodes,
        "exit_rate": float(exit_rate),
        "clear_rate": float(clear_rate),
        "avg_rooms": float(avg_rooms),
        "avg_reward": float(avg_reward),
    }


def _score_recent_episode_window(metrics: dict[str, float]) -> float:
    return (
        metrics["exit_rate"] * 100.0
        + metrics["clear_rate"] * 40.0
        + metrics["avg_rooms"] * 10.0
        + metrics["avg_reward"]
    )


def _resolve_pretrained_model_path(
    explicit_path: str | None,
    prefer_latest_rl_best: bool,
) -> str | None:
    if explicit_path:
        return explicit_path
    if prefer_latest_rl_best:
        latest_rl_best = _find_latest_rl_best_model()
        if latest_rl_best is not None:
            return str(latest_rl_best)
    try:
        from binding_rl_agent.inference import find_latest_model

        return str(find_latest_model())
    except FileNotFoundError:
        return None


def _find_latest_rl_best_model(rl_runs_dir: str | Path = "rl_runs") -> Path | None:
    root = Path(rl_runs_dir)
    if not root.exists():
        return None
    candidates = sorted(
        root.rglob("best_actor_critic.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _load_pretrained_weights(model: IsaacActorCritic, model_path: str, device: torch.device) -> None:
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    state_dict = checkpoint["model_state_dict"]
    if any(key.startswith("policy.") for key in state_dict):
        policy_state_dict = {
            key.removeprefix("policy."): value
            for key, value in state_dict.items()
            if key.startswith("policy.")
        }
        model.policy.load_state_dict(policy_state_dict, strict=False)
    else:
        model.policy.load_state_dict(state_dict, strict=False)


def _auto_reset_game() -> None:
    release_all_agent_keys()
    time.sleep(1.0)
    # Death/game-over screens often need an explicit confirm before restart.
    tap_key("space", hold_seconds=0.12)
    time.sleep(0.8)
    tap_key("space", hold_seconds=0.12)
    time.sleep(2.0)
    hold_keys(["r"], hold_seconds=2.0)
    time.sleep(2.0)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
