from __future__ import annotations

import atexit
from collections import defaultdict, deque
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
from binding_rl_agent.inference import (
    frame_size_from_checkpoint,
    motion_channels_from_checkpoint,
    obs_config_from_checkpoint,
    policy_kwargs_from_checkpoint,
)
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
    # PPO / training schedule — one update per episode.
    total_episodes: int = 1000
    max_steps_per_episode: int = 1024
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_epochs: int = 4
    ppo_minibatch_size: int = 64
    ppo_clip_epsilon: float = 0.2
    # Conservative fine-tuning learning rate. From-BC fine-tuning is fragile;
    # higher rates lead to BC forgetting + entropy explosion. Standard for
    # RLHF-style PPO-from-BC is in the 1e-6 to 1e-5 range.
    learning_rate: float = 1.0e-6
    value_coef: float = 0.5
    # No entropy bonus by default. The cooled sampling temperatures already
    # provide the right amount of stochasticity, and a positive entropy_coef
    # actively pushes the policy toward uniform when advantages are near-zero
    # (which happens often with sparse rewards), accelerating BC forgetting.
    entropy_coef: float = 0.0
    max_grad_norm: float = 0.5
    # Per-head sampling temperatures. Movement is cooled (was noisy at T=1.0);
    # shooting stays at T=1.0 (cooling it makes "no-shoot" dominate, since
    # BC's argmax for the shooting head is "idle" on most frames). Applied to
    # BOTH sampling and PPO update logits for ratio consistency.
    movement_sampling_temperature: float = 0.5
    shooting_sampling_temperature: float = 1.0
    bomb_sampling_temperature: float = 1.0
    # Force bomb action to 0 always (BC bomb head is undertrained; agent self-bombs).
    disable_bomb: bool = True
    output_dir: str = "rl_runs"
    pretrained_model_path: str | None = None
    prefer_latest_rl_best: bool = False
    device: str | None = None
    auto_reset: bool = False
    deterministic_actions: bool = False
    # Stronger BC trust region. With a low LR + no entropy bonus, this keeps
    # the policy close to BC while RL nudges it on the dimensions with reward.
    bc_anchor_coef: float = 0.8
    # Adaptive KL trust region. Tracks KL between behavior and current policy
    # per minibatch; aborts the rest of the PPO epochs if KL > 1.5 * target_kl,
    # and scales the optimizer LR after each update toward the target.
    # Target KL between behavior and current policy. Measured with Schulman's
    # k3 estimator (always non-negative, low variance). 0.015 gives more
    # headroom for the full PPO update to complete (4 epochs) without aborting
    # on moderate KL spikes; outliers are still caught by kl_early_stop_factor.
    target_kl: float = 0.015
    kl_early_stop_factor: float = 1.5
    kl_lr_scale_down_threshold: float = 2.0
    kl_lr_scale_up_threshold: float = 0.5
    kl_lr_scale_down_factor: float = 0.5
    kl_lr_scale_up_factor: float = 1.5
    min_learning_rate: float = 1.0e-8
    max_learning_rate: float = 1.0e-4
    bc_anchor_movement_weight: float = 1.0
    bc_anchor_shooting_weight: float = 2.0
    bc_anchor_bomb_weight: float = 0.5
    early_stop_patience: int = 100
    early_stop_min_delta: float = 0.0
    best_checkpoint_window_episodes: int = 50
    best_checkpoint_min_episodes: int = 20
    diagnostics_sample_frames: int = 16
    seed: int = 7
    # Asymmetric/positive-leaning rewards: drop time/stagnation/timeout penalties,
    # keep modest death/damage negatives. Termination at room timeout is the signal.
    reward_config: TelemetryRewardConfig = TelemetryRewardConfig(
        time_penalty=0.0,
        transition_reward=0.1,
        rooms_explored_reward=1.0,
        room_clear_reward=2.0,
        damage_penalty=-0.1,
        kill_reward=0.05,
        coin_reward=0.01,
        key_reward=0.05,
        collectible_reward=0.5,
        soul_heart_reward=0.1,
        black_heart_reward=0.15,
        stagnation_penalty=0.0,
        stagnation_window=90,
        # ~20s @ 20Hz (combat rooms are gated out via nav_hint==STAY).
        room_timeout_steps=400,
        room_timeout_penalty=0.0,
        # 2 minutes of no kills/damage in a STAY (combat) room → end the episode.
        # Targets the freeze attractor; without this the agent learns to stand
        # still in combat rooms because no timeout fires there.
        stay_idle_timeout_steps=2400,
        stay_idle_timeout_penalty=0.0,
        stay_idle_timeout_terminates=True,
        death_penalty=-0.5,
        exploratory_death_penalty=-0.25,
        recent_progress_window=120,
    )


class IsaacActorCritic(nn.Module):
    def __init__(self, policy: IsaacCNNPolicy) -> None:
        super().__init__()
        self.policy = policy
        hidden_dim = self.policy._trunk_linear[0].out_features
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        observations: torch.Tensor,
        nav_hint: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        cnn_out = self.policy.features(observations)
        flat = self.policy._flatten(cnn_out)
        if self.policy.use_nav_hint_embedding:
            if nav_hint is None:
                nav_hint = torch.zeros(flat.shape[0], dtype=torch.long, device=flat.device)
            emb = self.policy.nav_hint_embedding(nav_hint)
            flat = torch.cat([flat, emb], dim=1)
        hidden = self.policy._trunk_linear(flat)
        logits = {
            "movement": self.policy.movement_head(hidden),
            "shooting": self.policy.shooting_head(hidden),
            "bomb": self.policy.bomb_head(hidden),
        }
        value = self.value_head(hidden).squeeze(-1)
        return logits, value


def train_actor_critic(config: RLTrainConfig) -> Path:
    _seed_everything(config.seed)
    # Force deterministic CuDNN convolution kernels so batch=1 (collection) and
    # batch=N (PPO update) forward passes give bit-identical logits. Without
    # this, batch-size variation introduces ~5e-4 logit noise that adds tiny
    # but non-zero KL even before any gradient step.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Ensure held WASD/arrow/bomb keys are released on any exit path,
    # including Ctrl+C and unhandled exceptions.
    atexit.register(release_all_agent_keys)
    if config.deterministic_actions:
        print("Ignoring deterministic_actions=True during PPO training; sampling must stay stochastic.")
        config = replace(config, deterministic_actions=False)

    # Resolve pretrained checkpoint first so we can derive observation config from it.
    resolved_pretrained = _resolve_pretrained_model_path(
        explicit_path=config.pretrained_model_path,
        prefer_latest_rl_best=config.prefer_latest_rl_best,
    )

    # Build ObservationConfig from the BC checkpoint when available, so the RL env
    # uses the same frame_mode, stack_size, and resolution that the policy was
    # trained on.  Fall back to sensible defaults for from-scratch runs.
    device = torch.device(
        config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(
        f"Device: {device} (cuda available: {torch.cuda.is_available()}, "
        f"device_name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a'})"
    )
    bc_checkpoint: dict | None = None
    motion_channels: bool = False
    use_nav_hint_embedding: bool = False
    if resolved_pretrained:
        try:
            bc_checkpoint = torch.load(resolved_pretrained, map_location=device, weights_only=True)
        except TypeError:
            bc_checkpoint = torch.load(resolved_pretrained, map_location=device)
        # If we loaded an RL checkpoint, prefer its embedded BC checkpoint metadata.
        if "bc_checkpoint" in bc_checkpoint and isinstance(bc_checkpoint["bc_checkpoint"], dict):
            print("Resuming RL — using embedded bc_checkpoint metadata for architecture.")
            rl_state_dict = bc_checkpoint["model_state_dict"]
            bc_checkpoint = bc_checkpoint["bc_checkpoint"]
            bc_checkpoint["__rl_state_dict__"] = rl_state_dict
        obs_kwargs = obs_config_from_checkpoint(bc_checkpoint)
        frame_size = frame_size_from_checkpoint(bc_checkpoint)
        motion_channels = motion_channels_from_checkpoint(bc_checkpoint)
        use_nav_hint_embedding = bool(bc_checkpoint.get("use_nav_hint_embedding", False))
        print(
            f"Observation config from checkpoint: "
            f"frame_mode={obs_kwargs.get('frame_mode')!r} "
            f"stack_size={obs_kwargs.get('stack_size')} "
            f"frame_size={frame_size} "
            f"motion_channels={motion_channels} "
            f"use_nav_hint_embedding={use_nav_hint_embedding}"
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
    observation = _maybe_apply_motion_channels(observation, motion_channels)
    _wait_for_telemetry(env, config.warmup_seconds)
    # _wait_for_telemetry seeded reward_detector.previous_game_state. Use that
    # same state to seed env.current_nav_hint, so the very first action of the
    # very first episode is sampled with the correct nav_hint instead of the
    # default 0 (STAY).
    initial_game_state = env.reward_detector.previous_game_state
    if initial_game_state is not None:
        env.current_nav_hint = env._compute_nav_hint(initial_game_state)

    if bc_checkpoint is not None:
        policy_kwargs = policy_kwargs_from_checkpoint(bc_checkpoint)
        policy = IsaacCNNPolicy(**policy_kwargs)
        # When resuming an RL checkpoint, bc_checkpoint contains only architecture
        # metadata (train_config etc.) — its model_state_dict was stripped at save
        # time. The actual policy weights are loaded later from __rl_state_dict__.
        if "model_state_dict" in bc_checkpoint:
            bc_state_dict = bc_checkpoint["model_state_dict"]
            load_result = policy.load_state_dict(bc_state_dict, strict=False)
            if load_result.missing_keys or load_result.unexpected_keys:
                print(
                    f"BC weight load: missing={load_result.missing_keys} "
                    f"unexpected={load_result.unexpected_keys}"
                )
    else:
        print("No pretrained checkpoint supplied. RL will start from random weights.")
        policy = IsaacCNNPolicy(input_channels=int(observation.shape[0]))

    model = IsaacActorCritic(policy=policy)
    if bc_checkpoint is not None and "__rl_state_dict__" in bc_checkpoint:
        rl_load = model.load_state_dict(bc_checkpoint["__rl_state_dict__"], strict=False)
        print(
            f"RL state load: missing={rl_load.missing_keys} "
            f"unexpected={rl_load.unexpected_keys}"
        )

    # Build reference_policy independently so the BC anchor always pulls toward
    # the *original* BC, even when resuming from an RL checkpoint. Without this
    # the anchor would point at the already-drifted RL policy and weaken with
    # every resume.
    reference_policy: IsaacCNNPolicy | None = None
    if bc_checkpoint is not None and "model_state_dict" in bc_checkpoint:
        reference_policy = IsaacCNNPolicy(**policy_kwargs_from_checkpoint(bc_checkpoint))
        reference_policy.load_state_dict(bc_checkpoint["model_state_dict"], strict=False)
        reference_policy.to(device)
        reference_policy.eval()
        for parameter in reference_policy.parameters():
            parameter.requires_grad_(False)
        print("Reference policy initialized from BC checkpoint (BC anchor active).")
    else:
        # No BC weights available (from-scratch or legacy RL checkpoint without
        # embedded BC) — disable the BC anchor for this run.
        if config.bc_anchor_coef > 0.0:
            print(
                "No BC weights available for reference_policy; disabling BC anchor "
                "(bc_anchor_coef -> 0) for this run."
            )
            config = replace(config, bc_anchor_coef=0.0)
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
    # Compact run-level state for the post-run summary file.
    run_state = {
        "run_dir": str(run_dir),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": asdict(config),
        "resumed_from": resolved_pretrained,
        "best_episode": None,
        "best_metrics": None,
        "termination": "in_progress",
        "all_episode_summaries": [],
    }
    try:
     # Keep the model in eval mode for the entire training loop. Activating
     # train mode would enable the BC's dropout (p=0.2), which produces a
     # different forward pass per call on the same observation. That causes
     # the PPO update to compute new_log_probs that differ from the captured
     # old_log_probs by dropout noise rather than by real policy change,
     # which explodes the importance ratio and the KL estimator. The BC uses
     # LayerNorm, so eval mode has no other side effects.
     model.eval()
     for episode_idx in range(1, config.total_episodes + 1):
        rollout = _collect_episode(
            env, model, device, observation, config, motion_channels, episode_idx
        )
        observation = rollout["next_observation"]

        # If this episode ended at max_steps the game is mid-action while we
        # run PPO — arm a grace window so the next episode can absorb any
        # during-update damage/death without polluting training.
        if rollout["done_reason"] == "max_steps":
            env.reward_detector.grace_steps_remaining = (
                config.reward_config.post_update_grace_steps
            )

        # Skip the PPO update + window add if this episode was a "fake death"
        # caused by the agent being unattended during the previous PPO update.
        # The data is mostly noise (dying from grace-period damage with no
        # real action signal) and would corrupt rolling-window metrics.
        if rollout.get("is_grace_death"):
            print(
                f"ep={episode_idx:04d} steps={len(rollout['rewards']):04d} "
                f"end=grace_death       (skipping PPO update + window)",
                flush=True,
            )
            # Still arm grace if THIS one ended at max_steps (it didn't — was death),
            # otherwise nothing to do; just continue to the next collection.
            continue

        update_start = time.monotonic()
        losses = _update_model(
            model=model,
            reference_policy=reference_policy,
            optimizer=optimizer,
            rollout=rollout,
            device=device,
            config=config,
        )
        update_seconds = time.monotonic() - update_start
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
            "episode": episode_idx,
            "episode_length": int(len(rollout["rewards"])),
            "done_reason": rollout["done_reason"],
            "update_seconds": update_seconds,
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
            f"ep={episode_idx:04d} steps={summary['episode_length']:04d} "
            f"end={rollout['done_reason']:<17s} "
            f"reward={summary['sum_reward']:+7.3f} "
            f"rooms+={summary['rooms_explored_gained']} kills+={summary['kills_gained']} "
            f"vloss={summary['value_loss']:.3f} ploss={summary['policy_loss']:7.3f} "
            f"ent={summary['entropy']:.3f} "
            f"kl={summary.get('kl_mean', 0.0):.4f} kl_max={summary.get('kl_max', 0.0):.4f} ep_run={summary.get('epochs_ran', 0)} "
            f"lr={summary.get('lr_after', 0.0):.1e} upd={update_seconds:.1f}s "
            f"win_exit={summary['recent_window_exit_rate']:.3f}"
        )
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary) + "\n")
        run_state["all_episode_summaries"].append(_compact_summary(summary))
        save_rollout_diagnostics(
            diagnostics_dir=diagnostics_dir,
            update_idx=episode_idx,
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
                "bc_checkpoint": _strip_bc_checkpoint_for_save(bc_checkpoint),
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
                        "best_episode": episode_idx,
                        "best_window_score": best_window_score,
                        "best_window_metrics": recent_window_metrics,
                        "bc_checkpoint": _strip_bc_checkpoint_for_save(bc_checkpoint),
                    },
                    best_checkpoint_path,
                )
                print(
                    f"new_best episode={episode_idx:04d} "
                    f"exit_rate={recent_window_metrics['exit_rate']:.3f} "
                    f"clear_rate={recent_window_metrics['clear_rate']:.3f} "
                    f"avg_rooms={recent_window_metrics['avg_rooms']:.3f} "
                    f"checkpoint={best_checkpoint_path.name}"
                )
                run_state["best_episode"] = episode_idx
                run_state["best_metrics"] = dict(recent_window_metrics)
            else:
                updates_without_improvement += 1
            if updates_without_improvement >= config.early_stop_patience:
                print(
                    f"early_stop episode={episode_idx:04d} "
                    f"best_window_score={best_window_score:.4f} "
                    f"patience={config.early_stop_patience}"
                )
                run_state["termination"] = f"early_stop@{episode_idx}"
                break
        else:
            updates_without_improvement = 0
     if run_state["termination"] == "in_progress":
      run_state["termination"] = f"total_episodes_reached@{config.total_episodes}"
    except KeyboardInterrupt:
     run_state["termination"] = f"keyboard_interrupt@{episode_idx}"
     print(f"\nInterrupted at episode {episode_idx} — writing summary...")
    finally:
     release_all_agent_keys()
     _write_run_summary(run_dir, run_state)

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


def _collect_episode(
    env: IsaacVisualRLEnv,
    model: IsaacActorCritic,
    device: torch.device,
    observation: np.ndarray,
    config: RLTrainConfig,
    motion_channels: bool,
    episode_idx: int = 0,
) -> dict[str, object]:
    """Collect one episode (until done or max_steps_per_episode), returning the rollout buffer."""
    observations: list[np.ndarray] = []
    actions: list[IsaacAction] = []
    rewards: list[float] = []
    dones: list[bool] = []
    log_probs: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    infos: list[dict[str, object]] = []
    nav_hints: list[int] = []

    current_observation = observation
    done_reason = "max_steps"
    # Collect under no_grad — we don't need autograd here (gradients are only
    # computed during _update_model, which re-runs the forward pass with grad).
    # Without this, every forward retains its activation graph and 1024 steps
    # of graphs blow out GPU memory.
    with torch.no_grad():
      for step_idx in range(config.max_steps_per_episode):
        observation_tensor = (
            torch.from_numpy(current_observation).float().unsqueeze(0).to(device) / 255.0
        )
        current_nav_hint = int(env.current_nav_hint)
        nav_hint_tensor = torch.tensor([current_nav_hint], dtype=torch.long, device=device)
        logits, value = model(observation_tensor, nav_hint=nav_hint_tensor)
        sampling_logits = _disable_bomb_logits(logits) if config.disable_bomb else logits

        sampled_action, behavior_policy_distributions, behavior_policy_entropy = _sample_action(
            sampling_logits=sampling_logits,
            deterministic=config.deterministic_actions,
            movement_temperature=config.movement_sampling_temperature,
            shooting_temperature=config.shooting_sampling_temperature,
            bomb_temperature=config.bomb_sampling_temperature,
        )
        behavior_policy_log_prob = _action_log_prob(behavior_policy_distributions, sampled_action)

        step = env.step(sampled_action)
        next_observation = _maybe_apply_motion_channels(step.observation, motion_channels)

        observations.append(current_observation.copy())
        actions.append(sampled_action)
        rewards.append(step.reward)
        dones.append(step.done)
        # Squeeze the leading batch=1 dim so torch.stack later gives [N], not
        # [N, 1]. Without this, [N, 1] broadcasts against [B] in the PPO update
        # to produce a [B, B] cross-sample comparison rather than the intended
        # [B] paired comparison — which silently corrupts the importance ratio
        # and the KL estimator with completely fake B^2 noise.
        log_probs.append(behavior_policy_log_prob.squeeze(0))
        values.append(value.squeeze(0))
        entropies.append(behavior_policy_entropy.squeeze(0))
        nav_hints.append(current_nav_hint)
        infos.append(dict(step.info))

        current_observation = next_observation
        if step.done:
            done_reason = (
                "death" if step.info.get("death_candidate")
                else "room_timeout" if step.info.get("timeout_done")
                else "stay_idle_timeout" if step.info.get("stay_idle_timeout_done")
                else "other"
            )
            if config.auto_reset:
                _auto_reset_game()
            current_observation = _maybe_apply_motion_channels(env.reset(), motion_channels)
            break

    next_obs_tensor = (
        torch.from_numpy(current_observation).float().unsqueeze(0).to(device) / 255.0
    )
    next_nav_hint_tensor = torch.tensor(
        [int(env.current_nav_hint)], dtype=torch.long, device=device
    )
    with torch.no_grad():
        _, next_value = model(next_obs_tensor, nav_hint=next_nav_hint_tensor)

    return {
        "observations": np.stack(observations, axis=0),
        "actions": actions,
        "rewards": np.asarray(rewards, dtype=np.float32),
        "dones": np.asarray(dones, dtype=np.float32),
        "log_probs": torch.stack(log_probs),
        "values": torch.stack(values),
        "entropies": torch.stack(entropies),
        "infos": infos,
        "nav_hints": np.asarray(nav_hints, dtype=np.int64),
        "next_value": next_value.squeeze(0),
        "next_observation": current_observation,
        "done_reason": done_reason,
        # True when this episode ended quickly under grace — i.e., the agent
        # likely died from being unattended during the PPO update. Such
        # rollouts carry no real signal and are excluded from PPO + window.
        "is_grace_death": (
            done_reason == "death"
            and len(rewards) <= config.reward_config.post_update_grace_steps
            and any(info.get("grace_active") for info in infos)
        ),
    }


def _update_model(
    model: IsaacActorCritic,
    reference_policy: IsaacCNNPolicy | None,
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
    # Transfer uint8 first (~115 MB at 1024x7x128x128), then cast to float on
    # GPU. Casting before transfer would push ~470 MB across PCIe — 4x slower.
    observations_tensor = (
        torch.from_numpy(np.asarray(rollout["observations"], dtype=np.uint8)).to(device).float() / 255.0
    )

    old_log_probs_tensor = rollout["log_probs"].detach().to(device)
    old_values_tensor = rollout["values"].detach().to(device)
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
    nav_hints_tensor = torch.from_numpy(rollout["nav_hints"]).to(device=device)

    batch_size = observations_tensor.shape[0]
    minibatch_size = max(1, min(config.ppo_minibatch_size, batch_size))
    policy_loss_values: list[float] = []
    value_loss_values: list[float] = []
    entropy_values: list[float] = []
    bc_anchor_values: list[float] = []
    total_loss_values: list[float] = []
    kl_values: list[float] = []
    epochs_ran = 0
    kl_early_stop_threshold = config.kl_early_stop_factor * config.target_kl
    kl_early_stopped = False

    for _ in range(config.ppo_epochs):
        if kl_early_stopped:
            break
        epochs_ran += 1
        permutation = torch.randperm(batch_size, device=device)
        for start_idx in range(0, batch_size, minibatch_size):
            batch_indices = permutation[start_idx : start_idx + minibatch_size]
            batch_obs = observations_tensor[batch_indices]
            batch_advantages = advantages_tensor[batch_indices]
            batch_returns = returns_tensor[batch_indices]
            batch_old_log_probs = old_log_probs_tensor[batch_indices]

            batch_nav_hints = nav_hints_tensor[batch_indices]
            logits, value_predictions = model(batch_obs, nav_hint=batch_nav_hints)
            # Apply the same bomb-disable + per-head temperature transforms as during sampling.
            update_logits = _disable_bomb_logits(logits) if config.disable_bomb else logits
            head_temperatures = {
                "movement": max(config.movement_sampling_temperature, 1e-3),
                "shooting": max(config.shooting_sampling_temperature, 1e-3),
                "bomb": max(config.bomb_sampling_temperature, 1e-3),
            }
            behavior_logits = {k: v / head_temperatures[k] for k, v in update_logits.items()}
            new_log_probs = _action_log_prob_from_indices(
                logits=behavior_logits,
                movement_actions=movement_actions[batch_indices],
                shooting_actions=shooting_actions[batch_indices],
                bomb_actions=bomb_actions[batch_indices],
            )
            entropy = _true_policy_entropy_from_logits(behavior_logits).mean()
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            # Schulman's k3 KL estimator: always >= 0, low variance, unbiased.
            # See http://joschu.net/blog/kl-approx.html. The simpler estimator
            # (old - new).mean() can go negative and is much noisier, which
            # confuses the adaptive trust region logic.
            with torch.no_grad():
                log_ratio = new_log_probs - batch_old_log_probs
                approx_kl = ((ratio - 1) - log_ratio).mean()
            kl_values.append(float(approx_kl.item()))
            unclipped_objective = ratio * batch_advantages
            clipped_ratio = torch.clamp(
                ratio,
                1.0 - config.ppo_clip_epsilon,
                1.0 + config.ppo_clip_epsilon,
            )
            clipped_objective = clipped_ratio * batch_advantages
            policy_loss = -torch.min(unclipped_objective, clipped_objective).mean()
            # Clipped value loss (standard PPO): prevents the value head from
            # making huge jumps on outlier returns. Mirrors the policy ratio
            # clipping using ppo_clip_epsilon as the trust radius.
            batch_old_values = old_values_tensor[batch_indices]
            v_clipped = batch_old_values + torch.clamp(
                value_predictions - batch_old_values,
                -config.ppo_clip_epsilon,
                +config.ppo_clip_epsilon,
            )
            v_loss_unclipped = (batch_returns - value_predictions).pow(2)
            v_loss_clipped = (batch_returns - v_clipped).pow(2)
            value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
            bc_anchor_loss = _compute_bc_anchor_loss(
                current_logits=logits,
                reference_policy=reference_policy,
                observations=batch_obs,
                nav_hint=batch_nav_hints,
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

            # Within-update early stop: if this minibatch already pushed KL past
            # the threshold, abandon the rest of the epochs (and minibatches).
            # This is the main fix for the policy_loss=22+ ratio explosions.
            if float(approx_kl.item()) > kl_early_stop_threshold:
                kl_early_stopped = True
                break

    # Adaptive LR scaling based on observed KL across all minibatches that ran.
    kl_mean = float(np.mean(kl_values)) if kl_values else 0.0
    current_lr = float(optimizer.param_groups[0]["lr"])
    if kl_mean > config.kl_lr_scale_down_threshold * config.target_kl:
        new_lr = current_lr * config.kl_lr_scale_down_factor
    elif kl_mean < config.kl_lr_scale_up_threshold * config.target_kl:
        new_lr = current_lr * config.kl_lr_scale_up_factor
    else:
        new_lr = current_lr
    new_lr = max(config.min_learning_rate, min(config.max_learning_rate, new_lr))
    floor_eps = config.min_learning_rate * 1.01
    just_hit_floor = (new_lr <= floor_eps) and (current_lr > floor_eps)
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr
    # When we hit the LR floor, the trust region has fully given up — reset
    # Adam's momentum buffers so stale gradient signal from the high-LR era
    # doesn't keep pushing parameters at the new tiny LR. Without this, even
    # at lr=1e-8 the policy can move noticeably from accumulated momentum.
    if just_hit_floor:
        # Adam needs a defaultdict(dict) so that per-parameter state entries
        # auto-create on first access. Re-creating with the wrong factory (or
        # no factory) breaks the next optimizer.step() with a KeyError.
        optimizer.state = defaultdict(dict)
        print(f"  [adaptive] hit min_lr={new_lr:.1e}; reset Adam momentum buffers")

    return {
        "policy_loss": float(np.mean(policy_loss_values)) if policy_loss_values else 0.0,
        "value_loss": float(np.mean(value_loss_values)) if value_loss_values else 0.0,
        "entropy": float(np.mean(entropy_values)) if entropy_values else 0.0,
        "bc_anchor_loss": float(np.mean(bc_anchor_values)) if bc_anchor_values else 0.0,
        "total_loss": float(np.mean(total_loss_values)) if total_loss_values else 0.0,
        "kl_mean": kl_mean,
        "kl_max": float(np.max(kl_values)) if kl_values else 0.0,
        "epochs_ran": epochs_ran,
        "kl_early_stopped": kl_early_stopped,
        "lr_after": new_lr,
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
    temperatures = {
        "movement": max(movement_temperature, 1e-3),
        "shooting": max(shooting_temperature, 1e-3),
        "bomb": max(bomb_temperature, 1e-3),
    }
    scaled_logits = {head: sampling_logits[head] / temperatures[head] for head in sampling_logits}
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
    current_logits: dict[str, torch.Tensor],
    reference_policy: IsaacCNNPolicy | None,
    observations: torch.Tensor,
    nav_hint: torch.Tensor | None,
    movement_weight: float,
    shooting_weight: float,
    bomb_weight: float,
) -> torch.Tensor:
    if reference_policy is None:
        return torch.zeros((), device=observations.device)
    with torch.no_grad():
        reference_logits = reference_policy(observations, nav_hint=nav_hint)

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


def _compact_summary(summary: dict) -> dict:
    """Strip per-step / per-episode-detail fields from a summary line, keeping
    only the scalars worth scanning in the post-run summary file.
    """
    return {
        "episode": summary.get("episode"),
        "steps": summary.get("episode_length"),
        "done_reason": summary.get("done_reason"),
        "reward_sum": round(summary.get("sum_reward", 0.0), 4),
        "rooms_explored_gained": summary.get("rooms_explored_gained", 0),
        "kills_gained": summary.get("kills_gained", 0),
        "value_loss": round(summary.get("value_loss", 0.0), 4),
        "policy_loss": round(summary.get("policy_loss", 0.0), 4),
        "entropy": round(summary.get("entropy", 0.0), 4),
        "kl_mean": round(summary.get("kl_mean", 0.0), 5),
        "kl_max": round(summary.get("kl_max", 0.0), 5),
        "epochs_ran": summary.get("epochs_ran", 0),
        "kl_early_stopped": summary.get("kl_early_stopped", False),
        "lr_after": summary.get("lr_after", 0.0),
        "update_seconds": round(summary.get("update_seconds", 0.0), 2),
        "window_exit_rate": round(summary.get("recent_window_exit_rate", 0.0), 3),
        "window_clear_rate": round(summary.get("recent_window_clear_rate", 0.0), 3),
        "window_avg_rooms": round(summary.get("recent_window_avg_rooms", 0.0), 3),
    }


def _write_run_summary(run_dir: Path, run_state: dict) -> None:
    """Write a compact post-run summary to `run_dir/summary.json` for quick
    inspection. Includes config, best checkpoint, termination reason, full
    per-episode trajectory, and flagged anomalies (large policy_loss, large kl,
    frozen episodes).
    """
    summaries = list(run_state.get("all_episode_summaries", []))
    n = len(summaries)

    def _aggregate(field: str) -> dict:
        values = [s[field] for s in summaries if s.get(field) is not None]
        if not values:
            return {"min": None, "max": None, "mean": None, "last": None}
        return {
            "min": min(values),
            "max": max(values),
            "mean": float(np.mean(values)),
            "last": values[-1],
        }

    anomalies: list[dict] = []
    for s in summaries:
        flags = []
        if s.get("policy_loss", 0.0) > 5.0:
            flags.append(f"policy_loss={s['policy_loss']:.2f}")
        if s.get("kl_mean", 0.0) > 0.05:
            flags.append(f"kl_mean={s['kl_mean']:.4f}")
        if s.get("kl_max", 0.0) > 0.10:
            flags.append(f"kl_max={s['kl_max']:.4f}")
        if s.get("entropy", 0.0) > 2.5:
            flags.append(f"entropy={s['entropy']:.3f}")
        if s.get("done_reason") == "stay_idle_timeout":
            flags.append("stay_idle_timeout fired")
        if flags:
            anomalies.append({
                "episode": s["episode"],
                "flags": flags,
                "reward_sum": s.get("reward_sum"),
                "lr_after": s.get("lr_after"),
            })

    top_reward = sorted(summaries, key=lambda s: s.get("reward_sum", 0.0), reverse=True)[:5]

    summary_obj = {
        "run_dir": run_state.get("run_dir"),
        "started_at": run_state.get("started_at"),
        "ended_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "termination": run_state.get("termination"),
        "resumed_from": run_state.get("resumed_from"),
        "total_episodes_completed": n,
        "best_episode": run_state.get("best_episode"),
        "best_metrics": run_state.get("best_metrics"),
        "trajectory": {
            field: _aggregate(field)
            for field in (
                "reward_sum", "entropy", "policy_loss", "value_loss",
                "kl_mean", "kl_max", "lr_after", "epochs_ran",
                "window_exit_rate", "window_clear_rate", "window_avg_rooms",
            )
        },
        "top_reward_episodes": [
            {k: s[k] for k in ("episode", "reward_sum", "rooms_explored_gained", "kills_gained", "done_reason")}
            for s in top_reward
        ],
        "anomalies": anomalies,
        "config": run_state.get("config"),
        "all_episodes": summaries,
    }
    summary_path = run_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_obj, handle, indent=2, default=str)
    print(f"summary written to {summary_path}")


def _disable_bomb_logits(logits: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Replace bomb logits with a one-hot at index 0 so the bomb action is forced.

    With Categorical(logits=...) this gives P(0)=1, log_prob(0)=0, entropy=0,
    and the bomb head receives no gradient from policy/value/entropy losses
    (since the bomb logits no longer flow into them). The bomb head can still
    drift via the BC anchor, which operates on raw model logits.
    """
    bomb = logits["bomb"]
    forced = torch.full_like(bomb, -1e9)
    forced[..., 0] = 0.0
    return {**logits, "bomb": forced}


def _maybe_apply_motion_channels(observation: np.ndarray, motion_channels: bool) -> np.ndarray:
    """Append per-frame absolute-difference motion channels, mirroring predict_policy.

    Operates on uint8 stacked frames of shape (stack, H, W). Skips when the BC
    model wasn't trained with motion channels or when the stack has < 2 frames.
    """
    if not motion_channels or observation.shape[0] < 2:
        return observation
    obs_int = observation.astype(np.int16, copy=False)
    diffs = [
        np.abs(obs_int[t : t + 1] - obs_int[t - 1 : t]).astype(observation.dtype, copy=False)
        for t in range(1, observation.shape[0])
    ]
    return np.concatenate([observation, *diffs], axis=0)


def _strip_bc_checkpoint_for_save(bc_checkpoint: dict | None) -> dict | None:
    """Return a slim copy of the BC checkpoint, suitable for embedding in an RL save.

    Keeps train_config, input_channels, architecture flags, names, AND the BC
    model_state_dict. We need the BC weights so the BC anchor (reference_policy)
    on RL resume points at the *original* BC, not at the resumed RL policy
    (which would weaken the trust region with each successive resume).
    """
    if bc_checkpoint is None:
        return None
    keep_keys = {
        "train_config",
        "input_channels",
        "use_nav_hint_embedding",
        "movement_names",
        "shooting_names",
        "bomb_names",
        "model_state_dict",
    }
    return {key: bc_checkpoint[key] for key in keep_keys if key in bc_checkpoint}


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
