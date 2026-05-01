from __future__ import annotations

import argparse
from dataclasses import replace

from binding_rl_agent.rl_training import RLTrainConfig, train_actor_critic


def parse_args() -> argparse.Namespace:
    defaults = RLTrainConfig()
    reward_defaults = defaults.reward_config
    parser = argparse.ArgumentParser(
        description="Train a lightweight actor-critic policy against live Isaac telemetry."
    )
    parser.add_argument("--title", default=None, help="Optional Isaac window title substring.")
    parser.add_argument("--telemetry-port", type=int, default=8123, help="UDP telemetry port.")
    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=5.0,
        help="How long to wait before starting live RL.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=defaults.total_episodes,
        help="Number of episodes (one PPO update per episode).",
    )
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=defaults.max_steps_per_episode,
        help="Safety cap on steps before forcing an episode end.",
    )
    parser.add_argument("--ppo-epochs", type=int, default=defaults.ppo_epochs, help="PPO epochs per episode.")
    parser.add_argument(
        "--ppo-minibatch-size",
        type=int,
        default=defaults.ppo_minibatch_size,
        help="PPO minibatch size within the episode buffer.",
    )
    parser.add_argument(
        "--ppo-clip-epsilon",
        type=float,
        default=defaults.ppo_clip_epsilon,
        help="PPO clipped-surrogate epsilon.",
    )
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate, help="Adam learning rate.")
    parser.add_argument("--gamma", type=float, default=defaults.gamma, help="Discount factor.")
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=defaults.entropy_coef,
        help="Entropy regularization coefficient.",
    )
    parser.add_argument(
        "--movement-sampling-temperature",
        type=float,
        default=defaults.movement_sampling_temperature,
        help="Temperature applied to movement logits (lower = closer to argmax).",
    )
    parser.add_argument(
        "--shooting-sampling-temperature",
        type=float,
        default=defaults.shooting_sampling_temperature,
        help="Temperature applied to shooting logits (1.0 = BC's natural distribution).",
    )
    parser.add_argument(
        "--bomb-sampling-temperature",
        type=float,
        default=defaults.bomb_sampling_temperature,
        help="Temperature applied to bomb logits (ignored when --enable-bomb is not set).",
    )
    parser.add_argument(
        "--enable-bomb",
        dest="disable_bomb",
        action="store_false",
        help="Allow the agent to take bomb actions (default: bomb is forced to 0).",
    )
    parser.set_defaults(disable_bomb=defaults.disable_bomb)
    parser.add_argument(
        "--target-kl",
        type=float,
        default=defaults.target_kl,
        help="Target KL between behavior and current policy (per minibatch).",
    )
    parser.add_argument(
        "--min-learning-rate",
        type=float,
        default=defaults.min_learning_rate,
        help="Floor for adaptive LR scaling.",
    )
    parser.add_argument(
        "--max-learning-rate",
        type=float,
        default=defaults.max_learning_rate,
        help="Ceiling for adaptive LR scaling.",
    )
    parser.add_argument("--output-dir", default="rl_runs", help="Directory for RL checkpoints/logs.")
    parser.add_argument(
        "--pretrained-model",
        default=None,
        help="Optional checkpoint path. Defaults to the latest BC model unless RL resume is enabled.",
    )
    parser.add_argument(
        "--prefer-latest-rl-best",
        dest="prefer_latest_rl_best",
        action="store_true",
        help="Resume from the latest best RL checkpoint when no explicit pretrained path is given.",
    )
    parser.add_argument(
        "--no-prefer-latest-rl-best",
        dest="prefer_latest_rl_best",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(prefer_latest_rl_best=defaults.prefer_latest_rl_best)
    parser.add_argument(
        "--auto-reset",
        action="store_true",
        help="Try to restart the game automatically after death.",
    )
    parser.add_argument(
        "--diagnostics-sample-frames",
        type=int,
        default=16,
        help="How many evenly spaced rollout frames to save per update for diagnostics.",
    )
    parser.add_argument(
        "--bc-anchor-coef",
        type=float,
        default=defaults.bc_anchor_coef,
        help="Strength of the imitation-policy anchor during RL fine-tuning.",
    )
    parser.add_argument(
        "--bc-anchor-movement-weight",
        type=float,
        default=defaults.bc_anchor_movement_weight,
        help="Relative BC-anchor weight for the movement head.",
    )
    parser.add_argument(
        "--bc-anchor-shooting-weight",
        type=float,
        default=defaults.bc_anchor_shooting_weight,
        help="Relative BC-anchor weight for the shooting head.",
    )
    parser.add_argument(
        "--bc-anchor-bomb-weight",
        type=float,
        default=defaults.bc_anchor_bomb_weight,
        help="Relative BC-anchor weight for the bomb head.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=defaults.early_stop_patience,
        help="Stop training after this many episodes without a best-checkpoint improvement.",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=defaults.early_stop_min_delta,
        help="Minimum best-checkpoint metric improvement required to reset early-stopping patience.",
    )
    parser.add_argument(
        "--best-window-episodes",
        type=int,
        default=defaults.best_checkpoint_window_episodes,
        help="How many recent completed episodes to use when choosing the best checkpoint.",
    )
    parser.add_argument(
        "--best-window-min-episodes",
        type=int,
        default=defaults.best_checkpoint_min_episodes,
        help="Minimum completed episodes before the best-checkpoint metric becomes active.",
    )
    parser.add_argument(
        "--time-penalty",
        type=float,
        default=reward_defaults.time_penalty,
        help="Per-step time penalty (use 0.0 to disable).",
    )
    parser.add_argument(
        "--transition-reward",
        type=float,
        default=reward_defaults.transition_reward,
        help="Reward for room transitions.",
    )
    parser.add_argument(
        "--rooms-explored-reward",
        type=float,
        default=reward_defaults.rooms_explored_reward,
        help="Reward for exploring a genuinely new room.",
    )
    parser.add_argument(
        "--room-clear-reward",
        type=float,
        default=reward_defaults.room_clear_reward,
        help="Reward for clearing a room.",
    )
    parser.add_argument(
        "--kill-reward",
        type=float,
        default=reward_defaults.kill_reward,
        help="Reward for each kill.",
    )
    parser.add_argument(
        "--damage-penalty",
        type=float,
        default=reward_defaults.damage_penalty,
        help="Penalty applied per unit of damage taken.",
    )
    parser.add_argument(
        "--death-penalty",
        type=float,
        default=reward_defaults.death_penalty,
        help="Penalty for dying without recent progress.",
    )
    parser.add_argument(
        "--exploratory-death-penalty",
        type=float,
        default=reward_defaults.exploratory_death_penalty,
        help="Penalty for exploratory deaths after recent progress.",
    )
    parser.add_argument(
        "--stagnation-penalty",
        type=float,
        default=reward_defaults.stagnation_penalty,
        help="Penalty applied once the agent has been stagnant for too long.",
    )
    parser.add_argument(
        "--stagnation-window",
        type=int,
        default=reward_defaults.stagnation_window,
        help="Steps without progress before stagnation penalties begin.",
    )
    parser.add_argument(
        "--room-timeout-steps",
        type=int,
        default=reward_defaults.room_timeout_steps,
        help="Maximum steps allowed in a navigation room before the episode times out.",
    )
    parser.add_argument(
        "--room-timeout-penalty",
        type=float,
        default=reward_defaults.room_timeout_penalty,
        help="Penalty applied when a room timeout occurs.",
    )
    parser.add_argument(
        "--no-room-timeout-terminate",
        action="store_true",
        help="Apply room timeout penalty but do not end the episode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reward_config = replace(
        RLTrainConfig().reward_config,
        time_penalty=args.time_penalty,
        transition_reward=args.transition_reward,
        rooms_explored_reward=args.rooms_explored_reward,
        room_clear_reward=args.room_clear_reward,
        kill_reward=args.kill_reward,
        damage_penalty=args.damage_penalty,
        death_penalty=args.death_penalty,
        exploratory_death_penalty=args.exploratory_death_penalty,
        stagnation_penalty=args.stagnation_penalty,
        stagnation_window=args.stagnation_window,
        room_timeout_steps=args.room_timeout_steps,
        room_timeout_penalty=args.room_timeout_penalty,
        room_timeout_terminates=not args.no_room_timeout_terminate,
    )
    run_dir = train_actor_critic(
        RLTrainConfig(
            title_substring=args.title,
            telemetry_port=args.telemetry_port,
            warmup_seconds=args.warmup_seconds,
            total_episodes=args.episodes,
            max_steps_per_episode=args.max_steps_per_episode,
            ppo_epochs=args.ppo_epochs,
            ppo_minibatch_size=args.ppo_minibatch_size,
            ppo_clip_epsilon=args.ppo_clip_epsilon,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            entropy_coef=args.entropy_coef,
            movement_sampling_temperature=args.movement_sampling_temperature,
            shooting_sampling_temperature=args.shooting_sampling_temperature,
            bomb_sampling_temperature=args.bomb_sampling_temperature,
            disable_bomb=args.disable_bomb,
            target_kl=args.target_kl,
            min_learning_rate=args.min_learning_rate,
            max_learning_rate=args.max_learning_rate,
            output_dir=args.output_dir,
            pretrained_model_path=args.pretrained_model,
            prefer_latest_rl_best=args.prefer_latest_rl_best,
            auto_reset=args.auto_reset,
            diagnostics_sample_frames=args.diagnostics_sample_frames,
            bc_anchor_coef=args.bc_anchor_coef,
            bc_anchor_movement_weight=args.bc_anchor_movement_weight,
            bc_anchor_shooting_weight=args.bc_anchor_shooting_weight,
            bc_anchor_bomb_weight=args.bc_anchor_bomb_weight,
            early_stop_patience=args.early_stop_patience,
            early_stop_min_delta=args.early_stop_min_delta,
            best_checkpoint_window_episodes=args.best_window_episodes,
            best_checkpoint_min_episodes=args.best_window_min_episodes,
            reward_config=reward_config,
        )
    )
    print(f"Saved RL run to: {run_dir}")


if __name__ == "__main__":
    main()
