from __future__ import annotations

import argparse

from binding_rl_agent.preprocessing import FRAME_TRANSFORMS
from binding_rl_agent.training import TrainConfig, train_behavior_cloning


def parse_args() -> argparse.Namespace:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(
        description="Train a behavior cloning policy from recorded Isaac rollouts."
    )
    parser.add_argument("--rollouts-dir",   default=defaults.rollouts_dir)
    parser.add_argument("--output-dir",     default=defaults.output_dir)
    parser.add_argument("--epochs",         type=int,   default=defaults.epochs)
    parser.add_argument("--batch-size",     type=int,   default=defaults.batch_size)
    parser.add_argument("--learning-rate",  type=float, default=defaults.learning_rate)
    parser.add_argument("--train-fraction", type=float, default=defaults.train_fraction)
    parser.add_argument("--early-stop-patience", type=int, default=defaults.early_stop_patience,
                        help="0 = disabled")
    parser.add_argument("--weight-decay",   type=float, default=defaults.weight_decay)
    parser.add_argument("--no-augment",     action="store_true")
    # Observation
    parser.add_argument("--frame-size",  type=int, default=defaults.frame_size,
                        help="Resize target (square) for each frame.")
    parser.add_argument("--frame-mode",  default=defaults.frame_mode,
                        choices=list(FRAME_TRANSFORMS.keys()),
                        help="Per-frame preprocessing mode.")
    parser.add_argument("--stack-size",  type=int, default=defaults.stack_size,
                        help="Number of frames stacked per observation.")
    parser.add_argument("--motion-channels", action="store_true",
                        help="Append frame-diff channels after stacking.")
    # Model architecture
    parser.add_argument("--conv-channels", type=int, nargs="+",
                        default=list(defaults.conv_channels))
    parser.add_argument("--hidden-dim",  type=int,   default=defaults.hidden_dim)
    parser.add_argument("--dropout",     type=float, default=defaults.dropout)
    parser.add_argument("--no-batchnorm", action="store_true",
                        help="Disable batch normalisation in conv layers")
    # Misc
    parser.add_argument("--movement-idle-weight",      type=float, default=defaults.movement_idle_weight)
    parser.add_argument("--movement-direction-weight", type=float, default=defaults.movement_direction_weight)
    parser.add_argument("--shooting-idle-weight",      type=float, default=defaults.shooting_idle_weight)
    parser.add_argument("--max-runs",    type=int, default=None)
    parser.add_argument("--lr-scheduler", choices=["none", "cosine"], default=defaults.lr_scheduler)
    parser.add_argument("--movement-only", action="store_true")
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--label-smoothing", type=float, default=defaults.label_smoothing,
                        help="Label smoothing for all cross-entropy losses")
    parser.add_argument("--val-split-mode", choices=["random", "temporal"],
                        default=defaults.val_split_mode,
                        help="random (default): mixed split preserving temporal distribution; "
                             "temporal: first 80%% train / last 20%% val per run")
    parser.add_argument("--cache-dir", default=defaults.cache_dir,
                        help="Pre-processed frame cache directory ('' = disabled). "
                             "Speeds up training by caching bilateral/colour transforms to disk.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_behavior_cloning(
        TrainConfig(
            rollouts_dir=args.rollouts_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            train_fraction=args.train_fraction,
            early_stop_patience=args.early_stop_patience,
            weight_decay=args.weight_decay,
            augment=not args.no_augment,
            frame_size=args.frame_size,
            frame_mode=args.frame_mode,
            stack_size=args.stack_size,
            motion_channels=args.motion_channels,
            conv_channels=tuple(args.conv_channels),
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            use_batchnorm=not args.no_batchnorm,
            movement_idle_weight=args.movement_idle_weight,
            movement_direction_weight=args.movement_direction_weight,
            shooting_idle_weight=args.shooting_idle_weight,
            max_runs=args.max_runs,
            lr_scheduler=args.lr_scheduler,
            movement_only=args.movement_only,
            seed=args.seed,
            label_smoothing=args.label_smoothing,
            val_split_mode=args.val_split_mode,
            cache_dir=args.cache_dir,
        )
    )
    print(f"num_samples={result.num_samples}  model={result.model_path}")
    print(f"train_loss={result.final_train_loss:.4f}  val_loss={result.final_val_loss:.4f}")
    print(f"movement_acc={result.final_val_movement_accuracy:.3f}")
    if not args.movement_only:
        print(f"shooting_acc={result.final_val_shooting_accuracy:.3f}")
        print(f"joint_acc={result.final_val_joint_accuracy:.3f}")


if __name__ == "__main__":
    main()
