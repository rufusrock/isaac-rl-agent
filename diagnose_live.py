"""Capture live frames and training frames, run both through the model,
and save a visual comparison to disk.

This tests whether the live capture pipeline produces the same inputs
as the training pipeline.  If the model behaves differently on live
frames vs training frames, there's a pipeline mismatch.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from binding_rl_agent.dataset import MOVEMENT_NAMES, IsaacRolloutDataset
from binding_rl_agent.env import IsaacFrameEnv, ObservationConfig
from binding_rl_agent.inference import (
    find_latest_model,
    frame_size_from_checkpoint,
    load_policy_checkpoint,
    obs_config_from_checkpoint,
    predict_policy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare live vs training frames")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--rollouts-dir", default="rollouts")
    parser.add_argument("--holdout-run", default="run_20260413_160649")
    parser.add_argument("--num-frames", type=int, default=20,
                        help="Number of live frames to capture")
    parser.add_argument("--save-dir", default="diagnostics")
    parser.add_argument("--title", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.model_path or str(find_latest_model())
    model, device, checkpoint = load_policy_checkpoint(model_path)
    train_cfg = checkpoint.get("train_config", {})
    frame_mode = train_cfg.get("frame_mode", "gray")
    stack_size = int(train_cfg.get("stack_size", 4))
    frame_size = int(train_cfg.get("frame_size", 128))
    motion_channels = bool(train_cfg.get("motion_channels", False))

    print(f"Model: {model_path}")
    print(f"Config: {frame_mode}, stack={stack_size}, size={frame_size}")
    print()

    # --- Part 1: Sample training frames and predictions ---
    print("Loading holdout data...")
    holdout_ds = IsaacRolloutDataset(
        rollouts_dir=args.rollouts_dir,
        include_runs=(args.holdout_run,),
        stack_size=stack_size,
        frame_size=frame_size,
        frame_mode=frame_mode,
        motion_channels=False,  # predict_policy adds motion channels itself
    )

    # Sample evenly spaced frames from holdout
    n = len(holdout_ds)
    indices = [int(i * n / args.num_frames) for i in range(args.num_frames)]

    print(f"Running model on {args.num_frames} holdout frames...")
    for i, idx in enumerate(indices):
        obs_tensor, targets = holdout_ds[idx]
        obs_np = obs_tensor.numpy()

        # The observation is uint8 (stack_size, H, W) for gray
        # Save the last frame as an image
        if obs_np.ndim == 3:
            last_frame = obs_np[-1]  # last in stack
        else:
            last_frame = obs_np

        # Run through model (same path as predict_policy)
        pred = predict_policy(model, device, obs_np, checkpoint=checkpoint)
        true_mv = int(targets["movement"].item())

        # Build comparison image
        vis = cv2.cvtColor(last_frame, cv2.COLOR_GRAY2BGR)
        vis = cv2.resize(vis, (256, 256), interpolation=cv2.INTER_NEAREST)

        probs_str = " ".join(
            f"{MOVEMENT_NAMES[j]}:{p:.2f}"
            for j, p in enumerate(pred.movement.probabilities)
        )
        cv2.putText(vis, f"TRAIN frame {idx}", (5, 20),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(vis, f"true={MOVEMENT_NAMES[true_mv]}", (5, 40),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(vis, f"pred={pred.movement.label} ({pred.movement.confidence:.2f})",
                     (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                     (0, 255, 0) if pred.movement.index == true_mv else (0, 0, 255), 1)
        cv2.putText(vis, probs_str, (5, 80),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        cv2.imwrite(str(save_dir / f"train_{i:03d}.png"), vis)

    # --- Part 2: Capture live frames and predictions ---
    print(f"\nCapturing {args.num_frames} live frames...")
    print("Make sure the Isaac window is visible!")

    obs_kwargs = obs_config_from_checkpoint(checkpoint)
    env = IsaacFrameEnv(
        title_substring=args.title,
        observation_config=ObservationConfig(
            width=frame_size,
            height=frame_size,
            **obs_kwargs,
        ),
    )

    observation = env.reset()
    import time
    print("Capturing in 2 seconds...")
    time.sleep(2)

    for i in range(args.num_frames):
        observation = env.step(action=None)
        obs_np = observation

        if obs_np.ndim == 3:
            last_frame = obs_np[-1]
        else:
            last_frame = obs_np

        pred = predict_policy(model, device, obs_np, checkpoint=checkpoint)

        vis = cv2.cvtColor(last_frame, cv2.COLOR_GRAY2BGR)
        vis = cv2.resize(vis, (256, 256), interpolation=cv2.INTER_NEAREST)

        probs_str = " ".join(
            f"{MOVEMENT_NAMES[j]}:{p:.2f}"
            for j, p in enumerate(pred.movement.probabilities)
        )
        cv2.putText(vis, f"LIVE frame {i}", (5, 20),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(vis, f"pred={pred.movement.label} ({pred.movement.confidence:.2f})",
                     (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        cv2.putText(vis, probs_str, (5, 60),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        cv2.imwrite(str(save_dir / f"live_{i:03d}.png"), vis)
        time.sleep(0.1)

    # --- Part 3: Cross-pipeline test ---
    # Take a live frame and process it through the TRAINING pipeline
    # to see if there's a difference
    print("\n--- CROSS-PIPELINE TEST ---")
    print("Capturing one live frame and processing both ways...")

    raw_bgr = env.capture.grab()  # raw BGR from window capture

    # Live pipeline: resize BGR -> grayscale
    from binding_rl_agent.preprocessing import resize_frame, to_grayscale, to_rgb, frame_gray
    live_processed = to_grayscale(resize_frame(raw_bgr, frame_size, frame_size))

    # Training pipeline: BGR -> resize to 256 -> RGB CHW -> resize to 128 -> frame_gray
    from binding_rl_agent.preprocessing import resize_frame_rgb
    resized_256 = resize_frame(raw_bgr, 256, 256)
    rgb_chw = to_rgb(resized_256)  # (3, 256, 256) RGB
    rgb_128 = resize_frame_rgb(rgb_chw, frame_size)  # (3, 128, 128) RGB
    train_processed = frame_gray(rgb_128)  # (128, 128) grayscale

    # Compare
    diff = np.abs(live_processed.astype(np.float32) - train_processed.astype(np.float32))
    print(f"  Live frame shape: {live_processed.shape}, dtype: {live_processed.dtype}")
    print(f"  Train frame shape: {train_processed.shape}, dtype: {train_processed.dtype}")
    print(f"  Pixel value ranges: live=[{live_processed.min()}, {live_processed.max()}], "
          f"train=[{train_processed.min()}, {train_processed.max()}]")
    print(f"  Mean absolute difference: {diff.mean():.2f}")
    print(f"  Max absolute difference: {diff.max():.0f}")
    print(f"  Pixels with diff > 5: {(diff > 5).sum()} / {diff.size} "
          f"({(diff > 5).sum() / diff.size * 100:.1f}%)")

    # Save comparison
    cv2.imwrite(str(save_dir / "pipeline_live.png"), live_processed)
    cv2.imwrite(str(save_dir / "pipeline_train.png"), train_processed)
    diff_vis = (diff * 10).clip(0, 255).astype(np.uint8)  # amplify for visibility
    cv2.imwrite(str(save_dir / "pipeline_diff_10x.png"), diff_vis)

    print(f"\nAll diagnostics saved to {save_dir}/")
    print("Compare train_*.png vs live_*.png to see if inputs look different.")
    print("Check pipeline_diff_10x.png for pixel-level preprocessing differences.")


if __name__ == "__main__":
    main()
