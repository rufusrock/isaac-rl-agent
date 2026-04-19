"""Diagnose why the BC model has decent holdout accuracy but plays terribly.

Generates:
  1. Per-class accuracy breakdown on holdout (is it just getting idle right?)
  2. Confusion matrix for movement predictions
  3. Side-by-side visual comparison of training frames vs live capture
  4. Action diversity analysis (does the model collapse to a few actions?)
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from binding_rl_agent.dataset import (
    MOVEMENT_NAMES,
    SHOOTING_NAMES,
    IsaacRolloutDataset,
)
from binding_rl_agent.inference import load_policy_checkpoint
from binding_rl_agent.training import TrainConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose BC model quality")
    parser.add_argument("--model-path", required=True, help="Path to bc_policy.pt")
    parser.add_argument("--rollouts-dir", default="rollouts")
    parser.add_argument("--holdout-run", default="run_20260413_160649")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-live-frames", type=int, default=100,
                        help="Number of live frames to capture for comparison")
    parser.add_argument("--save-dir", default="diagnostics",
                        help="Directory to save diagnostic outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model, device, checkpoint = load_policy_checkpoint(args.model_path)
    train_cfg = checkpoint.get("train_config", {})
    frame_mode = train_cfg.get("frame_mode", "gray")
    stack_size = int(train_cfg.get("stack_size", 4))
    frame_size = int(train_cfg.get("frame_size", 128))
    motion_channels = bool(train_cfg.get("motion_channels", False))

    print(f"Model: {args.model_path}")
    print(f"Config: frame_mode={frame_mode}, stack_size={stack_size}, "
          f"frame_size={frame_size}, motion_channels={motion_channels}")
    print()

    # --- 1. Holdout evaluation with per-class breakdown ---
    print("=" * 60)
    print("HOLDOUT PER-CLASS ACCURACY")
    print("=" * 60)

    holdout_ds = IsaacRolloutDataset(
        rollouts_dir=args.rollouts_dir,
        include_runs=(args.holdout_run,),
        stack_size=stack_size,
        frame_size=frame_size,
        frame_mode=frame_mode,
        motion_channels=motion_channels,
    )
    print(f"Holdout samples: {len(holdout_ds)}")
    print(f"Movement distribution: {holdout_ds.summary.movement_counts}")
    print(f"Shooting distribution: {holdout_ds.summary.shooting_counts}")
    print()

    loader = DataLoader(holdout_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0)

    # Collect all predictions and ground truth
    all_mv_pred: list[int] = []
    all_mv_true: list[int] = []
    all_sh_pred: list[int] = []
    all_sh_true: list[int] = []
    all_mv_conf: list[float] = []

    model.eval()
    with torch.no_grad():
        for batch_obs, batch_targets in loader:
            obs = batch_obs.float().to(device)
            if obs.dtype == torch.uint8:
                obs = obs.float()
            obs = obs / 255.0 if obs.max() > 1.0 else obs

            logits = model(obs)

            mv_probs = torch.softmax(logits["movement"], dim=1)
            sh_probs = torch.softmax(logits["shooting"], dim=1)

            mv_pred = mv_probs.argmax(dim=1)
            sh_pred = sh_probs.argmax(dim=1)
            mv_conf = mv_probs.max(dim=1).values

            all_mv_pred.extend(mv_pred.cpu().tolist())
            all_mv_true.extend(batch_targets["movement"].tolist())
            all_sh_pred.extend(sh_pred.cpu().tolist())
            all_sh_true.extend(batch_targets["shooting"].tolist())
            all_mv_conf.extend(mv_conf.cpu().tolist())

    mv_pred = np.array(all_mv_pred)
    mv_true = np.array(all_mv_true)
    sh_pred = np.array(all_sh_pred)
    sh_true = np.array(all_sh_true)
    mv_conf = np.array(all_mv_conf)

    # --- Movement per-class accuracy ---
    print("Movement per-class accuracy:")
    print(f"  {'Class':>8}  {'Count':>6}  {'Correct':>7}  {'Acc':>6}  {'Predicted':>9}")
    total_correct = 0
    for cls in sorted(MOVEMENT_NAMES.keys()):
        mask = mv_true == cls
        count = mask.sum()
        if count == 0:
            continue
        correct = (mv_pred[mask] == cls).sum()
        total_correct += correct
        # How often does the model predict this class?
        pred_count = (mv_pred == cls).sum()
        print(f"  {MOVEMENT_NAMES[cls]:>8}  {count:>6}  {correct:>7}  "
              f"{correct/count:.3f}  {pred_count:>9}")
    overall = total_correct / len(mv_true)
    print(f"  {'OVERALL':>8}  {len(mv_true):>6}  {total_correct:>7}  {overall:.3f}")
    print()

    # --- Movement confusion matrix ---
    print("Movement confusion matrix (rows=true, cols=predicted):")
    n_mv = len(MOVEMENT_NAMES)
    cm = np.zeros((n_mv, n_mv), dtype=int)
    for t, p in zip(mv_true, mv_pred):
        cm[t, p] += 1

    header = "         " + "".join(f"{MOVEMENT_NAMES[i]:>8}" for i in range(n_mv))
    print(header)
    for i in range(n_mv):
        row = f"{MOVEMENT_NAMES[i]:>8} "
        for j in range(n_mv):
            row += f"{cm[i,j]:>8}"
        # Add row accuracy
        row_total = cm[i].sum()
        row_acc = cm[i, i] / row_total if row_total > 0 else 0
        row += f"  ({row_acc:.1%})"
        print(row)
    print()

    # --- Shooting per-class accuracy ---
    print("Shooting per-class accuracy:")
    print(f"  {'Class':>8}  {'Count':>6}  {'Correct':>7}  {'Acc':>6}")
    for cls in sorted(SHOOTING_NAMES.keys()):
        mask = sh_true == cls
        count = mask.sum()
        if count == 0:
            continue
        correct = (sh_pred[mask] == cls).sum()
        print(f"  {SHOOTING_NAMES[cls]:>8}  {count:>6}  {correct:>7}  "
              f"{correct/count:.3f}")
    sh_overall = (sh_pred == sh_true).mean()
    print(f"  {'OVERALL':>8}  {len(sh_true):>6}  {(sh_pred == sh_true).sum():>7}  "
          f"{sh_overall:.3f}")
    print()

    # --- Confidence analysis ---
    print("=" * 60)
    print("CONFIDENCE ANALYSIS (movement head)")
    print("=" * 60)

    # When the model is wrong, how confident is it?
    correct_mask = mv_pred == mv_true
    wrong_mask = ~correct_mask

    if correct_mask.any():
        print(f"  When CORRECT: mean conf = {mv_conf[correct_mask].mean():.3f}, "
              f"median = {np.median(mv_conf[correct_mask]):.3f}")
    if wrong_mask.any():
        print(f"  When WRONG:   mean conf = {mv_conf[wrong_mask].mean():.3f}, "
              f"median = {np.median(mv_conf[wrong_mask]):.3f}")

    # Confidence by class
    print()
    print("  Mean confidence by predicted class:")
    for cls in sorted(MOVEMENT_NAMES.keys()):
        mask = mv_pred == cls
        if mask.any():
            print(f"    {MOVEMENT_NAMES[cls]:>8}: {mv_conf[mask].mean():.3f} "
                  f"(n={mask.sum()})")
    print()

    # --- Prediction diversity ---
    print("=" * 60)
    print("PREDICTION DIVERSITY")
    print("=" * 60)

    mv_pred_counts = Counter(mv_pred.tolist())
    print("  Model movement prediction distribution:")
    for cls in sorted(mv_pred_counts.keys()):
        count = mv_pred_counts[cls]
        pct = count / len(mv_pred) * 100
        print(f"    {MOVEMENT_NAMES[cls]:>8}: {count:>6} ({pct:.1f}%)")

    mv_true_counts = Counter(mv_true.tolist())
    print("  Actual movement distribution:")
    for cls in sorted(mv_true_counts.keys()):
        count = mv_true_counts[cls]
        pct = count / len(mv_true) * 100
        print(f"    {MOVEMENT_NAMES[cls]:>8}: {count:>6} ({pct:.1f}%)")
    print()

    # --- Consecutive same-prediction runs ---
    print("=" * 60)
    print("ACTION PERSISTENCE (consecutive same-prediction runs)")
    print("=" * 60)

    run_lengths: list[int] = []
    current_run = 1
    for i in range(1, len(mv_pred)):
        if mv_pred[i] == mv_pred[i - 1]:
            current_run += 1
        else:
            run_lengths.append(current_run)
            current_run = 1
    run_lengths.append(current_run)

    run_lengths_arr = np.array(run_lengths)
    print(f"  Total prediction changes: {len(run_lengths)}")
    print(f"  Mean run length: {run_lengths_arr.mean():.1f} frames")
    print(f"  Median run length: {np.median(run_lengths_arr):.1f} frames")
    print(f"  Max run length: {run_lengths_arr.max()} frames")
    print(f"  Runs > 20 frames: {(run_lengths_arr > 20).sum()}")
    print(f"  Runs > 50 frames: {(run_lengths_arr > 50).sum()}")
    print()

    # Compare to ground truth persistence
    gt_run_lengths: list[int] = []
    current_run = 1
    for i in range(1, len(mv_true)):
        if mv_true[i] == mv_true[i - 1]:
            current_run += 1
        else:
            gt_run_lengths.append(current_run)
            current_run = 1
    gt_run_lengths.append(current_run)

    gt_arr = np.array(gt_run_lengths)
    print(f"  Ground truth action changes: {len(gt_run_lengths)}")
    print(f"  GT mean run length: {gt_arr.mean():.1f} frames")
    print(f"  GT median run length: {np.median(gt_arr):.1f} frames")
    print(f"  GT max run length: {gt_arr.max()} frames")
    print()

    # --- Directional accuracy when NOT idle ---
    print("=" * 60)
    print("DIRECTIONAL ACCURACY (excluding idle frames)")
    print("=" * 60)

    dir_mask = mv_true != 0  # frames where ground truth is a direction
    if dir_mask.any():
        dir_correct = (mv_pred[dir_mask] == mv_true[dir_mask]).sum()
        dir_total = dir_mask.sum()
        dir_acc = dir_correct / dir_total
        print(f"  Directional frames: {dir_total}")
        print(f"  Correct: {dir_correct} ({dir_acc:.3f})")
        # What does the model predict when it should be moving?
        pred_when_should_move = Counter(mv_pred[dir_mask].tolist())
        print(f"  Model predictions when ground truth is a direction:")
        for cls in sorted(pred_when_should_move.keys()):
            count = pred_when_should_move[cls]
            pct = count / dir_total * 100
            print(f"    {MOVEMENT_NAMES[cls]:>8}: {count:>6} ({pct:.1f}%)")

    idle_mask = mv_true == 0
    if idle_mask.any():
        idle_correct = (mv_pred[idle_mask] == 0).sum()
        idle_total = idle_mask.sum()
        print(f"\n  Idle frames: {idle_total}")
        print(f"  Correctly predicted idle: {idle_correct} ({idle_correct/idle_total:.3f})")
        pred_when_idle = Counter(mv_pred[idle_mask].tolist())
        print(f"  Model predictions when ground truth is idle:")
        for cls in sorted(pred_when_idle.keys()):
            count = pred_when_idle[cls]
            pct = count / idle_total * 100
            print(f"    {MOVEMENT_NAMES[cls]:>8}: {count:>6} ({pct:.1f}%)")
    print()

    print("=" * 60)
    print("DIAGNOSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
