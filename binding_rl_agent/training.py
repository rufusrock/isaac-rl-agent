from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from binding_rl_agent.dataset import (
    BOMB_NAMES,
    MOVEMENT_NAMES,
    SHOOTING_NAMES,
    IsaacRolloutDataset,
)
from binding_rl_agent.models import IsaacCNNPolicy


@dataclass(frozen=True)
class TrainConfig:
    rollouts_dir: str = "rollouts"
    output_dir: str = "models"
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 1e-3
    train_fraction: float = 0.8
    seed: int = 7
    num_workers: int = 0
    movement_idle_weight: float = 1.0
    movement_direction_weight: float = 1.0
    shooting_idle_weight: float = 1.0
    early_stop_patience: int = 0
    weight_decay: float = 1e-4
    augment: bool = True
    # Observation preprocessing
    frame_size: int = 128            # resize target (square)
    frame_mode: str = "gray"         # see FRAME_TRANSFORMS keys in preprocessing.py
    stack_size: int = 4              # number of frames stacked per observation
    motion_channels: bool = False    # append frame-diff channels after stacking
    # Model architecture
    conv_channels: tuple[int, ...] = (8, 16, 16)
    hidden_dim: int = 128
    dropout: float = 0.0
    use_batchnorm: bool = True
    arch: str = "plain"               # "plain" | "impala" | "nature"
    num_resblocks: int = 2            # only used when arch="impala"
    norm_type: str | None = None      # None = derive from use_batchnorm; else "batch"|"layer"|"group"|"none"
    aug_mode: str = "flip"            # "flip" | "flip+drq" | "flip+jitter" | "drq" | "jitter" | "none"
    # Held-out run name (excluded from train+val; scored post-training).
    holdout_run: str | None = None
    # None = auto-detect from dataset (use if nav_hints present), True/False = force
    use_nav_hint_embedding: bool | None = None
    # If True, only train and report movement head
    movement_only: bool = False
    # Limit number of runs loaded from rollouts_dir (None = all)
    max_runs: int | None = None
    # LR scheduler: "none" | "cosine"
    lr_scheduler: str = "none"
    # Label smoothing for all cross-entropy losses
    label_smoothing: float = 0.05
    # Pre-process frame cache directory ("" = disabled).
    # When set, bilateral/colour transforms run once and are stored as memmaps;
    # __getitem__ becomes pure array indexing → enables num_workers > 0 on Windows.
    cache_dir: str = ""
    # Train/val split mode within each run.
    # "random": randomly assign samples — train and val share the same temporal
    #            distribution.  Frame-stack leakage is minimal and acceptable.
    #            This is the historical default and what produced the 0.786 model.
    # "temporal": first train_fraction of each run → train, remainder → val.
    #             Creates early-game/late-game distribution shift that prevents
    #             generalisation.  Only use when you explicitly need future-frame
    #             holdout (e.g. online evaluation).
    val_split_mode: str = "random"


@dataclass(frozen=True)
class TrainResult:
    model_path: Path
    metrics_path: Path
    num_samples: int
    train_samples: int
    val_samples: int
    final_train_loss: float
    final_val_loss: float
    final_val_movement_accuracy: float
    final_val_shooting_accuracy: float
    final_val_bomb_accuracy: float
    final_val_joint_accuracy: float
    holdout_samples: int = 0
    holdout_loss: float = 0.0
    holdout_movement_accuracy: float = 0.0
    holdout_shooting_accuracy: float = 0.0
    holdout_bomb_accuracy: float = 0.0


def train_behavior_cloning(config: TrainConfig | None = None) -> TrainResult:
    train_config = config or TrainConfig()
    _seed_everything(train_config.seed)

    cache_dir = train_config.cache_dir or None
    exclude = (train_config.holdout_run,) if train_config.holdout_run else ()
    dataset = IsaacRolloutDataset(
        train_config.rollouts_dir,
        max_runs=train_config.max_runs,
        stack_size=train_config.stack_size,
        frame_size=train_config.frame_size,
        frame_mode=train_config.frame_mode,
        motion_channels=train_config.motion_channels,
        cache_dir=cache_dir,
        exclude_runs=exclude,
    )
    if len(dataset) < 2:
        raise ValueError("Need at least 2 rollout samples to train and validate.")

    holdout_dataset: IsaacRolloutDataset | None = None
    if train_config.holdout_run:
        holdout_dataset = IsaacRolloutDataset(
            train_config.rollouts_dir,
            max_runs=None,
            stack_size=train_config.stack_size,
            frame_size=train_config.frame_size,
            frame_mode=train_config.frame_mode,
            motion_channels=train_config.motion_channels,
            cache_dir=cache_dir,
            include_runs=(train_config.holdout_run,),
        )

    train_indices, val_indices = _temporal_train_val_split(
        dataset,
        train_fraction=train_config.train_fraction,
        mode=train_config.val_split_mode,
        seed=train_config.seed,
    )
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    train_size = len(train_indices)
    val_size = len(val_indices)

    # With a memmap cache, __getitem__ is pure array indexing → workers are safe on Windows.
    # Without cache, bilateral filter runs per-sample on main process (num_workers=0).
    effective_workers = train_config.num_workers if train_config.num_workers > 0 else (
        4 if dataset._use_cache else 0
    )
    use_pin = effective_workers > 0 and torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=effective_workers,
        persistent_workers=effective_workers > 0,
        pin_memory=use_pin,
        prefetch_factor=2 if effective_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=effective_workers,
        persistent_workers=effective_workers > 0,
        pin_memory=use_pin,
        prefetch_factor=2 if effective_workers > 0 else None,
    )

    sample_observation, _ = dataset[0]
    input_channels = int(sample_observation.shape[0])
    if train_config.use_nav_hint_embedding is None:
        use_nav_hint_embedding = dataset.summary.has_nav_hints
    else:
        use_nav_hint_embedding = train_config.use_nav_hint_embedding
    model = IsaacCNNPolicy(
        input_channels=input_channels,
        conv_channels=train_config.conv_channels,
        hidden_dim=train_config.hidden_dim,
        dropout=train_config.dropout,
        input_size=train_config.frame_size,
        use_nav_hint_embedding=use_nav_hint_embedding,
        use_batchnorm=train_config.use_batchnorm,
        arch=train_config.arch,
        norm_type=train_config.norm_type,
        num_resblocks=train_config.num_resblocks,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ls = train_config.label_smoothing
    movement_weights = torch.tensor(
        [
            train_config.movement_idle_weight,
            train_config.movement_direction_weight,
            train_config.movement_direction_weight,
            train_config.movement_direction_weight,
            train_config.movement_direction_weight,
        ],
        dtype=torch.float32,
        device=device,
    )
    criteria = {"movement": nn.CrossEntropyLoss(weight=movement_weights, label_smoothing=ls)}
    if not train_config.movement_only:
        shooting_weights = torch.tensor(
            [train_config.shooting_idle_weight, 1.0, 1.0, 1.0, 1.0],
            dtype=torch.float32,
            device=device,
        )
        criteria["shooting"] = nn.CrossEntropyLoss(weight=shooting_weights, label_smoothing=ls)
        criteria["bomb"] = nn.CrossEntropyLoss(label_smoothing=ls)
    # nav_hint is now an INPUT embedding — it is NOT a loss target.
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)

    loss_weights: dict[str, float] = {}

    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    if train_config.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=train_config.epochs, eta_min=train_config.learning_rate * 0.01
        )

    history: list[dict[str, float]] = []
    best_val_loss = float("inf")
    best_model_state: dict | None = None
    epochs_without_improvement = 0
    epoch_bar = tqdm(range(1, train_config.epochs + 1), desc="epochs", unit="ep")
    _epoch_start = time.time()
    for epoch in epoch_bar:
        train_loss = _run_epoch(
            model=model,
            loader=train_loader,
            criteria=criteria,
            optimizer=optimizer,
            device=device,
            training=True,
            augment=train_config.augment,
            aug_mode=train_config.aug_mode,
            loss_weights=loss_weights,
        )
        val_metrics = _evaluate(
            model=model,
            loader=val_loader,
            criteria=criteria,
            device=device,
            loss_weights=loss_weights,
        )
        # Separate the raw confusion matrix from per-epoch history (it goes
        # into the final metrics.json only).
        last_movement_confusion = val_metrics.pop("_movement_confusion", None)
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            **val_metrics,
        }
        history.append(epoch_metrics)
        epoch_elapsed = time.time() - _epoch_start
        _epoch_start = time.time()
        acc_str = " | ".join(
            f"{k.replace('val_', '').replace('_accuracy', '')}_acc={v:.3f}"
            for k, v in val_metrics.items() if k.endswith("_accuracy")
        )
        epoch_bar.set_postfix_str(
            f"train={train_loss:.4f} val={val_metrics['val_loss']:.4f} | {acc_str} | {epoch_elapsed:.0f}s/ep"
        )
        if scheduler is not None:
            scheduler.step()
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if train_config.early_stop_patience > 0 and epochs_without_improvement >= train_config.early_stop_patience:
            print(f"early_stop epoch={epoch:02d} best_val_loss={best_val_loss:.4f}")
            break
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    holdout_metrics: dict[str, float] = {}
    if holdout_dataset is not None:
        holdout_loader = DataLoader(
            holdout_dataset,
            batch_size=train_config.batch_size,
            shuffle=False,
            num_workers=effective_workers,
            persistent_workers=effective_workers > 0,
            pin_memory=use_pin,
            prefetch_factor=2 if effective_workers > 0 else None,
        )
        holdout_metrics = _evaluate(
            model=model,
            loader=holdout_loader,
            criteria=criteria,
            device=device,
            loss_weights=loss_weights,
        )
        holdout_metrics.pop("_movement_confusion", None)
        print(
            f"[holdout {train_config.holdout_run}] "
            f"loss={holdout_metrics['val_loss']:.4f} "
            f"move_acc={holdout_metrics.get('val_movement_accuracy', 0.0):.4f}"
        )

    output_dir = Path(train_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / _build_train_run_name()
    run_dir.mkdir(parents=True, exist_ok=False)

    model_path = run_dir / "bc_policy.pt"
    metrics_path = run_dir / "metrics.json"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_channels": int(sample_observation.shape[0]),
            "movement_names": MOVEMENT_NAMES,
            "shooting_names": SHOOTING_NAMES,
            "bomb_names": BOMB_NAMES,
            "train_config": asdict(train_config),
            "dataset_summary": asdict(dataset.summary),
            "use_nav_hint_embedding": use_nav_hint_embedding,
        },
        model_path,
    )

    # Convert confusion matrix from list-of-lists to dict-of-dicts keyed by
    # class name for readability in metrics.json.
    movement_confusion_named: dict[str, dict[str, int]] = {}
    if last_movement_confusion is not None:
        for true_idx, row in enumerate(last_movement_confusion):
            true_name = MOVEMENT_NAMES[true_idx]
            movement_confusion_named[true_name] = {
                MOVEMENT_NAMES[pred_idx]: count
                for pred_idx, count in enumerate(row)
            }

    metrics_payload = {
        "train_config": asdict(train_config),
        "dataset_summary": asdict(dataset.summary),
        "history": history,
        "device": str(device),
        "movement_confusion_matrix": movement_confusion_named,
        "holdout_metrics": holdout_metrics,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    final = history[-1]
    return TrainResult(
        model_path=model_path,
        metrics_path=metrics_path,
        num_samples=len(dataset),
        train_samples=train_size,
        val_samples=val_size,
        final_train_loss=float(final["train_loss"]),
        final_val_loss=float(final["val_loss"]),
        final_val_movement_accuracy=float(final.get("val_movement_accuracy", 0.0)),
        final_val_shooting_accuracy=float(final.get("val_shooting_accuracy", 0.0)),
        final_val_bomb_accuracy=float(final.get("val_bomb_accuracy", 0.0)),
        final_val_joint_accuracy=float(final.get("val_joint_accuracy", 0.0)),
        holdout_samples=len(holdout_dataset) if holdout_dataset is not None else 0,
        holdout_loss=float(holdout_metrics.get("val_loss", 0.0)),
        holdout_movement_accuracy=float(holdout_metrics.get("val_movement_accuracy", 0.0)),
        holdout_shooting_accuracy=float(holdout_metrics.get("val_shooting_accuracy", 0.0)),
        holdout_bomb_accuracy=float(holdout_metrics.get("val_bomb_accuracy", 0.0)),
    )


def _temporal_train_val_split(
    dataset: IsaacRolloutDataset,
    train_fraction: float = 0.8,
    mode: str = "random",
    seed: int = 7,
) -> tuple[list[int], list[int]]:
    """Within-run train/val split.

    mode="random" (default):
        Randomly assign each sample within a run to train or val.  Both sets
        cover the full temporal range of the run, so there is no early/late-game
        distribution shift.  This is the approach used by the historical 0.786
        model.  Frame-stack leakage (adjacent frames shared across the boundary)
        is minimal and does not affect generalisation in practice.

    mode="temporal":
        First train_fraction of each run -> train, remainder -> val.  Creates a
        genuine early/late-game distribution shift that prevents the model from
        generalising.  Only use when you explicitly need a causal holdout.
    """
    rng = random.Random(seed)
    num_runs = dataset.summary.num_runs
    train_indices: list[int] = []
    val_indices: list[int] = []

    for run_idx in range(num_runs):
        run_mask = dataset.sample_to_run == run_idx
        run_sample_indices = [i for i, m in enumerate(run_mask) if m]

        if mode == "temporal":
            split = max(1, int(len(run_sample_indices) * train_fraction))
            train_indices.extend(run_sample_indices[:split])
            val_indices.extend(run_sample_indices[split:])
        else:  # "random"
            shuffled = list(run_sample_indices)
            rng.shuffle(shuffled)
            split = max(1, int(len(shuffled) * train_fraction))
            train_indices.extend(shuffled[:split])
            val_indices.extend(shuffled[split:])

    if not val_indices:
        val_indices.append(train_indices.pop())

    return train_indices, val_indices


# movement/shooting: 0=idle,1=up,2=down,3=left,4=right
_HFLIP_ACTION = torch.tensor([0, 1, 2, 4, 3], dtype=torch.long)  # left<->right
_VFLIP_ACTION = torch.tensor([0, 2, 1, 3, 4], dtype=torch.long)  # up<->down

# nav_hint: 0=STAY,1=NORTH,2=SOUTH,3=WEST,4=EAST
_HFLIP_NAV = torch.tensor([0, 1, 2, 4, 3], dtype=torch.long)  # WEST<->EAST
_VFLIP_NAV = torch.tensor([0, 2, 1, 3, 4], dtype=torch.long)  # NORTH<->SOUTH


def _drq_shift(obs: torch.Tensor, max_shift: int = 4) -> torch.Tensor:
    """DrQ-v2 style random translation: reflect-pad by max_shift, random crop back."""
    B, C, H, W = obs.shape
    padded = torch.nn.functional.pad(
        obs, (max_shift,) * 4, mode="replicate"
    )
    # Per-sample random offsets in [0, 2*max_shift]
    dx = torch.randint(0, 2 * max_shift + 1, (B,), device=obs.device)
    dy = torch.randint(0, 2 * max_shift + 1, (B,), device=obs.device)
    out = torch.empty_like(obs)
    # Loop over batch — B is small enough (256) for this to be fine on GPU.
    for i in range(B):
        out[i] = padded[i, :, dy[i]:dy[i] + H, dx[i]:dx[i] + W]
    return out


def _color_jitter(obs: torch.Tensor, brightness: float = 0.2, contrast: float = 0.2) -> torch.Tensor:
    """Per-sample brightness + contrast jitter. Operates on [0,1] float tensors."""
    B = obs.shape[0]
    b = 1.0 + (torch.rand(B, 1, 1, 1, device=obs.device) * 2 - 1) * brightness
    c = 1.0 + (torch.rand(B, 1, 1, 1, device=obs.device) * 2 - 1) * contrast
    mean = obs.mean(dim=(2, 3), keepdim=True)
    return ((obs - mean) * c + mean * b).clamp_(0.0, 1.0)


def _augment_batch(
    observations: torch.Tensor,
    targets: dict[str, torch.Tensor],
    mode: str = "flip",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    do_flip = "flip" in mode
    do_drq = "drq" in mode
    do_jitter = "jitter" in mode

    if do_flip:
        if torch.rand(1).item() > 0.5:
            observations = observations.flip(-1)
            targets = {
                **targets,
                "movement": _HFLIP_ACTION.to(targets["movement"].device)[targets["movement"]],
                "shooting": _HFLIP_ACTION.to(targets["shooting"].device)[targets["shooting"]],
            }
            if "nav_hint" in targets:
                targets["nav_hint"] = _HFLIP_NAV.to(targets["nav_hint"].device)[targets["nav_hint"]]
        if torch.rand(1).item() > 0.5:
            observations = observations.flip(-2)
            targets = {
                **targets,
                "movement": _VFLIP_ACTION.to(targets["movement"].device)[targets["movement"]],
                "shooting": _VFLIP_ACTION.to(targets["shooting"].device)[targets["shooting"]],
            }
            if "nav_hint" in targets:
                targets["nav_hint"] = _VFLIP_NAV.to(targets["nav_hint"].device)[targets["nav_hint"]]
    if do_drq:
        observations = _drq_shift(observations)
    if do_jitter:
        observations = _color_jitter(observations)
    return observations, targets


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criteria: dict[str, nn.Module],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    training: bool,
    augment: bool = False,
    aug_mode: str = "flip",
    loss_weights: dict[str, float] | None = None,
) -> float:
    model.train(mode=training)
    total_loss = 0.0
    total_items = 0

    batch_bar = tqdm(loader, desc="train" if training else "val", unit="b", leave=False)
    for observations, targets_raw in batch_bar:
        observations = observations.to(device, non_blocking=True)
        if observations.dtype == torch.uint8:
            observations = observations.float().div_(255.0)

        # Move all targets to device; keep nav_hint separate (input, not loss target)
        all_targets = {name: t.to(device, non_blocking=True) for name, t in targets_raw.items()}
        nav_hint_input = all_targets.get("nav_hint")
        targets = {name: all_targets[name] for name in criteria if name in all_targets}

        if training and augment and aug_mode != "none":
            observations, augmented = _augment_batch(observations, all_targets, mode=aug_mode)
            nav_hint_input = augmented.get("nav_hint")
            targets = {name: augmented[name] for name in criteria if name in augmented}

        if training:
            optimizer.zero_grad()

        logits = model(observations, nav_hint=nav_hint_input)
        _w = loss_weights or {}
        loss = sum(
            _w.get(name, 1.0) * criteria[name](logits[name], targets[name])
            for name in criteria
        )

        if training:
            loss.backward()
            optimizer.step()

        batch_size = observations.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_items += int(batch_size)
        batch_bar.set_postfix_str(f"loss={total_loss / max(total_items, 1):.4f}")

    return total_loss / max(total_items, 1)


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criteria: dict[str, nn.Module],
    device: torch.device,
    loss_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    num_movement_classes = len(MOVEMENT_NAMES)
    model.eval()
    total_loss = 0.0
    total_items = 0
    total_correct: dict[str, int] = {name: 0 for name in criteria}

    # Per-class tracking for movement head
    movement_class_correct = [0] * num_movement_classes
    movement_class_total = [0] * num_movement_classes
    movement_confusion = [[0] * num_movement_classes for _ in range(num_movement_classes)]

    # Movement entropy accumulator
    total_movement_entropy = 0.0

    _w = loss_weights or {}

    with torch.no_grad():
        for observations, targets_raw in tqdm(loader, desc="val", unit="b", leave=False):
            observations = observations.to(device, non_blocking=True)
            if observations.dtype == torch.uint8:
                observations = observations.float().div_(255.0)

            all_targets = {name: t.to(device, non_blocking=True) for name, t in targets_raw.items()}
            nav_hint_input = all_targets.get("nav_hint")
            targets = {name: all_targets[name] for name in criteria if name in all_targets}

            logits = model(observations, nav_hint=nav_hint_input)
            loss = sum(
                _w.get(name, 1.0) * criteria[name](logits[name], targets[name])
                for name in criteria
            )

            batch_size = observations.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_items += int(batch_size)

            for name in criteria:
                preds = torch.argmax(logits[name], dim=1)
                total_correct[name] += int((preds == targets[name]).sum().item())

                if name == "movement":
                    # Compute entropy of movement softmax distribution
                    probs = torch.softmax(logits["movement"], dim=1)  # (B, 5)
                    log_probs = torch.log(probs + 1e-8)
                    entropy = -(probs * log_probs).sum(dim=1)  # (B,)
                    total_movement_entropy += float(entropy.sum().item())

                    true_np = targets[name].cpu().numpy()
                    pred_np = preds.cpu().numpy()
                    for cls in range(num_movement_classes):
                        mask = true_np == cls
                        movement_class_total[cls] += int(mask.sum())
                        movement_class_correct[cls] += int((pred_np[mask] == cls).sum())
                    for true_label, pred_label in zip(true_np, pred_np):
                        movement_confusion[int(true_label)][int(pred_label)] += 1

    metrics: dict[str, float] = {"val_loss": total_loss / max(total_items, 1)}
    for name in criteria:
        metrics[f"val_{name}_accuracy"] = total_correct[name] / max(total_items, 1)

    # Mean movement entropy — key RL-readiness metric
    metrics["val_movement_entropy"] = total_movement_entropy / max(total_items, 1)

    # Per-class accuracy for the movement head (list of 5 floats)
    metrics["val_movement_per_class_acc"] = [
        movement_class_correct[c] / max(movement_class_total[c], 1)
        for c in range(num_movement_classes)
    ]

    # Attach raw confusion matrix so callers can persist it
    metrics["_movement_confusion"] = movement_confusion

    return metrics


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_train_run_name() -> str:
    from time import strftime

    return strftime("bc_%Y%m%d_%H%M%S")
