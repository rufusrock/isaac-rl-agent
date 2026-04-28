from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from binding_rl_agent.dataset import BOMB_NAMES, MOVEMENT_NAMES, SHOOTING_NAMES
from binding_rl_agent.env import IsaacAction
from binding_rl_agent.models import IsaacCNNPolicy

if TYPE_CHECKING:
    from binding_rl_agent.room_graph import RoomGraph


@dataclass(frozen=True)
class HeadPrediction:
    index: int
    label: str
    confidence: float
    probabilities: tuple[float, ...]


@dataclass(frozen=True)
class PolicyPrediction:
    movement: HeadPrediction
    shooting: HeadPrediction
    bomb: HeadPrediction
    device: str


def find_latest_model(models_dir: str | Path = "models") -> Path:
    root = Path(models_dir)
    candidates = sorted(
        (path for path in root.iterdir() if path.is_dir() and (path / "bc_policy.pt").exists()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No trained models found under {root}")
    return candidates[0] / "bc_policy.pt"


def load_policy_checkpoint(
    model_path: str | Path,
) -> tuple[IsaacCNNPolicy, torch.device, dict]:
    checkpoint_path = Path(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    train_cfg = checkpoint.get("train_config", {})
    conv_channels = tuple(train_cfg.get("conv_channels", (8, 16, 16)))
    hidden_dim = int(train_cfg.get("hidden_dim", 128))
    frame_size = int(train_cfg.get("frame_size", 128))

    # Derive input_channels from train_config when not explicitly stored at top level.
    if "input_channels" in checkpoint:
        input_channels = int(checkpoint["input_channels"])
    else:
        _CHANNELS_PER_FRAME = {"gray": 1, "eq_gray": 1, "multichannel": 2,
                                "rgb": 3, "hsv_sv": 2, "mc_sat": 3}
        frame_mode = train_cfg.get("frame_mode", "gray")
        stack_size = int(train_cfg.get("stack_size", 4))
        motion_channels = bool(train_cfg.get("motion_channels", False))
        input_channels = _CHANNELS_PER_FRAME.get(frame_mode, 1) * stack_size
        if motion_channels:
            input_channels += stack_size - 1

    use_nav_hint_embedding = bool(checkpoint.get("use_nav_hint_embedding", False))
    arch = train_cfg.get("arch", "plain")
    norm_type = train_cfg.get("norm_type", None)
    num_resblocks = int(train_cfg.get("num_resblocks", 2))
    dropout = float(train_cfg.get("dropout", 0.0))

    common_kwargs = dict(
        input_channels=input_channels,
        input_size=frame_size,
        num_movement_actions=len(checkpoint.get("movement_names", MOVEMENT_NAMES)),
        num_shooting_actions=len(checkpoint.get("shooting_names", SHOOTING_NAMES)),
        num_bomb_actions=len(checkpoint.get("bomb_names", BOMB_NAMES)),
        conv_channels=conv_channels,
        hidden_dim=hidden_dim,
        use_nav_hint_embedding=use_nav_hint_embedding,
        arch=arch,
        norm_type=norm_type,
        num_resblocks=num_resblocks,
        dropout=dropout,
    )

    model = IsaacCNNPolicy(**common_kwargs)

    state_dict = checkpoint["model_state_dict"]
    result = model.load_state_dict(state_dict, strict=False)
    if result.unexpected_keys:
        import warnings
        warnings.warn(
            f"Checkpoint contains keys not present in the current model "
            f"(likely legacy BatchNorm weights): {result.unexpected_keys}. "
            f"These were ignored.",
            stacklevel=2,
        )
    model.to(device)
    model.eval()
    return model, device, checkpoint


def obs_config_from_checkpoint(checkpoint: dict) -> dict:
    """Return kwargs for ObservationConfig inferred from a checkpoint."""
    train_cfg = checkpoint.get("train_config", {})
    frame_mode = train_cfg.get("frame_mode", "gray")
    stack_size = int(train_cfg.get("stack_size", 4))
    return dict(frame_mode=frame_mode, stack_size=stack_size)


def frame_size_from_checkpoint(checkpoint: dict) -> int:
    """Return the frame size (width == height) used during training."""
    train_cfg = checkpoint.get("train_config", {})
    return int(train_cfg.get("frame_size", 128))


def predict_policy(
    model: IsaacCNNPolicy,
    device: torch.device,
    observation: np.ndarray,
    checkpoint: dict | None = None,
    nav_hint: int | None = None,
) -> PolicyPrediction:
    """Run a forward pass and return structured predictions.

    nav_hint is an integer class (0=STAY, 1=N, 2=S, 3=W, 4=E) — required when
    the model was trained with ``use_nav_hint_embedding=True``; if omitted the
    model falls back to STAY (class 0).
    """
    obs_f = torch.from_numpy(observation).float() / 255.0
    train_cfg = checkpoint.get("train_config", {}) if checkpoint else {}
    if train_cfg.get("motion_channels", False) and obs_f.shape[0] > 1:
        stack_size = obs_f.shape[0]
        diffs = [
            (obs_f[t:t + 1] - obs_f[t - 1:t]).abs()
            for t in range(1, stack_size)
        ]
        obs_f = torch.cat([obs_f, *diffs], dim=0)
    observation_tensor = obs_f.unsqueeze(0).to(device)

    nav_hint_tensor: torch.Tensor | None = None
    if nav_hint is not None:
        nav_hint_tensor = torch.tensor([nav_hint], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(observation_tensor, nav_hint=nav_hint_tensor)

    movement_names = checkpoint.get("movement_names", MOVEMENT_NAMES) if checkpoint else MOVEMENT_NAMES
    shooting_names = checkpoint.get("shooting_names", SHOOTING_NAMES) if checkpoint else SHOOTING_NAMES
    bomb_names = checkpoint.get("bomb_names", BOMB_NAMES) if checkpoint else BOMB_NAMES

    return PolicyPrediction(
        movement=_decode_head(logits["movement"], movement_names),
        shooting=_decode_head(logits["shooting"], shooting_names),
        bomb=_decode_head(logits["bomb"], bomb_names),
        device=str(device),
    )


def nav_hint_from_room_graph(
    room_graph: "RoomGraph",
    current_room_index: int,
) -> int:
    """Compute a nav_hint integer from the live room graph.

    Returns an integer in [0, 4] matching the NavHint enum
    (0=STAY, 1=N, 2=S, 3=W, 4=E).  Use this to feed nav_hint into
    predict_policy() during live play.
    """
    return int(room_graph.nav_hint(current_room_index))


def _decode_head(
    logits: torch.Tensor,
    label_map: dict[int, str],
) -> HeadPrediction:
    probabilities = torch.softmax(logits[0], dim=0)
    index = int(torch.argmax(probabilities).item())
    confidence = float(probabilities[index].item())
    return HeadPrediction(
        index=index,
        label=label_map[index],
        confidence=confidence,
        probabilities=tuple(float(p) for p in probabilities.tolist()),
    )


def prediction_to_action(
    prediction: PolicyPrediction,
    movement_threshold: float = 0.0,
    shooting_threshold: float = 0.0,
    bomb_threshold: float = 1.1,
) -> IsaacAction:
    movement = prediction.movement.index if prediction.movement.confidence >= movement_threshold else 0
    shooting = prediction.shooting.index if prediction.shooting.confidence >= shooting_threshold else 0
    bomb = prediction.bomb.index if prediction.bomb.confidence >= bomb_threshold else 0
    return IsaacAction(movement=movement, shooting=shooting, bomb=bomb)
