from __future__ import annotations

import torch
from torch import nn

NAV_HINT_CLASSES = 5
NAV_HINT_EMBED_DIM = 16


def _make_norm(norm_type: str, channels: int) -> nn.Module | None:
    if norm_type == "batch":
        return nn.BatchNorm2d(channels)
    if norm_type == "layer":
        # GroupNorm with 1 group == LayerNorm over (C,H,W) per sample
        return nn.GroupNorm(1, channels)
    if norm_type == "group":
        groups = 8 if channels % 8 == 0 else (4 if channels % 4 == 0 else 1)
        return nn.GroupNorm(groups, channels)
    if norm_type == "none":
        return None
    raise ValueError(f"Unknown norm_type: {norm_type}")


class _PlainCNN(nn.Module):
    def __init__(self, in_ch: int, conv_channels: tuple[int, ...], norm_type: str) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        ch = in_ch
        for out_ch in conv_channels:
            conv_bias = norm_type == "none"
            layers.append(nn.Conv2d(ch, out_ch, kernel_size=3, stride=2, bias=conv_bias))
            norm = _make_norm(norm_type, out_ch)
            if norm is not None:
                layers.append(norm)
            layers.append(nn.ReLU())
            ch = out_ch
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ImpalaResBlock(nn.Module):
    def __init__(self, channels: int, norm_type: str) -> None:
        super().__init__()
        conv_bias = norm_type == "none"
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=conv_bias)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=conv_bias)
        self.n1 = _make_norm(norm_type, channels)
        self.n2 = _make_norm(norm_type, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(x)
        h = self.conv1(h)
        if self.n1 is not None:
            h = self.n1(h)
        h = torch.relu(h)
        h = self.conv2(h)
        if self.n2 is not None:
            h = self.n2(h)
        return x + h


class _ImpalaCNN(nn.Module):
    """Impala-style 3-stage CNN. Each stage: conv -> maxpool -> 2x residual blocks.
    conv_channels = (c1, c2, c3) sets per-stage output channels."""

    def __init__(
        self,
        in_ch: int,
        conv_channels: tuple[int, ...],
        norm_type: str,
        num_resblocks: int = 2,
    ) -> None:
        super().__init__()
        stages: list[nn.Module] = []
        ch = in_ch
        for out_ch in conv_channels:
            conv_bias = norm_type == "none"
            stage_layers: list[nn.Module] = [
                nn.Conv2d(ch, out_ch, kernel_size=3, padding=1, bias=conv_bias),
            ]
            norm = _make_norm(norm_type, out_ch)
            if norm is not None:
                stage_layers.append(norm)
            stage_layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            for _ in range(num_resblocks):
                stage_layers.append(_ImpalaResBlock(out_ch, norm_type))
            stages.append(nn.Sequential(*stage_layers))
            ch = out_ch
        self.net = nn.Sequential(*stages, nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _NatureCNN(nn.Module):
    """Classic DQN Nature-paper CNN: 8x8/s4 -> 4x4/s2 -> 3x3/s1."""

    def __init__(self, in_ch: int, conv_channels: tuple[int, ...], norm_type: str) -> None:
        super().__init__()
        if len(conv_channels) != 3:
            raise ValueError("NatureCNN expects 3 conv channel sizes.")
        c1, c2, c3 = conv_channels
        conv_bias = norm_type == "none"
        layers: list[nn.Module] = []

        def _add_block(inp: int, outp: int, k: int, s: int) -> None:
            layers.append(nn.Conv2d(inp, outp, kernel_size=k, stride=s, bias=conv_bias))
            norm = _make_norm(norm_type, outp)
            if norm is not None:
                layers.append(norm)
            layers.append(nn.ReLU())

        _add_block(in_ch, c1, 8, 4)
        _add_block(c1, c2, 4, 2)
        _add_block(c2, c3, 3, 1)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _build_feature_extractor(
    arch: str,
    in_ch: int,
    conv_channels: tuple[int, ...],
    norm_type: str,
    num_resblocks: int = 2,
) -> nn.Module:
    if arch == "plain":
        return _PlainCNN(in_ch, conv_channels, norm_type)
    if arch == "impala":
        return _ImpalaCNN(in_ch, conv_channels, norm_type, num_resblocks=num_resblocks)
    if arch == "nature":
        return _NatureCNN(in_ch, conv_channels, norm_type)
    raise ValueError(f"Unknown arch: {arch}")


class IsaacCNNPolicy(nn.Module):
    def __init__(
        self,
        input_channels: int = 4,
        num_movement_actions: int = 5,
        num_shooting_actions: int = 5,
        num_bomb_actions: int = 2,
        conv_channels: tuple[int, ...] = (8, 16, 16),
        hidden_dim: int = 128,
        dropout: float = 0.0,
        input_size: int = 128,
        use_nav_hint_embedding: bool = False,
        use_batchnorm: bool = True,
        arch: str = "plain",
        norm_type: str | None = None,
        num_resblocks: int = 2,
    ) -> None:
        super().__init__()
        self.use_nav_hint_embedding = use_nav_hint_embedding

        if norm_type is None:
            norm_type = "batch" if use_batchnorm else "none"

        self.features = _build_feature_extractor(
            arch=arch,
            in_ch=input_channels,
            conv_channels=conv_channels,
            norm_type=norm_type,
            num_resblocks=num_resblocks,
        )

        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_size, input_size)
            flat_size = self.features(dummy).numel()

        if use_nav_hint_embedding:
            self.nav_hint_embedding = nn.Embedding(NAV_HINT_CLASSES, NAV_HINT_EMBED_DIM)
            trunk_in_size = flat_size + NAV_HINT_EMBED_DIM
        else:
            self.nav_hint_embedding = None  # type: ignore[assignment]
            trunk_in_size = flat_size

        trunk_layers: list[nn.Module] = [nn.Linear(trunk_in_size, hidden_dim), nn.ReLU()]
        if dropout > 0.0:
            trunk_layers.append(nn.Dropout(p=dropout))
        self._trunk_linear = nn.Sequential(*trunk_layers)
        self._flatten = nn.Flatten()

        self.movement_head = nn.Linear(hidden_dim, num_movement_actions)
        self.shooting_head = nn.Linear(hidden_dim, num_shooting_actions)
        self.bomb_head = nn.Linear(hidden_dim, num_bomb_actions)

    def forward(
        self,
        observations: torch.Tensor,
        nav_hint: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        cnn_out = self.features(observations)
        flat = self._flatten(cnn_out)

        if self.use_nav_hint_embedding and nav_hint is not None:
            emb = self.nav_hint_embedding(nav_hint)
            flat = torch.cat([flat, emb], dim=1)
        elif self.use_nav_hint_embedding and nav_hint is None:
            stay = torch.zeros(flat.shape[0], dtype=torch.long, device=flat.device)
            emb = self.nav_hint_embedding(stay)
            flat = torch.cat([flat, emb], dim=1)

        hidden = self._trunk_linear(flat)
        return {
            "movement": self.movement_head(hidden),
            "shooting": self.shooting_head(hidden),
            "bomb": self.bomb_head(hidden),
        }
