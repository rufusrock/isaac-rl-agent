from __future__ import annotations

import concurrent.futures
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from binding_rl_agent.preprocessing import FRAME_TRANSFORMS, resize_frame_rgb

# Module-level dict so each DataLoader worker opens memmap files once.
# Keyed by (pid, cache_path_str) so workers don't share state.
_WORKER_FRAME_CACHE: dict[tuple[int, str], np.memmap] = {}


def _build_or_load_frame_cache(
    run_dir: Path,
    raw_frames: np.ndarray,
    frame_transform: Callable,
    frame_mode: str,
    frame_size: int,
    cache_dir: Path,
    num_threads: int | None = None,
) -> tuple[Path, tuple[int, ...], str]:
    """Build (or load) a pre-processed frame cache for one rollout run.

    Returns (cache_path, shape, dtype_str).  The caller can then open the file
    as ``np.memmap(cache_path, dtype=dtype_str, mode='r', shape=shape)``.
    """
    stem = f"{run_dir.name}_{frame_mode}_{frame_size}"
    cache_path = cache_dir / f"{stem}.dat"
    meta_path  = cache_dir / f"{stem}.meta.json"

    if cache_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        shape = tuple(meta["shape"])
        dtype = meta["dtype"]
        return cache_path, shape, dtype

    # --- Build cache ---
    sample = frame_transform(raw_frames[0])
    frame_shape: tuple[int, ...] = sample.shape
    dtype = sample.dtype.str  # e.g. "|u1" for uint8

    n_frames = len(raw_frames)
    full_shape = (n_frames, *frame_shape)

    print(f"  [cache] Building {stem}  ({n_frames} frames, shape={full_shape}) ...", flush=True)

    mm = np.memmap(cache_path, dtype=sample.dtype, mode="w+", shape=full_shape)

    workers = num_threads or min(16, os.cpu_count() or 4)

    def _process(i: int) -> None:
        mm[i] = frame_transform(raw_frames[i])

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        # Report progress every ~10%
        total = n_frames
        step  = max(1, total // 10)
        futs  = {pool.submit(_process, i): i for i in range(total)}
        done  = 0
        for fut in concurrent.futures.as_completed(futs):
            fut.result()
            done += 1
            if done % step == 0 or done == total:
                print(f"    {done}/{total} frames cached", flush=True)

    mm.flush()
    del mm  # close write handle

    meta_path.write_text(
        json.dumps({"shape": list(full_shape), "dtype": dtype,
                    "frame_mode": frame_mode, "frame_size": frame_size}),
        encoding="utf-8",
    )
    print(f"  [cache] Done -> {cache_path}", flush=True)
    return cache_path, full_shape, dtype


MOVEMENT_NAMES: dict[int, str] = {
    0: "idle", 1: "up", 2: "down", 3: "left", 4: "right",
}

SHOOTING_NAMES: dict[int, str] = {
    0: "idle", 1: "up", 2: "down", 3: "left", 4: "right",
}

BOMB_NAMES: dict[int, str] = {
    0: "no_bomb", 1: "bomb",
}


@dataclass(frozen=True)
class DatasetSummary:
    num_runs: int
    num_samples: int
    observation_shape: tuple[int, ...]
    movement_counts: dict[int, int]
    shooting_counts: dict[int, int]
    bomb_counts: dict[int, int]
    run_dirs: list[str]
    has_nav_hints: bool = False


class IsaacRolloutDataset(
    Dataset[tuple[torch.Tensor, dict[str, torch.Tensor]]]
):
    def __init__(
        self,
        rollouts_dir: str | Path = "rollouts",
        max_runs: int | None = None,
        stack_size: int = 4,
        frame_size: int = 128,
        frame_mode: str = "gray",    # see FRAME_TRANSFORMS keys
        motion_channels: bool = False,
        cache_dir: str | Path | None = None,
        exclude_runs: tuple[str, ...] = (),
        include_runs: tuple[str, ...] | None = None,
    ) -> None:
        self.rollouts_dir = Path(rollouts_dir)
        self.stack_size = stack_size
        self.frame_size = frame_size
        self.frame_mode = frame_mode
        self.motion_channels = motion_channels
        self.frame_transform: Callable = FRAME_TRANSFORMS[frame_mode]

        self.run_dirs = self._discover_runs(
            self.rollouts_dir,
            max_runs=max_runs,
            exclude_runs=exclude_runs,
            include_runs=include_runs,
        )
        (
            raw_frames_per_run,
            self.movement_actions,
            self.shooting_actions,
            self.bomb_actions,
            self.nav_hints,
            self.sample_to_run,
            self.sample_to_local,
        ) = self._load_rollouts(self.run_dirs, frame_size=frame_size)

        # Build or load pre-processed frame cache if requested.
        # Storing only paths+shapes keeps the dataset picklable for DataLoader workers.
        if cache_dir is not None:
            cache_root = Path(cache_dir)
            cache_root.mkdir(parents=True, exist_ok=True)
            self._cache_paths: list[Path] = []
            self._cache_shapes: list[tuple[int, ...]] = []
            self._cache_dtype: str = "uint8"
            for run_idx, run_dir in enumerate(self.run_dirs):
                path, shape, dtype = _build_or_load_frame_cache(
                    run_dir=run_dir,
                    raw_frames=raw_frames_per_run[run_idx],
                    frame_transform=self.frame_transform,
                    frame_mode=frame_mode,
                    frame_size=frame_size,
                    cache_dir=cache_root,
                )
                self._cache_paths.append(path)
                self._cache_shapes.append(shape)
                self._cache_dtype = dtype
            # Release raw frames — cache on disk is the source of truth now.
            del raw_frames_per_run
            self.raw_frames_per_run: list[np.ndarray] = []
            self._use_cache = True
        else:
            self.raw_frames_per_run = raw_frames_per_run
            self._use_cache = False

        sample_obs, _ = self[0]
        self.summary = DatasetSummary(
            num_runs=len(self.run_dirs),
            num_samples=len(self.movement_actions),
            observation_shape=tuple(sample_obs.shape),
            movement_counts=self._count_actions(self.movement_actions),
            shooting_counts=self._count_actions(self.shooting_actions),
            bomb_counts=self._count_actions(self.bomb_actions),
            run_dirs=[str(path) for path in self.run_dirs],
            has_nav_hints=self.nav_hints is not None,
        )

    def __len__(self) -> int:
        return len(self.movement_actions)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        run_idx = int(self.sample_to_run[index])
        local_idx = int(self.sample_to_local[index])

        # Gather up to stack_size frames ending at frame local_idx+1
        end = local_idx + 2
        start = max(0, end - self.stack_size)

        if self._use_cache:
            # Lazily open memmap per worker (keyed by pid so workers don't share state)
            cache_key = (os.getpid(), str(self._cache_paths[run_idx]))
            if cache_key not in _WORKER_FRAME_CACHE:
                _WORKER_FRAME_CACHE[cache_key] = np.memmap(
                    self._cache_paths[run_idx],
                    dtype=self._cache_dtype,
                    mode="r",
                    shape=self._cache_shapes[run_idx],
                )
            cached = _WORKER_FRAME_CACHE[cache_key]
            frames = np.array(cached[start:end])  # copy slice from memmap
            if frames.shape[0] < self.stack_size:
                pad_n = self.stack_size - frames.shape[0]
                pad = np.repeat(np.array(cached[:1]), pad_n, axis=0)
                frames = np.concatenate([pad, frames], axis=0)
            processed = [frames[i] for i in range(self.stack_size)]
        else:
            raw = self.raw_frames_per_run[run_idx]  # (M, 3, H, W)
            frames = raw[start:end]
            if len(frames) < self.stack_size:
                pad_n = self.stack_size - len(frames)
                frames = np.concatenate(
                    [np.repeat(raw[:1], pad_n, axis=0), frames], axis=0
                )
            processed = [self.frame_transform(f) for f in frames]

        # Stack: list of (C,H,W) → (stack*C, H, W) or list of (H,W) → (stack, H, W)
        if processed[0].ndim == 2:
            stacked = np.stack(processed, axis=0)           # (stack, H, W)
        else:
            stacked = np.concatenate(processed, axis=0)     # (stack*C, H, W)

        # Return uint8 when possible — GPU normalises (/255) after transfer,
        # keeping worker batches 4× smaller and the pipeline fast.
        # motion_channels needs float for the diff; handle that path here.
        if self.motion_channels and self.stack_size > 1:
            obs_f = torch.from_numpy(stacked).float() / 255.0
            c = obs_f.shape[0] // self.stack_size
            diffs = [
                (obs_f[t * c:(t + 1) * c] - obs_f[(t - 1) * c:t * c]).abs()
                for t in range(1, self.stack_size)
            ]
            observation = torch.cat([obs_f] + diffs, dim=0)
        else:
            observation = torch.from_numpy(stacked)  # uint8, normalised on GPU

        targets = {
            "movement": torch.tensor(int(self.movement_actions[index]), dtype=torch.long),
            "shooting": torch.tensor(int(self.shooting_actions[index]), dtype=torch.long),
            "bomb":     torch.tensor(int(self.bomb_actions[index]),     dtype=torch.long),
        }
        if self.nav_hints is not None:
            targets["nav_hint"] = torch.tensor(int(self.nav_hints[index]), dtype=torch.long)
        return observation, targets

    @staticmethod
    def _discover_runs(
        rollouts_dir: Path,
        max_runs: int | None = None,
        exclude_runs: tuple[str, ...] = (),
        include_runs: tuple[str, ...] | None = None,
    ) -> list[Path]:
        if not rollouts_dir.exists():
            raise FileNotFoundError(f"Rollouts directory does not exist: {rollouts_dir}")
        if include_runs is not None and exclude_runs:
            raise ValueError("Pass include_runs OR exclude_runs, not both.")
        run_dirs = sorted(
            path for path in rollouts_dir.iterdir()
            if path.is_dir() and (path / "rollout_data.npz").exists()
        )
        if include_runs is not None:
            include_set = set(include_runs)
            run_dirs = [p for p in run_dirs if p.name in include_set]
            missing = include_set - {p.name for p in run_dirs}
            if missing:
                raise FileNotFoundError(f"include_runs not found: {sorted(missing)}")
        elif exclude_runs:
            exclude_set = set(exclude_runs)
            run_dirs = [p for p in run_dirs if p.name not in exclude_set]
        if not run_dirs:
            raise FileNotFoundError(f"No rollout runs found under {rollouts_dir}")
        if max_runs is not None:
            run_dirs = run_dirs[:max_runs]
        return run_dirs

    @staticmethod
    def _load_rollouts(
        run_dirs: list[Path],
        frame_size: int = 128,
    ) -> tuple[
        list[np.ndarray],   # raw_frames_per_run
        np.ndarray,         # movement_actions
        np.ndarray,         # shooting_actions
        np.ndarray,         # bomb_actions
        np.ndarray | None,  # nav_hints
        np.ndarray,         # sample_to_run
        np.ndarray,         # sample_to_local
    ]:
        raw_frames_per_run: list[np.ndarray] = []
        movement_list: list[np.ndarray] = []
        shooting_list: list[np.ndarray] = []
        bomb_list: list[np.ndarray] = []
        nav_hints_list: list[np.ndarray] = []
        sample_to_run_list: list[np.ndarray] = []
        sample_to_local_list: list[np.ndarray] = []
        any_nav_hints = False

        for run_idx, run_dir in enumerate(run_dirs):
            data = np.load(run_dir / "rollout_data.npz")
            metadata = load_metadata(run_dir)
            movement_actions, shooting_actions, bomb_actions = decode_action_heads(
                data=data,
                schema_version=int(metadata.get("action_schema_version", 1)),
            )
            n_actions = len(movement_actions)

            if "raw_frames" in data:
                raw = data["raw_frames"]  # (M, 3, H, W) uint8 RGB
                if len(raw) != n_actions + 1:
                    raise ValueError(
                        f"Run {run_dir}: {len(raw)} raw frames vs {n_actions} actions"
                    )
                # Resize if needed
                H = raw.shape[2]
                if H != frame_size:
                    raw = np.stack(
                        [resize_frame_rgb(f, frame_size) for f in raw], axis=0
                    )
            else:
                raise ValueError(
                    f"Run {run_dir} uses legacy stacked format which is no longer supported. "
                    "Please re-record rollouts."
                )

            raw_frames_per_run.append(raw)
            movement_list.append(movement_actions)
            shooting_list.append(shooting_actions)
            bomb_list.append(bomb_actions)
            sample_to_run_list.append(np.full(n_actions, run_idx, dtype=np.int32))
            sample_to_local_list.append(np.arange(n_actions, dtype=np.int32))

            if "nav_hints" in data:
                nav_hints_list.append(data["nav_hints"].astype(np.int64, copy=False))
                any_nav_hints = True
            else:
                nav_hints_list.append(np.zeros(n_actions, dtype=np.int64))

        movement = np.concatenate(movement_list).astype(np.int64, copy=False)
        shooting = np.concatenate(shooting_list).astype(np.int64, copy=False)
        bomb = np.concatenate(bomb_list).astype(np.int64, copy=False)
        nav_hints = np.concatenate(nav_hints_list) if any_nav_hints else None
        sample_to_run = np.concatenate(sample_to_run_list)
        sample_to_local = np.concatenate(sample_to_local_list)

        return (
            raw_frames_per_run,
            movement, shooting, bomb,
            nav_hints,
            sample_to_run, sample_to_local,
        )

    @staticmethod
    def _count_actions(actions: np.ndarray) -> dict[int, int]:
        unique, counts = np.unique(actions, return_counts=True)
        return {int(a): int(c) for a, c in zip(unique, counts, strict=True)}


def load_metadata(run_dir: str | Path) -> dict:
    path = Path(run_dir) / "metadata.json"
    return json.loads(path.read_text(encoding="utf-8"))


def decode_action_heads(
    data: np.lib.npyio.NpzFile,
    schema_version: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if schema_version >= 3:
        return (
            data["movement_actions"].astype(np.int64, copy=False),
            data["shooting_actions"].astype(np.int64, copy=False),
            data["bomb_actions"].astype(np.int64, copy=False),
        )
    flat_actions = data["actions"].astype(np.int64, copy=False)
    return remap_flat_actions_to_heads(flat_actions, schema_version)


def remap_flat_actions_to_heads(
    actions: np.ndarray,
    schema_version: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    movement = np.zeros(actions.shape, dtype=np.int64)
    shooting = np.zeros(actions.shape, dtype=np.int64)
    bomb = np.zeros(actions.shape, dtype=np.int64)

    if schema_version >= 2:
        mapping = {
            0: (0, 0, 0), 1: (1, 0, 0), 2: (2, 0, 0), 3: (3, 0, 0), 4: (4, 0, 0),
            5: (0, 1, 0), 6: (0, 2, 0), 7: (0, 3, 0), 8: (0, 4, 0), 9: (0, 0, 1),
        }
    else:
        mapping = {
            0: (0, 0, 0), 1: (1, 0, 0), 2: (2, 0, 0),  3: (3, 0, 0),  4: (4, 0, 0),
            5: (0, 1, 0), 6: (0, 2, 0), 7: (0, 3, 0),  8: (0, 4, 0),  9: (1, 1, 0),
           10: (1, 3, 0),11: (1, 4, 0),12: (2, 2, 0), 13: (2, 3, 0), 14: (2, 4, 0),
           15: (3, 3, 0),16: (4, 4, 0),
        }

    for idx, action in enumerate(actions):
        action_id = int(action)
        if action_id not in mapping:
            raise ValueError(f"Unknown action id {action_id} for schema version {schema_version}")
        move, shoot, drop_bomb = mapping[action_id]
        movement[idx] = move
        shooting[idx] = shoot
        bomb[idx] = drop_bomb

    return movement, shooting, bomb
