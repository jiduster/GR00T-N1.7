"""Attach precomputed DexWM DINO features to GR00T single-step shards.

The sidecar is one memory-mapped ``episode-XXXXXX.features.npy`` per episode,
with shape ``[T, 448, 1024]`` float16, aligned to parquet frame indices. Each
training step at index ``t`` reads a DexWM teacher-forcing window of 9 frames
and 8 transitions. With ``action_stride=s`` that window is

    features[t], features[t+s], ..., features[t+8s]
    actions[t+s-1], actions[t+2s-1], ..., actions[t+8s-1]

so stride=1 is consecutive (the original VLA aux path) and stride=5 linspaces
the 8 hops across GR00T's 40-step action chunk. The loss module gathers the
same action indices from the sampled VLA chunk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gr00t.data.dataset.sharded_single_step_dataset import (
    ShardedSingleStepDataset,
    extract_step_data,
)
from gr00t.data.types import MessageType
from gr00t.model.dexwm_auxiliary import (
    DEXWM_CONTEXT_FRAMES,
    RAW_ACTION_DIM,
    RAW_STATE_DIM,
    dexwm_window_offsets,
    max_dexwm_action_index,
)

BIMANUAL_ACTION_GROUPS = ("right_tcp", "right_hand", "left_tcp", "left_hand")
BIMANUAL_STATE_GROUPS = ("right_tcp", "left_tcp", "right_hand", "left_hand")


def _concat_groups(data: dict[str, np.ndarray], keys: tuple[str, ...]) -> np.ndarray:
    parts = []
    for key in keys:
        if key not in data:
            raise KeyError(f"Missing joint group '{key}' while building DexWM sidecar tensors")
        parts.append(np.asarray(data[key], dtype=np.float32))
    return np.concatenate(parts, axis=-1)


def _sidecar_path(feature_root: Path, episode_index: int) -> Path:
    return feature_root / f"episode-{int(episode_index):06d}.features.npy"


class DexWMSidecarDataset(ShardedSingleStepDataset):
    """``ShardedSingleStepDataset`` that appends DexWM sidecar tensors."""

    def __init__(
        self,
        *args: Any,
        feature_root: str | Path,
        context_frames: int = DEXWM_CONTEXT_FRAMES,
        action_stride: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.feature_root = Path(feature_root).expanduser().resolve()
        if not self.feature_root.exists():
            raise FileNotFoundError(f"DexWM feature root does not exist: {self.feature_root}")
        self.context_frames = int(context_frames)
        self.action_stride = int(action_stride)
        last_action = max_dexwm_action_index(self.action_stride, self.context_frames)
        if last_action >= int(self.action_horizon):
            raise ValueError(
                f"dexwm_action_stride={self.action_stride} needs action index {last_action}, "
                f"but action_horizon={self.action_horizon}"
            )
        self._feature_memmaps: dict[str, np.memmap] = {}
        self._frame_offsets, self._action_offsets = dexwm_window_offsets(
            self.action_stride, self.context_frames
        )

    def _memmap_features(self, episode_index: int) -> np.memmap:
        path = _sidecar_path(self.feature_root, episode_index)
        key = str(path)
        cached = self._feature_memmaps.get(key)
        if cached is not None:
            return cached
        if not path.exists():
            raise FileNotFoundError(
                f"DexWM sidecar missing for episode {episode_index}: {path}"
            )
        features = np.load(path, mmap_mode="r")
        if features.ndim != 3 or features.shape[1:] != (448, 1024):
            raise ValueError(
                f"Expected sidecar shape [T, 448, 1024] for {path}, got {tuple(features.shape)}"
            )
        self._feature_memmaps[key] = features
        return features

    def _feature_window(self, episode_index: int, step_index: int, episode_length: int) -> tuple[np.ndarray, np.ndarray]:
        features = self._memmap_features(episode_index)
        length = min(int(features.shape[0]), int(episode_length))
        last = max(length - 1, 0)
        frame_offsets, action_offsets = dexwm_window_offsets(
            getattr(self, "action_stride", 1), self.context_frames
        )
        indices = np.clip(int(step_index) + frame_offsets, 0, last)
        window = np.stack([np.asarray(features[int(i)]) for i in indices], axis=0)
        valid = (int(step_index) + action_offsets + 1) < length
        return window.astype(np.float16, copy=False), valid.astype(np.float32)

    def get_datapoint(
        self,
        episode_data: pd.DataFrame,
        step_index: int,
        episode_index: int | None = None,
        episode_length: int | None = None,
    ) -> dict:
        assert self.processor is not None, "Processor must be set before getting datapoints"
        vla_step_data = extract_step_data(
            episode_data,
            step_index,
            self.modality_configs,
            self.embodiment_tag,
            self.allow_padding,
        )
        processed = self.processor(
            [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
        )
        if episode_index is None:
            raise ValueError("DexWM sidecar loading requires episode_index")
        if episode_length is None:
            episode_length = len(episode_data)

        features, valid_mask = self._feature_window(episode_index, int(step_index), int(episode_length))
        raw_state = _concat_groups(vla_step_data.states, BIMANUAL_STATE_GROUPS)
        raw_action = _concat_groups(vla_step_data.actions, BIMANUAL_ACTION_GROUPS)
        if raw_state.ndim == 2:
            raw_state = raw_state[-1]
        if raw_state.shape[-1] != RAW_STATE_DIM:
            raise ValueError(f"Expected raw state dim {RAW_STATE_DIM}, got {raw_state.shape}")
        if raw_action.shape[-1] != RAW_ACTION_DIM:
            raise ValueError(f"Expected raw action dim {RAW_ACTION_DIM}, got {raw_action.shape}")
        available_actions = int(raw_action.shape[0])
        last_action = max(available_actions - 1, 0)
        action_offsets = dexwm_window_offsets(self.action_stride, self.context_frames)[1]
        gathered = []
        for i, offset in enumerate(action_offsets):
            idx = int(offset)
            if idx >= available_actions:
                valid_mask[i] = 0.0
            gathered.append(raw_action[min(idx, last_action)])
        gt_action = np.stack(gathered, axis=0)

        processed["dexwm_features"] = np.ascontiguousarray(features)
        processed["dexwm_valid_mask"] = np.ascontiguousarray(valid_mask)
        processed["dexwm_state"] = np.ascontiguousarray(raw_state.astype(np.float32))
        processed["dexwm_gt_action"] = np.ascontiguousarray(gt_action.astype(np.float32))
        return processed

    def get_shard(self, idx: int) -> list:
        episodes = self.sharded_episodes[idx]
        datapoints = []
        for ep_idx, step_indices in episodes:
            episode_data = self.episode_loader[ep_idx]
            episode_meta = self.episode_loader.episodes_metadata[ep_idx]
            episode_id = int(episode_meta["episode_index"])
            episode_length = int(self.episode_loader.get_episode_length(ep_idx))
            for step_index in step_indices:
                datapoints.append(
                    self.get_datapoint(
                        episode_data,
                        int(step_index),
                        episode_index=episode_id,
                        episode_length=episode_length,
                    )
                )
        return datapoints
