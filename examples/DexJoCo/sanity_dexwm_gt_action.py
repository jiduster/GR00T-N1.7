#!/usr/bin/env python3
"""Sanity-check frozen DexWM teacher forcing on GR00T DexJoCo data.

Phase 1 of the handoff: feed ground-truth 44D actions through the torch FK
adapter and the frozen DexWM, without updating GR00T. If this loss is already
huge or non-finite, do not start VLA training.

Example:

  cd /data/home/zyh/Isaac-GR00T
  uv run python examples/DexJoCo/sanity_dexwm_gt_action.py --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path("/data/home/zyh/Isaac-GR00T")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gr00t.model.dexjoco_raw_adapter import DexJoCoRawActionAdapter  # noqa: E402
from gr00t.model.dexwm_auxiliary import (  # noqa: E402
    DEFAULT_DEXWM_CHECKPOINT,
    FrozenDexWMAuxiliary,
    GroupWiseUnnormalizer,
    dexwm_window_offsets,
)

ACTION_GROUPS = (
    ("right_tcp", 0, 6),
    ("right_hand", 6, 22),
    ("left_tcp", 22, 28),
    ("left_hand", 28, 44),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        default="/mnt/ceph2/DexJoCo-Datasets-LeRobot/dexjoco_lerobot_datasets/bimanual_microwave_cook_gr00t_v2",
    )
    parser.add_argument(
        "--feature-root",
        default="/mnt/ceph2/DexJoCo-Datasets-LeRobot/dexjoco_dino_vla_cache/bimanual_microwave_cook",
    )
    parser.add_argument("--checkpoint", default=DEFAULT_DEXWM_CHECKPOINT)
    parser.add_argument("--stats", default=None, help="Optional GR00T dataset_statistics.json")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="DexWM action stride. 1=consecutive 8 hops; 5=linspace across the 40-step chunk.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    feature_root = Path(args.feature_root)
    parquet_path = dataset_root / f"data/chunk-000/episode_{args.episode:06d}.parquet"
    sidecar_path = feature_root / f"episode-{args.episode:06d}.features.npy"
    meta_path = dataset_root / "meta/episodes.jsonl"
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)
    if not sidecar_path.exists():
        raise FileNotFoundError(sidecar_path)

    episode_length = None
    with meta_path.open() as handle:
        for line in handle:
            record = json.loads(line)
            if int(record["episode_index"]) == args.episode:
                episode_length = int(record["length"])
                break
    if episode_length is None:
        raise ValueError(f"Episode {args.episode} not found in {meta_path}")

    df = pd.read_parquet(parquet_path)
    step = int(args.step)
    stride = int(args.stride)
    frame_offsets, action_offsets = dexwm_window_offsets(stride)
    last_frame = step + int(frame_offsets[-1])
    if last_frame >= episode_length:
        raise ValueError(
            f"step {step} with stride {stride} needs frame {last_frame} "
            f"in episode of length {episode_length}"
        )

    state = np.asarray(df["observation.state"].iloc[step], dtype=np.float32)
    actions = np.stack(
        [np.asarray(df["action"].iloc[step + int(i)], dtype=np.float32) for i in action_offsets],
        axis=0,
    )
    feature_indices = step + frame_offsets
    features = np.load(sidecar_path, mmap_mode="r")[feature_indices].astype(np.float32)
    print(
        f"episode={args.episode} step={step} stride={stride} T={episode_length} "
        f"state={state.shape} actions={actions.shape} features={features.shape} "
        f"frame_offsets={frame_offsets.tolist()} action_offsets={action_offsets.tolist()}",
        flush=True,
    )

    state_t = torch.from_numpy(np.array(state, copy=True))[None]
    action_t = torch.from_numpy(np.array(actions, copy=True))[None]
    adapter = DexJoCoRawActionAdapter(layout="dual_rotvec")
    action132 = adapter(action_t, state_t)
    print(
        f"adapter output shape={tuple(action132.shape)} "
        f"abs_mean={action132.abs().mean().item():.6f} "
        f"abs_max={action132.abs().max().item():.6f}",
        flush=True,
    )
    if not torch.isfinite(action132).all():
        raise SystemExit("adapter produced non-finite DexWM actions")

    if args.device.startswith("cpu"):
        print("skipping DexWM forward on CPU (flex-attention masks are GPU-oriented)", flush=True)
        return

    stats_path = args.stats
    if stats_path is None:
        stats_path = (
            "/mnt/ceph2/ckpt/gr00t_n1d7_dexjoco_bimanual_microwave/"
            "official_ft_8gpu/experiment_cfg/dataset_statistics.json"
        )
    stats = json.loads(Path(stats_path).read_text())["new_embodiment"]["action"]
    action_low = np.concatenate([np.asarray(stats[name]["q01"], dtype=np.float32) for name, _, _ in ACTION_GROUPS])
    action_high = np.concatenate([np.asarray(stats[name]["q99"], dtype=np.float32) for name, _, _ in ACTION_GROUPS])
    unnormalizer = GroupWiseUnnormalizer(action_low, action_high)
    aux = FrozenDexWMAuxiliary(
        args.checkpoint,
        unnormalizer,
        device=args.device,
        dtype=args.dtype,
    )
    loss = aux(
        torch.from_numpy(features)[None].to(args.device),
        action_t.to(args.device),
        state_t.to(args.device),
        actions_are_normalized=False,
    )
    print(f"GT-action DexWM teacher-forcing loss: {float(loss):.6f}", flush=True)
    if not torch.isfinite(loss):
        raise SystemExit("DexWM GT-action loss is not finite")


if __name__ == "__main__":
    main()
