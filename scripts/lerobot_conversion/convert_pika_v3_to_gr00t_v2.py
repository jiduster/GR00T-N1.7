#!/usr/bin/env python

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert Pika LeRobot v3 image-parquet datasets to GR00T-flavored LeRobot v2.

This converter is specialized for datasets whose v3 camera observations are stored
inside parquet files as ``struct<bytes, path>`` image columns instead of standard
LeRobot v3 concatenated MP4 files. GR00T training expects per-episode parquet
files plus per-episode MP4 videos, so this script:

1. Splits consolidated v3 parquet shards into v2 per-episode parquet files.
2. Materializes embedded image bytes into per-episode MP4 files.
3. Writes GR00T-specific metadata:
   - ``meta/episodes.jsonl``
   - ``meta/tasks.jsonl``
   - ``meta/modality.json``
4. Rewrites ``meta/info.json`` to a v2-compatible layout.

Typical usage:

    python scripts/lerobot_conversion/convert_pika_v3_to_gr00t_v2.py \
      --src-root /mnt/ceph2/SIRIUS_LAB_Pika_TeleOp_Bimanual/lerobot_pidata_pnp \
      --dst-root /tmp/lerobot_pidata_pnp_gr00t_v2
"""

from __future__ import annotations

import argparse
import json
from math import ceil
from pathlib import Path
import shutil
from typing import Any

import cv2
import jsonlines
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


V21 = "v2.1"
V30 = "v3.0"

LEGACY_DATA_PATH_TEMPLATE = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
LEGACY_VIDEO_PATH_TEMPLATE = (
    "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
)

EPISODES_PARQUET_GLOB = "meta/episodes/chunk-*/file-*.parquet"
TASKS_PARQUET_PATH = "meta/tasks.parquet"
INFO_PATH = "meta/info.json"
STATS_PATH = "meta/stats.json"

ANNOTATION_COLUMN = "annotation.human.task_description"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src-root",
        type=Path,
        required=True,
        help="Path to the source LeRobot v3 dataset root.",
    )
    parser.add_argument(
        "--dst-root",
        type=Path,
        required=True,
        help="Path to the output GR00T LeRobot v2 dataset root.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Override output FPS. Defaults to source info.json fps.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional cap for debugging. Converts the first N episodes only.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output directory if it already exists.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=4)


def validate_source_dataset(src_root: Path) -> dict[str, Any]:
    info = load_json(src_root / INFO_PATH)
    version = info.get("codebase_version")
    if version != V30:
        raise ValueError(
            f"Expected source dataset codebase_version={V30!r}, got {version!r} at {src_root}"
        )
    return info


def load_episode_records(src_root: Path, max_episodes: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for pq_path in sorted(src_root.glob(EPISODES_PARQUET_GLOB)):
        table = pq.read_table(pq_path)
        records.extend(table.to_pylist())

    records.sort(key=lambda rec: int(rec["episode_index"]))
    if max_episodes is not None:
        records = records[:max_episodes]
    return records


def load_tasks(src_root: Path) -> list[dict[str, Any]]:
    tasks_table = pq.read_table(src_root / TASKS_PARQUET_PATH)
    tasks = tasks_table.to_pylist()
    tasks.sort(key=lambda item: int(item["task_index"]))
    return tasks


def flatten_feature_names(feature: dict[str, Any]) -> dict[str, Any]:
    names = feature.get("names")
    if (
        isinstance(names, list)
        and len(names) == 1
        and isinstance(names[0], list)
        and all(isinstance(item, str) for item in names[0])
    ):
        feature = dict(feature)
        feature["names"] = names[0]
    return feature


def convert_image_feature_to_video(feature: dict[str, Any], fps: int) -> dict[str, Any]:
    feature = flatten_feature_names(feature)
    shape = feature.get("shape")
    if not isinstance(shape, list) or len(shape) != 3:
        raise ValueError(f"Unexpected image feature shape: {shape}")

    channels, height, width = shape
    if channels != 3:
        raise ValueError(f"Expected RGB image with 3 channels, got shape={shape}")

    return {
        "dtype": "video",
        "shape": [height, width, channels],
        "names": ["height", "width", "channels"],
        "info": {
            "video.height": height,
            "video.width": width,
            "video.codec": "mp4v",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": fps,
            "video.channels": channels,
            "has_audio": False,
        },
    }


def build_output_info(
    source_info: dict[str, Any],
    total_episodes: int,
    fps: int,
) -> tuple[dict[str, Any], list[str]]:
    info = dict(source_info)
    info["codebase_version"] = V21
    info["total_episodes"] = total_episodes
    info["fps"] = fps
    info["data_path"] = LEGACY_DATA_PATH_TEMPLATE
    info["video_path"] = LEGACY_VIDEO_PATH_TEMPLATE
    info["total_chunks"] = ceil(total_episodes / info["chunks_size"]) if total_episodes else 0

    video_keys: list[str] = []
    converted_features: dict[str, Any] = {}
    for feature_name, feature in info["features"].items():
        if feature_name.startswith("observation.images."):
            converted_features[feature_name] = convert_image_feature_to_video(feature, fps)
            video_keys.append(feature_name)
        else:
            converted_features[feature_name] = flatten_feature_names(feature)

    info["features"] = converted_features
    info["total_videos"] = total_episodes * len(video_keys)
    info.pop("data_files_size_in_mb", None)
    info.pop("video_files_size_in_mb", None)
    return info, video_keys


def build_modality_json() -> dict[str, Any]:
    return {
        "state": {
            "left_arm": {"start": 0, "end": 6},
            "right_arm": {"start": 6, "end": 12},
            "left_gripper": {"start": 12, "end": 13},
            "right_gripper": {"start": 13, "end": 14},
        },
        "action": {
            "left_arm": {"start": 0, "end": 6},
            "right_arm": {"start": 6, "end": 12},
            "left_gripper": {"start": 12, "end": 13},
            "right_gripper": {"start": 13, "end": 14},
        },
        "video": {
            "scene": {"original_key": "observation.images.scene"},
            "left_wrist": {"original_key": "observation.images.left_wrist"},
            "right_wrist": {"original_key": "observation.images.right_wrist"},
        },
        "annotation": {
            "human.task_description": {},
        },
    }


def prepare_output_root(dst_root: Path, force: bool) -> None:
    if dst_root.exists():
        if not force:
            raise FileExistsError(
                f"Output directory already exists: {dst_root}. Use --force to overwrite."
            )
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)


def write_tasks_jsonl(dst_root: Path, tasks: list[dict[str, Any]]) -> None:
    path = dst_root / "meta" / "tasks.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(path, mode="w") as writer:
        for task in tasks:
            writer.write(
                {
                    "task_index": int(task["task_index"]),
                    "task": str(task["task"]),
                }
            )


def write_episodes_jsonl(dst_root: Path, episode_records: list[dict[str, Any]]) -> None:
    path = dst_root / "meta" / "episodes.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(path, mode="w") as writer:
        for record in episode_records:
            payload = {
                "episode_index": int(record["episode_index"]),
                "tasks": list(record.get("tasks", [])),
                "length": int(record["length"]),
            }
            writer.write(payload)


def copy_stats_json(src_root: Path, dst_root: Path) -> None:
    src = src_root / STATS_PATH
    if src.exists():
        dst = dst_root / STATS_PATH
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _decode_image_bytes(encoded: bytes) -> np.ndarray:
    frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Failed to decode image bytes with OpenCV")
    return frame


def _write_episode_video(video_path: Path, encoded_frames: list[bytes], fps: int) -> None:
    if not encoded_frames:
        raise ValueError(f"No frames provided for video {video_path}")

    first_frame = _decode_image_bytes(encoded_frames[0])
    height, width = first_frame.shape[:2]

    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {video_path}")

    try:
        writer.write(first_frame)
        for encoded in encoded_frames[1:]:
            frame = _decode_image_bytes(encoded)
            if frame.shape[:2] != (height, width):
                raise ValueError(
                    f"Inconsistent frame size for {video_path}: "
                    f"expected {(height, width)}, got {frame.shape[:2]}"
                )
            writer.write(frame)
    finally:
        writer.release()


def extract_struct_bytes(column: pa.ChunkedArray) -> list[bytes]:
    values: list[bytes | None] = []
    for chunk in column.chunks:
        struct_chunk = pa.StructArray.from_arrays(
            [chunk.field("bytes"), chunk.field("path")],
            names=["bytes", "path"],
        )
        values.extend(struct_chunk.field("bytes").to_pylist())
    if any(value is None for value in values):
        raise ValueError("Encountered null image bytes in source parquet")
    return values  # type: ignore[return-value]


def convert_episode_table(
    episode_table: pa.Table,
    episode_index: int,
    dst_root: Path,
    video_keys: list[str],
    fps: int,
    chunks_size: int,
) -> None:
    arrays = []
    names = []
    video_payloads: dict[str, list[bytes]] = {}

    for name in episode_table.schema.names:
        column = episode_table.column(name)
        if name in video_keys:
            video_payloads[name] = extract_struct_bytes(column)
            continue
        arrays.append(column)
        names.append(name)

    task_idx = episode_table.column("task_index")
    annotation_col = pa.array(task_idx.to_pylist(), type=pa.int64())
    arrays.append(annotation_col)
    names.append(ANNOTATION_COLUMN)

    parquet_out = pa.Table.from_arrays(arrays, names=names)
    episode_chunk = episode_index // chunks_size
    parquet_path = dst_root / LEGACY_DATA_PATH_TEMPLATE.format(
        episode_chunk=episode_chunk,
        episode_index=episode_index,
    )
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(parquet_out, parquet_path)

    for video_key, encoded_frames in video_payloads.items():
        video_path = dst_root / LEGACY_VIDEO_PATH_TEMPLATE.format(
            episode_chunk=episode_chunk,
            video_key=video_key,
            episode_index=episode_index,
        )
        _write_episode_video(video_path, encoded_frames, fps=fps)


def group_records_by_data_file(
    episode_records: list[dict[str, Any]],
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for record in episode_records:
        key = (int(record["data/chunk_index"]), int(record["data/file_index"]))
        grouped.setdefault(key, []).append(record)
    return grouped


def convert_dataset(src_root: Path, dst_root: Path, fps_override: int | None, max_episodes: int | None) -> None:
    source_info = validate_source_dataset(src_root)
    fps = int(fps_override or source_info.get("fps", 30))
    episode_records = load_episode_records(src_root, max_episodes=max_episodes)
    tasks = load_tasks(src_root)
    output_info, video_keys = build_output_info(source_info, len(episode_records), fps)

    write_json(dst_root / INFO_PATH, output_info)
    write_json(dst_root / "meta" / "modality.json", build_modality_json())
    write_tasks_jsonl(dst_root, tasks)
    write_episodes_jsonl(dst_root, episode_records)
    copy_stats_json(src_root, dst_root)

    chunks_size = int(output_info["chunks_size"])
    grouped = group_records_by_data_file(episode_records)
    for (chunk_idx, file_idx), records in tqdm(grouped.items(), desc="convert data files"):
        source_path = src_root / f"data/chunk-{chunk_idx:03d}/file-{file_idx:03d}.parquet"
        if not source_path.exists():
            raise FileNotFoundError(f"Missing source parquet file: {source_path}")

        table = pq.read_table(source_path)
        records = sorted(records, key=lambda rec: int(rec["dataset_from_index"]))
        file_offset = int(records[0]["dataset_from_index"])

        for record in tqdm(records, desc=f"episodes from file-{file_idx:03d}", leave=False):
            episode_index = int(record["episode_index"])
            start = int(record["dataset_from_index"]) - file_offset
            stop = int(record["dataset_to_index"]) - file_offset
            length = stop - start
            if length <= 0:
                raise ValueError(
                    f"Invalid episode length for episode {episode_index}: start={start}, stop={stop}"
                )
            episode_table = table.slice(start, length)
            convert_episode_table(
                episode_table=episode_table,
                episode_index=episode_index,
                dst_root=dst_root,
                video_keys=video_keys,
                fps=fps,
                chunks_size=chunks_size,
            )


def main() -> None:
    args = parse_args()
    prepare_output_root(args.dst_root, args.force)
    convert_dataset(
        src_root=args.src_root,
        dst_root=args.dst_root,
        fps_override=args.fps,
        max_episodes=args.max_episodes,
    )


if __name__ == "__main__":
    main()
