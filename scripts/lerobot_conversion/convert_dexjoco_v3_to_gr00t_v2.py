#!/usr/bin/env python

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert a bimanual DexJoCo LeRobot v3 dataset to GR00T's v2 layout.

The source is never modified. Consolidated parquet and video files are split
into per-episode files, and the GR00T-specific metadata files are generated.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
from pathlib import Path
import shutil
import subprocess
from typing import Any

import jsonlines
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm


V30 = "v3.0"
V21 = "v2.1"
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = (
    "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-root", type=Path, required=True)
    parser.add_argument("--dst-root", type=Path, required=True)
    parser.add_argument("--video-workers", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=4)


def serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    return value


def prepare_destination(dst_root: Path, force: bool) -> None:
    if dst_root.exists():
        if not force:
            raise FileExistsError(f"Destination already exists: {dst_root}")
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True)


def load_episode_records(src_root: Path) -> list[dict[str, Any]]:
    paths = sorted(src_root.glob("meta/episodes/chunk-*/file-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No v3 episode metadata found under {src_root}")
    records: list[dict[str, Any]] = []
    for path in paths:
        records.extend(pq.read_table(path).to_pylist())
    records.sort(key=lambda record: int(record["episode_index"]))
    return records


def build_info(
    source_info: dict[str, Any], records: list[dict[str, Any]], video_keys: list[str]
) -> dict[str, Any]:
    info = dict(source_info)
    chunks_size = int(info["chunks_size"])
    info.update(
        {
            "codebase_version": V21,
            "total_episodes": len(records),
            "total_frames": sum(int(record["length"]) for record in records),
            "total_chunks": math.ceil(len(records) / chunks_size),
            "total_videos": len(records) * len(video_keys),
            "data_path": DATA_PATH,
            "video_path": VIDEO_PATH if video_keys else None,
        }
    )
    info.pop("data_files_size_in_mb", None)
    info.pop("video_files_size_in_mb", None)
    for feature in info["features"].values():
        if feature.get("dtype") != "video":
            feature.pop("fps", None)
    return info


def build_modality() -> dict[str, Any]:
    return {
        "state": {
            "right_tcp": {"start": 0, "end": 7},
            "left_tcp": {"start": 7, "end": 14},
            "right_hand": {"start": 14, "end": 30},
            "left_hand": {"start": 30, "end": 46},
        },
        "action": {
            "right_tcp": {"start": 0, "end": 6},
            "right_hand": {"start": 6, "end": 22},
            "left_tcp": {"start": 22, "end": 28},
            "left_hand": {"start": 28, "end": 44},
        },
        "video": {
            "ego": {"original_key": "observation.images.ego"},
            "wrist_left": {"original_key": "observation.images.wrist_left"},
            "wrist_right": {"original_key": "observation.images.wrist_right"},
        },
        "annotation": {
            "human.task_description": {"original_key": "task_index"},
        },
    }


def write_tasks(src_root: Path, dst_root: Path) -> None:
    tasks = pq.read_table(src_root / "meta/tasks.parquet").to_pandas()
    tasks = tasks.sort_values("task_index")
    with jsonlines.open(dst_root / "meta/tasks.jsonl", mode="w") as writer:
        for task, row in tasks.iterrows():
            writer.write({"task_index": int(row["task_index"]), "task": str(task)})


def nested_episode_stats(record: dict[str, Any]) -> dict[str, Any]:
    stats: dict[str, dict[str, Any]] = defaultdict(dict)
    for key, value in record.items():
        if not key.startswith("stats/"):
            continue
        _, feature, stat_name = key.split("/", maxsplit=2)
        stats[feature][stat_name] = serializable(value)
    return dict(stats)


def write_episode_metadata(dst_root: Path, records: list[dict[str, Any]]) -> None:
    with (
        jsonlines.open(dst_root / "meta/episodes.jsonl", mode="w") as episode_writer,
        jsonlines.open(dst_root / "meta/episodes_stats.jsonl", mode="w") as stats_writer,
    ):
        for record in records:
            episode_index = int(record["episode_index"])
            episode_writer.write(
                {
                    "episode_index": episode_index,
                    "tasks": serializable(record.get("tasks", [])),
                    "length": int(record["length"]),
                }
            )
            stats_writer.write(
                {"episode_index": episode_index, "stats": nested_episode_stats(record)}
            )


def convert_data(
    src_root: Path,
    dst_root: Path,
    records: list[dict[str, Any]],
    chunks_size: int,
) -> None:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(int(record["data/chunk_index"]), int(record["data/file_index"]))].append(
            record
        )

    for (chunk_index, file_index), file_records in tqdm(
        grouped.items(), desc="Splitting parquet files"
    ):
        src_path = src_root / f"data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
        table = pq.read_table(src_path)
        file_records.sort(key=lambda record: int(record["dataset_from_index"]))
        file_offset = int(file_records[0]["dataset_from_index"])
        for record in file_records:
            episode_index = int(record["episode_index"])
            start = int(record["dataset_from_index"]) - file_offset
            stop = int(record["dataset_to_index"]) - file_offset
            episode_table = table.slice(start, stop - start)
            if episode_table.num_rows != int(record["length"]):
                raise ValueError(f"Episode {episode_index} parquet length mismatch")
            dst_path = dst_root / DATA_PATH.format(
                episode_chunk=episode_index // chunks_size,
                episode_index=episode_index,
            )
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(episode_table, dst_path)


def split_one_video(
    src_path: Path,
    dst_path: Path,
    start: float,
    end: float,
    expected_frames: int,
) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start:.9f}",
            "-i",
            str(src_path),
            "-t",
            f"{end - start:.9f}",
            "-c",
            "copy",
            "-avoid_negative_ts",
            "1",
            "-y",
            str(dst_path),
        ],
        check=True,
        timeout=300,
    )
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(dst_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    actual_frames = int(probe.stdout.strip())
    if actual_frames != expected_frames:
        raise ValueError(
            f"Video frame mismatch for {dst_path}: {actual_frames} != {expected_frames}"
        )


def convert_videos(
    src_root: Path,
    dst_root: Path,
    records: list[dict[str, Any]],
    video_keys: list[str],
    source_pattern: str,
    chunks_size: int,
    workers: int,
) -> None:
    jobs = []
    for record in records:
        episode_index = int(record["episode_index"])
        for video_key in video_keys:
            prefix = f"videos/{video_key}"
            src_path = src_root / source_pattern.format(
                video_key=video_key,
                chunk_index=int(record[f"{prefix}/chunk_index"]),
                file_index=int(record[f"{prefix}/file_index"]),
            )
            dst_path = dst_root / VIDEO_PATH.format(
                episode_chunk=episode_index // chunks_size,
                video_key=video_key,
                episode_index=episode_index,
            )
            jobs.append(
                (
                    src_path,
                    dst_path,
                    float(record[f"{prefix}/from_timestamp"]),
                    float(record[f"{prefix}/to_timestamp"]),
                    int(record["length"]),
                )
            )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(split_one_video, *job) for job in jobs]
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Splitting and validating videos"
        ):
            future.result()


def convert(src_root: Path, dst_root: Path, video_workers: int, force: bool) -> None:
    info = load_json(src_root / "meta/info.json")
    if info.get("codebase_version") != V30:
        raise ValueError(f"Expected LeRobot {V30}, got {info.get('codebase_version')!r}")
    if int(info["features"]["action"]["shape"][0]) != 44:
        raise ValueError("Expected a 44-dimensional DexJoCo action")
    if int(info["features"]["observation.state"]["shape"][0]) != 46:
        raise ValueError("Expected a 46-dimensional DexJoCo state")

    records = load_episode_records(src_root)
    video_keys = [
        key for key, feature in info["features"].items() if feature.get("dtype") == "video"
    ]
    prepare_destination(dst_root, force)
    output_info = build_info(info, records, video_keys)
    write_json(dst_root / "meta/info.json", output_info)
    write_json(dst_root / "meta/modality.json", build_modality())
    # The custom config uses absolute action targets only. Keeping an explicit
    # empty file avoids a write attempt when training checks relative statistics.
    write_json(dst_root / "meta/relative_stats.json", {})
    write_tasks(src_root, dst_root)
    write_episode_metadata(dst_root, records)
    shutil.copy2(src_root / "meta/stats.json", dst_root / "meta/stats.json")
    convert_data(src_root, dst_root, records, int(info["chunks_size"]))
    convert_videos(
        src_root,
        dst_root,
        records,
        video_keys,
        info["video_path"],
        int(info["chunks_size"]),
        video_workers,
    )


if __name__ == "__main__":
    args = parse_args()
    convert(args.src_root, args.dst_root, args.video_workers, args.force)
