#!/usr/bin/env python3
"""Correlate bc_loss and wm_loss from GR00T DexWM trainer_state.json logs.

The HuggingFace log_history stores `loss` and `{bc,wm}_loss` in separate
records at the same step. Interval > 1 writes wm_loss=0 on skipped steps.
Both curves also fall over training, so a raw Pearson can look strongly
positive even when the two heads are not co-moving. This script reports:

  pearson / spearman   on (bc, wm) after dropping wm_loss==0
  pearson_diff         on consecutive differences (do they move together?)
  pearson_detrended    after subtracting a linear fit on step
  late_pearson         same Pearson on the last 50% of kept points
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CKPT_ROOT = Path("/mnt/ceph2/ckpt/gr00t_n1d7_dexjoco_bimanual_microwave")


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return _pearson(rx.astype(np.float64), ry.astype(np.float64))


def _linear_residual(values: np.ndarray, step: np.ndarray) -> np.ndarray:
    if values.size < 3 or np.std(step) < 1e-12:
        return values - values.mean()
    coef = np.polyfit(step.astype(np.float64), values.astype(np.float64), deg=1)
    return values - np.polyval(coef, step)


def merge_log_history(log_history: list[dict]) -> list[dict]:
    by_step: dict[int, dict] = {}
    for row in log_history:
        if "step" not in row:
            continue
        by_step.setdefault(int(row["step"]), {}).update(row)
    return [by_step[k] for k in sorted(by_step)]


def kept_pairs(merged: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    steps, bc, wm = [], [], []
    for row in merged:
        if "bc_loss" not in row or "wm_loss" not in row:
            continue
        w = float(row["wm_loss"])
        if not np.isfinite(w) or w <= 0.0:
            continue
        b = float(row["bc_loss"])
        if not np.isfinite(b):
            continue
        steps.append(int(row["step"]))
        bc.append(b)
        wm.append(w)
    return np.asarray(steps), np.asarray(bc, dtype=np.float64), np.asarray(wm, dtype=np.float64)


def correlate(steps: np.ndarray, bc: np.ndarray, wm: np.ndarray) -> dict[str, float]:
    n = int(bc.size)
    out = {
        "n": n,
        "pearson": float("nan"),
        "spearman": float("nan"),
        "pearson_diff": float("nan"),
        "pearson_detrended": float("nan"),
        "late_pearson": float("nan"),
        "bc_mean": float("nan"),
        "wm_mean": float("nan"),
        "bc_last": float("nan"),
        "wm_last": float("nan"),
    }
    if n < 3:
        return out
    out["pearson"] = _pearson(bc, wm)
    out["spearman"] = _spearman(bc, wm)
    out["pearson_diff"] = _pearson(np.diff(bc), np.diff(wm))
    out["pearson_detrended"] = _pearson(_linear_residual(bc, steps), _linear_residual(wm, steps))
    half = n // 2
    if n - half >= 3:
        out["late_pearson"] = _pearson(bc[half:], wm[half:])
    out["bc_mean"] = float(bc.mean())
    out["wm_mean"] = float(wm.mean())
    out["bc_last"] = float(bc[-1])
    out["wm_last"] = float(wm[-1])
    return out


def latest_trainer_state(run_dir: Path) -> Path | None:
    best_step = -1
    best_path: Path | None = None
    try:
        children = run_dir.iterdir()
    except FileNotFoundError:
        return None
    for child in children:
        if not child.is_dir() or not child.name.startswith("checkpoint-"):
            continue
        suffix = child.name.split("-")[-1]
        if not suffix.isdigit():
            continue
        step = int(suffix)
        if step <= best_step:
            continue
        path = child / "trainer_state.json"
        if path.is_file():
            best_step = step
            best_path = path
    return best_path


def analyze_run(run_dir: Path) -> dict:
    state_path = latest_trainer_state(run_dir)
    if state_path is None:
        return {"run": run_dir.name, "error": "no trainer_state.json"}
    payload = json.loads(state_path.read_text())
    steps, bc, wm = kept_pairs(merge_log_history(payload.get("log_history", [])))
    stats = correlate(steps, bc, wm)
    stats["run"] = run_dir.name
    stats["checkpoint"] = state_path.parent.name
    stats["global_step"] = payload.get("global_step")
    return stats


def discover_runs(root: Path) -> list[Path]:
    runs = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        nested = child / child.name
        candidate = nested if nested.is_dir() else child
        if latest_trainer_state(candidate) is not None:
            runs.append(candidate)
    return runs


def format_row(stats: dict) -> str:
    if "error" in stats:
        return f"{stats['run']:<62} {stats['error']}"
    return (
        f"{stats['run']:<62} {stats['checkpoint']:<18} n={stats['n']:<5} "
        f"r={stats['pearson']:+.3f}  rs={stats['spearman']:+.3f}  "
        f"d={stats['pearson_diff']:+.3f}  det={stats['pearson_detrended']:+.3f}  "
        f"late={stats['late_pearson']:+.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=CKPT_ROOT,
        help="Directory that contains experiment folders",
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Experiment directory name or absolute path. Repeatable. Default: all runs under --root.",
    )
    args = parser.parse_args()

    if args.run:
        run_dirs = []
        for name in args.run:
            path = Path(name)
            if not path.is_absolute():
                nested = args.root / name / name
                path = nested if nested.is_dir() else args.root / name
            run_dirs.append(path)
    else:
        run_dirs = discover_runs(args.root)

    print(
        "run                                                              ckpt               "
        "n     pearson   spearman  diff      detrended late"
    )
    for run_dir in run_dirs:
        print(format_row(analyze_run(run_dir)))
    print()
    print(
        "Keep rows with wm_loss>0 only. |r|~0.2 is weak, ~0.5 moderate, >0.8 strong. "
        "Raw pearson is inflated by both losses falling over time; prefer diff/detrended."
    )


if __name__ == "__main__":
    main()
