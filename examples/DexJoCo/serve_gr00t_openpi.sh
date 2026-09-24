#!/usr/bin/env bash
# OpenPI-compatible GR00T policy server for DexJoCo evaluation.
# Run this in the Isaac-GR00T environment. DexJoCo eval connects on --port.

set -euo pipefail

cd /data/home/zyh/Isaac-GR00T

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL_PATH="${MODEL_PATH:-/mnt/ceph2/ckpt/gr00t_n1d7_dexjoco_bimanual_microwave/visual_ft_8gpu/visual_ft_8gpu}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
DEVICE="${DEVICE:-cuda:0}"

/data/home/zyh/Isaac-GR00T/.venv/bin/python examples/DexJoCo/serve_gr00t_openpi.py \
  --model-path "$MODEL_PATH" \
  --embodiment-tag NEW_EMBODIMENT \
  --device "$DEVICE" \
  --host "$HOST" \
  --port "$PORT"
