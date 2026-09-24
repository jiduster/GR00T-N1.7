#!/usr/bin/env bash
# Direct 8-GPU LoRA finetune for DexJoCo bimanual_microwave_cook.
# Bypass Slurm: run this on the compute node as a user who can open all 8 GPUs.

set -euo pipefail

cd /data/home/zyh/Isaac-GR00T

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MASTER_PORT="${MASTER_PORT:-29671}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"

OUTPUT_DIR="${OUTPUT_DIR:-/mnt/ceph2/ckpt/gr00t_n1d7_dexjoco_bimanual_microwave_lora}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-rank16_8gpu}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
EPISODE_SAMPLING_RATE="${EPISODE_SAMPLING_RATE:-1.0}"
MAX_STEPS="${MAX_STEPS:-10000}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

uv run bash examples/finetune.sh \
  --base-model-path /mnt/ceph2/ckpt/GR00T-N1.7-3B \
  --backbone-model-path /mnt/ceph2/ckpt/Cosmos-Reason2-2B \
  --dataset-path /mnt/ceph2/DexJoCo-Datasets-LeRobot/dexjoco_lerobot_datasets/bimanual_microwave_cook_gr00t_v2 \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/DexJoCo/dexjoco_bimanual_config.py \
  --output-dir "$OUTPUT_DIR" \
  --experiment-name "$EXPERIMENT_NAME" \
  --num-gpus 8 \
  --global-batch-size "$GLOBAL_BATCH_SIZE" \
  --dataloader-num-workers "$DATALOADER_NUM_WORKERS" \
  --episode-sampling-rate "$EPISODE_SAMPLING_RATE" \
  --max-steps "$MAX_STEPS" \
  --save-steps "$SAVE_STEPS" \
  --save-total-limit "$SAVE_TOTAL_LIMIT" \
  --use-lora \
  --lora-rank "$LORA_RANK" \
  --lora-alpha "$LORA_ALPHA" \
  --lora-dropout "$LORA_DROPOUT" \
  --no-use-wandb
