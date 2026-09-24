#!/usr/bin/env bash
# Optional Slurm wrapper. Prefer examples/DexJoCo/train_gr00t_n1d7_lora_8gpu.sh
# when you can open GPUs directly.

#SBATCH --job-name=groot-dexjoco-lora
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=64

set -euo pipefail

cd /data/home/zyh/Isaac-GR00T

export MASTER_PORT="${MASTER_PORT:-29671}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

OUTPUT_DIR="${OUTPUT_DIR:-/mnt/ceph2/ckpt/gr00t_n1d7_dexjoco_bimanual_microwave_lora}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-rank16_8gpu}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
EPISODE_SAMPLING_RATE="${EPISODE_SAMPLING_RATE:-1.0}"
MAX_STEPS="${MAX_STEPS:-10000}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"

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
  --dataloader-num-workers 4 \
  --episode-sampling-rate "$EPISODE_SAMPLING_RATE" \
  --max-steps "$MAX_STEPS" \
  --save-steps "$SAVE_STEPS" \
  --save-total-limit "$SAVE_TOTAL_LIMIT" \
  --use-lora \
  --lora-rank 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --no-use-wandb
