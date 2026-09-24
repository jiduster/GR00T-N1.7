#!/usr/bin/env bash
# Stage-2 visual finetune for DexJoCo bimanual_microwave_cook.
# Submit on the compute node (lyg0383). Do not salloc. Do not srun inside this script.
#
# Submit:
#   sbatch --gres=gpu:0,gpu:1,gpu:2,gpu:3,gpu:4,gpu:5,gpu:6,gpu:7 \
#     -c 64 \
#     -o /data/home/zyh/job-%j.out \
#     examples/DexJoCo/train_gr00t_n1d7_visual_8gpu.sbatch

#SBATCH --job-name=groot-dexjoco-visual
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=64

set -euo pipefail

cd /data/home/zyh/Isaac-GR00T

# Slurm Job Bind remaps the named Indexes to CUDA ordinals 0..7.
# Do not set CUDA_VISIBLE_DEVICES to physical GPU Indexes.
unset CUDA_VISIBLE_DEVICES || true

export MASTER_PORT="${MASTER_PORT:-29681}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NO_ALBUMENTATIONS_UPDATE=1

BASE_MODEL_PATH="${BASE_MODEL_PATH:-/mnt/ceph2/ckpt/gr00t_n1d7_dexjoco_bimanual_microwave/official_ft_8gpu/checkpoint-10000}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/ceph2/ckpt/gr00t_n1d7_dexjoco_bimanual_microwave/visual_ft_8gpu}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-visual_ft_8gpu}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
EPISODE_SAMPLING_RATE="${EPISODE_SAMPLING_RATE:-1.0}"
MAX_STEPS="${MAX_STEPS:-3000}"
SAVE_STEPS="${SAVE_STEPS:-500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
NUM_GPUS="${NUM_GPUS:-8}"

uv run torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
  gr00t/experiment/launch_finetune.py \
  --base_model_path "$BASE_MODEL_PATH" \
  --backbone_model_path /mnt/ceph2/ckpt/Cosmos-Reason2-2B \
  --dataset_path /mnt/ceph2/DexJoCo-Datasets-LeRobot/dexjoco_lerobot_datasets/bimanual_microwave_cook_gr00t_v2 \
  --embodiment_tag NEW_EMBODIMENT \
  --modality_config_path examples/DexJoCo/dexjoco_bimanual_config.py \
  --output_dir "$OUTPUT_DIR" \
  --experiment_name "$EXPERIMENT_NAME" \
  --num_gpus "$NUM_GPUS" \
  --global_batch_size "$GLOBAL_BATCH_SIZE" \
  --dataloader_num_workers 4 \
  --episode_sampling_rate "$EPISODE_SAMPLING_RATE" \
  --max_steps "$MAX_STEPS" \
  --save_steps "$SAVE_STEPS" \
  --save_total_limit "$SAVE_TOTAL_LIMIT" \
  --warmup_ratio 0.05 \
  --weight_decay 1e-5 \
  --learning_rate "$LEARNING_RATE" \
  --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
  --tune_visual \
  --no-use-wandb
