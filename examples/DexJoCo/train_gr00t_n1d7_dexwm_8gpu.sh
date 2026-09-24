#!/usr/bin/env bash
# 8-GPU GR00T N1.7 finetune with a frozen DexWM objective.
# The model flags keep the VLM frozen while training the action head and
# multimodal projector, matching launch_finetune.py defaults.
# Do not reuse the official_ft or visual_ft output directories: experiment.py
# always passes resume_from_checkpoint=True.
#
# The default objective keeps BC training and samples WM metrics periodically
# with zero WM weight. Override the variables below for auxiliary or WM-only
# training. Always use a new OUTPUT_DIR when changing the objective.
# Inference is unchanged; do not load DexWM in serve_gr00t_openpi.py.
# For a WM-only diagnostic run, set DEXWM_OBJECTIVE=wm_only,
# DEXWM_UPDATE_INTERVAL=1, and DEXWM_LOSS_WEIGHT to a positive value.
#
# Run on the compute node as a user who can open all 8 GPUs:
#   bash examples/DexJoCo/train_gr00t_n1d7_dexwm_8gpu.sh

set -euo pipefail

cd /data/home/zyh/Isaac-GR00T

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MASTER_PORT="${MASTER_PORT:-29691}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-/mnt/ceph2/ckpt/GR00T-N1.7-3B}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/ceph2/ckpt/gr00t_n1d7_dexjoco_bimanual_microwave/official_ft_dexwm_debugging_loss_corr_8gpu}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-official_ft_dexwm_debugging_loss_corr_8gpu}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
EPISODE_SAMPLING_RATE="${EPISODE_SAMPLING_RATE:-1.0}"
MAX_STEPS="${MAX_STEPS:-30000}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
NUM_GPUS="${NUM_GPUS:-8}"

DEXWM_CKPT="${DEXWM_CKPT:-/mnt/ceph3/dexwm/outputs/dexjoco_bimanual_teacher_forcing_rollout_s012/checkpoints/dexjoco_bimanual_teacher_forcing_rollout_9.pth.tar}"
DEXWM_FEATURE_ROOT="${DEXWM_FEATURE_ROOT:-/mnt/ceph2/DexJoCo-Datasets-LeRobot/dexjoco_dino_vla_cache/bimanual_microwave_cook}"
DEXWM_OBJECTIVE="${DEXWM_OBJECTIVE:-bc_plus_wm}"
DEXWM_LOSS_WEIGHT="${DEXWM_LOSS_WEIGHT:-0.0}"
DEXWM_UPDATE_INTERVAL="${DEXWM_UPDATE_INTERVAL:-50}"
DEXWM_ACTION_NUM_STEPS="${DEXWM_ACTION_NUM_STEPS:-1}"
DEXWM_ACTION_STRIDE="${DEXWM_ACTION_STRIDE:-4}"
DEXWM_DTYPE="${DEXWM_DTYPE:-bfloat16}"

source .venv/bin/activate
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
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
  --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
  --episode_sampling_rate "$EPISODE_SAMPLING_RATE" \
  --max_steps "$MAX_STEPS" \
  --save_steps "$SAVE_STEPS" \
  --save_total_limit "$SAVE_TOTAL_LIMIT" \
  --warmup_ratio 0.05 \
  --weight_decay 1e-5 \
  --learning_rate "$LEARNING_RATE" \
  --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
  --enable_dexwm_auxiliary \
  --dexwm_checkpoint_path "$DEXWM_CKPT" \
  --dexwm_feature_root "$DEXWM_FEATURE_ROOT" \
  --dexwm_objective "$DEXWM_OBJECTIVE" \
  --dexwm_loss_weight "$DEXWM_LOSS_WEIGHT" \
  --dexwm_update_interval "$DEXWM_UPDATE_INTERVAL" \
  --dexwm_action_num_steps "$DEXWM_ACTION_NUM_STEPS" \
  --dexwm_action_stride "$DEXWM_ACTION_STRIDE" \
  --dexwm_dtype "$DEXWM_DTYPE" \
  --no-use-wandb
