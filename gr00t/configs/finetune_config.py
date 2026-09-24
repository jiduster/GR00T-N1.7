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

# Finetune config used for single node post-training.
from dataclasses import dataclass


@dataclass
class FinetuneConfig:
    """
    Configuration for fine-tuning a Vision-Language-Action (VLA) model.

    This dataclass defines all parameters needed to launch a fine-tuning job
    on a pretrained base model using a custom dataset and embodiment-specific
    modality configuration. It controls model tuning options, data augmentation,
    and training hyperparameters.
    """

    # --- Data and Model Paths ---
    base_model_path: str
    """Path to the pretrained base model checkpoint (e.g., Hugging Face model hub or local directory)."""

    dataset_path: str
    """Path to the dataset root directory containing trajectory data for fine-tuning."""

    embodiment_tag: str
    """Embodiment tag (name or value, case-insensitive). See EmbodimentTag for known tags."""

    modality_config_path: str | None = None
    """
    Path to a Python file defining the modality configuration for the given embodiment. 
    If None, use the pre-registered modality config in `gr00t/configs/data/embodiment_configs.py`. 
    """

    backbone_model_path: str | None = None
    """Optional path to the VLM backbone resources used by GR00T N1.7.
    Set this for fully offline runs, e.g. a local Cosmos-Reason2-2B snapshot.
    If None, defaults to the public Hub identifier baked into the model config.
    """

    # --- Model Tuning Flags ---
    tune_llm: bool = False
    """If True, fine-tune the language model (LLM) backbone during training."""

    tune_visual: bool = False
    """If True, fine-tune the visual encoder (e.g., ViT or CNN backbone)."""

    tune_projector: bool = True
    """If True, fine-tune the multimodal projector layers that map vision/language features to a shared space."""

    tune_diffusion_model: bool = True
    """If True, fine-tune the diffusion-based action decoder (if present in the model)."""

    use_lora: bool = False
    """Enable LoRA for the language backbone while training the action head normally."""

    lora_rank: int = 16
    """Rank of each LoRA adapter."""

    lora_alpha: int = 32
    """LoRA scaling factor."""

    lora_dropout: float = 0.05
    """Dropout probability applied inside LoRA adapters."""

    state_dropout_prob: float = 0.2
    """
    Dropout probability applied to state inputs for regularization during training.
    """

    # --- Data Augmentation ---
    random_rotation_angle: int | None = None
    """Maximum rotation angle (in degrees) for random rotation augmentation of input images."""

    color_jitter_params: dict[str, float] | None = None
    """
    Parameters for color jitter augmentation on images.

    Expected keys include:
      - "brightness": float
      - "contrast": float
      - "saturation": float
      - "hue": float
    Example: {"brightness": 0.4, "contrast": 0.4, "saturation": 0.4, "hue": 0.1}

    If None, applying the default color jitter augmentation from the pretrained model.
    """
    extra_augmentation_config: str | None = None
    """
    JSON string for extra image augmentations (mask-based and others).

    Expected keys include:
      - "background_noise_transforms": list of dicts for noise on mask regions
          - "target_mask_values": list of int (e.g., [0])
          - "p": float (probability of applying)
      - "masked_region_transforms": list of dicts for color tint on mask regions
          - "target_mask_values": list of int (e.g., [4] or [5])
          - "p": float (probability of applying)
          - "alpha_range": [min, max] for random_tint intensity

    Example: {"background_noise_transforms": [{"target_mask_values": [0], "p": 0.9}],
              "masked_region_transforms": [{"target_mask_values": [4], "p": 1.0, "alpha_range": [0, 1]}]}

    If None, no extra augmentations are applied.
    """

    # --- Training Configuration ---
    global_batch_size: int = 64
    """Total effective batch size across all GPUs and accumulation steps."""

    dataloader_num_workers: int = 2
    """Number of parallel worker processes used for data loading."""

    learning_rate: float = 1e-4
    """Initial learning rate for optimizer."""

    gradient_accumulation_steps: int = 1
    """Number of forward passes to accumulate before performing a backward/update step."""

    output_dir: str = "./outputs"
    """Directory where model checkpoints, logs, and outputs are saved."""

    experiment_name: str | None = None
    """Optional experiment name used as the W&B run name. Defaults to the output directory basename."""

    wandb_project: str = "finetune-gr00t-n1d7"
    """W&B project name to log runs to."""

    save_steps: int = 1000
    """Frequency (in training steps) at which to save checkpoints."""

    save_total_limit: int = 5
    """Maximum number of checkpoints to keep before older ones are deleted."""

    num_gpus: int = 1
    """Number of GPUs available for distributed or single-node training."""

    use_wandb: bool = False
    """
    If True, log metrics and artifacts to Weights & Biases (wandb).
    The project is `finetune-gr00t-n1d7`.
    You need to login to wandb to view the logs.
    """

    max_steps: int = 10000
    """Total number of training steps to run before stopping."""

    weight_decay: float = 1e-5
    """Weight decay coefficient for optimizer (L2 regularization)."""

    warmup_ratio: float = 0.05
    """Proportion of total training steps used for learning rate warm-up."""

    shard_size: int = 2**10
    """Size of the shard to use for the dataset during preloading."""

    episode_sampling_rate: float = 0.1
    """Sampling rate for the episodes."""

    num_shards_per_epoch: int = int(1e5)
    """Number of shards to use for the dataset. reduce this number if vram is limited."""

    save_only_model: bool = False
    """If True, save only model weights (skip optimizer/scheduler/RNG states). Cannot resume training from these checkpoints."""

    skip_weight_loading: bool = False
    """If True, skip loading model weights from base_model_path (architecture only).
    The processor (tokenizer/config) is still loaded from base_model_path.
    Useful for CI/testing to skip the slow checkpoint shard loading."""

    # --- Frozen DexWM auxiliary loss ---
    enable_dexwm_auxiliary: bool = False
    """If True, add a frozen DexWM teacher-forcing feature MSE to the GR00T loss."""

    dexwm_checkpoint_path: str | None = (
        "/mnt/ceph3/dexwm/outputs/dexjoco_bimanual_teacher_forcing_4task/"
        "checkpoints/dexjoco_bimanual_teacher_forcing_8.pth.tar"
    )
    """Teacher-forcing DexWM checkpoint. The ``_9`` filename in the handoff doc
    is no longer on disk; ``_8`` is the last snapshot of that run."""

    dexwm_feature_root: str | None = None
    """Directory of ``episode-XXXXXX.features.npy`` DINO sidecars."""

    dexwm_objective: str = "bc_plus_wm"
    """Training objective: ``bc_plus_wm`` or ``wm_only``. In ``wm_only`` mode,
    BC loss is still computed and logged as a diagnostic but is not backpropagated."""

    dexwm_loss_weight: float = 0.05
    """Weight applied on steps that compute the DexWM loss."""

    dexwm_update_interval: int = 1
    """Compute DexWM every N optimizer steps. OpenPI's microwave recipe used 8
    with ``dexwm_loss_weight=0.4`` so the expected coefficient stays ~0.05."""

    dexwm_action_num_steps: int = 1
    """Differentiable flow-matching Euler steps used to sample VLA actions."""

    dexwm_dtype: str = "bfloat16"
    """Frozen DexWM compute dtype. ``bfloat16`` or ``float32``."""

    dexwm_root: str = "/data/home/zyh/dexwm"
    """DexWM source tree used to import ``models.model.DexWM``."""

    dexwm_use_gt_actions: bool = False
    """If True, feed dataset GT actions into DexWM instead of sampled VLA
    actions. For alignment sanity checks only; it does not train the VLA."""

    dexwm_action_stride: int = 1
    """Dataset-frame gap between the 8 DexWM hops. ``1`` is consecutive
    (first 8 of the 40-step chunk). ``5`` linspaces those hops across the
    full GR00T action horizon. Must satisfy ``8 * stride <= action_horizon``."""
