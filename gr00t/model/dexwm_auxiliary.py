"""Frozen DexWM teacher-forcing auxiliary loss for GR00T N1.7 training.

DexWM parameters stay frozen and are *not* registered on the GR00T module, so
they are excluded from the optimizer, DDP/DeepSpeed wrapping, and checkpoints.
The DexWM forward itself must still run with autograd enabled so ``wm_loss``
can flow back into sampled GR00T actions.
"""

from __future__ import annotations

import importlib
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import nn

from gr00t.model.dexjoco_raw_adapter import DexJoCoRawActionAdapter

logger = logging.getLogger(__name__)

DEFAULT_DEXWM_ROOT = "/data/home/zyh/dexwm"
DEFAULT_DEXWM_CHECKPOINT = (
    "/mnt/ceph3/dexwm/outputs/dexjoco_bimanual_teacher_forcing_4task/"
    "checkpoints/dexjoco_bimanual_teacher_forcing_8.pth.tar"
)
DEXWM_FEATURE_SHAPE = (9, 448, 1024)
DEXWM_CONTEXT_FRAMES = 8
DEXWM_ACTION_DIM = 132
DEXWM_OBJECTIVES = ("bc_plus_wm", "wm_only")
RAW_ACTION_DIM = 44
RAW_STATE_DIM = 46
DEXWM_SIDECAR_KEYS = (
    "dexwm_features",
    "dexwm_valid_mask",
    "dexwm_state",
    "dexwm_gt_action",
)


def dexwm_window_offsets(
    stride: int,
    context_frames: int = DEXWM_CONTEXT_FRAMES,
) -> tuple[np.ndarray, np.ndarray]:
    """Frame and action-chunk offsets for a DexWM teacher-forcing window.

    DexWM always sees 9 frames and 8 transitions. ``stride`` is the gap in
    dataset frames between those slots, so stride=1 is consecutive and
    stride=5 linspaces the 8 hops across a 40-step GR00T action chunk:

        frames  t, t+s, ..., t+8s
        actions a[t+s-1], a[t+2s-1], ..., a[t+8s-1]

    The last action index is ``8 * stride - 1``, which must fit inside the
    VLA action horizon (39 when horizon=40 and stride=5).
    """
    stride = int(stride)
    if stride < 1:
        raise ValueError(f"dexwm_action_stride must be >= 1, got {stride}")
    if context_frames < 1:
        raise ValueError(f"context_frames must be >= 1, got {context_frames}")
    frame_offsets = np.arange(context_frames + 1, dtype=np.int64) * stride
    action_offsets = (np.arange(context_frames, dtype=np.int64) + 1) * stride - 1
    return frame_offsets, action_offsets


def max_dexwm_action_index(stride: int, context_frames: int = DEXWM_CONTEXT_FRAMES) -> int:
    return int(dexwm_window_offsets(stride, context_frames)[1][-1])


BIMANUAL_ACTION_GROUPS = ("right_tcp", "right_hand", "left_tcp", "left_hand")
BIMANUAL_STATE_GROUPS = ("right_tcp", "left_tcp", "right_hand", "left_hand")


def unnormalize_minmax(normalized: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    """Invert GR00T min/max (or q01/q99) normalization without clipping.

    Training-time predicted actions may sit slightly outside ``[-1, 1]``.
    Clipping those values would zero the gradient at the boundary.
    """
    low = low.to(device=normalized.device, dtype=normalized.dtype)
    high = high.to(device=normalized.device, dtype=normalized.dtype)
    return (normalized + 1.0) * 0.5 * (high - low) + low


def unnormalize_meanstd(normalized: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    mean = mean.to(device=normalized.device, dtype=normalized.dtype)
    std = std.to(device=normalized.device, dtype=normalized.dtype)
    return normalized * (std + 1e-6) + mean


class GroupWiseUnnormalizer(nn.Module):
    """Concatenate per-group GR00T stats and invert normalization in torch."""

    def __init__(
        self,
        action_low: np.ndarray,
        action_high: np.ndarray,
        *,
        mode: Literal["minmax", "meanstd"] = "minmax",
        action_mean: np.ndarray | None = None,
        action_std: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.register_buffer("action_low", torch.as_tensor(action_low, dtype=torch.float32), persistent=False)
        self.register_buffer("action_high", torch.as_tensor(action_high, dtype=torch.float32), persistent=False)
        if action_mean is None:
            action_mean = np.zeros_like(action_low)
        if action_std is None:
            action_std = np.ones_like(action_low)
        self.register_buffer("action_mean", torch.as_tensor(action_mean, dtype=torch.float32), persistent=False)
        self.register_buffer("action_std", torch.as_tensor(action_std, dtype=torch.float32), persistent=False)

    def unnormalize_action(self, normalized_action: torch.Tensor) -> torch.Tensor:
        dim = self.action_low.shape[-1]
        if normalized_action.shape[-1] < dim:
            raise ValueError(
                f"Expected at least {dim} action dims to unnormalize, got {tuple(normalized_action.shape)}"
            )
        action = normalized_action[..., :dim]
        if self.mode == "meanstd":
            return unnormalize_meanstd(action, self.action_mean, self.action_std)
        return unnormalize_minmax(action, self.action_low, self.action_high)


def _concat_group_stats(norm_params: dict[str, dict[str, Any]], group_keys: tuple[str, ...], stat: str) -> np.ndarray:
    pieces = []
    for key in group_keys:
        if key not in norm_params:
            raise KeyError(f"Missing GR00T normalization stats for group '{key}'")
        pieces.append(np.asarray(norm_params[key][stat], dtype=np.float32).reshape(-1))
    return np.concatenate(pieces, axis=0)


def build_group_unnormalizer(processor, embodiment_tag: str) -> GroupWiseUnnormalizer:
    """Build an action unnormalizer from the live GR00T processor stats."""
    sap = processor.state_action_processor
    action_keys = tuple(processor.modality_configs[embodiment_tag]["action"].modality_keys)
    action_params = sap.norm_params[embodiment_tag]["action"]
    mode: Literal["minmax", "meanstd"] = "meanstd" if getattr(processor, "use_mean_std", False) else "minmax"
    return GroupWiseUnnormalizer(
        action_low=_concat_group_stats(action_params, action_keys, "min"),
        action_high=_concat_group_stats(action_params, action_keys, "max"),
        mode=mode,
        action_mean=_concat_group_stats(action_params, action_keys, "mean"),
        action_std=_concat_group_stats(action_params, action_keys, "std"),
    )


def _import_dexwm(dexwm_root: str):
    root = Path(dexwm_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"DexWM source tree does not exist: {root}")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    try:
        return importlib.import_module("models.model").DexWM
    except Exception as exc:  # pragma: no cover - depends on optional DexWM deps
        raise RuntimeError(
            "Could not import DexWM. Install DexWM dependencies in the Isaac-GR00T "
            "environment (at least `timm`) and keep /data/home/zyh/dexwm on disk."
        ) from exc


@dataclass
class DexWMAuxiliaryRuntimeConfig:
    checkpoint_path: str = DEFAULT_DEXWM_CHECKPOINT
    dexwm_root: str = DEFAULT_DEXWM_ROOT
    loss_weight: float = 0.05
    update_interval: int = 1
    action_num_steps: int = 1
    dtype: str = "bfloat16"
    use_gt_actions: bool = False
    layout: str = "dual_rotvec"


class FrozenDexWMAuxiliary(nn.Module):
    """Teacher-forcing feature MSE conditioned on converted VLA actions."""

    def __init__(
        self,
        checkpoint_path: str,
        unnormalizer: GroupWiseUnnormalizer,
        *,
        device: torch.device | str = "cpu",
        dexwm_root: str = DEFAULT_DEXWM_ROOT,
        dtype: str = "bfloat16",
        layout: str = "dual_rotvec",
    ) -> None:
        super().__init__()
        if dtype not in {"float32", "bfloat16"}:
            raise ValueError(f"Unsupported DexWM dtype: {dtype}")
        self.dtype_name = dtype
        device = torch.device(device)

        dexwm_class = _import_dexwm(dexwm_root)
        self.unnormalizer = unnormalizer
        self.adapter = DexJoCoRawActionAdapter(layout=layout)
        self.world_model = dexwm_class(
            backbone_name="dinov2",
            num_patches=448,
            patch_size=14,
            hidden_dim=1024,
            action_dim=DEXWM_ACTION_DIM,
            depth=32,
            num_heads=16,
            mlp_ratio=2.0,
            num_context=DEXWM_CONTEXT_FRAMES,
            is_eval=False,
            emb_loss_fn=nn.MSELoss(reduction="mean"),
            use_gradient_checkpointing=False,
            load_image_encoder=False,
        )

        checkpoint_path = Path(checkpoint_path).expanduser()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"DexWM checkpoint does not exist: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("model", checkpoint)
        state = {key.replace("_orig_mod.", ""): value for key, value in state.items()}
        incompatible = self.world_model.load_state_dict(state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "DexWM checkpoint does not match teacher-forcing architecture: "
                f"missing={incompatible.missing_keys[:8]}, "
                f"unexpected={incompatible.unexpected_keys[:8]}"
            )
        del state, checkpoint

        self.world_model.eval()
        for parameter in self.world_model.parameters():
            parameter.requires_grad_(False)
        self.to(device)
        if dtype == "bfloat16":
            self.world_model.to(dtype=torch.bfloat16)
            self.adapter.float()
            self.unnormalizer.float()

        n_params = sum(p.numel() for p in self.world_model.parameters())
        logger.info(
            "Loaded frozen DexWM from %s (%s params, dtype=%s, device=%s)",
            checkpoint_path,
            f"{n_params:,}",
            dtype,
            device,
        )

    def to_model_device(self, device: torch.device, dtype: torch.dtype | None = None) -> None:
        """Move the unregistered module after the trainer places GR00T on GPU."""
        self.to(device)
        if self.dtype_name == "bfloat16" or dtype == torch.bfloat16:
            self.world_model.to(device=device, dtype=torch.bfloat16)
            self.adapter.to(device=device, dtype=torch.float32)
            self.unnormalizer.to(device=device, dtype=torch.float32)
        else:
            self.world_model.to(device=device, dtype=torch.float32)
            self.adapter.to(device=device, dtype=torch.float32)
            self.unnormalizer.to(device=device, dtype=torch.float32)

    def convert_actions(
        self,
        predicted_actions: torch.Tensor,
        physical_state: torch.Tensor,
        *,
        actions_are_normalized: bool,
    ) -> torch.Tensor:
        if predicted_actions.ndim != 3 or predicted_actions.shape[1] != DEXWM_CONTEXT_FRAMES:
            raise ValueError(
                "DexWM auxiliary expects predicted_actions [B, 8, 44], "
                f"got {tuple(predicted_actions.shape)}"
            )
        actions = predicted_actions[..., :RAW_ACTION_DIM]
        if actions_are_normalized:
            actions = self.unnormalizer.unnormalize_action(actions)
        if physical_state.ndim == 3:
            physical_state = physical_state[:, -1]
        if physical_state.shape[-1] < RAW_STATE_DIM:
            raise ValueError(
                f"DexWM auxiliary expects physical state dim >= {RAW_STATE_DIM}, "
                f"got {tuple(physical_state.shape)}"
            )
        return self.adapter(actions.float(), physical_state[..., :RAW_STATE_DIM].float())

    def forward(
        self,
        real_features: torch.Tensor,
        predicted_actions: torch.Tensor,
        physical_state: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        *,
        actions_are_normalized: bool = True,
    ) -> torch.Tensor:
        if real_features.ndim != 4 or tuple(real_features.shape[1:]) != DEXWM_FEATURE_SHAPE:
            raise ValueError(
                "DexWM teacher forcing expects real_features shaped "
                f"[B, 9, 448, 1024], got {tuple(real_features.shape)}"
            )
        if predicted_actions.shape[0] != real_features.shape[0]:
            raise ValueError("real_features and predicted_actions batch sizes differ")
        if valid_mask is not None and tuple(valid_mask.shape) != (real_features.shape[0], DEXWM_CONTEXT_FRAMES):
            raise ValueError(f"valid_mask must have shape [B, 8], got {tuple(valid_mask.shape)}")

        action132 = self.convert_actions(
            predicted_actions,
            physical_state,
            actions_are_normalized=actions_are_normalized,
        )
        rel_t = torch.zeros(
            (real_features.shape[0], DEXWM_CONTEXT_FRAMES),
            device=real_features.device,
            dtype=torch.long,
        )
        wm_features = real_features
        wm_actions = action132
        if self.dtype_name == "bfloat16":
            wm_features = wm_features.to(dtype=torch.bfloat16)
            wm_actions = wm_actions.to(dtype=torch.bfloat16)
        predicted, _, _, _emb_loss, _ = self.world_model(
            None,
            wm_actions,
            rel_t,
            action_diff=True,
            x_emb=wm_features,
            predict_kp=False,
        )
        per_transition = (predicted[:, 1:].float() - wm_features[:, 1:].float()).square().mean(dim=(2, 3))
        if valid_mask is not None:
            mask = valid_mask.to(device=per_transition.device, dtype=per_transition.dtype)
            return (per_transition * mask).sum() / mask.sum().clamp_min(1.0)
        return per_transition.mean()


def attach_frozen_dexwm(
    model: nn.Module,
    auxiliary: FrozenDexWMAuxiliary,
    *,
    objective: str = "bc_plus_wm",
    loss_weight: float,
    update_interval: int,
    action_num_steps: int,
    use_gt_actions: bool = False,
    action_stride: int = 1,
) -> None:
    """Attach DexWM without registering it as a GR00T child module."""
    if objective not in DEXWM_OBJECTIVES:
        raise ValueError(
            f"dexwm_objective must be one of {DEXWM_OBJECTIVES}, got {objective!r}"
        )
    if update_interval < 1:
        raise ValueError(f"dexwm_update_interval must be >= 1, got {update_interval}")
    if objective == "wm_only":
        if update_interval != 1:
            raise ValueError("dexwm_objective='wm_only' requires dexwm_update_interval=1")
        if loss_weight <= 0:
            raise ValueError("dexwm_objective='wm_only' requires dexwm_loss_weight > 0")
        if use_gt_actions:
            raise ValueError(
                "dexwm_objective='wm_only' requires predicted actions; "
                "GT actions make wm_loss independent of the VLA"
            )
    action_stride = int(action_stride)
    action_horizon = int(getattr(getattr(model, "config", None), "action_horizon", 40))
    last_action = max_dexwm_action_index(action_stride)
    if last_action >= action_horizon:
        raise ValueError(
            f"dexwm_action_stride={action_stride} needs action index {last_action}, "
            f"but action_horizon={action_horizon}. Use stride <= {action_horizon // DEXWM_CONTEXT_FRAMES}."
        )
    object.__setattr__(model, "dexwm_auxiliary", auxiliary)
    object.__setattr__(model, "dexwm_objective", objective)
    object.__setattr__(model, "dexwm_loss_weight", float(loss_weight))
    object.__setattr__(model, "dexwm_update_interval", int(update_interval))
    object.__setattr__(model, "dexwm_action_num_steps", int(action_num_steps))
    object.__setattr__(model, "dexwm_use_gt_actions", bool(use_gt_actions))
    object.__setattr__(model, "dexwm_action_stride", action_stride)
    object.__setattr__(model, "_dexwm_due", True)
    logger.info(
        "Attached frozen DexWM auxiliary loss (objective=%s, weight=%s, interval=%s, "
        "flow_steps=%s, gt_actions=%s, action_stride=%s)",
        objective,
        loss_weight,
        update_interval,
        action_num_steps,
        use_gt_actions,
        action_stride,
    )
