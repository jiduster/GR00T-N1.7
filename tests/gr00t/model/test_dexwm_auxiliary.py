"""Unit tests for the GR00T DexWM auxiliary path.

These tests cover the pieces that must be correct before a long GPU run:
group-wise unnormalization, the torch FK adapter, sidecar windows, and
differentiable action sampling. Loading the 5GB DexWM checkpoint is opt-in.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.dexjoco_raw_adapter import DexJoCoRawActionAdapter
from gr00t.model.dexwm_auxiliary import (
    DEXWM_CONTEXT_FRAMES,
    GroupWiseUnnormalizer,
    attach_frozen_dexwm,
    dexwm_window_offsets,
    unnormalize_minmax,
)
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead


def _small_config(**overrides) -> Gr00tN1d7Config:
    defaults = dict(
        backbone_embedding_dim=64,
        hidden_size=64,
        input_embedding_dim=64,
        max_state_dim=7,
        max_action_dim=7,
        action_horizon=4,
        state_history_length=1,
        num_inference_timesteps=2,
        max_num_embodiments=4,
        add_pos_embed=True,
        use_vlln=True,
        max_seq_len=32,
        use_alternate_vl_dit=False,
        attend_text_every_n_blocks=2,
        tune_projector=True,
        tune_diffusion_model=True,
        tune_vlln=True,
        state_dropout_prob=0.0,
        noise_beta_alpha=1.5,
        noise_beta_beta=1.0,
        noise_s=0.999,
        num_timestep_buckets=1000,
        attn_dropout=0.0,
        diffusion_model_cfg={
            "positional_embeddings": None,
            "num_layers": 2,
            "num_attention_heads": 2,
            "attention_head_dim": 32,
            "norm_type": "ada_norm",
            "dropout": 0.0,
            "final_dropout": False,
            "output_dim": 64,
            "interleave_self_attention": True,
        },
    )
    defaults.update(overrides)
    return Gr00tN1d7Config(**defaults)


def _make_backbone_output(config, batch_size=2, seq_len=8):
    return BatchFeature(
        data={
            "backbone_features": torch.randn(batch_size, seq_len, config.backbone_embedding_dim),
            "backbone_attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
            "image_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
        }
    )


def _make_action_input(config, batch_size=2):
    return BatchFeature(
        data={
            "state": torch.randn(batch_size, config.state_history_length, config.max_state_dim),
            "action": torch.randn(batch_size, config.action_horizon, config.max_action_dim),
            "embodiment_id": torch.zeros(batch_size, dtype=torch.long),
            "action_mask": torch.ones(batch_size, config.action_horizon, config.max_action_dim),
        }
    )


def test_unnormalize_minmax_roundtrip():
    low = torch.tensor([-2.0, 0.5, 1.0])
    high = torch.tensor([2.0, 1.5, 3.0])
    physical = torch.tensor([[-2.0, 0.5, 3.0], [0.0, 1.0, 2.0]])
    normalized = 2.0 * (physical - low) / (high - low) - 1.0
    recovered = unnormalize_minmax(normalized, low, high)
    torch.testing.assert_close(recovered, physical)


def test_groupwise_unnormalizer_concatenates_action_groups():
    unnormalizer = GroupWiseUnnormalizer(
        action_low=np.array([0.0, 10.0, 20.0], dtype=np.float32),
        action_high=np.array([2.0, 12.0, 24.0], dtype=np.float32),
    )
    normalized = torch.tensor([[[-1.0, 0.0, 1.0]]], dtype=torch.float32)
    physical = unnormalizer.unnormalize_action(normalized)
    torch.testing.assert_close(physical, torch.tensor([[[0.0, 11.0, 24.0]]]))


def test_dual_rotvec_adapter_shape_and_grad():
    adapter = DexJoCoRawActionAdapter(layout="dual_rotvec")
    actions = torch.zeros(2, 8, 44, dtype=torch.float32)
    actions[..., 0] = 0.02
    actions[..., 22] = -0.02
    actions = actions.clone().requires_grad_(True)
    state = torch.zeros(2, 46, dtype=torch.float32)
    state[:, 3] = 1.0  # right quaternion w
    state[:, 10] = 1.0  # left quaternion w
    out = adapter(actions, state)
    assert out.shape == (2, 8, 132)
    assert torch.isfinite(out).all()
    loss = out.square().mean()
    grad = torch.autograd.grad(loss, actions, allow_unused=False)[0]
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0


def test_dexwm_window_offsets_unit_stride_is_consecutive():
    frames, actions = dexwm_window_offsets(1)
    np.testing.assert_array_equal(frames, np.arange(9))
    np.testing.assert_array_equal(actions, np.arange(8))


def test_dexwm_window_offsets_stride5_covers_horizon():
    frames, actions = dexwm_window_offsets(5)
    np.testing.assert_array_equal(frames, np.array([0, 5, 10, 15, 20, 25, 30, 35, 40]))
    np.testing.assert_array_equal(actions, np.array([4, 9, 14, 19, 24, 29, 34, 39]))


def test_sidecar_window_valid_mask_at_episode_tail():
    from gr00t.data.dataset.dexwm_sidecar import DexWMSidecarDataset

    class _Dummy(DexWMSidecarDataset):
        def __init__(self, action_stride: int = 1):
            self.feature_root = Path("/tmp")
            self.context_frames = DEXWM_CONTEXT_FRAMES
            self.action_stride = action_stride
            self._feature_memmaps = {}

        def _memmap_features(self, episode_index: int):
            features = np.zeros((50, 448, 1024), dtype=np.float16)
            for t in range(features.shape[0]):
                features[t, 0, 0] = t
            return features

    dummy = _Dummy()
    features, valid = dummy._feature_window(episode_index=0, step_index=10, episode_length=12)
    assert features.shape == (9, 448, 1024)
    # transitions 10->11 is valid, 11->12 and later are not
    np.testing.assert_array_equal(valid, np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32))


def test_sidecar_window_stride5_picks_linspace_frames():
    from gr00t.data.dataset.dexwm_sidecar import DexWMSidecarDataset

    class _Dummy(DexWMSidecarDataset):
        def __init__(self):
            self.feature_root = Path("/tmp")
            self.context_frames = DEXWM_CONTEXT_FRAMES
            self.action_stride = 5
            self._feature_memmaps = {}

        def _memmap_features(self, episode_index: int):
            features = np.zeros((50, 448, 1024), dtype=np.float16)
            for t in range(features.shape[0]):
                features[t, 0, 0] = t
            return features

    dummy = _Dummy()
    features, valid = dummy._feature_window(episode_index=0, step_index=2, episode_length=50)
    assert features.shape == (9, 448, 1024)
    np.testing.assert_array_equal(
        features[:, 0, 0].astype(np.float32),
        np.array([2, 7, 12, 17, 22, 27, 32, 37, 42], dtype=np.float32),
    )
    np.testing.assert_array_equal(valid, np.ones(8, dtype=np.float32))


def test_action_head_sampled_actions_require_grad():
    config = _small_config()
    head = Gr00tN1d7ActionHead(config)
    head.train()
    out = head.forward(
        _make_backbone_output(config),
        _make_action_input(config),
        return_actions=True,
        action_num_steps=1,
    )
    sampled = out["sampled_actions"]
    assert sampled.shape == (2, config.action_horizon, config.max_action_dim)
    assert sampled.requires_grad
    sampled.square().mean().backward()
    grads = [p.grad.detach().norm() for p in head.parameters() if p.grad is not None]
    assert grads and torch.stack(grads).sum() > 0


def test_wm_only_attachment_requires_every_step_and_vla_actions():
    model = torch.nn.Linear(2, 2)
    model.config = type("Config", (), {"action_horizon": 40})()
    auxiliary = object()

    with pytest.raises(ValueError, match="update_interval=1"):
        attach_frozen_dexwm(
            model,
            auxiliary,
            objective="wm_only",
            loss_weight=1.0,
            update_interval=2,
            action_num_steps=1,
        )

    with pytest.raises(ValueError, match="predicted actions"):
        attach_frozen_dexwm(
            model,
            auxiliary,
            objective="wm_only",
            loss_weight=1.0,
            update_interval=1,
            action_num_steps=1,
            use_gt_actions=True,
        )


def test_get_action_still_detached():
    config = _small_config()
    head = Gr00tN1d7ActionHead(config)
    action_input = _make_action_input(config)
    del action_input["action"]
    out = head.get_action(_make_backbone_output(config), action_input)
    assert not out["action_pred"].requires_grad


def test_frozen_dexwm_loads_and_returns_finite_loss():
    if os.environ.get("RUN_DEXWM_CHECKPOINT_TEST") != "1":
        return
    if not torch.cuda.is_available():
        return
    from gr00t.model.dexwm_auxiliary import DEFAULT_DEXWM_CHECKPOINT, FrozenDexWMAuxiliary

    unnormalizer = GroupWiseUnnormalizer(
        action_low=np.zeros(44, dtype=np.float32),
        action_high=np.ones(44, dtype=np.float32),
    )
    aux = FrozenDexWMAuxiliary(
        DEFAULT_DEXWM_CHECKPOINT,
        unnormalizer,
        device="cuda",
        dtype="bfloat16",
    )
    features = torch.randn(1, 9, 448, 1024, device="cuda", dtype=torch.float32)
    actions = torch.zeros(1, 8, 44, device="cuda", dtype=torch.float32, requires_grad=True)
    state = torch.zeros(1, 46, device="cuda", dtype=torch.float32)
    state[:, 3] = 1.0
    state[:, 10] = 1.0
    loss = aux(features, actions, state, actions_are_normalized=False)
    assert torch.isfinite(loss)
    grad = torch.autograd.grad(loss, actions, allow_unused=False)[0]
    assert torch.isfinite(grad).all()
    for parameter in aux.world_model.parameters():
        assert parameter.grad is None
