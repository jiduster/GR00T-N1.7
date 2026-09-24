"""Differentiable DexJoCo TCP+Allegro to DexWM action conversion.

Ported from ``openpi/src/openpi/training/dexjoco_raw_adapter.py``. Keep this
copy torch-only: a NumPy/MuJoCo FK would detach VLA action gradients.

DexJoCo policies emit absolute TCP poses and Allegro joint targets.  The
released DexWM checkpoint conditions on camera-frame deltas of 21 points per
hand.  This module mirrors the fixed Allegro kinematic chain from DexJoCo's
MuJoCo XML using torch-only operations, so gradients can flow back to a VLA
that predicts raw actions.
"""

from __future__ import annotations

from typing import ClassVar, Literal

import torch
from torch import Tensor, nn


RAW_SINGLE_ROTVEC_DIM = 22
RAW_SINGLE_QUAT_DIM = 23
RAW_DUAL_ROTVEC_DIM = 44
RAW_DUAL_QUAT_DIM = 46
DEXWM_ACTION_DIM = 132

# This is the fixed ego camera used by the bimanual DexJoCo label generator.
# It maps world coordinates to the positive-depth optical frame used by the
# DINO/keypoint labels (MuJoCo frame followed by diag(1, -1, -1)).  The
# bimanual ``observation.images.ego`` streams use this transform.
DEFAULT_DEXWM_CAMERA_WORLD_TO_CAMERA = (
    (0.0, -1.0, 0.0, 0.0),
    (-0.707106702, 0.0, -0.707106860, 0.636396379),
    (0.707106860, 0.0, -0.707106702, 2.474873663),
    (0.0, 0.0, 0.0, 1.0),
)

# MuJoCo ``front`` camera in the single-arm fold-glasses arena.  The raw
# dataset's ``observation.images.front`` stream is rendered from this camera,
# so a fold-glasses config should pass this matrix explicitly when generating
# DexWM labels/features.
DEFAULT_FOLD_GLASSES_FRONT_CAMERA_WORLD_TO_CAMERA = (
    (0.0, 1.0, 0.0, 0.0),
    (0.422618240, 0.0, -0.906307797, 0.900688764),
    (-0.906307797, 0.0, -0.422618240, 1.854389320),
    (0.0, 0.0, 0.0, 1.0),
)


def _rotation_from_quaternion_wxyz(quaternion: Tensor) -> Tensor:
    """Convert ``[..., 4]`` scalar-first quaternions to rotation matrices."""
    q = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def _rotation_from_rotvec(rotvec: Tensor) -> Tensor:
    """Differentiable Rodrigues formula for ``[..., 3]`` rotation vectors."""
    theta2 = (rotvec * rotvec).sum(dim=-1, keepdim=True)
    small = theta2 < 1e-8
    # ``torch.where`` evaluates both branches, so the closed-form path must
    # stay finite at the identity.  Take sqrt of a clamped theta^2.
    safe_theta2 = torch.where(small, torch.full_like(theta2, 1e-8), theta2)
    theta = safe_theta2.sqrt()
    a = torch.where(
        small,
        1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0,
        torch.sin(theta) / theta,
    )
    b = torch.where(
        small,
        0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0,
        (1.0 - torch.cos(theta)) / safe_theta2,
    )
    x, y, z = rotvec.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    skew = torch.stack((zeros, -z, y, z, zeros, -x, -y, x, zeros), dim=-1).reshape(
        *rotvec.shape[:-1], 3, 3
    )
    eye = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device)
    eye = eye.expand(*rotvec.shape[:-1], 3, 3)
    return eye + a[..., None] * skew + b[..., None] * (skew @ skew)


def _compose(parent_rotation: Tensor, parent_translation: Tensor, fixed_rotation: Tensor,
             fixed_translation: Tensor, axis: Tensor, angle: Tensor) -> tuple[Tensor, Tensor]:
    """Compose one MuJoCo body transform and its revolute joint."""
    joint_rotation = _rotation_from_rotvec(axis * angle[..., None])
    rotation = parent_rotation @ fixed_rotation @ joint_rotation
    translation = parent_translation + (parent_rotation @ fixed_translation[..., None]).squeeze(-1)
    return rotation, translation


class DexJoCoRawActionAdapter(nn.Module):
    """Map absolute raw DexJoCo actions to ``[B,T,132]`` DexWM deltas.

    ``single_rotvec`` is the fold-glasses contract (xyz, rotvec, 16 joints).
    ``single_quat``, ``dual_rotvec`` and ``dual_quat`` are accepted for the
    corresponding DexJoCo evaluation contracts.  The initial state is the
    normalized or unnormalized observation state supplied to ``forward``;
    callers should unnormalize it with the same statistics as the dataset.
    """

    _CHAIN_NAMES: ClassVar = ("ff", "mf", "rf", "th")
    _POINT_ORDER: ClassVar = ("th", "ff", "mf", "rf", "rf")

    # Values are body-local transforms and joint axes copied from
    # panda_allegro_{right,left}.xml.  The TCP frame is attachment_site_*.
    _PALM_POS: ClassVar = {
        # Position is declared in the attachment body's frame in the XML.
        # It is rotated below into the TCP/site frame.
        "right": (0.05, 0.0, 0.03),
        "left": (0.05, 0.0, 0.03),
    }
    _PALM_QUAT: ClassVar = {
        # Relative to attachment_site (whose XML quat is [0,0,0,1]), not the
        # attachment body's frame.  This is [0, sqrt(.5), 0, sqrt(.5)].
        "right": (0.0, 0.707106781, 0.0, 0.707106781),
        "left": (0.0, 0.707106781, 0.0, 0.707106781),
    }
    _BODY_POS: ClassVar = {
        "right": {
            "ff": ((0.0, 0.0435, -0.001542), (0.0, 0.0, 0.0164), (0.0, 0.0, 0.054), (0.0, 0.0, 0.0384)),
            "mf": ((0.0, 0.0, 0.0007), (0.0, 0.0, 0.0164), (0.0, 0.0, 0.054), (0.0, 0.0, 0.0384)),
            "rf": ((0.0, -0.0435, -0.001542), (0.0, 0.0, 0.0164), (0.0, 0.0, 0.054), (0.0, 0.0, 0.0384)),
            "th": ((-0.0182, 0.019333, -0.045987), (-0.027, 0.005, 0.0399), (0.0, 0.0, 0.0177), (0.0, 0.0, 0.0514)),
        },
        "left": {
            "ff": ((0.0, -0.0435, -0.001542), (0.0, 0.0, 0.0164), (0.0, 0.0, 0.054), (0.0, 0.0, 0.0384)),
            "mf": ((0.0, 0.0, 0.0007), (0.0, 0.0, 0.0164), (0.0, 0.0, 0.054), (0.0, 0.0, 0.0384)),
            "rf": ((0.0, 0.0435, -0.001542), (0.0, 0.0, 0.0164), (0.0, 0.0, 0.054), (0.0, 0.0, 0.0384)),
            "th": ((-0.0182, -0.019333, -0.045987), (-0.027, -0.005, 0.0399), (0.0, 0.0, 0.0177), (0.0, 0.0, 0.0514)),
        },
    }
    _BODY_QUAT: ClassVar = {
        "right": {
            "ff": ((0.999048221, -0.04361941, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
            "mf": ((1.0, 0.0, 0.0, 0.0),) * 4,
            "rf": ((0.999048221, 0.04361941, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
            "th": ((0.477714093, -0.521334101, -0.521334101, -0.477714093), (1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        },
        "left": {
            "ff": ((0.999048221, 0.04361941, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
            "mf": ((1.0, 0.0, 0.0, 0.0),) * 4,
            "rf": ((0.999048221, -0.04361941, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
            "th": ((0.477714093, 0.521334101, -0.521334101, 0.477714093), (1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        },
    }
    _JOINT_AXES: ClassVar = {
        "right": {"ff": ((0, 0, 1), (0, 1, 0), (0, 1, 0), (0, 1, 0)),
                   "mf": ((0, 0, 1), (0, 1, 0), (0, 1, 0), (0, 1, 0)),
                   "rf": ((0, 0, 1), (0, 1, 0), (0, 1, 0), (0, 1, 0)),
                   "th": ((-1, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 0))},
        "left": {"ff": ((0, 0, 1), (0, 1, 0), (0, 1, 0), (0, 1, 0)),
                  "mf": ((0, 0, 1), (0, 1, 0), (0, 1, 0), (0, 1, 0)),
                  "rf": ((0, 0, 1), (0, 1, 0), (0, 1, 0), (0, 1, 0)),
                  "th": ((1, 0, 0), (0, 0, -1), (0, 1, 0), (0, 1, 0))},
    }

    def __init__(
        self,
        layout: Literal["single_rotvec", "single_quat", "dual_rotvec", "dual_quat"] = "single_rotvec",
        camera_world_to_camera: Tensor | None = None,
    ) -> None:
        super().__init__()
        if layout not in {"single_rotvec", "single_quat", "dual_rotvec", "dual_quat"}:
            raise ValueError(f"Unsupported DexJoCo raw action layout: {layout}")
        self.layout = layout
        camera = (
            torch.as_tensor(DEFAULT_DEXWM_CAMERA_WORLD_TO_CAMERA, dtype=torch.float32)
            if camera_world_to_camera is None
            else torch.as_tensor(camera_world_to_camera, dtype=torch.float32)
        )
        if camera.shape != (4, 4):
            raise ValueError(f"camera_world_to_camera must have shape [4,4], got {tuple(camera.shape)}")
        self.register_buffer("camera_world_to_camera", camera, persistent=False)

        for side in ("right", "left"):
            palm_q = torch.as_tensor(self._PALM_QUAT[side], dtype=torch.float32)
            self.register_buffer(f"{side}_palm_rotation", _rotation_from_quaternion_wxyz(palm_q), persistent=False)
            for finger in self._CHAIN_NAMES:
                self.register_buffer(f"{side}_{finger}_pos", torch.as_tensor(self._BODY_POS[side][finger], dtype=torch.float32), persistent=False)
                self.register_buffer(f"{side}_{finger}_quat", torch.as_tensor(self._BODY_QUAT[side][finger], dtype=torch.float32), persistent=False)
                self.register_buffer(f"{side}_{finger}_axis", torch.as_tensor(self._JOINT_AXES[side][finger], dtype=torch.float32), persistent=False)

    @property
    def action_dim(self) -> int:
        return {"single_rotvec": 22, "single_quat": 23, "dual_rotvec": 44, "dual_quat": 46}[self.layout]

    @staticmethod
    def _reorder_hand(hand: Tensor, side: str) -> Tensor:
        # DexJoCo's bimanual state serializes the left hand RF,MF,FF,TH.
        if side == "left":
            return hand[..., [8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15]]
        return hand

    def _hand_points(self, hand: Tensor, side: str) -> Tensor:
        batch_shape = hand.shape[:-1]
        palm_pos = torch.as_tensor(self._PALM_POS[side], dtype=hand.dtype, device=hand.device).expand(*batch_shape, 3)
        palm_rotation = getattr(self, f"{side}_palm_rotation").to(dtype=hand.dtype).expand(*batch_shape, 3, 3)
        current = (palm_rotation, palm_pos)
        by_finger = {}
        for finger_index, finger in enumerate(self._CHAIN_NAMES):
            rotation, translation = current
            pos = getattr(self, f"{side}_{finger}_pos").to(dtype=hand.dtype)
            quat = getattr(self, f"{side}_{finger}_quat").to(dtype=hand.dtype)
            axis = getattr(self, f"{side}_{finger}_axis").to(dtype=hand.dtype)
            points = []
            for joint_index in range(4):
                fixed_rotation = _rotation_from_quaternion_wxyz(quat[joint_index]).expand(*batch_shape, 3, 3)
                fixed_translation = pos[joint_index].expand(*batch_shape, 3)
                rotation, translation = _compose(
                    rotation, translation, fixed_rotation, fixed_translation,
                    axis[joint_index].expand(*batch_shape, 3),
                    self._reorder_hand(hand, side)[..., finger_index * 4 + joint_index],
                )
                points.append(translation)
            by_finger[finger] = torch.stack(points, dim=-2)
        # DexWM's 21 points begin with the palm, followed by thumb, index,
        # middle, ring, and a repeated ring chain as the pinky proxy.
        return torch.cat([palm_pos[..., None, :], *[by_finger[finger] for finger in self._POINT_ORDER]], dim=-2)

    def _absolute_points(self, pose: Tensor, hand: Tensor, side: str) -> Tensor:
        if pose.shape[-1] == 6:
            rotation = _rotation_from_rotvec(pose[..., 3:6])
        else:
            rotation = _rotation_from_quaternion_wxyz(pose[..., 3:7])
        local_points = self._hand_points(hand, side)
        world_points = (rotation @ local_points.transpose(-1, -2)).transpose(-1, -2) + pose[..., None, :3]
        camera_rotation = self.camera_world_to_camera[:3, :3].to(dtype=pose.dtype, device=pose.device)
        camera_translation = self.camera_world_to_camera[:3, 3].to(dtype=pose.dtype, device=pose.device)
        return (camera_rotation @ world_points.transpose(-1, -2)).transpose(-1, -2) + camera_translation

    def _split(self, actions: Tensor, state: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        if self.layout == "single_rotvec":
            if actions.shape[-1] != 22 or state.shape[-1] < 23:
                raise ValueError(f"single_rotvec expects actions[...,22] and state[...,23], got {actions.shape}, {state.shape}")
            return actions[..., :6], actions[..., 6:22], state[..., :7], state[..., 7:23], None, None
        if self.layout == "single_quat":
            if actions.shape[-1] != 23 or state.shape[-1] < 23:
                raise ValueError(f"single_quat expects actions[...,23] and state[...,23], got {actions.shape}, {state.shape}")
            return actions[..., :7], actions[..., 7:23], state[..., :7], state[..., 7:23], None, None
        if self.layout == "dual_rotvec":
            if actions.shape[-1] != 44 or state.shape[-1] < 46:
                raise ValueError(f"dual_rotvec expects actions[...,44] and state[...,46], got {actions.shape}, {state.shape}")
            return actions[..., :6], actions[..., 6:22], state[..., :7], state[..., 14:30], actions[..., 22:28], actions[..., 28:44]
        if actions.shape[-1] != 46 or state.shape[-1] < 46:
            raise ValueError(f"dual_quat expects actions[...,46] and state[...,46], got {actions.shape}, {state.shape}")
        return actions[..., :7], actions[..., 7:23], state[..., :7], state[..., 14:30], actions[..., 23:30], actions[..., 30:46]

    def forward(self, actions: Tensor, state: Tensor) -> Tensor:
        if actions.ndim != 3 or state.ndim != 2:
            raise ValueError(f"Expected actions [B,T,A] and state [B,S], got {actions.shape}, {state.shape}")
        if actions.shape[0] != state.shape[0]:
            raise ValueError("Raw action and state batch dimensions differ")
        pose, hand, state_pose, state_hand, left_pose, left_hand = self._split(actions, state)
        if self.layout.startswith("single"):
            current = self._absolute_points(state_pose, state_hand, "right")
            right = self._absolute_points(pose, hand, "right")
            right_delta = torch.diff(right, dim=1, prepend=current[:, None])
            left_delta = torch.zeros_like(right_delta)
        else:
            current_right = self._absolute_points(state_pose, state_hand, "right")
            current_left = self._absolute_points(state[..., 7:14], state[..., 30:46], "left")
            right = self._absolute_points(pose, hand, "right")
            left = self._absolute_points(left_pose, left_hand, "left")
            right_delta = torch.diff(right, dim=1, prepend=current_right[:, None])
            left_delta = torch.diff(left, dim=1, prepend=current_left[:, None])
        camera = torch.zeros((*right_delta.shape[:-2], 6), dtype=actions.dtype, device=actions.device)
        return torch.cat([left_delta.flatten(-2), right_delta.flatten(-2), camera], dim=-1)
