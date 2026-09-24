# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GR00T N1.7 modality configuration for bimanual DexJoCo datasets."""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


def _absolute_eef_config() -> ActionConfig:
    return ActionConfig(
        rep=ActionRepresentation.ABSOLUTE,
        type=ActionType.EEF,
        format=ActionFormat.XYZ_ROTVEC,
    )


def _absolute_joint_config() -> ActionConfig:
    return ActionConfig(
        rep=ActionRepresentation.ABSOLUTE,
        type=ActionType.NON_EEF,
        format=ActionFormat.DEFAULT,
    )


dexjoco_bimanual_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["ego", "wrist_left", "wrist_right"],
    ),
    # Raw state layout:
    # [right_xyz(3), right_quat_wxyz(4), left_xyz(3), left_quat_wxyz(4),
    #  right_hand(16), left_hand(16)].
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["right_tcp", "left_tcp", "right_hand", "left_hand"],
    ),
    # Raw action layout:
    # [right_xyz(3), right_rotvec(3), right_hand(16),
    #  left_xyz(3), left_rotvec(3), left_hand(16)].
    #
    # These are absolute controller targets. Keeping every group ABSOLUTE is
    # important because state TCP rotations are quaternions while action TCP
    # rotations are rotation vectors, so they cannot be subtracted directly.
    "action": ModalityConfig(
        delta_indices=list(range(40)),
        modality_keys=["right_tcp", "right_hand", "left_tcp", "left_hand"],
        action_configs=[
            _absolute_eef_config(),
            _absolute_joint_config(),
            _absolute_eef_config(),
            _absolute_joint_config(),
        ],
    ),
    "language": ModalityConfig(delta_indices=[0], modality_keys=["task"]),
}


register_modality_config(
    dexjoco_bimanual_config,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)
