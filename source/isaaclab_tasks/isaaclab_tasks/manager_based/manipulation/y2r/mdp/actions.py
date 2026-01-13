# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Custom action terms for Y2R trajectory manipulation tasks."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
import isaaclab.utils.string as string_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# Canonical Allegro hand joint order - this is the order assumed by ALLEGRO_PCA_MATRIX columns
# and must be used when indexing joint positions for eigengrasp computations.
# NOTE: PhysX joint order from USD may differ! Always use find_joints() with this list.
ALLEGRO_HAND_JOINT_NAMES = [
    "index_joint_0", "index_joint_1", "index_joint_2", "index_joint_3",
    "middle_joint_0", "middle_joint_1", "middle_joint_2", "middle_joint_3",
    "ring_joint_0", "ring_joint_1", "ring_joint_2", "ring_joint_3",
    "thumb_joint_0", "thumb_joint_1", "thumb_joint_2", "thumb_joint_3",
]

# PCA matrix for eigen grasp (5 eigen components -> 16 hand joints)
# Each row is an eigen grasp basis vector
# Column order matches ALLEGRO_HAND_JOINT_NAMES above
ALLEGRO_PCA_MATRIX = torch.tensor([
    [-3.8872e-02, 3.7917e-01, 4.4703e-01, 7.1016e-03, 2.1159e-03, 3.2014e-01, 4.4660e-01, 5.2108e-02,
      5.6869e-05, 2.9845e-01, 3.8575e-01, 7.5774e-03, -1.4790e-02, 9.8163e-02, 4.3551e-02, 3.1699e-01],
    [-5.1148e-02, -1.3007e-01, 5.7727e-02, 5.7914e-01, 1.0156e-02, -1.8469e-01, 5.3809e-02, 5.4888e-01,
      1.3351e-04, -1.7747e-01, 2.7809e-02, 4.8187e-01, 2.9753e-02, 2.6149e-02, 6.6994e-02, 1.8117e-01],
    [-5.7137e-02, -3.4707e-01, 3.3365e-01, -1.8029e-01, -4.3560e-02, -4.7666e-01, 3.2517e-01, -1.5208e-01,
     -5.9691e-05, -4.5790e-01, 3.6536e-01, -1.3916e-01, 2.3925e-03, 3.7238e-02, -1.0124e-01, -1.7442e-02],
    [ 2.2795e-02, -3.4090e-02, 3.4366e-02, -2.6531e-02, 2.3471e-02, 4.6123e-02, 9.8059e-02, -1.2619e-03,
     -1.6452e-04, -1.3741e-02, 1.3813e-01, 2.8677e-02, 2.2661e-01, -5.9911e-01, 7.0257e-01, -2.4525e-01],
    [-4.4911e-02, -4.7156e-01, 9.3124e-02, 2.3135e-01, -2.4607e-03, 9.5564e-02, 1.2470e-01, 3.6613e-02,
      1.3821e-04, 4.6072e-01, 9.9315e-02, -8.1080e-02, -4.7617e-01, -2.7734e-01, -2.3989e-01, -3.1222e-01]
])  # Shape: (5, 16)


def get_allegro_hand_joint_ids(env, robot: Articulation) -> torch.Tensor:
    """Get Allegro hand joint IDs in canonical order (cached on env).
    
    Uses find_joints to map joint names to PhysX indices, ensuring correct order
    regardless of how joints are ordered in the USD file.
    
    Args:
        env: The environment (used for caching).
        robot: The robot articulation.
        
    Returns:
        Tensor of joint indices in canonical order (16,).
    """
    cache_attr = "_allegro_hand_joint_ids"
    joint_ids = getattr(env, cache_attr, None)
    if joint_ids is None:
        ids, _ = robot.find_joints(ALLEGRO_HAND_JOINT_NAMES, preserve_order=True)
        joint_ids = torch.tensor(list(ids), device=env.device, dtype=torch.long)
        setattr(env, cache_attr, joint_ids)
    return joint_ids


class EigenGraspRelativeJointPositionAction(ActionTerm):
    """Eigen grasp action term with relative joint position control.
    
    Takes (7 + eigen_dim + 16)D input [arm_7, eigen_k, hand_raw_16] and outputs 23D joint targets.
    The hand joints are computed as: hand_delta = (eigen_k @ PCA_kx16) + hand_raw_16
    """
    
    cfg: "EigenGraspRelativeJointPositionActionCfg"
    _asset: Articulation
    
    def __init__(self, cfg: "EigenGraspRelativeJointPositionActionCfg", env: "ManagerBasedEnv") -> None:
        super().__init__(cfg, env)
        
        # Resolve joint names
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=True
        )
        self._num_joints = len(self._joint_ids)
        
        # Validate joint count
        expected_joints = cfg.arm_joint_count + cfg.hand_joint_count
        if self._num_joints != expected_joints:
            raise ValueError(
                f"Expected {expected_joints} joints (arm={cfg.arm_joint_count}, hand={cfg.hand_joint_count}), "
                f"but found {self._num_joints} joints matching pattern."
            )
        
        
        # Store dimensions
        self._arm_dim = cfg.arm_joint_count
        self._eigen_dim = cfg.eigen_dim
        self._hand_dim = cfg.hand_joint_count
        self._input_dim = self._arm_dim + self._eigen_dim + self._hand_dim  # 28
        self._output_dim = self._arm_dim + self._hand_dim  # 23

        # Validate eigen dimension against the PCA matrix.
        # We support taking the first K principal components (K <= 5).
        if not (1 <= self._eigen_dim <= ALLEGRO_PCA_MATRIX.shape[0]):
            raise ValueError(
                f"Invalid eigen_dim={self._eigen_dim}. Must be in [1, {ALLEGRO_PCA_MATRIX.shape[0]}] "
                f"to match ALLEGRO_PCA_MATRIX shape={tuple(ALLEGRO_PCA_MATRIX.shape)}."
            )
        
        # Create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self._input_dim, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self._output_dim, device=self.device)
        
        # Register PCA matrix as buffer (use only the first eigen_dim components)
        self._pca_matrix = ALLEGRO_PCA_MATRIX[: self._eigen_dim].to(self.device)  # (eigen_dim, 16)
        
        # Parse scale
        if isinstance(cfg.scale, (float, int)):
            self._scale = float(cfg.scale)
        elif isinstance(cfg.scale, dict):
            self._scale = torch.ones(self.num_envs, self._output_dim, device=self.device)
            index_list, _, value_list = string_utils.resolve_matching_names_values(cfg.scale, self._joint_names)
            self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported scale type: {type(cfg.scale)}. Supported types are float and dict.")
    
    @property
    def action_dim(self) -> int:
        """The dimension of the action space (7 + eigen_dim + 16)."""
        return self._input_dim
    
    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions
    
    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions
    
    def process_actions(self, actions: torch.Tensor):
        """Process (7 + eigen_dim + 16)D input to 23D joint deltas.
        
        Args:
            actions: Input actions of shape (num_envs, 7 + eigen_dim + 16)
                     [arm_7, eigen_k, hand_raw_16]
        """
        # Store raw actions
        self._raw_actions[:] = actions
        
        # Split input
        arm_actions = actions[:, :self._arm_dim]  # (N, 7)
        eigen_actions = actions[:, self._arm_dim : self._arm_dim + self._eigen_dim]  # (N, eigen_dim)
        hand_raw_actions = actions[:, self._arm_dim + self._eigen_dim:]  # (N, 16)
        
        # Transform eigen to hand joints: (N, eigen_dim) @ (eigen_dim, 16) = (N, 16)
        hand_pca = torch.matmul(eigen_actions, self._pca_matrix)
        
        # Add residual
        hand_delta = hand_pca + hand_raw_actions  # (N, 16)
        
        # Concatenate arm and hand
        full_delta = torch.cat([arm_actions, hand_delta], dim=-1)  # (N, 23)
        
        # Apply scale
        self._processed_actions[:] = full_delta * self._scale
    
    def apply_actions(self):
        """Apply relative position control."""
        # Add current joint positions to the processed actions
        current_actions = self._processed_actions + self._asset.data.joint_pos[:, self._joint_ids]
        # Set position targets
        self._asset.set_joint_position_target(current_actions, joint_ids=self._joint_ids)
    
    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0


@configclass
class EigenGraspRelativeJointPositionActionCfg(ActionTermCfg):
    """Configuration for eigen grasp relative joint position action term.
    
    This action term takes a 28D input:
    - 7 arm joint deltas
    - 5 eigen grasp coefficients
    - 16 raw hand joint deltas (residual)
    
    And produces 23D joint position targets by:
    1. Transforming eigen coefficients to hand joints via PCA: hand_pca = eigen_5 @ PCA_matrix
    2. Adding residual: hand_delta = hand_pca + hand_raw_16
    3. Concatenating: full_delta = [arm_7, hand_delta]
    4. Applying scale and relative position control
    """
    
    class_type: type[ActionTerm] = EigenGraspRelativeJointPositionAction
    
    joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""
    
    scale: float | dict[str, float] = 1.0
    """Scale factor for the action (float or dict of regex expressions). Defaults to 1.0."""
    
    arm_joint_count: int = 7
    """Number of arm joints (first N joints). Defaults to 7 for Kuka."""
    
    hand_joint_count: int = 16
    """Number of hand joints. Defaults to 16 for Allegro."""
    
    eigen_dim: int = 5
    """Number of eigen grasp dimensions. Defaults to 5."""


__all__ = [
    "EigenGraspRelativeJointPositionAction",
    "EigenGraspRelativeJointPositionActionCfg",
    "ALLEGRO_HAND_JOINT_NAMES",
    "ALLEGRO_PCA_MATRIX",
    "get_allegro_hand_joint_ids",
]
