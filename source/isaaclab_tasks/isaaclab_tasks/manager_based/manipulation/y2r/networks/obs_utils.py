# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Observation utilities for Point-Transformer network.

This module provides functions to reshape the flat concatenated observation
vector from the environment into structured point-centric format suitable
for transformer processing.

Observation Structure (from base.yaml):
    policy:       last_action (28) × 5 history = 140
    proprio:      joint_pos (23) + joint_vel (23) + hand_tips (65) + contact (~10) × 5 = ~605
    current_pc:   object_point_cloud (32×3) × 3 history = 288
    current_poses: object_pose (7) + hand_pose (7) × 5 history = 70
    targets:      target_point_clouds (5×32×3) + target_poses (5×7) × 1 = 515

Note: object_pc and poses can have different history lengths.

Point-centric view:
    - 32 point tokens, each with dims based on history:
      - Current + history frames of xyz from current_pc
      - Target trajectory xyz from targets
    - 1 proprio token with all remaining features
"""

from __future__ import annotations

import os
import torch
from dataclasses import dataclass


@dataclass
class ObsConfig:
    """Configuration for observation splitting. No defaults - load from Y2R config."""
    # Point cloud settings
    num_points: int
    point_dim: int  # (object_pc_history + window_size) × 3

    # History lengths (split: object_pc vs poses)
    object_pc_history: int
    poses_history: int
    targets_history: int

    # Observation group sizes
    action_dim: int
    action_history: int
    joint_pos_dim: int
    joint_vel_dim: int
    hand_tips_dim: int
    contact_dim: int
    proprio_history: int
    pose_dim: int
    window_size: int


def obs_config_from_y2r() -> ObsConfig:
    """Create ObsConfig from Y2R base.yaml config (single source of truth)."""
    from ..config_loader import get_config

    mode = os.getenv("Y2R_MODE")
    cfg = get_config(mode)

    object_pc_history = cfg.observations.history.object_pc
    window_size = cfg.trajectory.window_size

    return ObsConfig(
        num_points=cfg.observations.num_points,
        point_dim=(object_pc_history + window_size) * 3,
        object_pc_history=object_pc_history,
        poses_history=cfg.observations.history.poses,
        targets_history=cfg.observations.history.targets,
        action_dim=cfg.robot.arm_joint_count + cfg.robot.hand_joint_count,  # 7 + 16 or eigen
        action_history=cfg.observations.history.policy,
        joint_pos_dim=cfg.robot.arm_joint_count + cfg.robot.hand_joint_count,
        joint_vel_dim=cfg.robot.arm_joint_count + cfg.robot.hand_joint_count,
        hand_tips_dim=65,  # 5 tips × 13 (pos+quat+linvel+angvel) - fixed
        contact_dim=10,  # Approximate - fixed
        proprio_history=cfg.observations.history.proprio,
        pose_dim=7,  # pos(3) + quat(4) - fixed
        window_size=window_size,
    )


def compute_obs_layout(cfg: ObsConfig | None = None) -> dict:
    """Compute the observation layout and offsets.

    Args:
        cfg: Observation configuration. Loads from Y2R config if None.

    Returns:
        Dictionary with observation layout information.
    """
    if cfg is None:
        cfg = obs_config_from_y2r()

    # Policy group (action history)
    policy_size = cfg.action_dim * cfg.action_history

    # Proprio group (joint state, hand tips, contact) × history
    proprio_per_step = cfg.joint_pos_dim + cfg.joint_vel_dim + cfg.hand_tips_dim + cfg.contact_dim
    proprio_size = proprio_per_step * cfg.proprio_history

    # Current PC group (object point cloud) × object_pc_history
    pc_per_step = cfg.num_points * 3
    current_pc_size = pc_per_step * cfg.object_pc_history

    # Current poses group (object + hand poses) × poses_history
    poses_per_step = cfg.pose_dim * 2  # object_pose + hand_pose
    current_poses_size = poses_per_step * cfg.poses_history

    # Targets group (target point clouds + target poses + timing) × targets_history
    target_pc_size = cfg.window_size * cfg.num_points * 3
    target_poses_size = cfg.window_size * cfg.pose_dim * 2  # object + hand targets
    target_timing_size = 1  # trajectory_timing scalar
    targets_size = (target_pc_size + target_poses_size + target_timing_size) * cfg.targets_history

    # Compute offsets
    policy_start = 0
    proprio_start = policy_size
    current_pc_start = proprio_start + proprio_size
    current_poses_start = current_pc_start + current_pc_size
    targets_start = current_poses_start + current_poses_size
    total_size = targets_start + targets_size

    return {
        'policy': {'start': policy_start, 'size': policy_size},
        'proprio': {'start': proprio_start, 'size': proprio_size},
        'current_pc': {'start': current_pc_start, 'size': current_pc_size},
        'current_poses': {'start': current_poses_start, 'size': current_poses_size},
        'targets': {'start': targets_start, 'size': targets_size},
        'total': total_size,
        'point_cloud_per_step': pc_per_step,
        'target_pc_size': target_pc_size,
        'config': cfg,
    }


def split_point_and_proprio(
    obs: torch.Tensor,
    num_points: int = 32,
    point_dim: int = 30,
    cfg: ObsConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split flat observation into point cloud and proprio tensors.
    
    This is a simplified version that assumes the point cloud data
    (perception + targets) is at the end of the observation vector.
    
    The point-centric format combines:
        - perception.object_point_cloud history: (history_len, num_points, 3)
        - targets.target_point_clouds: (window_size, num_points, 3)
    
    Into: (num_points, point_dim) where point_dim = (history + window) × 3
    
    Args:
        obs: Flat observation tensor (B, obs_dim)
        num_points: Number of points per object
        point_dim: Features per point (default 30 = 10 frames × 3)
        cfg: Optional observation config for detailed layout
        
    Returns:
        point_obs: Point cloud in point-centric format (B, num_points, point_dim)
        proprio_obs: All remaining features (B, proprio_dim)
    """
    B = obs.shape[0]
    obs_dim = obs.shape[1]
    
    # Total point cloud dimensions
    point_cloud_dim = num_points * point_dim
    proprio_dim = obs_dim - point_cloud_dim
    
    if proprio_dim < 0:
        raise ValueError(
            f"Observation dim ({obs_dim}) too small for {num_points} points × {point_dim} dims = {point_cloud_dim}"
        )
    
    # Split observation
    # Assume point cloud is at the END of observation
    # proprio = [policy | proprio | pose parts]
    # point_cloud = [perception point clouds | target point clouds]
    proprio_obs = obs[:, :proprio_dim]
    point_flat = obs[:, proprio_dim:]
    
    # Reshape to point-centric format
    point_obs = point_flat.view(B, num_points, point_dim)
    
    return point_obs, proprio_obs


def split_point_and_proprio_detailed(
    obs: torch.Tensor,
    cfg: ObsConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split observation with detailed knowledge of observation structure.

    This version uses the exact observation layout to properly extract
    and reshape point cloud data from current_pc and targets groups.

    Args:
        obs: Flat observation tensor (B, obs_dim)
        cfg: Observation configuration (loads from Y2R config if None)

    Returns:
        point_obs: Point cloud in point-centric format (B, num_points, point_dim)
        proprio_obs: All remaining features (B, proprio_dim)
    """
    if cfg is None:
        cfg = obs_config_from_y2r()

    B = obs.shape[0]
    layout = compute_obs_layout(cfg)

    # Extract groups
    policy = obs[:, layout['policy']['start']:layout['policy']['start'] + layout['policy']['size']]
    proprio = obs[:, layout['proprio']['start']:layout['proprio']['start'] + layout['proprio']['size']]
    current_pc = obs[:, layout['current_pc']['start']:layout['current_pc']['start'] + layout['current_pc']['size']]
    current_poses = obs[:, layout['current_poses']['start']:layout['current_poses']['start'] + layout['current_poses']['size']]
    targets = obs[:, layout['targets']['start']:layout['targets']['start'] + layout['targets']['size']]

    # Extract point clouds from current_pc group
    # current_pc = [pc_0, pc_1, ..., pc_n] where n = object_pc_history
    # Each pc is (num_points × 3)
    pc_per_step = layout['point_cloud_per_step']

    current_pcs = current_pc.view(B, cfg.object_pc_history, cfg.num_points, 3)

    # Extract target point clouds (ignoring poses)
    target_pc_size = cfg.window_size * cfg.num_points * 3
    target_pcs = targets[:, :target_pc_size]
    target_pcs = target_pcs.view(B, cfg.window_size, cfg.num_points, 3)

    # Combine into point-centric format
    # Per point: [hist_0, ..., hist_n, tgt_0, ..., tgt_m] each is 3D xyz
    # Shape: (B, n+m, num_points, 3) → permute → (B, num_points, n+m, 3) → reshape
    all_pcs = torch.cat([current_pcs, target_pcs], dim=1)
    point_obs = all_pcs.permute(0, 2, 1, 3)
    point_obs = point_obs.reshape(B, cfg.num_points, -1)

    # Proprio includes: policy (actions), proprio (joints/tips), poses, and timing
    target_poses_size = cfg.window_size * cfg.pose_dim * 2
    target_poses = targets[:, target_pc_size:target_pc_size + target_poses_size]
    target_timing = targets[:, target_pc_size + target_poses_size:]

    proprio_obs = torch.cat([policy, proprio, current_poses, target_poses, target_timing], dim=1)

    return point_obs, proprio_obs

