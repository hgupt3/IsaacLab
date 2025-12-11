# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Observation utilities for Point-Transformer network.

This module provides functions to reshape the flat concatenated observation
vector from the environment into structured point-centric format suitable
for transformer processing.

Observation Structure (from y2r_config.yaml):
    policy:     last_action (28) × 5 history = 140
    proprio:    joint_pos (23) + joint_vel (23) + hand_tips (65) + contact (~10) × 5 = ~605
    perception: object_point_cloud (32×3) + object_pose (7) × 5 = 515
    targets:    target_point_clouds (5×32×3) + target_poses (5×7) × 1 = 515

Point-centric view:
    - 32 point tokens, each with 30 dims:
      - Current + 4 history frames of xyz (5 × 3 = 15 dims from perception)
      - 5 target trajectory xyz (5 × 3 = 15 dims from targets)
    - 1 proprio token with all remaining features
"""

from __future__ import annotations

import torch
from dataclasses import dataclass


@dataclass
class ObsConfig:
    """Configuration for observation splitting."""
    # Point cloud settings
    num_points: int = 32
    point_dim: int = 30  # 5 history + 5 targets × 3 xyz
    
    # History lengths
    perception_history: int = 5
    targets_history: int = 1  # Targets already encode future, no history
    
    # Observation group sizes (from y2r env)
    action_dim: int = 28
    action_history: int = 5
    joint_pos_dim: int = 23
    joint_vel_dim: int = 23
    hand_tips_dim: int = 65  # 5 tips × 13 (pos+quat+linvel+angvel)
    contact_dim: int = 10  # Approximate
    proprio_history: int = 5
    pose_dim: int = 7  # pos(3) + quat(4)
    window_size: int = 5  # Target window


def compute_obs_layout(cfg: ObsConfig | None = None) -> dict:
    """Compute the observation layout and offsets.
    
    Args:
        cfg: Observation configuration. Uses defaults if None.
        
    Returns:
        Dictionary with observation layout information.
    """
    if cfg is None:
        cfg = ObsConfig()
    
    # Policy group (action history)
    policy_size = cfg.action_dim * cfg.action_history
    
    # Proprio group (joint state, hand tips, contact) × history
    proprio_per_step = cfg.joint_pos_dim + cfg.joint_vel_dim + cfg.hand_tips_dim + cfg.contact_dim
    proprio_size = proprio_per_step * cfg.proprio_history
    
    # Perception group (point cloud + pose) × history
    pc_per_step = cfg.num_points * 3  # 32 × 3 = 96
    perception_per_step = pc_per_step + cfg.pose_dim  # 96 + 7 = 103
    perception_size = perception_per_step * cfg.perception_history
    
    # Targets group (target point clouds + target poses) × 1
    target_pc_size = cfg.window_size * cfg.num_points * 3  # 5 × 32 × 3 = 480
    target_poses_size = cfg.window_size * cfg.pose_dim  # 5 × 7 = 35
    targets_size = target_pc_size + target_poses_size
    
    # Compute offsets
    policy_start = 0
    proprio_start = policy_size
    perception_start = proprio_start + proprio_size
    targets_start = perception_start + perception_size
    total_size = targets_start + targets_size
    
    return {
        'policy': {'start': policy_start, 'size': policy_size},
        'proprio': {'start': proprio_start, 'size': proprio_size},
        'perception': {'start': perception_start, 'size': perception_size},
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
    and reshape point cloud data from perception and targets groups.
    
    Args:
        obs: Flat observation tensor (B, obs_dim)
        cfg: Observation configuration
        
    Returns:
        point_obs: Point cloud in point-centric format (B, num_points, point_dim)
        proprio_obs: All remaining features (B, proprio_dim)
    """
    if cfg is None:
        cfg = ObsConfig()
    
    B = obs.shape[0]
    layout = compute_obs_layout(cfg)
    
    # Extract groups
    policy = obs[:, layout['policy']['start']:layout['policy']['start'] + layout['policy']['size']]
    proprio = obs[:, layout['proprio']['start']:layout['proprio']['start'] + layout['proprio']['size']]
    perception = obs[:, layout['perception']['start']:layout['perception']['start'] + layout['perception']['size']]
    targets = obs[:, layout['targets']['start']:layout['targets']['start'] + layout['targets']['size']]
    
    # Extract point clouds from perception (ignoring pose)
    # perception = [pc_0, pose_0, pc_1, pose_1, ..., pc_4, pose_4]
    # Each pc is (num_points × 3), pose is 7
    pc_per_step = layout['point_cloud_per_step']
    perception_per_step = pc_per_step + cfg.pose_dim
    
    perception_pcs = []
    perception_poses = []
    for i in range(cfg.perception_history):
        start = i * perception_per_step
        pc = perception[:, start:start + pc_per_step]
        pose = perception[:, start + pc_per_step:start + perception_per_step]
        perception_pcs.append(pc)
        perception_poses.append(pose)
    
    # Stack perception point clouds: (B, history, num_points × 3)
    perception_pcs = torch.stack(perception_pcs, dim=1)  # (B, 5, 96)
    perception_pcs = perception_pcs.view(B, cfg.perception_history, cfg.num_points, 3)  # (B, 5, 32, 3)
    
    # Extract target point clouds (ignoring poses)
    target_pc_size = cfg.window_size * cfg.num_points * 3
    target_pcs = targets[:, :target_pc_size]
    target_pcs = target_pcs.view(B, cfg.window_size, cfg.num_points, 3)  # (B, 5, 32, 3)
    
    # Combine into point-centric format
    # Per point: [hist_0, hist_1, ..., hist_4, tgt_0, tgt_1, ..., tgt_4] each is 3D xyz
    # Shape: (B, 5+5, 32, 3) → permute → (B, 32, 10, 3) → reshape → (B, 32, 30)
    all_pcs = torch.cat([perception_pcs, target_pcs], dim=1)  # (B, 10, 32, 3)
    point_obs = all_pcs.permute(0, 2, 1, 3)  # (B, 32, 10, 3)
    point_obs = point_obs.reshape(B, cfg.num_points, -1)  # (B, 32, 30)
    
    # Proprio includes: policy (actions), proprio (joints/tips), and poses
    perception_poses = torch.cat(perception_poses, dim=1)  # (B, 5*7=35)
    target_poses = targets[:, target_pc_size:]  # (B, 35)
    
    proprio_obs = torch.cat([policy, proprio, perception_poses, target_poses], dim=1)
    
    return point_obs, proprio_obs

