# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import math as math_utils
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def action_rate_l2_clamped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1).clamp(-1000, 1000)


def action_l2_clamped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action), dim=1).clamp(-1000, 1000)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward reaching the object using a tanh-kernel on end-effector distance.

    The reward is close to 1 when the maximum distance between the object and any end-effector body is small.
    Disabled during release phase for trajectory task.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    asset_pos = asset.data.body_pos_w[:, asset_cfg.body_ids]
    object_pos = object.data.root_pos_w
    object_ee_distance = torch.norm(asset_pos - object_pos[:, None, :], dim=-1).max(dim=-1).values
    reward = 1 - torch.tanh(object_ee_distance / std)
    
    # Disable during release phase
    in_release = env.trajectory_manager.is_in_release_phase()
    reward = torch.where(in_release, -2*reward, reward)
    
    return reward


def contacts(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""

    thumb_contact_sensor: ContactSensor = env.scene.sensors["thumb_link_3_object_s"]
    index_contact_sensor: ContactSensor = env.scene.sensors["index_link_3_object_s"]
    middle_contact_sensor: ContactSensor = env.scene.sensors["middle_link_3_object_s"]
    ring_contact_sensor: ContactSensor = env.scene.sensors["ring_link_3_object_s"]
    # check if contact force is above threshold
    thumb_contact = thumb_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    index_contact = index_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    middle_contact = middle_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    ring_contact = ring_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)

    thumb_contact_mag = torch.norm(thumb_contact, dim=-1)
    index_contact_mag = torch.norm(index_contact, dim=-1)
    middle_contact_mag = torch.norm(middle_contact, dim=-1)
    ring_contact_mag = torch.norm(ring_contact, dim=-1)
    good_contact_cond1 = (thumb_contact_mag > threshold) & (
        (index_contact_mag > threshold) | (middle_contact_mag > threshold) | (ring_contact_mag > threshold)
    )

    return good_contact_cond1


def success_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    align_asset_cfg: SceneEntityCfg,
    pos_std: float,
    rot_std: float | None = None,
) -> torch.Tensor:
    """Reward success by comparing commanded pose to the object pose using tanh kernels on error."""

    asset: RigidObject = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[align_asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, des_quat_w = combine_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, command[:, :3], command[:, 3:7]
    )
    pos_err, rot_err = compute_pose_error(des_pos_w, des_quat_w, object.data.root_pos_w, object.data.root_quat_w)
    pos_dist = torch.norm(pos_err, dim=1)
    if not rot_std:
        # square is not necessary but this help to keep the final value between having rot_std or not roughly the same
        return (1 - torch.tanh(pos_dist / pos_std)) ** 2
    rot_dist = torch.norm(rot_err, dim=1)
    return (1 - torch.tanh(pos_dist / pos_std)) * (1 - torch.tanh(rot_dist / rot_std))


def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg, align_asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of commanded position using tanh kernel, gated by contact presence."""

    asset: RigidObject = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[align_asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    distance = torch.norm(object.data.root_pos_w - des_pos_w, dim=1)
    return (1 - torch.tanh(distance / std)) * contacts(env, 1.0).float()


def orientation_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg, align_asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of commanded orientation using tanh kernel, gated by contact presence."""

    asset: RigidObject = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[align_asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = math_utils.quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    quat_distance = math_utils.quat_error_magnitude(object.data.root_quat_w, des_quat_w)

    return (1 - torch.tanh(quat_distance / std)) * contacts(env, 1.0).float()


# ==================== Trajectory-based rewards ====================


def lookahead_tracking(
    env: ManagerBasedRLEnv,
    std: float = 0.1,
    decay: float = 0.5,
    contact_threshold: float = 0.05,
    rot_std: float = 0.5,
) -> torch.Tensor:
    """
    Reward tracking future trajectory targets.
    
    Supports two modes based on cached errors:
    - Point cloud mode: Uses _cached_mean_errors (point-to-point distance)
    - Pose mode: Uses _cached_pose_errors (position + rotation errors)
    
    GATED BY CONTACT: Only gives reward when fingers are in contact with object.
    Exponential decay weights: weight_i = decay^i (i=0 is current, i=W-1 is furthest)
    
    Args:
        env: The environment.
        std: Standard deviation for position/point cloud tanh kernel.
        decay: Exponential decay factor (0-1). Lower = focus on current target.
        contact_threshold: Force threshold for contact detection (N).
        rot_std: Standard deviation for rotation tanh kernel (pose mode only).
    
    Returns:
        Reward tensor (num_envs,).
    """
    # Check which mode via trajectory config flag
    use_point_cloud = env.trajectory_manager.cfg.use_point_cloud
    
    # DEBUG: Print all rewards for env 0 every 10 steps (controlled by trajectory_cfg.debug_print_rewards)
    cfg = env.trajectory_manager.cfg
    step = env.episode_length_buf[0].item()
    # Debug: print flag value on first step
    if cfg.debug_print_rewards and step % 10 == 0:
        rm = env.reward_manager
        print(f"\n=== Episode Step {step} | Env 0 Rewards ===")
        # Use get_active_iterable_terms to get per-term values
        for name, values in rm.get_active_iterable_terms(0):
            term_cfg = rm.get_term_cfg(name)
            print(f"  {name:20s}: weighted={values[0]:+.4f} (w={term_cfg.weight:+.2f})")
    
    if not use_point_cloud:
        # Pose mode: combine position and rotation rewards
        pos_errors = env._cached_pose_errors['pos']  # (N, W)
        rot_errors = env._cached_pose_errors['rot']  # (N, W)
        W = pos_errors.shape[1]
        
        # Exponential decay weights
        weights = decay ** torch.arange(W, device=env.device, dtype=torch.float32)
        weights = weights / weights.sum()
        
        # Weighted average errors
        weighted_pos_error = (pos_errors * weights.unsqueeze(0)).sum(dim=-1)  # (N,)
        weighted_rot_error = (rot_errors * weights.unsqueeze(0)).sum(dim=-1)  # (N,)
        
        # Reward as product: (1 - tanh(pos_err/std)) * (1 - tanh(rot_err/rot_std))
        pos_reward = 1.0 - torch.tanh(weighted_pos_error / std)
        rot_reward = 1.0 - torch.tanh(weighted_rot_error / rot_std)
        reward = (pos_reward + rot_reward*2)/3.0
    else:
        # Point cloud mode: use mean errors
        mean_errors = env._cached_mean_errors  # (N, W)
        W = mean_errors.shape[1]
        
        # Exponential decay weights
        weights = decay ** torch.arange(W, device=env.device, dtype=torch.float32)
        weights = weights / weights.sum()
        
        # Weighted average error
        weighted_error = (mean_errors * weights.unsqueeze(0)).sum(dim=-1)  # (N,)
        
        # Convert to reward using tanh kernel
        reward = 1.0 - torch.tanh(weighted_error / std)
    
    # Gate by contact: only reward when grasping (thumb + at least one other finger)
    has_contact = contacts(env, contact_threshold)  # (N,) bool
    reward = torch.where(has_contact, reward, torch.zeros_like(reward))
    
    in_release = env.trajectory_manager.is_in_release_phase()
    reward = torch.where(in_release, torch.zeros_like(reward), reward)
    
    return reward


def trajectory_success(
    env: ManagerBasedRLEnv,
    error_threshold: float = 0.02,
    rot_threshold: float = 0.2,
) -> torch.Tensor:
    """
    Success reward: object at current target in release phase.
    
    Supports two modes based on cached errors:
    - Point cloud mode: Uses _cached_mean_errors (point-to-point distance)
    - Pose mode: Uses _cached_pose_errors (position + rotation errors)
    
    Gives reward when conditions are met:
    1. In release phase
    2. Object close to current target
       - Point cloud: mean error < error_threshold
       - Pose: pos_error < error_threshold AND rot_error < rot_threshold
    
    Args:
        env: The environment.
        error_threshold: Max position/mean error for success (meters).
        rot_threshold: Max rotation error for success (radians, pose mode only).
    
    Returns:
        Reward tensor (num_envs,) - 1.0 if success, 0.0 otherwise.
    """
    trajectory_manager = env.trajectory_manager
    use_point_cloud = trajectory_manager.cfg.use_point_cloud
    
    # Condition 1: In release phase
    in_release = trajectory_manager.is_in_release_phase()  # (N,) bool
    
    # Condition 2: Object at current target
    if not use_point_cloud:
        # Pose mode: check both position and rotation
        pos_error = env._cached_pose_errors['pos'][:, 0]  # (N,)
        rot_error = env._cached_pose_errors['rot'][:, 0]  # (N,)
        at_goal = (pos_error < error_threshold) & (rot_error < rot_threshold)
    else:
        # Point cloud mode: use mean errors
        mean_errors = env._cached_mean_errors  # (N, W)
        current_error = mean_errors[:, 0]  # (N,)
        at_goal = current_error < error_threshold
    
    # Success = both conditions
    success_mask = in_release & at_goal
    return success_mask.float()


def arm_table_binary_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    table_z: float = 0.255,
    threshold_mid: float = 0.08,
    threshold_distal: float = 0.04,
) -> torch.Tensor:
    """
    Binary penalty for arm links below safe height thresholds.
    
    Checks body_ids from asset_cfg (expected to be links 3-7 + palm_link).
    Links are split into two groups by their index in body_ids:
    - First 3 bodies (links 3,4,5): threshold = table_z + threshold_mid
    - Remaining bodies (links 6,7,palm): threshold = table_z + threshold_distal
    
    Returns fraction of links below their threshold (0.0 to 1.0).
    
    Args:
        env: The environment.
        asset_cfg: SceneEntityCfg with body_ids for links to check.
        table_z: Base table height (without env origin offset).
        threshold_mid: Safe height above table for mid-arm links (m).
        threshold_distal: Safe height above table for distal links + palm (m).
    
    Returns:
        Penalty tensor (num_envs,) - fraction of links below threshold.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    
    # Get Z positions for selected bodies: (N, L)
    link_z = robot.data.body_pos_w[:, asset_cfg.body_ids, 2]
    
    # Account for env origins
    env_table_z = table_z + env.scene.env_origins[:, 2]  # (N,)
    
    num_bodies = len(asset_cfg.body_ids)
    
    # Split into mid (first 3) and distal (remaining)
    # Assumes body_ids are ordered: [link3, link4, link5, link6, link7, palm]
    num_mid = 3
    
    # Mid links (3,4,5): table_z + threshold_mid
    mid_threshold = env_table_z.unsqueeze(1) + threshold_mid  # (N, 1)
    mid_violations = (link_z[:, :num_mid] < mid_threshold).float()  # (N, 3)
    
    # Distal links (6,7,palm): table_z + threshold_distal
    distal_threshold = env_table_z.unsqueeze(1) + threshold_distal  # (N, 1)
    distal_violations = (link_z[:, num_mid:] < distal_threshold).float()  # (N, remaining)
    
    # Concatenate and compute fraction
    all_violations = torch.cat([mid_violations, distal_violations], dim=1)  # (N, L)
    fraction_below = all_violations.mean(dim=1)  # (N,)
    
    return fraction_below
