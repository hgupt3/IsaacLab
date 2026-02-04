# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination functions for Y2R trajectory following task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_error_magnitude

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Time out based on per-env effective episode duration.
    
    Unlike Isaac Lab's default time_out which uses a fixed episode_length_s for all envs,
    this uses per-env t_episode_end computed at reset based on:
    - Actual waypoint count (0 to max_waypoints)
    - skip_manipulation flag (grasp-only episodes)
    
    This ensures envs with fewer waypoints or skip_manipulation terminate earlier
    rather than waiting for the maximum possible episode duration.
    
    Returns:
        Boolean tensor (num_envs,) - True if episode should end.
    """
    # Use env step counter (updated before termination check) instead of
    # trajectory_manager.phase_time (updated during observation compute, later in the step).
    # This keeps termination aligned with the current step and avoids running
    # past the computed episode end.
    tm = env.trajectory_manager
    return (env.episode_length_buf * env.step_dt) >= tm.t_episode_end


def abnormal_robot_state(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminating environment when violation of velocity limits detects, this usually indicates unstable physics caused
    by very bad, or aggressive action"""
    robot: Articulation = env.scene[asset_cfg.name]
    return (robot.data.joint_vel.abs() > (robot.data.joint_vel_limits * 2)).any(dim=1)


def trajectory_deviation(
    env: ManagerBasedRLEnv,
    threshold: float = 0.15,
    rot_threshold: float = 1.0,
) -> torch.Tensor:
    """Terminate if object deviates too far from current trajectory target.
    
    Uses use_point_cloud flag from trajectory config to decide which error metric:
    - Point cloud mode: Uses aligned mean error (point-to-point distance)
    - Pose mode: Uses aligned position + rotation errors
    
    Args:
        env: The environment.
        threshold: Max allowed position/mean error (meters).
        rot_threshold: Max allowed rotation error (radians, pose mode only).
    
    Returns:
        Boolean tensor (num_envs,) - True if should terminate.
    """
    use_point_cloud = env.trajectory_manager.cfg.mode.use_point_cloud
    
    if not use_point_cloud:
        # Pose mode: check both position AND rotation thresholds
        pos_error = env._cached_aligned_pose_errors['pos']  # (N,)
        rot_error = env._cached_aligned_pose_errors['rot']  # (N,)
        # Terminate if EITHER threshold exceeded
        return (pos_error > threshold) | (rot_error > rot_threshold)
    
    # Point cloud mode: use mean errors
    return env._cached_aligned_mean_error > threshold


def hand_pose_deviation(
    env: ManagerBasedRLEnv,
    pos_threshold: float = 0.1,
    rot_threshold: float = 1.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate if hand deviates too far from target during grasp/release phases.
    
    Only active during grasp (phase 0) and release (phase 2) phases.
    During manipulation phase (phase 1), returns False (no termination).
    
    Args:
        env: The environment.
        pos_threshold: Max allowed position error (meters).
        rot_threshold: Max allowed rotation error (radians).
        robot_cfg: Scene entity config for robot.
    
    Returns:
        Boolean tensor (num_envs,) - True if should terminate.
    """
    cfg = env.cfg.y2r_cfg
    N = env.num_envs
    
    # If hand trajectory is disabled, don't terminate
    if not cfg.hand_trajectory.enabled:
        return torch.zeros(N, dtype=torch.bool, device=env.device)
    
    trajectory_manager = env.trajectory_manager
    robot: Articulation = env.scene[robot_cfg.name]
    
    # Get palm body index (cache it)
    if not hasattr(env, '_term_palm_body_idx'):
        palm_ids = robot.find_bodies("palm_link")[0]
        if len(palm_ids) == 0:
            return torch.zeros(N, dtype=torch.bool, device=env.device)
        env._term_palm_body_idx = palm_ids[0]
    
    palm_idx = env._term_palm_body_idx
    
    # Get current phase (0=grasp, 1=manipulation, 2=release)
    phase = trajectory_manager.get_phase()  # (N,)
    
    # Only terminate during grasp and release phases
    in_gated_phase = (phase == 0) | (phase == 2)
    
    # If no envs are in gated phase, skip computation
    if not in_gated_phase.any():
        return torch.zeros(N, dtype=torch.bool, device=env.device)
    
    # Get actual palm pose in world frame
    palm_pos_w = robot.data.body_pos_w[:, palm_idx]  # (N, 3)
    palm_quat_w = robot.data.body_quat_w[:, palm_idx]  # (N, 4)
    
    # Get target palm pose from trajectory
    hand_target = trajectory_manager.get_current_hand_target()  # (N, 7)
    target_pos_w = hand_target[:, :3]  # (N, 3)
    target_quat_w = hand_target[:, 3:7]  # (N, 4)
    
    # Compute errors
    pos_error = (palm_pos_w - target_pos_w).norm(dim=-1)  # (N,)
    rot_error = quat_error_magnitude(palm_quat_w, target_quat_w)  # (N,)
    
    # Terminate if EITHER threshold exceeded (only during gated phases)
    exceeds_threshold = (pos_error > pos_threshold) | (rot_error > rot_threshold)
    
    # Only terminate in grasp/release phases
    return in_gated_phase & exceeds_threshold
