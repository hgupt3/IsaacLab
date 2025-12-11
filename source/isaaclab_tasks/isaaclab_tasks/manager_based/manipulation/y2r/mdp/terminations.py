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

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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
    - Point cloud mode: Uses _cached_mean_errors (point-to-point distance)
    - Pose mode: Uses _cached_pose_errors (position + rotation errors)
    
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
        pos_error = env._cached_pose_errors['pos'][:, 0]  # (N,)
        rot_error = env._cached_pose_errors['rot'][:, 0]  # (N,)
        # Terminate if EITHER threshold exceeded
        return (pos_error > threshold) | (rot_error > rot_threshold)
    
    # Point cloud mode: use mean errors
    mean_errors = env._cached_mean_errors  # (N, W)
    current_error = mean_errors[:, 0]  # (N,)
    return current_error > threshold
