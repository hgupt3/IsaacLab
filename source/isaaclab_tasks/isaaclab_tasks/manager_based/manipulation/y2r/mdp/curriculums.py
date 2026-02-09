# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.envs import mdp
from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import quat_error_magnitude, quat_apply


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def initial_final_interpolate_fn(env: ManagerBasedRLEnv, env_id, data, initial_value, final_value, difficulty_term_str):
    """
    Interpolate between initial value iv and final value fv, for any arbitrarily
    nested structure of lists/tuples in 'data'. Scalars (int/float) are handled
    at the leaves.
    """
    # get the fraction scalar on the device
    difficulty_term: DifficultyScheduler = getattr(env.curriculum_manager.cfg, difficulty_term_str).func
    frac = difficulty_term.difficulty_frac
    if frac < 0.1:
        # no-op during start, since the difficulty fraction near 0 is wasting of resource.
        return mdp.modify_env_param.NO_CHANGE

    # convert iv/fv to tensors, but we'll peel them apart in recursion
    initial_value_tensor = torch.tensor(initial_value, device=env.device)
    final_value_tensor = torch.tensor(final_value, device=env.device)

    return _recurse(initial_value_tensor.tolist(), final_value_tensor.tolist(), data, frac)


def _recurse(iv_elem, fv_elem, data_elem, frac):
    # If it's a sequence, rebuild the same type with each element recursed
    if isinstance(data_elem, Sequence) and not isinstance(data_elem, (str, bytes)):
        # Note: we assume initial value element and final value element have the same structure as data
        return type(data_elem)(_recurse(iv_e, fv_e, d_e, frac) for iv_e, fv_e, d_e in zip(iv_elem, fv_elem, data_elem))
    # Otherwise it's a leaf scalar: do the interpolation
    new_val = frac * (fv_elem - iv_elem) + iv_elem
    if isinstance(data_elem, int):
        return int(new_val.item())
    else:
        # cast floats or any numeric
        return new_val.item()


class DifficultyScheduler(ManagerTermBase):
    """Adaptive difficulty scheduler for trajectory curriculum learning.

    Tracks per-environment difficulty levels and adjusts them based on task performance
    and/or a step-based schedule.
    
    Step-based scheduler (optional):
    - If step_interval is set, difficulty floor increases every step_interval steps
    - Floor guarantees minimum progress regardless of agent performance
    - Performance-based logic can still push difficulty higher than floor
    
    Performance-based advancement:
    - Advances when episode timed out (full trajectory completed) and object is at goal
    - Can be disabled by setting use_performance=False

    The normalized average difficulty is exposed as `difficulty_frac` for curriculum interpolation.
    
    Tolerance parameters (pos_tol, rot_tol, pc_tol):
    - If None: uses current success threshold (adaptive with curriculum)
    - If float: uses fixed tolerance value
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        init_difficulty = self.cfg.params.get("init_difficulty", 0)
        self.current_adr_difficulties = torch.ones(env.num_envs, device=env.device) * init_difficulty
        self.difficulty_frac = 0
        
        # Step-based scheduler state
        self.step_based_floor = init_difficulty

    def get_state(self):
        return self.current_adr_difficulties

    def set_state(self, state: torch.Tensor):
        self.current_adr_difficulties = state.clone().to(self._env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        init_difficulty: int = 0,
        min_difficulty: int = 0,
        max_difficulty: int = 50,
        promotion_only: bool = False,
        use_trajectory: bool = True,  # Always True for trajectory tasks
        pos_tol: float | None = None,  # None = use current success threshold
        rot_tol: float | None = None,  # None = use current success threshold
        pc_tol: float | None = None,   # None = use current success threshold
        step_interval: int | None = None,  # Steps between floor increases (None = disabled)
        use_performance: bool = True,  # Whether to use performance-based advancement
    ):
        # Compute step-based difficulty floor using Isaac Lab's global step counter
        if step_interval is not None and step_interval > 0:
            self.step_based_floor = min(
                max_difficulty,
                init_difficulty + (env.common_step_counter // step_interval)
            )
        
        # Performance-based logic
        if use_performance:
            trajectory_manager = env.trajectory_manager
            use_point_cloud = trajectory_manager.cfg.mode.use_point_cloud
            
            # Get current success thresholds from reward manager (already curriculum-adjusted)
            success_cfg = env.reward_manager.get_term_cfg("trajectory_success")
            current_pos_threshold = success_cfg.params.get("pos_threshold", 0.05)
            current_rot_threshold = success_cfg.params.get("rot_threshold", 0.3)
            
            # Use provided tolerances or fall back to current success thresholds
            effective_pos_tol = pos_tol if pos_tol is not None else current_pos_threshold
            effective_rot_tol = rot_tol if rot_tol is not None else current_rot_threshold
            effective_pc_tol = pc_tol if pc_tol is not None else current_pos_threshold
            
            # Compare object pose against FINAL GOAL (not current lookahead target)
            goal_poses = trajectory_manager.goal_poses[env_ids]  # (n, 7) world frame
            goal_pos = goal_poses[:, :3]  # (n, 3)
            goal_quat = goal_poses[:, 3:7]  # (n, 4)
            
            # Object current pose (world frame)
            object_pos_w = env.scene["object"].data.root_pos_w[env_ids]  # (n, 3)
            object_quat_w = env.scene["object"].data.root_quat_w[env_ids]  # (n, 4)
            
            if use_point_cloud:
                # Point cloud mode: compute mean point error against goal point cloud
                points_local = trajectory_manager.points_local  # (num_envs, P, 3)
                if points_local is not None:
                    pts = points_local[env_ids]  # (n, P, 3)
                    goal_pts_w = quat_apply(
                        goal_quat.unsqueeze(1).expand_as(pts), pts
                    ) + goal_pos.unsqueeze(1)  # (n, P, 3)
                    curr_pts_w = quat_apply(
                        object_quat_w.unsqueeze(1).expand_as(pts), pts
                    ) + object_pos_w.unsqueeze(1)  # (n, P, 3)
                    mean_errors = (curr_pts_w - goal_pts_w).norm(dim=-1).mean(dim=-1)  # (n,)
                    at_goal = mean_errors < effective_pc_tol
                else:
                    at_goal = torch.zeros(len(env_ids), dtype=torch.bool, device=env.device)
            else:
                # Pose mode: position + rotation error against final goal
                pos_errors = (object_pos_w - goal_pos).norm(dim=-1)  # (n,)
                rot_errors = quat_error_magnitude(object_quat_w, goal_quat)  # (n,)
                at_goal = (pos_errors < effective_pos_tol) & (rot_errors < effective_rot_tol)
            
            # Promote only if episode timed out (completed full trajectory) AND object is at goal
            if hasattr(env, 'reset_time_outs') and env.reset_time_outs is not None:
                timed_out = env.reset_time_outs[env_ids]
            else:
                timed_out = torch.zeros(len(env_ids), dtype=torch.bool, device=env.device)
            move_up = timed_out & at_goal
            
            demot = self.current_adr_difficulties[env_ids] if promotion_only else self.current_adr_difficulties[env_ids] - 1
            perf_difficulties = torch.where(
                move_up,
                self.current_adr_difficulties[env_ids] + 1,
                demot,
            )
        else:
            # No performance-based logic, just use current difficulties
            perf_difficulties = self.current_adr_difficulties[env_ids]
        
        # Combine: step-based floor is the minimum, performance can exceed it
        floor_tensor = torch.tensor(self.step_based_floor, device=env.device, dtype=perf_difficulties.dtype)
        self.current_adr_difficulties[env_ids] = torch.maximum(
            perf_difficulties, floor_tensor
        ).clamp(min=min_difficulty, max=max_difficulty)
        
        self.difficulty_frac = torch.mean(self.current_adr_difficulties) / max(max_difficulty, 1)
        return self.difficulty_frac
