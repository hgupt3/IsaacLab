# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Trajectory Manager for manipulation tasks.

Generates entire trajectory at reset, then uses sliding window for observations.
Simple and deterministic - no dynamic regeneration.
"""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import quat_mul, quat_from_euler_xyz, quat_apply, sample_uniform

if TYPE_CHECKING:
    from .trajectory_cfg import TrajectoryParamsCfg


class TrajectoryManager:
    """
    Generates full trajectory at reset, provides sliding window view.
    
    Simple approach:
    - At reset: generate ALL targets for entire episode
    - At step: advance window index
    - Collision: just clamp z to table surface (no penetration)
    """

    def __init__(
        self,
        cfg: "TrajectoryParamsCfg",
        num_envs: int,
        device: str,
        table_height: float = 0.255,
    ):
        """
        Initialize trajectory manager.
        
        Args:
            cfg: TrajectoryParamsCfg with timing and trajectory parameters.
            num_envs: Number of parallel environments.
            device: Torch device.
            table_height: Z-height of table surface for penetration check.
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        self.table_height = table_height
        
        # Timing
        self.target_dt = 1.0 / cfg.target_hz
        self.total_duration = (
            cfg.pickup_duration
            + cfg.manipulate_duration
            + cfg.place_duration
            + cfg.release_duration
        )
        
        # Total number of targets for entire episode
        self.total_targets = int(self.total_duration / self.target_dt) + 1
        
        # Full trajectory buffer: (num_envs, total_targets, 7) for pos(3) + quat(4)
        self.trajectory = torch.zeros(num_envs, self.total_targets, 7, device=device)
        
        # Current window index (which target is "current")
        self.current_idx = torch.zeros(num_envs, dtype=torch.long, device=device)
        
        # Phase time (seconds since episode start)
        self.phase_time = torch.zeros(num_envs, device=device)
        
        # Start/goal poses
        self.start_poses = torch.zeros(num_envs, 7, device=device)
        self.goal_poses = torch.zeros(num_envs, 7, device=device)
        
        # Environment origins (XY offset for each cloned env)
        self.env_origins = torch.zeros(num_envs, 3, device=device)
        
        # Waypoints: (num_envs, max_waypoints, 7)
        self.waypoints = torch.zeros(num_envs, cfg.num_waypoints_max, 7, device=device)
        self.num_waypoints = torch.zeros(num_envs, dtype=torch.long, device=device)
        
        # Local point cloud for penetration checking (set by observation term)
        self.points_local: torch.Tensor | None = None

    def reset(
        self,
        env_ids: torch.Tensor,
        start_poses: torch.Tensor,
        env_origins: torch.Tensor,
    ):
        """
        Reset trajectory for specified environments.
        
        Generates the ENTIRE trajectory upfront.
        
        Args:
            env_ids: Environment indices to reset.
            start_poses: Starting object poses (n, 7) - pos(3) + quat(4).
            env_origins: World origin offset for each env (n, 3).
        """
        if len(env_ids) == 0:
            return

        n = len(env_ids)
        
        # Reset timing
        self.phase_time[env_ids] = 0.0
        self.current_idx[env_ids] = 0
        
        # Store start poses and environment origins
        self.start_poses[env_ids] = start_poses
        self.env_origins[env_ids] = env_origins
        
        # Sample waypoints
        self._sample_waypoints(env_ids, start_poses, env_origins)
        
        # Sample goal (use start_poses Z for stable placement)
        self._sample_goal(env_ids, env_origins, start_poses)
        
        # Generate entire trajectory
        self._generate_full_trajectory(env_ids)

    def step(self, dt: float):
        """
        Advance trajectory by one timestep.
        
        Args:
            dt: Simulation timestep.
        """
        self.phase_time += dt
        
        # Compute current target index from time
        new_idx = (self.phase_time / self.target_dt).long()
        self.current_idx = torch.clamp(new_idx, max=self.total_targets - self.cfg.window_size)

    def get_window_targets(self) -> torch.Tensor:
        """
        Get current window of targets.
        
        Returns:
            (num_envs, window_size, 7) tensor of pos(3) + quat(4).
        """
        window_size = self.cfg.window_size
        
        # Build indices: (num_envs, window_size)
        batch_idx = torch.arange(self.num_envs, device=self.device).unsqueeze(1)
        window_offset = torch.arange(window_size, device=self.device).unsqueeze(0)
        target_idx = self.current_idx.unsqueeze(1) + window_offset
        
        # Clamp to valid range
        target_idx = torch.clamp(target_idx, max=self.total_targets - 1)
        
        return self.trajectory[batch_idx, target_idx]

    def get_current_target(self) -> torch.Tensor:
        """Get the current (first) target. Returns (num_envs, 7)."""
        batch_idx = torch.arange(self.num_envs, device=self.device)
        return self.trajectory[batch_idx, self.current_idx]

    def _sample_waypoints(self, env_ids: torch.Tensor, start_poses: torch.Tensor, env_origins: torch.Tensor):
        """Sample 0-2 intermediate waypoints."""
        n = len(env_ids)
        cfg = self.cfg
        
        # Random number of waypoints
        self.num_waypoints[env_ids] = torch.randint(
            cfg.num_waypoints_min, cfg.num_waypoints_max + 1, (n,), device=self.device
        )
        
        # Convert start to local frame for clamping
        start_local = start_poses[:, :3] - env_origins
        
        for wp_idx in range(cfg.num_waypoints_max):
            # Position: sample in local frame (None = use workspace, else offset from start)
            if cfg.waypoint_position_range_x is None:
                x_local = sample_uniform(cfg.workspace_x[0], cfg.workspace_x[1], (n,), self.device)
            else:
                x_local = start_local[:, 0] + sample_uniform(cfg.waypoint_position_range_x[0], cfg.waypoint_position_range_x[1], (n,), self.device)
                x_local = x_local.clamp(cfg.workspace_x[0], cfg.workspace_x[1])
            
            if cfg.waypoint_position_range_y is None:
                y_local = sample_uniform(cfg.workspace_y[0], cfg.workspace_y[1], (n,), self.device)
            else:
                y_local = start_local[:, 1] + sample_uniform(cfg.waypoint_position_range_y[0], cfg.waypoint_position_range_y[1], (n,), self.device)
                y_local = y_local.clamp(cfg.workspace_y[0], cfg.workspace_y[1])
            
            if cfg.waypoint_position_range_z is None:
                z_local = sample_uniform(cfg.workspace_z_min, cfg.workspace_z_max, (n,), self.device)
            else:
                z_local = sample_uniform(cfg.waypoint_position_range_z[0], cfg.waypoint_position_range_z[1], (n,), self.device)
                z_local = z_local.clamp(cfg.workspace_z_min, cfg.workspace_z_max)
            
            pos_local = torch.stack([x_local, y_local, z_local], dim=-1)
            
            # Convert back to world frame
            pos_world = pos_local + env_origins
            
            # Orientation: random rotation in WORLD frame if enabled
            if cfg.vary_waypoint_orientation:
                euler = sample_uniform(-cfg.waypoint_max_rotation, cfg.waypoint_max_rotation, (n, 3), self.device)
                delta_quat = quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])
                # delta_quat * start_quat = apply delta in world frame
                quat = quat_mul(delta_quat, start_poses[:, 3:7])
            else:
                quat = start_poses[:, 3:7].clone()
            
            self.waypoints[env_ids, wp_idx, :3] = pos_world
            self.waypoints[env_ids, wp_idx, 3:7] = quat

    def _sample_goal(self, env_ids: torch.Tensor, env_origins: torch.Tensor, start_poses: torch.Tensor):
        """Sample goal pose (on table).
        
        Uses start_poses Z to preserve stable object placement height.
        """
        n = len(env_ids)
        cfg = self.cfg
        
        # Random XY (None = use workspace bounds)
        x_range = cfg.goal_x_range if cfg.goal_x_range is not None else cfg.workspace_x
        y_range = cfg.goal_y_range if cfg.goal_y_range is not None else cfg.workspace_y
        
        goal_x_local = sample_uniform(x_range[0], x_range[1], (n,), self.device)
        goal_y_local = sample_uniform(y_range[0], y_range[1], (n,), self.device)
        
        # Convert to world frame
        goal_x = goal_x_local + env_origins[:, 0]
        goal_y = goal_y_local + env_origins[:, 1]
        
        # Z: use same height as start (object is stable on table at start)
        # This preserves the stable placement height regardless of object size
        goal_z_val = start_poses[:, 2]
        
        # Random yaw rotation in WORLD frame (around world Z axis)
        yaw_delta = sample_uniform(-math.pi, math.pi, (n,), self.device)
        delta_quat = quat_from_euler_xyz(
            torch.zeros(n, device=self.device),
            torch.zeros(n, device=self.device),
            yaw_delta,
        )
        # delta_quat * start_quat = apply yaw in world frame
        goal_quat = quat_mul(delta_quat, start_poses[:, 3:7])
        
        self.goal_poses[env_ids, 0] = goal_x
        self.goal_poses[env_ids, 1] = goal_y
        self.goal_poses[env_ids, 2] = goal_z_val
        self.goal_poses[env_ids, 3:7] = goal_quat

    def _generate_full_trajectory(self, env_ids: torch.Tensor):
        """Generate entire trajectory for given environments."""
        n = len(env_ids)
        cfg = self.cfg
        T = self.total_targets
        
        # Time array for all targets
        times = torch.arange(T, device=self.device).float() * self.target_dt  # (T,)
        times = times.unsqueeze(0).expand(n, T)  # (n, T)
        
        # Compute progress (0 to 1) based on phase
        progress = self._compute_progress(times)  # (n, T)
        
        # Interpolate positions through start -> waypoints -> goal
        positions = self._interpolate_positions(env_ids, progress)  # (n, T, 3)
        
        # Interpolate orientations
        orientations = self._interpolate_orientations(env_ids, progress)  # (n, T, 4)
        
        # Apply penetration check (clamp z so point cloud stays above table)
        positions = self._apply_penetration_check(positions, orientations, env_ids)
        
        # Store trajectory
        self.trajectory[env_ids, :, :3] = positions
        self.trajectory[env_ids, :, 3:7] = orientations

    def _compute_progress(self, times: torch.Tensor) -> torch.Tensor:
        """
        Compute progress (0 to 1) with phase-based easing.
        
        Phases:
        - pickup: stationary at start (progress = 0)
        - manipulate: ease-in movement (0 -> ~0.7)
        - place: ease-out to goal (~0.7 -> 1.0)
        - release: stationary at goal (progress = 1)
        """
        cfg = self.cfg
        n, T = times.shape
        
        # Phase boundaries
        t1 = cfg.pickup_duration
        t2 = t1 + cfg.manipulate_duration
        t3 = t2 + cfg.place_duration
        # t4 = t3 + cfg.release_duration (end)
        
        # Movement duration (manipulate + place)
        movement_duration = cfg.manipulate_duration + cfg.place_duration
        
        progress = torch.zeros_like(times)
        
        # Pickup phase: progress = 0 (stationary)
        # Already zeros
        
        # Manipulate phase: ease-in (slow start, speeds up)
        mask1 = (times >= t1) & (times < t2)
        if mask1.any():
            local_t = (times[mask1] - t1) / cfg.manipulate_duration
            # Ease-in: slow start, accelerate
            eased = local_t ** cfg.ease_power
            # Scale to fraction of total movement
            progress[mask1] = eased * (cfg.manipulate_duration / movement_duration)
        
        # Place phase: ease-out (slows down to goal)
        mask2 = (times >= t2) & (times < t3)
        if mask2.any():
            local_t = (times[mask2] - t2) / cfg.place_duration
            # Ease-out: decelerate to stop
            eased = 1.0 - (1.0 - local_t) ** cfg.ease_power
            base = cfg.manipulate_duration / movement_duration
            progress[mask2] = base + eased * (cfg.place_duration / movement_duration)
        
        # Release phase: progress = 1 (stationary at goal)
        mask3 = times >= t3
        progress[mask3] = 1.0
        
        return progress.clamp(0.0, 1.0)

    def _interpolate_positions(self, env_ids: torch.Tensor, progress: torch.Tensor) -> torch.Tensor:
        """
        Interpolate positions using Catmull-Rom spline through control points.
        
        Unlike Bezier, Catmull-Rom splines PASS THROUGH all control points.
        Control points: start -> waypoints -> goal
        """
        n, T = progress.shape
        cfg = self.cfg
        max_wp = cfg.num_waypoints_max
        
        start = self.start_poses[env_ids, :3]  # (n, 3)
        goal = self.goal_poses[env_ids, :3]    # (n, 3)
        waypoints = self.waypoints[env_ids, :, :3]  # (n, max_wp, 3)
        num_wp = self.num_waypoints[env_ids]   # (n,)
        
        # Build control points: (n, max_wp+2, 3)
        control = torch.zeros(n, max_wp + 2, 3, device=self.device)
        control[:, 0] = start
        control[:, 1:1+max_wp] = waypoints
        control[:, -1] = goal
        
        # For unused waypoints, interpolate between start and goal
        for i in range(max_wp):
            unused = num_wp <= i
            if unused.any():
                t_val = (i + 1) / (max_wp + 1)
                control[unused, i + 1] = (1 - t_val) * start[unused] + t_val * goal[unused]
        
        # Catmull-Rom spline interpolation
        num_segments = max_wp + 1  # Number of segments between control points
        
        # Map progress [0, 1] to segment index and local t
        p_scaled = progress * num_segments  # (n, T)
        seg_idx = p_scaled.long().clamp(0, num_segments - 1)  # (n, T)
        t_local = p_scaled - seg_idx.float()  # (n, T) in [0, 1]
        
        # For each point, get 4 control points (p0, p1, p2, p3) for Catmull-Rom
        # Pad control points at ends for boundary handling
        padded = torch.cat([control[:, :1], control, control[:, -1:]], dim=1)  # (n, max_wp+4, 3)
        
        # Gather control points: p0=seg_idx, p1=seg_idx+1, p2=seg_idx+2, p3=seg_idx+3
        batch_idx = torch.arange(n, device=self.device).unsqueeze(1).expand(n, T)
        p0 = padded[batch_idx, seg_idx]      # (n, T, 3)
        p1 = padded[batch_idx, seg_idx + 1]  # (n, T, 3)
        p2 = padded[batch_idx, seg_idx + 2]  # (n, T, 3)
        p3 = padded[batch_idx, seg_idx + 3]  # (n, T, 3)
        
        # Catmull-Rom formula: P(t) = 0.5 * ((2*p1) + (-p0+p2)*t + (2*p0-5*p1+4*p2-p3)*t^2 + (-p0+3*p1-3*p2+p3)*t^3)
        t = t_local.unsqueeze(-1)  # (n, T, 1)
        t2 = t * t
        t3 = t2 * t
        
        positions = 0.5 * (
            2 * p1 +
            (-p0 + p2) * t +
            (2*p0 - 5*p1 + 4*p2 - p3) * t2 +
            (-p0 + 3*p1 - 3*p2 + p3) * t3
        )
        
        return positions  # (n, T, 3)

    def _interpolate_orientations(self, env_ids: torch.Tensor, progress: torch.Tensor) -> torch.Tensor:
        """Interpolate orientations using piecewise SLERP through waypoints."""
        n, T = progress.shape
        cfg = self.cfg
        max_wp = cfg.num_waypoints_max
        
        # Build orientation control points: start -> waypoints -> goal
        quats = torch.zeros(n, max_wp + 2, 4, device=self.device)
        quats[:, 0] = self.start_poses[env_ids, 3:7]
        quats[:, 1:1+max_wp] = self.waypoints[env_ids, :, 3:7]
        quats[:, -1] = self.goal_poses[env_ids, 3:7]
        
        # For unused waypoints, use start orientation
        num_wp = self.num_waypoints[env_ids]
        for i in range(max_wp):
            unused = num_wp <= i
            if unused.any():
                # SLERP between start and goal for unused
                t_val = (i + 1) / (max_wp + 1)
                quats[unused, i + 1] = self._batched_slerp(
                    quats[unused, 0:1].expand(-1, 1, 4),
                    quats[unused, -1:].expand(-1, 1, 4),
                    torch.full((unused.sum(), 1), t_val, device=self.device)
                ).squeeze(1)
        
        # Piecewise SLERP through control points
        num_segments = max_wp + 1
        p_scaled = progress * num_segments  # (n, T)
        seg_idx = p_scaled.long().clamp(0, num_segments - 1)  # (n, T)
        t_local = p_scaled - seg_idx.float()  # (n, T)
        
        # Gather segment endpoints
        batch_idx = torch.arange(n, device=self.device).unsqueeze(1).expand(n, T)
        q0 = quats[batch_idx, seg_idx]      # (n, T, 4)
        q1 = quats[batch_idx, seg_idx + 1]  # (n, T, 4)
        
        return self._batched_slerp(q0, q1, t_local)

    def _batched_slerp(self, q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Batched spherical linear interpolation."""
        # Ensure shortest path
        dot = (q0 * q1).sum(dim=-1, keepdim=True)
        q1 = torch.where(dot < 0, -q1, q1)
        dot = dot.abs().clamp(-1.0, 1.0)
        
        # Linear interpolation for nearly parallel quaternions
        linear_mask = dot > 0.9995
        
        # Spherical interpolation
        theta_0 = torch.acos(dot)
        theta = theta_0 * t.unsqueeze(-1)
        
        q2 = q1 - q0 * dot
        q2 = q2 / (q2.norm(dim=-1, keepdim=True) + 1e-8)
        
        result = q0 * torch.cos(theta) + q2 * torch.sin(theta)
        
        # Linear fallback
        linear_result = q0 + t.unsqueeze(-1) * (q1 - q0)
        linear_result = linear_result / (linear_result.norm(dim=-1, keepdim=True) + 1e-8)
        
        return torch.where(linear_mask, linear_result, result)

    def _apply_penetration_check(
        self, 
        positions: torch.Tensor, 
        orientations: torch.Tensor,
        env_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Point cloud-based penetration check: ensure no point goes below table.
        
        Fully vectorized - transforms all point clouds for all trajectory poses
        in parallel and adjusts positions so lowest points stay above table.
        
        Args:
            positions: (n, T, 3) trajectory positions in world frame.
            orientations: (n, T, 4) trajectory orientations (wxyz).
            env_ids: Environment indices for these positions.
            
        Returns:
            Adjusted positions with all point cloud points above table.
        """
        if self.points_local is None:
            # Fallback to start pose Z if no point cloud available
            min_z = self.start_poses[env_ids, 2:3].unsqueeze(1)
            positions[:, :, 2:3] = torch.maximum(positions[:, :, 2:3], min_z)
            return positions
        
        n, T, _ = positions.shape
        P = self.points_local.shape[1]  # num points
        
        # Get local points for these envs: (n, P, 3)
        local_pts = self.points_local[env_ids]
        
        # Table surface Z for each env: (n,)
        table_z = self.table_height + self.env_origins[env_ids, 2]
        
        # Expand for vectorized computation:
        # local_pts: (n, P, 3) -> (n, 1, P, 3) -> (n, T, P, 3)
        # positions: (n, T, 3) -> (n, T, 1, 3)
        # orientations: (n, T, 4) -> (n, T, 1, 4) -> (n, T, P, 4)
        
        local_exp = local_pts.unsqueeze(1).expand(n, T, P, 3)  # (n, T, P, 3)
        pos_exp = positions.unsqueeze(2)  # (n, T, 1, 3)
        quat_exp = orientations.unsqueeze(2).expand(n, T, P, 4)  # (n, T, P, 4)
        
        # Transform all points in parallel: (n, T, P, 3)
        # Reshape for quat_apply, then reshape back
        world_pts = quat_apply(
            quat_exp.reshape(-1, 4), 
            local_exp.reshape(-1, 3)
        ).reshape(n, T, P, 3) + pos_exp
        
        # Find min Z per (env, time): (n, T)
        min_pt_z = world_pts[:, :, :, 2].min(dim=2).values
        
        # Penetration depth: how much below table (n, T)
        penetration = table_z.unsqueeze(1) - min_pt_z
        
        # Shift up where penetrating
        shift = torch.clamp(penetration, min=0.0)
        positions[:, :, 2] = positions[:, :, 2] + shift
        
        return positions

    # ---- Phase helpers ----
    
    def get_phase(self) -> torch.Tensor:
        """Get current phase index for each environment (0=pickup, 1=manipulate, 2=place, 3=release)."""
        cfg = self.cfg
        t = self.phase_time
        
        phase = torch.zeros_like(t, dtype=torch.long)
        t1 = cfg.pickup_duration
        t2 = t1 + cfg.manipulate_duration
        t3 = t2 + cfg.place_duration
        
        phase[t >= t1] = 1  # manipulate
        phase[t >= t2] = 2  # place
        phase[t >= t3] = 3  # release
        
        return phase

    def is_in_release_phase(self) -> torch.Tensor:
        """Check if environments are in release phase."""
        return self.get_phase() == 3
    
    def is_in_pickup_phase(self) -> torch.Tensor:
        """Check if environments are in pickup phase."""
        return self.get_phase() == 0
