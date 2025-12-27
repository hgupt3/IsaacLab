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

from isaaclab.utils.math import quat_mul, quat_from_euler_xyz, quat_apply, quat_apply_inverse, quat_inv, sample_uniform, quat_from_matrix

from .mdp.utils import get_stable_object_placement, compute_z_offset_from_usd, read_object_scales_from_usd

if TYPE_CHECKING:
    from .config_loader import Y2RConfig


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
        cfg: "Y2RConfig",
        num_envs: int,
        device: str,
        table_height: float = 0.255,
        object_prim_path: str | None = None,
    ):
        """
        Initialize trajectory manager.
        
        Args:
            cfg: Y2RConfig with timing and trajectory parameters.
            num_envs: Number of parallel environments.
            device: Torch device.
            table_height: Z-height of table surface for penetration check.
            object_prim_path: USD prim path pattern for objects (for stable pose computation).
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        self.table_height = table_height
        self.object_prim_path = object_prim_path
        
        # Extract trajectory settings from config
        traj = cfg.trajectory
        
        # Timing
        self.target_dt = 1.0 / traj.target_hz
        # For push-T, the trajectory horizon is the rolling window (e.g., 7s), not the full episode
        if cfg.push_t.enabled:
            self.total_duration = cfg.push_t.rolling_window
        else:
            max_keypoints = cfg.hand_trajectory.keypoints.count[1]
            max_waypoints = cfg.waypoints.count[1]
            self.total_duration = (
                traj.phases.grasp
                + traj.phases.manipulate_base
                + traj.phases.manipulate_per_keypoint * max_keypoints
                + cfg.waypoints.pause_duration * max_waypoints  # Pause at each waypoint
                + traj.phases.hand_release
            )
        
        # Total number of targets for the trajectory buffer
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
        wp_cfg = cfg.waypoints
        self.waypoints = torch.zeros(num_envs, wp_cfg.count[1], 7, device=device)
        self.num_waypoints = torch.zeros(num_envs, dtype=torch.long, device=device)
        
        # Local point cloud for penetration checking (set by observation term)
        self.points_local: torch.Tensor | None = None
        
        # For push-T replanning: track when to regenerate trajectory
        self.last_replan_idx = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.current_object_poses = torch.zeros(num_envs, 7, device=device)
        
        # ========== Hand trajectory buffers ==========
        # Hand trajectory buffer: (num_envs, total_targets, 7) for palm pos(3) + quat(4)
        self.hand_trajectory = torch.zeros(num_envs, self.total_targets, 7, device=device)
        
        # Grasp pose: initial grasp pose relative to object (num_envs, 7)
        # This is the transform from object frame to palm frame at grasp
        self.grasp_pose = torch.zeros(num_envs, 7, device=device)
        
        # Grasp keypoints: chain of perturbed grasp transforms (num_envs, max_keypoints, 7)
        hand_cfg = cfg.hand_trajectory
        max_keypoints = hand_cfg.keypoints.count[1]
        self.grasp_keypoints = torch.zeros(num_envs, max_keypoints, 7, device=device)
        self.num_grasp_keypoints = torch.zeros(num_envs, dtype=torch.long, device=device)
        
        # Release pose: target palm position for release phase (num_envs, 7)
        self.release_pose = torch.zeros(num_envs, 7, device=device)
        
        # Starting palm pose (captured at reset for interpolation)
        self.start_palm_pose = torch.zeros(num_envs, 7, device=device)
        
        # Object scales for variable scale support
        # Read from USD prims after prestartup scale randomization
        if object_prim_path is not None:
            scales = read_object_scales_from_usd(object_prim_path, num_envs)
            self._object_scales = scales.to(device)
        else:
            self._object_scales = torch.ones(num_envs, device=device)
        
        # Pre-compute push_t base z-offset (scale=1.0) for goal positioning
        # At reset time, we multiply by actual object scale
        self._push_t_base_z_offset = None
        if cfg.push_t.enabled and cfg.push_t.object_usd:
            from pathlib import Path
            config_dir = Path(__file__).parent
            usd_path = str(config_dir / cfg.push_t.object_usd)
            
            if cfg.push_t.object_z_offset is not None:
                # Config specifies exact z_offset, use as-is (no scaling)
                self._push_t_base_z_offset = cfg.push_t.object_z_offset
            else:
                # Compute base z_offset with scale=1.0
                # Will be multiplied by actual object scale at reset time
                goal_rot = cfg.push_t.goal_rotation if cfg.push_t.goal_rotation else cfg.push_t.outline_rotation
                self._push_t_base_z_offset = compute_z_offset_from_usd(
                    usd_path=usd_path,
                    rotation_wxyz=tuple(goal_rot),
                    scale=1.0,  # Base z_offset at unit scale
                )

    def reset(
        self,
        env_ids: torch.Tensor,
        start_poses: torch.Tensor,
        env_origins: torch.Tensor,
        start_palm_poses: torch.Tensor | None = None,
    ):
        """
        Reset trajectory for specified environments.
        
        Generates the ENTIRE trajectory upfront.
        
        Args:
            env_ids: Environment indices to reset.
            start_poses: Starting object poses (n, 7) - pos(3) + quat(4).
            env_origins: World origin offset for each env (n, 3).
            start_palm_poses: Starting palm poses (n, 7) - optional, for hand trajectory.
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
        
        # Object scales are read once at init from USD prims (after prestartup randomization)
        
        # Sample waypoints
        self._sample_waypoints(env_ids, start_poses, env_origins)
        
        # Sample goal (use start_poses Z for stable placement)
        self._sample_goal(env_ids, env_origins, start_poses)
        
        # === Hand trajectory generation (if enabled) ===
        if self.cfg.hand_trajectory.enabled:
            # Store starting palm pose
            if start_palm_poses is not None:
                self.start_palm_pose[env_ids] = start_palm_poses
            else:
                # Default: palm above object, facing down
                default_palm_pos = start_poses[:, :3].clone()
                default_palm_pos[:, 2] += 0.15  # 15cm above object
                default_palm_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).expand(n, 4)
                self.start_palm_pose[env_ids, :3] = default_palm_pos
                self.start_palm_pose[env_ids, 3:7] = default_palm_quat
            
            # Sample grasp region and compute grasp pose
            grasp_pos, grasp_quat = self._sample_grasp_region(env_ids, start_poses)
            
            # Store grasp pose relative to object (for manipulation phase)
            obj_pos = start_poses[:, :3]
            obj_quat = start_poses[:, 3:7]
            # grasp_pos_local = R_obj^-1 * (grasp_pos_world - obj_pos)
            grasp_pos_local = quat_apply_inverse(obj_quat, grasp_pos - obj_pos)
            grasp_quat_local = quat_mul(quat_inv(obj_quat), grasp_quat)
            
            self.grasp_pose[env_ids, :3] = grasp_pos_local
            self.grasp_pose[env_ids, 3:7] = grasp_quat_local
            
            # Sample grasp keypoints (perturbed from grasp pose)
            self._sample_grasp_keypoints(env_ids)
            
            # Sample release pose
            self._sample_release_pose(env_ids, env_origins)
        
        # Generate entire object trajectory
        # For push-T mode, use adaptive speed from the start
        if self.cfg.push_t.enabled:
            # Initialize current_object_poses with start poses for initial trajectory
            self.current_object_poses[env_ids] = start_poses
            self._replan_from_current(env_ids)
            # Reset replan tracking so first replan happens at correct time
            self.last_replan_idx[env_ids] = 0
        else:
            self._generate_full_trajectory(env_ids)
        
        # Generate hand trajectory (after object trajectory, since it depends on it)
        if self.cfg.hand_trajectory.enabled:
            self._generate_hand_trajectory(env_ids)

    def step(self, dt: float, object_poses: torch.Tensor | None = None):
        """
        Advance trajectory by one timestep.
        
        For push-T mode: replans trajectory from current object position every window_size timesteps.
        
        Args:
            dt: Simulation timestep.
            object_poses: Current object poses (num_envs, 7) for replanning. Only used in push-T mode.
        """
        self.phase_time += dt
        
        # Compute current target index from time
        new_idx = (self.phase_time / self.target_dt).long()
        
        # For push-T: don't clamp idx (need continuous replanning)
        # For normal: clamp to trajectory length
        if self.cfg.push_t.enabled:
            self.current_idx = new_idx
        else:
            self.current_idx = torch.clamp(new_idx, max=self.total_targets - self.cfg.trajectory.window_size)
        
        # Push-T replanning: regenerate trajectory from current position every window_size timesteps
        if self.cfg.push_t.enabled and object_poses is not None:
            self.current_object_poses = object_poses
            replan_interval = self.cfg.trajectory.window_size
            
            # Check which envs need replanning (every window_size steps, but not at idx 0)
            steps_since_replan = self.current_idx - self.last_replan_idx
            needs_replan = (steps_since_replan >= replan_interval) & (self.current_idx > 0)
            
            if needs_replan.any():
                env_ids = torch.where(needs_replan)[0]
                self._replan_from_current(env_ids)
                # Reset to start of trajectory buffer after replanning
                # This ensures current_idx stays within buffer bounds
                self.current_idx[env_ids] = 0
                self.last_replan_idx[env_ids] = 0
                self.phase_time[env_ids] = 0.0

    def get_window_targets(self) -> torch.Tensor:
        """
        Get current window of targets.
        
        Returns:
            (num_envs, window_size, 7) tensor of pos(3) + quat(4).
        """
        window_size = self.cfg.trajectory.window_size
        
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
        wp_cfg = cfg.waypoints
        ws_cfg = cfg.workspace
        
        # Random number of waypoints
        self.num_waypoints[env_ids] = torch.randint(
            wp_cfg.count[0], wp_cfg.count[1] + 1, (n,), device=self.device
        )
        
        # Convert start to local frame for clamping
        start_local = start_poses[:, :3] - env_origins
        
        max_wp = wp_cfg.count[1]
        for wp_idx in range(max_wp):
            # Position: sample in local frame (None = use workspace, else offset from start)
            if wp_cfg.position_range.x is None:
                x_local = sample_uniform(ws_cfg.x[0], ws_cfg.x[1], (n,), self.device)
            else:
                x_local = start_local[:, 0] + sample_uniform(wp_cfg.position_range.x[0], wp_cfg.position_range.x[1], (n,), self.device)
                x_local = x_local.clamp(ws_cfg.x[0], ws_cfg.x[1])
            
            if wp_cfg.position_range.y is None:
                y_local = sample_uniform(ws_cfg.y[0], ws_cfg.y[1], (n,), self.device)
            else:
                y_local = start_local[:, 1] + sample_uniform(wp_cfg.position_range.y[0], wp_cfg.position_range.y[1], (n,), self.device)
                y_local = y_local.clamp(ws_cfg.y[0], ws_cfg.y[1])
            
            if wp_cfg.position_range.z is None:
                z_local = sample_uniform(ws_cfg.z[0], ws_cfg.z[1], (n,), self.device)
            else:
                z_local = sample_uniform(wp_cfg.position_range.z[0], wp_cfg.position_range.z[1], (n,), self.device)
                z_local = z_local.clamp(ws_cfg.z[0], ws_cfg.z[1])
            
            pos_local = torch.stack([x_local, y_local, z_local], dim=-1)
            
            # Convert back to world frame
            pos_world = pos_local + env_origins
            
            # Orientation: random rotation in WORLD frame if enabled
            if wp_cfg.vary_orientation:
                euler = sample_uniform(-wp_cfg.max_rotation, wp_cfg.max_rotation, (n, 3), self.device)
                delta_quat = quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])
                # delta_quat * start_quat = apply delta in world frame
                quat = quat_mul(delta_quat, start_poses[:, 3:7])
            else:
                quat = start_poses[:, 3:7].clone()
            
            self.waypoints[env_ids, wp_idx, :3] = pos_world
            self.waypoints[env_ids, wp_idx, 3:7] = quat

    def _sample_goal(self, env_ids: torch.Tensor, env_origins: torch.Tensor, start_poses: torch.Tensor):
        """Sample goal pose (on table).
        
        Uses stable poses from trimesh for goal placement (z-position + orientation).
        Different stable poses have different z-positions (e.g., cube on edge vs flat).
        """
        n = len(env_ids)
        cfg = self.cfg
        goal_cfg = cfg.goal
        ws_cfg = cfg.workspace
        
        # Random XY (None = use workspace bounds)
        x_range = goal_cfg.x_range if goal_cfg.x_range is not None else ws_cfg.x
        y_range = goal_cfg.y_range if goal_cfg.y_range is not None else ws_cfg.y
        
        goal_x_local = sample_uniform(x_range[0], x_range[1], (n,), self.device)
        goal_y_local = sample_uniform(y_range[0], y_range[1], (n,), self.device)
        
        # Convert to world frame
        goal_x = goal_x_local + env_origins[:, 0]
        goal_y = goal_y_local + env_origins[:, 1]
        
        # Get stable placement (z-position + orientation) from trimesh
        # Skip for push-T mode (use config values or auto-computed z-offset)
        if cfg.push_t.enabled:
            # Base goal on table surface + center-to-bottom distance + safety margin
            # Multiply base_z_offset by actual object scale for variable scale support
            base_z_offset = self._push_t_base_z_offset
            actual_z_offset = base_z_offset * self._object_scales[env_ids]
            goal_z_local = self.table_height + actual_z_offset + cfg.randomization.reset.z_offset
            
            # Apply goal offset from config (if specified)
            if cfg.push_t.goal_offset is not None:
                goal_x_local = goal_x_local + cfg.push_t.goal_offset[0]
                goal_y_local = goal_y_local + cfg.push_t.goal_offset[1]
                goal_z_local = goal_z_local + cfg.push_t.goal_offset[2]
                # Re-compute world frame positions with offset
                goal_x = goal_x_local + env_origins[:, 0]
                goal_y = goal_y_local + env_origins[:, 1]
            
            # Use goal_rotation from config, or fall back to outline_rotation
            if cfg.push_t.goal_rotation is not None:
                goal_quat = torch.tensor([cfg.push_t.goal_rotation], device=self.device).repeat(n, 1)
            else:
                goal_quat = torch.tensor([cfg.push_t.outline_rotation], device=self.device).repeat(n, 1)
        else:
            env_ids_list = env_ids.tolist()
            goal_z_local, goal_quat = get_stable_object_placement(
                env_ids=env_ids_list,
                prim_path=self.object_prim_path,
                table_surface_z=self.table_height,
                randomize_pose=False,  # Use most stable pose
                z_offset=cfg.randomization.reset.z_offset,
            )
            goal_z_local = goal_z_local.to(self.device)
            goal_quat = goal_quat.to(self.device)
        
        # Add env origin z offset
        goal_z = goal_z_local + env_origins[:, 2]
        
        # Apply random yaw on top of stable pose (like start poses)
        # Skip random yaw for push-T mode - keep deterministic orientation
        if not cfg.push_t.enabled:
            yaw_delta = sample_uniform(-math.pi, math.pi, (n,), self.device)
            delta_quat = quat_from_euler_xyz(
                torch.zeros(n, device=self.device),
                torch.zeros(n, device=self.device),
                yaw_delta,
            )
            # Apply yaw in world frame on top of stable orientation
            goal_quat = quat_mul(delta_quat, goal_quat)
        
        self.goal_poses[env_ids, 0] = goal_x
        self.goal_poses[env_ids, 1] = goal_y
        self.goal_poses[env_ids, 2] = goal_z
        self.goal_poses[env_ids, 3:7] = goal_quat

    # ========== Hand Trajectory Methods ==========
    
    def _sample_grasp_region(
        self, 
        env_ids: torch.Tensor, 
        object_poses: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample grasp pose from object surface normals.
        
        Samples a surface point (excluding bottom hemisphere), then computes
        palm pose such that palm Z axis faces toward the object.
        
        Args:
            env_ids: Environment indices to sample for.
            object_poses: Object poses (n, 7) for these envs.
            
        Returns:
            Tuple of:
            - grasp_pos: (n, 3) palm position in world frame
            - grasp_quat: (n, 4) palm quaternion in world frame (wxyz)
        """
        from .mdp.utils import get_point_cloud_cache
        
        n = len(env_ids)
        cfg = self.cfg.hand_trajectory.grasp_sampling
        
        # Get cached points and normals
        cache = get_point_cloud_cache(
            env_ids=env_ids.tolist(),
            prim_path=self.object_prim_path,
            pool_size=256,
        )
        
        # Get points and normals for these envs
        # Note: cache tensors are on CPU, so use CPU indices for indexing
        env_ids_cpu = env_ids.cpu()
        geo_indices = cache.geo_indices[env_ids_cpu].to(self.device)
        scales = cache.scales[env_ids_cpu].to(self.device)
        all_points = cache.all_base_points.to(self.device)
        all_normals = cache.all_base_normals.to(self.device)
        
        # Get base points/normals per env (in object-local frame)
        env_points = all_points[geo_indices]  # (n, pool_size, 3)
        env_normals = all_normals[geo_indices]  # (n, pool_size, 3)
        
        # Apply scale to points (normals don't need scaling)
        env_points = env_points * scales.unsqueeze(1)
        
        # Transform normals to WORLD frame for filtering
        # Objects are placed on table with only yaw variation, so world-Z is always "up"
        obj_pos = object_poses[:, :3]  # (n, 3)
        obj_quat = object_poses[:, 3:7]  # (n, 4)
        
        # Rotate normals to world frame: (n, pool_size, 3)
        env_normals_flat = env_normals.reshape(-1, 3)  # (n*pool_size, 3)
        obj_quat_exp = obj_quat.unsqueeze(1).expand(n, 256, 4).reshape(-1, 4)  # (n*pool_size, 4)
        env_normals_w = quat_apply(obj_quat_exp, env_normals_flat).reshape(n, 256, 3)
        
        # Filter out bottom hemisphere in WORLD frame
        # Exclude normals pointing too far downward (toward table)
        exclude_z = -1.0 + cfg.exclude_bottom_fraction * 2  # e.g., 0.3 -> -0.4
        valid_mask = env_normals_w[:, :, 2] > exclude_z  # (n, pool_size)
        
        # Randomly select one valid point per env
        # Set invalid points to very low priority
        rand_vals = torch.rand(n, 256, device=self.device)
        rand_vals[~valid_mask] = -1.0  # Invalid points won't be selected
        selected_idx = rand_vals.argmax(dim=1)  # (n,)
        
        batch_idx = torch.arange(n, device=self.device)
        surface_points = env_points[batch_idx, selected_idx]  # (n, 3) in object-local frame
        surface_normals = env_normals[batch_idx, selected_idx]  # (n, 3) in object-local frame
        
        # Rotate surface point and normal to world frame
        surface_points_w = quat_apply(obj_quat, surface_points) + obj_pos
        surface_normals_w = quat_apply(obj_quat, surface_normals)
        surface_normals_w = surface_normals_w / (surface_normals_w.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Sample standoff distance
        standoff = sample_uniform(
            cfg.standoff_range[0], cfg.standoff_range[1], (n,), self.device
        )
        
        # Palm position: surface point + standoff along normal
        grasp_pos = surface_points_w + standoff.unsqueeze(-1) * surface_normals_w
        
        # Palm orientation: Z axis points toward object (opposite of normal)
        # We need to construct a rotation where palm_z = -surface_normal
        grasp_quat = self._quat_from_z_axis(-surface_normals_w)
        
        # Filter: exclude grasps where fingers point toward robot (+X direction)
        # If fingers point toward robot, flip 180° around palm's local Z axis
        if cfg.exclude_toward_robot:
            # Get finger direction (palm X axis) in world frame
            x_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device)
            finger_dir_w = quat_apply(grasp_quat, x_axis.expand(n, 3))  # (n, 3)
            
            # Check if finger X component is too positive (pointing toward robot at origin)
            toward_robot = finger_dir_w[:, 0] > cfg.toward_robot_threshold
            
            if toward_robot.any():
                # Rotate 180° around LOCAL Z axis (palm Z) to flip finger direction
                # Order: quat_mul(grasp_quat, flip_quat) applies flip in local frame first
                flip_quat = quat_from_euler_xyz(
                    torch.zeros(toward_robot.sum(), device=self.device),
                    torch.zeros(toward_robot.sum(), device=self.device),
                    torch.full((toward_robot.sum(),), math.pi, device=self.device),
                )
                # Apply flip in local frame: keeps palm facing object, flips finger direction
                grasp_quat[toward_robot] = quat_mul(grasp_quat[toward_robot], flip_quat)
        
        return grasp_pos, grasp_quat
    
    def _quat_from_z_axis(self, z_target: torch.Tensor) -> torch.Tensor:
        """Construct quaternion where Z axis points along z_target.
        
        Args:
            z_target: (n, 3) target Z axis direction (normalized).
            
        Returns:
            (n, 4) quaternion (wxyz format).
        """
        n = z_target.shape[0]
        
        # Normalize target
        z = z_target / (z_target.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Choose an arbitrary perpendicular vector for X axis
        # Use world up (0, 0, 1) unless z is parallel to it
        world_up = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(n, 3)
        
        # If z is nearly parallel to world_up, use world_forward instead
        dot = (z * world_up).sum(dim=-1, keepdim=True).abs()
        alt_ref = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(n, 3)
        ref = torch.where(dot > 0.9, alt_ref, world_up)
        
        # X = ref × Z (perpendicular to Z)
        x = torch.cross(ref, z, dim=-1)
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Y = Z × X
        y = torch.cross(z, x, dim=-1)
        
        # Build rotation matrix and convert to quaternion
        rot_mat = torch.stack([x, y, z], dim=-1)  # (n, 3, 3)
        
        return quat_from_matrix(rot_mat)
    
    def _sample_grasp_keypoints(self, env_ids: torch.Tensor):
        """Sample chain of perturbed grasp transforms for in-hand manipulation.
        
        Each keypoint is perturbed from the previous one (not from initial).
        Stored as transforms relative to object.
        
        Args:
            env_ids: Environment indices.
        """
        n = len(env_ids)
        cfg = self.cfg.hand_trajectory.keypoints
        
        # Random number of keypoints per env
        self.num_grasp_keypoints[env_ids] = torch.randint(
            cfg.count[0], cfg.count[1] + 1, (n,), device=self.device
        )
        
        max_kp = cfg.count[1]
        if max_kp == 0:
            return
        
        # Start from initial grasp pose
        prev_pos = self.grasp_pose[env_ids, :3].clone()  # (n, 3) relative to object
        prev_quat = self.grasp_pose[env_ids, 3:7].clone()  # (n, 4)
        
        for kp_idx in range(max_kp):
            # Position perturbation
            pos_delta = sample_uniform(
                -cfg.pos_perturbation, cfg.pos_perturbation, (n, 3), self.device
            )
            pos_delta[:, 2] *= 0.5  # Less vertical perturbation
            new_pos = prev_pos + pos_delta
            
            # Rotation perturbation (small euler angles)
            euler_delta = sample_uniform(
                -cfg.rot_perturbation, cfg.rot_perturbation, (n, 3), self.device
            )
            rot_delta = quat_from_euler_xyz(euler_delta[:, 0], euler_delta[:, 1], euler_delta[:, 2])
            new_quat = quat_mul(rot_delta, prev_quat)
            
            # Store keypoint
            self.grasp_keypoints[env_ids, kp_idx, :3] = new_pos
            self.grasp_keypoints[env_ids, kp_idx, 3:7] = new_quat
            
            # Chain: next keypoint perturbs from this one
            prev_pos = new_pos
            prev_quat = new_quat

    def _sample_release_pose(self, env_ids: torch.Tensor, env_origins: torch.Tensor):
        """Sample release pose in workspace region.
        
        Args:
            env_ids: Environment indices.
            env_origins: World origin offset for each env (n, 3).
        """
        n = len(env_ids)
        cfg = self.cfg.hand_trajectory.release
        ws_cfg = self.cfg.workspace
        
        # Sample position in release region
        x_range = cfg.position_range.x if cfg.position_range.x is not None else ws_cfg.x
        y_range = cfg.position_range.y if cfg.position_range.y is not None else ws_cfg.y
        z_range = cfg.position_range.z if cfg.position_range.z is not None else ws_cfg.z
        
        x_local = sample_uniform(x_range[0], x_range[1], (n,), self.device)
        y_local = sample_uniform(y_range[0], y_range[1], (n,), self.device)
        z_local = sample_uniform(z_range[0], z_range[1], (n,), self.device)
        
        # Convert to world frame
        release_pos = torch.stack([x_local, y_local, z_local], dim=-1) + env_origins
        
        # Ensure minimum distance from goal object
        goal_pos = self.goal_poses[env_ids, :3]
        to_release = release_pos - goal_pos
        dist = to_release.norm(dim=-1, keepdim=True)
        min_dist = cfg.min_distance_from_object
        
        # If too close, push away
        too_close = dist < min_dist
        if too_close.any():
            direction = to_release / (dist + 1e-8)
            release_pos = torch.where(
                too_close,
                goal_pos + direction * min_dist,
                release_pos
            )
        
        # Release orientation: palm faces down (neutral pose)
        # Palm Z points down = [0, 0, -1]
        palm_z_down = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(n, 3)
        release_quat = self._quat_from_z_axis(palm_z_down)
        
        self.release_pose[env_ids, :3] = release_pos
        self.release_pose[env_ids, 3:7] = release_quat

    def _generate_hand_trajectory(self, env_ids: torch.Tensor):
        """Generate hand (palm) trajectory through grasp → manipulation → release.
        
        Args:
            env_ids: Environment indices.
        """
        n = len(env_ids)
        T = self.total_targets
        cfg = self.cfg
        phases = cfg.trajectory.phases
        
        # Phase timing - use CONSISTENT duration matching object trajectory
        grasp_duration = phases.grasp
        release_duration = phases.hand_release
        
        # Get actual num_kp per env (for interpolation) but use MAX for timing
        num_kp = self.num_grasp_keypoints[env_ids]  # (n,)
        max_kp = cfg.hand_trajectory.keypoints.count[1]
        max_wp = cfg.waypoints.count[1]
        
        # Manipulation duration matches object trajectory (fixed, not per-env)
        manip_base = phases.manipulate_base
        manip_per_kp = phases.manipulate_per_keypoint
        pause_duration = cfg.waypoints.pause_duration
        manip_duration = manip_base + manip_per_kp * max_kp + pause_duration * max_wp
        
        # Time array
        times = torch.arange(T, device=self.device).float() * self.target_dt  # (T,)
        times = times.unsqueeze(0).expand(n, T)  # (n, T)
        
        # Phase boundaries (consistent across all envs)
        t_grasp_end = grasp_duration
        t_manip_end = t_grasp_end + manip_duration  # scalar, not per-env
        t_release_end = t_manip_end + release_duration
        
        # Initialize trajectory
        positions = torch.zeros(n, T, 3, device=self.device)
        orientations = torch.zeros(n, T, 4, device=self.device)
        orientations[:, :, 0] = 1.0  # Identity quaternion default
        
        # Get start and end poses
        start_pos = self.start_palm_pose[env_ids, :3]  # (n, 3)
        start_quat = self.start_palm_pose[env_ids, 3:7]  # (n, 4)
        grasp_pos = self.grasp_pose[env_ids, :3]  # (n, 3) relative to object
        grasp_quat = self.grasp_pose[env_ids, 3:7]  # (n, 4)
        release_pos = self.release_pose[env_ids, :3]  # (n, 3)
        release_quat = self.release_pose[env_ids, 3:7]  # (n, 4)
        
        # Object start pose (for converting grasp to world frame)
        obj_pos = self.start_poses[env_ids, :3]
        obj_quat = self.start_poses[env_ids, 3:7]
        
        # Convert grasp pose to world frame
        grasp_pos_w = quat_apply(obj_quat, grasp_pos) + obj_pos
        grasp_quat_w = quat_mul(obj_quat, grasp_quat)
        
        # === Phase 1: Grasp (interpolate start → grasp) ===
        grasp_mask = times < t_grasp_end  # (n, T)
        grasp_progress = (times / grasp_duration).clamp(0.0, 1.0)  # (n, T)
        
        # Vectorized linear interpolation for position
        p = grasp_progress.unsqueeze(-1)  # (n, T, 1)
        grasp_lerp_pos = (1 - p) * start_pos.unsqueeze(1) + p * grasp_pos_w.unsqueeze(1)  # (n, T, 3)
        grasp_lerp_quat = self._batched_slerp(
            start_quat.unsqueeze(1).expand(n, T, 4),
            grasp_quat_w.unsqueeze(1).expand(n, T, 4),
            grasp_progress
        )  # (n, T, 4)
        positions = torch.where(grasp_mask.unsqueeze(-1), grasp_lerp_pos, positions)
        orientations = torch.where(grasp_mask.unsqueeze(-1), grasp_lerp_quat, orientations)
        
        # === Phase 2: Manipulation (interpolate through keypoints, relative to object) ===
        manip_mask = (times >= t_grasp_end) & (times < t_manip_end)
        
        # Get object trajectory for transforming relative poses to world frame
        obj_trajectory = self.trajectory[env_ids]  # (n, T, 7)
        obj_pos_t = obj_trajectory[:, :, :3]  # (n, T, 3)
        obj_quat_t = obj_trajectory[:, :, 3:7]  # (n, T, 4)
        
        # Keypoints for interpolation (already have max_kp from above)
        keypoints = self.grasp_keypoints[env_ids]  # (n, max_kp, 7)
        
        # Compute manipulation progress (0 to 1) within manipulation phase
        manip_time = (times - t_grasp_end).clamp(min=0.0)  # (n, T)
        manip_progress = (manip_time / manip_duration).clamp(0.0, 1.0)  # (n, T)
        
        # Build padded keypoint chain: [grasp_pose, kp_0, kp_1, ..., kp_{max-1}]
        # Shape: (n, max_kp + 1, 7)
        kp_chain = torch.zeros(n, max_kp + 1, 7, device=self.device)
        kp_chain[:, 0, :3] = grasp_pos
        kp_chain[:, 0, 3:7] = grasp_quat
        if max_kp > 0:
            kp_chain[:, 1:, :] = keypoints
        
        # Number of segments per env (at least 1)
        num_segments = (num_kp + 1).float().clamp(min=1.0)  # (n,)
        
        # Compute segment index for each timestep: (n, T)
        seg_idx = (manip_progress * num_segments.unsqueeze(1)).long()
        seg_idx = seg_idx.clamp(0, max_kp)  # Clamp to valid range
        
        # Local progress within segment
        seg_start = seg_idx.float() / num_segments.unsqueeze(1)  # (n, T)
        seg_end = (seg_idx.float() + 1) / num_segments.unsqueeze(1)  # (n, T)
        seg_duration = seg_end - seg_start
        local_progress = ((manip_progress - seg_start) / seg_duration.clamp(min=1e-6)).clamp(0.0, 1.0)  # (n, T)
        
        # Get "from" and "to" keypoints using advanced indexing
        batch_idx = torch.arange(n, device=self.device).unsqueeze(1).expand(n, T)  # (n, T)
        
        from_pos = kp_chain[batch_idx, seg_idx, :3]  # (n, T, 3)
        from_quat = kp_chain[batch_idx, seg_idx, 3:7]  # (n, T, 4)
        
        # "to" is next keypoint, but clamped to last valid one
        to_idx = (seg_idx + 1).clamp(max=max_kp)
        # For envs with 0 keypoints, to_idx should point to grasp_pose (idx 0)
        to_idx = torch.where(num_kp.unsqueeze(1) == 0, torch.zeros_like(to_idx), to_idx)
        
        to_pos = kp_chain[batch_idx, to_idx, :3]  # (n, T, 3)
        to_quat = kp_chain[batch_idx, to_idx, 3:7]  # (n, T, 4)
        
        # Interpolate position
        lp = local_progress.unsqueeze(-1)  # (n, T, 1)
        interp_pos_rel = (1 - lp) * from_pos + lp * to_pos  # (n, T, 3)
        
        # Interpolate orientation
        interp_quat_rel = self._batched_slerp(from_quat, to_quat, local_progress)  # (n, T, 4)
        
        # Transform interpolated relative pose to world frame using object trajectory
        manip_pos = quat_apply(
            obj_quat_t.reshape(-1, 4), 
            interp_pos_rel.reshape(-1, 3)
        ).reshape(n, T, 3) + obj_pos_t
        manip_quat = quat_mul(
            obj_quat_t.reshape(-1, 4), 
            interp_quat_rel.reshape(-1, 4)
        ).reshape(n, T, 4)
        
        positions = torch.where(manip_mask.unsqueeze(-1), manip_pos, positions)
        orientations = torch.where(manip_mask.unsqueeze(-1), manip_quat, orientations)
        
        # === Phase 3: Release (interpolate to release pose) ===
        release_mask = times >= t_manip_end
        release_progress = ((times - t_manip_end) / release_duration).clamp(0.0, 1.0)
        
        # Get final manipulation pose (last keypoint or grasp_pose if no keypoints)
        # Use kp_chain built above: index is num_kp (last keypoint) or 0 if no keypoints
        final_kp_idx = num_kp.clamp(max=max_kp)  # (n,)
        batch_idx_1d = torch.arange(n, device=self.device)
        final_rel_pos = kp_chain[batch_idx_1d, final_kp_idx, :3]  # (n, 3)
        final_rel_quat = kp_chain[batch_idx_1d, final_kp_idx, 3:7]  # (n, 4)
        
        # Transform final relative pose using object position at END of manipulation
        # (NOT goal_poses - the object is at this position when release starts)
        manip_end_idx = int(t_manip_end / self.target_dt)
        manip_end_idx = min(manip_end_idx, T - 1)
        obj_pos_at_release = obj_pos_t[:, manip_end_idx, :]  # (n, 3)
        obj_quat_at_release = obj_quat_t[:, manip_end_idx, :]  # (n, 4)
        final_grasp_pos_w = quat_apply(obj_quat_at_release, final_rel_pos) + obj_pos_at_release
        final_grasp_quat_w = quat_mul(obj_quat_at_release, final_rel_quat)
        
        # Vectorized linear interpolation for release
        p_rel = release_progress.unsqueeze(-1)  # (n, T, 1)
        release_lerp_pos = (1 - p_rel) * final_grasp_pos_w.unsqueeze(1) + p_rel * release_pos.unsqueeze(1)
        release_lerp_quat = self._batched_slerp(
            final_grasp_quat_w.unsqueeze(1).expand(n, T, 4),
            release_quat.unsqueeze(1).expand(n, T, 4),
            release_progress
        )
        positions = torch.where(release_mask.unsqueeze(-1), release_lerp_pos, positions)
        orientations = torch.where(release_mask.unsqueeze(-1), release_lerp_quat, orientations)
        
        # Store hand trajectory
        self.hand_trajectory[env_ids, :, :3] = positions
        self.hand_trajectory[env_ids, :, 3:7] = orientations

    def get_hand_window_targets(self) -> torch.Tensor:
        """Get current window of hand pose targets.
        
        Returns:
            (num_envs, window_size, 7) tensor of palm pos(3) + quat(4).
        """
        window_size = self.cfg.trajectory.window_size
        
        # Build indices: (num_envs, window_size)
        batch_idx = torch.arange(self.num_envs, device=self.device).unsqueeze(1)
        window_offset = torch.arange(window_size, device=self.device).unsqueeze(0)
        target_idx = self.current_idx.unsqueeze(1) + window_offset
        
        # Clamp to valid range
        target_idx = torch.clamp(target_idx, max=self.total_targets - 1)
        
        return self.hand_trajectory[batch_idx, target_idx]

    def get_current_hand_target(self) -> torch.Tensor:
        """Get the current (first) hand target. Returns (num_envs, 7)."""
        batch_idx = torch.arange(self.num_envs, device=self.device)
        return self.hand_trajectory[batch_idx, self.current_idx]

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

    def _replan_from_current(self, env_ids: torch.Tensor):
        """Regenerate trajectory from current object pose to goal (push-T mode).
        
        Generates a trajectory from current object position to goal with minimum speed
        enforcement. Uses rolling_window as the base planning horizon, but adjusts it
        if the distance is small to maintain at least min_speed.
        
        This prevents infinite slowdown when very close to goal while ensuring the
        object stays at goal once reached.
        """
        # Update start poses to current actual object positions
        self.start_poses[env_ids] = self.current_object_poses[env_ids]
        
        # Build trajectories per env (need per-env durations for min_speed enforcement)
        for env_id in env_ids:
            env_id = env_id.item()
            
            # Calculate distance to goal
            start_pos = self.start_poses[env_id:env_id+1, :3]  # (1, 3)
            goal_pos = self.goal_poses[env_id:env_id+1, :3]    # (1, 3)
            distance_to_goal = torch.norm(goal_pos - start_pos).item()
            
            # Enforce minimum speed: adjust duration if natural speed would be too slow
            rolling_window = self.cfg.push_t.rolling_window
            min_speed = self.cfg.push_t.min_speed
            natural_speed = distance_to_goal / rolling_window if rolling_window > 0 else float('inf')
            
            if natural_speed < min_speed and distance_to_goal > 1e-4:
                # Too slow - use min_speed to determine duration
                duration = max(distance_to_goal / min_speed, self.target_dt)
            else:
                # Normal speed or already at goal - use full rolling window
                duration = rolling_window
            
            # Generate targets for this duration
            # Start from target_dt ahead (skip the "current position" target)
            num_tgt = min(int(duration / self.target_dt) + 1, self.total_targets)
            
            # Time array starting from target_dt (not 0)
            times = (torch.arange(num_tgt, device=self.device).float() + 1) * self.target_dt
            progress = (times / duration).clamp(0.0, 1.0) if duration > 0 else torch.ones(num_tgt, device=self.device)
            
            # Interpolate positions (lerp)
            positions = start_pos + progress.unsqueeze(-1) * (goal_pos - start_pos)
            
            # Interpolate orientations (slerp)
            start_quat = self.start_poses[env_id:env_id+1, 3:7][0]  # (4,)
            goal_quat = self.goal_poses[env_id:env_id+1, 3:7][0]    # (4,)
            orientations = torch.zeros(num_tgt, 4, device=self.device)
            for t, t_val in enumerate(progress):
                orientations[t] = self._slerp(start_quat, goal_quat, t_val.item())
            
            # Store in trajectory buffer (pad remaining with goal pose)
            self.trajectory[env_id, :num_tgt, :3] = positions
            self.trajectory[env_id, :num_tgt, 3:7] = orientations
            if num_tgt < self.total_targets:
                # Fill rest of trajectory with goal pose (stay at goal)
                self.trajectory[env_id, num_tgt:, :3] = goal_pos
                self.trajectory[env_id, num_tgt:, 3:7] = goal_quat
    
    def _slerp(self, q1: torch.Tensor, q2: torch.Tensor, t: float) -> torch.Tensor:
        """Spherical linear interpolation between two quaternions."""
        dot = (q1 * q2).sum()
        
        # If negative dot, negate one quat to take shorter path
        if dot < 0:
            q2 = -q2
            dot = -dot
        
        dot = torch.clamp(dot, -1.0, 1.0)
        
        # If very close, use linear interpolation
        if dot > 0.9995:
            result = q1 + t * (q2 - q1)
            return result / torch.norm(result)
        
        theta = torch.acos(dot)
        sin_theta = torch.sin(theta)
        
        w1 = torch.sin((1.0 - t) * theta) / sin_theta
        w2 = torch.sin(t * theta) / sin_theta
        
        return w1 * q1 + w2 * q2

    def _compute_progress(self, times: torch.Tensor) -> torch.Tensor:
        """
        Compute progress (0 to 1) with phase-based easing and waypoint pauses.
        
        For push-T mode: Always in manipulate phase (no grasp/release).
        For normal mode:
        - grasp: stationary at start (progress = 0) - object waits while hand approaches
        - manipulate: move segments with pauses at waypoints
        - hand_release: stationary at goal (progress = 1) - object stays while hand retreats
        
        Waypoint pauses: Object pauses briefly at each waypoint before continuing.
        """
        cfg = self.cfg.trajectory
        hand_cfg = self.cfg.hand_trajectory
        wp_cfg = self.cfg.waypoints
        n, T = times.shape
        
        # Compute manipulation duration components
        max_kp = hand_cfg.keypoints.count[1]
        max_wp = wp_cfg.count[1]
        move_duration = cfg.phases.manipulate_base + cfg.phases.manipulate_per_keypoint * max_kp
        pause_duration = wp_cfg.pause_duration
        total_pause_time = pause_duration * max_wp
        manip_duration = move_duration + total_pause_time
        
        # Push-T mode: always manipulate phase, no grasp/release, no pauses
        if self.cfg.push_t.enabled:
            # Map time directly to progress with symmetric easing
            local_t = (times / move_duration).clamp(0.0, 1.0)  # 0 to 1
            
            # Split into first half (ease-in) and second half (ease-out)
            first_half = local_t < 0.5
            second_half = ~first_half
            
            progress = torch.zeros_like(local_t)
            
            # First half: ease-in (0 -> 0.5)
            if first_half.any():
                t_normalized = local_t[first_half] * 2.0  # Map [0, 0.5] -> [0, 1]
                progress[first_half] = (t_normalized ** cfg.easing_power) * 0.5
            
            # Second half: ease-out (0.5 -> 1.0)
            if second_half.any():
                t_normalized = (local_t[second_half] - 0.5) * 2.0  # Map [0.5, 1] -> [0, 1]
                progress[second_half] = 0.5 + (1.0 - (1.0 - t_normalized) ** cfg.easing_power) * 0.5
            
            return progress.clamp(0.0, 1.0)
        
        # Normal mode: phases
        # Phase boundaries  
        t1 = cfg.phases.grasp  # End of grasp phase
        t2 = t1 + manip_duration  # End of manipulate phase (including pauses)
        
        progress = torch.zeros_like(times)
        
        # Grasp phase: progress = 0 (object stationary while hand approaches)
        # Already zeros
        
        # Manipulate phase with waypoint pauses (vectorized)
        mask = (times >= t1) & (times < t2)
        if mask.any():
            manip_t = times[mask] - t1  # Time within manipulation phase
            
            # Structure: move1 → pause1 → move2 → pause2 → ... → moveN (no pause after last)
            num_seg = max_wp + 1
            move_per_seg = move_duration / num_seg
            block_duration = move_per_seg + pause_duration  # Time per block (move + pause)
            
            # Which block are we in?
            seg_idx = (manip_t / block_duration).long().clamp(0, num_seg - 1)
            block_t = manip_t - seg_idx.float() * block_duration  # Time within block
            
            # Are we moving or pausing?
            in_move = block_t < move_per_seg
            
            # Progress during move: interpolate within segment
            move_progress = (seg_idx.float() + block_t / move_per_seg) / num_seg
            # Progress during pause: hold at segment end
            pause_progress = (seg_idx.float() + 1.0) / num_seg
            
            raw_progress = torch.where(in_move, move_progress, pause_progress).clamp(0.0, 1.0)
            
            # Apply symmetric easing
            eased = torch.where(
                raw_progress < 0.5,
                ((raw_progress * 2.0) ** cfg.easing_power) * 0.5,
                0.5 + (1.0 - ((1.0 - raw_progress) * 2.0) ** cfg.easing_power) * 0.5
            )
            
            progress[mask] = eased
        
        # Release phase: progress = 1 (object stationary at goal)
        mask_release = times >= t2
        progress[mask_release] = 1.0
        
        return progress.clamp(0.0, 1.0)

    def _interpolate_positions(self, env_ids: torch.Tensor, progress: torch.Tensor) -> torch.Tensor:
        """
        Interpolate positions using Catmull-Rom spline through control points.
        
        Unlike Bezier, Catmull-Rom splines PASS THROUGH all control points.
        Control points: start -> waypoints -> goal
        """
        n, T = progress.shape
        cfg = self.cfg
        max_wp = cfg.waypoints.count[1]
        
        start = self.start_poses[env_ids, :3]  # (n, 3)
        goal = self.goal_poses[env_ids, :3]    # (n, 3)
        waypoints = self.waypoints[env_ids, :, :3]  # (n, max_wp, 3)
        num_wp = self.num_waypoints[env_ids]   # (n,)
        
        # Build control points: (n, max_wp+2, 3)
        control = torch.zeros(n, max_wp + 2, 3, device=self.device)
        control[:, 0] = start
        control[:, 1:1+max_wp] = waypoints
        control[:, -1] = goal
        
        # For unused waypoints, interpolate between start and goal (vectorized)
        if max_wp > 0:
            t_vals = torch.arange(1, max_wp + 1, device=self.device).float() / (max_wp + 1)  # (max_wp,)
            slot_indices = torch.arange(max_wp, device=self.device)  # (max_wp,)
            unused_mask = num_wp.unsqueeze(1) <= slot_indices.unsqueeze(0)  # (n, max_wp)
            
            t_exp = t_vals.view(1, max_wp, 1)  # (1, max_wp, 1)
            interpolated = (1 - t_exp) * start.unsqueeze(1) + t_exp * goal.unsqueeze(1)  # (n, max_wp, 3)
            control[:, 1:1+max_wp] = torch.where(unused_mask.unsqueeze(-1), interpolated, control[:, 1:1+max_wp])
        
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
        max_wp = cfg.waypoints.count[1]
        
        # Build orientation control points: start -> waypoints -> goal
        quats = torch.zeros(n, max_wp + 2, 4, device=self.device)
        quats[:, 0] = self.start_poses[env_ids, 3:7]
        quats[:, 1:1+max_wp] = self.waypoints[env_ids, :, 3:7]
        quats[:, -1] = self.goal_poses[env_ids, 3:7]
        
        # For unused waypoints, SLERP between start and goal (vectorized)
        num_wp = self.num_waypoints[env_ids]
        if max_wp > 0:
            t_vals = torch.arange(1, max_wp + 1, device=self.device).float() / (max_wp + 1)  # (max_wp,)
            slot_indices = torch.arange(max_wp, device=self.device)  # (max_wp,)
            unused_mask = num_wp.unsqueeze(1) <= slot_indices.unsqueeze(0)  # (n, max_wp)
            
            # SLERP all slots at once: (n, max_wp, 4)
            start_q = quats[:, 0:1, :].expand(n, max_wp, 4)  # (n, max_wp, 4)
            goal_q = quats[:, -1:, :].expand(n, max_wp, 4)   # (n, max_wp, 4)
            t_exp = t_vals.view(1, max_wp).expand(n, max_wp)  # (n, max_wp)
            interpolated_q = self._batched_slerp(start_q, goal_q, t_exp)  # (n, max_wp, 4)
            
            quats[:, 1:1+max_wp] = torch.where(unused_mask.unsqueeze(-1), interpolated_q, quats[:, 1:1+max_wp])
        
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
        # Minimum clearance above table surface (in local frame)
        min_clearance = self.cfg.randomization.reset.z_offset
        
        if self.points_local is None:
            # Fallback: use a conservative estimate based on object half-height
            # Clamp positions to ensure center is at least table + clearance + estimated half-height
            # Use start pose Z as proxy for object half-height above table
            start_height_above_table = self.start_poses[env_ids, 2:3] - self.env_origins[env_ids, 2:3]
            min_z = self.table_height + self.env_origins[env_ids, 2:3].unsqueeze(1) + min_clearance
            # Also ensure we don't go below start height (which should be valid)
            min_z = torch.maximum(min_z, self.start_poses[env_ids, 2:3].unsqueeze(1))
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
        """Get current phase index for each environment (0=grasp, 1=manipulate, 2=release)."""
        cfg = self.cfg.trajectory
        hand_cfg = self.cfg.hand_trajectory
        wp_cfg = self.cfg.waypoints
        t = self.phase_time
        
        # Compute manipulation duration (including waypoint pauses)
        max_kp = hand_cfg.keypoints.count[1]
        max_wp = wp_cfg.count[1]
        move_duration = cfg.phases.manipulate_base + cfg.phases.manipulate_per_keypoint * max_kp
        manip_duration = move_duration + wp_cfg.pause_duration * max_wp
        
        phase = torch.zeros_like(t, dtype=torch.long)
        t1 = cfg.phases.grasp
        t2 = t1 + manip_duration
        
        phase[t >= t1] = 1  # manipulate
        phase[t >= t2] = 2  # release
        
        return phase

    def is_in_release_phase(self) -> torch.Tensor:
        """Check if environments are in release phase."""
        return self.get_phase() == 2
    
    def is_in_grasp_phase(self) -> torch.Tensor:
        """Check if environments are in grasp phase."""
        return self.get_phase() == 0
