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
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

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

        traj = cfg.trajectory
        self.target_dt = 1.0 / traj.target_hz

        # Max duration for buffer allocation
        max_duration = 0.0
        for seg in traj.segments:
            if seg.type == "random_waypoint":
                max_duration += seg.count[1] * (seg.movement_duration + seg.pause_duration)
            else:
                max_duration += seg.duration

        self.total_duration = max_duration
        self.total_targets = int(self.total_duration / self.target_dt) + 1

        # Object trajectory buffer
        self.trajectory = torch.zeros(num_envs, self.total_targets, 7, device=device)
        self.current_idx = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.phase_time = torch.zeros(num_envs, device=device)

        self.start_poses = torch.zeros(num_envs, 7, device=device)
        self.goal_poses = torch.zeros(num_envs, 7, device=device)
        self.env_origins = torch.zeros(num_envs, 3, device=device)
        self.points_local: torch.Tensor | None = None

        # Push-T replanning tracking
        self.last_replan_idx = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.current_object_poses = torch.zeros(num_envs, 7, device=device)

        self.skip_manipulation = torch.zeros(num_envs, dtype=torch.bool, device=device)

        # Phase boundaries (computed at reset)
        self.t_grasp_end = torch.zeros(num_envs, device=device)
        self.t_manip_end = torch.zeros(num_envs, device=device)
        self.t_episode_end = torch.zeros(num_envs, device=device)

        # Hand trajectory
        self.hand_trajectory = torch.zeros(num_envs, self.total_targets, 7, device=device)

        # Grasp pose in object frame
        self.grasp_pose = torch.zeros(num_envs, 7, device=device)

        # Surface-based grasp
        hand_cfg = cfg.hand_trajectory
        self.grasp_surface_point = torch.zeros(num_envs, 3, device=device)
        self.grasp_surface_normal = torch.zeros(num_envs, 3, device=device)
        self.grasp_roll = torch.zeros(num_envs, device=device)
        self.grasp_standoff = torch.zeros(num_envs, device=device)
        self.grasp_surface_idx = torch.zeros(num_envs, dtype=torch.long, device=device)

        max_keypoints = hand_cfg.keypoints.count[1]
        self.keypoint_surface_points = torch.zeros(num_envs, max_keypoints, 3, device=device)
        self.keypoint_surface_normals = torch.zeros(num_envs, max_keypoints, 3, device=device)
        self.keypoint_rolls = torch.zeros(num_envs, max_keypoints, device=device)
        self.num_grasp_keypoints = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.grasp_keypoints = torch.zeros(num_envs, max_keypoints, 7, device=device)

        self.release_pose = torch.zeros(num_envs, 7, device=device)
        self.start_palm_pose = torch.zeros(num_envs, 7, device=device)

        # Debug: surface points in object-local frame
        self.debug_surface_point_local = torch.zeros(num_envs, 3, device=device)
        self.debug_keypoint_points_local = torch.zeros(num_envs, max_keypoints, 3, device=device)

        # Segment-based trajectory
        # Max segments after random_waypoint expansion (each becomes movement + pause)
        max_segments_after_expansion = sum(
            seg.count[1] * 2 if seg.type == "random_waypoint" else 1
            for seg in cfg.trajectory.segments
        )
        self.segment_poses = torch.zeros(num_envs, max_segments_after_expansion, 7, device=device)
        self.segment_boundaries = torch.zeros(num_envs, max_segments_after_expansion + 1, device=device)
        self.coupling_modes = torch.zeros(num_envs, max_segments_after_expansion, dtype=torch.long, device=device)  # 0=full, 1=position_only, 2=none
        self.num_segments = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.hand_pose_at_segment_boundary = torch.zeros(num_envs, max_segments_after_expansion + 1, 7, device=device)

        self.segment_durations = torch.zeros(num_envs, max_segments_after_expansion, device=device)
        self.segment_is_helical = torch.zeros(num_envs, max_segments_after_expansion, dtype=torch.bool, device=device)
        self.segment_is_grasp = torch.zeros(num_envs, max_segments_after_expansion, dtype=torch.bool, device=device)
        self.segment_is_release = torch.zeros(num_envs, max_segments_after_expansion, dtype=torch.bool, device=device)
        self.segment_is_goal = torch.zeros(num_envs, max_segments_after_expansion, dtype=torch.bool, device=device)
        self.segment_is_return = torch.zeros(num_envs, max_segments_after_expansion, dtype=torch.bool, device=device)

        # Helical segment parameters (only used when segment_is_helical=True)
        self.segment_helical_axis = torch.zeros(num_envs, max_segments_after_expansion, 3, device=device)
        self.segment_helical_rotation = torch.zeros(num_envs, max_segments_after_expansion, device=device)
        self.segment_helical_translation = torch.zeros(num_envs, max_segments_after_expansion, 3, device=device)

        # Custom pose override (for waypoint segments with explicit pose in config)
        self.segment_has_custom_pose = torch.zeros(num_envs, max_segments_after_expansion, dtype=torch.bool, device=device)
        self.segment_custom_pose = torch.zeros(num_envs, max_segments_after_expansion, 7, device=device)

        # Hand orientation override (for position_only coupling)
        self.segment_has_hand_orientation = torch.zeros(num_envs, max_segments_after_expansion, dtype=torch.bool, device=device)
        self.segment_hand_orientation = torch.zeros(num_envs, max_segments_after_expansion, 4, device=device)

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

            # Resolve USD path: expand variables, then handle absolute/Nucleus paths
            object_usd_expanded = cfg.push_t.object_usd.format(
                ISAACLAB_NUCLEUS_DIR=ISAACLAB_NUCLEUS_DIR,
                ISAAC_NUCLEUS_DIR=ISAAC_NUCLEUS_DIR,
            )
            if object_usd_expanded.startswith(("omniverse://", "http://", "https://")) or Path(object_usd_expanded).is_absolute():
                usd_path = object_usd_expanded
            else:
                usd_path = str(config_dir / object_usd_expanded)

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

    def _expand_segments(self, env_ids: torch.Tensor, start_poses: torch.Tensor, env_origins: torch.Tensor):
        """FULLY VECTORIZED: Expand segment configs (random_waypoint → N waypoint segments).

        Process ALL envs in parallel - ZERO LOOPS!

        Args:
            env_ids: Environment indices to expand for
            start_poses: Starting object poses (n, 7)
            env_origins: World origin offset for each env (n, 3)
        """
        from isaaclab.utils.math import quat_mul, quat_from_euler_xyz

        n = len(env_ids)

        # Track output segment index per env
        output_seg_idx = torch.zeros(n, dtype=torch.long, device=self.device)

        # Reset segment buffers for these envs to avoid stale data
        self.segment_durations[env_ids] = 0.0
        self.segment_is_helical[env_ids] = False
        self.segment_is_grasp[env_ids] = False
        self.segment_is_release[env_ids] = False
        self.segment_is_goal[env_ids] = False
        self.segment_is_return[env_ids] = False
        self.segment_has_custom_pose[env_ids] = False
        self.segment_custom_pose[env_ids] = 0.0
        self.segment_has_hand_orientation[env_ids] = False
        self.segment_hand_orientation[env_ids] = 0.0
        self.coupling_modes[env_ids] = 0

        # Process each segment config
        for seg_config in self.cfg.trajectory.segments:
            if seg_config.type == "random_waypoint":
                # Each waypoint becomes two segments: movement + pause
                num_wp_per_env = torch.randint(
                    seg_config.count[0],
                    seg_config.count[1] + 1,
                    (n,),
                    device=self.device
                )

                max_wp = num_wp_per_env.max().item()

                if max_wp > 0:
                    ws_cfg = self.cfg.workspace
                    start_local = start_poses[:, :3] - env_origins
                    wp_valid = torch.arange(max_wp, device=self.device).unsqueeze(0) < num_wp_per_env.unsqueeze(1)
                    if seg_config.position_range and seg_config.position_range.x is not None:
                        x_offsets = sample_uniform(
                            seg_config.position_range.x[0],
                            seg_config.position_range.x[1],
                            (n, max_wp),
                            self.device
                        )
                        x_local = start_local[:, 0].unsqueeze(1) + x_offsets
                        x_local = torch.clamp(x_local, min=ws_cfg.x[0], max=ws_cfg.x[1])
                    else:
                        x_local = sample_uniform(ws_cfg.x[0], ws_cfg.x[1], (n, max_wp), self.device)

                    if seg_config.position_range and seg_config.position_range.y is not None:
                        y_offsets = sample_uniform(
                            seg_config.position_range.y[0],
                            seg_config.position_range.y[1],
                            (n, max_wp),
                            self.device
                        )
                        y_local = start_local[:, 1].unsqueeze(1) + y_offsets
                        y_local = torch.clamp(y_local, min=ws_cfg.y[0], max=ws_cfg.y[1])
                    else:
                        y_local = sample_uniform(ws_cfg.y[0], ws_cfg.y[1], (n, max_wp), self.device)

                    if seg_config.position_range and seg_config.position_range.z is not None:
                        z_values = sample_uniform(
                            seg_config.position_range.z[0],
                            seg_config.position_range.z[1],
                            (n, max_wp),
                            self.device
                        )
                        z_local = torch.clamp(z_values, min=ws_cfg.z[0], max=ws_cfg.z[1])
                    else:
                        z_local = sample_uniform(ws_cfg.z[0], ws_cfg.z[1], (n, max_wp), self.device)

                    pos_local = torch.stack([x_local, y_local, z_local], dim=2)  # (n, max_wp, 3)

                    # Sample orientations for ALL at once (n, max_wp, 4)
                    if seg_config.vary_orientation:
                        euler_all = sample_uniform(
                            -seg_config.max_rotation,
                            seg_config.max_rotation,
                            (n, max_wp, 3),
                            self.device
                        )  # (n, max_wp, 3)

                        # Batched euler to quaternion
                        delta_quats = quat_from_euler_xyz(
                            euler_all[..., 0].reshape(-1),
                            euler_all[..., 1].reshape(-1),
                            euler_all[..., 2].reshape(-1)
                        ).reshape(n, max_wp, 4)  # (n, max_wp, 4)

                        # Apply to start orientation
                        start_quat_expanded = start_poses[:, 3:7].unsqueeze(1).expand(-1, max_wp, -1)  # (n, max_wp, 4)
                        quat_all = quat_mul(delta_quats.reshape(n * max_wp, 4), start_quat_expanded.reshape(n * max_wp, 4)).reshape(n, max_wp, 4)
                    else:
                        quat_all = start_poses[:, 3:7].unsqueeze(1).expand(-1, max_wp, -1)  # (n, max_wp, 4)

                    # Write to segment buffers (vectorized scatter)
                    # Each waypoint becomes TWO segments: movement + pause
                    env_idx_flat, wp_idx_flat = torch.where(wp_valid)

                    # Get global env IDs
                    global_env_idx = env_ids[env_idx_flat]

                    # Coupling mode (same for both movement and pause)
                    coupling_map = {"full": 0, "position_only": 1, "none": 2}
                    coupling_value = coupling_map[seg_config.hand_coupling]

                    # ===== MOVEMENT SEGMENT (interpolates to waypoint) =====
                    # Segment index: base + 2*wp_idx (even indices are movement)
                    move_seg_idx = output_seg_idx[env_idx_flat] + wp_idx_flat * 2

                    self.segment_durations[global_env_idx, move_seg_idx] = seg_config.movement_duration
                    self.coupling_modes[global_env_idx, move_seg_idx] = coupling_value
                    self.segment_is_helical[global_env_idx, move_seg_idx] = False
                    self.segment_is_grasp[global_env_idx, move_seg_idx] = False
                    self.segment_is_release[global_env_idx, move_seg_idx] = False
                    self.segment_is_goal[global_env_idx, move_seg_idx] = False
                    self.segment_is_return[global_env_idx, move_seg_idx] = False
                    self.segment_has_custom_pose[global_env_idx, move_seg_idx] = True
                    # IMPORTANT: segment_custom_pose positions are stored in ENV-LOCAL frame.
                    self.segment_custom_pose[global_env_idx, move_seg_idx, :3] = pos_local[env_idx_flat, wp_idx_flat]
                    self.segment_custom_pose[global_env_idx, move_seg_idx, 3:7] = quat_all[env_idx_flat, wp_idx_flat]

                    # ===== PAUSE SEGMENT (holds at waypoint) =====
                    # Segment index: base + 2*wp_idx + 1 (odd indices are pause)
                    pause_seg_idx = move_seg_idx + 1

                    self.segment_durations[global_env_idx, pause_seg_idx] = seg_config.pause_duration
                    self.coupling_modes[global_env_idx, pause_seg_idx] = coupling_value
                    self.segment_is_helical[global_env_idx, pause_seg_idx] = False
                    self.segment_is_grasp[global_env_idx, pause_seg_idx] = False
                    self.segment_is_release[global_env_idx, pause_seg_idx] = False
                    self.segment_is_goal[global_env_idx, pause_seg_idx] = False
                    self.segment_is_return[global_env_idx, pause_seg_idx] = False
                    self.segment_has_custom_pose[global_env_idx, pause_seg_idx] = True
                    # Same pose as movement segment end (object holds at waypoint)
                    self.segment_custom_pose[global_env_idx, pause_seg_idx, :3] = pos_local[env_idx_flat, wp_idx_flat]
                    self.segment_custom_pose[global_env_idx, pause_seg_idx, 3:7] = quat_all[env_idx_flat, wp_idx_flat]

                    # Advance output index: 2 segments per waypoint
                    output_seg_idx += num_wp_per_env * 2

            else:
                # ===== REGULAR SEGMENTS (waypoint, helical) - ALL ENVS AT ONCE =====
                target_seg_idx = output_seg_idx  # (n,) - current segment index per env
                global_env_idx = env_ids  # (n,)

                # Duration
                self.segment_durations[global_env_idx, target_seg_idx] = seg_config.duration

                # Coupling mode
                coupling_map = {"full": 0, "position_only": 1, "none": 2}
                self.coupling_modes[global_env_idx, target_seg_idx] = coupling_map[seg_config.hand_coupling]

                # Segment type markers
                name_lower = seg_config.name.lower()
                self.segment_is_grasp[global_env_idx, target_seg_idx] = "grasp" in name_lower
                self.segment_is_release[global_env_idx, target_seg_idx] = "release" in name_lower
                self.segment_is_goal[global_env_idx, target_seg_idx] = ("goal" in name_lower) or ("place" in name_lower)
                self.segment_is_return[global_env_idx, target_seg_idx] = "return" in name_lower
                self.segment_is_helical[global_env_idx, target_seg_idx] = (seg_config.type == "helical")

                # Type-specific fields
                if seg_config.type == "helical":
                    self.segment_helical_axis[global_env_idx, target_seg_idx] = torch.tensor(seg_config.axis, dtype=torch.float32, device=self.device).unsqueeze(0).expand(n, 3)
                    self.segment_helical_rotation[global_env_idx, target_seg_idx] = seg_config.rotation
                    self.segment_helical_translation[global_env_idx, target_seg_idx] = torch.tensor(seg_config.translation, dtype=torch.float32, device=self.device).unsqueeze(0).expand(n, 3)
                elif seg_config.type == "waypoint":
                    if seg_config.pose is not None:
                        self.segment_has_custom_pose[global_env_idx, target_seg_idx] = True
                        self.segment_custom_pose[global_env_idx, target_seg_idx] = torch.tensor(seg_config.pose, dtype=torch.float32, device=self.device).unsqueeze(0).expand(n, 7)

                # Hand orientation override
                if hasattr(seg_config, 'hand_orientation') and seg_config.hand_orientation is not None:
                    self.segment_has_hand_orientation[global_env_idx, target_seg_idx] = True
                    self.segment_hand_orientation[global_env_idx, target_seg_idx] = torch.tensor(seg_config.hand_orientation, dtype=torch.float32, device=self.device).unsqueeze(0).expand(n, 4)

                # Advance output index
                output_seg_idx += 1

        # Store num_segments for each env
        self.num_segments[env_ids] = output_seg_idx


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

        # Expand segments (random_waypoint → N waypoints)
        self._expand_segments(env_ids, start_poses, env_origins)
        
        # Reset timing
        self.phase_time[env_ids] = 0.0
        self.current_idx[env_ids] = 0
        
        # Sample skip_manipulation per env (grasp-only episodes)
        skip_prob = self.cfg.trajectory.skip_manipulation_probability
        if skip_prob > 0:
            self.skip_manipulation[env_ids] = torch.rand(n, device=self.device) < skip_prob
        else:
            self.skip_manipulation[env_ids] = False
        
        # Store start poses and environment origins
        self.start_poses[env_ids] = start_poses
        self.env_origins[env_ids] = env_origins

        # Sample goal (use start_poses Z for stable placement)
        self._sample_goal(env_ids, env_origins, start_poses)

        # For skip envs: goal = start (object stays in place)
        skip_mask = self.skip_manipulation[env_ids]
        if skip_mask.any():
            self.goal_poses[env_ids[skip_mask]] = self.start_poses[env_ids[skip_mask]]

        # FULLY VECTORIZED: Compute phase boundaries for ALL envs in parallel
        max_segs = self.segment_durations.shape[1]

        # Create valid segment mask: (n, max_segs)
        seg_indices = torch.arange(max_segs, device=self.device).unsqueeze(0).expand(n, max_segs)
        valid_mask = seg_indices < self.num_segments[env_ids].unsqueeze(1)

        # For skip envs: zero out manipulation segment durations (grasp → release directly)
        # This matches old behavior where skip envs had zero manipulation time
        if skip_mask.any():
            skip_env_ids = env_ids[skip_mask]
            is_manip_skip = ~self.segment_is_grasp[skip_env_ids] & ~self.segment_is_release[skip_env_ids]  # (num_skip, max_segs)
            self.segment_durations[skip_env_ids] = torch.where(
                is_manip_skip,
                torch.full_like(self.segment_durations[skip_env_ids], 0.01),  # Small value to avoid div-by-zero
                self.segment_durations[skip_env_ids]
            )
            # BUG FIX: Also override custom poses to start pose so object doesn't jerk to random waypoints
            # For skip envs, manipulation segments should keep object at start position.
            # NOTE: segment_custom_pose positions are stored in ENV-LOCAL frame.
            start_pose_local = self.start_poses[skip_env_ids].clone()
            start_pose_local[:, :3] -= self.env_origins[skip_env_ids]
            start_pose_expanded = start_pose_local.unsqueeze(1).expand(-1, max_segs, -1)  # (num_skip, max_segs, 7)
            is_manip_skip_3d = is_manip_skip.unsqueeze(2).expand(-1, -1, 7)  # (num_skip, max_segs, 7)
            self.segment_custom_pose[skip_env_ids] = torch.where(
                is_manip_skip_3d,
                start_pose_expanded,
                self.segment_custom_pose[skip_env_ids]
            )
            # Mark these as having custom pose so they use start_poses instead of computed poses
            self.segment_has_custom_pose[skip_env_ids] = torch.where(
                is_manip_skip,
                torch.ones_like(is_manip_skip),
                self.segment_has_custom_pose[skip_env_ids]
            )

        # Compute cumulative segment boundaries via masked cumsum
        self.segment_boundaries[env_ids, 0] = 0.0
        durations = self.segment_durations[env_ids]  # (n, max_segs)
        durations_masked = durations * valid_mask.float()
        cumsum_durations = torch.cumsum(durations_masked, dim=1)  # (n, max_segs)
        self.segment_boundaries[env_ids, 1:max_segs+1] = cumsum_durations

        # Get segment type masks
        is_grasp = self.segment_is_grasp[env_ids] & valid_mask  # (n, max_segs)
        is_release = self.segment_is_release[env_ids] & valid_mask

        # Find LAST occurrence of each segment type using index masking
        batch_range = torch.arange(n, device=self.device)

        # t_grasp_end: boundary after last grasp segment
        grasp_indices = torch.where(is_grasp, seg_indices, -1)
        last_grasp_idx = grasp_indices.max(dim=1).values  # (n,)
        grasp_boundary_idx = (last_grasp_idx + 1).clamp(min=0, max=max_segs)
        self.t_grasp_end[env_ids] = self.segment_boundaries[env_ids[batch_range], grasp_boundary_idx]
        self.t_grasp_end[env_ids[last_grasp_idx == -1]] = 0.0  # No grasp segment

        # t_manip_end: boundary after last manipulation segment (not grasp, not release)
        manip_mask = valid_mask & ~is_grasp & ~is_release
        manip_indices = torch.where(manip_mask, seg_indices, -1)
        last_manip_idx = manip_indices.max(dim=1).values
        manip_boundary_idx = (last_manip_idx + 1).clamp(min=0, max=max_segs)
        self.t_manip_end[env_ids] = self.segment_boundaries[env_ids[batch_range], manip_boundary_idx]
        no_manip = (last_manip_idx == -1)
        self.t_manip_end[env_ids[no_manip]] = self.t_grasp_end[env_ids[no_manip]]

        # t_episode_end: total duration (boundary after last valid segment)
        self.t_episode_end[env_ids] = self.segment_boundaries[env_ids[batch_range], self.num_segments[env_ids]]
        
        # === Hand trajectory generation (if enabled) ===
        if self.cfg.hand_trajectory.enabled:
            # Store starting palm pose (required when hand trajectory is enabled)
            self.start_palm_pose[env_ids] = start_palm_poses
            
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

            # Sample release pose
            self._sample_release_pose(env_ids, env_origins)

        # Generate entire object trajectory (same for all modes - uses phases)
        # Push_t will replan during manipulation phase in step()
        # NOTE: Must be before _sample_grasp_keypoints since keypoints need object poses at future times
        self._generate_full_trajectory(env_ids)
        # Hold final object target after episode end (matches old phase-based behavior)
        end_idx = (self.t_episode_end[env_ids] / self.target_dt).long().clamp(0, self.total_targets - 1)
        time_idx = torch.arange(self.total_targets, device=self.device).unsqueeze(0)
        hold_mask = time_idx >= end_idx.unsqueeze(1)  # (n, T)
        end_pose = self.trajectory[env_ids, end_idx]  # (n, 7)
        self.trajectory[env_ids] = torch.where(
            hold_mask.unsqueeze(2),
            end_pose.unsqueeze(1),
            self.trajectory[env_ids],
        )

        # Sample grasp keypoints (perturbed from grasp pose)
        # Must be AFTER trajectory generation so we can look up object poses at keypoint times
        if self.cfg.hand_trajectory.enabled:
            self._sample_grasp_keypoints(env_ids)
        
        # For push_t: initialize tracking for manipulation phase replanning
        if self.cfg.push_t.enabled:
            self.current_object_poses[env_ids] = start_poses
            self.last_replan_idx[env_ids] = 0
        
        # Generate hand trajectory (after object trajectory, since it depends on it)
        if self.cfg.hand_trajectory.enabled:
            self._generate_hand_trajectory_from_segments(env_ids)
            # Hold final hand target after episode end (keep release pose)
            end_hand = self.hand_trajectory[env_ids, end_idx]  # (n, 7)
            self.hand_trajectory[env_ids] = torch.where(
                hold_mask.unsqueeze(2),
                end_hand.unsqueeze(1),
                self.hand_trajectory[env_ids],
            )

    def step(self, dt: float, object_poses: torch.Tensor | None = None):
        """
        Advance trajectory by one timestep.
        
        For push-T mode: replans manipulation portion of trajectory from current object 
        position every window_size timesteps, but ONLY during manipulation phase.
        Grasp and release phases use fixed trajectories (no replanning).
        
        Args:
            dt: Simulation timestep.
            object_poses: Current object poses (num_envs, 7) for replanning. Only used in push-T mode.
        """
        self.phase_time += dt
        
        # Compute current target index from time
        new_idx = (self.phase_time / self.target_dt).long()
        
        # Clamp to trajectory length (window access clamps per-target)
        self.current_idx = torch.clamp(new_idx, max=self.total_targets - 1)
        
        # Push-T replanning: ONLY during manipulation phase
        if self.cfg.push_t.enabled and object_poses is not None:
            self.current_object_poses = object_poses
            replan_interval = self.cfg.trajectory.window_size
            
            # Only replan if in manipulation phase (phase == 1)
            in_manipulation = self.get_phase() == 1
            steps_since_replan = self.current_idx - self.last_replan_idx
            needs_replan = in_manipulation & (steps_since_replan >= replan_interval)
            
            if needs_replan.any():
                env_ids = torch.where(needs_replan)[0]
                self._replan_manipulation_phase(env_ids)
                self.last_replan_idx[env_ids] = self.current_idx[env_ids]
                
                # Regenerate hand trajectory for manipulation phase
                if self.cfg.hand_trajectory.enabled:
                    self._update_hand_trajectory_manipulation(env_ids)

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

    def _sample_goal(self, env_ids: torch.Tensor, env_origins: torch.Tensor, start_poses: torch.Tensor):
        """Sample goal pose (on table).
        
        Uses stable poses from trimesh for goal placement (z-position + orientation).
        Different stable poses have different z-positions (e.g., cube on edge vs flat).
        If return_to_start is True, goal = start position (for return tasks like cup pouring).
        """
        n = len(env_ids)
        cfg = self.cfg
        goal_cfg = cfg.goal
        ws_cfg = cfg.workspace
        
        # Return to start position if configured (e.g., cup pouring returns bottle)
        if goal_cfg.return_to_start:
            self.goal_poses[env_ids] = start_poses.clone()
            return
        
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
                goal_quat = torch.tensor([cfg.push_t.goal_rotation], dtype=torch.float32, device=self.device).repeat(n, 1)
            else:
                goal_quat = torch.tensor([cfg.push_t.outline_rotation], dtype=torch.float32, device=self.device).repeat(n, 1)
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
        
        Two modes:
        1. fixed_direction specified: Find surface point in that direction from 
           (object_center + fixed_origin_offset), use its surface normal.
        2. fixed_direction is None: Randomly sample surface point (excluding bottom).
        
        In both cases, grasp pose is derived from the surface normal:
        - grasp_pos = surface_point + standoff * surface_normal
        - grasp_quat = palm Z points toward object (opposite of surface normal)
        
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
        
        obj_pos = object_poses[:, :3]  # (n, 3)
        obj_quat = object_poses[:, 3:7]  # (n, 4)
        
        # Rotate normals to world frame: (n, pool_size, 3)
        env_normals_flat = env_normals.reshape(-1, 3)  # (n*pool_size, 3)
        obj_quat_exp = obj_quat.unsqueeze(1).expand(n, 256, 4).reshape(-1, 4)  # (n*pool_size, 4)
        env_normals_w = quat_apply(obj_quat_exp, env_normals_flat).reshape(n, 256, 3)
        
        # Select surface point based on mode
        if cfg.fixed_direction is not None:
            # Fixed direction mode: find surface point in specified direction
            
            # Compute grasp origin in object-local frame (object center + offset)
            if cfg.fixed_origin_offset is not None:
                origin_offset = torch.tensor(cfg.fixed_origin_offset, dtype=torch.float32, device=self.device)
            else:
                origin_offset = torch.zeros(3, device=self.device)
            
            # Direction in object-local frame (normalized)
            direction = torch.tensor(cfg.fixed_direction, dtype=torch.float32, device=self.device)
            direction = direction / (direction.norm() + 1e-8)
            
            # Find surface point with maximum projection onto direction from origin
            # For each point: score = (point - origin) · direction
            # Higher score = further along the direction = on the surface facing that way
            points_from_origin = env_points - origin_offset.unsqueeze(0).unsqueeze(0)  # (n, pool_size, 3)
            projections = (points_from_origin * direction.unsqueeze(0).unsqueeze(0)).sum(dim=-1)  # (n, pool_size)
            
            # Select point with maximum projection (furthest in the direction)
            selected_idx = projections.argmax(dim=1)  # (n,)
        else:
            # Random sampling mode: filter bottom hemisphere and randomly select
            
            # Filter out bottom hemisphere in WORLD frame
            exclude_z = -1.0 + cfg.exclude_bottom_fraction * 2  # e.g., 0.3 -> -0.4
            valid_mask = env_normals_w[:, :, 2] > exclude_z  # (n, pool_size)
            
            # Randomly select one valid point per env
            rand_vals = torch.rand(n, 256, device=self.device)
            rand_vals[~valid_mask] = -1.0  # Invalid points won't be selected
            selected_idx = rand_vals.argmax(dim=1)  # (n,)
        
        batch_idx = torch.arange(n, device=self.device)
        surface_points = env_points[batch_idx, selected_idx]  # (n, 3) in object-local frame
        surface_normals = env_normals[batch_idx, selected_idx]  # (n, 3) in object-local frame
        
        # Fix potentially inverted normals: if we selected a point in +Y direction,
        # the outward normal should also point roughly +Y. If it points opposite, flip it.
        if cfg.fixed_direction is not None:
            # Check alignment: dot(normal, direction) should be positive for outward normal
            alignment = (surface_normals * direction.unsqueeze(0)).sum(dim=-1)  # (n,)
            flip_mask = alignment < 0  # Normal points wrong way
            surface_normals[flip_mask] = -surface_normals[flip_mask]
        
        # Apply origin offset to surface points (in object-local frame)
        if cfg.fixed_origin_offset is not None:
            surface_points = surface_points + origin_offset.unsqueeze(0)

        # Rotate surface point and normal to world frame
        surface_points_w = quat_apply(obj_quat, surface_points) + obj_pos
        surface_normals_w = quat_apply(obj_quat, surface_normals)
        surface_normals_w = surface_normals_w / (surface_normals_w.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Store surface point in LOCAL frame for debug visualization (moves with object)
        self.debug_surface_point_local[env_ids] = surface_points
        
        # Sample standoff distance
        standoff = sample_uniform(
            cfg.standoff_range[0], cfg.standoff_range[1], (n,), self.device
        )
        
        # Palm position: surface point + standoff along normal
        grasp_pos = surface_points_w + standoff.unsqueeze(-1) * surface_normals_w
        
        # Palm orientation: Z axis points toward object (opposite of normal)
        grasp_quat = self._quat_from_z_axis(-surface_normals_w)

        # Determine desired roll (either fixed or zero)
        desired_roll = torch.full((n,), cfg.fixed_hand_roll, device=self.device) if cfg.fixed_hand_roll is not None else torch.zeros(n, device=self.device)

        # Compute optimal roll to satisfy all finger direction constraints simultaneously
        constraints = []
        if cfg.exclude_toward_robot:
            constraints.append((0, cfg.toward_robot_threshold))  # x-component
        if cfg.exclude_upward:
            constraints.append((2, cfg.upward_threshold))  # z-component

        if len(constraints) > 0:
            roll = self._compute_feasible_roll_multi_constraint(
                base_quat=grasp_quat,
                desired_roll=desired_roll,
                constraints=constraints,
            )
        else:
            roll = desired_roll

        # Apply the final roll to grasp quaternion
        roll_quat = quat_from_euler_xyz(
            torch.zeros(n, device=self.device),
            torch.zeros(n, device=self.device),
            roll,
        )
        grasp_quat = quat_mul(grasp_quat, roll_quat)

        # Store surface-based grasp representation for keypoint sampling
        self.grasp_surface_point[env_ids] = surface_points  # Local frame
        self.grasp_surface_normal[env_ids] = surface_normals  # Local frame
        self.grasp_roll[env_ids] = roll
        self.grasp_standoff[env_ids] = standoff
        self.grasp_surface_idx[env_ids] = selected_idx

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

    def _quat_from_z_and_roll(self, z_target: torch.Tensor, roll: torch.Tensor) -> torch.Tensor:
        """Construct quaternion where Z axis points along z_target with given roll.

        Args:
            z_target: (n, 3) or (n, k, 3) target Z axis direction.
            roll: (n,) or (n, k) roll angle around Z axis in radians.

        Returns:
            (n, 4) or (n, k, 4) quaternion (wxyz format).
        """
        original_shape = z_target.shape[:-1]
        z_target_flat = z_target.reshape(-1, 3)
        roll_flat = roll.reshape(-1)

        # Get base quaternion (no roll)
        base_quat = self._quat_from_z_axis(z_target_flat)

        # Apply roll around Z axis
        n = roll_flat.shape[0]
        zeros = torch.zeros(n, device=self.device)
        roll_quat = quat_from_euler_xyz(zeros, zeros, roll_flat)
        result = quat_mul(base_quat, roll_quat)

        return result.reshape(*original_shape, 4)

    def _compute_closest_feasible_roll(
        self,
        base_quat: torch.Tensor,  # (n, 4) base quaternion at roll=0
        desired_roll: torch.Tensor,  # (n,) desired roll value
        threshold: float,
        component_idx: int = 0,  # 0=x, 1=y, 2=z
    ) -> torch.Tensor:
        """Find roll closest to desired_roll that satisfies finger[component] < threshold.

        As roll varies around the palm Z axis, finger direction rotates in 3D:
            finger[i](roll) = A * cos(roll) + B * sin(roll) = R * cos(roll - φ)
        where:
            A = base_X[i] (finger[i] at roll=0)
            B = base_Y[i] (finger[i] at roll=π/2)
            R = sqrt(A² + B²)
            φ = atan2(B, A)

        The constraint finger[component] < threshold defines a feasible region.
        This function returns the roll closest to desired_roll within that region.

        Args:
            base_quat: Base quaternion with roll=0 (from _quat_from_z_axis).
            desired_roll: The roll we'd like to use if feasible.
            threshold: Maximum allowed finger[component] value.
            component_idx: Which component to constrain (0=x, 1=y, 2=z).

        Returns:
            Roll values that satisfy the constraint, as close to desired_roll as possible.
        """
        n = base_quat.shape[0]
        device = base_quat.device

        # Get X and Y axes of palm frame in world coordinates
        x_axis = torch.tensor([1.0, 0.0, 0.0], device=device)
        y_axis = torch.tensor([0.0, 1.0, 0.0], device=device)
        base_X = quat_apply(base_quat, x_axis.expand(n, 3))  # (n, 3)
        base_Y = quat_apply(base_quat, y_axis.expand(n, 3))  # (n, 3)

        # Coefficients for finger[component] = A*cos(roll) + B*sin(roll)
        A = base_X[:, component_idx]
        B = base_Y[:, component_idx]

        # Convert to R*cos(roll - φ) form
        R = torch.sqrt(A**2 + B**2)
        phi = torch.atan2(B, A)

        # Handle edge cases:
        # - If R < threshold: all rolls feasible (finger.x max is R)
        # - If R < -threshold: no rolls feasible (finger.x min is -R)
        # For R < -threshold, we return desired_roll (nothing we can do)
        always_feasible = R <= threshold
        never_feasible = -R >= threshold  # i.e., threshold <= -R

        # Compute feasible region boundaries
        # Constraint: R * cos(roll - φ) < threshold
        # => cos(roll - φ) < threshold / R
        # Feasible when |roll - φ| > arccos(threshold / R)
        ratio = (threshold / R).clamp(-1, 1)
        half_width = torch.acos(ratio)

        # Boundaries in absolute roll coordinates
        boundary_lo = phi - half_width  # Lower boundary
        boundary_hi = phi + half_width  # Upper boundary

        # Check if desired_roll is feasible
        # Normalize desired_roll relative to phi, wrap to [-π, π]
        rel_desired = desired_roll - phi
        rel_desired = (rel_desired + math.pi) % (2 * math.pi) - math.pi

        # Desired is infeasible if |rel_desired| < half_width
        desired_infeasible = torch.abs(rel_desired) < half_width

        # Find closest boundary
        # If rel_desired < 0, closer to lower boundary (-half_width)
        # If rel_desired >= 0, closer to upper boundary (+half_width)
        epsilon = 0.01  # Small margin to be strictly inside feasible region
        closest_boundary = torch.where(
            rel_desired < 0,
            phi - half_width - epsilon,  # Just below lower boundary
            phi + half_width + epsilon,  # Just above upper boundary
        )

        # Final result:
        # - If always/never feasible: use desired_roll
        # - If desired is feasible: use desired_roll
        # - If desired is infeasible: use closest_boundary
        result = torch.where(
            always_feasible | never_feasible | ~desired_infeasible,
            desired_roll,
            closest_boundary,
        )

        return result

    def _compute_feasible_roll_multi_constraint(
        self,
        base_quat: torch.Tensor,  # (n, 4) base quaternion at roll=0
        desired_roll: torch.Tensor,  # (n,) desired roll value
        constraints: list[tuple[int, float]],  # [(component_idx, threshold), ...]
    ) -> torch.Tensor:
        """Find roll satisfying ALL constraints, or minimize worst violation if impossible.

        For each constraint (component_idx, threshold), computes the feasible region
        where finger[component] < threshold. Then finds the intersection of all regions.

        If intersection exists: Returns roll from intersection closest to desired_roll.
        If intersection is empty: Returns roll that minimizes max(violations).

        Args:
            base_quat: Base quaternion with roll=0.
            desired_roll: The roll we'd like to use if feasible.
            constraints: List of (component_idx, threshold) pairs to satisfy.

        Returns:
            Roll values that best satisfy all constraints.
        """
        if len(constraints) == 0:
            return desired_roll

        n = base_quat.shape[0]
        device = base_quat.device

        # Get X and Y axes of palm frame in world coordinates
        x_axis = torch.tensor([1.0, 0.0, 0.0], device=device)
        y_axis = torch.tensor([0.0, 1.0, 0.0], device=device)
        base_X = quat_apply(base_quat, x_axis.expand(n, 3))  # (n, 3)
        base_Y = quat_apply(base_quat, y_axis.expand(n, 3))  # (n, 3)

        # For each constraint, compute feasible region
        # Each region is represented as a complement of an arc: feasible when |roll - phi| > half_width
        feasible_regions = []  # List of (phi, half_width, always_feasible, never_feasible)

        for component_idx, threshold in constraints:
            # Coefficients for finger[component] = A*cos(roll) + B*sin(roll)
            A = base_X[:, component_idx]
            B = base_Y[:, component_idx]

            # Convert to R*cos(roll - φ) form
            R = torch.sqrt(A**2 + B**2)
            phi = torch.atan2(B, A)

            # Edge cases
            always_feasible = R <= threshold
            never_feasible = -R >= threshold

            # Compute feasible region boundaries
            # Constraint: R * cos(roll - φ) < threshold => |roll - φ| > arccos(threshold / R)
            ratio = (threshold / R).clamp(-1, 1)
            half_width = torch.acos(ratio)

            feasible_regions.append((phi, half_width, always_feasible, never_feasible))

        # Find intersection of all feasible regions
        # Strategy: Sample N candidate rolls uniformly around circle, check which satisfy ALL constraints
        num_samples = 32
        candidate_rolls = torch.linspace(-math.pi, math.pi, num_samples, device=device).unsqueeze(0).expand(n, num_samples)  # (n, num_samples)

        # Evaluate all constraints for all candidates
        all_feasible = torch.ones(n, num_samples, dtype=torch.bool, device=device)

        for (phi, half_width, always_feas, never_feas) in feasible_regions:
            # Normalize candidate_rolls relative to phi
            rel_rolls = candidate_rolls - phi.unsqueeze(1)
            rel_rolls = (rel_rolls + math.pi) % (2 * math.pi) - math.pi

            # Feasible if |rel_rolls| > half_width (outside the infeasible arc)
            is_feasible = torch.abs(rel_rolls) > half_width.unsqueeze(1)

            # Handle edge cases
            is_feasible = torch.where(always_feas.unsqueeze(1), torch.ones_like(is_feasible), is_feasible)
            is_feasible = torch.where(never_feas.unsqueeze(1), torch.zeros_like(is_feasible), is_feasible)

            all_feasible = all_feasible & is_feasible

        # Check if any candidates are fully feasible
        any_feasible = all_feasible.any(dim=1)  # (n,)

        # Case 1: Intersection exists - pick closest to desired_roll
        if any_feasible.any():
            # Compute distance from desired_roll to each candidate (wrap-around distance)
            dist_to_desired = candidate_rolls - desired_roll.unsqueeze(1)
            dist_to_desired = (dist_to_desired + math.pi) % (2 * math.pi) - math.pi
            dist_to_desired = torch.abs(dist_to_desired)

            # Mask out infeasible candidates
            dist_to_desired[~all_feasible] = float('inf')

            # Pick closest feasible candidate
            best_idx = dist_to_desired.argmin(dim=1)
            result_feasible = candidate_rolls[torch.arange(n, device=device), best_idx]
        else:
            result_feasible = desired_roll  # Placeholder, will be overwritten

        # Case 2: No intersection - minimize worst violation
        # Compute violation for each constraint at each candidate roll
        max_violations = torch.zeros(n, num_samples, device=device)

        for component_idx, threshold in constraints:
            A = base_X[:, component_idx].unsqueeze(1)  # (n, 1)
            B = base_Y[:, component_idx].unsqueeze(1)  # (n, 1)

            # Compute finger[component] = A*cos(roll) + B*sin(roll)
            finger_component = A * torch.cos(candidate_rolls) + B * torch.sin(candidate_rolls)

            # Violation = max(0, finger_component - threshold)
            violation = torch.relu(finger_component - threshold)

            # Track maximum violation across constraints
            max_violations = torch.maximum(max_violations, violation)

        # Pick candidate with smallest maximum violation
        best_idx_fallback = max_violations.argmin(dim=1)
        result_fallback = candidate_rolls[torch.arange(n, device=device), best_idx_fallback]

        # Final result: use feasible solution if exists, otherwise use fallback
        result = torch.where(any_feasible, result_feasible, result_fallback)

        return result

    def _get_object_pose_at_keypoint_time(self, env_ids: torch.Tensor, kp_idx: int) -> torch.Tensor:
        """Get object pose at the time when keypoint kp_idx is reached.

        Keypoints are evenly distributed through the manipulation phase.

        Args:
            env_ids: Environment indices.
            kp_idx: Keypoint index (0-based).

        Returns:
            (n, 7) object poses at keypoint time.
        """
        n = len(env_ids)
        cfg = self.cfg
        max_kp = cfg.hand_trajectory.keypoints.count[1]

        # Keypoints are distributed through manipulation phase
        # kp_idx=0 is near start of manipulation, kp_idx=max_kp-1 is near end

        # Get per-env manipulation duration
        grasp_end = self.t_grasp_end[env_ids]  # (n,)
        manip_duration = self.t_manip_end[env_ids] - grasp_end  # (n,)

        # Time within manipulation phase for this keypoint
        # For forward-only traversal with N segments: grasp → kp_0 → ... → kp_{N-1}
        # kp_k is reached at progress (k+1)/N, where N = max_kp
        # Final keypoint (kp_{N-1}) is at progress 1.0 (goal pose)
        kp_fraction = (kp_idx + 1) / max_kp
        kp_time = grasp_end + kp_fraction * manip_duration  # (n,)

        # Convert to trajectory index
        kp_traj_idx = (kp_time / self.target_dt).long()  # (n,)
        kp_traj_idx = kp_traj_idx.clamp(max=self.total_targets - 1)

        # Gather object poses from trajectory
        batch_idx = torch.arange(n, device=self.device)
        return self.trajectory[env_ids][batch_idx, kp_traj_idx]  # (n, 7)

    def _compute_palm_world_batch(
        self,
        surface_points: torch.Tensor,
        surface_normals: torch.Tensor,
        rolls: torch.Tensor,
        standoffs: torch.Tensor,
        obj_poses: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute world-frame palm poses from surface-based representation.

        Args:
            surface_points: (n, 3) or (n, k, 3) surface points in object-local frame.
            surface_normals: (n, 3) or (n, k, 3) surface normals in object-local frame.
            rolls: (n,) or (n, k) roll angles around approach axis.
            standoffs: (n,) standoff distances.
            obj_poses: (n, 7) object poses (pos + quat).

        Returns:
            palm_pos_world: (n, 3) or (n, k, 3) palm positions in world frame.
            palm_quat_world: (n, 4) or (n, k, 4) palm quaternions in world frame.
        """
        has_k_dim = surface_points.dim() == 3
        if has_k_dim:
            n, k, _ = surface_points.shape
            # Expand standoffs and obj_poses for k dimension
            standoffs_exp = standoffs.unsqueeze(1).expand(n, k)
            obj_pos = obj_poses[:, :3].unsqueeze(1).expand(n, k, 3)
            obj_quat = obj_poses[:, 3:7].unsqueeze(1).expand(n, k, 4)
        else:
            n = surface_points.shape[0]
            standoffs_exp = standoffs
            obj_pos = obj_poses[:, :3]
            obj_quat = obj_poses[:, 3:7]

        # Palm position in local frame: point + standoff * normal
        palm_pos_local = surface_points + standoffs_exp.unsqueeze(-1) * surface_normals

        # Palm orientation in local frame: Z axis points opposite to normal (toward object)
        palm_quat_local = self._quat_from_z_and_roll(-surface_normals, rolls)

        # Transform to world frame
        if has_k_dim:
            # Flatten for quat operations
            palm_pos_local_flat = palm_pos_local.reshape(-1, 3)
            palm_quat_local_flat = palm_quat_local.reshape(-1, 4)
            obj_pos_flat = obj_pos.reshape(-1, 3)
            obj_quat_flat = obj_quat.reshape(-1, 4)

            palm_pos_world = quat_apply(obj_quat_flat, palm_pos_local_flat) + obj_pos_flat
            palm_quat_world = quat_mul(obj_quat_flat, palm_quat_local_flat)

            palm_pos_world = palm_pos_world.reshape(n, k, 3)
            palm_quat_world = palm_quat_world.reshape(n, k, 4)
        else:
            palm_pos_world = quat_apply(obj_quat, palm_pos_local) + obj_pos
            palm_quat_world = quat_mul(obj_quat, palm_quat_local)

        return palm_pos_world, palm_quat_world

    def _sample_grasp_keypoints(self, env_ids: torch.Tensor):
        """Sample feasibility-checked keypoints from nearby surface points.

        Each keypoint is a perturbed surface point with perturbed roll that results
        in a feasible world-frame palm pose. Applies filter chain:
        1. Distance: nearby previous point
        2. Normal similarity: prevent flipping to opposite face
        3. Height: palm above table
        4. Finger direction: fingers not toward robot (same as grasp sampling)
        5. (Final only) Exclude bottom fraction: surface normal points up

        Args:
            env_ids: Environment indices.
        """
        from .mdp.utils import get_point_cloud_cache

        n = len(env_ids)
        cfg = self.cfg.hand_trajectory.keypoints
        max_kp = cfg.count[1]

        # Random number of keypoints per env
        self.num_grasp_keypoints[env_ids] = torch.randint(
            cfg.count[0], cfg.count[1] + 1, (n,), device=self.device
        )

        if max_kp == 0:
            return

        # Get point cloud pool for these envs
        cache = get_point_cloud_cache(
            env_ids=env_ids.tolist(),
            prim_path=self.object_prim_path,
            pool_size=256,
        )
        env_ids_cpu = env_ids.cpu()
        geo_indices = cache.geo_indices[env_ids_cpu].to(self.device)
        scales = cache.scales[env_ids_cpu].to(self.device)
        pool_points = cache.all_base_points.to(self.device)[geo_indices]  # (n, pool_size, 3)
        pool_normals = cache.all_base_normals.to(self.device)[geo_indices]  # (n, pool_size, 3)

        # Apply object scale to points (scales is (n, 3) for per-axis scaling)
        pool_points = pool_points * scales.unsqueeze(1)  # (n, 1, 3) * (n, 256, 3) -> (n, 256, 3)
        pool_size = pool_points.shape[1]

        # Start from initial grasp surface representation
        prev_points = self.grasp_surface_point[env_ids].clone()  # (n, 3)
        prev_normals = self.grasp_surface_normal[env_ids].clone()  # (n, 3)
        prev_rolls = self.grasp_roll[env_ids].clone()  # (n,)
        standoffs = self.grasp_standoff[env_ids]  # (n,)

        # Get goal poses for final keypoint (at progress 1.0)
        goal_poses = self.goal_poses[env_ids]  # (n, 7)

        feas_cfg = self.cfg.hand_trajectory.feasibility

        # Config for finger direction check (same as grasp sampling)
        grasp_cfg = self.cfg.hand_trajectory.grasp_sampling
        toward_robot_threshold = grasp_cfg.toward_robot_threshold
        exclude_bottom_frac = grasp_cfg.exclude_bottom_fraction
        exclude_z = -1.0 + exclude_bottom_frac * 2

        for kp_idx in range(max_kp):
            is_final = (kp_idx == max_kp - 1)

            # Final keypoint is at progress 1.0 (goal pose), others at (k+1)/max_kp
            if is_final:
                obj_pose_at_kp = goal_poses
            else:
                obj_pose_at_kp = self._get_object_pose_at_keypoint_time(env_ids, kp_idx)  # (n, 7)

            # ===== FILTER 1: Distance =====
            distances = (pool_points - prev_points.unsqueeze(1)).norm(dim=-1)  # (n, pool_size)
            dist_ok = distances < cfg.max_surface_perturbation

            # ===== FILTER 2: Normal similarity =====
            normal_sim = (pool_normals * prev_normals.unsqueeze(1)).sum(dim=-1)  # (n, pool_size)
            normal_ok = normal_sim > cfg.normal_similarity_threshold

            # ===== Compute palm poses EXACTLY like grasp: from WORLD normal =====
            obj_quat_at_kp = obj_pose_at_kp[:, 3:7]
            obj_pos_at_kp = obj_pose_at_kp[:, :3]
            obj_quat_exp = obj_quat_at_kp.unsqueeze(1).expand(n, pool_size, 4)
            obj_pos_exp = obj_pos_at_kp.unsqueeze(1).expand(n, pool_size, 3)

            # Step 1: Transform normals to WORLD frame (like grasp line 623)
            pool_normals_world = quat_apply(
                obj_quat_exp.reshape(-1, 4),
                pool_normals.reshape(-1, 3)
            ).reshape(n, pool_size, 3)

            # Step 2: Compute base quaternion from WORLD normal (like grasp line 638)
            base_quat_world = self._quat_from_z_axis(
                -pool_normals_world.reshape(-1, 3)
            ).reshape(n, pool_size, 4)

            # Step 3: Compute closest feasible roll for each candidate point
            # Desired roll = prev_roll + small perturbation (for variation)
            desired_rolls = prev_rolls.unsqueeze(1) + (
                (torch.rand(n, pool_size, device=self.device) * 2 - 1) * cfg.roll_perturbation
            )  # (n, pool_size)

            # Compute closest feasible roll for each point (apply all constraints simultaneously)
            constraints = [(0, toward_robot_threshold)]  # x-component (toward robot)
            if grasp_cfg.exclude_upward:
                constraints.append((2, grasp_cfg.upward_threshold))  # z-component (upward)

            candidate_rolls = self._compute_feasible_roll_multi_constraint(
                base_quat=base_quat_world.reshape(-1, 4),
                desired_roll=desired_rolls.reshape(-1),
                constraints=constraints,
            ).reshape(n, pool_size)

            # Step 4: Apply the computed rolls
            roll_quat = quat_from_euler_xyz(
                torch.zeros(n * pool_size, device=self.device),
                torch.zeros(n * pool_size, device=self.device),
                candidate_rolls.reshape(-1)
            ).reshape(n, pool_size, 4)
            palm_quat_world = quat_mul(
                base_quat_world.reshape(-1, 4),
                roll_quat.reshape(-1, 4)
            ).reshape(n, pool_size, 4)

            # Palm position in world frame
            standoffs_exp = standoffs.unsqueeze(1).expand(n, pool_size)
            pool_points_world = quat_apply(
                obj_quat_exp.reshape(-1, 4),
                pool_points.reshape(-1, 3)
            ).reshape(n, pool_size, 3) + obj_pos_exp
            palm_pos_world = pool_points_world + standoffs_exp.unsqueeze(-1) * pool_normals_world

            # ===== FILTER 3: Height check =====
            height_ok = palm_pos_world[..., 2] > (self.table_height + feas_cfg.min_height)

            # ===== FILTER 4: Finger direction - verify the computed roll is actually feasible =====
            # (It might not be if no feasible roll exists for this surface normal)
            x_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device)
            finger_dir_world = quat_apply(
                palm_quat_world.reshape(-1, 4),
                x_axis.expand(n * pool_size, 3)
            ).reshape(n, pool_size, 3)

            # ===== FILTER 4a: Exclude toward robot =====
            finger_ok = finger_dir_world[..., 0] < toward_robot_threshold

            # ===== FILTER 4b: Exclude upward =====
            if grasp_cfg.exclude_upward:
                upward_ok = finger_dir_world[..., 2] < grasp_cfg.upward_threshold
            else:
                upward_ok = torch.ones(n, pool_size, dtype=torch.bool, device=self.device)

            # ===== FILTER 5: Exclude bottom fraction (final keypoint only) =====
            if is_final:
                bottom_ok = pool_normals_world[..., 2] > exclude_z
                valid_mask = dist_ok & normal_ok & height_ok & finger_ok & upward_ok & bottom_ok
            else:
                valid_mask = dist_ok & normal_ok & height_ok & finger_ok & upward_ok

            # ===== Sample from valid candidates =====
            scores = torch.rand(n, pool_size, device=self.device)
            scores[~valid_mask] = -float('inf')
            selected_idx = scores.argmax(dim=1)  # (n,)

            has_valid = valid_mask.any(dim=1)  # (n,)

            # Gather results - select directly from already-computed values
            batch_idx = torch.arange(n, device=self.device)

            # Select from valid candidates
            selected_palm_quat_world = palm_quat_world[batch_idx, selected_idx]  # (n, 4)
            selected_points = pool_points[batch_idx, selected_idx]  # (n, 3)
            selected_normals = pool_normals[batch_idx, selected_idx]  # (n, 3)
            selected_rolls = candidate_rolls[batch_idx, selected_idx]  # (n,)

            # For fallback, compute palm_quat_world for prev values (might be invalid!)
            prev_normals_world = quat_apply(obj_quat_at_kp, prev_normals)
            prev_base_quat_world = self._quat_from_z_axis(-prev_normals_world)
            prev_roll_quat = quat_from_euler_xyz(
                torch.zeros(n, device=self.device),
                torch.zeros(n, device=self.device),
                prev_rolls
            )
            prev_palm_quat_world = quat_mul(prev_base_quat_world, prev_roll_quat)

            # Apply fallback where no valid candidate
            new_palm_quat_world = torch.where(
                has_valid.unsqueeze(-1),
                selected_palm_quat_world,
                prev_palm_quat_world,
            )
            new_points = torch.where(has_valid.unsqueeze(-1), selected_points, prev_points)
            new_normals = torch.where(has_valid.unsqueeze(-1), selected_normals, prev_normals)
            new_rolls = torch.where(has_valid, selected_rolls, prev_rolls)

            # Convert to local frame for storage (like grasp line 280)
            palm_quat_local = quat_mul(quat_inv(obj_quat_at_kp), new_palm_quat_world)
            palm_pos_local = new_points + standoffs.unsqueeze(-1) * new_normals

            # Store keypoint surface representation
            self.keypoint_surface_points[env_ids, kp_idx] = new_points
            self.keypoint_surface_normals[env_ids, kp_idx] = new_normals
            self.keypoint_rolls[env_ids, kp_idx] = new_rolls

            # Store pose representation for trajectory generation
            self.grasp_keypoints[env_ids, kp_idx, :3] = palm_pos_local
            self.grasp_keypoints[env_ids, kp_idx, 3:7] = palm_quat_local

            # Store for visualization
            self.debug_keypoint_points_local[env_ids, kp_idx] = new_points

            # Chain: next keypoint perturbs from this one
            prev_points = new_points
            prev_normals = new_normals
            prev_rolls = new_rolls

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

    def _generate_hand_trajectory_from_segments(self, env_ids: torch.Tensor):
        """Generate hand trajectory from segments with coupling modes + keypoints.

        Keypoints and coupling modes work together:
        - Keypoints define relative hand pose in object frame (micro-adjustments)
        - Coupling modes control how that relative pose tracks the object

        Args:
            env_ids: Environment indices
        """
        from isaaclab.utils.math import quat_apply, quat_mul

        n = len(env_ids)
        times = torch.arange(self.total_targets, device=self.device).float() * self.target_dt

        # Get per-env keypoint counts (used later in trajectory generation)
        num_kp = self.num_grasp_keypoints[env_ids]  # (n,)

        # ===== FULLY VECTORIZED: Process ALL (env × segment × timestep) at once - ZERO LOOPS! =====

        max_segs = self.segment_durations.shape[1]

        # Step 1: Precompute segment masks for ALL segments at once (n, max_segs, total_targets)
        times_3d = times.view(1, 1, -1).expand(n, max_segs, -1)  # (n, max_segs, total_targets)
        seg_starts_3d = self.segment_boundaries[env_ids, :-1].unsqueeze(2)  # (n, max_segs, 1)
        seg_ends_3d = self.segment_boundaries[env_ids, 1:].unsqueeze(2)  # (n, max_segs, 1)
        valid_segs_3d = (torch.arange(max_segs, device=self.device).view(1, -1, 1) < self.num_segments[env_ids].view(n, 1, 1))  # (n, max_segs, 1)

        # For LAST segment: use <= for end boundary to include exact endpoint
        is_last_seg_3d = (torch.arange(max_segs, device=self.device).view(1, -1, 1) == (self.num_segments[env_ids].view(n, 1, 1) - 1))
        seg_masks_normal = (times_3d >= seg_starts_3d) & (times_3d < seg_ends_3d)
        seg_masks_last = (times_3d >= seg_starts_3d) & (times_3d <= seg_ends_3d)
        seg_masks_3d = torch.where(is_last_seg_3d, seg_masks_last, seg_masks_normal) & valid_segs_3d  # (n, max_segs, total_targets)

        # Step 2: Compute total manipulation duration per env (vectorized)
        manip_mask_2d = ~self.segment_is_grasp[env_ids] & ~self.segment_is_release[env_ids]  # (n, max_segs)
        manip_durations_per_seg = self.segment_durations[env_ids] * manip_mask_2d.float()  # (n, max_segs)
        manip_duration_total = manip_durations_per_seg.sum(dim=1).clamp(min=1e-6)  # (n,)

        # Step 3: Find LAST timestep per (env, segment) for smooth transitions - FULLY VECTORIZED!
        # Reverse timestep dimension, find first True, convert back to get LAST True
        seg_masks_reversed = seg_masks_3d.flip(dims=[2])  # (n, max_segs, total_targets)
        first_true_reversed = seg_masks_reversed.long().argmax(dim=2)  # (n, max_segs) - index of first True in reversed
        last_timestep_per_segment = (self.total_targets - 1) - first_true_reversed  # (n, max_segs)
        has_any_timesteps = seg_masks_3d.any(dim=2)  # (n, max_segs)
        last_timestep_per_segment = torch.where(has_any_timesteps, last_timestep_per_segment, torch.zeros_like(last_timestep_per_segment))

        # Initialize output
        positions_all = torch.zeros(n, self.total_targets, 3, device=self.device)
        orientations_all = torch.zeros(n, self.total_targets, 4, device=self.device)
        # Default orientation to identity for any timesteps not covered by masks
        orientations_all[:, :, 0] = 1.0

        # Step 4: Process GRASP segments - ALL at once
        is_grasp_3d = self.segment_is_grasp[env_ids].unsqueeze(2).expand(-1, -1, self.total_targets)  # (n, max_segs, total_targets)
        grasp_mask_3d = seg_masks_3d & is_grasp_3d  # (n, max_segs, total_targets)

        env_idx_g, seg_idx_g, time_idx_g = torch.where(grasp_mask_3d)

        if len(env_idx_g) > 0:
            # Compute local progress for ALL grasp timesteps
            local_t_g = ((times[time_idx_g] - self.segment_boundaries[env_ids[env_idx_g], seg_idx_g]) /
                         self.segment_durations[env_ids[env_idx_g], seg_idx_g].clamp(min=1e-6)).clamp(0, 1)

            # Gather object poses
            global_env_idx = env_ids[env_idx_g]
            obj_pos_g = self.trajectory[global_env_idx, time_idx_g, :3]
            obj_quat_g = self.trajectory[global_env_idx, time_idx_g, 3:7]

            # Transform grasp pose to world
            grasp_pos_world_g = quat_apply(obj_quat_g, self.grasp_pose[global_env_idx, :3]) + obj_pos_g
            grasp_quat_world_g = quat_mul(obj_quat_g, self.grasp_pose[global_env_idx, 3:7])

            # Start pose (MUST be in world coordinates!)
            start_pos_g = self.start_palm_pose[global_env_idx, :3]
            start_quat_g = self.start_palm_pose[global_env_idx, 3:7]

            # Handle approach_distance if configured
            cfg_grasp = self.cfg.hand_trajectory.grasp_sampling
            if cfg_grasp.approach_distance is not None and cfg_grasp.approach_distance > 0:
                # Compute pre-grasp pose: grasp_pos - approach_distance * palm_z_axis
                # Palm Z points toward object (approach direction)
                palm_z_local = torch.tensor([0., 0., 1.], device=self.device).expand(len(env_idx_g), 3)
                palm_z_world_g = quat_apply(grasp_quat_world_g, palm_z_local)  # (num_grasp_timesteps, 3)

                pre_grasp_pos_g = grasp_pos_world_g - cfg_grasp.approach_distance * palm_z_world_g
                pre_grasp_quat_g = grasp_quat_world_g.clone()  # Same orientation as grasp

                # Two-phase interpolation:
                # Phase 1 (0 → align_fraction): start → pre-grasp
                # Phase 2 (align_fraction → 1.0): pre-grasp → grasp
                align_frac = cfg_grasp.align_fraction
                in_phase1 = local_t_g < align_frac

                # Fast-track rotation: completes earlier than position
                rot_fraction = self.cfg.hand_trajectory.grasp_rot_completion_fraction
                rot_progress = (local_t_g / rot_fraction).clamp(0, 1)

                # Phase 1: Interpolate start → pre-grasp
                t_phase1 = (local_t_g / align_frac).clamp(0, 1)
                pos_phase1 = start_pos_g + t_phase1.unsqueeze(1) * (pre_grasp_pos_g - start_pos_g)
                # Rotation uses fast-tracked progress during align phase
                rot_progress_phase1 = (rot_progress / align_frac).clamp(0, 1)
                quat_phase1 = self._batched_slerp(start_quat_g, pre_grasp_quat_g, rot_progress_phase1)

                # Phase 2: Interpolate pre-grasp → grasp
                t_phase2 = ((local_t_g - align_frac) / (1.0 - align_frac)).clamp(0, 1)
                pos_phase2 = pre_grasp_pos_g + t_phase2.unsqueeze(1) * (grasp_pos_world_g - pre_grasp_pos_g)
                # Rotation already complete during align phase, so just hold grasp orientation
                quat_phase2 = grasp_quat_world_g.expand(len(env_idx_g), 4)

                # Select based on phase
                hand_pos_g = torch.where(in_phase1.unsqueeze(1), pos_phase1, pos_phase2)
                quat_mask = in_phase1.unsqueeze(1).expand(-1, 4)
                hand_quat_g = torch.where(quat_mask, quat_phase1, quat_phase2)
            else:
                # No approach distance: direct interpolation start → grasp
                # Fast-track rotation: completes earlier than position
                rot_fraction = self.cfg.hand_trajectory.grasp_rot_completion_fraction
                rot_progress = (local_t_g / rot_fraction).clamp(0, 1)

                hand_pos_g = start_pos_g + local_t_g.unsqueeze(1) * (grasp_pos_world_g - start_pos_g)
                hand_quat_g = self._batched_slerp(start_quat_g, grasp_quat_world_g, rot_progress)

            # Scatter write
            positions_all[env_idx_g, time_idx_g] = hand_pos_g
            orientations_all[env_idx_g, time_idx_g] = hand_quat_g

        # Step 5: Pre-compute segment boundary orientations for position_only coupling
        # We can't use orientations_all here because it hasn't been filled yet!
        # Instead, compute what the orientation WILL BE at each segment boundary.
        # FULLY VECTORIZED: Process all (env, seg_idx) pairs at once - NO LOOP!

        # Build keypoint chain: [grasp, kp_0, kp_1, ..., kp_{N-1}]
        max_kp = self.grasp_keypoints.shape[1]
        kp_chain = torch.zeros(n, max_kp + 1, 7, device=self.device)
        kp_chain[:, 0, :3] = self.grasp_pose[env_ids, :3]
        kp_chain[:, 0, 3:7] = self.grasp_pose[env_ids, 3:7]
        if max_kp > 0:
            kp_chain[:, 1:, :3] = self.grasp_keypoints[env_ids, :, :3]
            kp_chain[:, 1:, 3:7] = self.grasp_keypoints[env_ids, :, 3:7]

        # Per-env: if num_kp == 0, duplicate grasp at index 1 (ignore kp_0).
        # Only do this if max_kp > 0, otherwise kp_chain has no index 1.
        zero_kp_mask = num_kp == 0
        if max_kp > 0 and zero_kp_mask.any():
            kp_chain[zero_kp_mask, 1, :3] = kp_chain[zero_kp_mask, 0, :3]
            kp_chain[zero_kp_mask, 1, 3:7] = kp_chain[zero_kp_mask, 0, 3:7]

        # Get segment start times for ALL segments at once: (n, max_segs)
        seg_start_times = self.segment_boundaries[env_ids, :max_segs]  # (n, max_segs)
        seg_start_indices = (seg_start_times / self.target_dt).long().clamp(0, self.total_targets - 1)  # (n, max_segs)

        # Get object quaternions at segment starts: (n, max_segs, 4)
        env_range_2d = torch.arange(n, device=self.device).unsqueeze(1).expand(n, max_segs)  # (n, max_segs)
        # NOTE: self.trajectory is indexed by GLOBAL env ids, not local batch indices.
        global_env_range_2d = env_ids.unsqueeze(1).expand(n, max_segs)  # (n, max_segs)
        obj_quat_at_starts = self.trajectory[global_env_range_2d, seg_start_indices, 3:7]  # (n, max_segs, 4)

        # Compute global manipulation progress at ALL segment boundaries: (n, max_segs)
        # Must be measured in "manipulation-only time" (exclude grasp/release durations),
        # matching the progress definition used in Step 6.
        cumsum_manip_time = torch.cumsum(manip_durations_per_seg, dim=1)  # (n, max_segs)
        seg_manip_start_time = cumsum_manip_time - manip_durations_per_seg  # (n, max_segs)
        global_manip_progress = (seg_manip_start_time / manip_duration_total.unsqueeze(1)).clamp(0, 1)  # (n, max_segs)

        # Interpolate through keypoint chain at all progress values
        num_kp_per_env = num_kp.unsqueeze(1)  # (n, 1) - num_kp is already indexed by env_ids
        num_segments_kp = num_kp_per_env.clamp(min=1)  # (n, 1)

        seg_idx_kp = (global_manip_progress * num_segments_kp).long().clamp(min=0)  # (n, max_segs)
        seg_start_kp = seg_idx_kp.float() / num_segments_kp  # (n, max_segs)
        seg_end_kp = (seg_idx_kp + 1).float() / num_segments_kp  # (n, max_segs)
        local_progress_kp = ((global_manip_progress - seg_start_kp) / (seg_end_kp - seg_start_kp).clamp(min=1e-6)).clamp(0, 1)  # (n, max_segs)

        # Clamp keypoint indices
        # kp_chain has shape (n, max_kp + 1, 7), so valid indices are [0, max_kp]
        # When max_kp=0, only index 0 is valid, so we must clamp to 0 not 1.
        actual_max_chain_idx = max_kp  # Max valid index in kp_chain's second dimension
        max_chain_idx = torch.where(num_kp_per_env == 0,
                                     torch.zeros_like(num_kp_per_env),
                                     num_kp_per_env.clamp(max=actual_max_chain_idx))  # (n, 1)
        from_idx_kp = seg_idx_kp.clamp(max=max_chain_idx)  # (n, max_segs)
        to_idx_kp = (seg_idx_kp + 1).clamp(max=max_chain_idx)  # (n, max_segs)

        # Gather keypoints using advanced indexing (flatten, gather, reshape)
        env_idx_flat = env_range_2d.reshape(-1)  # (n * max_segs,)
        from_idx_flat = from_idx_kp.reshape(-1)  # (n * max_segs,)
        to_idx_flat = to_idx_kp.reshape(-1)  # (n * max_segs,)

        from_quat_local_flat = kp_chain[env_idx_flat, from_idx_flat, 3:7]  # (n*max_segs, 4)
        to_quat_local_flat = kp_chain[env_idx_flat, to_idx_flat, 3:7]  # (n*max_segs, 4)

        # Interpolate (batched SLERP across all env-segment pairs)
        interp_quat_local = self._batched_slerp(
            from_quat_local_flat,
            to_quat_local_flat,
            local_progress_kp.reshape(n * max_segs)
        ).reshape(n, max_segs, 4)

        # Transform to world frame: batched quaternion multiplication
        hand_quat_world = quat_mul(
            obj_quat_at_starts.reshape(n * max_segs, 4),
            interp_quat_local.reshape(n * max_segs, 4)
        ).reshape(n, max_segs, 4)

        # Use these pre-computed orientations for position_only coupling
        prev_hand_quat_per_seg = hand_quat_world  # (n, max_segs, 4)

        # Step 6: Process MANIPULATION segments (coupling mode 0 or 1) - ALL at once
        is_manip_3d = manip_mask_2d.unsqueeze(2).expand(-1, -1, self.total_targets)  # (n, max_segs, total_targets)
        coupling_full_or_pos = ((self.coupling_modes[env_ids] == 0) | (self.coupling_modes[env_ids] == 1)).unsqueeze(2).expand(-1, -1, self.total_targets)
        manip_mask_3d = seg_masks_3d & is_manip_3d & coupling_full_or_pos

        env_idx_m, seg_idx_m, time_idx_m = torch.where(manip_mask_3d)

        if len(env_idx_m) > 0:
            # Local time within segment
            local_t_within_seg = times[time_idx_m] - self.segment_boundaries[env_ids[env_idx_m], seg_idx_m]

            # Global manipulation progress [0, 1]
            global_manip_progress = ((seg_manip_start_time[env_idx_m, seg_idx_m] + local_t_within_seg) /
                                     manip_duration_total[env_idx_m]).clamp(0, 1)

            # Keypoint interpolation
            num_kp_per_env = num_kp[env_idx_m]  # (num_manip_timesteps,)
            num_segments_kp = num_kp_per_env.clamp(min=1)

            seg_idx_kp = (global_manip_progress * num_segments_kp).long().clamp(min=0)
            seg_start_kp = seg_idx_kp.float() / num_segments_kp
            seg_end_kp = (seg_idx_kp + 1).float() / num_segments_kp
            local_progress_kp = ((global_manip_progress - seg_start_kp) / (seg_end_kp - seg_start_kp).clamp(min=1e-6)).clamp(0, 1)

            # Gather keypoints with PER-ENV chain index clamping
            # For env with num_kp keypoints, max chain idx = num_kp (final keypoint)
            # num_kp=0: max_chain_idx=0 (only grasp at position 0, when max_kp=0)
            # num_kp=N: max_chain_idx=N (chain has grasp + N keypoints)
            # IMPORTANT: When max_kp=0, kp_chain has shape (n, 1, 7), so only index 0 is valid
            actual_max_chain_idx = max_kp  # Max valid index in kp_chain's second dimension
            max_chain_idx_per_timestep = torch.where(num_kp_per_env == 0,
                                                      torch.zeros_like(num_kp_per_env),
                                                      num_kp_per_env.clamp(max=actual_max_chain_idx))
            from_idx_kp = seg_idx_kp.clamp(max=max_chain_idx_per_timestep)
            to_idx_kp = (seg_idx_kp + 1).clamp(max=max_chain_idx_per_timestep)

            from_pos_local_m = kp_chain[env_idx_m, from_idx_kp, :3]
            from_quat_local_m = kp_chain[env_idx_m, from_idx_kp, 3:7]
            to_pos_local_m = kp_chain[env_idx_m, to_idx_kp, :3]
            to_quat_local_m = kp_chain[env_idx_m, to_idx_kp, 3:7]

            # Interpolate keypoints
            interp_pos_local_m = from_pos_local_m + local_progress_kp.unsqueeze(1) * (to_pos_local_m - from_pos_local_m)
            interp_quat_local_m = self._batched_slerp(from_quat_local_m, to_quat_local_m, local_progress_kp)

            # Gather object trajectory
            global_env_idx_m = env_ids[env_idx_m]
            obj_pos_m = self.trajectory[global_env_idx_m, time_idx_m, :3]
            obj_quat_m = self.trajectory[global_env_idx_m, time_idx_m, 3:7]

            # Apply coupling (split by coupling mode)
            coupling_m = self.coupling_modes[env_ids[env_idx_m], seg_idx_m]

            # FULL coupling (mode 0)
            full_mask = coupling_m == 0
            if full_mask.any():
                hand_pos_full = quat_apply(obj_quat_m[full_mask], interp_pos_local_m[full_mask]) + obj_pos_m[full_mask]
                hand_quat_full = quat_mul(obj_quat_m[full_mask], interp_quat_local_m[full_mask])
                positions_all[env_idx_m[full_mask], time_idx_m[full_mask]] = hand_pos_full
                orientations_all[env_idx_m[full_mask], time_idx_m[full_mask]] = hand_quat_full

            # POSITION_ONLY coupling (mode 1)
            pos_only_mask = coupling_m == 1
            if pos_only_mask.any():
                hand_pos_posonly = quat_apply(obj_quat_m[pos_only_mask], interp_pos_local_m[pos_only_mask]) + obj_pos_m[pos_only_mask]
                # Orientation: use previous segment's last orientation
                hand_quat_posonly = prev_hand_quat_per_seg[env_idx_m[pos_only_mask], seg_idx_m[pos_only_mask]]
                positions_all[env_idx_m[pos_only_mask], time_idx_m[pos_only_mask]] = hand_pos_posonly
                orientations_all[env_idx_m[pos_only_mask], time_idx_m[pos_only_mask]] = hand_quat_posonly

        # Step 7: Process DECOUPLED segments (coupling mode 2) - ALL at once
        is_decouple = (self.coupling_modes[env_ids] == 2).unsqueeze(2).expand(-1, -1, self.total_targets)
        decouple_mask_3d = seg_masks_3d & is_decouple

        env_idx_d, seg_idx_d, time_idx_d = torch.where(decouple_mask_3d)

        if len(env_idx_d) > 0:
            # Target: release pose (default - custom poses handled separately if needed)
            global_env_idx_d = env_ids[env_idx_d]
            target_pos_d = self.release_pose[global_env_idx_d, :3]
            target_quat_d = self.release_pose[global_env_idx_d, 3:7]

            # Compute starting pose for interpolation:
            # Transform grasp keypoint using object pose at END of last manipulation segment
            # This matches old code: final_grasp_pos_w = quat_apply(obj_quat_at_release, final_rel_pos) + obj_pos_at_release

            # Find the last timestep of the previous segment
            prev_seg_idx_d = (seg_idx_d - 1).clamp(min=0)
            prev_seg_end_time = self.segment_boundaries[env_ids[env_idx_d], prev_seg_idx_d + 1]
            prev_seg_end_idx = (prev_seg_end_time / self.target_dt).long().clamp(0, self.total_targets - 1)

            # Get object pose at that timestep
            obj_pos_at_release = self.trajectory[global_env_idx_d, prev_seg_end_idx, :3]
            obj_quat_at_release = self.trajectory[global_env_idx_d, prev_seg_end_idx, 3:7]

            # Get grasp pose in object-local frame (last keypoint in chain)
            # kp_chain structure: [grasp, kp_0, kp_1, ..., kp_{N-1}]
            # - If num_kp == 0: final is grasp (from self.grasp_pose)
            # - If num_kp == N: final is kp_{N-1} (from self.grasp_keypoints[:, N-1])
            num_kp = self.num_grasp_keypoints[global_env_idx_d]
            has_keypoints = num_kp > 0

            # For envs WITH keypoints: use last keypoint (index num_kp-1)
            # For envs WITHOUT keypoints: use grasp pose
            final_rel_pos = torch.zeros(len(global_env_idx_d), 3, device=self.device)
            final_rel_quat = torch.zeros(len(global_env_idx_d), 4, device=self.device)

            if has_keypoints.any():
                kp_idx = (num_kp[has_keypoints] - 1).clamp(min=0)  # Last keypoint index (0-based)
                final_rel_pos[has_keypoints] = self.grasp_keypoints[global_env_idx_d[has_keypoints], kp_idx, :3]
                final_rel_quat[has_keypoints] = self.grasp_keypoints[global_env_idx_d[has_keypoints], kp_idx, 3:7]

            if (~has_keypoints).any():
                final_rel_pos[~has_keypoints] = self.grasp_pose[global_env_idx_d[~has_keypoints], :3]
                final_rel_quat[~has_keypoints] = self.grasp_pose[global_env_idx_d[~has_keypoints], 3:7]

            # Transform to world frame
            final_grasp_pos_w = quat_apply(obj_quat_at_release, final_rel_pos) + obj_pos_at_release
            final_grasp_quat_w = quat_mul(obj_quat_at_release, final_rel_quat)

            # Local progress (linear)
            # Speed up retreat to complete before buffer ends, leaving room for hold at release
            # This ensures the last window_size targets are all at release pose
            window_buffer = self.cfg.trajectory.window_size * self.target_dt
            segment_duration_d = self.segment_durations[env_ids[env_idx_d], seg_idx_d]
            effective_duration_d = (segment_duration_d - window_buffer).clamp(min=self.target_dt)

            local_t_d = ((times[time_idx_d] - self.segment_boundaries[env_ids[env_idx_d], seg_idx_d]) /
                         effective_duration_d).clamp(0, 1)

            # Apply ease-in: slow start, fast end (t^power)
            release_ease_power = getattr(self.cfg.trajectory, 'release_ease_power', 2.0)
            eased_t_d = local_t_d ** release_ease_power

            # Interpolate with easing from final_grasp_pos_w to release pose
            hand_pos_d = final_grasp_pos_w + eased_t_d.unsqueeze(1) * (target_pos_d - final_grasp_pos_w)
            hand_quat_d = self._batched_slerp(final_grasp_quat_w, target_quat_d, eased_t_d)

            # Scatter write
            positions_all[env_idx_d, time_idx_d] = hand_pos_d
            orientations_all[env_idx_d, time_idx_d] = hand_quat_d

        # Step 8: Write to hand_trajectory buffer (vectorized scatter)
        env_idx_all = env_ids.unsqueeze(1).expand(-1, self.total_targets)  # (n, total_targets)
        time_idx_all = torch.arange(self.total_targets, device=self.device).unsqueeze(0).expand(n, -1)  # (n, total_targets)

        env_flat = env_idx_all.reshape(-1)
        time_flat = time_idx_all.reshape(-1)

        self.hand_trajectory[env_flat, time_flat, :3] = positions_all.reshape(-1, 3)
        self.hand_trajectory[env_flat, time_flat, 3:7] = orientations_all.reshape(-1, 4)

        # Store hand poses at segment boundaries for position_only coupling during replanning
        # FULLY VECTORIZED: Process all (env, seg_idx) pairs at once - NO LOOP!
        max_segs_plus_one = self.segment_boundaries.shape[1]  # max_segs + 1

        # Get all boundary times: (n, max_segs+1)
        boundary_times_all = self.segment_boundaries[env_ids]  # (n, max_segs+1)

        # Convert to timestep indices
        boundary_indices_all = (boundary_times_all / self.target_dt).long().clamp(min=0, max=self.total_targets - 1)  # (n, max_segs+1)

        # Gather hand poses at all boundary indices using advanced indexing
        # For each (env, seg_idx), get hand_trajectory[env, boundary_indices_all[env, seg_idx]]
        env_indices_expanded = env_ids.unsqueeze(1).expand(-1, max_segs_plus_one)  # (n, max_segs+1)

        # Flatten for gather
        env_flat = env_indices_expanded.reshape(-1)  # (n * (max_segs+1),)
        time_flat = boundary_indices_all.reshape(-1)  # (n * (max_segs+1),)

        # Gather: (n * (max_segs+1), 7)
        hand_poses_flat = self.hand_trajectory[env_flat, time_flat]

        # Reshape and store: (n, max_segs+1, 7)
        self.hand_pose_at_segment_boundary[env_ids] = hand_poses_flat.reshape(n, max_segs_plus_one, 7)

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

    def _compute_segment_poses_vectorized(self, env_ids: torch.Tensor):
        """VECTORIZED: Compute segment target poses for all specified envs in parallel.

        Directly writes to self.segment_poses[env_ids, :, :] buffer.

        Args:
            env_ids: (n,) Environment indices to compute poses for
        """
        from isaaclab.utils.math import quat_mul, quat_from_angle_axis

        n = len(env_ids)
        max_segs = self.segment_poses.shape[1]

        # Initialize
        self.segment_poses[env_ids] = 0.0

        # FULLY VECTORIZED with sequential dependency handling via iterative updates
        # Process segments in order (small loop over ~10 segments, NOT 16k envs!)
        # This loop is NECESSARY due to chaining (segment i depends on segment i-1)
        # BUT it's over segments (max ~10), processing ALL 16k envs in parallel per segment!

        for seg_idx in range(max_segs):
            has_segment = (seg_idx < self.num_segments[env_ids])
            if not has_segment.any():
                break

            # Get previous pose (vectorized)
            prev_poses = self.start_poses[env_ids] if seg_idx == 0 else self.segment_poses[env_ids, seg_idx - 1]

            # HELICAL: Compute ALL helical poses for this segment index (vectorized)
            is_helical = self.segment_is_helical[env_ids, seg_idx] & has_segment
            if is_helical.any():
                hel_envs = env_ids[is_helical]
                translation = self.segment_helical_translation[hel_envs, seg_idx]
                axes = self.segment_helical_axis[hel_envs, seg_idx] / torch.norm(self.segment_helical_axis[hel_envs, seg_idx], dim=1, keepdim=True)
                angles = self.segment_helical_rotation[hel_envs, seg_idx]
                delta_quats = quat_from_angle_axis(angles, axes)
                self.segment_poses[hel_envs, seg_idx, :3] = prev_poses[is_helical, :3] + translation
                self.segment_poses[hel_envs, seg_idx, 3:7] = quat_mul(prev_poses[is_helical, 3:7], delta_quats)

            # WAYPOINT: Compute ALL waypoint poses for this segment index (vectorized)
            is_waypoint = ~is_helical & has_segment
            if is_waypoint.any():
                wp_envs = env_ids[is_waypoint]
                has_custom = self.segment_has_custom_pose[wp_envs, seg_idx]

                # Custom poses (convert from env-local to world frame)
                if has_custom.any():
                    custom_envs = wp_envs[has_custom]
                    custom_pose_local = self.segment_custom_pose[custom_envs, seg_idx]
                    # Position: add env_origins
                    custom_pos_world = custom_pose_local[:, :3] + self.env_origins[custom_envs]
                    # Orientation: keep as-is (quaternions are orientation, not affected by translation)
                    custom_quat = custom_pose_local[:, 3:7]
                    self.segment_poses[custom_envs, seg_idx] = torch.cat([custom_pos_world, custom_quat], dim=1)

                # Computed poses (vectorized with where)
                needs_compute = ~has_custom
                if needs_compute.any():
                    comp_envs = wp_envs[needs_compute]
                    comp_prev_poses = prev_poses[is_waypoint][needs_compute]

                    # Start with previous pose, then overwrite based on segment type
                    result_poses = comp_prev_poses.clone()

                    # Grasp → start_poses
                    is_grasp = self.segment_is_grasp[comp_envs, seg_idx]
                    result_poses = torch.where(is_grasp.unsqueeze(1), self.start_poses[comp_envs], result_poses)

                    # Goal → goal_poses
                    is_goal = self.segment_is_goal[comp_envs, seg_idx]
                    result_poses = torch.where(is_goal.unsqueeze(1), self.goal_poses[comp_envs], result_poses)

                    # Return → start_poses
                    is_return = self.segment_is_return[comp_envs, seg_idx]
                    result_poses = torch.where(is_return.unsqueeze(1), self.start_poses[comp_envs], result_poses)

                    self.segment_poses[comp_envs, seg_idx] = result_poses

    def _generate_waypoint_segment_batched(
        self, start_poses, end_poses, all_times, seg_start_times, durations, env_masks
    ):
        """FULLY VECTORIZED: Generate waypoint trajectories for multiple envs in parallel.

        Args:
            start_poses: (n, 7) start poses for each env
            end_poses: (n, 7) end poses for each env
            all_times: (T,) global time array
            seg_start_times: (n,) segment start time per env
            durations: (n,) segment duration per env
            env_masks: (n, T) boolean mask of valid timesteps per env

        Returns:
            positions: (n, T, 3) positions (invalid timesteps are zeros)
            orientations: (n, T, 4) orientations (invalid timesteps are zeros)
        """
        n = start_poses.shape[0]
        T = all_times.shape[0]

        # Compute local progress for each env at each timestep
        raw_t = ((all_times.unsqueeze(0) - seg_start_times.unsqueeze(1)) / durations.unsqueeze(1)).clamp(0, 1)  # (n, T)

        # Apply easing (fully vectorized)
        easing_power = self.cfg.trajectory.easing_power
        local_t = torch.where(
            raw_t < 0.5,
            ((raw_t * 2.0) ** easing_power) * 0.5,
            0.5 + (1.0 - ((1.0 - raw_t) * 2.0) ** easing_power) * 0.5
        )  # (n, T)

        # Interpolate positions (fully vectorized)
        start_pos = start_poses[:, :3].unsqueeze(1)  # (n, 1, 3)
        end_pos = end_poses[:, :3].unsqueeze(1)  # (n, 1, 3)
        positions = start_pos + local_t.unsqueeze(-1) * (end_pos - start_pos)  # (n, T, 3)

        # Interpolate orientations (FULLY VECTORIZED - no loops!)
        start_quat = start_poses[:, 3:7]  # (n, 4)
        end_quat = end_poses[:, 3:7]  # (n, 4)

        # Expand to (n, T, 4) for batched SLERP
        start_quat_expanded = start_quat.unsqueeze(1).expand(n, T, 4)  # (n, T, 4)
        end_quat_expanded = end_quat.unsqueeze(1).expand(n, T, 4)  # (n, T, 4)

        # Reshape to (n*T, 4) for batched SLERP
        start_flat = start_quat_expanded.reshape(n * T, 4)
        end_flat = end_quat_expanded.reshape(n * T, 4)
        t_flat = local_t.reshape(n * T)

        # Batched SLERP for ALL (n*T) pairs at once
        orientations_flat = self._batched_slerp(start_flat, end_flat, t_flat)  # (n*T, 4)
        orientations = orientations_flat.reshape(n, T, 4)  # (n, T, 4)

        # Mask invalid timesteps
        positions = positions * env_masks.unsqueeze(-1).float()
        orientations = orientations * env_masks.unsqueeze(-1).float()

        return positions, orientations

    def _generate_helical_segment_batched(
        self, start_poses, end_poses, all_times, seg_start_times, durations, axes, rotations, translations, env_masks
    ):
        """FULLY VECTORIZED: Generate helical trajectories for multiple envs - NO LOOPS.

        Args:
            start_poses: (n, 7) start poses
            end_poses: (n, 7) end poses
            all_times: (T,) global time array
            seg_start_times: (n,) segment start times
            durations: (n,) segment durations
            axes: (n, 3) rotation axes
            rotations: (n,) total rotations in radians
            translations: (n, 3) translation vectors
            env_masks: (n, T) boolean mask

        Returns:
            positions: (n, T, 3)
            orientations: (n, T, 4)
        """
        from isaaclab.utils.math import quat_mul

        n = start_poses.shape[0]
        T = all_times.shape[0]

        # Normalize axes
        axes_norm = axes / torch.norm(axes, dim=1, keepdim=True)  # (n, 3)

        # Compute local progress: (n, T)
        local_t = ((all_times.unsqueeze(0) - seg_start_times.unsqueeze(1)) / durations.unsqueeze(1).clamp(min=1e-6)).clamp(0, 1)

        # Positions: start + t * translation
        start_pos = start_poses[:, :3].unsqueeze(1)  # (n, 1, 3)
        positions = start_pos + local_t.unsqueeze(-1) * translations.unsqueeze(1)  # (n, T, 3)

        # Orientations: incremental rotation around axis
        start_quat = start_poses[:, 3:7]  # (n, 4)
        angles = rotations.unsqueeze(1) * local_t  # (n, T)

        # Convert axis-angle to quaternions (batched)
        half_angles = angles / 2  # (n, T)
        sin_half = torch.sin(half_angles)  # (n, T)
        cos_half = torch.cos(half_angles)  # (n, T)

        # delta_quats: (n, T, 4)
        delta_quats = torch.zeros(n, T, 4, device=self.device)
        delta_quats[:, :, 0] = cos_half
        delta_quats[:, :, 1:4] = axes_norm.unsqueeze(1) * sin_half.unsqueeze(-1)  # (n, T, 3)

        # Apply rotation: quat_mul for all (n*T) pairs
        delta_flat = delta_quats.reshape(n * T, 4)
        start_flat = start_quat.unsqueeze(1).expand(n, T, 4).reshape(n * T, 4)
        orientations_flat = quat_mul(delta_flat, start_flat)
        orientations = orientations_flat.reshape(n, T, 4)

        # Mask invalid timesteps
        positions = positions * env_masks.unsqueeze(-1).float()
        orientations = orientations * env_masks.unsqueeze(-1).float()

        return positions, orientations

    def _generate_full_trajectory(self, env_ids: torch.Tensor):
        """FULLY VECTORIZED: Generate entire trajectory from segments - NO ENV LOOPS.

        Args:
            env_ids: Environment indices to generate for
        """
        n = len(env_ids)
        max_segs = self.segment_poses.shape[1]
        T = self.total_targets

        # Clear buffers for these envs to avoid stale targets if any timestep is not written
        self.trajectory[env_ids] = 0.0
        # Default orientation to identity
        self.trajectory[env_ids, :, 3] = 1.0

        # Compute all segment poses in one vectorized call
        self._compute_segment_poses_vectorized(env_ids)

        # Time array for timestep indices
        times = torch.arange(T, device=self.device).float() * self.target_dt  # (T,)

        # FULLY VECTORIZED: Process ALL (env × segment) at once - NO SEGMENT LOOP!
        # Build 3D start/end poses
        start_poses_3d = torch.zeros(n, max_segs, 7, device=self.device)
        start_poses_3d[:, 0] = self.start_poses[env_ids]
        start_poses_3d[:, 1:] = self.segment_poses[env_ids, :-1]
        end_poses_3d = self.segment_poses[env_ids]

        # Build 3D timing
        seg_start_times_3d = self.segment_boundaries[env_ids, :-1]
        seg_end_times_3d = self.segment_boundaries[env_ids, 1:]
        seg_durations_3d = seg_end_times_3d - seg_start_times_3d

        # Build 3D masks
        valid_seg_mask = torch.arange(max_segs, device=self.device).unsqueeze(0) < self.num_segments[env_ids].unsqueeze(1)
        times_3d = times.view(1, 1, -1).expand(n, max_segs, -1)
        
        # For LAST segment: use <= for end boundary to include exact endpoint
        # This fixes the boundary condition where t=episode_end would be excluded
        is_last_seg = torch.arange(max_segs, device=self.device).unsqueeze(0) == (self.num_segments[env_ids].unsqueeze(1) - 1)
        seg_time_mask_normal = (times_3d >= seg_start_times_3d.unsqueeze(2)) & (times_3d < seg_end_times_3d.unsqueeze(2))
        seg_time_mask_last = (times_3d >= seg_start_times_3d.unsqueeze(2)) & (times_3d <= seg_end_times_3d.unsqueeze(2))
        seg_time_mask = torch.where(is_last_seg.unsqueeze(2), seg_time_mask_last, seg_time_mask_normal)
        seg_mask_3d = seg_time_mask & valid_seg_mask.unsqueeze(2)

        # Separate waypoint/helical
        waypoint_mask_3d = seg_mask_3d & ~self.segment_is_helical[env_ids].unsqueeze(2)
        helical_mask_3d = seg_mask_3d & self.segment_is_helical[env_ids].unsqueeze(2)

        # Generate ALL waypoint segments at once
        env_idx_wp, seg_idx_wp, time_idx_wp = torch.where(waypoint_mask_3d)
        if len(env_idx_wp) > 0:
            global_env_idx_wp = env_ids[env_idx_wp]
            local_t = ((times[time_idx_wp] - seg_start_times_3d[env_idx_wp, seg_idx_wp]) /
                       seg_durations_3d[env_idx_wp, seg_idx_wp].clamp(min=1e-6)).clamp(0, 1)
            easing_power = self.cfg.trajectory.easing_power
            eased_t = torch.where(local_t < 0.5,
                                  0.5 * torch.pow(2 * local_t, easing_power),
                                  1.0 - 0.5 * torch.pow(2 * (1 - local_t), easing_power))
            pos = (start_poses_3d[env_idx_wp, seg_idx_wp, :3] +
                   eased_t.unsqueeze(1) * (end_poses_3d[env_idx_wp, seg_idx_wp, :3] - start_poses_3d[env_idx_wp, seg_idx_wp, :3]))
            quat = self._batched_slerp(start_poses_3d[env_idx_wp, seg_idx_wp, 3:7],
                                       end_poses_3d[env_idx_wp, seg_idx_wp, 3:7], eased_t)

            # Scatter write using advanced indexing
            self.trajectory[global_env_idx_wp, time_idx_wp, :3] = pos
            self.trajectory[global_env_idx_wp, time_idx_wp, 3:7] = quat

        # Generate ALL helical segments at once
        env_idx_hel, seg_idx_hel, time_idx_hel = torch.where(helical_mask_3d)
        if len(env_idx_hel) > 0:
            global_env_idx_hel = env_ids[env_idx_hel]
            axes = self.segment_helical_axis[global_env_idx_hel, seg_idx_hel]
            axes = axes / torch.norm(axes, dim=1, keepdim=True).clamp(min=1e-8)
            local_t = ((times[time_idx_hel] - seg_start_times_3d[env_idx_hel, seg_idx_hel]) /
                       seg_durations_3d[env_idx_hel, seg_idx_hel].clamp(min=1e-6)).clamp(0, 1)
            pos = start_poses_3d[env_idx_hel, seg_idx_hel, :3] + local_t.unsqueeze(1) * self.segment_helical_translation[global_env_idx_hel, seg_idx_hel]
            angles = self.segment_helical_rotation[global_env_idx_hel, seg_idx_hel] * local_t
            delta_quats = torch.zeros(len(env_idx_hel), 4, device=self.device)
            delta_quats[:, 0] = torch.cos(angles / 2)
            delta_quats[:, 1:4] = axes * torch.sin(angles / 2).unsqueeze(1)
            quat = quat_mul(delta_quats, start_poses_3d[env_idx_hel, seg_idx_hel, 3:7])

            # Scatter write using advanced indexing
            self.trajectory[global_env_idx_hel, time_idx_hel, :3] = pos
            self.trajectory[global_env_idx_hel, time_idx_hel, 3:7] = quat

        # Apply penetration check (clamp z so point cloud stays above table)
        # Keep as a post-pass since we generated via sparse scatter writes.
        positions = self.trajectory[env_ids, :, :3]
        orientations = self.trajectory[env_ids, :, 3:7]
        positions = self._apply_penetration_check(positions, orientations, env_ids)
        self.trajectory[env_ids, :, :3] = positions

    def _replan_manipulation_phase(self, env_ids: torch.Tensor):
        """Replan manipulation portion of trajectory to current segment target.

        Segment-aware replanning: determines which segment we're in based on
        current phase_time vs segment boundaries, then replans to that segment's target.

        This method:
        - Only updates trajectory indices from current_idx to segment end
        - Does not reset phase_time or current_idx (episode continues normally)
        - Uses rolling_window as planning horizon
        - Enforces minimum speed to prevent infinite slowdown near target

        Args:
            env_ids: Environment indices to replan.
        """
        n = len(env_ids)
        if n == 0:
            return

        # Get current object positions
        current_pos = self.current_object_poses[env_ids, :3]  # (n, 3)
        current_quat = self.current_object_poses[env_ids, 3:7]  # (n, 4)

        # Get per-env manipulation end index
        manip_end_idx = (self.t_manip_end[env_ids] / self.target_dt).long()  # (n,)
        current_idx = self.current_idx[env_ids]  # (n,)

        # Planning horizon from config
        rolling_window_targets = int(self.cfg.push_t.rolling_window / self.target_dt)

        # FULLY VECTORIZED: Process all envs in parallel - NO LOOP!

        # 1. Find current segment index for all envs using vectorized searchsorted - NO LOOP!
        current_times = self.phase_time[env_ids]  # (n,)
        num_segs = self.num_segments[env_ids]  # (n,)

        # Vectorized segment finding: compare current_time against all boundaries at once
        # segment_boundaries[env, :] = [0, t1, t2, ..., t_num_segs, 0, 0, ...]
        max_segs_plus_one = self.segment_boundaries.shape[1]
        boundaries = self.segment_boundaries[env_ids]  # (n, max_segs+1)

        # Find which segment: boundaries[seg] <= current_time < boundaries[seg+1]
        # Create mask for all segments: (n, max_segs+1)
        current_times_expanded = current_times.unsqueeze(1)  # (n, 1)

        # Check if current_time >= boundary[i] for all i
        after_start = current_times_expanded >= boundaries  # (n, max_segs+1)

        # Find the LAST True value (rightmost segment start that we've passed)
        # Multiply by segment indices and take max
        seg_indices = torch.arange(max_segs_plus_one, device=self.device).unsqueeze(0).expand(n, -1)  # (n, max_segs+1)
        # Set invalid segments to -1 so they don't interfere with max
        seg_indices_masked = torch.where(after_start, seg_indices, -1)
        current_seg_idx = seg_indices_masked.max(dim=1).values  # (n,)

        # Clamp to valid range [0, num_segs-1]
        current_seg_idx = torch.clamp(current_seg_idx, min=0, max=num_segs.max() - 1)
        # Per-env clamping to account for different num_segments
        current_seg_idx = torch.minimum(current_seg_idx, num_segs - 1)

        # 2. Gather target poses, segment types, and start poses (batched advanced indexing)
        batch_range = torch.arange(n, device=self.device)
        target_pos = self.segment_poses[env_ids, current_seg_idx, :3]  # (n, 3)
        target_quat = self.segment_poses[env_ids, current_seg_idx, 3:7]  # (n, 4)
        is_helical = self.segment_is_helical[env_ids, current_seg_idx]  # (n,)

        # Get segment start poses (for helical rotation calculation)
        # If seg_idx == 0: use start_poses, else use previous segment's end pose
        is_first_seg = (current_seg_idx == 0)
        seg_start_poses = torch.zeros(n, 7, device=self.device)
        if is_first_seg.any():
            seg_start_poses[is_first_seg] = self.start_poses[env_ids[is_first_seg]]
        if (~is_first_seg).any():
            prev_seg_idx = (current_seg_idx[~is_first_seg] - 1).clamp(min=0)
            seg_start_poses[~is_first_seg] = self.segment_poses[env_ids[~is_first_seg], prev_seg_idx]

        # 3. Compute remaining targets and horizons (vectorized)
        num_remaining = torch.clamp(manip_end_idx - current_idx, min=1)  # (n,)
        horizon = torch.minimum(
            torch.full_like(num_remaining, rolling_window_targets),
            num_remaining
        )  # (n,)

        # 4. Calculate distances and speeds (vectorized)
        distance_to_target = torch.norm(target_pos - current_pos, dim=1)  # (n,)
        remaining_time = num_remaining.float() * self.target_dt  # (n,)
        min_speed = self.cfg.push_t.min_speed

        # Avoid division by zero
        safe_remaining_time = torch.where(remaining_time > 0, remaining_time, torch.ones_like(remaining_time))
        natural_speed = distance_to_target / safe_remaining_time  # (n,)

        # Determine actual_targets based on speed constraint
        # HELICAL segments: ignore min_speed, always use horizon (maintain coupled motion)
        # WAYPOINT segments: apply min_speed constraint for catch-up
        slow_mask = (natural_speed < min_speed) & (distance_to_target > 1e-4) & ~is_helical  # (n,)
        actual_duration = distance_to_target / min_speed  # (n,)
        speed_based_targets = torch.clamp((actual_duration / self.target_dt).long() + 1, min=1, max=horizon.max())  # (n,)

        actual_targets = torch.where(slow_mask, speed_based_targets, horizon)  # (n,)

        # 5. Generate trajectories with variable lengths (padded batching)
        max_targets = actual_targets.max().item()

        if max_targets > 0:
            # Create progress array for all envs (padded to max_targets)
            t_steps = torch.linspace(0, 1, max_targets, device=self.device)  # (max_targets,)
            valid_mask = torch.arange(max_targets, device=self.device).unsqueeze(0) < actual_targets.unsqueeze(1)  # (n, max_targets)

            # Expand start/end poses
            start_pos_expanded = current_pos.unsqueeze(1).expand(n, max_targets, 3)  # (n, max_targets, 3)
            target_pos_expanded = target_pos.unsqueeze(1).expand(n, max_targets, 3)  # (n, max_targets, 3)
            start_quat_expanded = current_quat.unsqueeze(1).expand(n, max_targets, 4)  # (n, max_targets, 4)
            target_quat_expanded = target_quat.unsqueeze(1).expand(n, max_targets, 4)  # (n, max_targets, 4)

            # Position and orientation: split by segment type
            positions = torch.zeros(n, max_targets, 3, device=self.device)
            orientations = torch.zeros(n, max_targets, 4, device=self.device)

            # WAYPOINT segments: linear interpolation from current to target
            waypoint_mask = ~is_helical
            if waypoint_mask.any():
                positions[waypoint_mask] = (start_pos_expanded[waypoint_mask] +
                                           t_steps.unsqueeze(0).unsqueeze(-1) *
                                           (target_pos_expanded[waypoint_mask] - start_pos_expanded[waypoint_mask]))

            # Waypoint segments: SLERP from current to target
            waypoint_mask = ~is_helical
            if waypoint_mask.any():
                wp_count = waypoint_mask.sum().item()
                wp_indices = torch.where(waypoint_mask)[0]
                start_flat = start_quat_expanded[wp_indices].reshape(wp_count * max_targets, 4)
                target_flat = target_quat_expanded[wp_indices].reshape(wp_count * max_targets, 4)
                t_flat = t_steps.unsqueeze(0).expand(wp_count, max_targets).reshape(wp_count * max_targets)
                orientations_flat = self._batched_slerp(start_flat, target_flat, t_flat)
                orientations[wp_indices] = orientations_flat.reshape(wp_count, max_targets, 4)

            # Helical segments: position to target, rotation with alignment correction + helical component
            # During replanning: interpolate position AND non-helical rotation to target, maintain min helical rotation
            if is_helical.any():
                hel_indices = torch.where(is_helical)[0]
                n_hel = len(hel_indices)

                # Get helical parameters
                axes = self.segment_helical_axis[env_ids[hel_indices], current_seg_idx[hel_indices]]  # (n_hel, 3)
                axes = axes / torch.norm(axes, dim=1, keepdim=True).clamp(min=1e-8)  # Normalize

                # Current state and targets
                current_pos_hel = current_pos[hel_indices]  # (n_hel, 3)
                current_quat_hel = current_quat[hel_indices]  # (n_hel, 4)
                target_pos_hel = target_pos[hel_indices]  # (n_hel, 3)
                target_quat_hel = target_quat[hel_indices]  # (n_hel, 4)

                # Minimum angular velocity
                min_angular_vel = self.cfg.push_t.min_angular_velocity

                # Generate trajectory for planning window
                for t_idx in range(max_targets):
                    # Time step and progress
                    dt = t_idx * self.target_dt  # (scalar)
                    progress = t_steps[t_idx]  # (scalar) [0, 1]

                    # Position: linear interpolation from current to ACTUAL target
                    positions[hel_indices, t_idx] = current_pos_hel + progress * (target_pos_hel - current_pos_hel)

                    # Rotation: SLERP to target (fixes all axes) + extra helical rotation
                    # Step 1: SLERP from current to target orientation (fixes misalignment in all axes)
                    base_quat = self._batched_slerp(current_quat_hel, target_quat_hel, progress)

                    # Step 2: Add additional rotation around helical axis at min angular velocity
                    angle_delta = torch.full((n_hel,), min_angular_vel * dt, device=self.device)  # (n_hel,)

                    # Convert to quaternion
                    delta_quats = torch.zeros(n_hel, 4, device=self.device)
                    delta_quats[:, 0] = torch.cos(angle_delta / 2)
                    delta_quats[:, 1:4] = axes * torch.sin(angle_delta / 2).unsqueeze(1)

                    # Apply extra helical rotation: final_quat = delta_quat * base_quat
                    orientations[hel_indices, t_idx] = quat_mul(delta_quats, base_quat)

            # 6. Scatter write to trajectory (only valid timesteps)
            write_indices = current_idx.unsqueeze(1) + torch.arange(max_targets, device=self.device).unsqueeze(0)  # (n, max_targets)
            write_indices = torch.clamp(write_indices, max=self.total_targets - 1)
            write_mask = valid_mask & (write_indices < self.total_targets)  # (n, max_targets)

            env_idx_flat = env_ids.unsqueeze(1).expand(n, max_targets)[write_mask]
            time_idx_flat = write_indices[write_mask]
            pos_flat = positions[write_mask]
            quat_flat = orientations[write_mask]

            self.trajectory[env_idx_flat, time_idx_flat, :3] = pos_flat
            self.trajectory[env_idx_flat, time_idx_flat, 3:7] = quat_flat

        # 7. Fill remaining manipulation indices with segment target pose (vectorized)
        fill_start = current_idx + actual_targets  # (n,)
        fill_end = torch.clamp(manip_end_idx, max=self.total_targets)  # (n,)
        needs_fill = fill_start < fill_end  # (n,)

        if needs_fill.any():
            fill_envs = env_ids[needs_fill]
            fill_starts = fill_start[needs_fill]
            fill_ends = fill_end[needs_fill]
            fill_target_pos = target_pos[needs_fill]
            fill_target_quat = target_quat[needs_fill]

            max_fill_len = (fill_ends - fill_starts).max().item()
            fill_range = torch.arange(max_fill_len, device=self.device).unsqueeze(0)  # (1, max_fill_len)
            fill_mask = fill_range < (fill_ends - fill_starts).unsqueeze(1)  # (num_fill, max_fill_len)

            fill_time_idx = fill_starts.unsqueeze(1) + fill_range  # (num_fill, max_fill_len)
            fill_env_idx = fill_envs.unsqueeze(1).expand(-1, max_fill_len)  # (num_fill, max_fill_len)

            fill_env_flat = fill_env_idx[fill_mask]
            fill_time_flat = fill_time_idx[fill_mask]
            fill_pos_flat = fill_target_pos.unsqueeze(1).expand(-1, max_fill_len, 3)[fill_mask]
            fill_quat_flat = fill_target_quat.unsqueeze(1).expand(-1, max_fill_len, 4)[fill_mask]

            self.trajectory[fill_env_flat, fill_time_flat, :3] = fill_pos_flat
            self.trajectory[fill_env_flat, fill_time_flat, 3:7] = fill_quat_flat

    def _update_hand_trajectory_manipulation(self, env_ids: torch.Tensor):
        """Update hand trajectory after object replanning.

        Respects coupling modes:
        - full (0): Hand position and orientation follow object
        - position_only (1): Hand position follows, orientation frozen at segment start
        - none (2): Should not be called during manipulation phase

        Args:
            env_ids: Environment indices to update.
        """
        from isaaclab.utils.math import quat_mul, quat_apply

        n = len(env_ids)
        if n == 0:
            return

        # Get per-env manipulation end index (do NOT overwrite release phase)
        manip_end_idx = (self.t_manip_end[env_ids] / self.target_dt).long()  # (n,)
        current_idx = self.current_idx[env_ids]  # (n,)

        # Get grasp pose relative to object (stored at reset)
        grasp_pos_rel = self.grasp_pose[env_ids, :3]  # (n, 3)
        grasp_quat_rel = self.grasp_pose[env_ids, 3:7]  # (n, 4)

        # FULLY VECTORIZED: Update all envs at once - NO LOOP!
        # Clamp manipulation end indices to trajectory bounds
        manip_end_idx = torch.clamp(manip_end_idx, max=self.total_targets)

        # Find max length needed for batching
        max_len = (manip_end_idx - current_idx).max().item()

        if max_len <= 0:
            return

        # Build batch indices: range_mask determines valid timesteps for each env
        range_mask = torch.arange(max_len, device=self.device).unsqueeze(0) < (manip_end_idx - current_idx).unsqueeze(1)  # (n, max_len)
        time_indices = current_idx.unsqueeze(1) + torch.arange(max_len, device=self.device).unsqueeze(0)  # (n, max_len)
        time_indices = torch.clamp(time_indices, max=self.total_targets - 1)  # Prevent out-of-bounds

        # Gather object trajectories (batched)
        env_range = env_ids.unsqueeze(1).expand(n, max_len)  # (n, max_len)
        obj_pos = self.trajectory[env_range, time_indices, :3]  # (n, max_len, 3)
        obj_quat = self.trajectory[env_range, time_indices, 3:7]  # (n, max_len, 4)

        # Expand grasp poses to match all timesteps
        grasp_pos_expanded = grasp_pos_rel.unsqueeze(1).expand(n, max_len, 3)  # (n, max_len, 3)
        grasp_quat_expanded = grasp_quat_rel.unsqueeze(1).expand(n, max_len, 4)  # (n, max_len, 4)

        # Batched transforms: flatten, transform, reshape
        obj_pos_flat = obj_pos.reshape(n * max_len, 3)
        obj_quat_flat = obj_quat.reshape(n * max_len, 4)
        grasp_pos_flat = grasp_pos_expanded.reshape(n * max_len, 3)
        grasp_quat_flat = grasp_quat_expanded.reshape(n * max_len, 4)

        # Position ALWAYS follows object (for both full and position_only coupling)
        hand_pos_flat = quat_apply(obj_quat_flat, grasp_pos_flat) + obj_pos_flat  # (n*max_len, 3)
        hand_pos = hand_pos_flat.reshape(n, max_len, 3)

        # Orientation depends on coupling mode
        # Need to determine which segment each timestep belongs to
        times = time_indices.float() * self.target_dt  # (n, max_len)

        # Find current segment index for each timestep
        # For each (env, timestep), find which segment it's in
        max_segs = self.segment_boundaries.shape[1] - 1
        boundaries = self.segment_boundaries[env_ids]  # (n, max_segs+1)

        # Expand for comparison: (n, max_len, max_segs+1)
        times_expanded = times.unsqueeze(2)  # (n, max_len, 1)
        boundaries_expanded = boundaries.unsqueeze(1)  # (n, 1, max_segs+1)

        # Find which boundary each time has passed
        after_start = times_expanded >= boundaries_expanded  # (n, max_len, max_segs+1)

        # Find rightmost True (last boundary passed)
        seg_indices_range = torch.arange(max_segs + 1, device=self.device).view(1, 1, -1)  # (1, 1, max_segs+1)
        seg_indices_masked = torch.where(after_start, seg_indices_range, -1)
        current_seg_per_timestep = seg_indices_masked.max(dim=2).values  # (n, max_len)
        current_seg_per_timestep = torch.clamp(current_seg_per_timestep, min=0, max=max_segs-1)

        # Clamp to valid range per env
        num_segs_expanded = self.num_segments[env_ids].unsqueeze(1).expand(n, max_len)  # (n, max_len)
        current_seg_per_timestep = torch.minimum(current_seg_per_timestep, num_segs_expanded - 1)

        # Get coupling mode for each timestep
        env_idx_flat = env_range.reshape(-1)  # (n*max_len,)
        seg_idx_flat = current_seg_per_timestep.reshape(-1)  # (n*max_len,)
        coupling_modes_flat = self.coupling_modes[env_idx_flat, seg_idx_flat]  # (n*max_len,)

        # Apply coupling modes
        hand_quat = torch.zeros(n, max_len, 4, device=self.device)

        # FULL coupling (mode 0): orientation follows object
        full_mask = (coupling_modes_flat == 0)
        if full_mask.any():
            hand_quat_full = quat_mul(obj_quat_flat[full_mask], grasp_quat_flat[full_mask])
            hand_quat.reshape(n * max_len, 4)[full_mask] = hand_quat_full

        # POSITION_ONLY coupling (mode 1): orientation frozen at segment start
        # Use pre-computed hand_pose_at_segment_boundary buffer
        pos_only_mask = (coupling_modes_flat == 1)
        if pos_only_mask.any():
            pos_only_env_idx = env_idx_flat[pos_only_mask]
            pos_only_seg_idx = seg_idx_flat[pos_only_mask]

            # Get hand orientation at segment start (stored during initial trajectory generation)
            # hand_pose_at_segment_boundary[env, K] = hand pose at time when segment K starts
            frozen_quat = self.hand_pose_at_segment_boundary[pos_only_env_idx, pos_only_seg_idx, 3:7]  # (num_pos_only, 4)

            hand_quat.reshape(n * max_len, 4)[pos_only_mask] = frozen_quat

        # Scatter write using mask (only write valid timesteps)
        valid_env_idx = env_range[range_mask]  # (num_valid,)
        valid_time_idx = time_indices[range_mask]  # (num_valid,)
        valid_hand_pos = hand_pos[range_mask]  # (num_valid, 3)
        valid_hand_quat = hand_quat[range_mask]  # (num_valid, 4)

        self.hand_trajectory[valid_env_idx, valid_time_idx, :3] = valid_hand_pos
        self.hand_trajectory[valid_env_idx, valid_time_idx, 3:7] = valid_hand_quat

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
        """FULLY VECTORIZED: Get current phase index for each environment (segment-based).

        Returns:
            0 = grasp (hand approaches object)
            1 = manipulate (hand moves with object)
            2 = release (hand retreats, fingers open)

        Maps segment names to phase indices:
        - "grasp" → phase 0
        - "release" → phase 2
        - everything else → phase 1 (manipulation)
        """
        t = self.phase_time  # (num_envs,)
        phase = torch.ones(self.num_envs, dtype=torch.long, device=self.device)  # Default: manipulation

        # Find current segment index for ALL envs using vectorized searchsorted
        max_segs_plus_one = self.segment_boundaries.shape[1]

        # Create mask: (num_envs, max_segs+1) - True where t >= boundary
        t_expanded = t.unsqueeze(1)  # (num_envs, 1)
        after_start = t_expanded >= self.segment_boundaries  # (num_envs, max_segs+1)

        # Find LAST True (rightmost boundary we've passed)
        # Boundaries: [0, end_seg0, end_seg1, ..., end_segN] (max_segs+1 entries)
        # If we're between boundaries[K] and boundaries[K+1], we're IN segment K
        # The rightmost boundary we've passed IS the current segment index
        seg_indices = torch.arange(max_segs_plus_one, device=self.device).unsqueeze(0).expand(self.num_envs, -1)
        seg_indices_masked = torch.where(after_start, seg_indices, torch.full_like(seg_indices, -1))
        current_seg_idx = seg_indices_masked.max(dim=1).values  # (num_envs,)

        # Clamp to valid segment range [0, num_segs-1]
        # If we're past all boundaries (idx >= num_segs), clamp to last segment
        current_seg_idx = current_seg_idx.clamp(min=0)
        max_segs_tensor = self.segment_boundaries.shape[1] - 1  # max_segs
        current_seg_idx = current_seg_idx.clamp(max=max_segs_tensor - 1)
        current_seg_idx = torch.minimum(current_seg_idx, self.num_segments - 1)

        # Gather segment type flags for current segments
        env_range = torch.arange(self.num_envs, device=self.device)
        is_grasp_current = self.segment_is_grasp[env_range, current_seg_idx]  # (num_envs,)
        is_release_current = self.segment_is_release[env_range, current_seg_idx]  # (num_envs,)

        # Map to phase indices (vectorized)
        phase = torch.where(is_grasp_current, 0, phase)
        phase = torch.where(is_release_current, 2, phase)
        # Remaining are already set to 1 (manipulation)

        return phase

    def is_in_release_phase(self) -> torch.Tensor:
        """Check if environments are in release phase (includes settle + retreat)."""
        return self.get_phase() == 2
    
    def is_in_grasp_phase(self) -> torch.Tensor:
        """Check if environments are in grasp phase."""
        return self.get_phase() == 0
