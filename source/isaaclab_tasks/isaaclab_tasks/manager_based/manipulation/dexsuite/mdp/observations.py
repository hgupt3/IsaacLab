# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import (
    quat_apply,
    quat_apply_inverse,
    quat_error_magnitude,
    quat_inv,
    quat_mul,
    subtract_frame_transforms,
)

from .utils import sample_object_point_cloud, sample_object_point_cloud_random

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from ..trajectory_manager import TrajectoryManager
    from ..trajectory_cfg import TrajectoryParamsCfg


def zeros_placeholder(
    env: ManagerBasedRLEnv,
    size: int = 7,
) -> torch.Tensor:
    """Placeholder observation that returns zeros.
    
    Useful for maintaining checkpoint compatibility when observations are removed.
    
    Args:
        env: The environment.
        size: Size of the zeros tensor. Defaults to 7 (pose size).
    
    Returns:
        Tensor of shape (num_envs, size) filled with zeros.
    """
    return torch.zeros(env.num_envs, size, device=env.device)


def object_pos_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """Object position in the robot's root frame.

    Args:
        env: The environment.
        robot_cfg: Scene entity for the robot (reference frame). Defaults to ``SceneEntityCfg("robot")``.
        object_cfg: Scene entity for the object. Defaults to ``SceneEntityCfg("object")``.

    Returns:
        Tensor of shape ``(num_envs, 3)``: object position [x, y, z] expressed in the robot root frame.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    return quat_apply_inverse(robot.data.root_quat_w, object.data.root_pos_w - robot.data.root_pos_w)


def object_quat_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object orientation in the robot's root frame.

    Args:
        env: The environment.
        robot_cfg: Scene entity for the robot (reference frame). Defaults to ``SceneEntityCfg("robot")``.
        object_cfg: Scene entity for the object. Defaults to ``SceneEntityCfg("object")``.

    Returns:
        Tensor of shape ``(num_envs, 4)``: object quaternion ``(w, x, y, z)`` in the robot root frame.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    return quat_mul(quat_inv(robot.data.root_quat_w), object.data.root_quat_w)


def object_pose_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object pose (position + quaternion) in the robot's root frame.

    Combines object_pos_b and object_quat_b into a single 7-dim observation.
    Used in pose mode instead of point cloud observations.

    Args:
        env: The environment.
        robot_cfg: Scene entity for the robot (reference frame). Defaults to ``SceneEntityCfg("robot")``.
        object_cfg: Scene entity for the object. Defaults to ``SceneEntityCfg("object")``.

    Returns:
        Tensor of shape ``(num_envs, 7)``: [x, y, z, qw, qx, qy, qz] in robot root frame.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    
    # Position in robot frame
    pos_b = quat_apply_inverse(robot.data.root_quat_w, object.data.root_pos_w - robot.data.root_pos_w)
    
    # Orientation in robot frame
    quat_b = quat_mul(quat_inv(robot.data.root_quat_w), object.data.root_quat_w)
    
    return torch.cat([pos_b, quat_b], dim=-1)


def body_state_b(
    env: ManagerBasedRLEnv,
    body_asset_cfg: SceneEntityCfg,
    base_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Body state (pos, quat, lin vel, ang vel) in the base asset's root frame.

    The state for each body is stacked horizontally as
    ``[position(3), quaternion(4)(wxyz), linvel(3), angvel(3)]`` and then concatenated over bodies.

    Args:
        env: The environment.
        body_asset_cfg: Scene entity for the articulated body whose links are observed.
        base_asset_cfg: Scene entity providing the reference (root) frame.

    Returns:
        Tensor of shape ``(num_envs, num_bodies * 13)`` with per-body states expressed in the base root frame.
    """
    body_asset: Articulation = env.scene[body_asset_cfg.name]
    base_asset: Articulation = env.scene[base_asset_cfg.name]
    # get world pose of bodies
    body_pos_w = body_asset.data.body_pos_w[:, body_asset_cfg.body_ids].view(-1, 3)
    body_quat_w = body_asset.data.body_quat_w[:, body_asset_cfg.body_ids].view(-1, 4)
    body_lin_vel_w = body_asset.data.body_lin_vel_w[:, body_asset_cfg.body_ids].view(-1, 3)
    body_ang_vel_w = body_asset.data.body_ang_vel_w[:, body_asset_cfg.body_ids].view(-1, 3)
    num_bodies = int(body_pos_w.shape[0] / env.num_envs)
    # get world pose of base frame
    root_pos_w = base_asset.data.root_link_pos_w.unsqueeze(1).repeat_interleave(num_bodies, dim=1).view(-1, 3)
    root_quat_w = base_asset.data.root_link_quat_w.unsqueeze(1).repeat_interleave(num_bodies, dim=1).view(-1, 4)
    # transform from world body pose to local body pose
    body_pos_b, body_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, body_pos_w, body_quat_w)
    body_lin_vel_b = quat_apply_inverse(root_quat_w, body_lin_vel_w)
    body_ang_vel_b = quat_apply_inverse(root_quat_w, body_ang_vel_w)
    # concate and return
    out = torch.cat((body_pos_b, body_quat_b, body_lin_vel_b, body_ang_vel_b), dim=1)
    return out.view(env.num_envs, -1)


class object_point_cloud_b(ManagerTermBase):
    """Object surface point cloud expressed in a reference asset's root frame.

    Uses RANDOM surface sampling and caches points on env._object_points_local
    for sharing with target_sequence_point_clouds_b.

    Args (from ``cfg.params``):
        object_cfg: Scene entity for the object to sample. Defaults to ``SceneEntityCfg("object")``.
        ref_asset_cfg: Scene entity providing the reference frame. Defaults to ``SceneEntityCfg("robot")``.
        num_points: Number of points to sample on the object surface. Defaults to ``10``.
        visualize: Whether to draw markers for the points. Defaults to ``True``.

    Returns (from ``__call__``):
        If ``flatten=False``: tensor of shape ``(num_envs, num_points, 3)``.
        If ``flatten=True``: tensor of shape ``(num_envs, 3 * num_points)``.
    
    Note:
        Caches points_local on env._object_points_local for use by target_sequence_point_clouds_b.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.object_cfg: SceneEntityCfg = cfg.params.get("object_cfg", SceneEntityCfg("object"))
        self.ref_asset_cfg: SceneEntityCfg = cfg.params.get("ref_asset_cfg", SceneEntityCfg("robot"))
        self.num_points: int = cfg.params.get("num_points", 10)
        self.object: RigidObject = env.scene[self.object_cfg.name]
        self.ref_asset: Articulation = env.scene[self.ref_asset_cfg.name]
        
        # Visualizer
        self.visualizer = None
        self.visualize_env_ids = cfg.params.get("visualize_env_ids", [0])  # Default: only env 0
        if cfg.params.get("visualize", True):
            from isaaclab.markers import VisualizationMarkers
            from isaaclab.markers.config import RAY_CASTER_MARKER_CFG
            ray_cfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/ObservationPointCloud")
            ray_cfg.markers["hit"].radius = 0.0025
            self.visualizer = VisualizationMarkers(ray_cfg)
        
        # Sample points using RANDOM sampling (not FPS) for consistency with targets
        all_env_ids = list(range(env.num_envs))
        self.points_local = sample_object_point_cloud_random(
            all_env_ids, self.num_points, self.object.cfg.prim_path, device=env.device
        )
        self.points_w = torch.zeros_like(self.points_local)
        
        # Cache on env for target_sequence_point_clouds_b to use the SAME points
        env._object_points_local = self.points_local

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        ref_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        num_points: int = 10,
        flatten: bool = False,
        visualize: bool = True,
    ):
        """Compute the object point cloud in the reference asset's root frame.

        Args:
            env: The environment.
            ref_asset_cfg: Reference frame provider (root). Defaults to ``SceneEntityCfg("robot")``.
            object_cfg: Object to sample. Defaults to ``SceneEntityCfg("object")``.
            num_points: Unused at runtime (set at init).
            flatten: If ``True``, return a flattened tensor ``(num_envs, 3 * num_points)``.
            visualize: If ``True``, draw markers for the points.

        Returns:
            Tensor of shape ``(num_envs, num_points, 3)`` or flattened if requested.
        """
        P = self.num_points
        ref_pos_w = self.ref_asset.data.root_pos_w.unsqueeze(1).expand(-1, P, -1)
        ref_quat_w = self.ref_asset.data.root_quat_w.unsqueeze(1).expand(-1, P, -1)

        object_pos_w = self.object.data.root_pos_w.unsqueeze(1).expand(-1, P, -1)
        object_quat_w = self.object.data.root_quat_w.unsqueeze(1).expand(-1, P, -1)
        
        # Transform local points to world
        self.points_w = quat_apply(object_quat_w, self.points_local) + object_pos_w
        
        if visualize and self.visualizer is not None:
            if self.visualize_env_ids is not None:
                # Only visualize points for specified env IDs
                vis_points = self.points_w[self.visualize_env_ids].reshape(-1, 3)
            else:
                vis_points = self.points_w.view(-1, 3)
            self.visualizer.visualize(translations=vis_points)
        
        # Transform to robot base frame
        object_point_cloud_pos_b, _ = subtract_frame_transforms(ref_pos_w, ref_quat_w, self.points_w, None)

        return object_point_cloud_pos_b.view(env.num_envs, -1) if flatten else object_point_cloud_pos_b


def fingers_contact_force_b(
    env: ManagerBasedRLEnv,
    contact_sensor_names: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """base-frame contact forces from listed sensors, concatenated per env.

    Args:
        env: The environment.
        contact_sensor_names: Names of contact sensors in ``env.scene.sensors`` to read.

    Returns:
        Tensor of shape ``(num_envs, 3 * num_sensors)`` with forces stacked horizontally as
        ``[fx, fy, fz]`` per sensor.
    """
    force_w = [env.scene.sensors[name].data.force_matrix_w.view(env.num_envs, 3) for name in contact_sensor_names]
    force_w = torch.stack(force_w, dim=1)
    robot: Articulation = env.scene[asset_cfg.name]
    forces_b = quat_apply_inverse(robot.data.root_link_quat_w.unsqueeze(1).repeat(1, force_w.shape[1], 1), force_w)
    return forces_b


class target_sequence_obs_b(ManagerTermBase):
    """Trajectory target observations expressed in robot's base frame.
    
    Supports two modes based on trajectory_cfg.use_point_cloud:
    - Point cloud mode (default): Transforms object point cloud to each target pose.
      Returns (num_envs, window_size * num_points * 3).
    - Pose mode: Returns target poses directly.
      Returns (num_envs, window_size * 7).
    
    Also caches errors for reward/termination functions:
    - Point cloud mode: env._cached_mean_errors (N, W) - point-to-point mean error
    - Pose mode: env._cached_pose_errors dict with 'pos' (N, W) and 'rot' (N, W)
    
    Args (from ``cfg.params``):
        object_cfg: Scene entity for the object. Defaults to ``SceneEntityCfg("object")``.
        ref_asset_cfg: Reference frame (robot). Defaults to ``SceneEntityCfg("robot")``.
        trajectory_cfg: TrajectoryParamsCfg with mode settings.
    
    Returns:
        Point cloud mode: Tensor of shape ``(num_envs, window_size * num_points * 3)``.
        Pose mode: Tensor of shape ``(num_envs, window_size * 7)``.
    """
    
    # Inferno colormap (dark purple → bright yellow)
    INFERNO_COLORS = [
        (0.07, 0.04, 0.11),  # 0: nearly black
        (0.32, 0.06, 0.38),  # 1: dark purple
        (0.55, 0.09, 0.38),  # 2: magenta
        (0.75, 0.19, 0.27),  # 3: red-purple
        (0.89, 0.35, 0.13),  # 4: orange-red
        (0.96, 0.55, 0.04),  # 5: orange
        (0.99, 0.77, 0.17),  # 6: yellow-orange
        (0.99, 0.99, 0.64),  # 7: bright yellow
    ]
    CURRENT_OBJECT_COLOR = (1.0, 0.0, 0.0)  # Full red for current object
    POINT_RADIUS = 0.0025
    
    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        
        # Import here to avoid circular import
        from ..trajectory_manager import TrajectoryManager
        from ..trajectory_cfg import TrajectoryParamsCfg
        
        # Get trajectory config (all settings come from here)
        traj_cfg: TrajectoryParamsCfg = cfg.params.get("trajectory_cfg", TrajectoryParamsCfg())
        
        self.object_cfg: SceneEntityCfg = cfg.params.get("object_cfg", SceneEntityCfg("object"))
        self.ref_asset_cfg: SceneEntityCfg = cfg.params.get("ref_asset_cfg", SceneEntityCfg("robot"))
        
        # Use values from trajectory config
        self.traj_cfg = traj_cfg
        self.use_point_cloud = traj_cfg.use_point_cloud
        self.num_points = traj_cfg.num_points
        self.window_size = traj_cfg.window_size
        self.visualize = traj_cfg.visualize_targets
        self.visualize_current = traj_cfg.visualize_current
        self.visualize_waypoint_region = traj_cfg.visualize_waypoint_region
        self.visualize_goal_region = traj_cfg.visualize_goal_region
        self.visualize_pose_axes = traj_cfg.visualize_pose_axes
        self.visualize_env_ids = traj_cfg.visualize_env_ids  # None = all envs
        
        self.object: RigidObject = env.scene[self.object_cfg.name]
        self.ref_asset: Articulation = env.scene[self.ref_asset_cfg.name]
        self.env = env
        self.dt = env.step_dt
        
        # Create trajectory manager and attach to env for other components to access
        self.trajectory_manager = TrajectoryManager(
            cfg=traj_cfg,
            num_envs=env.num_envs,
            device=env.device,
            table_height=traj_cfg.table_surface_z,
            object_prim_path=self.object.cfg.prim_path,
        )
        env.trajectory_manager = self.trajectory_manager
        
        # Point cloud initialization (always needed for visualization)
        self.points_local = None
        # Use cached points from object_point_cloud_b (must be initialized first)
        # This ensures SAME local points are used for both current object and targets
        if hasattr(env, '_object_points_local'):
            self.points_local = env._object_points_local
        else:
            # Fallback if object_point_cloud_b not initialized yet
            all_env_ids = list(range(env.num_envs))
            self.points_local = sample_object_point_cloud_random(
                all_env_ids, self.num_points, self.object.cfg.prim_path, device=env.device
            )
            env._object_points_local = self.points_local
        
        # Pass point cloud to trajectory manager for penetration checking
        self.trajectory_manager.points_local = self.points_local
        
        # Visualizers
        self.markers = None
        self._debug_draw = None
        self._visualizer_initialized = False
        if self.visualize:
            self._init_visualizer()
        if self.visualize_waypoint_region or self.visualize_goal_region or self.visualize_pose_axes:
            self._init_region_visualizer()
    
    def _init_visualizer(self):
        """Initialize visualization markers for targets + current object."""
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        
        # Target markers (Inferno colors) + current object marker (red)
        markers_dict = {
            f"target_{i}": sim_utils.SphereCfg(
                radius=self.POINT_RADIUS,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=self.INFERNO_COLORS[i]),
            )
            for i in range(self.window_size)
        }
        markers_dict["current"] = sim_utils.SphereCfg(
            radius=self.POINT_RADIUS,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=self.CURRENT_OBJECT_COLOR,
            ),
        )
        
        self.markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/TargetSequencePointClouds",
                markers=markers_dict,
            )
        )
    
    def _init_region_visualizer(self):
        """Initialize debug draw for waypoint/goal region visualization."""
        try:
            # Try different import paths for different Isaac Sim versions
            try:
                import isaacsim.util.debug_draw._debug_draw as omni_debug_draw
            except ModuleNotFoundError:
                try:
                    from omni.isaac.debug_draw import _debug_draw as omni_debug_draw
                except ModuleNotFoundError:
                    from omni.debugdraw import get_debug_draw_interface
                    self._debug_draw = get_debug_draw_interface()
                    return
            self._debug_draw = omni_debug_draw.acquire_debug_draw_interface()
        except Exception as e:
            print(f"[WARN] Could not initialize debug draw: {e}. Region visualization disabled.")
            self._debug_draw = None
    
    def _visualize_regions(self, env_origins: torch.Tensor, start_pos: torch.Tensor):
        """Visualize waypoint and goal sampling regions using debug lines."""
        if self._debug_draw is None:
            return
        
        cfg = self.traj_cfg
        device = env_origins.device
        N = env_origins.shape[0]
        
        # Select envs to visualize
        if self.visualize_env_ids is None:
            env_ids = torch.arange(N, device=device)
        else:
            env_ids = torch.tensor([i for i in self.visualize_env_ids if i < N], device=device)
        E = len(env_ids)
        if E == 0:
            return
        
        origins = env_origins[env_ids]  # (E, 3)
        starts = start_pos[env_ids]     # (E, 3)
        
        # Note: clear_lines is done in _visualize_pose_axes which runs every frame
        
        # Waypoint region: 3D box (12 edges)
        if self.visualize_waypoint_region:
            start_local = starts - origins
            
            # X bounds (None = workspace, else offset from start)
            if cfg.waypoint_position_range_x is None:
                x_min = torch.full((E,), cfg.workspace_x[0], device=device)
                x_max = torch.full((E,), cfg.workspace_x[1], device=device)
            else:
                x_min = torch.clamp(start_local[:, 0] + cfg.waypoint_position_range_x[0], min=cfg.workspace_x[0])
                x_max = torch.clamp(start_local[:, 0] + cfg.waypoint_position_range_x[1], max=cfg.workspace_x[1])
            
            # Y bounds
            if cfg.waypoint_position_range_y is None:
                y_min = torch.full((E,), cfg.workspace_y[0], device=device)
                y_max = torch.full((E,), cfg.workspace_y[1], device=device)
            else:
                y_min = torch.clamp(start_local[:, 1] + cfg.waypoint_position_range_y[0], min=cfg.workspace_y[0])
                y_max = torch.clamp(start_local[:, 1] + cfg.waypoint_position_range_y[1], max=cfg.workspace_y[1])
            
            # Z bounds
            if cfg.waypoint_position_range_z is None:
                z_min = torch.full((E,), cfg.workspace_z_min, device=device)
                z_max = torch.full((E,), cfg.workspace_z_max, device=device)
            else:
                z_min = torch.full((E,), max(cfg.waypoint_position_range_z[0], cfg.workspace_z_min), device=device)
                z_max = torch.full((E,), min(cfg.waypoint_position_range_z[1], cfg.workspace_z_max), device=device)
            
            wp_min = torch.stack([x_min, y_min, z_min], dim=-1) + origins
            wp_max = torch.stack([x_max, y_max, z_max], dim=-1) + origins
            
            starts_wp, ends_wp = self._box_edges_lines(wp_min, wp_max)
            color_wp = (0.2, 0.9, 0.2, 1.0)  # Green
            self._draw_lines(starts_wp, ends_wp, color_wp)
        
        # Goal region: 2D rectangle on table (4 edges)
        if self.visualize_goal_region:
            x_range = cfg.goal_x_range if cfg.goal_x_range is not None else cfg.workspace_x
            y_range = cfg.goal_y_range if cfg.goal_y_range is not None else cfg.workspace_y
            
            g_min = origins[:, :2] + torch.tensor([[x_range[0], y_range[0]]], device=device)
            g_max = origins[:, :2] + torch.tensor([[x_range[1], y_range[1]]], device=device)
            g_z = cfg.table_surface_z + origins[:, 2] + 0.01
            
            starts_g, ends_g = self._rect_edges_lines(g_min, g_max, g_z)
            color_g = (0.2, 0.9, 0.9, 1.0)  # Cyan
            self._draw_lines(starts_g, ends_g, color_g)
    
    def _draw_lines(self, starts: torch.Tensor, ends: torch.Tensor, color: tuple):
        """Draw lines using debug draw interface."""
        # Non-blocking transfer + direct tolist (skip numpy intermediate)
        starts_list = starts.cpu().tolist()
        ends_list = ends.cpu().tolist()
        n = len(starts_list)
        self._debug_draw.draw_lines(starts_list, ends_list, [color] * n, [2.0] * n)
    
    def _box_edges_lines(self, mins: torch.Tensor, maxs: torch.Tensor):
        """Get start/end points for 12 edges of 3D boxes. Returns (E*12, 3) each."""
        E = mins.shape[0]
        x0, y0, z0 = mins[:, 0], mins[:, 1], mins[:, 2]
        x1, y1, z1 = maxs[:, 0], maxs[:, 1], maxs[:, 2]
        
        # 8 vertices of box: (E, 8, 3)
        verts = torch.stack([
            torch.stack([x0, y0, z0], dim=-1),  # 0: min corner
            torch.stack([x1, y0, z0], dim=-1),  # 1
            torch.stack([x0, y1, z0], dim=-1),  # 2
            torch.stack([x1, y1, z0], dim=-1),  # 3
            torch.stack([x0, y0, z1], dim=-1),  # 4
            torch.stack([x1, y0, z1], dim=-1),  # 5
            torch.stack([x0, y1, z1], dim=-1),  # 6
            torch.stack([x1, y1, z1], dim=-1),  # 7: max corner
        ], dim=1)  # (E, 8, 3)
        
        # 12 edges: pairs of vertex indices
        edge_pairs = [(0,1), (2,3), (4,5), (6,7),  # X edges
                      (0,2), (1,3), (4,6), (5,7),  # Y edges
                      (0,4), (1,5), (2,6), (3,7)]  # Z edges
        
        starts = torch.cat([verts[:, i] for i, j in edge_pairs], dim=0)  # (E*12, 3)
        ends = torch.cat([verts[:, j] for i, j in edge_pairs], dim=0)    # (E*12, 3)
        return starts, ends
    
    def _rect_edges_lines(self, mins: torch.Tensor, maxs: torch.Tensor, z: torch.Tensor):
        """Get start/end points for 4 edges of 2D rectangles. Returns (E*4, 3) each."""
        E = mins.shape[0]
        x0, y0 = mins[:, 0], mins[:, 1]
        x1, y1 = maxs[:, 0], maxs[:, 1]
        
        # 4 vertices: (E, 4, 3)
        verts = torch.stack([
            torch.stack([x0, y0, z], dim=-1),  # 0: bottom-left
            torch.stack([x1, y0, z], dim=-1),  # 1: bottom-right
            torch.stack([x0, y1, z], dim=-1),  # 2: top-left
            torch.stack([x1, y1, z], dim=-1),  # 3: top-right
        ], dim=1)  # (E, 4, 3)
        
        # 4 edges
        edge_pairs = [(0,1), (2,3), (0,2), (1,3)]
        starts = torch.cat([verts[:, i] for i, j in edge_pairs], dim=0)
        ends = torch.cat([verts[:, j] for i, j in edge_pairs], dim=0)
        return starts, ends
    
    def _visualize_pose_axes(
        self,
        object_pos_w: torch.Tensor,
        object_quat_w: torch.Tensor,
        target_pos_w: torch.Tensor,
        target_quat_w: torch.Tensor,
        num_envs: int,
    ):
        """Visualize pose axes (XYZ arrows) for current object and targets.
        
        Args:
            object_pos_w: (N, 3) current object position in world frame.
            object_quat_w: (N, 4) current object quaternion in world frame.
            target_pos_w: (N, W, 3) target positions in world frame.
            target_quat_w: (N, W, 4) target quaternions in world frame.
            num_envs: Number of environments.
        """
        if self._debug_draw is None:
            return
        
        # Axis arrow length
        AXIS_LENGTH = 0.08
        
        # Select envs to visualize
        if self.visualize_env_ids is None:
            env_ids = list(range(num_envs))
        else:
            env_ids = [i for i in self.visualize_env_ids if i < num_envs]
        
        if not env_ids:
            return
        
        E = len(env_ids)
        W = target_pos_w.shape[1]
        device = object_pos_w.device
        
        # Local axis directions
        x_axis = torch.tensor([1.0, 0.0, 0.0], device=device)
        y_axis = torch.tensor([0.0, 1.0, 0.0], device=device)
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=device)
        
        all_starts = []
        all_ends = []
        all_colors = []
        
        # === Current object axes (bright colors) ===
        obj_pos = object_pos_w[env_ids]  # (E, 3)
        obj_quat = object_quat_w[env_ids]  # (E, 4)
        
        # Transform local axes to world
        for axis, color in [(x_axis, (1.0, 0.0, 0.0, 1.0)),    # Red for X
                            (y_axis, (0.0, 1.0, 0.0, 1.0)),    # Green for Y
                            (z_axis, (0.0, 0.0, 1.0, 1.0))]:   # Blue for Z
            axis_exp = axis.unsqueeze(0).expand(E, 3)
            axis_world = quat_apply(obj_quat, axis_exp) * AXIS_LENGTH
            
            all_starts.append(obj_pos)
            all_ends.append(obj_pos + axis_world)
            all_colors.extend([color] * E)
        
        # === Target axes (dimmer colors, only first target for clarity) ===
        # Using first target in window (current target)
        tgt_pos = target_pos_w[env_ids, 0]  # (E, 3) - first target
        tgt_quat = target_quat_w[env_ids, 0]  # (E, 4)
        
        for axis, color in [(x_axis, (0.5, 0.0, 0.0, 0.8)),    # Dark red for X
                            (y_axis, (0.0, 0.5, 0.0, 0.8)),    # Dark green for Y
                            (z_axis, (0.0, 0.0, 0.5, 0.8))]:   # Dark blue for Z
            axis_exp = axis.unsqueeze(0).expand(E, 3)
            axis_world = quat_apply(tgt_quat, axis_exp) * AXIS_LENGTH
            
            all_starts.append(tgt_pos)
            all_ends.append(tgt_pos + axis_world)
            all_colors.extend([color] * E)
        
        # Concatenate and draw
        starts = torch.cat(all_starts, dim=0)  # (6*E, 3)
        ends = torch.cat(all_ends, dim=0)      # (6*E, 3)
        
        starts_list = starts.cpu().tolist()
        ends_list = ends.cpu().tolist()
        n = len(starts_list)
        self._debug_draw.draw_lines(starts_list, ends_list, all_colors, [3.0] * n)
    
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        trajectory_cfg: "TrajectoryParamsCfg | None" = None,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        ref_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        """Compute target sequence observations in robot base frame.
        
        Also manages trajectory manager: resets on env reset, steps each call.
        
        Returns:
            Point cloud mode: Flattened tensor (num_envs, window_size * num_points * 3).
            Pose mode: Flattened tensor (num_envs, window_size * 7).
        """
        N = env.num_envs
        W = self.window_size
        
        # Handle resets: check which envs just reset (episode_length_buf == 0)
        just_reset = env.episode_length_buf == 0
        reset_ids = just_reset.nonzero(as_tuple=True)[0]
        
        if len(reset_ids) > 0:
            # Get current object poses for resetting envs
            object_pos = self.object.data.root_pos_w[reset_ids]
            object_quat = self.object.data.root_quat_w[reset_ids]
            start_poses = torch.cat([object_pos, object_quat], dim=-1)
            
            # Get environment origins
            env_origins = env.scene.env_origins[reset_ids]
            
            self.trajectory_manager.reset(reset_ids, start_poses, env_origins)
        
        # Step trajectory (advance time)
        self.trajectory_manager.step(self.dt)
        
        # Get window targets: (N, W, 7) - pos(3) + quat(4)
        window_targets = self.trajectory_manager.get_window_targets()
        
        # Current object pose
        object_pos_w = self.object.data.root_pos_w  # (N, 3)
        object_quat_w = self.object.data.root_quat_w  # (N, 4)
        
        # Reference frame
        ref_pos_w = self.ref_asset.data.root_pos_w   # (N, 3)
        ref_quat_w = self.ref_asset.data.root_quat_w  # (N, 4)
        
        # Always compute point cloud observations and cache BOTH error types
        # use_point_cloud flag only affects which errors are used in rewards/terminations
        return self._compute_observations(
            env, N, W, window_targets, object_pos_w, object_quat_w, ref_pos_w, ref_quat_w
        )
    
    def _compute_observations(
        self,
        env: ManagerBasedRLEnv,
        N: int,
        W: int,
        window_targets: torch.Tensor,
        object_pos_w: torch.Tensor,
        object_quat_w: torch.Tensor,
        ref_pos_w: torch.Tensor,
        ref_quat_w: torch.Tensor,
    ) -> torch.Tensor:
        """Compute point cloud observations and cache BOTH point cloud and pose errors.
        
        Always outputs point clouds. Caches both error types for rewards/terminations
        to use based on use_point_cloud flag.
        """
        P = self.num_points
        
        # Target poses in world frame
        target_pos_w = window_targets[:, :, :3]   # (N, W, 3)
        target_quat_w = window_targets[:, :, 3:7]  # (N, W, 4)
        
        # === Point cloud computation ===
        target_pos = target_pos_w.unsqueeze(2)   # (N, W, 1, 3)
        target_quat = target_quat_w.unsqueeze(2)  # (N, W, 1, 4)
        
        local_exp = self.points_local.unsqueeze(1).expand(N, W, P, 3)  # (N, W, P, 3)
        quat_exp = target_quat.expand(N, W, P, 4)  # (N, W, P, 4)
        pos_exp = target_pos.expand(N, W, P, 3)    # (N, W, P, 3)
        
        # Apply rotation + translation: (N, W, P, 3)
        all_points_w = quat_apply(quat_exp.reshape(-1, 4), local_exp.reshape(-1, 3)).reshape(N, W, P, 3) + pos_exp
        
        # Transform to robot base frame
        ref_pos_exp = ref_pos_w.view(N, 1, 1, 3).expand(N, W, P, 3)
        ref_quat_exp = ref_quat_w.view(N, 1, 1, 4).expand(N, W, P, 4)
        
        all_points_b, _ = subtract_frame_transforms(
            ref_pos_exp.reshape(-1, 3), ref_quat_exp.reshape(-1, 4),
            all_points_w.reshape(-1, 3), None
        )
        all_points_b = all_points_b.reshape(N, W, P, 3)
        
        # Current object points (world space)
        object_quat_exp = object_quat_w.unsqueeze(1).expand(-1, P, -1)
        object_pos_exp = object_pos_w.unsqueeze(1).expand(-1, P, -1)
        current_points_w = quat_apply(object_quat_exp, self.points_local) + object_pos_exp
        
        # === Cache point cloud errors ===
        current_exp = current_points_w.unsqueeze(1).expand(N, W, P, 3)
        point_errors = (current_exp - all_points_w).norm(dim=-1)  # (N, W, P)
        env._cached_mean_errors = point_errors.mean(dim=-1)  # (N, W)
        
        # === Cache pose errors ===
        # Position error: (N, W)
        object_pos_exp_w = object_pos_w.unsqueeze(1).expand(N, W, 3)
        pos_errors = (object_pos_exp_w - target_pos_w).norm(dim=-1)  # (N, W)
        
        # Rotation error using quat_error_magnitude: (N, W)
        object_quat_exp_w = object_quat_w.unsqueeze(1).expand(N, W, 4)
        rot_errors = quat_error_magnitude(
            object_quat_exp_w.reshape(-1, 4),
            target_quat_w.reshape(-1, 4)
        ).reshape(N, W)  # (N, W)
        
        env._cached_pose_errors = {
            'pos': pos_errors,  # (N, W) position error in meters
            'rot': rot_errors,  # (N, W) rotation error in radians
        }
        
        # Visualization (point clouds every frame)
        if self.visualize and self.markers is not None:
            self._visualize(all_points_w, current_points_w, N)
        
        # Debug draw visualizations (lines) - must redraw every frame after clear
        if self._debug_draw is not None and (self.visualize_pose_axes or self.visualize_waypoint_region or self.visualize_goal_region):
            # Clear previous frame's lines first
            self._debug_draw.clear_lines()
            
            # Pose axes visualization (current + targets)
            if self.visualize_pose_axes:
                self._visualize_pose_axes(object_pos_w, object_quat_w, target_pos_w, target_quat_w, N)
            
            # Region boxes (waypoint/goal)
            if self.visualize_waypoint_region or self.visualize_goal_region:
                self._visualize_regions(self.env.scene.env_origins, self.trajectory_manager.start_poses[:, :3])
        
        # Flatten: (N, W * P * 3)
        return all_points_b.reshape(N, -1)
    
    def _visualize(
        self,
        target_points_w: torch.Tensor,
        current_points_w: torch.Tensor,
        num_envs: int,
    ):
        """Visualize target and current point clouds.
        
        Args:
            target_points_w: (N, W, P, 3) all target point clouds in world space.
            current_points_w: (N, P, 3) current object points in world space.
            num_envs: Number of environments.
        """
        # Determine which envs to visualize
        if self.visualize_env_ids is None:
            env_ids = list(range(num_envs))
        else:
            env_ids = [i for i in self.visualize_env_ids if i < num_envs]
        
        if not env_ids:
            return
        
        W = target_points_w.shape[1]
        P = target_points_w.shape[2]
        E = len(env_ids)
        device = target_points_w.device
        
        # Gather points for selected envs (stay on GPU)
        target_pts = target_points_w[env_ids]  # (E, W, P, 3)
        current_pts = current_points_w[env_ids]  # (E, P, 3)
        
        # Flatten targets: (E, W, P, 3) -> (E*W*P, 3)
        target_flat = target_pts.reshape(-1, 3)
        
        # Marker indices for targets: [0,0,...,1,1,...,W-1,W-1,...] repeated E times
        single_env_indices = torch.arange(W, device=device).repeat_interleave(P)  # (W*P,)
        target_indices = single_env_indices.repeat(E)  # (E*W*P,)
        
        if self.visualize_current:
            # Flatten current: (E, P, 3) -> (E*P, 3)
            current_flat = current_pts.reshape(-1, 3)
            current_indices = torch.full((E * P,), W, dtype=torch.long, device=device)
            
            # Combine targets + current
            all_positions = torch.cat([target_flat, current_flat], dim=0)
            all_indices = torch.cat([target_indices, current_indices])
        else:
            # Targets only
            all_positions = target_flat
            all_indices = target_indices
        
        self.markers.visualize(
            translations=all_positions,
            marker_indices=all_indices,
        )


def target_sequence_poses_b(
    env: ManagerBasedRLEnv,
    ref_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Target sequence poses in robot's base frame.
    
    Returns the window of target poses from the trajectory manager.
    Must be called AFTER target_sequence_obs_b which initializes the trajectory manager.
    
    Args:
        env: The environment (must have trajectory_manager attached).
        ref_asset_cfg: Reference frame (robot). Defaults to ``SceneEntityCfg("robot")``.
    
    Returns:
        Flattened tensor (num_envs, window_size * 7) - target poses in robot frame.
    """
    trajectory_manager = env.trajectory_manager
    ref_asset: Articulation = env.scene[ref_asset_cfg.name]
    
    N = env.num_envs
    W = trajectory_manager.cfg.window_size
    
    # Get window targets: (N, W, 7) - pos(3) + quat(4)
    window_targets = trajectory_manager.get_window_targets()
    target_pos_w = window_targets[:, :, :3]   # (N, W, 3)
    target_quat_w = window_targets[:, :, 3:7]  # (N, W, 4)
    
    # Reference frame
    ref_pos_w = ref_asset.data.root_pos_w   # (N, 3)
    ref_quat_w = ref_asset.data.root_quat_w  # (N, 4)
    
    # Transform target poses to robot base frame
    target_pos_rel = target_pos_w - ref_pos_w.unsqueeze(1)  # (N, W, 3)
    ref_quat_inv = quat_inv(ref_quat_w)  # (N, 4)
    
    # Rotate each target position to robot frame
    target_pos_b = quat_apply(
        ref_quat_inv.unsqueeze(1).expand(N, W, 4).reshape(-1, 4),
        target_pos_rel.reshape(-1, 3)
    ).reshape(N, W, 3)
    
    # Orientation: multiply by inverse robot quat
    target_quat_b = quat_mul(
        ref_quat_inv.unsqueeze(1).expand(N, W, 4).reshape(-1, 4),
        target_quat_w.reshape(-1, 4)
    ).reshape(N, W, 4)
    
    # Combine to poses in robot frame: (N, W, 7)
    target_poses_b = torch.cat([target_pos_b, target_quat_b], dim=-1)
    
    # Flatten: (N, W * 7)
    return target_poses_b.reshape(N, -1)
