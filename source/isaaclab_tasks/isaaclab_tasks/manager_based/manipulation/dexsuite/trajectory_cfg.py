# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Trajectory parameters configuration - separate file to avoid circular imports."""

from isaaclab.utils import configclass


@configclass
class TrajectoryParamsCfg:
    """Config for trajectory following task, used by env and TrajectoryManager."""

    # Point cloud settings
    num_points: int = 32  # Points per point cloud
    object_pc_history_length: int = 5  # History frames for object point cloud
    
    # Visualization (for debugging - set all to False for performance)
    visualize_targets: bool = True  # Show target point clouds
    visualize_current: bool = True  # Show current object point cloud
    visualize_waypoint_region: bool = True  # Show waypoint sampling box
    visualize_goal_region: bool = True  # Show goal sampling region on table
    visualize_env_ids: list[int] | None = [0]  # Which envs to visualize (None = all envs)

    # Timing
    trajectory_duration: float = 15.0  # Total trajectory time (seconds)
    target_hz: float = 20.0/3.0  # Target rate, target_dt = 1/target_hz
    window_size: int = 5  # Lookahead targets in observation

    # Phase durations (seconds)
    pickup_duration: float = 2.0   # Stationary at start for grasping
    manipulate_duration: float = 8.0  # Main movement with ease-in
    place_duration: float = 3.0    # Ease-out slowdown to table
    release_duration: float = 1.0  # Stationary at goal for release

    # Intermediate waypoints (0-2 sampled uniformly)
    num_waypoints_min: int = 0
    num_waypoints_max: int = 2
    waypoint_position_range_x: tuple[float, float] | None = None  # None = use workspace_x
    waypoint_position_range_y: tuple[float, float] | None = None  # None = use workspace_y
    waypoint_position_range_z: tuple[float, float] | None = None  # None = use workspace_z
    vary_waypoint_orientation: bool = True
    waypoint_max_rotation: float = 1.57  # Max rotation (rad) ~90°

    # End goal (stable on table) - None = use workspace bounds
    goal_x_range: tuple[float, float] | None = None  # None = use workspace_x
    goal_y_range: tuple[float, float] | None = None  # None = use workspace_y  
    table_surface_z: float = 0.255
    table_margin: float = 0.0  # Keep goal this far from table edge

    # Easing
    ease_power: float = 2.0  # Quadratic ease curves for phase transitions

    # Thresholds (point-to-point mean error) - final values after curriculum
    success_threshold: float = 0.015  # Success distance (m)
    termination_threshold: float = 0.15  # Max deviation before termination (m)
    
    # Curriculum threshold ranges (initial -> final)
    success_threshold_initial: float = 0.06  # Start lenient
    termination_threshold_initial: float = 0.3  # Start lenient (30cm)
    
    # Curriculum difficulty progression
    curriculum_error_tol: float = 0.05  # Promote difficulty if error < this (m)

    # Reward
    lookahead_decay: float = 0.1  # Exponential decay for lookahead reward

    # Workspace bounds (relative to robot base)
    workspace_x: tuple[float, float] = (-0.8, -0.3)
    workspace_y: tuple[float, float] = (-0.35, 0.35)
    workspace_z_min: float = 0.28  # Above table surface
    workspace_z_max: float = 0.75

# Shared config instance
TRAJECTORY_PARAMS = TrajectoryParamsCfg()

