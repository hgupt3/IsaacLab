# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Trajectory parameters configuration - separate file to avoid circular imports."""

from isaaclab.utils import configclass


@configclass
class TrajectoryParamsCfg:
    """Config for trajectory following task, used by env and TrajectoryManager."""

    # Representation mode
    use_point_cloud: bool = False  # True = point cloud, False = pose (pos + quat)
    
    # Action space
    use_eigen_grasp: bool = True  # True = eigen grasp (28D), False = full joints (23D)

    # Point cloud settings (when use_point_cloud=True)
    num_points: int = 32  # Points per point cloud
    object_pc_history_length: int = 5  # History frames for object point cloud
    
    # Pose mode settings (when use_point_cloud=False)
    pose_rot_std: float = 0.3  # Std for rotation tanh kernel in rewards ()
    object_pose_history_length: int = 5  # History frames for object pose
    
    # Visualization (for debugging - set all to False for performance)
    visualize_targets: bool = True  # Show target point clouds
    visualize_current: bool = True  # Show current object point cloud
    visualize_waypoint_region: bool = True  # Show waypoint sampling box
    visualize_goal_region: bool = True  # Show goal sampling region on table
    visualize_pose_axes: bool = True  # Show axis arrows for current and target poses
    visualize_env_ids: list[int] | None = []  # Which envs to visualize (None = all envs)
    debug_print_rewards: bool = False  # Print individual rewards every N steps (enabled in PLAY config)

    # Timing
    trajectory_duration: float = 10.0  # Total trajectory time (seconds)
    target_hz: float = 20.0/3.0  # Target rate, target_dt = 1/target_hz
    window_size: int = 5  # Lookahead targets in observation

    # Phase durations (seconds)
    pickup_duration: float = 0.5   # Stationary at start for grasping
    manipulate_duration: float = 7.5  # Main movement with symmetric ease-in/out
    release_duration: float = 2.0  # Stationary at goal for release

    # Intermediate waypoints (0-2 sampled uniformly)
    num_waypoints_min: int = 0
    num_waypoints_max: int = 2
    waypoint_position_range_x: tuple[float, float] | None = None  # None = use workspace_x
    waypoint_position_range_y: tuple[float, float] | None = None  # None = use workspace_y
    waypoint_position_range_z: tuple[float, float] | None = None  # None = use workspace_z
    vary_waypoint_orientation: bool = True
    waypoint_max_rotation: float = 3.14  # Max rotation (rad)

    # End goal (stable on table) - None = use workspace bounds
    goal_x_range: tuple[float, float] | None = None  # None = use workspace_x
    goal_y_range: tuple[float, float] | None = None  # None = use workspace_y  
    table_surface_z: float = 0.255
    table_margin: float = 0.0  # Keep goal this far from table edge

    # Easing
    ease_power: float = 2.0  # Quadratic ease curves for phase transitions

    # Thresholds (point-to-point mean error for point cloud mode) - final values after curriculum
    success_threshold: float = 0.02  # Success distance (m)
    termination_threshold: float = 0.06  # Max deviation before termination (m)
    
    # Curriculum threshold ranges (initial -> final) for point cloud mode
    # Curriculum advances when success_threshold is met at current difficulty
    success_threshold_initial: float = 0.06  # Start lenient
    termination_threshold_initial: float = 0.18  # Start lenient
    
    # Pose mode thresholds (position in meters, rotation in radians)
    pose_success_pos_threshold: float = 0.02  # Success position distance (m)
    pose_success_rot_threshold: float = 0.2  # Success rotation error (rad, ~11 deg)
    pose_termination_pos_threshold: float = 0.05  # Max position deviation (m)
    pose_termination_rot_threshold: float = 0.5  # Max rotation deviation (rad, ~29 deg)
    
    # Pose mode curriculum threshold ranges (initial -> final)
    pose_success_pos_threshold_initial: float = 0.08  # Start lenient
    pose_success_rot_threshold_initial: float = 0.8  # Start lenient (~46 deg)
    pose_termination_pos_threshold_initial: float = 0.25  # Start lenient
    pose_termination_rot_threshold_initial: float = 2.5  # Start lenient

    # Reward
    lookahead_decay: float = 0.2  # Exponential decay for lookahead reward

    # Workspace bounds (relative to robot base)
    workspace_x: tuple[float, float] = (-0.8, -0.3)
    workspace_y: tuple[float, float] = (-0.35, 0.35)
    workspace_z_min: float = 0.28  # Above table surface
    workspace_z_max: float = 0.75

# Shared config instance
TRAJECTORY_PARAMS = TrajectoryParamsCfg()

