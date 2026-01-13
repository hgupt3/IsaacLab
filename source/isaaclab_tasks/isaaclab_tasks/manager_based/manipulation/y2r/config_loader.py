# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration loader for Y2R task with layer-based YAML composition.

YAML is the single source of truth - all values must be defined in configs/base.yaml.
Layers (mode, task) are composed on top of base config at runtime.

Usage:
    cfg = get_config()                                   # train mode, base task
    cfg = get_config(mode="distill")                     # distill mode (student training)
    cfg = get_config(mode="play", task="push")           # teacher play on task
    cfg = get_config(mode="play_student", task="cup")    # student play on task

To add a new task:
    1. Create configs/layers/tasks/<name>.yaml with task-specific overrides
    2. Run: ./scripts/play.sh --task <name> --continue
"""

from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, get_origin, get_type_hints

import yaml


# ==============================================================================
# YAML LOADING WITH INHERITANCE
# ==============================================================================

def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override dict into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml(name: str) -> dict:
    """Load YAML config with inheritance support."""
    config_dir = Path(__file__).parent / "configs"
    filename = f"{name}.yaml" if not name.endswith(".yaml") else name
    config_path = config_dir / filename
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    if "_base_" in config:
        base_name = config.pop("_base_").replace(".yaml", "")
        base_config = _load_yaml(base_name)
        config = _deep_merge(base_config, config)
    
    return config


# ==============================================================================
# AUTO-PARSER
# ==============================================================================

def _auto_parse(cls, data: dict) -> Any:
    """Recursively parse dict into dataclass. All values must come from data."""
    hints = get_type_hints(cls)
    kwargs = {}
    
    for field_name, field_type in hints.items():
        if field_name.startswith("_"):
            kwargs[field_name] = data.get(field_name, {})
            continue
            
        value = data.get(field_name)
        origin = get_origin(field_type)
        
        if is_dataclass(field_type):
            kwargs[field_name] = _auto_parse(field_type, value or {})
        elif value is None:
            kwargs[field_name] = None
        elif origin is tuple or field_type is tuple:
            kwargs[field_name] = tuple(value) if isinstance(value, list) else value
        elif origin is list:
            kwargs[field_name] = list(value)
        else:
            kwargs[field_name] = value
    
    return cls(**kwargs)


# ==============================================================================
# CONFIG DATACLASSES (structure only - values come from YAML)
# ==============================================================================

@dataclass
class ModeConfig:
    use_point_cloud: bool
    use_eigen_grasp: bool
    use_student_mode: bool


@dataclass
class HistoryConfig:
    policy: int
    proprio: int
    perception: int
    targets: int


@dataclass
class ObservationsConfig:
    num_points: int
    student_num_points: int
    point_pool_size: int
    history: HistoryConfig


@dataclass
class PhasesConfig:
    grasp: float
    manipulate_base: float
    settle: float = 0.0  # Pause at goal before retreat (fingers can open)
    hand_release: float = 2.5


@dataclass
class TrajectoryConfig:
    target_hz: float
    window_size: int
    phases: PhasesConfig
    easing_power: float
    release_ease_power: float = 2.0  # Ease-in power for retreat (slow start)
    skip_manipulation_probability: float = 0.0  # Probability of grasp-only episodes


# ==============================================================================
# HAND TRAJECTORY CONFIG
# ==============================================================================

@dataclass
class GraspSamplingConfig:
    standoff_range: tuple[float, float]
    exclude_bottom_fraction: float
    exclude_toward_robot: bool
    toward_robot_threshold: float
    approach_distance: float | None  # Pre-grasp distance (null = disabled)
    align_fraction: float            # Fraction of grasp phase for alignment
    fixed_origin_offset: list[float] | None  # [x,y,z] offset from object center (null = center)
    fixed_direction: list[float] | None      # [x,y,z] look direction (null = random sampling)
    fixed_hand_roll: float | None            # Rotation around approach axis (radians, null = auto)


@dataclass
class GraspKeypointsConfig:
    count: tuple[int, int]
    pos_perturbation: float
    rot_perturbation: float


@dataclass
class ReleasePositionRangeConfig:
    x: tuple[float, float] | None
    y: tuple[float, float] | None
    z: tuple[float, float] | None


@dataclass
class ReleaseConfig:
    position_range: ReleasePositionRangeConfig
    min_distance_from_object: float


@dataclass
class HandTrajectoryConfig:
    enabled: bool
    grasp_rot_completion_fraction: float  # Rotation completes at this fraction of grasp duration
    grasp_sampling: GraspSamplingConfig
    keypoints: GraspKeypointsConfig
    release: ReleaseConfig


@dataclass
class WorkspaceConfig:
    x: tuple[float, float]
    y: tuple[float, float]
    z: tuple[float, float]
    table_surface_z: float


@dataclass
class PositionRangeConfig:
    x: tuple[float, float] | None
    y: tuple[float, float] | None
    z: tuple[float, float] | None


@dataclass
class WaypointsConfig:
    count: tuple[int, int]
    movement_duration: float  # Seconds of movement time per waypoint
    pause_duration: float     # Seconds to pause at each waypoint
    position_range: PositionRangeConfig
    vary_orientation: bool
    max_rotation: float
    fixed_waypoints: list[list[float]] | None  # [[x,y,z,qw,qx,qy,qz], ...] overrides random


@dataclass
class GoalConfig:
    x_range: tuple[float, float] | None
    y_range: tuple[float, float] | None
    table_margin: float
    return_to_start: bool  # If true, goal = start position


@dataclass
class RewardConfig:
    weight: float
    params: dict = field(default_factory=dict)  # All reward-specific params go here


@dataclass
class RewardsConfig:
    action_l2: RewardConfig
    action_rate_l2: RewardConfig
    fingers_to_object: RewardConfig
    lookahead_tracking: RewardConfig
    trajectory_success: RewardConfig
    early_termination: RewardConfig
    arm_table_penalty: RewardConfig
    good_finger_contact: RewardConfig
    finger_manipulation: RewardConfig
    palm_velocity_penalty: RewardConfig
    palm_orientation_penalty: RewardConfig
    distal_joint3_penalty: RewardConfig
    joint_limits_margin: RewardConfig
    tracking_progress: RewardConfig
    hand_pose_following: RewardConfig
    finger_release: RewardConfig
    finger_regularizer: RewardConfig
    object_stillness: RewardConfig


@dataclass
class TrajectoryDeviationConfig:
    point_cloud_threshold: tuple[float, float]
    position_threshold: tuple[float, float]
    rotation_threshold: tuple[float, float]


@dataclass
class HandPoseDeviationConfig:
    position_threshold: tuple[float, float]
    rotation_threshold: tuple[float, float]


@dataclass
class TerminationsConfig:
    trajectory_deviation: TrajectoryDeviationConfig
    hand_pose_deviation: HandPoseDeviationConfig


@dataclass
class DifficultyConfig:
    initial: int
    min: int
    max: int


@dataclass
class SchedulerConfig:
    step_interval: int | None  # Steps between difficulty floor increases (null to disable)
    use_performance: bool      # Whether performance can advance faster than floor


@dataclass
class AdvancementTolerancesConfig:
    position: float | None
    rotation: float | None
    point_cloud: float | None


@dataclass
class NoiseConfig:
    joint_pos: tuple[float, float]
    joint_vel: tuple[float, float]
    hand_tips: tuple[float, float]
    object_point_cloud: tuple[float, float]
    object_pose: tuple[float, float]
    target_point_clouds: tuple[float, float]
    target_poses: tuple[float, float]


@dataclass
class GravityConfig:
    initial: tuple[float, float, float]
    final: tuple[float, float, float]


@dataclass
class CurriculumConfig:
    difficulty: DifficultyConfig
    scheduler: SchedulerConfig
    advancement_tolerances: AdvancementTolerancesConfig
    noise: NoiseConfig
    gravity: GravityConfig


@dataclass
class ObjectRandomizationConfig:
    scale: tuple[float, float]
    mass_scale: tuple[float, float]
    static_friction: tuple[float, float]
    dynamic_friction: tuple[float, float]
    restitution: tuple[float, float]


@dataclass
class RobotRandomizationConfig:
    static_friction: tuple[float, float]
    dynamic_friction: tuple[float, float]
    restitution: tuple[float, float]
    stiffness_scale: tuple[float, float]
    damping_scale: tuple[float, float]
    joint_friction_scale: tuple[float, float]


@dataclass
class ResetRandomizationConfig:
    table_xy: tuple[float, float]
    object_x: tuple[float, float]
    object_y: tuple[float, float]
    object_yaw: tuple[float, float]
    robot_joints: tuple[float, float]
    robot_wrist: tuple[float, float]
    z_offset: float


@dataclass
class RandomizationConfig:
    object: ObjectRandomizationConfig
    robot: RobotRandomizationConfig
    reset: ResetRandomizationConfig


@dataclass
class VisualizationConfig:
    targets: bool
    current_object: bool
    student_visible: bool
    student_target: bool
    waypoint_region: bool
    goal_region: bool
    pose_axes: bool
    hand_pose_targets: bool
    grasp_surface_point: bool
    env_ids: list[int] | None
    debug_print_rewards: bool


@dataclass
class SimulationConfig:
    physics_dt: float
    decimation: int
    num_envs: int
    env_spacing: float
    replicate_physics: bool


@dataclass
class RobotConfig:
    action_scale: float
    arm_joint_count: int
    hand_joint_count: int
    eigen_dim: int


@dataclass
class PushTConfig:
    enabled: bool
    object_usd: str | None
    outline_usd: str | None
    object_scale: float
    outline_position: tuple[float, float]
    object_rotation: tuple[float, float, float, float]
    outline_rotation: tuple[float, float, float, float]
    goal_offset: tuple[float, float, float] | None
    goal_rotation: tuple[float, float, float, float] | None
    rolling_window: float
    min_speed: float
    include_in_primitives: bool
    object_z_offset: float | None  # null = auto-compute from mesh geometry
    outline_z_offset: float | None  # null = auto-compute from mesh geometry


@dataclass
class GenerationConfig:
    num_shapes: int
    primitives_per_shape: tuple[int, int]
    base_size: tuple[float, float]
    size_decay: float
    primitive_types: dict[str, float]
    seed: int | None


@dataclass
class ProceduralObjectsConfig:
    enabled: bool
    percentage: float
    asset_dir: str
    regenerate: bool
    generation: GenerationConfig


@dataclass
class PseudoCameraConfig:
    position: tuple[float, float, float]


@dataclass
class CameraOffsetConfig:
    pos: tuple[float, float, float]
    rot: tuple[float, float, float]


@dataclass
class WristCameraConfig:
    enabled: bool
    resolution: int
    focal_length: float
    horizontal_aperture: float
    clipping_range: tuple[float, float]
    offset: CameraOffsetConfig
    web_viewer: bool
    viewer_port: int
    viewer_update_hz: int


@dataclass
class Y2RConfig:
    """Complete Y2R task configuration."""
    mode: ModeConfig
    observations: ObservationsConfig
    trajectory: TrajectoryConfig
    hand_trajectory: HandTrajectoryConfig
    workspace: WorkspaceConfig
    waypoints: WaypointsConfig
    goal: GoalConfig
    rewards: RewardsConfig
    terminations: TerminationsConfig
    curriculum: CurriculumConfig
    randomization: RandomizationConfig
    visualization: VisualizationConfig
    simulation: SimulationConfig
    robot: RobotConfig
    push_t: PushTConfig
    procedural_objects: ProceduralObjectsConfig
    wrist_camera: WristCameraConfig
    pseudo_camera: PseudoCameraConfig


# ==============================================================================
# PUBLIC API
# ==============================================================================

def get_config(mode: str = "train", task: str = "base") -> Y2RConfig:
    """Load Y2R config with layer-based composition.
    
    Args:
        mode: One of "train", "distill", "play", "play_student"
            - train: Base config only (teacher training, 16k envs)
            - distill: Base + student layer (student distillation, 4k envs)
            - play: Base + play layer (teacher evaluation, 32 envs)
            - play_student: Base + student + play + student_play layers (4 envs)
        task: Task name - "base" or any YAML file in configs/layers/tasks/
            - base: Uses base config (random objects)
            - <name>: Loads configs/layers/tasks/<name>.yaml
            
    Returns:
        Y2RConfig instance with layers composed
        
    To add a new task:
        1. Create configs/layers/tasks/<name>.yaml with task-specific overrides
        2. Run: ./scripts/play.sh --task <name> --continue
    """
    cfg = _load_yaml("base")
    
    # Mode layers (mutually exclusive paths)
    if mode == "distill":
        cfg = _deep_merge(cfg, _load_yaml("layers/student"))
    elif mode == "play":
        cfg = _deep_merge(cfg, _load_yaml("layers/play"))
    elif mode == "play_student":
        cfg = _deep_merge(cfg, _load_yaml("layers/student"))
        cfg = _deep_merge(cfg, _load_yaml("layers/play"))
        cfg = _deep_merge(cfg, _load_yaml("layers/student_play"))
    elif mode == "keyboard":
        cfg = _deep_merge(cfg, _load_yaml("layers/keyboard"))
    # mode == "train" uses base only
    
    # Task layer (optional, applied last)
    if task != "base":
        cfg = _deep_merge(cfg, _load_yaml(f"layers/tasks/{task}"))
    
    return _auto_parse(Y2RConfig, cfg)
