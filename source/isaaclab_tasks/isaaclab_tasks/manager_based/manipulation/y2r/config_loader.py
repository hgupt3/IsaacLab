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
from enum import Enum
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


def _normalize_curriculum_keys(data: Any) -> Any:
    """Ensure dict keys are strings for config serialization."""
    if isinstance(data, dict):
        normalized: dict = {}
        for key, value in data.items():
            normalized[str(key)] = _normalize_curriculum_keys(value)
        return normalized
    if isinstance(data, list):
        return [_normalize_curriculum_keys(item) for item in data]
    return data


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
    # Teacher observations
    policy: int
    proprio: int
    object_pc: int
    poses: int
    targets: int
    # Student observations
    student_pc: int
    student_targets: int
    student_camera: int


@dataclass
class ObservationsConfig:
    num_points: int
    student_num_points: int
    point_pool_size: int
    history: HistoryConfig
    point_cloud_filter: dict | None = None  # Dict with keys: enabled, axis, min, max


@dataclass
class TrajectoryConfig:
    target_hz: float
    window_size: int
    easing_power: float
    skip_manipulation_probability: float = 0.0  # Probability of grasp-only episodes
    segments: list[dict] = field(default_factory=list)  # List of segment configs (REQUIRED)


# ==============================================================================
# SEGMENT-BASED TRAJECTORY SYSTEM
# ==============================================================================

class HandCouplingMode(str, Enum):
    """Hand coupling modes for segments."""
    FULL = "full"                # Hand rotates + translates with object (default)
    POSITION_ONLY = "position_only"  # Hand translates but orientation fixed
    NONE = "none"                # Hand fully decoupled


@dataclass
class BaseSegmentConfig:
    """Base class for all trajectory segments."""
    name: str                    # Human-readable identifier
    duration: float              # Seconds for this segment
    type: str = "base"           # Segment type (overridden by children)
    hand_coupling: str = "full"  # Defaults to "full" - only specify if different
    hand_orientation: list[float] | None = None  # Optional fixed hand quat [qw,qx,qy,qz] for position_only mode
    curriculum: dict | None = None  # Optional per-difficulty overrides (e.g., count ranges)


@dataclass
class WaypointSegmentConfig(BaseSegmentConfig):
    """Move to discrete waypoint target."""
    pose: list[float] | None = None  # [x,y,z,qw,qx,qy,qz] or null (computed at reset)
    type: str = "waypoint"


@dataclass
class HelicalSegmentConfig(BaseSegmentConfig):
    """Helical motion: coupled rotation + translation."""
    axis: list[float] = field(default_factory=list)  # Rotation axis unit vector [x,y,z]
    angular_velocity: float = 1.0  # Angular velocity in rad/s (total rotation = angular_velocity * duration)
    translation: list[float] = field(default_factory=list)  # Translation vector [x,y,z]
    type: str = "helical"


@dataclass
class RandomWaypointSegmentConfig(BaseSegmentConfig):
    """Generates random waypoints at reset time (expands to N waypoint segments)."""
    count: tuple[int, int] = (0, 0)  # [min, max] waypoints to sample
    movement_duration: float = 3.0   # Duration PER waypoint (total = N * movement_duration)
    pause_duration: float = 0.0      # Pause PER waypoint
    position_range: PositionRangeConfig | None = None
    vary_orientation: bool = False
    max_rotation: float = 0.0
    type: str = "random_waypoint"
    duration: float = 0.0  # Ignored - computed at reset based on sampled count


# ==============================================================================
# HAND TRAJECTORY CONFIG
# ==============================================================================

@dataclass
class GraspSamplingConfig:
    standoff_range: tuple[float, float]
    exclude_bottom_fraction: float
    exclude_toward_robot: bool
    toward_robot_threshold: float
    exclude_upward: bool
    upward_threshold: float
    approach_distance: float | None  # Pre-grasp distance (null = disabled)
    align_fraction: float            # Fraction of grasp phase for alignment
    fixed_origin_offset: list[float] | None  # [x,y,z] offset from object center (null = center)
    fixed_direction: list[float] | None      # [x,y,z] look direction (null = random sampling)
    fixed_hand_roll: float | None            # Rotation around approach axis (radians, null = auto)


@dataclass
class GraspKeypointsConfig:
    count: tuple[int, int]
    max_surface_perturbation: float  # Max distance to search for nearby surface points (meters)
    roll_perturbation: float         # Max roll perturbation around approach axis (radians)
    normal_similarity_threshold: float  # Min dot product of normals (prevents flipping to opposite face)
    curriculum: dict | None = None  # Optional per-difficulty overrides for keypoint counts


@dataclass
class FeasibilityConfig:
    min_height: float                # Palm must be above table surface + this margin (meters)


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
    feasibility: FeasibilityConfig
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
    hand_eigen: tuple[float, float]
    hand_tips: tuple[float, float]
    object_point_cloud: tuple[float, float]
    object_pose: tuple[float, float]
    hand_pose: tuple[float, float]
    target_point_clouds: tuple[float, float]
    target_poses: tuple[float, float]
    hand_pose_targets: tuple[float, float]


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
    gate_floor: tuple[float, float]
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
    outline_has_contacts: bool = False  # Enable collision on outline
    outline_is_kinematic: bool = False  # Make outline fixed/unmovable
    outline_has_gravity: bool = False   # Enable gravity on outline


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

def _validate_segment_fields(seg_dict: dict, expected_fields: set[str], seg_type: str) -> None:
    """Validate that segment dict only contains expected fields.

    Args:
        seg_dict: Segment dictionary from YAML
        expected_fields: Set of allowed field names
        seg_type: Segment type name for error messages

    Raises:
        ValueError: If unknown fields are found
    """
    actual_fields = set(seg_dict.keys())
    unexpected_fields = actual_fields - expected_fields

    if unexpected_fields:
        raise ValueError(
            f"Unexpected fields in '{seg_type}' segment '{seg_dict.get('name', 'unknown')}': "
            f"{sorted(unexpected_fields)}. Expected fields: {sorted(expected_fields)}"
        )


def _parse_segments(yaml_segments: list[dict]) -> list[BaseSegmentConfig]:
    """Parse segment configs from YAML.

    Args:
        yaml_segments: List of segment dicts from YAML

    Returns:
        List of parsed segment config objects
    """
    segments = []
    for seg_dict in yaml_segments:
        seg_type = seg_dict["type"]

        if seg_type == "waypoint":
            # Validate fields
            expected = {"type", "name", "duration", "pose", "hand_coupling", "hand_orientation", "curriculum"}
            _validate_segment_fields(seg_dict, expected, "waypoint")

            seg = WaypointSegmentConfig(
                name=seg_dict["name"],
                duration=seg_dict["duration"],
                pose=seg_dict.get("pose"),  # Can be null
                hand_coupling=seg_dict.get("hand_coupling", "full"),  # Has default in dataclass
                hand_orientation=seg_dict.get("hand_orientation"),  # Optional fixed orientation
                curriculum=seg_dict.get("curriculum"),
            )
        elif seg_type == "helical":
            # Validate fields
            expected = {"type", "name", "duration", "axis", "angular_velocity", "translation", "hand_coupling", "hand_orientation", "curriculum"}
            _validate_segment_fields(seg_dict, expected, "helical")

            seg = HelicalSegmentConfig(
                name=seg_dict["name"],
                duration=seg_dict["duration"],
                axis=seg_dict["axis"],
                angular_velocity=seg_dict["angular_velocity"],
                translation=seg_dict["translation"],
                hand_coupling=seg_dict.get("hand_coupling", "full"),  # Has default in dataclass
                hand_orientation=seg_dict.get("hand_orientation"),  # Optional fixed orientation
                curriculum=seg_dict.get("curriculum"),
            )
        elif seg_type == "random_waypoint":
            # Validate fields
            expected = {"type", "name", "count", "movement_duration", "pause_duration", "position_range",
                       "vary_orientation", "max_rotation", "hand_coupling", "curriculum"}
            _validate_segment_fields(seg_dict, expected, "random_waypoint")

            # Parse position_range if present
            pos_range = None
            if "position_range" in seg_dict:
                pr = seg_dict["position_range"]
                pos_range = PositionRangeConfig(
                    x=tuple(pr["x"]) if pr.get("x") else None,
                    y=tuple(pr["y"]) if pr.get("y") else None,
                    z=tuple(pr["z"]) if pr.get("z") else None,
                )

            seg = RandomWaypointSegmentConfig(
                name=seg_dict["name"],
                count=tuple(seg_dict["count"]),
                movement_duration=seg_dict["movement_duration"],
                pause_duration=seg_dict["pause_duration"],
                position_range=pos_range,
                vary_orientation=seg_dict["vary_orientation"],
                max_rotation=seg_dict["max_rotation"],
                hand_coupling=seg_dict.get("hand_coupling", "full"),  # Has default in dataclass
                curriculum=seg_dict.get("curriculum"),
            )
        else:
            raise ValueError(f"Unknown segment type: {seg_type}")

        segments.append(seg)

    return segments


def get_config_file_paths(mode: str = "train", task: str = "base") -> list[Path]:
    """Return list of YAML files that will be loaded for given mode/task.

    This ensures wandb logging matches actual config composition by maintaining
    a single source of truth for which files are loaded.

    Args:
        mode: One of "train", "distill", "play", "play_student", "keyboard"
        task: Task name - "base" or any YAML file in configs/layers/tasks/

    Returns:
        List of Path objects for YAML files that will be loaded
    """
    config_dir = Path(__file__).parent / "configs"
    files = [config_dir / "base.yaml"]

    # Mode layers (mirrors get_config() logic exactly)
    if mode == "distill":
        files.append(config_dir / "layers" / "student.yaml")
    elif mode == "play":
        files.append(config_dir / "layers" / "play.yaml")
    elif mode == "play_student":
        files.extend([
            config_dir / "layers" / "student.yaml",
            config_dir / "layers" / "play.yaml",
            config_dir / "layers" / "student_play.yaml"
        ])
    elif mode == "keyboard":
        files.append(config_dir / "layers" / "keyboard.yaml")
    # mode == "train" uses base only

    # Task layer (optional, applied last)
    if task != "base":
        files.append(config_dir / "layers" / "tasks" / f"{task}.yaml")

    return files


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

    cfg = _normalize_curriculum_keys(cfg)

    # Parse segments from YAML dict to segment config objects
    if "trajectory" in cfg and "segments" in cfg["trajectory"]:
        cfg["trajectory"]["segments"] = _parse_segments(cfg["trajectory"]["segments"])

    return _auto_parse(Y2RConfig, cfg)
