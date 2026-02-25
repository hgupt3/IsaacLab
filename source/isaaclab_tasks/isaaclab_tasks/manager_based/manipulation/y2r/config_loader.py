# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration loader for Y2R task with layer-based YAML composition.

YAML is the single source of truth - all values must be defined in configs/base.yaml.
Layers (robot, mode, task) are composed on top of base config at runtime.

Layer order: base → robot → mode → task

All parameters default to env vars (Y2R_MODE, Y2R_TASK, Y2R_ROBOT),
with fallback defaults defined in _DEFAULT_* constants below.

Usage:
    cfg = get_config()                                          # reads all from env vars
    cfg = get_config(robot="kuka_allegro")                      # override robot only
    cfg = get_config(mode="play", task="push")                  # override mode+task
"""

from __future__ import annotations

import types
from dataclasses import dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

import os

import yaml

# ==============================================================================
# DEFAULTS (single source of truth for env var fallbacks)
# ==============================================================================
_DEFAULT_MODE = "train"
_DEFAULT_TASK = "base"
_DEFAULT_ROBOT = "ur5e_leap"


def _resolve_defaults(mode: str | None, task: str | None, robot: str | None) -> tuple[str, str, str]:
    """Resolve None parameters from environment variables, falling back to defaults."""
    return (
        mode if mode is not None else os.environ.get("Y2R_MODE", _DEFAULT_MODE),
        task if task is not None else os.environ.get("Y2R_TASK", _DEFAULT_TASK),
        robot if robot is not None else os.environ.get("Y2R_ROBOT", _DEFAULT_ROBOT),
    )


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
        config = yaml.safe_load(f) or {}

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

        # Handle Optional[DataclassType] (i.e. DataclassType | None)
        inner_dc = None
        if origin is Union or origin is types.UnionType:
            args = get_args(field_type)
            for a in args:
                if is_dataclass(a):
                    inner_dc = a
                    break

        if is_dataclass(field_type):
            kwargs[field_name] = _auto_parse(field_type, value or {})
        elif inner_dc is not None:
            kwargs[field_name] = _auto_parse(inner_dc, value) if value is not None else None
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
    student_proprio: int
    student_poses: int
    student_future_poses: int
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
    path_mode: bool = False
    timing_aware: bool = True
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
    hand_coupling_weights: dict[str, float] = field(default_factory=lambda: {"full": 1.0})  # Per-env coupling weights
    hand_orientation: list[float] | None = None  # Optional fixed hand quat [qw,qx,qy,qz] for position_only mode
    curriculum: dict | None = None  # Optional per-difficulty overrides


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
    count_weights: list[float] = field(default_factory=lambda: [1.0])  # P(count=i) ~ weights[i]
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
    count_weights: list[float]       # P(count=i) ~ weights[i]
    max_surface_perturbation: float  # Max distance to search for nearby surface points (meters)
    roll_perturbation: float         # Max roll perturbation around approach axis (radians)
    normal_similarity_threshold: float  # Min dot product of normals (prevents flipping to opposite face)
    curriculum: dict | None = None  # Optional per-difficulty overrides for count_weights


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
class ContactFactorConfig:
    nail_gate: float
    nail_gate_min: float
    finger_weights: list[float]
    threshold: float
    ramp: float
    min_factor: float
    thumb_gate_floor: float


@dataclass
class RewardsConfig:
    contact_factor: ContactFactorConfig
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
class AbnormalRobotConfig:
    enabled: bool
    debug: bool


@dataclass
class TerminationsConfig:
    trajectory_deviation: TrajectoryDeviationConfig
    hand_pose_deviation: HandPoseDeviationConfig
    abnormal_robot: AbnormalRobotConfig


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
    grasp_scale: tuple[float, float]
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
    object_x: tuple[float, float] | None  # null = inherit workspace bounds
    object_y: tuple[float, float] | None  # null = inherit workspace bounds
    object_yaw: tuple[float, float]
    robot_joints: tuple[float, float]
    robot_wrist: tuple[float, float]
    z_offset: float
    robot_base_x: tuple[float, float]
    robot_base_y: tuple[float, float]
    robot_base_z: tuple[float, float]
    camera_forward: tuple[float, float]
    camera_lateral: tuple[float, float]
    camera_vertical: tuple[float, float]


@dataclass
class RandomizationConfig:
    object: ObjectRandomizationConfig
    robot: RobotRandomizationConfig
    reset: ResetRandomizationConfig


@dataclass
class CameraViewerConfig:
    enabled: bool
    port: int
    update_hz: int
    env_id: int


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
    contact_forces: bool
    camera_pose: bool
    env_ids: list[int] | None
    debug_print_rewards: bool
    camera_viewer: CameraViewerConfig


@dataclass
class SimulationConfig:
    physics_dt: float
    decimation: int
    num_envs: int
    env_spacing: float
    replicate_physics: bool
    solver_position_iteration_count: int
    solver_velocity_iteration_count: int


@dataclass
class PalmFrameOffsetConfig:
    pos: tuple[float, float, float]
    rot_euler: tuple[float, float, float]


@dataclass
class RobotConfig:
    action_scale: float
    arm_joint_count: int
    hand_joint_count: int
    eigen_dim: int
    palm_body_name: str
    wrist_joint_name: str
    arm_joint_regex: str
    hand_body_regex: str
    palm_frame_offset: PalmFrameOffsetConfig | None = None
    tip_offsets: dict[str, list[float]] | None = None


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
class VisibilityCameraConfig:
    width: int
    height: int
    focal_length: float
    horizontal_aperture: float
    clipping_range: tuple[float, float]
    distance: tuple[float, float]
    yaw: tuple[float, float]
    pitch: tuple[float, float]
    look_at_offset: tuple[float, float, float]  # (x, y, z) fixed offset from workspace center


@dataclass
class CameraOffsetConfig:
    pos: tuple[float, float, float]
    rot: tuple[float, float, float]


@dataclass
class WristCameraConfig:
    width: int
    height: int
    focal_length: float
    horizontal_aperture: float
    clipping_range: tuple[float, float]
    offset: CameraOffsetConfig


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
    visibility_camera: VisibilityCameraConfig


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

        # Parse hand_coupling_weights (common to all segment types)
        hcw = seg_dict.get("hand_coupling_weights", {"full": 1.0})
        valid_modes = {"full", "position_only", "none"}
        invalid = set(hcw.keys()) - valid_modes
        if invalid:
            raise ValueError(
                f"Invalid hand_coupling_weights keys in segment '{seg_dict['name']}': {invalid}. "
                f"Valid modes: {sorted(valid_modes)}"
            )

        if seg_type == "waypoint":
            expected = {"type", "name", "duration", "pose", "hand_coupling_weights", "hand_orientation", "curriculum"}
            _validate_segment_fields(seg_dict, expected, "waypoint")

            seg = WaypointSegmentConfig(
                name=seg_dict["name"],
                duration=seg_dict["duration"],
                pose=seg_dict.get("pose"),
                hand_coupling_weights=hcw,
                hand_orientation=seg_dict.get("hand_orientation"),
                curriculum=seg_dict.get("curriculum"),
            )
        elif seg_type == "helical":
            expected = {"type", "name", "duration", "axis", "angular_velocity", "translation", "hand_coupling_weights", "hand_orientation", "curriculum"}
            _validate_segment_fields(seg_dict, expected, "helical")

            seg = HelicalSegmentConfig(
                name=seg_dict["name"],
                duration=seg_dict["duration"],
                axis=seg_dict["axis"],
                angular_velocity=seg_dict["angular_velocity"],
                translation=seg_dict["translation"],
                hand_coupling_weights=hcw,
                hand_orientation=seg_dict.get("hand_orientation"),
                curriculum=seg_dict.get("curriculum"),
            )
        elif seg_type == "random_waypoint":
            expected = {"type", "name", "count_weights", "movement_duration", "pause_duration", "position_range",
                       "vary_orientation", "max_rotation", "hand_coupling_weights", "curriculum"}
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
                count_weights=seg_dict["count_weights"],
                movement_duration=seg_dict["movement_duration"],
                pause_duration=seg_dict["pause_duration"],
                position_range=pos_range,
                vary_orientation=seg_dict["vary_orientation"],
                max_rotation=seg_dict["max_rotation"],
                hand_coupling_weights=hcw,
                curriculum=seg_dict.get("curriculum"),
            )
        else:
            raise ValueError(f"Unknown segment type: {seg_type}")

        segments.append(seg)

    return segments


def get_config_file_paths(mode: str | None = None, task: str | None = None, robot: str | None = None) -> list[Path]:
    """Return list of YAML files that will be loaded for given mode/task/robot.

    This ensures wandb logging matches actual config composition by maintaining
    a single source of truth for which files are loaded.

    Args:
        mode: One of "train", "distill", "play", "play_student", "keyboard" (None = Y2R_MODE env var)
        task: Task name - "base" or any YAML file in configs/layers/tasks/ (None = Y2R_TASK env var)
        robot: Robot name - "ur5e_leap" or "kuka_allegro" (None = Y2R_ROBOT env var)

    Returns:
        List of Path objects for YAML files that will be loaded
    """
    mode, task, robot = _resolve_defaults(mode, task, robot)

    config_dir = Path(__file__).parent / "configs"
    files = [config_dir / "base.yaml"]

    # Robot layer (applied first, before mode/task)
    robot_layer = config_dir / "layers" / "robots" / f"{robot}.yaml"
    if robot_layer.exists():
        files.append(robot_layer)

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


def get_config(mode: str | None = None, task: str | None = None, robot: str | None = None) -> Y2RConfig:
    """Load Y2R config with layer-based composition.

    Layer order: base → robot → mode → task
    All parameters default to their respective env vars (Y2R_MODE, Y2R_TASK, Y2R_ROBOT).

    Args:
        mode: One of "train", "distill", "play", "play_student" (None = Y2R_MODE env var)
        task: Task name - "base" or tasks YAML name (None = Y2R_TASK env var)
        robot: Robot name - "ur5e_leap" or "kuka_allegro" (None = Y2R_ROBOT env var)

    Returns:
        Y2RConfig instance with layers composed
    """
    mode, task, robot = _resolve_defaults(mode, task, robot)

    cfg = _load_yaml("base")

    # Robot layer (applied first, before mode/task)
    robot_layer = Path(__file__).parent / "configs" / "layers" / "robots" / f"{robot}.yaml"
    if robot_layer.exists():
        cfg = _deep_merge(cfg, _load_yaml(f"layers/robots/{robot}"))

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
