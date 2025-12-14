# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration loader for Y2R task with YAML inheritance support."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ==============================================================================
# YAML Loading with Inheritance
# ==============================================================================

def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override dict into base dict.
    
    Args:
        base: Base dictionary to merge into.
        override: Override dictionary with values to apply.
        
    Returns:
        New merged dictionary (does not modify inputs).
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_name: str = "y2r_config") -> dict:
    """Load YAML config with inheritance support.
    
    Supports `_base_` key for inheriting from another config.
    
    Args:
        config_name: Name of config file (without .yaml extension).
        
    Returns:
        Merged configuration dictionary.
    """
    config_dir = Path(__file__).parent
    config_path = config_dir / f"{config_name}.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Handle inheritance
    if "_base_" in config:
        base_name = config.pop("_base_").replace(".yaml", "")
        base_config = load_config(base_name)
        config = deep_merge(base_config, config)
    
    return config


# ==============================================================================
# Config Singleton Cache
# ==============================================================================

_config_cache: dict[str, "Y2RConfig"] = {}


def get_config(config_name: str = "y2r_config", force_reload: bool = False) -> "Y2RConfig":
    """Get cached Y2RConfig instance.
    
    Args:
        config_name: Name of config file (without .yaml extension).
        force_reload: If True, reload from disk even if cached.
        
    Returns:
        Y2RConfig instance with all settings.
    """
    global _config_cache
    
    if force_reload or config_name not in _config_cache:
        raw = load_config(config_name)
        _config_cache[config_name] = Y2RConfig.from_dict(raw)
    
    return _config_cache[config_name]


def get_play_config() -> "Y2RConfig":
    """Convenience function to get play/evaluation config."""
    return get_config("y2r_play_config")


def get_push_config() -> "Y2RConfig":
    """Convenience function to get push-T evaluation config."""
    return get_config("y2r_push_config")


# ==============================================================================
# Type-Safe Config Classes
# ==============================================================================

@dataclass
class ModeConfig:
    """Mode selection configuration."""
    use_point_cloud: bool = False
    use_eigen_grasp: bool = True


@dataclass
class HistoryConfig:
    """History lengths for observation groups."""
    policy: int = 5
    proprio: int = 5
    perception: int = 5
    targets: int = 1


@dataclass
class ObservationsConfig:
    """Observation configuration."""
    num_points: int = 32
    clip_range: tuple[float, float] = (-2.0, 2.0)
    history: HistoryConfig = field(default_factory=HistoryConfig)


@dataclass
class PhasesConfig:
    """Phase duration configuration."""
    pickup: float = 0.5
    manipulate: float = 7.5
    release: float = 2.0


@dataclass
class TrajectoryConfig:
    """Trajectory timing configuration."""
    duration: float = 10.0
    target_hz: float = 6.667
    window_size: int = 5
    phases: PhasesConfig = field(default_factory=PhasesConfig)
    easing_power: float = 2.0


@dataclass
class WorkspaceConfig:
    """Workspace bounds configuration."""
    x: tuple[float, float] = (-0.8, -0.3)
    y: tuple[float, float] = (-0.35, 0.35)
    z: tuple[float, float] = (0.28, 0.75)
    table_surface_z: float = 0.255


@dataclass
class PositionRangeConfig:
    """Waypoint position range configuration."""
    x: tuple[float, float] | None = None
    y: tuple[float, float] | None = None
    z: tuple[float, float] | None = None


@dataclass
class WaypointsConfig:
    """Waypoint configuration."""
    count: tuple[int, int] = (0, 2)
    position_range: PositionRangeConfig = field(default_factory=PositionRangeConfig)
    vary_orientation: bool = True
    max_rotation: float = 3.14


@dataclass
class GoalConfig:
    """Goal sampling configuration."""
    x_range: tuple[float, float] | None = None
    y_range: tuple[float, float] | None = None
    table_margin: float = 0.0


@dataclass
class RewardConfig:
    """Individual reward configuration."""
    weight: float = 0.0
    # Additional params stored as dict
    params: dict = field(default_factory=dict)


@dataclass
class RewardsConfig:
    """All rewards configuration."""
    action_l2: RewardConfig = field(default_factory=lambda: RewardConfig(weight=-0.005))
    action_rate_l2: RewardConfig = field(default_factory=lambda: RewardConfig(weight=-0.005))
    fingers_to_object: RewardConfig = field(default_factory=lambda: RewardConfig(
        weight=1.0,
        params={
            "std": 0.4,
            # When tracking error is large, reduce this reward to avoid squeeze-and-freeze.
            # Defaults: disabled (thresholds None).
            "error_gate_pos_threshold": None,
            "error_gate_pos_slope": 0.02,
            "error_gate_rot_threshold": None,
            "error_gate_rot_slope": 0.5,
        },
    ))
    lookahead_tracking: RewardConfig = field(default_factory=lambda: RewardConfig(
        weight=5.0, params={
            "std": 0.03,
            "decay": 0.2,
            "contact_threshold": 2.0,
            "contact_ramp": 1.0,
            "contact_min_factor": 0.05,
            "rot_std": 0.3,
        }
    ))
    trajectory_success: RewardConfig = field(default_factory=lambda: RewardConfig(
        weight=10.0, params={
            "point_cloud_threshold": (0.06, 0.02),
            "position_threshold": (0.08, 0.02),
            "rotation_threshold": (0.8, 0.2),
        }
    ))
    early_termination: RewardConfig = field(default_factory=lambda: RewardConfig(
        weight=-5.0, params={"term_keys": ["abnormal_robot", "trajectory_deviation"]}
    ))
    arm_table_penalty: RewardConfig = field(default_factory=lambda: RewardConfig(
        weight=-1.0, params={"table_z": 0.255, "threshold_mid": 0.06, "threshold_distal": 0.03}
    ))
    good_finger_contact: RewardConfig = field(default_factory=lambda: RewardConfig(
        weight=0.5, params={"threshold": 1.0}
    ))
    finger_manipulation: RewardConfig = field(default_factory=lambda: RewardConfig(
        weight=0.5, params={"pos_std": 0.01, "rot_std": 0.1}
    ))
    palm_velocity_penalty: RewardConfig = field(default_factory=lambda: RewardConfig(
        weight=-0.3, params={"angular_std": 0.5, "linear_std": 0.3, "linear_scale": 0.2}
    ))
    palm_orientation_penalty: RewardConfig = field(default_factory=lambda: RewardConfig(
        weight=-0.1, params={"std": 0.5}
    ))
    distal_joint3_penalty: RewardConfig = field(default_factory=lambda: RewardConfig(
        weight=-0.3,
        params={"std": 1.0, "joint_name_regex": ".*_joint_3", "only_when_contact": True, "contact_threshold": 1.0, "only_in_manipulation": True,
        },
    ))
    joint_limits_margin: RewardConfig = field(default_factory=lambda: RewardConfig(
        weight=-0.05, params={"threshold": 0.95, "power": 2.0}
    ))
    tracking_progress: RewardConfig = field(default_factory=lambda: RewardConfig(
        weight=0.0,
        params={
            "pos_weight": 1.0,
            "rot_weight": 0.5,
            "positive_only": False,
            "clip": 1.0,
        },
    ))


@dataclass
class TrajectoryDeviationConfig:
    """Trajectory deviation termination thresholds."""
    point_cloud_threshold: tuple[float, float] = (0.18, 0.06)
    position_threshold: tuple[float, float] = (0.25, 0.05)
    rotation_threshold: tuple[float, float] = (2.5, 0.5)


@dataclass
class TerminationsConfig:
    """Termination conditions configuration."""
    trajectory_deviation: TrajectoryDeviationConfig = field(default_factory=TrajectoryDeviationConfig)


@dataclass
class DifficultyConfig:
    """Curriculum difficulty configuration."""
    initial: int = 0
    min: int = 0
    max: int = 10


@dataclass
class AdvancementTolerancesConfig:
    """Tolerances for advancing difficulty.
    
    When None, uses the current success threshold (adaptive with curriculum).
    """
    position: float | None = None
    rotation: float | None = None
    point_cloud: float | None = None


@dataclass
class NoiseConfig:
    """Observation noise curriculum."""
    joint_pos: tuple[float, float] = (0.0, 0.1)
    joint_vel: tuple[float, float] = (0.0, 0.2)
    hand_tips: tuple[float, float] = (0.0, 0.01)
    object_point_cloud: tuple[float, float] = (0.0, 0.01)
    object_pose: tuple[float, float] = (0.0, 0.01)
    target_point_clouds: tuple[float, float] = (0.0, 0.01)
    target_poses: tuple[float, float] = (0.0, 0.01)


@dataclass
class GravityConfig:
    """Gravity curriculum."""
    initial: tuple[float, float, float] = (0.0, 0.0, 0.0)
    final: tuple[float, float, float] = (0.0, 0.0, -9.81)


@dataclass
class CurriculumConfig:
    """Full curriculum configuration."""
    difficulty: DifficultyConfig = field(default_factory=DifficultyConfig)
    advancement_tolerances: AdvancementTolerancesConfig = field(default_factory=AdvancementTolerancesConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    gravity: GravityConfig = field(default_factory=GravityConfig)


@dataclass
class ObjectRandomizationConfig:
    """Object randomization configuration."""
    scale: tuple[float, float] = (0.75, 1.5)
    mass_scale: tuple[float, float] = (0.2, 2.0)
    static_friction: tuple[float, float] = (0.5, 1.0)
    dynamic_friction: tuple[float, float] = (0.5, 1.0)
    restitution: tuple[float, float] = (0.0, 0.0)


@dataclass
class RobotRandomizationConfig:
    """Robot randomization configuration."""
    static_friction: tuple[float, float] = (0.5, 1.0)
    dynamic_friction: tuple[float, float] = (0.5, 1.0)
    restitution: tuple[float, float] = (0.0, 0.0)
    stiffness_scale: tuple[float, float] = (0.5, 2.0)
    damping_scale: tuple[float, float] = (0.5, 2.0)
    joint_friction_scale: tuple[float, float] = (0.0, 5.0)


@dataclass
class ResetRandomizationConfig:
    """Reset randomization configuration."""
    table_xy: tuple[float, float] = (-0.05, 0.05)
    object_x: tuple[float, float] = (-0.25, 0.25)
    object_y: tuple[float, float] = (-0.35, 0.35)
    object_yaw: tuple[float, float] = (-3.14, 3.14)
    robot_joints: tuple[float, float] = (-0.5, 0.5)
    robot_wrist: tuple[float, float] = (-3.0, 3.0)
    z_offset: float = 0.0005


@dataclass
class RandomizationConfig:
    """Full randomization configuration."""
    object: ObjectRandomizationConfig = field(default_factory=ObjectRandomizationConfig)
    robot: RobotRandomizationConfig = field(default_factory=RobotRandomizationConfig)
    reset: ResetRandomizationConfig = field(default_factory=ResetRandomizationConfig)


@dataclass
class VisualizationConfig:
    """Visualization configuration."""
    targets: bool = True
    current_object: bool = True
    waypoint_region: bool = True
    goal_region: bool = True
    pose_axes: bool = True
    env_ids: list[int] | None = None  # None=all envs, []=no envs, [0,1,2]=specific
    debug_print_rewards: bool = False


@dataclass
class SimulationConfig:
    """Simulation configuration."""
    physics_dt: float = 0.00833
    decimation: int = 2
    num_envs: int = 4096
    env_spacing: float = 3.0
    replicate_physics: bool = False


@dataclass
class RobotConfig:
    """Robot-specific configuration."""
    action_scale: float = 0.1
    arm_joint_count: int = 7
    hand_joint_count: int = 16
    eigen_dim: int = 5


@dataclass
class PushTConfig:
    """Push-T mode configuration."""
    enabled: bool = False
    object_usd: str | None = None
    outline_usd: str | None = None
    object_scale: float = 1.0
    outline_position: tuple[float, float] = (-0.55, 0.0)
    episode_duration: float = 20.0  # Episode timeout (seconds)
    rolling_window: float = 3.0     # Trajectory planning horizon (seconds)
    min_speed: float = 0.02         # Minimum speed (m/s)
    include_in_primitives: bool = False  # Include T-shape in random object selection


@dataclass
class GenerationConfig:
    """Generation parameters for procedural shapes."""
    num_shapes: int = 100
    primitives_per_shape: tuple[int, int] = (3, 8)
    base_size: tuple[float, float] = (0.03, 0.08)
    size_decay: float = 0.8
    primitive_types: dict[str, float] = field(default_factory=lambda: {
        "ellipsoid": 0.4,
        "box": 0.3,
        "capsule": 0.2,
        "cylinder": 0.1,
    })
    seed: int | None = None


@dataclass
class ProceduralObjectsConfig:
    """Procedural object generation configuration."""
    enabled: bool = True
    percentage: float = 0.5  # Fraction of shapes that are procedural
    asset_dir: str = "assets/procedural"
    regenerate: bool = False
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    _raw: dict = field(default_factory=dict)  # Store raw dict for procedural_shapes.py


@dataclass
class Y2RConfig:
    """Complete Y2R task configuration.
    
    Access config values like:
        cfg = get_config()
        window_size = cfg.trajectory.window_size
        use_pc = cfg.mode.use_point_cloud
        
        # Success thresholds from rewards
        success_pc = cfg.rewards.trajectory_success.params["point_cloud_threshold"]
        
        # Termination thresholds
        term_pc = cfg.terminations.trajectory_deviation.point_cloud_threshold
        
        # Push-T mode
        if cfg.push_t.enabled:
            object_usd = cfg.push_t.object_usd
    """
    mode: ModeConfig = field(default_factory=ModeConfig)
    observations: ObservationsConfig = field(default_factory=ObservationsConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    waypoints: WaypointsConfig = field(default_factory=WaypointsConfig)
    goal: GoalConfig = field(default_factory=GoalConfig)
    rewards: RewardsConfig = field(default_factory=RewardsConfig)
    terminations: TerminationsConfig = field(default_factory=TerminationsConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    randomization: RandomizationConfig = field(default_factory=RandomizationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    push_t: PushTConfig = field(default_factory=PushTConfig)
    procedural_objects: ProceduralObjectsConfig = field(default_factory=ProceduralObjectsConfig)
    
    @classmethod
    def from_dict(cls, data: dict) -> "Y2RConfig":
        """Create Y2RConfig from dictionary.
        
        Args:
            data: Raw config dictionary from YAML.
            
        Returns:
            Populated Y2RConfig instance.
        """
        return cls(
            mode=_parse_mode(data.get("mode", {})),
            observations=_parse_observations(data.get("observations", {})),
            trajectory=_parse_trajectory(data.get("trajectory", {})),
            workspace=_parse_workspace(data.get("workspace", {})),
            waypoints=_parse_waypoints(data.get("waypoints", {})),
            goal=_parse_goal(data.get("goal", {})),
            rewards=_parse_rewards(data.get("rewards", {})),
            terminations=_parse_terminations(data.get("terminations", {})),
            curriculum=_parse_curriculum(data.get("curriculum", {})),
            randomization=_parse_randomization(data.get("randomization", {})),
            visualization=_parse_visualization(data.get("visualization", {})),
            simulation=_parse_simulation(data.get("simulation", {})),
            robot=_parse_robot(data.get("robot", {})),
            push_t=_parse_push_t(data.get("push_t", {})),
            procedural_objects=_parse_procedural_objects(data.get("procedural_objects", {})),
        )


# ==============================================================================
# Parser Helpers
# ==============================================================================

def _to_tuple(val: list | tuple | None, default: tuple = None) -> tuple | None:
    """Convert list to tuple, or return None."""
    if val is None:
        return default
    return tuple(val) if isinstance(val, list) else val


def _parse_mode(data: dict) -> ModeConfig:
    return ModeConfig(
        use_point_cloud=data.get("use_point_cloud", False),
        use_eigen_grasp=data.get("use_eigen_grasp", True),
    )


def _parse_observations(data: dict) -> ObservationsConfig:
    history_data = data.get("history", {})
    return ObservationsConfig(
        num_points=data.get("num_points", 32),
        clip_range=_to_tuple(data.get("clip_range"), (-2.0, 2.0)),
        history=HistoryConfig(
            policy=history_data.get("policy", 5),
            proprio=history_data.get("proprio", 5),
            perception=history_data.get("perception", 5),
            targets=history_data.get("targets", 1),
        ),
    )


def _parse_trajectory(data: dict) -> TrajectoryConfig:
    phases_data = data.get("phases", {})
    return TrajectoryConfig(
        duration=data.get("duration", 10.0),
        target_hz=data.get("target_hz", 6.667),
        window_size=data.get("window_size", 5),
        phases=PhasesConfig(
            pickup=phases_data.get("pickup", 0.5),
            manipulate=phases_data.get("manipulate", 7.5),
            release=phases_data.get("release", 2.0),
        ),
        easing_power=data.get("easing_power", 2.0),
    )


def _parse_workspace(data: dict) -> WorkspaceConfig:
    return WorkspaceConfig(
        x=_to_tuple(data.get("x"), (-0.8, -0.3)),
        y=_to_tuple(data.get("y"), (-0.35, 0.35)),
        z=_to_tuple(data.get("z"), (0.28, 0.75)),
        table_surface_z=data.get("table_surface_z", 0.255),
    )


def _parse_waypoints(data: dict) -> WaypointsConfig:
    pos_data = data.get("position_range", {})
    return WaypointsConfig(
        count=_to_tuple(data.get("count"), (0, 2)),
        position_range=PositionRangeConfig(
            x=_to_tuple(pos_data.get("x")),
            y=_to_tuple(pos_data.get("y")),
            z=_to_tuple(pos_data.get("z")),
        ),
        vary_orientation=data.get("vary_orientation", True),
        max_rotation=data.get("max_rotation", 3.14),
    )


def _parse_goal(data: dict) -> GoalConfig:
    return GoalConfig(
        x_range=_to_tuple(data.get("x_range")),
        y_range=_to_tuple(data.get("y_range")),
        table_margin=data.get("table_margin", 0.0),
    )


def _parse_rewards(data: dict) -> RewardsConfig:
    def parse_reward(name: str, defaults: dict) -> RewardConfig:
        r = data.get(name, {})
        weight = r.get("weight", defaults.get("weight", 0.0))
        # All other keys go into params, converting lists to tuples
        params = {}
        for k, v in r.items():
            if k != "weight":
                params[k] = tuple(v) if isinstance(v, list) else v
        # Merge with defaults
        for k, v in defaults.items():
            if k != "weight" and k not in params:
                params[k] = v
        return RewardConfig(weight=weight, params=params)
    
    return RewardsConfig(
        action_l2=parse_reward("action_l2", {"weight": -0.005}),
        action_rate_l2=parse_reward("action_rate_l2", {"weight": -0.005}),
        fingers_to_object=parse_reward("fingers_to_object", {
            "weight": 1.0,
            "std": 0.4,
            "error_gate_pos_threshold": None,
            "error_gate_pos_slope": 0.02,
            "error_gate_rot_threshold": None,
            "error_gate_rot_slope": 0.5,
        }),
        lookahead_tracking=parse_reward("lookahead_tracking", {
            "weight": 5.0,
            "std": 0.03,
            "decay": 0.2,
            "contact_threshold": 2.0,
            "contact_ramp": 1.0,
            "contact_min_factor": 0.05,
            "rot_std": 0.3,
        }),
        trajectory_success=parse_reward("trajectory_success", {
            "weight": 10.0,
            "point_cloud_threshold": (0.06, 0.02),
            "position_threshold": (0.08, 0.02),
            "rotation_threshold": (0.8, 0.2),
        }),
        early_termination=parse_reward("early_termination", {
            "weight": -5.0, "term_keys": ["abnormal_robot", "trajectory_deviation"]
        }),
        arm_table_penalty=parse_reward("arm_table_penalty", {
            "weight": -1.0, "table_z": 0.255, "threshold_mid": 0.06, "threshold_distal": 0.03
        }),
        good_finger_contact=parse_reward("good_finger_contact", {"weight": 0.5, "threshold": 1.0}),
        finger_manipulation=parse_reward("finger_manipulation", {"weight": 0.5, "pos_std": 0.01, "rot_std": 0.1}),
        palm_velocity_penalty=parse_reward("palm_velocity_penalty", {
            "weight": -0.3, "angular_std": 0.5, "linear_std": 0.3, "linear_scale": 0.2
        }),
        palm_orientation_penalty=parse_reward("palm_orientation_penalty", {"weight": -0.1, "std": 0.5}),
        distal_joint3_penalty=parse_reward("distal_joint3_penalty", {
            "weight": -0.3,
            "std": 1.0,
            "joint_name_regex": ".*_joint_3",
            "only_when_contact": True,
            "contact_threshold": 1.0,
            "only_in_manipulation": True,
        }),
        joint_limits_margin=parse_reward("joint_limits_margin", {"weight": -0.05, "threshold": 0.95, "power": 2.0}),
        tracking_progress=parse_reward(
            "tracking_progress",
            {
                "weight": 0.0,
                "pos_weight": 1.0,
                "rot_weight": 0.5,
                "positive_only": False,
                "clip": 1.0,
            },
        ),
    )


def _parse_terminations(data: dict) -> TerminationsConfig:
    td_data = data.get("trajectory_deviation", {})
    return TerminationsConfig(
        trajectory_deviation=TrajectoryDeviationConfig(
            point_cloud_threshold=_to_tuple(td_data.get("point_cloud_threshold"), (0.18, 0.06)),
            position_threshold=_to_tuple(td_data.get("position_threshold"), (0.25, 0.05)),
            rotation_threshold=_to_tuple(td_data.get("rotation_threshold"), (2.5, 0.5)),
        ),
    )


def _parse_curriculum(data: dict) -> CurriculumConfig:
    diff_data = data.get("difficulty", {})
    adv_data = data.get("advancement_tolerances", {})
    noise_data = data.get("noise", {})
    grav_data = data.get("gravity", {})
    
    return CurriculumConfig(
        difficulty=DifficultyConfig(
            initial=diff_data.get("initial", 0),
            min=diff_data.get("min", 0),
            max=diff_data.get("max", 10),
        ),
        advancement_tolerances=AdvancementTolerancesConfig(
            position=adv_data.get("position"),  # None = use success threshold
            rotation=adv_data.get("rotation"),  # None = use success threshold
            point_cloud=adv_data.get("point_cloud"),  # None = use success threshold
        ),
        noise=NoiseConfig(
            joint_pos=_to_tuple(noise_data.get("joint_pos"), (0.0, 0.1)),
            joint_vel=_to_tuple(noise_data.get("joint_vel"), (0.0, 0.2)),
            hand_tips=_to_tuple(noise_data.get("hand_tips"), (0.0, 0.01)),
            object_point_cloud=_to_tuple(noise_data.get("object_point_cloud"), (0.0, 0.01)),
            object_pose=_to_tuple(noise_data.get("object_pose"), (0.0, 0.01)),
            target_point_clouds=_to_tuple(noise_data.get("target_point_clouds"), (0.0, 0.01)),
            target_poses=_to_tuple(noise_data.get("target_poses"), (0.0, 0.01)),
        ),
        gravity=GravityConfig(
            initial=_to_tuple(grav_data.get("initial"), (0.0, 0.0, 0.0)),
            final=_to_tuple(grav_data.get("final"), (0.0, 0.0, -9.81)),
        ),
    )


def _parse_randomization(data: dict) -> RandomizationConfig:
    obj_data = data.get("object", {})
    robot_data = data.get("robot", {})
    reset_data = data.get("reset", {})
    
    return RandomizationConfig(
        object=ObjectRandomizationConfig(
            scale=_to_tuple(obj_data.get("scale"), (0.75, 1.5)),
            mass_scale=_to_tuple(obj_data.get("mass_scale"), (0.2, 2.0)),
            static_friction=_to_tuple(obj_data.get("static_friction"), (0.5, 1.0)),
            dynamic_friction=_to_tuple(obj_data.get("dynamic_friction"), (0.5, 1.0)),
            restitution=_to_tuple(obj_data.get("restitution"), (0.0, 0.0)),
        ),
        robot=RobotRandomizationConfig(
            static_friction=_to_tuple(robot_data.get("static_friction"), (0.5, 1.0)),
            dynamic_friction=_to_tuple(robot_data.get("dynamic_friction"), (0.5, 1.0)),
            restitution=_to_tuple(robot_data.get("restitution"), (0.0, 0.0)),
            stiffness_scale=_to_tuple(robot_data.get("stiffness_scale"), (0.5, 2.0)),
            damping_scale=_to_tuple(robot_data.get("damping_scale"), (0.5, 2.0)),
            joint_friction_scale=_to_tuple(robot_data.get("joint_friction_scale"), (0.0, 5.0)),
        ),
        reset=ResetRandomizationConfig(
            table_xy=_to_tuple(reset_data.get("table_xy"), (-0.05, 0.05)),
            object_x=_to_tuple(reset_data.get("object_x"), (-0.25, 0.25)),
            object_y=_to_tuple(reset_data.get("object_y"), (-0.35, 0.35)),
            object_yaw=_to_tuple(reset_data.get("object_yaw"), (-3.14, 3.14)),
            robot_joints=_to_tuple(reset_data.get("robot_joints"), (-0.5, 0.5)),
            robot_wrist=_to_tuple(reset_data.get("robot_wrist"), (-3.0, 3.0)),
            z_offset=reset_data.get("z_offset", 0.0005),
        ),
    )


def _parse_visualization(data: dict) -> VisualizationConfig:
    env_ids = data.get("env_ids")  # Keep None as None, [] as []
    if env_ids is not None and not isinstance(env_ids, list):
        env_ids = list(env_ids)
    return VisualizationConfig(
        targets=data.get("targets", True),
        current_object=data.get("current_object", True),
        waypoint_region=data.get("waypoint_region", True),
        goal_region=data.get("goal_region", True),
        pose_axes=data.get("pose_axes", True),
        env_ids=env_ids,  # Pass through: None=all, []=none, [x]=specific
        debug_print_rewards=data.get("debug_print_rewards", False),
    )


def _parse_simulation(data: dict) -> SimulationConfig:
    return SimulationConfig(
        physics_dt=data.get("physics_dt", 0.00833),
        decimation=data.get("decimation", 2),
        num_envs=data.get("num_envs", 4096),
        env_spacing=data.get("env_spacing", 3.0),
        replicate_physics=data.get("replicate_physics", False),
    )


def _parse_robot(data: dict) -> RobotConfig:
    return RobotConfig(
        action_scale=data.get("action_scale", 0.1),
        arm_joint_count=data.get("arm_joint_count", 7),
        hand_joint_count=data.get("hand_joint_count", 16),
        eigen_dim=data.get("eigen_dim", 5),
    )


def _parse_push_t(data: dict) -> PushTConfig:
    return PushTConfig(
        enabled=data.get("enabled", False),
        object_usd=data.get("object_usd"),
        outline_usd=data.get("outline_usd"),
        object_scale=data.get("object_scale", 1.0),
        outline_position=_to_tuple(data.get("outline_position"), (-0.55, 0.0)),
        episode_duration=data.get("episode_duration", 20.0),
        rolling_window=data.get("rolling_window", 3.0),
        min_speed=data.get("min_speed", 0.02),
        include_in_primitives=data.get("include_in_primitives", False),
    )


def _parse_procedural_objects(data: dict) -> ProceduralObjectsConfig:
    gen_data = data.get("generation", {})
    return ProceduralObjectsConfig(
        enabled=data.get("enabled", True),
        percentage=data.get("percentage", 0.5),
        asset_dir=data.get("asset_dir", "assets/procedural"),
        regenerate=data.get("regenerate", False),
        generation=GenerationConfig(
            num_shapes=gen_data.get("num_shapes", 100),
            primitives_per_shape=_to_tuple(gen_data.get("primitives_per_shape"), (3, 8)),
            base_size=_to_tuple(gen_data.get("base_size"), (0.03, 0.08)),
            size_decay=gen_data.get("size_decay", 0.8),
            primitive_types=gen_data.get("primitive_types", {
                "ellipsoid": 0.4,
                "box": 0.3,
                "capsule": 0.2,
                "cylinder": 0.1,
            }),
            seed=gen_data.get("seed"),
        ),
        _raw=data,  # Store raw dict for procedural_shapes.py
    )
