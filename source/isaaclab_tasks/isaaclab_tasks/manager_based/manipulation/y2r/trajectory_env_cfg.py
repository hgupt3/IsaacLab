# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trajectory following task for dexterous manipulation.

This task trains the robot to follow a sequence of targets along a smooth trajectory.
The trajectory includes pickup from table, variable-speed manipulation, and place-back phases.

Supports two representation modes (controlled by config mode.use_point_cloud):
- Point cloud mode (default): Uses point-to-point distance for tracking
- Pose mode: Uses position + rotation (quaternion) for tracking
"""

from dataclasses import MISSING
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import CapsuleCfg, ConeCfg, CuboidCfg, CylinderCfg, RigidBodyMaterialCfg, SphereCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp
from .adr_curriculum import build_curriculum_cfg
from .config_loader import get_config, Y2RConfig
from .mdp.utils import compute_z_offset_from_usd
from .procedural_shapes import get_procedural_shape_paths


# ==============================================================================
# SCENE CONFIG (static - no config dependencies)
# ==============================================================================

# Built-in primitive asset configs
_PRIMITIVE_ASSETS = [
    CuboidCfg(size=(0.05, 0.1, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
    CuboidCfg(size=(0.05, 0.05, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
    CuboidCfg(size=(0.025, 0.1, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
    CuboidCfg(size=(0.025, 0.05, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
    CuboidCfg(size=(0.025, 0.025, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
    CuboidCfg(size=(0.01, 0.1, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
    SphereCfg(radius=0.035, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
    SphereCfg(radius=0.065, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
    CapsuleCfg(radius=0.04, height=0.025, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
    CapsuleCfg(radius=0.04, height=0.1, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
    CapsuleCfg(radius=0.025, height=0.1, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
    CapsuleCfg(radius=0.025, height=0.2, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
    CapsuleCfg(radius=0.01, height=0.2, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
    CylinderCfg(radius=0.05, height=0.1, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
    CylinderCfg(radius=0.025, height=0.15, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
    CylinderCfg(radius=0.05, height=0.05, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
    CylinderCfg(radius=0.05, height=0.025, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
    CuboidCfg(size=(0.06, 0.06, 0.06), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
    CuboidCfg(size=(0.025, 0.025, 0.025), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
]


def _build_object_cfg(cfg: Y2RConfig) -> RigidObjectCfg:
    """Build object config with mix of primitives and procedural shapes.
    
    If procedural_objects.enabled is True, generates/loads procedural shapes
    and mixes them with built-in primitives according to the percentage setting.
    
    If push_t.include_in_primitives is True, adds the T-shape USD to the primitives pool.
    """
    import math
    
    y2r_dir = Path(__file__).parent
    proc_cfg = cfg.procedural_objects
    
    # Start with primitive assets
    primitive_assets = list(_PRIMITIVE_ASSETS)
    procedural_assets = []
    
    # Add T-shape to primitives if configured
    if cfg.push_t.include_in_primitives and cfg.push_t.object_usd:
        t_shape_path = y2r_dir / cfg.push_t.object_usd
        if t_shape_path.exists():
            t_shape_cfg = sim_utils.UsdFileCfg(
                usd_path=str(t_shape_path),
                scale=(cfg.push_t.object_scale,) * 3,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=0,
                    disable_gravity=False,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            )
            primitive_assets.append(t_shape_cfg)
    
    # Add procedural shapes if enabled
    if proc_cfg.enabled:
        shape_paths = get_procedural_shape_paths(proc_cfg, y2r_dir)
        
        if shape_paths:
            # Create UsdFileCfg for each procedural shape
            for path in shape_paths:
                usd_cfg = sim_utils.UsdFileCfg(
                    usd_path=str(path),
                    scale=(1.0, 1.0, 1.0),  # Must specify scale for randomize_rigid_body_scale to work
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=0,
                        disable_gravity=False,
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
                )
                procedural_assets.append(usd_cfg)
    
    # Build final asset list based on percentage
    # Since we can't weight, we repeat assets to achieve desired ratio
    pct = proc_cfg.percentage if proc_cfg.enabled and procedural_assets else 0.0
    
    if pct >= 1.0:
        # All procedural
        all_assets = procedural_assets
    elif pct <= 0.0 or not procedural_assets:
        # All primitives
        all_assets = primitive_assets
    else:
        # Mix: repeat assets to approximate the desired ratio
        # Target: procedural_count / total_count ≈ pct
        n_prim = len(primitive_assets)
        n_proc = len(procedural_assets)
        
        # Find repeat counts to achieve ratio
        # We want: (n_proc * r_proc) / (n_prim * r_prim + n_proc * r_proc) = pct
        # Simplify by setting r_prim = 1 and solving for r_proc:
        # r_proc = pct * n_prim / ((1 - pct) * n_proc)
        if pct < 1.0:
            r_proc = pct * n_prim / ((1 - pct) * n_proc)
            r_proc = max(1, round(r_proc))  # At least 1 repeat
        else:
            r_proc = 1
        
        all_assets = primitive_assets + procedural_assets * r_proc
    
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=all_assets,
            random_choice=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            activate_contact_sensors=True,  # Required for object_table_contact sensor
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.55, 0.0, 0.45)),
    )


@configclass
class TrajectorySceneCfg(InteractiveSceneCfg):
    """Scene for trajectory following task with object starting on table."""

    # robot
    robot: ArticulationCfg = MISSING

    # object - default config, overridden in TrajectoryEnvCfg.__post_init__
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=_PRIMITIVE_ASSETS,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            activate_contact_sensors=True,  # Required for object_table_contact sensor
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.55, 0.0, 0.45)),
    )

    # table
    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/table",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 1.5, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.3, 0.2), roughness=0.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.55, 0.0, 0.235), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(),
        spawn=sim_utils.GroundPlaneCfg(),
        collision_group=-1,
    )

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # Visual outline for push-T mode (set dynamically in TrajectoryEnvCfg_PUSH)
    outline: AssetBaseCfg | None = None


@configclass
class CommandsCfg:
    """Command terms for the MDP (empty for trajectory task - uses TrajectoryManager instead)."""
    pass


@configclass
class ActionsCfg:
    pass


# ==============================================================================
# CONFIG BUILDERS - Create manager configs dynamically from Y2RConfig
# ==============================================================================

def _build_observations_cfg(cfg: Y2RConfig):
    """Build ObservationsCfg from config."""
    
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = cfg.observations.history.policy

    @configclass
    class ProprioObsCfg(ObsGroup):
        """Observations for proprioception group."""
        joint_pos = ObsTerm(func=mdp.joint_pos, noise=Unoise(n_min=-0.0, n_max=0.0))
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-0.0, n_max=0.0))
        hand_eigen = ObsTerm(
            func=mdp.allegro_hand_eigen_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={
                "arm_joint_count": cfg.robot.arm_joint_count,
                "hand_joint_count": cfg.robot.hand_joint_count,
                "eigen_dim": cfg.robot.eigen_dim,
                "use_default_delta": True,
            },
        )
        hand_tips_state_b = ObsTerm(
            func=mdp.body_state_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={
                "body_asset_cfg": SceneEntityCfg("robot"),
                "base_asset_cfg": SceneEntityCfg("robot"),
            },
        )
        contact: ObsTerm = MISSING

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = cfg.observations.history.proprio

    @configclass
    class PerceptionObsCfg(ObsGroup):
        """Perception observations: current object point cloud and pose with history."""
        object_point_cloud = ObsTerm(
            func=mdp.object_point_cloud_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={
                "num_points": cfg.observations.num_points,
                "flatten": True,
            },
        )
        object_pose = ObsTerm(
            func=mdp.object_pose_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_dim = 0
            self.concatenate_terms = True
            self.flatten_history_dim = True
            self.history_length = cfg.observations.history.perception
    
    @configclass
    class TargetObsCfg(ObsGroup):
        """Target observations: point clouds AND poses from trajectory."""
        target_point_clouds = ObsTerm(
            func=mdp.target_sequence_obs_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={},
        )
        target_poses = ObsTerm(
            func=mdp.target_sequence_poses_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
        )
        hand_pose_targets = ObsTerm(
            func=mdp.hand_pose_targets_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_dim = 0
            self.concatenate_terms = True
            self.flatten_history_dim = True
            self.history_length = cfg.observations.history.targets

    @configclass
    class StudentPerceptionObsCfg(ObsGroup):
        """Student perception: visible-only object point cloud from pseudo-camera viewpoint."""
        visible_point_cloud = ObsTerm(
            func=mdp.visible_object_point_cloud_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={
                "num_points": cfg.observations.student_num_points,
                "flatten": True,
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_dim = 0
            self.concatenate_terms = True
            self.flatten_history_dim = True
            self.history_length = 1

    @configclass
    class StudentTargetObsCfg(ObsGroup):
        """Student targets: visible point clouds through target trajectory."""
        visible_target_sequence = ObsTerm(
            func=mdp.visible_target_sequence_obs_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={},
        )
        hand_pose_targets = ObsTerm(
            func=mdp.hand_pose_targets_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_dim = 0
            self.concatenate_terms = True
            self.flatten_history_dim = True
            self.history_length = cfg.observations.history.targets

    @configclass
    class StudentCameraObsCfg(ObsGroup):
        """Student camera: wrist-mounted 32x32 depth."""
        wrist_depth = ObsTerm(
            func=mdp.wrist_depth_image,
            params={"sensor_cfg": SceneEntityCfg("wrist_camera")},
        )

        def __post_init__(self):
            self.enable_corruption = False  # No noise on raw camera
            self.concatenate_terms = True
            self.history_length = 1  # No history for camera

    @configclass
    class ObservationsCfg:
        """Observation specifications for the MDP."""
        policy: PolicyCfg = PolicyCfg()
        proprio: ProprioObsCfg = ProprioObsCfg()
        perception: PerceptionObsCfg = PerceptionObsCfg()
        targets: TargetObsCfg = TargetObsCfg()
        # Student groups (only when mode.use_student_mode is True)
        student_perception: StudentPerceptionObsCfg | None = None
        student_targets: StudentTargetObsCfg | None = None
        student_camera: StudentCameraObsCfg | None = None

    obs_cfg = ObservationsCfg()
    if cfg.mode.use_student_mode:
        obs_cfg.student_perception = StudentPerceptionObsCfg()
        obs_cfg.student_targets = StudentTargetObsCfg()
    if cfg.wrist_camera.enabled:
        obs_cfg.student_camera = StudentCameraObsCfg()
    return obs_cfg


def _build_events_cfg(cfg: Y2RConfig):
    """Build EventCfg from config."""
    
    @configclass
    class EventCfg:
        """Configuration for randomization."""

        randomize_object_scale = EventTerm(
            func=mdp.randomize_rigid_body_scale,
            mode="prestartup",
            params={
                "scale_range": list(cfg.randomization.object.scale),
                "asset_cfg": SceneEntityCfg("object"),
            },
        )

        robot_physics_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": list(cfg.randomization.robot.static_friction),
                "dynamic_friction_range": list(cfg.randomization.robot.dynamic_friction),
                "restitution_range": list(cfg.randomization.robot.restitution),
                "num_buckets": 250,
            },
        )

        object_physics_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("object", body_names=".*"),
                "static_friction_range": list(cfg.randomization.object.static_friction),
                "dynamic_friction_range": list(cfg.randomization.object.dynamic_friction),
                "restitution_range": list(cfg.randomization.object.restitution),
                "num_buckets": 250,
            },
        )

        joint_stiffness_and_damping = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stiffness_distribution_params": list(cfg.randomization.robot.stiffness_scale),
                "damping_distribution_params": list(cfg.randomization.robot.damping_scale),
                "operation": "scale",
            },
        )

        joint_friction = EventTerm(
            func=mdp.randomize_joint_parameters,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "friction_distribution_params": list(cfg.randomization.robot.joint_friction_scale),
                "operation": "scale",
            },
        )

        object_scale_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("object"),
                "mass_distribution_params": list(cfg.randomization.object.mass_scale),
                "operation": "scale",
            },
        )

        reset_table = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": list(cfg.randomization.reset.table_xy),
                    "y": list(cfg.randomization.reset.table_xy),
                    "z": [0.0, 0.0],
                },
                "velocity_range": {"x": [-0.0, 0.0], "y": [-0.0, 0.0], "z": [-0.0, 0.0]},
                "asset_cfg": SceneEntityCfg("table"),
            },
        )

        reset_object = EventTerm(
            func=mdp.reset_object_on_table_stable,
            mode="reset",
            params={
                "xy_position_range": {
                    "x": list(cfg.randomization.reset.object_x),
                    "y": list(cfg.randomization.reset.object_y),
                    "yaw": list(cfg.randomization.reset.object_yaw),
                },
                "table_surface_z": cfg.workspace.table_surface_z,
                "randomize_stable_pose": False,
                "asset_cfg": SceneEntityCfg("object"),
                "z_offset": cfg.randomization.reset.z_offset,
            },
        )

        reset_root = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": [-0.0, 0.0], "y": [-0.0, 0.0], "yaw": [-0.0, 0.0]},
                "velocity_range": {"x": [-0.0, 0.0], "y": [-0.0, 0.0], "z": [-0.0, 0.0]},
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": list(cfg.randomization.reset.robot_joints),
                "velocity_range": [0.0, 0.0],
            },
        )

        reset_robot_wrist_joint = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names="iiwa7_joint_7"),
                "position_range": list(cfg.randomization.reset.robot_wrist),
                "velocity_range": [0.0, 0.0],
            },
        )

        variable_gravity = EventTerm(
            func=mdp.randomize_physics_scene_gravity,
            mode="reset",
            params={
                "gravity_distribution_params": (
                    tuple(cfg.curriculum.gravity.initial),
                    tuple(cfg.curriculum.gravity.initial),
                ),
                "operation": "abs",
            },
        )

    return EventCfg()


def _build_rewards_cfg(cfg: Y2RConfig):
    """Build RewardsCfg from config."""
    
    @configclass
    class RewardsCfg:
        """Reward terms for the MDP."""

        # MUST BE FIRST - computes hand_pose_gate used by other rewards
        # (lookahead_tracking, trajectory_success, finger_manipulation, tracking_progress)
        hand_pose_following = RewTerm(
            func=mdp.hand_pose_following,
            weight=cfg.rewards.hand_pose_following.weight,
            params={
                "grasp_pos_tol": cfg.rewards.hand_pose_following.params["grasp_pos_tol"],
                "grasp_rot_tol": cfg.rewards.hand_pose_following.params["grasp_rot_tol"],
                "manipulation_pos_tol": cfg.rewards.hand_pose_following.params["manipulation_pos_tol"],
                "manipulation_rot_tol": cfg.rewards.hand_pose_following.params["manipulation_rot_tol"],
                "release_pos_tol": cfg.rewards.hand_pose_following.params["release_pos_tol"],
                "release_rot_tol": cfg.rewards.hand_pose_following.params["release_rot_tol"],
                "gate_in_grasp": cfg.rewards.hand_pose_following.params["gate_in_grasp"],
                "gate_in_manipulation": cfg.rewards.hand_pose_following.params["gate_in_manipulation"],
                "gate_in_release": cfg.rewards.hand_pose_following.params["gate_in_release"],
                "gate_pos_threshold": cfg.rewards.hand_pose_following.params["gate_pos_threshold"],
                "gate_rot_threshold": cfg.rewards.hand_pose_following.params["gate_rot_threshold"],
                "manipulation_gate_pos_threshold": cfg.rewards.hand_pose_following.params.get("manipulation_gate_pos_threshold"),
                "manipulation_gate_rot_threshold": cfg.rewards.hand_pose_following.params.get("manipulation_gate_rot_threshold"),
                "gate_floor": cfg.rewards.hand_pose_following.params["gate_floor"],
                "robot_cfg": SceneEntityCfg("robot"),
            },
        )

        action_l2 = RewTerm(
            func=mdp.action_l2_clamped,
            weight=cfg.rewards.action_l2.weight,
            params={
                "arm_joint_count": cfg.robot.arm_joint_count,
                "finger_scale": cfg.rewards.action_l2.params["finger_scale"],
            },
        )

        action_rate_l2 = RewTerm(
            func=mdp.action_rate_l2_clamped,
            weight=cfg.rewards.action_rate_l2.weight,
            params={
                "arm_joint_count": cfg.robot.arm_joint_count,
                "finger_scale": cfg.rewards.action_rate_l2.params["finger_scale"],
            },
        )

        fingers_to_object = RewTerm(
            func=mdp.object_ee_distance,
            weight=cfg.rewards.fingers_to_object.weight,
            params={
                "phases": cfg.rewards.fingers_to_object.params.get("phases"),
                "use_hand_pose_gate": cfg.rewards.fingers_to_object.params.get("use_hand_pose_gate"),
                "std": cfg.rewards.fingers_to_object.params["std"],
                "error_gate_pos_threshold": cfg.rewards.fingers_to_object.params["error_gate_pos_threshold"],
                "error_gate_pos_slope": cfg.rewards.fingers_to_object.params["error_gate_pos_slope"],
                "error_gate_rot_threshold": cfg.rewards.fingers_to_object.params["error_gate_rot_threshold"],
                "error_gate_rot_slope": cfg.rewards.fingers_to_object.params["error_gate_rot_slope"],
            },
        )

        lookahead_tracking = RewTerm(
            func=mdp.lookahead_tracking,
            weight=cfg.rewards.lookahead_tracking.weight,
            params={
                "phases": cfg.rewards.lookahead_tracking.params.get("phases"),
                "use_hand_pose_gate": cfg.rewards.lookahead_tracking.params.get("use_hand_pose_gate"),
                "use_contact_gating": cfg.rewards.lookahead_tracking.params.get("use_contact_gating"),
                "std": cfg.rewards.lookahead_tracking.params["std"],
                "decay": cfg.rewards.lookahead_tracking.params["decay"],
                "contact_threshold": cfg.rewards.lookahead_tracking.params["contact_threshold"],
                "contact_ramp": cfg.rewards.lookahead_tracking.params["contact_ramp"],
                "contact_min_factor": cfg.rewards.lookahead_tracking.params["contact_min_factor"],
                "rot_std": cfg.rewards.lookahead_tracking.params["rot_std"],
                "neg_threshold": cfg.rewards.lookahead_tracking.params["neg_threshold"],
                "neg_std": cfg.rewards.lookahead_tracking.params["neg_std"],
                "neg_scale": cfg.rewards.lookahead_tracking.params["neg_scale"],
                "rot_neg_threshold": cfg.rewards.lookahead_tracking.params["rot_neg_threshold"],
                "rot_neg_std": cfg.rewards.lookahead_tracking.params["rot_neg_std"],
            },
        )

        trajectory_success = RewTerm(
            func=mdp.trajectory_success,
            weight=cfg.rewards.trajectory_success.weight,
            params={
                "phases": cfg.rewards.trajectory_success.params.get("phases"),
                "use_hand_pose_gate": cfg.rewards.trajectory_success.params.get("use_hand_pose_gate"),
                "use_finger_release_gate": cfg.rewards.trajectory_success.params.get("use_finger_release_gate"),
                "pos_threshold": cfg.rewards.trajectory_success.params["pos_threshold"][0],
                "rot_threshold": cfg.rewards.trajectory_success.params["rot_threshold"][0],
                "pos_std": cfg.rewards.trajectory_success.params["pos_std"],
                "rot_std": cfg.rewards.trajectory_success.params["rot_std"],
                "sparse_weight": cfg.rewards.trajectory_success.params["sparse_weight"],
                "contact_gate_threshold": cfg.rewards.trajectory_success.params["contact_gate_threshold"],
                "contact_gate_ramp": cfg.rewards.trajectory_success.params["contact_gate_ramp"],
                "contact_gate_floor": cfg.rewards.trajectory_success.params["contact_gate_floor"],
            },
        )

        early_termination = RewTerm(
            func=mdp.is_terminated_term,
            weight=cfg.rewards.early_termination.weight,
            params={"term_keys": cfg.rewards.early_termination.params["term_keys"]},
        )

        arm_table_penalty = RewTerm(
            func=mdp.arm_table_binary_penalty,
            weight=cfg.rewards.arm_table_penalty.weight,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["iiwa7_link_(3|4|5|6|7)|palm_link"]),
                "table_z": cfg.rewards.arm_table_penalty.params["table_z"],
                "threshold_mid": cfg.rewards.arm_table_penalty.params["threshold_mid"],
                "threshold_distal": cfg.rewards.arm_table_penalty.params["threshold_distal"],
            },
        )

        finger_manipulation = RewTerm(
            func=mdp.finger_manipulation,
            weight=cfg.rewards.finger_manipulation.weight,
            params={
                "phases": cfg.rewards.finger_manipulation.params.get("phases"),
                "use_hand_pose_gate": cfg.rewards.finger_manipulation.params.get("use_hand_pose_gate"),
                "pos_std": cfg.rewards.finger_manipulation.params["pos_std"],
                "rot_std": cfg.rewards.finger_manipulation.params["rot_std"],
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("object"),
            },
        )

        palm_velocity_penalty = RewTerm(
            func=mdp.palm_velocity_penalty,
            weight=cfg.rewards.palm_velocity_penalty.weight,
            params={
                "angular_std": cfg.rewards.palm_velocity_penalty.params["angular_std"],
                "linear_std": cfg.rewards.palm_velocity_penalty.params["linear_std"],
                "linear_scale": cfg.rewards.palm_velocity_penalty.params["linear_scale"],
                "robot_cfg": SceneEntityCfg("robot"),
            },
        )

        palm_orientation_penalty = RewTerm(
            func=mdp.palm_orientation_penalty,
            weight=cfg.rewards.palm_orientation_penalty.weight,
            params={
                "std": cfg.rewards.palm_orientation_penalty.params["std"],
                "robot_cfg": SceneEntityCfg("robot"),
            },
        )

        joint_limits_margin = RewTerm(
            func=mdp.joint_pos_limits_margin,
            weight=cfg.rewards.joint_limits_margin.weight,
            params={
                "threshold": cfg.rewards.joint_limits_margin.params["threshold"],
                "power": cfg.rewards.joint_limits_margin.params["power"],
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            },
        )

        tracking_progress = RewTerm(
            func=mdp.tracking_progress,
            weight=cfg.rewards.tracking_progress.weight,
            params={
                "phases": cfg.rewards.tracking_progress.params.get("phases"),
                "use_hand_pose_gate": cfg.rewards.tracking_progress.params.get("use_hand_pose_gate"),
                "pos_weight": cfg.rewards.tracking_progress.params["pos_weight"],
                "rot_weight": cfg.rewards.tracking_progress.params["rot_weight"],
                "positive_only": cfg.rewards.tracking_progress.params["positive_only"],
                "clip": cfg.rewards.tracking_progress.params["clip"],
            },
        )

        finger_release = RewTerm(
            func=mdp.finger_release,
            weight=cfg.rewards.finger_release.weight,
            params={
                "phases": cfg.rewards.finger_release.params.get("phases"),
                "scale": cfg.rewards.finger_release.params["scale"],
                "arm_joint_count": cfg.robot.arm_joint_count,
                "hand_joint_count": cfg.robot.hand_joint_count,
                "robot_cfg": SceneEntityCfg("robot"),
            },
        )

        finger_regularizer = RewTerm(
            func=mdp.finger_regularizer,
            weight=cfg.rewards.finger_regularizer.weight,
            params={
                "phases": cfg.rewards.finger_regularizer.params.get("phases"),
                "default_joints": cfg.rewards.finger_regularizer.params["default_joints"],
                "std": cfg.rewards.finger_regularizer.params["std"],
                "arm_joint_count": cfg.robot.arm_joint_count,
                "robot_cfg": SceneEntityCfg("robot"),
            },
        )

        object_stillness = RewTerm(
            func=mdp.object_stillness,
            weight=cfg.rewards.object_stillness.weight,
            params={
                "phases": cfg.rewards.object_stillness.params.get("phases"),
                "lin_std": cfg.rewards.object_stillness.params["lin_std"],
                "ang_std": cfg.rewards.object_stillness.params["ang_std"],
                "object_cfg": SceneEntityCfg("object"),
            },
        )

    return RewardsCfg()


def _build_terminations_cfg(cfg: Y2RConfig):
    """Build TerminationsCfg from config."""
    
    @configclass
    class TerminationsCfg:
        """Termination terms for the MDP."""

        time_out = DoneTerm(func=mdp.time_out, time_out=True)

        trajectory_deviation = DoneTerm(
            func=mdp.trajectory_deviation,
            params={
                "threshold": cfg.terminations.trajectory_deviation.point_cloud_threshold[0],
                "rot_threshold": cfg.terminations.trajectory_deviation.rotation_threshold[0],
            },
        )

        hand_pose_deviation = DoneTerm(
            func=mdp.hand_pose_deviation,
            params={
                "pos_threshold": cfg.terminations.hand_pose_deviation.position_threshold[0],
                "rot_threshold": cfg.terminations.hand_pose_deviation.rotation_threshold[0],
            },
        )

    return TerminationsCfg()


# ==============================================================================
# MAIN ENV CONFIG
# ==============================================================================

import os as _os
_Y2R_MODE = _os.environ.get("Y2R_MODE", "train")
_Y2R_TASK = _os.environ.get("Y2R_TASK", "base")

@configclass
class TrajectoryEnvCfg(ManagerBasedEnvCfg):
    """Trajectory following task definition.

    The robot must follow a sequence of point cloud targets along a smooth Bezier trajectory.
    Includes pickup from table, variable-speed manipulation, and place-back phases.
    
    All config values are loaded dynamically in __post_init__ from the YAML file.
    Config is determined by Y2R_MODE and Y2R_TASK environment variables.
    
    Y2R_MODE: train | distill | play | play_student
    Y2R_TASK: base | push | cup
    
    Example: Y2R_MODE=play Y2R_TASK=cup ./scripts/play.sh --continue
    """

    # Config parameters - read from env vars at module load time
    _config_mode: str = _Y2R_MODE
    _config_task: str = _Y2R_TASK
    
    # Static settings
    viewer: ViewerCfg = ViewerCfg(eye=(-2.25, 0.0, 0.75), lookat=(0.0, 0.0, 0.45), origin_type="env")
    
    # Scene - configured in __post_init__
    scene: TrajectorySceneCfg = TrajectorySceneCfg(num_envs=1, env_spacing=3.0)
    
    # Built dynamically in __post_init__
    observations = None
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards = None
    terminations = None
    events = None
    curriculum = None
    
    # Config reference (set in __post_init__)
    y2r_cfg: Y2RConfig = None

    def __post_init__(self):
        """Build all config-dependent managers."""
        cfg = get_config(mode=self._config_mode, task=self._config_task)
        self.y2r_cfg = cfg
        
        # Build scene
        self.scene = TrajectorySceneCfg(
            num_envs=cfg.simulation.num_envs,
            env_spacing=cfg.simulation.env_spacing,
            replicate_physics=cfg.simulation.replicate_physics,
        )
        
        # Build object config (primitives + procedural shapes OR push_t USD)
        push_t_mode = cfg.push_t.enabled and cfg.push_t.object_usd
        if push_t_mode:
            self._setup_push_t_scene(cfg)
        else:
            self.scene.object = _build_object_cfg(cfg)
        
        # Build managers
        self.observations = _build_observations_cfg(cfg)
        self.events = _build_events_cfg(cfg)
        self.rewards = _build_rewards_cfg(cfg)
        self.terminations = _build_terminations_cfg(cfg)
        self.curriculum = build_curriculum_cfg(cfg)
        
        # Override object reset for push_t mode (after events is built)
        if push_t_mode:
            self._setup_push_t_events(cfg)
        
        # Set curriculum initial difficulty
        if self.curriculum is not None:
            self.curriculum.adr.params["init_difficulty"] = cfg.curriculum.difficulty.initial
        
        # Simulation parameters
        self.decimation = cfg.simulation.decimation
        
        # Compute episode duration (same for all modes - push_t uses phases too)
        phases = cfg.trajectory.phases
        max_waypoints = cfg.waypoints.count[1]  # For push_t, this is [0,0] so 0
        settle_duration = getattr(phases, 'settle', 0.0)
        # Per-waypoint time = movement + pause
        waypoint_duration = cfg.waypoints.movement_duration + cfg.waypoints.pause_duration
        self.episode_length_s = (
            phases.grasp
            + phases.manipulate_base
            + waypoint_duration * max_waypoints  # Movement + pause per waypoint
            + settle_duration  # Settle phase before retreat
            + phases.hand_release
        )
        
        self.is_finite_horizon = True
        self.sim.dt = cfg.simulation.physics_dt
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_max_rigid_patch_count = 4 * 5 * 2**15

    def _setup_push_t_scene(self, cfg: Y2RConfig):
        """Setup push_t scene: USD object and outline.
        
        Z-offsets (center-to-bottom distance) can be set manually in config or auto-computed:
        - If object_z_offset is None: auto-compute from mesh geometry
        - If object_z_offset is set: use that value directly
        Same for outline_z_offset.
        """
        config_dir = Path(__file__).parent
        safety_margin = cfg.randomization.reset.z_offset  # 5mm default
        
        # Compute object z-offset: use config value or auto-compute from mesh
        usd_path = str(config_dir / cfg.push_t.object_usd)
        # Use max of scale_range for init_state z to ensure no penetration at largest scale
        # Smaller scales will spawn slightly higher, but physics will settle them
        scale_range = cfg.randomization.object.scale
        max_scale = max(scale_range)
        
        if cfg.push_t.object_z_offset is not None:
            object_z_offset = cfg.push_t.object_z_offset
        else:
            object_z_offset = compute_z_offset_from_usd(
                usd_path=usd_path,
                rotation_wxyz=tuple(cfg.push_t.object_rotation),
                scale=max_scale,
            )
        object_z = cfg.workspace.table_surface_z + object_z_offset + safety_margin
        
        # Object from USD
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            spawn=sim_utils.UsdFileCfg(
                usd_path=usd_path,
                scale=(cfg.push_t.object_scale,) * 3,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=0,
                    disable_gravity=False,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
                activate_contact_sensors=True,  # Required for object_table_contact sensor
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(-0.55, 0.0, object_z),
                rot=cfg.push_t.object_rotation,
            ),
        )
        
        # Visual outline at goal position
        if cfg.push_t.outline_usd:
            outline_path = str(config_dir / cfg.push_t.outline_usd)
            
            # Compute outline z-offset: use config value or auto-compute from mesh
            if cfg.push_t.outline_z_offset is not None:
                outline_z_offset = cfg.push_t.outline_z_offset
            else:
                outline_z_offset = compute_z_offset_from_usd(
                    usd_path=outline_path,
                    rotation_wxyz=tuple(cfg.push_t.outline_rotation),
                    scale=cfg.push_t.object_scale,
                )
            outline_z = cfg.workspace.table_surface_z + outline_z_offset + 0.001  # Small offset to avoid z-fighting
            
            self.scene.outline = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Outline",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=outline_path,
                    scale=(cfg.push_t.object_scale,) * 3,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(*cfg.push_t.outline_position, outline_z),
                    rot=cfg.push_t.outline_rotation,
                ),
            )

    def _setup_push_t_events(self, cfg: Y2RConfig):
        """Override object reset for push_t mode.

        Uses reset_push_t_object which applies yaw in WORLD frame (around world Z axis).
        This is necessary because the T-shape has a non-identity default rotation to lay
        flat on the table, and we want yaw to rotate it on the table surface.
        """
        self.events.reset_object = EventTerm(
            func=mdp.reset_push_t_object,
            mode="reset",
            params={
                "pose_range": {
                    "x": list(cfg.randomization.reset.object_x),
                    "y": list(cfg.randomization.reset.object_y),
                    "yaw": list(cfg.randomization.reset.object_yaw),
                },
                "asset_cfg": SceneEntityCfg("object"),
            },
        )


