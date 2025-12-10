# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trajectory following task for dexterous manipulation.

This task trains the robot to follow a sequence of targets along a smooth trajectory.
The trajectory includes pickup from table, variable-speed manipulation, and place-back phases.

Supports two representation modes (controlled by trajectory_params.use_point_cloud):
- Point cloud mode (default): Uses point-to-point distance for tracking
- Pose mode: Uses position + rotation (quaternion) for tracking
"""

from dataclasses import MISSING

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
from isaaclab.sim import CapsuleCfg, ConeCfg, CuboidCfg, RigidBodyMaterialCfg, SphereCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp
from .adr_curriculum import TrajectoryCurriculumCfg
from .trajectory_cfg import TrajectoryParamsCfg, TRAJECTORY_PARAMS


@configclass
class TrajectorySceneCfg(InteractiveSceneCfg):
    """Scene for trajectory following task with object starting on table."""

    # robot
    robot: ArticulationCfg = MISSING

    # object - starts on table surface (table top at z=0.255, plus object half-height ~0.05)
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                CuboidCfg(size=(0.05, 0.1, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                CuboidCfg(size=(0.05, 0.05, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                CuboidCfg(size=(0.025, 0.1, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                CuboidCfg(size=(0.025, 0.05, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                CuboidCfg(size=(0.025, 0.025, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                CuboidCfg(size=(0.01, 0.1, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                SphereCfg(radius=0.05, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                SphereCfg(radius=0.025, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                CapsuleCfg(radius=0.04, height=0.025, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                CapsuleCfg(radius=0.04, height=0.01, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                CapsuleCfg(radius=0.04, height=0.1, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                CapsuleCfg(radius=0.025, height=0.1, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                CapsuleCfg(radius=0.025, height=0.2, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                CapsuleCfg(radius=0.01, height=0.2, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                ConeCfg(radius=0.05, height=0.1, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                ConeCfg(radius=0.025, height=0.1, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
            ],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
        ),
        # Object starts above table, will settle via physics
        # TODO: Compute z dynamically based on object type and scale
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


@configclass
class CommandsCfg:
    """Command terms for the MDP (empty for trajectory task - uses TrajectoryManager instead)."""
    pass


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5

    @configclass
    class ProprioObsCfg(ObsGroup):
        """Observations for proprioception group."""

        joint_pos = ObsTerm(func=mdp.joint_pos, noise=Unoise(n_min=-0.0, n_max=0.0))
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-0.0, n_max=0.0))
        hand_tips_state_b = ObsTerm(
            func=mdp.body_state_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            # good behaving number for position in m, velocity in m/s, rad/s,
            # and quaternion are unlikely to exceed -2 to 2 range
            clip=(-2.0, 2.0),
            params={
                "body_asset_cfg": SceneEntityCfg("robot"),
                "base_asset_cfg": SceneEntityCfg("robot"),
            },
        )
        contact: ObsTerm = MISSING

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5

    @configclass
    class PerceptionObsCfg(ObsGroup):
        """Perception observations: current object point cloud and pose with history."""

        # Current object point cloud with history
        # Shape per frame: num_points × 3, with history_length frames
        object_point_cloud = ObsTerm(
            func=mdp.object_point_cloud_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            clip=(-2.0, 2.0),
            params={
                "num_points": TRAJECTORY_PARAMS.num_points,
                "flatten": True,
                "visualize": True,
            },
        )
        
        # Current object pose (position + quaternion) with history
        # Shape per frame: 7 (pos_x, pos_y, pos_z, qw, qx, qy, qz)
        object_pose = ObsTerm(
            func=mdp.object_pose_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            clip=(-2.0, 2.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_dim = 0
            self.concatenate_terms = True
            self.flatten_history_dim = True
            self.history_length = TRAJECTORY_PARAMS.object_pc_history_length
    
    @configclass
    class TargetObsCfg(ObsGroup):
        """Target observations: point clouds AND poses from trajectory."""
        
        # Target point clouds (window_size × num_points × 3)
        # Also manages trajectory generation and caches both error types
        target_point_clouds = ObsTerm(
            func=mdp.target_sequence_obs_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            clip=(-2.0, 2.0),
            params={
                "trajectory_cfg": TRAJECTORY_PARAMS,
            },
        )
        
        # Target poses (window_size × 7)
        # Must come after target_point_clouds which initializes trajectory_manager
        target_poses = ObsTerm(
            func=mdp.target_sequence_poses_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            clip=(-2.0, 2.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_dim = 0
            self.concatenate_terms = True
            self.flatten_history_dim = True
            self.history_length = 1  # No history for targets (they already encode future)

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    proprio: ProprioObsCfg = ProprioObsCfg()
    perception: PerceptionObsCfg = PerceptionObsCfg()
    targets: TargetObsCfg = TargetObsCfg()


@configclass
class EventCfg:
    """Configuration for randomization."""

    # -- pre-startup
    randomize_object_scale = EventTerm(
        func=mdp.randomize_rigid_body_scale,
        mode="prestartup",
        params={"scale_range": (0.75, 1.5), "asset_cfg": SceneEntityCfg("object")},
    )

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": [0.5, 1.0],
            "dynamic_friction_range": [0.5, 1.0],
            "restitution_range": [0.0, 0.0],
            "num_buckets": 250,
        },
    )

    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
            "static_friction_range": [0.5, 1.0],
            "dynamic_friction_range": [0.5, 1.0],
            "restitution_range": [0.0, 0.0],
            "num_buckets": 250,
        },
    )

    joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": [0.5, 2.0],
            "damping_distribution_params": [0.5, 2.0],
            "operation": "scale",
        },
    )

    joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": [0.0, 5.0],
            "operation": "scale",
        },
    )

    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": [0.2, 2.0],
            "operation": "scale",
        },
    )

    reset_table = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": [-0.05, 0.05], "y": [-0.05, 0.05], "z": [0.0, 0.0]},
            "velocity_range": {"x": [-0.0, 0.0], "y": [-0.0, 0.0], "z": [-0.0, 0.0]},
            "asset_cfg": SceneEntityCfg("table"),
        },
    )

    # For trajectory task: object resets on table in a stable pose
    # Uses trimesh.poses.compute_stable_poses() to find physically stable orientations
    # Then randomizes x, y position on the table
    reset_object = EventTerm(
        func=mdp.reset_object_on_table_stable,
        mode="reset",
        params={
            "xy_position_range": {
                "x": [-0.25, 0.25],  # Relative to object init pos (-0.55)
                "y": [-0.35, 0.35],  # Relative to object init pos (0.0)
                "yaw": [-3.14, 3.14],  # Random yaw rotation on top of stable pose
            },
            "table_surface_z": 0.255,  # Table surface height
            "randomize_stable_pose": False,  # Use most stable pose only
            "asset_cfg": SceneEntityCfg("object"),
            "z_offset": 0.0005,  # safety margin
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
            "position_range": [-0.50, 0.50],
            "velocity_range": [0.0, 0.0],
        },
    )

    reset_robot_wrist_joint = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="iiwa7_joint_7"),
            "position_range": [-3, 3],
            "velocity_range": [0.0, 0.0],
        },
    )

    # Note (Octi): This is a deliberate trick in Remake to accelerate learning.
    # By scheduling gravity as a curriculum — starting with no gravity (easy)
    # and gradually introducing full gravity (hard) — the agent learns more smoothly.
    # This removes the need for a special "Lift" reward (often required to push the
    # agent to counter gravity), which has bonus effect of simplifying reward composition overall.
    variable_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="reset",
        params={
            "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            "operation": "abs",
        },
    )


@configclass
class ActionsCfg:
    pass


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    action_l2 = RewTerm(func=mdp.action_l2_clamped, weight=-0.005)

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2_clamped, weight=-0.005)

    fingers_to_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.4}, weight=1.0)

    # Trajectory tracking: lookahead with exponential decay, gated by contact
    # Works for both point cloud and pose modes (auto-detects based on cached errors)
    lookahead_tracking = RewTerm(
        func=mdp.lookahead_tracking,
        weight=5.0,
        params={
            "std": 0.03,  # Position std (point cloud mode: mean error std)
            "decay": TRAJECTORY_PARAMS.lookahead_decay,
            "contact_threshold": 2.0,  # Must have contact to get tracking reward
            "rot_std": TRAJECTORY_PARAMS.pose_rot_std,  # Rotation std (pose mode only)
        },
    )

    # Success: at goal + in release phase
    # Works for both point cloud and pose modes (auto-detects based on cached errors)
    # Success: at goal + in release phase
    # NOTE: Default to INITIAL (lenient) values - curriculum tightens these as difficulty increases
    trajectory_success = RewTerm(
        func=mdp.trajectory_success,
        weight=10.0,
        params={
            "error_threshold": TRAJECTORY_PARAMS.success_threshold_initial,  # Start lenient (6cm)
            "rot_threshold": TRAJECTORY_PARAMS.pose_success_rot_threshold_initial,  # Start lenient (~46 deg)
        },
    )

    early_termination = RewTerm(
        func=mdp.is_terminated_term,
        weight=-5.0,
        params={"term_keys": ["abnormal_robot", "trajectory_deviation"]},
    )

    # Arm table avoidance: penalize links 3-7 + palm below safe heights
    arm_table_penalty = RewTerm(
        func=mdp.arm_table_binary_penalty,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["iiwa7_link_(3|4|5|6|7)|palm_link"]),
            "table_z": 0.255,
            "threshold_mid": 0.06,
            "threshold_distal": 0.03,
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    abnormal_robot = DoneTerm(func=mdp.abnormal_robot_state)

    # Terminate if object deviates too far from current target
    # Works for both point cloud and pose modes (auto-detects based on cached errors)
    # NOTE: Default to INITIAL (lenient) values - curriculum tightens these as difficulty increases
    trajectory_deviation = DoneTerm(
        func=mdp.trajectory_deviation,
        params={
            "threshold": TRAJECTORY_PARAMS.termination_threshold_initial,  # Start lenient (30cm)
            "rot_threshold": TRAJECTORY_PARAMS.pose_termination_rot_threshold_initial,  # Start lenient (~86 deg)
        },
    )


@configclass
class TrajectoryEnvCfg(ManagerBasedEnvCfg):
    """Trajectory following task definition.

    The robot must follow a sequence of point cloud targets along a smooth Bezier trajectory.
    Includes pickup from table, variable-speed manipulation, and place-back phases.
    """

    # Trajectory parameters (configurable)
    trajectory_params: TrajectoryParamsCfg = TrajectoryParamsCfg()

    # Scene settings
    viewer: ViewerCfg = ViewerCfg(eye=(-2.25, 0.0, 0.75), lookat=(0.0, 0.0, 0.45), origin_type="env")
    scene: TrajectorySceneCfg = TrajectorySceneCfg(num_envs=4096, env_spacing=3, replicate_physics=False)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()  # TODO: Replace with trajectory commands
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()  # TODO: Replace with trajectory rewards
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: TrajectoryCurriculumCfg | None = TrajectoryCurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # General settings
        self.decimation = 2  # 60 Hz control (120 Hz physics / 2)

        # Episode length matches trajectory duration
        self.episode_length_s = self.trajectory_params.trajectory_duration
        self.is_finite_horizon = True

        # Simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_max_rigid_patch_count = 4 * 5 * 2**15


class TrajectoryEnvCfg_PLAY(TrajectoryEnvCfg):
    """Trajectory task evaluation environment definition."""

    def __post_init__(self):
        super().__post_init__()
        # Start at max difficulty for evaluation
        if self.curriculum is not None:
            self.curriculum.adr.params["init_difficulty"] = self.curriculum.adr.params["min_difficulty"]
        # Enable reward debug printing for play mode
        TRAJECTORY_PARAMS.debug_print_rewards = True
