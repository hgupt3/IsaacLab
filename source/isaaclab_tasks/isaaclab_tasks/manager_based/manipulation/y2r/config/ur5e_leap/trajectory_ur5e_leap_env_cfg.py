"""UR5e + LEAP Hand configuration for trajectory following task."""

from isaaclab_assets.robots import UR5E_LEAP_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg, ContactSensorCfg
from isaaclab.sim import PinholeCameraCfg
from isaaclab.utils import configclass

from ... import trajectory_env_cfg as trajectory
from ...config_loader import Y2RConfig
from ... import mdp


# ==============================================================================
# JOINT ORDER
# ==============================================================================

UR5E_LEAP_JOINT_ORDER = [
    # arm (6)
    "ur5e_joint_1", "ur5e_joint_2", "ur5e_joint_3",
    "ur5e_joint_4", "ur5e_joint_5", "ur5e_joint_6",
    # hand (16)
    "index_joint_0", "index_joint_1", "index_joint_2", "index_joint_3",
    "middle_joint_0", "middle_joint_1", "middle_joint_2", "middle_joint_3",
    "ring_joint_0",   "ring_joint_1",   "ring_joint_2",   "ring_joint_3",
    "thumb_joint_0",  "thumb_joint_1",  "thumb_joint_2",  "thumb_joint_3",
]


# ==============================================================================
# ACTION CONFIG BUILDERS
# ==============================================================================

def _build_rel_joint_action_cfg(cfg: Y2RConfig):
    """Build standard relative joint position action config (22D)."""
    @configclass
    class UR5eLeapRelJointPosActionCfg:
        action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=UR5E_LEAP_JOINT_ORDER,
            scale=cfg.robot.action_scale,
        )
    return UR5eLeapRelJointPosActionCfg()


def _build_eigen_grasp_action_cfg(cfg: Y2RConfig):
    """Build eigen grasp action config (27D input -> 22D output)."""
    @configclass
    class UR5eLeapEigenGraspActionCfg:
        action = mdp.EigenGraspRelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=UR5E_LEAP_JOINT_ORDER,
            scale=cfg.robot.action_scale,
            arm_joint_count=6,
            hand_joint_count=16,
            eigen_dim=cfg.robot.eigen_dim,
        )
    return UR5eLeapEigenGraspActionCfg()


def _build_ur5e_leap_rewards_cfg(cfg: Y2RConfig, base_rewards):
    """Add UR5e LEAP specific rewards to base rewards."""
    base_rewards.good_finger_contact = RewTerm(
        func=mdp.good_finger_contact,
        weight=cfg.rewards.good_finger_contact.weight,
        params={
            "phases": cfg.rewards.good_finger_contact.params.get("phases"),
            "use_hand_pose_gate": cfg.rewards.good_finger_contact.params.get("use_hand_pose_gate"),
            "invert_in_release": cfg.rewards.good_finger_contact.params.get("invert_in_release"),
        },
    )
    base_rewards.distal_joint3_penalty = RewTerm(
        func=mdp.distal_joint3_penalty,
        weight=cfg.rewards.distal_joint3_penalty.weight,
        params={
            "phases": cfg.rewards.distal_joint3_penalty.params.get("phases"),
            "std": cfg.rewards.distal_joint3_penalty.params["std"],
            "joint_name_regex": cfg.rewards.distal_joint3_penalty.params["joint_name_regex"],
            "only_when_contact": cfg.rewards.distal_joint3_penalty.params["only_when_contact"],
        },
    )
    return base_rewards


# ==============================================================================
# UR5E LEAP MIXIN
# ==============================================================================

# The URDF is converted to USD with merge_fixed_joints=True.
# Fixed-joint links (camera chain, palm_frame, tip frames, palm_link) are merged into parents.
# Surviving bodies (connected via revolute joints) remain as direct children of /World/Robot/.
# hand_base_joint is fixed — palm_link is merged into ur5e_link_6.

@configclass
class UR5eLeapTrajectoryMixinCfg:
    """Mixin config for UR5e + LEAP Hand robot in trajectory task."""

    actions = None

    def __post_init__(self: trajectory.TrajectoryEnvCfg):
        super().__post_init__()

        cfg = self.y2r_cfg

        # Set up robot — UR5e is mounted ON the table
        table_surface_z = cfg.workspace.table_surface_z
        self.scene.robot = UR5E_LEAP_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = (0.0, 0.0, table_surface_z)

        # Override table: real-world dimensions (1.125m depth × 1.267m width)
        # Robot at X=0, centered in Y, 11.3cm from back edge
        # Table center X = 0.113 - 1.13/2 = -0.452 ≈ -0.45
        self.scene.table = RigidObjectCfg(
            prim_path="/World/envs/env_.*/table",
            spawn=sim_utils.CuboidCfg(
                size=(1.13, 1.27, 0.04),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.91, 0.90), roughness=0.4),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(-0.45, 0.0, table_surface_z - 0.02),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )

        # Select action space
        if cfg.mode.use_eigen_grasp:
            self.actions = _build_eigen_grasp_action_cfg(cfg)
        else:
            self.actions = _build_rel_joint_action_cfg(cfg)

        # Add robot-specific rewards
        _build_ur5e_leap_rewards_cfg(cfg, self.rewards)

        # Override base config terms that hardcode Kuka joint/body names
        # arm_table_penalty: iiwa7_link_(3|4|5|6|7) -> ur5e_link_(3|4|5|6)
        self.rewards.arm_table_penalty.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=["ur5e_link_(3|4|5|6)"]
        )

        # Setup contact sensors for fingertips (link_3) and mid-phalanx (link_2)
        # USD hierarchy is flat: all bodies are at {ENV_REGEX_NS}/Robot/{link_name}
        finger_tip_body_list = ["index_link_3", "middle_link_3", "ring_link_3", "thumb_link_3"]
        finger_mid_body_list = ["index_link_2", "middle_link_2", "ring_link_2", "thumb_link_2"]
        for link_name in finger_tip_body_list + finger_mid_body_list:
            setattr(
                self.scene,
                f"{link_name}_object_s",
                ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/" + link_name,
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
                ),
            )

        # Contact force observation
        self.observations.proprio.contact = ObsTerm(
            func=mdp.fingers_contact_force_b,
            params={"contact_sensor_names": [f"{link}_object_s" for link in finger_tip_body_list]},
            clip=(-20.0, 20.0),
        )

        # Link_2 (mid-phalanx) contact force observation (teacher-only privileged)
        self.observations.proprio.link_2_contact = ObsTerm(
            func=mdp.fingers_contact_force_b,
            params={"contact_sensor_names": ["index_link_2_object_s", "middle_link_2_object_s", "ring_link_2_object_s"]},
            clip=(-20.0, 20.0),
        )

        # Object pose in palm frame (teacher-only privileged)
        self.observations.proprio.object_pose_palm = ObsTerm(func=mdp.object_pose_palm_b)

        # Hand tips state observation (link_3 bodies with computed tip offsets)
        self.observations.proprio.hand_tips_state_b = ObsTerm(
            func=mdp.hand_tips_state_with_offsets_b,
            noise=self.observations.proprio.hand_tips_state_b.noise,
            params={
                "body_asset_cfg": SceneEntityCfg("robot", body_names=["ur5e_link_6", "(index|middle|ring|thumb)_link_3"]),
                "base_asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # Fingers to object reward (link_3 bodies with tip offsets applied internally)
        self.rewards.fingers_to_object.params["asset_cfg"] = SceneEntityCfg("robot", body_names=["ur5e_link_6", "(index|middle|ring|thumb)_link_3"])

        # Add wrist camera in student mode
        if cfg.mode.use_student_mode:
            from scipy.spatial.transform import Rotation

            rot_euler = cfg.wrist_camera.offset.rot
            rot_swapped = (rot_euler[0], rot_euler[2], -rot_euler[1])

            r = Rotation.from_euler('xyz', rot_swapped, degrees=True)
            quat = r.as_quat()  # [x, y, z, w]
            quat_wxyz = (float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2]))
            pos = tuple(float(x) for x in cfg.wrist_camera.offset.pos)

            self.scene.wrist_camera = TiledCameraCfg(
                prim_path="{ENV_REGEX_NS}/Robot/ur5e_link_6/wrist_camera",
                offset=TiledCameraCfg.OffsetCfg(
                    pos=pos,
                    rot=quat_wxyz,
                    convention="opengl"
                ),
                data_types=["distance_to_image_plane"],
                spawn=PinholeCameraCfg(
                    focal_length=cfg.wrist_camera.focal_length,
                    horizontal_aperture=cfg.wrist_camera.horizontal_aperture,
                    clipping_range=cfg.wrist_camera.clipping_range,
                ),
                width=cfg.wrist_camera.width,
                height=cfg.wrist_camera.height,
            )

        # Add visibility camera for true occlusion-based point cloud filtering
        # Camera under env root Xform (not attached to robot), pose set at reset
        if cfg.mode.use_student_mode:
            self.scene.visibility_camera = TiledCameraCfg(
                prim_path="{ENV_REGEX_NS}/visibility_camera",
                data_types=["distance_to_image_plane", "instance_id_segmentation_fast"],
                spawn=PinholeCameraCfg(
                    focal_length=cfg.visibility_camera.focal_length,
                    horizontal_aperture=cfg.visibility_camera.horizontal_aperture,
                    clipping_range=cfg.visibility_camera.clipping_range,
                ),
                colorize_instance_id_segmentation=False,
                width=cfg.visibility_camera.width,
                height=cfg.visibility_camera.height,
            )


# ==============================================================================
# FINAL ENV CONFIG
# ==============================================================================

@configclass
class TrajectoryUR5eLeapEnvCfg(UR5eLeapTrajectoryMixinCfg, trajectory.TrajectoryEnvCfg):
    """UR5e + LEAP Hand trajectory following environment config.

    Config variant is selected via Y2R_VARIANT environment variable:
        Y2R_VARIANT=push ./scripts/push.sh --continue
    """
    pass
