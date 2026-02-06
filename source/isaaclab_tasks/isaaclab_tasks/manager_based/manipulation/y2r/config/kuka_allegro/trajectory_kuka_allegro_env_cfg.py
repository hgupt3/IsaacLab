# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kuka Allegro configuration for trajectory following task."""

from isaaclab_assets.robots import KUKA_ALLEGRO_CFG

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
# ACTION CONFIG BUILDERS - Create action configs dynamically from Y2RConfig
# ==============================================================================

def _build_rel_joint_action_cfg(cfg: Y2RConfig):
    """Build standard relative joint position action config."""
    # Explicit ordered joint list: arm joints first, then hand joints.
    KUKA_ALLEGRO_JOINT_ORDER = [
        # arm (7)
        "iiwa7_joint_1", "iiwa7_joint_2", "iiwa7_joint_3", "iiwa7_joint_4",
        "iiwa7_joint_5", "iiwa7_joint_6", "iiwa7_joint_7",
        # hand (16)
        "index_joint_0", "index_joint_1", "index_joint_2", "index_joint_3",
        "middle_joint_0", "middle_joint_1", "middle_joint_2", "middle_joint_3",
        "ring_joint_0",   "ring_joint_1",   "ring_joint_2",   "ring_joint_3",
        "thumb_joint_0",  "thumb_joint_1",  "thumb_joint_2",  "thumb_joint_3",
    ]
    
    @configclass
    class KukaAllegroRelJointPosActionCfg:
        """Standard relative joint position action (23D)."""
        action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=KUKA_ALLEGRO_JOINT_ORDER,
            scale=cfg.robot.action_scale,
        )
    
    return KukaAllegroRelJointPosActionCfg()


def _build_eigen_grasp_action_cfg(cfg: Y2RConfig):
    """Build eigen grasp action config."""
    # Explicit ordered joint list: arm joints first, then hand joints.
    KUKA_ALLEGRO_JOINT_ORDER = [
        # arm (7)
        "iiwa7_joint_1", "iiwa7_joint_2", "iiwa7_joint_3", "iiwa7_joint_4",
        "iiwa7_joint_5", "iiwa7_joint_6", "iiwa7_joint_7",
        # hand (16)
        "index_joint_0", "index_joint_1", "index_joint_2", "index_joint_3",
        "middle_joint_0", "middle_joint_1", "middle_joint_2", "middle_joint_3",
        "ring_joint_0",   "ring_joint_1",   "ring_joint_2",   "ring_joint_3",
        "thumb_joint_0",  "thumb_joint_1",  "thumb_joint_2",  "thumb_joint_3",
    ]
    
    @configclass
    class KukaAllegroEigenGraspActionCfg:
        """Eigen grasp action with residual (28D input -> 23D output)."""
        action = mdp.EigenGraspRelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=KUKA_ALLEGRO_JOINT_ORDER,
            scale=cfg.robot.action_scale,
            arm_joint_count=cfg.robot.arm_joint_count,
            hand_joint_count=cfg.robot.hand_joint_count,
            eigen_dim=cfg.robot.eigen_dim,
        )

    return KukaAllegroEigenGraspActionCfg()


def _build_kuka_allegro_rewards_cfg(cfg: Y2RConfig, base_rewards):
    """Add Kuka Allegro specific rewards to base rewards."""
    base_rewards.good_finger_contact = RewTerm(
        func=mdp.contacts,
        weight=cfg.rewards.good_finger_contact.weight,
        params={
            "phases": cfg.rewards.good_finger_contact.params.get("phases"),
            "use_hand_pose_gate": cfg.rewards.good_finger_contact.params.get("use_hand_pose_gate"),
            "invert_in_release": cfg.rewards.good_finger_contact.params.get("invert_in_release"),
            "threshold": cfg.rewards.good_finger_contact.params["threshold"],
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
            "contact_threshold": cfg.rewards.distal_joint3_penalty.params["contact_threshold"],
        },
    )
    return base_rewards


# ==============================================================================
# KUKA ALLEGRO MIXIN
# ==============================================================================

@configclass
class KukaAllegroTrajectoryMixinCfg:
    """Mixin config for Kuka Allegro robot in trajectory task.
    
    This mixin adds robot-specific configurations in __post_init__,
    using self.y2r_cfg which is set by the parent class.
    """
    
    # Placeholders - set in __post_init__
    actions = None

    def __post_init__(self: trajectory.TrajectoryEnvCfg):
        # Call parent __post_init__ first - this sets self.y2r_cfg
        super().__post_init__()
        
        cfg = self.y2r_cfg
        
        # Set up robot
        self.scene.robot = KUKA_ALLEGRO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # Select action space based on config flag
        if cfg.mode.use_eigen_grasp:
            self.actions = _build_eigen_grasp_action_cfg(cfg)
        else:
            self.actions = _build_rel_joint_action_cfg(cfg)
        
        # Add Kuka-specific rewards
        _build_kuka_allegro_rewards_cfg(cfg, self.rewards)

        # Setup contact sensors for fingertips
        finger_tip_body_list = ["index_link_3", "middle_link_3", "ring_link_3", "thumb_link_3"]
        for link_name in finger_tip_body_list:
            setattr(
                self.scene,
                f"{link_name}_object_s",
                ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ee_link/" + link_name,
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
                ),
            )

        # Contact force observation
        self.observations.proprio.contact = ObsTerm(
            func=mdp.fingers_contact_force_b,
            params={"contact_sensor_names": [f"{link}_object_s" for link in finger_tip_body_list]},
            clip=(-20.0, 20.0),
        )

        # Hand tips state observation
        self.observations.proprio.hand_tips_state_b.params["body_asset_cfg"].body_names = ["palm_link", ".*_tip"]

        # Fingers to object reward uses palm and fingertips
        self.rewards.fingers_to_object.params["asset_cfg"] = SceneEntityCfg("robot", body_names=["palm_link", ".*_tip"])

        # Add wrist camera if enabled (using TiledCamera for efficiency)
        if cfg.wrist_camera.enabled:
            from scipy.spatial.transform import Rotation
            
            # Config has [rx, ry, rz], transform for opengl convention
            rot_euler = cfg.wrist_camera.offset.rot
            rot_swapped = (rot_euler[0], rot_euler[2], -rot_euler[1]) # no idea whats going on here, but it works
            
            r = Rotation.from_euler('xyz', rot_swapped, degrees=True)
            quat = r.as_quat()  # [x, y, z, w]
            quat_wxyz = (float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2]))
            pos = tuple(float(x) for x in cfg.wrist_camera.offset.pos)
            
            self.scene.wrist_camera = TiledCameraCfg(
                prim_path="{ENV_REGEX_NS}/Robot/ee_link/palm_link/wrist_camera",
                offset=TiledCameraCfg.OffsetCfg(
                    pos=pos,
                    rot=quat_wxyz,
                    convention="opengl"
                ),
                data_types=["distance_to_image_plane"],  # Depth only
                spawn=PinholeCameraCfg(
                    focal_length=cfg.wrist_camera.focal_length,
                    horizontal_aperture=cfg.wrist_camera.horizontal_aperture,
                    clipping_range=cfg.wrist_camera.clipping_range,
                ),
                width=cfg.wrist_camera.width,
                height=cfg.wrist_camera.height,
            )


# ==============================================================================
# FINAL ENV CONFIG
# ==============================================================================

@configclass
class TrajectoryKukaAllegroEnvCfg(KukaAllegroTrajectoryMixinCfg, trajectory.TrajectoryEnvCfg):
    """Kuka Allegro trajectory following environment config.
    
    Config variant is selected via Y2R_VARIANT environment variable:
        Y2R_VARIANT=push ./scripts/push.sh --continue
    
    This loads the corresponding YAML from configs/{variant}.yaml
    """
    pass  # _config_name inherited from TrajectoryEnvCfg (reads Y2R_VARIANT env var)
