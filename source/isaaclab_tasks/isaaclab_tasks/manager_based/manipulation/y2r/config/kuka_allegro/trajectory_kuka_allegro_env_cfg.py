# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kuka Allegro configuration for trajectory following task."""

from isaaclab_assets.robots import KUKA_ALLEGRO_CFG

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from ... import trajectory_env_cfg as trajectory
from ...config_loader import Y2RConfig
from ... import mdp


# ==============================================================================
# ACTION CONFIG BUILDERS - Create action configs dynamically from Y2RConfig
# ==============================================================================

def _build_rel_joint_action_cfg(cfg: Y2RConfig):
    """Build standard relative joint position action config."""
    
    @configclass
    class KukaAllegroRelJointPosActionCfg:
        """Standard relative joint position action (23D)."""
        action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            scale=cfg.robot.action_scale,
        )
    
    return KukaAllegroRelJointPosActionCfg()


def _build_eigen_grasp_action_cfg(cfg: Y2RConfig):
    """Build eigen grasp action config."""
    
    @configclass
    class KukaAllegroEigenGraspActionCfg:
        """Eigen grasp action with residual (28D input -> 23D output)."""
        action = mdp.EigenGraspRelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            scale=cfg.robot.action_scale,
            arm_joint_count=cfg.robot.arm_joint_count,
            hand_joint_count=cfg.robot.hand_joint_count,
            eigen_dim=cfg.robot.eigen_dim,
        )

    return KukaAllegroEigenGraspActionCfg()


def _build_kuka_allegro_rewards_cfg(cfg: Y2RConfig, base_rewards):
    """Add Kuka Allegro specific rewards to base rewards."""
    # Add good_finger_contact reward to the base rewards
    base_rewards.good_finger_contact = RewTerm(
        func=mdp.contacts,
        weight=cfg.rewards.good_finger_contact.weight,
        params={"threshold": cfg.rewards.good_finger_contact.params.get("threshold", 1.0)},
    )
    # Penalize over-curling distal joints (joint_3) to discourage nail-side grasps
    base_rewards.distal_joint3_penalty = RewTerm(
        func=mdp.distal_joint3_penalty,
        weight=cfg.rewards.distal_joint3_penalty.weight,
        params={
            "std": cfg.rewards.distal_joint3_penalty.params.get("std", 1.0),
            "joint_name_regex": cfg.rewards.distal_joint3_penalty.params.get("joint_name_regex", ".*_joint_3"),
            "only_when_contact": cfg.rewards.distal_joint3_penalty.params.get("only_when_contact", True),
            "contact_threshold": cfg.rewards.distal_joint3_penalty.params.get("contact_threshold", 1.0),
            "only_in_manipulation": cfg.rewards.distal_joint3_penalty.params.get("only_in_manipulation", True),
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


# ==============================================================================
# FINAL ENV CONFIGS
# ==============================================================================

@configclass
class TrajectoryKukaAllegroEnvCfg(KukaAllegroTrajectoryMixinCfg, trajectory.TrajectoryEnvCfg):
    """Kuka Allegro trajectory following task - Training."""
    pass


@configclass
class TrajectoryKukaAllegroEnvCfg_PLAY(KukaAllegroTrajectoryMixinCfg, trajectory.TrajectoryEnvCfg_PLAY):
    """Kuka Allegro trajectory following task - Evaluation."""
    pass


@configclass
class TrajectoryKukaAllegroEnvCfg_PUSH(KukaAllegroTrajectoryMixinCfg, trajectory.TrajectoryEnvCfg_PUSH):
    """Kuka Allegro Push-T task - T-shape object with direct trajectory to outline goal."""
    pass
