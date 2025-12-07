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
from ... import mdp


@configclass
class KukaAllegroRelJointPosActionCfg:
    action = mdp.RelativeJointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.1)


@configclass
class KukaAllegroTrajectoryRewardCfg(trajectory.RewardsCfg):
    """Rewards for Kuka Allegro trajectory task."""

    # Reward if 2+ finger tips in contact with object, one must be thumb
    good_finger_contact = RewTerm(
        func=mdp.contacts,
        weight=0.5,
        params={"threshold": 1.0},
    )


@configclass
class KukaAllegroTrajectoryMixinCfg:
    """Mixin config for Kuka Allegro robot in trajectory task."""

    rewards: KukaAllegroTrajectoryRewardCfg = KukaAllegroTrajectoryRewardCfg()
    actions: KukaAllegroRelJointPosActionCfg = KukaAllegroRelJointPosActionCfg()

    def __post_init__(self: trajectory.TrajectoryEnvCfg):
        super().__post_init__()
        self.commands.object_pose.body_name = "palm_link"
        self.scene.robot = KUKA_ALLEGRO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

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


@configclass
class TrajectoryKukaAllegroEnvCfg(KukaAllegroTrajectoryMixinCfg, trajectory.TrajectoryEnvCfg):
    """Kuka Allegro trajectory following task - Training."""

    pass


@configclass
class TrajectoryKukaAllegroEnvCfg_PLAY(KukaAllegroTrajectoryMixinCfg, trajectory.TrajectoryEnvCfg_PLAY):
    """Kuka Allegro trajectory following task - Evaluation."""

    pass
