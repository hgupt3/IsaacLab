"""UR5e + Gemini 305 + WSG50 configuration for trajectory following."""

from pathlib import Path

from isaaclab_assets.robots import UR5E_GEMINI_WSG50_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
from isaaclab.sim import PinholeCameraCfg
from isaaclab.utils import configclass

from ... import mdp
from ... import trajectory_env_cfg as trajectory
from ...config_loader import Y2RConfig


def _build_parallel_gripper_action_cfg(cfg: Y2RConfig):
    """Build UR5e arm + scalar WSG opening action config."""

    @configclass
    class UR5eGeminiWSG50ActionCfg:
        action = mdp.ParallelGripperRelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=cfg.robot.arm_joint_names + cfg.robot.hand_joint_names,
            arm_scale=cfg.robot.action_scale,
            gripper_scale=cfg.robot.gripper_action_scale,
            arm_joint_count=cfg.robot.arm_joint_count,
            gripper_opening_limits=(0.0054, 0.11),
            use_target_tracking=cfg.robot.use_target_tracking,
            ema_alpha_arm=cfg.robot.ema_alpha_arm,
            ema_alpha_gripper=cfg.robot.ema_alpha_hand,
        )

    return UR5eGeminiWSG50ActionCfg()


def _build_wsg_rewards_cfg(cfg: Y2RConfig, base_rewards):
    """Add WSG-specific contact reward and leave dexterous rewards disabled by YAML."""
    base_rewards.good_finger_contact = RewTerm(
        func=mdp.good_finger_contact,
        weight=cfg.rewards.good_finger_contact.weight,
        params={
            "phases": cfg.rewards.good_finger_contact.params.get("phases"),
            "use_hand_pose_gate": cfg.rewards.good_finger_contact.params.get("use_hand_pose_gate"),
            "invert_in_release": cfg.rewards.good_finger_contact.params.get("invert_in_release"),
        },
    )
    return base_rewards


@configclass
class UR5eGeminiWSG50TrajectoryMixinCfg:
    """Mixin config for UR5e + Gemini + WSG50 in the Y2R trajectory task."""

    actions = None

    def __post_init__(self: trajectory.TrajectoryEnvCfg):
        super().__post_init__()

        cfg = self.y2r_cfg
        table_surface_z = cfg.workspace.table_surface_z
        self.scene.robot = UR5E_GEMINI_WSG50_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = (0.0, 0.0, table_surface_z)

        self.scene.table = RigidObjectCfg(
            prim_path="/World/envs/env_.*/table",
            spawn=sim_utils.CuboidCfg(
                size=(1.13, 1.27, 0.01),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.91, 0.90), roughness=0.4),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(-0.45, 0.0, table_surface_z - 0.005),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )

        self.actions = _build_parallel_gripper_action_cfg(cfg)
        _build_wsg_rewards_cfg(cfg, self.rewards)

        self.rewards.arm_table_penalty.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=["ur5e_link_(3|4|5|6)|wsg_base_link|wsg_(left|right)_slider_link"]
        )

        layout = cfg.robot.contact_layout
        all_sensor_bodies = sorted({body for group in layout.parallel_jaw_bodies for body in group})
        for link_name in all_sensor_bodies:
            setattr(
                self.scene,
                f"{link_name}_object_s",
                ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/" + layout.sensor_prim_prefix + link_name,
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
                ),
            )

        fin_sensor_names = [f"{group[0]}_object_s" for group in layout.parallel_jaw_bodies]
        self.observations.proprio.contact = ObsTerm(
            func=mdp.fingers_contact_force_b,
            params={"contact_sensor_names": fin_sensor_names},
            clip=(-20.0, 20.0),
        )
        self.observations.proprio.link_2_contact = ObsTerm(
            func=mdp.zeros_placeholder,
            params={"size": 0},
        )
        self.observations.proprio.object_pose_palm = ObsTerm(func=mdp.object_pose_palm_b)
        self.observations.proprio.hand_tips_state_b = ObsTerm(
            func=mdp.body_state_with_offsets_b,
            noise=self.observations.proprio.hand_tips_state_b.noise,
            params={
                "body_asset_cfg": SceneEntityCfg("robot", body_names=cfg.robot.tip_state_body_names),
                "base_asset_cfg": SceneEntityCfg("robot"),
                "body_names": cfg.robot.tip_state_body_names,
            },
        )

        self.rewards.fingers_to_object.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=cfg.robot.tip_state_body_names
        )
        self.rewards.fingers_to_object.params["body_names"] = cfg.robot.tip_state_body_names

        if cfg.mode.use_student_mode and cfg.mode.use_depth_camera:
            from scipy.spatial.transform import Rotation

            rot_euler = cfg.wrist_camera.offset.rot
            rot_swapped = (rot_euler[0], rot_euler[2], -rot_euler[1])
            quat = Rotation.from_euler("xyz", rot_swapped, degrees=True).as_quat()
            quat_wxyz = (float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2]))
            pos = tuple(float(x) for x in cfg.wrist_camera.offset.pos)

            self.scene.wrist_camera = TiledCameraCfg(
                prim_path="{ENV_REGEX_NS}/Robot/ur5e_link_6/wrist_camera",
                offset=TiledCameraCfg.OffsetCfg(pos=pos, rot=quat_wxyz, convention="opengl"),
                data_types=["distance_to_image_plane"],
                spawn=PinholeCameraCfg(
                    focal_length=cfg.wrist_camera.focal_length,
                    horizontal_aperture=cfg.wrist_camera.horizontal_aperture,
                    clipping_range=cfg.wrist_camera.clipping_range,
                ),
                width=cfg.wrist_camera.width,
                height=cfg.wrist_camera.height,
            )

        if cfg.mode.use_student_mode:
            policy_dt = float(cfg.simulation.physics_dt * cfg.simulation.decimation)
            steps_per_target = 1.0 / (cfg.trajectory.target_hz * policy_dt)
            visibility_resample_interval = max(1, int(cfg.trajectory.window_size * steps_per_target))
            visibility_update_period = policy_dt * visibility_resample_interval

            self.scene.visibility_camera = TiledCameraCfg(
                prim_path="{ENV_REGEX_NS}/visibility_camera",
                update_period=visibility_update_period,
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

            realsense_assets = Path(__file__).resolve().parent.parent.parent / "assets" / "realsense_d435"
            self.scene.visibility_camera_body = RigidObjectCfg(
                prim_path="/World/envs/env_.*/visibility_camera_body",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=str(realsense_assets / "camera" / "d435.usd"),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.75, 0.75), roughness=0.5),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            )


@configclass
class TrajectoryUR5eGeminiWSG50EnvCfg(UR5eGeminiWSG50TrajectoryMixinCfg, trajectory.TrajectoryEnvCfg):
    """UR5e + Gemini + WSG50 trajectory following environment config."""

    pass
