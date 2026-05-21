"""Configuration for the UR5e arm with WSG50 gripper and Gemini 305 camera.

The following configurations are available:

* :obj:`UR5E_GEMINI_WSG50_CFG`: UR5e + WSG50 + Gemini 305 with implicit actuator model.

Reference:

* https://www.universal-robots.com/products/ur5-robot/
* https://github.com/caelan/pybullet-planning/tree/master/models/drake/wsg_50_description

"""

import math
from pathlib import Path

import yaml

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Path to URDF
##

_Y2R_WSG50_ASSETS_DIR = (
    Path(__file__).resolve().parents[3]
    / "isaaclab_tasks"
    / "isaaclab_tasks"
    / "manager_based"
    / "manipulation"
    / "y2r"
    / "assets"
    / "ur5e_gemini_wsg50"
)
_CALIBRATION_PATH = _Y2R_WSG50_ASSETS_DIR / "calibration.yaml"

with _CALIBRATION_PATH.open("r") as f:
    _CALIBRATION = yaml.safe_load(f)

_DEFAULT_POSE = _CALIBRATION["default_pose"]
_DEFAULT_ARM_JOINTS_DEG = _DEFAULT_POSE["arm_joints_deg"]
_DEFAULT_WSG_OPENING_M = float(_DEFAULT_POSE["wsg_opening_m"])
if _DEFAULT_WSG_OPENING_M < 0.0054 or _DEFAULT_WSG_OPENING_M > 0.11:
    raise ValueError(f"default_pose.wsg_opening_m must be in [0.0054, 0.11], got {_DEFAULT_WSG_OPENING_M}")

_DEFAULT_WSG_HALF_OPENING_M = _DEFAULT_WSG_OPENING_M / 2.0

##
# Configuration
##

UR5E_GEMINI_WSG50_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(_Y2R_WSG50_ASSETS_DIR / "ur5e_gemini_wsg50.urdf"),
        usd_dir=str(_Y2R_WSG50_ASSETS_DIR / "ur5e_gemini_wsg50_usd"),
        fix_base=True,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=True,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=True,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=None,
                damping=None,
            ),
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "ur5e_joint_1": math.radians(float(_DEFAULT_ARM_JOINTS_DEG["ur5e_joint_1"])),
            "ur5e_joint_2": math.radians(float(_DEFAULT_ARM_JOINTS_DEG["ur5e_joint_2"])),
            "ur5e_joint_3": math.radians(float(_DEFAULT_ARM_JOINTS_DEG["ur5e_joint_3"])),
            "ur5e_joint_4": math.radians(float(_DEFAULT_ARM_JOINTS_DEG["ur5e_joint_4"])),
            "ur5e_joint_5": math.radians(float(_DEFAULT_ARM_JOINTS_DEG["ur5e_joint_5"])),
            "ur5e_joint_6": math.radians(float(_DEFAULT_ARM_JOINTS_DEG["ur5e_joint_6"])),
            "wsg_left_finger_joint": -_DEFAULT_WSG_HALF_OPENING_M,
            "wsg_right_finger_joint": _DEFAULT_WSG_HALF_OPENING_M,
        },
    ),
    actuators={
        "ur5e_arm_actuators": ImplicitActuatorCfg(
            joint_names_expr=["ur5e_joint_(1|2|3|4|5|6)"],
            effort_limit_sim={
                "ur5e_joint_(1|2|3)": 150.0,
                "ur5e_joint_(4|5|6)": 28.0,
            },
            velocity_limit_sim={"ur5e_joint_(1|2|3|4|5|6)": 3.14159},
            stiffness={
                "ur5e_joint_(1|2)": 600.0,
                "ur5e_joint_3": 300.0,
                "ur5e_joint_(4|5)": 100.0,
                "ur5e_joint_6": 50.0,
            },
            damping={
                "ur5e_joint_(1|2)": 49.0,
                "ur5e_joint_3": 34.6,
                "ur5e_joint_(4|5)": 20.0,
                "ur5e_joint_6": 14.1,
            },
            armature={"ur5e_joint_(1|2|3|4|5|6)": 0.0},
            friction={
                "ur5e_joint_1": 1.0,
                "ur5e_joint_2": 15.0,
                "ur5e_joint_3": 7.0,
                "ur5e_joint_4": 1.5,
                "ur5e_joint_5": 1.0,
                "ur5e_joint_6": 0.3,
            },
        ),
        "wsg50_gripper_actuators": ImplicitActuatorCfg(
            joint_names_expr=["wsg_(left|right)_finger_joint"],
            effort_limit_sim={"wsg_(left|right)_finger_joint": 100.0},
            velocity_limit_sim={"wsg_(left|right)_finger_joint": 0.2},
            stiffness={"wsg_(left|right)_finger_joint": 2000.0},
            damping={"wsg_(left|right)_finger_joint": 80.0},
            armature={"wsg_(left|right)_finger_joint": 0.001},
            friction={"wsg_(left|right)_finger_joint": 1.0},
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
