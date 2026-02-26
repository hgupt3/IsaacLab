"""Configuration for the UR5e arm with LEAP Hand.

The following configurations are available:

* :obj:`UR5E_LEAP_CFG`: UR5e + LEAP Hand with implicit actuator model.

Reference:

* https://www.universal-robots.com/products/ur5-robot/
* https://leaphand.com/

"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Path to URDF
##

_Y2R_ASSETS_DIR = (
    Path(__file__).resolve().parents[3]
    / "isaaclab_tasks"
    / "isaaclab_tasks"
    / "manager_based"
    / "manipulation"
    / "y2r"
    / "assets"
    / "ur5e_leap"
)

##
# Configuration
##

UR5E_LEAP_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(_Y2R_ASSETS_DIR / "ur5e_leap_right_gemini305.urdf"),
        usd_dir=str(_Y2R_ASSETS_DIR / "ur5e_leap_right_gemini305_usd"),
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
                stiffness=None, damping=None,
            ),
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            # UR5e: base=90, shoulder=-70, elbow=80, wrist1=-10, wrist2=90, wrist3=0 (degrees)
            "ur5e_joint_1": 1.5708,
            "ur5e_joint_2": -1.2217,
            "ur5e_joint_3": 1.3963,
            "ur5e_joint_4": -0.1745,
            "ur5e_joint_5": 1.5708,
            "ur5e_joint_6": 0.0,
            "(index|middle|ring)_joint_0": 0.0,
            "(index|middle|ring)_joint_1": 0.3,
            "(index|middle|ring)_joint_2": 0.3,
            "(index|middle|ring)_joint_3": 0.3,
            "thumb_joint_0": 1.5,
            "thumb_joint_1": 0.60147215,
            "thumb_joint_2": 0.33795027,
            "thumb_joint_3": 0.60845138,
        },
    ),
    actuators={
        "ur5e_leap_actuators": ImplicitActuatorCfg(
            joint_names_expr=[
                "ur5e_joint_(1|2|3|4|5|6)",
                "index_joint_(0|1|2|3)",
                "middle_joint_(0|1|2|3)",
                "ring_joint_(0|1|2|3)",
                "thumb_joint_(0|1|2|3)",
            ],
            effort_limit_sim={
                "ur5e_joint_(1|2|3)": 150.0,
                "ur5e_joint_(4|5|6)": 28.0,
                "index_joint_(0|1|2|3)": 0.5,
                "middle_joint_(0|1|2|3)": 0.5,
                "ring_joint_(0|1|2|3)": 0.5,
                "thumb_joint_(0|1|2|3)": 0.5,
            },
            velocity_limit_sim={
                "ur5e_joint_(1|2|3|4|5|6)": 3.14159,
                "index_joint_(0|1|2|3)": 8.48,
                "middle_joint_(0|1|2|3)": 8.48,
                "ring_joint_(0|1|2|3)": 8.48,
                "thumb_joint_(0|1|2|3)": 8.48,
            },
            stiffness={
                "ur5e_joint_(1|2)": 600.0,
                "ur5e_joint_3": 300.0,
                "ur5e_joint_(4|5)": 100.0,
                "ur5e_joint_6": 50.0,
                "index_joint_(0|1|2|3)": 3.0,
                "middle_joint_(0|1|2|3)": 3.0,
                "ring_joint_(0|1|2|3)": 3.0,
                "thumb_joint_(0|1|2|3)": 3.0,
            },
            damping={
                "ur5e_joint_(1|2)": 60.0,
                "ur5e_joint_3": 35.0,
                "ur5e_joint_(4|5)": 20.0,
                "ur5e_joint_6": 15.0,
                "index_joint_(0|1|2|3)": 0.1,
                "middle_joint_(0|1|2|3)": 0.1,
                "ring_joint_(0|1|2|3)": 0.1,
                "thumb_joint_(0|1|2|3)": 0.1,
            },
            friction={
                "ur5e_joint_(1|2|3)": 1.0,
                "ur5e_joint_(4|5|6)": 0.5,
                "index_joint_(0|1|2|3)": 0.01,
                "middle_joint_(0|1|2|3)": 0.01,
                "ring_joint_(0|1|2|3)": 0.01,
                "thumb_joint_(0|1|2|3)": 0.01,
            },
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
