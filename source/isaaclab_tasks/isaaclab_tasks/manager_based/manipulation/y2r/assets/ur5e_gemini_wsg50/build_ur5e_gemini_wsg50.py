#!/usr/bin/env python3
"""Build and validate the UR5e + Gemini + WSG50 calibration URDF."""

from __future__ import annotations

import copy
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml


ASSET_DIR = Path(__file__).resolve().parent
CALIBRATION_PATH = ASSET_DIR / "calibration.yaml"

TRANSFORM_KEYS = [
    "ee_to_wsg_mount",
    "wsg_mount_to_wsg_base",
    "wsg_base_to_wsg_mount_visual",
    "wsg_base_to_camera_mount",
    "camera_mount_to_gemini",
    "left_slider_to_soft_finger",
    "right_slider_to_soft_finger",
    "left_soft_finger_to_holder_visual",
    "right_soft_finger_to_holder_visual",
    "left_soft_finger_to_tip",
    "right_soft_finger_to_tip",
    "wsg_base_to_palm_frame",
]

KEEP_LINKS = [
    "base",
    "ur5e_base_link",
    "ur5e_link_0",
    "ur5e_link_1",
    "ur5e_link_2",
    "ur5e_link_3",
    "ur5e_link_4",
    "ur5e_link_5",
    "ur5e_link_6",
    "ur5e_base_fixed_link",
    "ee_link",
]

KEEP_JOINTS = [
    "base_yaw_joint",
    "ur5e_base_inertia_joint",
    "ur5e_joint_1",
    "ur5e_joint_2",
    "ur5e_joint_3",
    "ur5e_joint_4",
    "ur5e_joint_5",
    "ur5e_joint_6",
    "ur5e_base_fixed_joint",
    "ur5e_flange_joint",
]

EXPECTED_LINKS = KEEP_LINKS + [
    "wsg_mount_link",
    "wsg_base_link",
    "wsg_mount_visual_link",
    "wsg_left_slider_link",
    "wsg_right_slider_link",
    "soft_finger_left_link",
    "soft_finger_right_link",
    "finger_holder_left_visual_link",
    "finger_holder_right_visual_link",
    "gemini_mount_link",
    "gemini_305_link",
    "gemini_305_left_camera_frame",
    "gemini_305_left_camera_optical_frame",
    "gemini_305_right_camera_frame",
    "gemini_305_right_camera_optical_frame",
    "palm_frame",
    "left_tip",
    "right_tip",
]

CUSTOM_PHYSICAL_LINKS = [
    "wsg_mount_link",
    "wsg_base_link",
    "wsg_left_slider_link",
    "wsg_right_slider_link",
    "soft_finger_left_link",
    "soft_finger_right_link",
    "gemini_mount_link",
    "gemini_305_link",
]

CUSTOM_VISUAL_ONLY_LINKS = [
    "wsg_mount_visual_link",
    "finger_holder_left_visual_link",
    "finger_holder_right_visual_link",
]

REQUIRED_COLLISION_LINKS = [
    "wsg_base_link",
    "soft_finger_left_link",
    "soft_finger_right_link",
    "gemini_mount_link",
    "gemini_305_link",
]

EXPECTED_JOINTS = KEEP_JOINTS + [
    "wsg_mount_joint",
    "wsg_base_mount_joint",
    "wsg_mount_visual_joint",
    "wsg_left_finger_joint",
    "wsg_right_finger_joint",
    "soft_finger_left_joint",
    "soft_finger_right_joint",
    "finger_holder_left_visual_joint",
    "finger_holder_right_visual_joint",
    "left_tip_joint",
    "right_tip_joint",
    "palm_frame_joint",
    "gemini_mount_joint",
    "gemini_305_mount_joint",
    "gemini_305_left_camera_joint",
    "gemini_305_left_optical_joint",
    "gemini_305_right_camera_joint",
    "gemini_305_right_optical_joint",
]


def main() -> None:
    config = load_calibration(CALIBRATION_PATH)
    output_path = ASSET_DIR / str(config["model"]["generated_urdf"])
    robot = build_robot(config)
    write_robot(robot, output_path)
    validate_robot(output_path)
    print(f"Wrote {output_path}")


def load_calibration(path: Path) -> dict:
    with path.open("r") as f:
        config = yaml.safe_load(f)

    transforms = config["transforms"]
    for key in TRANSFORM_KEYS:
        transform = transforms[key]
        xyz = transform["xyz"]
        rpy_deg = transform["rpy_deg"]
        if len(xyz) != 3:
            raise ValueError(f"{key}.xyz must contain exactly 3 values")
        if len(rpy_deg) != 3:
            raise ValueError(f"{key}.rpy_deg must contain exactly 3 values")
        for value in list(xyz) + list(rpy_deg):
            float(value)

    default_pose = config["default_pose"]
    for joint_name in KEEP_JOINTS:
        if joint_name.startswith("ur5e_joint_"):
            float(default_pose["arm_joints_deg"][joint_name])
    float(default_pose["wsg_opening_m"])
    validate_contact_markers(config["contact_markers"])
    validate_collision_primitives(config["collision_primitives"])
    int(config["editor"]["port"])
    source_urdf = (ASSET_DIR / str(config["model"]["source_urdf"])).resolve()
    if not source_urdf.exists():
        raise FileNotFoundError(f"Missing source URDF: {source_urdf}")
    return config


def validate_contact_markers(contact_markers: dict) -> None:
    pad_normals = contact_markers["pad_normals"]
    expected = {
        "soft_finger_left_link": "left_tip",
        "soft_finger_right_link": "right_tip",
    }
    for body_link, origin_link in expected.items():
        marker = pad_normals[body_link]
        if marker["origin_link"] != origin_link:
            raise ValueError(f"{body_link}.origin_link expected {origin_link}, got {marker['origin_link']}")
        normal = marker["normal"]
        if len(normal) != 3:
            raise ValueError(f"{body_link}.normal must contain exactly 3 values")
        normal_values = [float(value) for value in normal]
        if sum(value * value for value in normal_values) <= 1e-12:
            raise ValueError(f"{body_link}.normal must be non-zero")


def validate_collision_primitives(collision_primitives: dict) -> None:
    for link_name in CUSTOM_PHYSICAL_LINKS:
        primitives = collision_primitives[link_name]
        if link_name in REQUIRED_COLLISION_LINKS and len(primitives) == 0:
            raise ValueError(f"{link_name} must have at least one collision primitive")
        for primitive in primitives:
            primitive_type = primitive["type"]
            xyz = primitive["xyz"]
            rpy_deg = primitive["rpy_deg"]
            if len(xyz) != 3:
                raise ValueError(f"{link_name} collision xyz must contain exactly 3 values")
            if len(rpy_deg) != 3:
                raise ValueError(f"{link_name} collision rpy_deg must contain exactly 3 values")
            for value in list(xyz) + list(rpy_deg):
                float(value)
            if primitive_type == "box":
                size = primitive["size"]
                if len(size) != 3:
                    raise ValueError(f"{link_name} box size must contain exactly 3 values")
                for value in size:
                    if float(value) <= 0.0:
                        raise ValueError(f"{link_name} box size values must be positive")
            elif primitive_type == "cylinder":
                if float(primitive["radius"]) <= 0.0:
                    raise ValueError(f"{link_name} cylinder radius must be positive")
                if float(primitive["length"]) <= 0.0:
                    raise ValueError(f"{link_name} cylinder length must be positive")
            else:
                raise ValueError(f"{link_name} unsupported collision primitive type: {primitive_type}")


def build_robot(config: dict) -> ET.Element:
    source_urdf = (ASSET_DIR / str(config["model"]["source_urdf"])).resolve()
    source_root = ET.parse(source_urdf).getroot()
    robot = ET.Element("robot", {"name": "ur5e_gemini_wsg50"})
    robot.append(ET.Comment("UR5e + WSG50 + Gemini 305. WSG50 source: caelan/pybullet-planning models/drake/wsg_50_description."))

    for material in source_root.findall("material"):
        robot.append(copy.deepcopy(material))
    add_material(robot, "wsg_gray", (0.50, 0.50, 0.50, 1.0))
    add_material(robot, "wsg_black", (0.02, 0.02, 0.02, 1.0))
    add_material(robot, "mount_black", (0.0, 0.0, 0.0, 1.0))
    add_material(robot, "fin_mount_white", (1.0, 1.0, 1.0, 1.0))
    add_material(robot, "soft_finger_black", (0.0, 0.0, 0.0, 1.0))
    add_material(robot, "gemini_305_gray", (0.45, 0.45, 0.45, 1.0))

    copy_named_elements(source_root, robot, "link", KEEP_LINKS)
    copy_named_elements(source_root, robot, "joint", KEEP_JOINTS)

    add_custom_links(robot, config["collision_primitives"])
    add_custom_joints(robot, config["transforms"])
    return robot


def copy_named_elements(source_root: ET.Element, robot: ET.Element, tag: str, names: list[str]) -> None:
    source = {element.attrib["name"]: element for element in source_root.findall(tag)}
    for name in names:
        robot.append(copy.deepcopy(source[name]))


def add_material(robot: ET.Element, name: str, rgba: tuple[float, float, float, float]) -> None:
    material = ET.SubElement(robot, "material", {"name": name})
    ET.SubElement(material, "color", {"rgba": fmt_values(rgba)})


def add_custom_links(robot: ET.Element, collision_primitives: dict) -> None:
    add_inertial_only_link(
        robot,
        "wsg_mount_link",
        mass=0.15,
        inertia_diag=(0.00012, 0.00012, 0.00012),
    )
    add_mesh_link(
        robot,
        "wsg_base_link",
        "wsg50/meshes/WSG50_110.stl",
        "wsg_gray",
        mass=1.20,
        inertia_diag=(0.0020, 0.0020, 0.0020),
        collision_primitives=collision_primitives["wsg_base_link"],
    )
    add_mesh_link(
        robot,
        "wsg_mount_visual_link",
        "external_meshes/wsg50_mount.stl",
        "mount_black",
        mass=0.001,
        inertia_diag=(1e-9, 1e-9, 1e-9),
        collision_primitives=[],
    )
    add_mesh_link(
        robot,
        "wsg_left_slider_link",
        "wsg50/meshes/GUIDE_WSG50_110.stl",
        "wsg_black",
        mass=0.10,
        inertia_diag=(0.00008, 0.00008, 0.00008),
        scale=(0.001, 0.001, 0.001),
        collision_primitives=collision_primitives["wsg_left_slider_link"],
    )
    add_mesh_link(
        robot,
        "wsg_right_slider_link",
        "wsg50/meshes/GUIDE_WSG50_110.stl",
        "wsg_black",
        mass=0.10,
        inertia_diag=(0.00008, 0.00008, 0.00008),
        scale=(0.001, 0.001, 0.001),
        visual_rpy=(0.0, 0.0, math.pi),
        collision_rpy=(0.0, 0.0, math.pi),
        collision_primitives=collision_primitives["wsg_right_slider_link"],
    )
    add_mesh_link(
        robot,
        "soft_finger_left_link",
        "external_meshes/soft_gripper_finger_left.stl",
        "soft_finger_black",
        mass=0.05,
        inertia_diag=(0.00004, 0.00004, 0.00004),
        collision_primitives=collision_primitives["soft_finger_left_link"],
    )
    add_mesh_link(
        robot,
        "soft_finger_right_link",
        "external_meshes/soft_gripper_finger_right.stl",
        "soft_finger_black",
        mass=0.05,
        inertia_diag=(0.00004, 0.00004, 0.00004),
        collision_primitives=collision_primitives["soft_finger_right_link"],
    )
    add_mesh_link(
        robot,
        "finger_holder_left_visual_link",
        "external_meshes/finger_holder_left.obj",
        "fin_mount_white",
        mass=0.001,
        inertia_diag=(1e-9, 1e-9, 1e-9),
        collision_primitives=[],
    )
    add_mesh_link(
        robot,
        "finger_holder_right_visual_link",
        "external_meshes/finger_holder_right.obj",
        "fin_mount_white",
        mass=0.001,
        inertia_diag=(1e-9, 1e-9, 1e-9),
        collision_primitives=[],
    )
    add_mesh_link(
        robot,
        "gemini_mount_link",
        "external_meshes/gemini_mount.stl",
        "mount_black",
        mass=0.04,
        inertia_diag=(0.00001, 0.00001, 0.00001),
        collision_primitives=collision_primitives["gemini_mount_link"],
    )
    add_mesh_link(
        robot,
        "gemini_305_link",
        "gemini_305/meshes/gemini_305_rotated.obj",
        "gemini_305_gray",
        mass=0.065,
        inertia_diag=(0.00001, 0.00001, 0.00001),
        scale=(0.001, 0.001, 0.001),
        visual_rpy=(math.pi, 0.0, 0.0),
        collision_box=(0.042, 0.042, 0.025),
        collision_primitives=collision_primitives["gemini_305_link"],
    )
    for name in (
        "gemini_305_left_camera_frame",
        "gemini_305_left_camera_optical_frame",
        "gemini_305_right_camera_frame",
        "gemini_305_right_camera_optical_frame",
        "palm_frame",
        "left_tip",
        "right_tip",
    ):
        add_virtual_link(robot, name)


def add_mesh_link(
    robot: ET.Element,
    name: str,
    mesh_filename: str,
    material_name: str,
    mass: float,
    inertia_diag: tuple[float, float, float],
    scale: tuple[float, float, float] | None = None,
    visual_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
    collision_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
    collision_box: tuple[float, float, float] | None = None,
    collision_primitives: list[dict] | None = None,
) -> None:
    link = ET.SubElement(robot, "link", {"name": name})
    visual = ET.SubElement(link, "visual")
    ET.SubElement(visual, "origin", {"xyz": "0 0 0", "rpy": fmt_values(visual_rpy)})
    geometry = ET.SubElement(visual, "geometry")
    mesh_attrib = {"filename": mesh_filename}
    if scale is not None:
        mesh_attrib["scale"] = fmt_values(scale)
    ET.SubElement(geometry, "mesh", mesh_attrib)
    ET.SubElement(visual, "material", {"name": material_name})

    if collision_primitives is None:
        collision = ET.SubElement(link, "collision")
        ET.SubElement(collision, "origin", {"xyz": "0 0 0", "rpy": fmt_values(collision_rpy)})
        collision_geometry = ET.SubElement(collision, "geometry")
        if collision_box is None:
            collision_mesh_attrib = {"filename": mesh_filename}
            if scale is not None:
                collision_mesh_attrib["scale"] = fmt_values(scale)
            ET.SubElement(collision_geometry, "mesh", collision_mesh_attrib)
        else:
            ET.SubElement(collision_geometry, "box", {"size": fmt_values(collision_box)})
    else:
        for primitive in collision_primitives:
            add_collision_primitive(link, primitive)
    add_inertial(link, mass, inertia_diag)


def add_inertial_only_link(
    robot: ET.Element,
    name: str,
    mass: float,
    inertia_diag: tuple[float, float, float],
) -> None:
    link = ET.SubElement(robot, "link", {"name": name})
    add_inertial(link, mass, inertia_diag)


def add_collision_primitive(link: ET.Element, primitive: dict) -> None:
    collision = ET.SubElement(link, "collision")
    xyz = tuple(float(value) for value in primitive["xyz"])
    rpy = tuple(math.radians(float(value)) for value in primitive["rpy_deg"])
    ET.SubElement(collision, "origin", {"xyz": fmt_values(xyz), "rpy": fmt_values(rpy)})
    geometry = ET.SubElement(collision, "geometry")
    primitive_type = primitive["type"]
    if primitive_type == "box":
        size = tuple(float(value) for value in primitive["size"])
        ET.SubElement(geometry, "box", {"size": fmt_values(size)})
    elif primitive_type == "cylinder":
        ET.SubElement(
            geometry,
            "cylinder",
            {
                "radius": fmt_float(float(primitive["radius"])),
                "length": fmt_float(float(primitive["length"])),
            },
        )
    else:
        raise ValueError(f"Unsupported collision primitive type: {primitive_type}")


def add_virtual_link(robot: ET.Element, name: str) -> None:
    link = ET.SubElement(robot, "link", {"name": name})
    add_inertial(link, 0.001, (1e-9, 1e-9, 1e-9))


def add_inertial(link: ET.Element, mass: float, inertia_diag: tuple[float, float, float]) -> None:
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "mass", {"value": fmt_float(mass)})
    ET.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(
        inertial,
        "inertia",
        {
            "ixx": fmt_float(inertia_diag[0]),
            "ixy": "0",
            "ixz": "0",
            "iyy": fmt_float(inertia_diag[1]),
            "iyz": "0",
            "izz": fmt_float(inertia_diag[2]),
        },
    )


def add_custom_joints(robot: ET.Element, transforms: dict) -> None:
    add_fixed_joint(robot, "wsg_mount_joint", "ee_link", "wsg_mount_link", transforms["ee_to_wsg_mount"])
    add_fixed_joint(robot, "wsg_base_mount_joint", "wsg_mount_link", "wsg_base_link", transforms["wsg_mount_to_wsg_base"])
    add_fixed_joint(
        robot,
        "wsg_mount_visual_joint",
        "wsg_base_link",
        "wsg_mount_visual_link",
        transforms["wsg_base_to_wsg_mount_visual"],
    )
    add_prismatic_joint(robot, "wsg_left_finger_joint", "wsg_base_link", "wsg_left_slider_link", (1.0, 0.0, 0.0), -0.055, -0.0027)
    add_prismatic_joint(robot, "wsg_right_finger_joint", "wsg_base_link", "wsg_right_slider_link", (1.0, 0.0, 0.0), 0.0027, 0.055)
    add_fixed_joint(
        robot,
        "soft_finger_left_joint",
        "wsg_left_slider_link",
        "soft_finger_left_link",
        transforms["left_slider_to_soft_finger"],
    )
    add_fixed_joint(
        robot,
        "soft_finger_right_joint",
        "wsg_right_slider_link",
        "soft_finger_right_link",
        transforms["right_slider_to_soft_finger"],
    )
    add_fixed_joint(
        robot,
        "finger_holder_left_visual_joint",
        "soft_finger_left_link",
        "finger_holder_left_visual_link",
        transforms["left_soft_finger_to_holder_visual"],
    )
    add_fixed_joint(
        robot,
        "finger_holder_right_visual_joint",
        "soft_finger_right_link",
        "finger_holder_right_visual_link",
        transforms["right_soft_finger_to_holder_visual"],
    )
    add_fixed_joint(robot, "left_tip_joint", "soft_finger_left_link", "left_tip", transforms["left_soft_finger_to_tip"])
    add_fixed_joint(robot, "right_tip_joint", "soft_finger_right_link", "right_tip", transforms["right_soft_finger_to_tip"])
    add_fixed_joint(robot, "palm_frame_joint", "wsg_base_link", "palm_frame", transforms["wsg_base_to_palm_frame"])
    add_fixed_joint(robot, "gemini_mount_joint", "wsg_base_link", "gemini_mount_link", transforms["wsg_base_to_camera_mount"])
    add_fixed_joint(robot, "gemini_305_mount_joint", "gemini_mount_link", "gemini_305_link", transforms["camera_mount_to_gemini"])
    add_fixed_joint(robot, "gemini_305_left_camera_joint", "gemini_305_link", "gemini_305_left_camera_frame", xyz=(0.009, 0.0, 0.0126))
    add_fixed_joint(
        robot,
        "gemini_305_left_optical_joint",
        "gemini_305_left_camera_frame",
        "gemini_305_left_camera_optical_frame",
        rpy=(0.0, 0.0, math.pi),
    )
    add_fixed_joint(robot, "gemini_305_right_camera_joint", "gemini_305_link", "gemini_305_right_camera_frame", xyz=(-0.009, 0.0, 0.0126))
    add_fixed_joint(
        robot,
        "gemini_305_right_optical_joint",
        "gemini_305_right_camera_frame",
        "gemini_305_right_camera_optical_frame",
        rpy=(0.0, 0.0, math.pi),
    )


def add_fixed_joint(
    robot: ET.Element,
    name: str,
    parent: str,
    child: str,
    transform: dict | None = None,
    xyz: tuple[float, float, float] | None = None,
    rpy: tuple[float, float, float] | None = None,
) -> None:
    joint = ET.SubElement(robot, "joint", {"name": name, "type": "fixed"})
    ET.SubElement(joint, "parent", {"link": parent})
    ET.SubElement(joint, "child", {"link": child})
    if transform is not None:
        origin_xyz = tuple(float(v) for v in transform["xyz"])
        origin_rpy = tuple(math.radians(float(v)) for v in transform["rpy_deg"])
    else:
        origin_xyz = xyz if xyz is not None else (0.0, 0.0, 0.0)
        origin_rpy = rpy if rpy is not None else (0.0, 0.0, 0.0)
    ET.SubElement(joint, "origin", {"xyz": fmt_values(origin_xyz), "rpy": fmt_values(origin_rpy)})


def add_prismatic_joint(
    robot: ET.Element,
    name: str,
    parent: str,
    child: str,
    axis: tuple[float, float, float],
    lower: float,
    upper: float,
) -> None:
    joint = ET.SubElement(robot, "joint", {"name": name, "type": "prismatic"})
    ET.SubElement(joint, "parent", {"link": parent})
    ET.SubElement(joint, "child", {"link": child})
    ET.SubElement(joint, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(joint, "axis", {"xyz": fmt_values(axis)})
    ET.SubElement(
        joint,
        "limit",
        {"lower": fmt_float(lower), "upper": fmt_float(upper), "effort": "100", "velocity": "0.2"},
    )
    ET.SubElement(joint, "dynamics", {"damping": "10", "friction": "1"})


def write_robot(robot: ET.Element, output_path: Path) -> None:
    tree = ET.ElementTree(robot)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


def validate_robot(output_path: Path) -> None:
    root = ET.parse(output_path).getroot()
    link_names = [link.attrib["name"] for link in root.findall("link")]
    joint_names = [joint.attrib["name"] for joint in root.findall("joint")]

    missing_links = [name for name in EXPECTED_LINKS if name not in link_names]
    if missing_links:
        raise ValueError(f"Missing expected links: {missing_links}")
    missing_joints = [name for name in EXPECTED_JOINTS if name not in joint_names]
    if missing_joints:
        raise ValueError(f"Missing expected joints: {missing_joints}")

    prismatic_wsg_joints = [
        joint.attrib["name"]
        for joint in root.findall("joint")
        if joint.attrib["type"] == "prismatic" and joint.attrib["name"].startswith("wsg_")
    ]
    if sorted(prismatic_wsg_joints) != ["wsg_left_finger_joint", "wsg_right_finger_joint"]:
        raise ValueError(f"Expected exactly two WSG prismatic joints, got {prismatic_wsg_joints}")

    parent_child = {
        joint.attrib["name"]: (joint.find("parent").attrib["link"], joint.find("child").attrib["link"])
        for joint in root.findall("joint")
    }
    expected_parent_child = {
        "wsg_mount_joint": ("ee_link", "wsg_mount_link"),
        "wsg_base_mount_joint": ("wsg_mount_link", "wsg_base_link"),
        "wsg_mount_visual_joint": ("wsg_base_link", "wsg_mount_visual_link"),
        "wsg_left_finger_joint": ("wsg_base_link", "wsg_left_slider_link"),
        "wsg_right_finger_joint": ("wsg_base_link", "wsg_right_slider_link"),
        "soft_finger_left_joint": ("wsg_left_slider_link", "soft_finger_left_link"),
        "soft_finger_right_joint": ("wsg_right_slider_link", "soft_finger_right_link"),
        "finger_holder_left_visual_joint": ("soft_finger_left_link", "finger_holder_left_visual_link"),
        "finger_holder_right_visual_joint": ("soft_finger_right_link", "finger_holder_right_visual_link"),
        "gemini_mount_joint": ("wsg_base_link", "gemini_mount_link"),
        "gemini_305_mount_joint": ("gemini_mount_link", "gemini_305_link"),
    }
    for joint_name, expected in expected_parent_child.items():
        if parent_child[joint_name] != expected:
            raise ValueError(f"{joint_name} expected {expected}, got {parent_child[joint_name]}")

    for link in root.findall("link"):
        link_name = link.attrib["name"]
        if link_name not in CUSTOM_PHYSICAL_LINKS:
            continue
        collisions = link.findall("collision")
        if link_name in REQUIRED_COLLISION_LINKS and len(collisions) == 0:
            raise ValueError(f"{link_name} is missing collision primitives")
        for collision in collisions:
            geometry = collision.find("geometry")
            if geometry.find("mesh") is not None:
                raise ValueError(f"{link_name} collision must use primitives, not mesh")
            if geometry.find("box") is None and geometry.find("cylinder") is None:
                raise ValueError(f"{link_name} collision must be box or cylinder")

    for link in root.findall("link"):
        link_name = link.attrib["name"]
        if link_name not in CUSTOM_VISUAL_ONLY_LINKS:
            continue
        if link.findall("collision"):
            raise ValueError(f"{link_name} must remain visual-only and not define collision geometry")
        if not link.findall("visual"):
            raise ValueError(f"{link_name} must define visual geometry")

    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib["filename"]
        if filename.startswith("package://"):
            raise ValueError(f"Generated URDF should not contain package URI: {filename}")
        mesh_path = ASSET_DIR / filename
        if not mesh_path.exists():
            raise FileNotFoundError(f"Missing mesh referenced by URDF: {mesh_path}")


def fmt_values(values) -> str:
    return " ".join(fmt_float(float(value)) for value in values)


def fmt_float(value: float) -> str:
    if abs(value) < 1e-12:
        value = 0.0
    return f"{value:.10g}"


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
