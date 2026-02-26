"""Procedural articulated object generator.

Design goals:
- Hand-scale objects with constrained, realistic size ranges.
- Geometry built from collider-friendly primitives (box/capsule/cylinder/sphere).
- Reproducible parameter sampling from a single RNG per object.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Tuple
import os
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


AXIS_TO_VECTOR: Dict[str, np.ndarray] = {
    "x": np.array([1.0, 0.0, 0.0]),
    "y": np.array([0.0, 1.0, 0.0]),
    "z": np.array([0.0, 0.0, 1.0]),
}


def _jitter(rng: np.random.Generator, nominal: float, rel: float) -> float:
    """Sample around a nominal value with relative jitter."""
    return nominal * rng.uniform(1.0 - rel, 1.0 + rel)


@dataclass
class ArticulatedObject:
    name: str
    base_mesh: trimesh.Trimesh
    child_mesh: trimesh.Trimesh
    joint_type: str
    joint_origin: np.ndarray
    joint_axis: np.ndarray
    joint_limits: Tuple[float, float]
    joint_default: float
    goal_joint_value: float | None = None

    def __post_init__(self):
        self.joint_origin = np.asarray(self.joint_origin, dtype=float)
        axis = np.asarray(self.joint_axis, dtype=float)
        norm = np.linalg.norm(axis)
        if norm <= 1e-9:
            raise ValueError(f"{self.name}: joint axis has near-zero norm")
        self.joint_axis = axis / norm
        lower, upper = self.joint_limits
        if lower > upper:
            raise ValueError(f"{self.name}: invalid joint limits {self.joint_limits}")
        if not (lower <= self.joint_default <= upper):
            raise ValueError(
                f"{self.name}: joint default {self.joint_default:.4f} outside limits {self.joint_limits}"
            )
        if self.goal_joint_value is not None and not (lower <= self.goal_joint_value <= upper):
            raise ValueError(
                f"{self.name}: goal joint value {self.goal_joint_value:.4f} outside limits {self.joint_limits}"
            )

    def get_child_transform(self, joint_value: float) -> np.ndarray:
        T = np.eye(4)
        if self.joint_type == "revolute":
            R = Rotation.from_rotvec(joint_value * self.joint_axis).as_matrix()
            T[:3, :3] = R
            T[:3, 3] = self.joint_origin - R @ self.joint_origin
        elif self.joint_type == "prismatic":
            T[:3, 3] = joint_value * self.joint_axis
        return T

    def get_meshes_at(self, joint_value: float):
        child = self.child_mesh.copy()
        child.apply_transform(self.get_child_transform(joint_value))
        return self.base_mesh.copy(), child


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def _box(extents, center):
    m = trimesh.creation.box(extents=extents)
    m.apply_translation(center)
    return m

def _cylinder(radius, height, center, axis="z"):
    m = trimesh.creation.cylinder(radius=radius, height=height)
    if axis == "x":
        m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    elif axis == "y":
        m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    m.apply_translation(center)
    return m

def _capsule(radius, length, center, axis="x"):
    m = trimesh.creation.capsule(height=length, radius=radius)
    if axis == "x":
        m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    elif axis == "y":
        m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    m.apply_translation(center)
    return m

def _sphere(radius, center):
    m = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    m.apply_translation(center)
    return m

def _merge(meshes):
    return trimesh.util.concatenate(meshes)


def _capsule_between(p0, p1, radius):
    """Create a capsule from point p0 to p1."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    vec = p1 - p0
    length = np.linalg.norm(vec)
    if length <= 2.0 * radius + 1e-6:
        return _sphere(radius, (p0 + p1) / 2.0)

    mesh = trimesh.creation.capsule(height=length - 2.0 * radius, radius=radius)
    align = trimesh.geometry.align_vectors(np.array([0.0, 0.0, 1.0]), vec / length)
    if align is not None:
        mesh.apply_transform(align)
    mesh.apply_translation((p0 + p1) / 2.0)
    return mesh


def _capsule_ring(radius, tube_radius, center, axis="z", stretch=1.0, segments=14):
    """Ring approximated by tangent capsules (collider-friendly primitive composition)."""
    points = []
    for i in range(segments):
        t = 2.0 * np.pi * i / segments
        points.append(np.array([radius * np.cos(t), stretch * radius * np.sin(t), 0.0]))

    parts = []
    for i in range(segments):
        parts.append(_capsule_between(points[i], points[(i + 1) % segments], tube_radius))
    ring = _merge(parts)

    if axis not in AXIS_TO_VECTOR:
        raise ValueError(f"Unsupported axis '{axis}'")
    align = trimesh.geometry.align_vectors(np.array([0.0, 0.0, 1.0]), AXIS_TO_VECTOR[axis])
    if align is not None:
        ring.apply_transform(align)
    ring.apply_translation(center)
    return ring


_URDF_PATH = os.path.join(
    os.path.dirname(__file__), "..", "IsaacLab", "source", "isaaclab_tasks",
    "isaaclab_tasks", "manager_based", "manipulation", "y2r", "assets",
    "ur5e_leap", "ur5e_leap_right_gemini305.urdf",
)

# LEAP hand links to include (palm + all finger links, no arm/camera)
_HAND_LINKS = {
    "palm_link",
    "index_link_0", "index_link_1", "index_link_2", "index_link_3",
    "middle_link_0", "middle_link_1", "middle_link_2", "middle_link_3",
    "ring_link_0", "ring_link_1", "ring_link_2", "ring_link_3",
    "thumb_link_0", "thumb_link_1", "thumb_link_2", "thumb_link_3",
}


def _parse_origin(elem):
    """Parse a URDF <origin> element into a 4x4 transform."""
    T = np.eye(4)
    if elem is None:
        return T
    xyz = elem.get("xyz", "0 0 0")
    rpy = elem.get("rpy", "0 0 0")
    T[:3, 3] = [float(v) for v in xyz.split()]
    r, p, y = [float(v) for v in rpy.split()]
    # URDF uses extrinsic XYZ (sxyz) = intrinsic ZYX
    T[:3, :3] = Rotation.from_euler("xyz", [r, p, y]).as_matrix()
    return T


def make_hand_reference(joint_values=None):
    """Load LEAP hand from URDF with meshes assembled via forward kinematics.

    Args:
        joint_values: dict mapping joint_name -> angle (rad). Defaults to 0 for all.
    """
    if not os.path.isfile(_URDF_PATH):
        # Fallback: simple schematic if URDF not available
        return _make_schematic_hand()

    urdf_dir = os.path.dirname(os.path.abspath(_URDF_PATH))
    tree = ET.parse(_URDF_PATH)
    root = tree.getroot()
    if joint_values is None:
        joint_values = {}

    # Parse links: name -> list of (visual_origin_T, mesh_filename)
    link_visuals = {}
    for link_elem in root.findall("link"):
        name = link_elem.get("name")
        if name not in _HAND_LINKS:
            continue
        visuals = []
        for vis in link_elem.findall("visual"):
            origin_T = _parse_origin(vis.find("origin"))
            geom = vis.find("geometry")
            if geom is None:
                continue
            mesh_elem = geom.find("mesh")
            if mesh_elem is not None:
                fname = mesh_elem.get("filename")
                scale_str = mesh_elem.get("scale")
                scale = np.ones(3)
                if scale_str:
                    scale = np.array([float(v) for v in scale_str.split()])
                visuals.append((origin_T, fname, scale))
        link_visuals[name] = visuals

    # Parse joints: build parent->child tree with transforms
    joints = []
    for joint_elem in root.findall("joint"):
        jtype = joint_elem.get("type")
        parent = joint_elem.find("parent").get("link")
        child = joint_elem.find("child").get("link")
        origin_T = _parse_origin(joint_elem.find("origin"))
        axis = np.array([0, 0, 1], dtype=float)
        axis_elem = joint_elem.find("axis")
        if axis_elem is not None:
            axis = np.array([float(v) for v in axis_elem.get("xyz").split()])
        jname = joint_elem.get("name")
        joints.append((jname, jtype, parent, child, origin_T, axis))

    # Build world transforms via BFS from palm_link
    world_T = {"palm_link": np.eye(4)}
    # Multiple passes to resolve all children
    resolved = {"palm_link"}
    for _ in range(10):
        for jname, jtype, parent, child, origin_T, axis in joints:
            if parent in resolved and child not in resolved and child in _HAND_LINKS:
                T_parent = world_T[parent]
                T_joint = origin_T.copy()
                if jtype == "revolute":
                    q = joint_values.get(jname, 0.0)
                    R_q = np.eye(4)
                    R_q[:3, :3] = Rotation.from_rotvec(q * axis).as_matrix()
                    T_joint = T_joint @ R_q
                world_T[child] = T_parent @ T_joint
                resolved.add(child)

    # Load and transform meshes
    meshes = []
    for link_name, visuals in link_visuals.items():
        if link_name not in world_T:
            continue
        T_link = world_T[link_name]
        for vis_origin_T, fname, scale in visuals:
            fpath = os.path.join(urdf_dir, fname)
            if not os.path.isfile(fpath):
                continue
            try:
                m = trimesh.load(fpath, force="mesh")
                if not hasattr(m, "vertices") or m.vertices.shape[0] == 0:
                    continue
                m.vertices *= scale
                T_full = T_link @ vis_origin_T
                m.apply_transform(T_full)
                meshes.append(m)
            except Exception:
                continue

    if not meshes:
        return _make_schematic_hand()

    hand = trimesh.util.concatenate(meshes)
    hand.apply_translation(-hand.centroid)
    return hand


def _make_schematic_hand():
    """Fallback schematic hand from primitives."""
    parts = []
    palm_w, palm_d, palm_t = 0.090, 0.055, 0.025
    parts.append(_box([palm_w, palm_d, palm_t], [0, 0, 0]))
    finger_r = 0.009
    finger_spacing = palm_w / 5
    for i in range(4):
        fx = -palm_w / 2 + finger_spacing * (i + 1)
        seg_lens = [0.035, 0.030, 0.025]
        y_offset = palm_d / 2
        for seg_len in seg_lens:
            parts.append(_capsule(finger_r, seg_len - 2 * finger_r,
                                  [fx, y_offset + seg_len / 2, 0], axis="y"))
            y_offset += seg_len
    thumb_r = 0.010
    thumb_segs = [0.030, 0.025, 0.020]
    tx, ty = -palm_w / 2 - 0.010, -palm_d / 4
    for seg_len in thumb_segs:
        parts.append(_capsule(thumb_r, seg_len - 2 * thumb_r,
                              [tx, ty + seg_len / 2, 0], axis="y"))
        ty += seg_len
        tx -= 0.005
    hand = _merge(parts)
    hand.apply_translation(-hand.centroid)
    return hand


# ---------------------------------------------------------------------------
# Templates — all sized for LEAP hand interaction
# ---------------------------------------------------------------------------


def generate_scissors(rng=None):
    """Scissors with finger holes large enough for robotic fingers (~18mm dia).

    Diversity: handle shape (round ring vs oblong), blade proportions,
    asymmetric handle sizes, blade-to-handle ratio.
    """
    rng = rng or np.random.default_rng()

    style = rng.choice(["compact", "standard", "long"])
    if style == "compact":
        blade_len = _jitter(rng, 0.095, 0.10)
        handle_len = _jitter(rng, 0.055, 0.10)
    elif style == "long":
        blade_len = _jitter(rng, 0.135, 0.08)
        handle_len = _jitter(rng, 0.075, 0.10)
    else:
        blade_len = _jitter(rng, 0.115, 0.08)
        handle_len = _jitter(rng, 0.065, 0.10)

    blade_w = blade_len * rng.uniform(0.10, 0.14)
    blade_t = rng.uniform(0.0035, 0.0055)
    z_gap = blade_t * rng.uniform(0.7, 1.1)

    handle_shape = rng.choice(["round", "oblong"])
    loop_inner_r = rng.uniform(0.012, 0.016)
    loop_tube_r = rng.uniform(0.0035, 0.0050)
    loop_r = loop_inner_r + loop_tube_r
    oblong_stretch = rng.uniform(1.45, 1.9) if handle_shape == "oblong" else 1.0

    eff_r = (loop_r + loop_tube_r) * max(oblong_stretch, 1.0)
    handle_y_offset = eff_r + rng.uniform(0.003, 0.008)

    child_loop_r = loop_r * rng.uniform(0.9, 1.1)
    child_loop_tube_r = loop_tube_r * rng.uniform(0.9, 1.1)

    def _make_handle(r_mid, r_tube, y_off, z):
        ring = _capsule_ring(
            radius=r_mid,
            tube_radius=r_tube,
            center=[-handle_len / 2, y_off, z],
            axis="z",
            stretch=oblong_stretch,
            segments=14,
        )
        conn = _box(
            [handle_len * 0.35, abs(y_off) * 0.6, blade_t],
            [-handle_len * 0.17, y_off * 0.45, z],
        )
        return [ring, conn]

    blade_style = rng.choice(["straight", "tapered"])
    if blade_style == "straight":
        base_blade = _box([blade_len, blade_w, blade_t], [blade_len / 2, 0, -z_gap])
        child_blade = _box([blade_len, blade_w, blade_t], [blade_len / 2, 0, z_gap])
    else:
        root_len = blade_len * rng.uniform(0.50, 0.62)
        tip_len = blade_len - root_len
        root_w = blade_w * rng.uniform(1.00, 1.15)
        tip_w = blade_w * rng.uniform(0.45, 0.70)
        base_blade = _merge([
            _box([root_len, root_w, blade_t], [root_len / 2, 0, -z_gap]),
            _box([tip_len, tip_w, blade_t], [root_len + tip_len / 2, 0, -z_gap]),
        ])
        child_blade = _merge([
            _box([root_len, root_w, blade_t], [root_len / 2, 0, z_gap]),
            _box([tip_len, tip_w, blade_t], [root_len + tip_len / 2, 0, z_gap]),
        ])

    # Base arm
    base_parts = _make_handle(loop_r, loop_tube_r, -handle_y_offset, -z_gap)
    pivot = _cylinder(blade_t * 2, z_gap * 4, [0, 0, 0])
    base_mesh = _merge([base_blade, pivot] + base_parts)

    # Child arm
    child_parts = _make_handle(child_loop_r, child_loop_tube_r, handle_y_offset, z_gap)
    child_mesh = _merge([child_blade] + child_parts)

    open_angle = np.radians(rng.uniform(20, 45))

    return ArticulatedObject(
        name="scissors",
        base_mesh=base_mesh, child_mesh=child_mesh,
        joint_type="revolute",
        joint_origin=np.array([0.0, 0.0, 0.0]),
        joint_axis=np.array([0.0, 0.0, -1.0]),
        joint_limits=(0.0, open_angle),
        joint_default=open_angle,
    )


def generate_book(rng=None):
    """Book / notebook for single-hand manipulation.

    Diversity: aspect ratio (portrait/landscape/square), thickness,
    spine style (flat/rounded), grasp feature on open edge (tab/lip/ribbon/none).
    """
    rng = rng or np.random.default_rng()

    aspect = rng.choice(["portrait", "landscape", "square"])
    if aspect == "portrait":
        cover_w = rng.uniform(0.145, 0.200)
        cover_h = cover_w * rng.uniform(1.30, 1.55)
    elif aspect == "landscape":
        cover_h = rng.uniform(0.135, 0.185)
        cover_w = cover_h * rng.uniform(1.35, 1.60)
    else:
        s = rng.uniform(0.145, 0.185)
        cover_w = s * rng.uniform(0.9, 1.1)
        cover_h = s * rng.uniform(0.9, 1.1)

    cover_t = rng.uniform(0.004, 0.007)
    pages_t = rng.uniform(0.012, 0.030)
    total_t = cover_t * 2 + pages_t

    back_cover = _box([cover_w, cover_h, cover_t], [cover_w / 2, 0, cover_t / 2])
    pages = _box(
        [cover_w * 0.95, cover_h * 0.95, pages_t],
        [cover_w / 2, 0, cover_t + pages_t / 2],
    )

    # Spine style
    spine_style = rng.choice(["flat", "rounded", "ribbed"])
    base_parts = [back_cover, pages]
    if spine_style == "flat":
        base_parts.append(_box(
            [cover_t, cover_h, total_t],
            [-cover_t / 2, 0, total_t / 2],
        ))
    elif spine_style == "rounded":
        spine_r = total_t / 2 * rng.uniform(0.6, 0.9)
        base_parts.append(_cylinder(
            spine_r, cover_h, [0, 0, total_t / 2], axis="y",
        ))
    else:
        base_parts.append(_box(
            [cover_t, cover_h, total_t],
            [-cover_t / 2, 0, total_t / 2],
        ))
        rib_count = int(rng.integers(2, 5))
        rib_r = cover_t * rng.uniform(0.3, 0.6)
        for i in range(rib_count):
            ry = cover_h * ((i + 1) / (rib_count + 1) - 0.5)
            base_parts.append(_cylinder(
                rib_r, total_t * 0.8,
                [-cover_t, ry, total_t / 2], axis="z",
            ))

    base_mesh = _merge(base_parts)

    hinge_z = cover_t + pages_t
    front_cover = _box([cover_w, cover_h, cover_t], [cover_w / 2, 0, cover_t / 2])

    # Grasp feature on the opening edge for fingers to hook under
    grasp_style = rng.choice(["tab", "full_lip", "ribbon", "none"])
    child_parts = [front_cover]
    if grasp_style == "tab":
        tab_w = rng.uniform(0.012, 0.022)
        tab_h = cover_h * rng.uniform(0.15, 0.30)
        tab_t = cover_t * rng.uniform(2.5, 5.0)
        child_parts.append(_box(
            [tab_w, tab_h, tab_t],
            [cover_w + tab_w / 2, 0, cover_t / 2],
        ))
    elif grasp_style == "full_lip":
        lip_w = rng.uniform(0.005, 0.010)
        lip_t = cover_t * rng.uniform(2.0, 3.5)
        child_parts.append(_box(
            [lip_w, cover_h * 0.85, lip_t],
            [cover_w + lip_w / 2, 0, cover_t / 2],
        ))
    elif grasp_style == "ribbon":
        ribbon_w = rng.uniform(0.004, 0.008)
        ribbon_hang = rng.uniform(0.015, 0.035)
        ry = cover_h * rng.uniform(-0.3, 0.3)
        child_parts.append(_box(
            [ribbon_w, ribbon_w, pages_t + ribbon_hang],
            [cover_w * 0.5, ry, -(ribbon_hang / 2)],
        ))

    child_mesh = _merge(child_parts)
    child_mesh.apply_translation([0, 0, hinge_z])

    return ArticulatedObject(
        name="book",
        base_mesh=base_mesh, child_mesh=child_mesh,
        joint_type="revolute",
        joint_origin=np.array([0.0, 0.0, hinge_z]),
        joint_axis=np.array([0.0, -1.0, 0.0]),
        joint_limits=(0.0, np.radians(170)),
        joint_default=0.0,
    )


def generate_laptop(rng=None):
    """Laptop large enough for hand interaction.

    Diversity: profile, hinge style, front lip geometry for opening.
    """
    rng = rng or np.random.default_rng()

    profile = rng.choice(["13in", "15in", "ultrabook"])
    if profile == "13in":
        width = rng.uniform(0.220, 0.250)
        depth = width * rng.uniform(0.62, 0.72)
    elif profile == "15in":
        width = rng.uniform(0.255, 0.300)
        depth = width * rng.uniform(0.60, 0.70)
    else:
        width = rng.uniform(0.230, 0.270)
        depth = width * rng.uniform(0.54, 0.63)

    base_t = rng.uniform(0.009, 0.013)
    screen_t = rng.uniform(0.005, 0.008)

    base_body = _box([width, depth, base_t], [0, 0, base_t / 2])
    hinge_r = min(base_t, screen_t) * rng.uniform(0.4, 0.6)
    hinge_style = rng.choice(["bar", "dual_pivot"])
    if hinge_style == "bar":
        hinge_parts = [_cylinder(hinge_r, width * 0.95, [0, depth / 2, base_t], axis="x")]
    else:
        span = width * rng.uniform(0.50, 0.65)
        x_off = span / 2
        hinge_len = width * rng.uniform(0.14, 0.20)
        hinge_parts = [
            _cylinder(hinge_r * 1.1, hinge_len, [-x_off, depth / 2, base_t], axis="x"),
            _cylinder(hinge_r * 1.1, hinge_len, [x_off, depth / 2, base_t], axis="x"),
        ]
    base_mesh = _merge([base_body] + hinge_parts)

    hinge_y = depth / 2
    hinge_z = base_t

    screen = _box([width, depth, screen_t], [0, -depth / 2, screen_t / 2])
    # Front grasp lip for opening (protrudes outward in -Y, not upward in +Z).
    lip_style = rng.choice(["center_tab", "full_lip", "dual_tabs", "offset_tab"])
    lip_protrude = rng.uniform(0.006, 0.014)
    lip_height = screen_t * rng.uniform(0.45, 0.95)
    lip_z = lip_height / 2 + screen_t * rng.uniform(0.00, 0.08)
    lip_y = -depth - lip_protrude / 2
    grasp_parts = []
    if lip_style == "center_tab":
        lip_w = width * rng.uniform(0.18, 0.32)
        grasp_parts.append(_box([lip_w, lip_protrude, lip_height], [0.0, lip_y, lip_z]))
    elif lip_style == "full_lip":
        lip_w = width * rng.uniform(0.70, 0.92)
        grasp_parts.append(_box([lip_w, lip_protrude, lip_height], [0.0, lip_y, lip_z]))
    elif lip_style == "dual_tabs":
        tab_w = width * rng.uniform(0.10, 0.18)
        tab_x = width * rng.uniform(0.22, 0.34)
        grasp_parts.append(_box([tab_w, lip_protrude, lip_height], [-tab_x, lip_y, lip_z]))
        grasp_parts.append(_box([tab_w, lip_protrude, lip_height], [tab_x, lip_y, lip_z]))
    else:
        side = -1.0 if rng.random() < 0.5 else 1.0
        lip_w = width * rng.uniform(0.16, 0.28)
        lip_x = side * width * rng.uniform(0.20, 0.34)
        grasp_parts.append(_box([lip_w, lip_protrude, lip_height], [lip_x, lip_y, lip_z]))

    child_mesh = _merge([screen] + grasp_parts)
    child_mesh.apply_translation([0, hinge_y, hinge_z])

    return ArticulatedObject(
        name="laptop",
        base_mesh=base_mesh, child_mesh=child_mesh,
        joint_type="revolute",
        joint_origin=np.array([0.0, hinge_y, hinge_z]),
        joint_axis=np.array([-1.0, 0.0, 0.0]),
        joint_limits=(0.0, np.radians(135)),
        joint_default=0.0,
    )


def generate_tongs(rng=None):
    """Tongs for squeezing/gripping.

    Diversity: arm cross-section (flat/round/capsule), tip shape (paddle/disc/bulb),
    arm profile (straight/curved), hinge bridge (box/cylinder/spring-coil),
    optional grip bumps on handle section.
    """
    rng = rng or np.random.default_rng()

    arm_len = rng.uniform(0.165, 0.220)
    arm_shape = rng.choice(["flat", "round", "capsule"])
    tip_shape = rng.choice(["paddle", "disc", "bulb"])
    arm_profile = rng.choice(["straight", "curved"])

    if arm_shape == "flat":
        arm_w = rng.uniform(0.011, 0.017)
        arm_t = rng.uniform(0.005, 0.008)
        arm_r = arm_t / 2
    elif arm_shape == "round":
        arm_r = rng.uniform(0.0055, 0.0085)
        arm_w = arm_r * 2
        arm_t = arm_r * 2
    else:
        arm_r = rng.uniform(0.0050, 0.0075)
        arm_w = arm_r * 2
        arm_t = arm_r * 2

    z_gap = max(arm_t * 0.6, 0.004)
    tip_len = rng.uniform(0.022, 0.035)
    tip_w = rng.uniform(0.024, 0.036)

    def _arm_segment(x0, x1, z):
        cx = (x0 + x1) / 2
        seg_len = abs(x1 - x0)
        if arm_shape == "flat":
            return _box([seg_len, arm_w, arm_t], [cx, 0, z])
        elif arm_shape == "round":
            return _cylinder(arm_r, seg_len, [cx, 0, z], axis="x")
        else:
            return _capsule(arm_r, max(0.001, seg_len - 2 * arm_r), [cx, 0, z], axis="x")

    def _arm(z):
        if arm_profile == "straight":
            return _arm_segment(-arm_len, 0, z)
        else:
            # Two segments with a kink — slight outward bend at midpoint
            bend_x = -arm_len * rng.uniform(0.40, 0.60)
            bend_z_off = z_gap * rng.uniform(0.3, 0.7) * np.sign(z)
            seg1 = _arm_segment(-arm_len, bend_x, z)
            seg2 = _arm_segment(bend_x, 0, z + bend_z_off)
            return _merge([seg1, seg2])

    def _tip(z):
        x = -arm_len + tip_len / 2
        if tip_shape == "paddle":
            return _box([tip_len, tip_w, arm_t * rng.uniform(1.0, 1.5)], [x, 0, z])
        elif tip_shape == "disc":
            return _cylinder(tip_w / 2, arm_t * 1.2, [x, 0, z])
        else:
            return _sphere(min(tip_len, tip_w) * 0.32, [x, 0, z])

    # Hinge / bridge piece connecting the two arms
    bridge_style = rng.choice(["box", "cylinder", "coil"])
    bridge_parts = []
    if bridge_style == "box":
        bridge_parts.append(_box(
            [arm_w * 1.3, arm_w * 1.3, z_gap * 2 + arm_t],
            [0, 0, 0],
        ))
    elif bridge_style == "cylinder":
        bridge_parts.append(_cylinder(
            arm_w * 0.7, z_gap * 2 + arm_t,
            [0, 0, 0], axis="z",
        ))
    else:
        # Spring coil: stack of small offset cylinders
        coil_turns = int(rng.integers(3, 7))
        coil_r = arm_w * rng.uniform(0.5, 0.8)
        coil_h = (z_gap * 2 + arm_t) / coil_turns
        for i in range(coil_turns):
            cz = -z_gap - arm_t / 2 + (i + 0.5) * coil_h
            bridge_parts.append(_cylinder(coil_r, coil_h * 0.85, [0, 0, cz]))

    # Optional grip bumps on handle section
    grip_parts = []
    if rng.random() < 0.45:
        bump_count = int(rng.integers(2, 5))
        bump_r = arm_w * rng.uniform(0.25, 0.45)
        for i in range(bump_count):
            bx = -arm_len * rng.uniform(0.05, 0.30)
            bx_i = bx - (arm_len * 0.05 * i)
            for sign in [-1, 1]:
                grip_parts.append(_sphere(bump_r, [bx_i, 0, sign * z_gap]))

    base_mesh = _merge([_arm(-z_gap), _tip(-z_gap)] + bridge_parts + grip_parts)
    child_mesh = _merge([_arm(z_gap), _tip(z_gap)])

    open_angle = np.radians(rng.uniform(16, 30))

    return ArticulatedObject(
        name="tongs",
        base_mesh=base_mesh, child_mesh=child_mesh,
        joint_type="revolute",
        joint_origin=np.array([0.0, 0.0, 0.0]),
        joint_axis=np.array([0.0, 1.0, 0.0]),
        joint_limits=(0.0, open_angle),
        joint_default=open_angle,
    )


def generate_box_lid(rng=None):
    """Box with hinged lid.

    Diversity: proportions (wide/deep/square/tall),
    grasp tab style (center_tab/full_lip/dual_tabs/capsule_handle),
    optional corner feet, optional front clasp, lid overhang.
    """
    rng = rng or np.random.default_rng()

    shape = rng.choice(["wide", "deep", "square", "tall"])
    if shape == "wide":
        box_w = rng.uniform(0.14, 0.20)
        box_d = box_w * rng.uniform(0.58, 0.75)
    elif shape == "deep":
        box_d = rng.uniform(0.13, 0.20)
        box_w = box_d * rng.uniform(0.58, 0.75)
    elif shape == "tall":
        box_w = rng.uniform(0.11, 0.16)
        box_d = rng.uniform(0.11, 0.16)
    else:
        s = rng.uniform(0.12, 0.18)
        box_w = s
        box_d = s

    box_h = rng.uniform(0.05, 0.085) if shape != "tall" else rng.uniform(0.08, 0.12)
    wall_t = rng.uniform(0.004, 0.0065)
    lid_t = rng.uniform(0.004, 0.0065)

    bottom = _box([box_w, box_d, wall_t], [0, 0, wall_t / 2])
    wf = _box([box_w, wall_t, box_h], [0, -box_d / 2 + wall_t / 2, wall_t + box_h / 2])
    wb = _box([box_w, wall_t, box_h], [0, box_d / 2 - wall_t / 2, wall_t + box_h / 2])
    wl = _box([wall_t, box_d, box_h], [-box_w / 2 + wall_t / 2, 0, wall_t + box_h / 2])
    wr = _box([wall_t, box_d, box_h], [box_w / 2 - wall_t / 2, 0, wall_t + box_h / 2])
    box_parts = [bottom, wf, wb, wl, wr]

    # Optional corner feet
    if rng.random() < 0.40:
        foot_r = wall_t * rng.uniform(0.8, 1.5)
        foot_h = rng.uniform(0.003, 0.008)
        for sx in [-1, 1]:
            for sy in [-1, 1]:
                fx = sx * (box_w / 2 - wall_t)
                fy = sy * (box_d / 2 - wall_t)
                box_parts.append(_cylinder(foot_r, foot_h, [fx, fy, -foot_h / 2]))

    # Optional front clasp on box body (decorative latch nub)
    if rng.random() < 0.35:
        clasp_w = box_w * rng.uniform(0.06, 0.12)
        clasp_h = box_h * rng.uniform(0.12, 0.22)
        clasp_d = rng.uniform(0.003, 0.006)
        clasp_z = wall_t + box_h - clasp_h / 2
        box_parts.append(_box(
            [clasp_w, clasp_d, clasp_h],
            [0, -box_d / 2 - clasp_d / 2, clasp_z],
        ))

    base_mesh = _merge(box_parts)

    hinge_y = box_d / 2
    hinge_z = wall_t + box_h

    # Lid with optional overhang
    lid_overhang = rng.uniform(0.0, 0.008)
    lid_w = box_w + lid_overhang * 2
    lid_d = box_d + lid_overhang
    lid = _box([lid_w, lid_d, lid_t], [0, -lid_d / 2 + lid_overhang, lid_t / 2])

    # Grasp tab
    tab_style = rng.choice(["center_tab", "full_lip", "dual_tabs", "capsule_handle"])
    tab_protrude = rng.uniform(0.010, 0.016)
    tab_height = lid_t * rng.uniform(2.5, 5.0)
    front_edge_y = -lid_d + lid_overhang
    lid_parts = [lid]
    if tab_style == "center_tab":
        tab_w = box_w * rng.uniform(0.25, 0.40)
        lid_parts.append(_box(
            [tab_w, tab_protrude, tab_height],
            [0, front_edge_y - tab_protrude / 2, tab_height / 2],
        ))
    elif tab_style == "full_lip":
        lid_parts.append(_box(
            [lid_w * 0.9, tab_protrude, tab_height],
            [0, front_edge_y - tab_protrude / 2, tab_height / 2],
        ))
    elif tab_style == "dual_tabs":
        tab_w = box_w * rng.uniform(0.12, 0.20)
        tab_x = box_w * rng.uniform(0.25, 0.32)
        lid_parts.append(_box(
            [tab_w, tab_protrude, tab_height],
            [-tab_x, front_edge_y - tab_protrude / 2, tab_height / 2],
        ))
        lid_parts.append(_box(
            [tab_w, tab_protrude, tab_height],
            [tab_x, front_edge_y - tab_protrude / 2, tab_height / 2],
        ))
    else:
        handle_r = rng.uniform(0.003, 0.005)
        handle_w = box_w * rng.uniform(0.25, 0.45)
        standoff = rng.uniform(0.008, 0.014)
        hy = front_edge_y - standoff
        lid_parts.append(_capsule(
            handle_r, handle_w - 2 * handle_r,
            [0, hy, lid_t / 2], axis="x",
        ))
        lid_parts.append(_cylinder(
            handle_r, standoff,
            [-handle_w / 2, front_edge_y - standoff / 2, lid_t / 2], axis="y",
        ))
        lid_parts.append(_cylinder(
            handle_r, standoff,
            [handle_w / 2, front_edge_y - standoff / 2, lid_t / 2], axis="y",
        ))

    child_mesh = _merge(lid_parts)
    child_mesh.apply_translation([0, hinge_y, hinge_z])

    return ArticulatedObject(
        name="box_lid",
        base_mesh=base_mesh, child_mesh=child_mesh,
        joint_type="revolute",
        joint_origin=np.array([0.0, hinge_y, hinge_z]),
        joint_axis=np.array([-1.0, 0.0, 0.0]),
        joint_limits=(0.0, np.radians(120)),
        joint_default=0.0,
    )


def generate_drawer(rng=None):
    """Drawer large enough for hand interaction.

    Diversity: proportions, handle style (bar/groove/knob).
    """
    rng = rng or np.random.default_rng()

    prop = rng.choice(["wide", "deep", "cubic"])
    if prop == "wide":
        inner_w = rng.uniform(0.15, 0.22)
        inner_d = inner_w * rng.uniform(0.60, 0.78)
        inner_h = rng.uniform(0.05, 0.08)
    elif prop == "deep":
        inner_d = rng.uniform(0.13, 0.19)
        inner_w = inner_d * rng.uniform(0.62, 0.80)
        inner_h = rng.uniform(0.05, 0.08)
    else:
        s = rng.uniform(0.12, 0.17)
        inner_w = s * rng.uniform(0.9, 1.1)
        inner_d = s * rng.uniform(0.9, 1.1)
        inner_h = s * rng.uniform(0.45, 0.70)

    wall_t = rng.uniform(0.0045, 0.0065)
    outer_w = inner_w + 2 * wall_t
    outer_d = inner_d + 2 * wall_t
    outer_h = inner_h + 2 * wall_t

    bottom = _box([outer_w, outer_d, wall_t], [0, 0, wall_t / 2])
    top = _box([outer_w, outer_d, wall_t], [0, 0, outer_h - wall_t / 2])
    back = _box([outer_w, wall_t, outer_h], [0, outer_d / 2 - wall_t / 2, outer_h / 2])
    left = _box([wall_t, outer_d, outer_h], [-outer_w / 2 + wall_t / 2, 0, outer_h / 2])
    right = _box([wall_t, outer_d, outer_h], [outer_w / 2 - wall_t / 2, 0, outer_h / 2])
    base_mesh = _merge([bottom, top, back, left, right])

    front_y = -outer_d / 2 + wall_t / 2
    d_parts = [
        _box([inner_w, inner_d, wall_t], [0, 0, wall_t + wall_t / 2]),
        _box([outer_w, wall_t, outer_h], [0, front_y, outer_h / 2]),
        _box([wall_t, inner_d, inner_h], [-inner_w / 2 + wall_t / 2, 0, wall_t + inner_h / 2]),
        _box([wall_t, inner_d, inner_h], [inner_w / 2 - wall_t / 2, 0, wall_t + inner_h / 2]),
        _box([inner_w, wall_t, inner_h], [0, inner_d / 2 - wall_t / 2, wall_t + inner_h / 2]),
    ]

    handle_style = rng.choice(["bar", "knob", "dual_knob", "ring_pull"])
    handle_z = outer_h * rng.uniform(0.4, 0.65)
    front_face_y = front_y - wall_t / 2

    if handle_style == "bar":
        hw = outer_w * rng.uniform(0.45, 0.75)
        hr = rng.uniform(0.005, 0.008)
        standoff = rng.uniform(0.015, 0.025)
        d_parts.append(_capsule(hr, hw - 2 * hr, [0, front_face_y - standoff, handle_z], axis="x"))
        d_parts.append(_cylinder(hr, standoff, [-hw / 2, front_face_y - standoff / 2, handle_z], axis="y"))
        d_parts.append(_cylinder(hr, standoff, [hw / 2, front_face_y - standoff / 2, handle_z], axis="y"))
    elif handle_style == "knob":
        kr = rng.uniform(0.010, 0.018)
        kd = rng.uniform(0.012, 0.022)
        d_parts.append(_cylinder(kr, kd, [0, front_face_y - kd / 2, handle_z], axis="y"))
    elif handle_style == "dual_knob":
        kr = rng.uniform(0.008, 0.014)
        kd = rng.uniform(0.010, 0.018)
        kx = outer_w * rng.uniform(0.18, 0.26)
        d_parts.append(_cylinder(kr, kd, [-kx, front_face_y - kd / 2, handle_z], axis="y"))
        d_parts.append(_cylinder(kr, kd, [kx, front_face_y - kd / 2, handle_z], axis="y"))
    else:
        ring_mid = outer_w * rng.uniform(0.10, 0.16)
        ring_tube = rng.uniform(0.0025, 0.0040)
        standoff = rng.uniform(0.012, 0.020)
        d_parts.append(_capsule_ring(ring_mid, ring_tube, [0, front_face_y - standoff, handle_z], axis="y", segments=12))
        lug_w = ring_tube * 2.2
        d_parts.append(_box([lug_w, standoff, lug_w], [-ring_mid * 0.7, front_face_y - standoff / 2, handle_z]))
        d_parts.append(_box([lug_w, standoff, lug_w], [ring_mid * 0.7, front_face_y - standoff / 2, handle_z]))

    child_mesh = _merge(d_parts)
    slide_dist = inner_d * 0.75

    return ArticulatedObject(
        name="drawer",
        base_mesh=base_mesh, child_mesh=child_mesh,
        joint_type="prismatic",
        joint_origin=np.array([0.0, 0.0, 0.0]),
        joint_axis=np.array([0.0, -1.0, 0.0]),
        joint_limits=(0.0, slide_dist),
        joint_default=0.0,
    )


def generate_door_handle(rng=None):
    """Door panel with randomized handle assemblies.

    Diversity:
    - left/right handed placement
    - escutcheon style (rect/round/dual-post)
    - handle style (straight/angled/capsule/knob)
    - scale + standoff variation
    """
    rng = rng or np.random.default_rng()

    # Door panel.
    door_style = rng.choice(["narrow", "regular", "wide"])
    if door_style == "narrow":
        door_w = rng.uniform(0.16, 0.20)
        door_h = door_w * rng.uniform(1.45, 1.75)
    elif door_style == "wide":
        door_w = rng.uniform(0.22, 0.28)
        door_h = door_w * rng.uniform(1.20, 1.45)
    else:
        door_w = rng.uniform(0.18, 0.24)
        door_h = door_w * rng.uniform(1.30, 1.65)
    door_t = rng.uniform(0.012, 0.018)
    door = _box([door_w, door_t, door_h], [0, 0, door_h / 2])
    door_front_y = door_t / 2

    # Left/right handed placement near edge.
    side = rng.choice([-1, 1])
    edge_margin = rng.uniform(0.025, 0.042)
    mount_x = side * (door_w / 2 - edge_margin)
    mount_z = door_h * rng.uniform(0.46, 0.68)
    lever_dir = -side  # Handle extends toward the door center.

    # Escutcheon / mount style.
    plate_style = rng.choice(["rect", "round", "dual_post"])
    plate_depth = rng.uniform(0.003, 0.007)
    plate_parts = []
    if plate_style == "rect":
        esc_w = rng.uniform(0.024, 0.040)
        esc_h = rng.uniform(0.050, 0.082)
        plate_parts.append(_box([esc_w, plate_depth, esc_h], [mount_x, door_front_y + plate_depth / 2, mount_z]))
    elif plate_style == "round":
        esc_r = rng.uniform(0.014, 0.024)
        plate_parts.append(_cylinder(esc_r, plate_depth, [mount_x, door_front_y + plate_depth / 2, mount_z], axis="y"))
    else:
        post_r = rng.uniform(0.006, 0.010)
        post_h = rng.uniform(0.048, 0.076)
        z_sep = post_h * rng.uniform(0.38, 0.52)
        plate_parts.append(_cylinder(post_r, plate_depth, [mount_x, door_front_y + plate_depth / 2, mount_z - z_sep], axis="y"))
        plate_parts.append(_cylinder(post_r, plate_depth, [mount_x, door_front_y + plate_depth / 2, mount_z + z_sep], axis="y"))
        plate_parts.append(_box([post_r * 1.4, plate_depth, z_sep * 2], [mount_x, door_front_y + plate_depth / 2, mount_z]))

    base_mesh = _merge([door] + plate_parts)

    # Handle geometry in front of plate.
    handle_style = rng.choice(["straight", "angled", "capsule", "knob"])
    lever_len = door_w * rng.uniform(0.42, 0.62)
    lever_r = rng.uniform(0.0055, 0.0105)
    shaft_r = lever_r * rng.uniform(0.8, 1.15)
    standoff = rng.uniform(0.045, 0.080)
    handle_base_y = door_front_y + plate_depth
    handle_y = handle_base_y + standoff / 2

    child_parts = [
        _cylinder(shaft_r, standoff, [mount_x, handle_base_y + standoff / 2, mount_z], axis="y")
    ]

    if handle_style == "straight":
        child_parts.append(_cylinder(lever_r, lever_len, [mount_x + lever_dir * lever_len / 2, handle_y, mount_z], axis="x"))
        child_parts.append(_sphere(lever_r * 1.5, [mount_x + lever_dir * lever_len, handle_y, mount_z]))
    elif handle_style == "angled":
        h_len = lever_len * rng.uniform(0.55, 0.78)
        v_len = lever_len * rng.uniform(0.18, 0.34)
        child_parts.append(_cylinder(lever_r, h_len, [mount_x + lever_dir * h_len / 2, handle_y, mount_z], axis="x"))
        child_parts.append(_cylinder(lever_r, v_len, [mount_x + lever_dir * h_len, handle_y, mount_z - v_len / 2], axis="z"))
        child_parts.append(_sphere(lever_r * 1.5, [mount_x + lever_dir * h_len, handle_y, mount_z - v_len]))
    elif handle_style == "capsule":
        child_parts.append(_capsule(lever_r, lever_len - 2 * lever_r, [mount_x + lever_dir * lever_len / 2, handle_y, mount_z], axis="x"))
    else:
        stem_len = standoff * rng.uniform(0.55, 0.85)
        knob_r = rng.uniform(0.011, 0.020)
        child_parts.append(_cylinder(shaft_r * 0.9, stem_len, [mount_x, handle_base_y + stem_len / 2, mount_z], axis="y"))
        child_parts.append(_sphere(knob_r, [mount_x, handle_base_y + stem_len + knob_r * 0.7, mount_z]))

    child_mesh = _merge(child_parts)

    # Make positive motion always push lever downward regardless handedness.
    # If the handle points in +X or -X, choosing axis sign = lever_dir yields dz/dtheta < 0.
    joint_axis_y = float(lever_dir)
    max_press = np.radians(rng.uniform(45, 72))

    return ArticulatedObject(
        name="door_handle",
        base_mesh=base_mesh,
        child_mesh=child_mesh,
        joint_type="revolute",
        joint_origin=np.array([mount_x, handle_y, mount_z]),
        joint_axis=np.array([0.0, joint_axis_y, 0.0]),
        joint_limits=(0.0, max_press),
        joint_default=0.0,
    )


def generate_bottle_cap(rng=None):
    """Bottle + cap object.

    Task intent: open the cap by rotating it around the bottle neck axis.
    """
    rng = rng or np.random.default_rng()

    # Smaller, hand-scale bottle classes.
    size_class = rng.choice(["mini", "small", "medium"])
    if size_class == "mini":
        base_r = rng.uniform(0.016, 0.021)
        body_h = rng.uniform(0.070, 0.102)
    elif size_class == "medium":
        base_r = rng.uniform(0.021, 0.028)
        body_h = rng.uniform(0.105, 0.145)
    else:
        base_r = rng.uniform(0.018, 0.025)
        body_h = rng.uniform(0.085, 0.125)

    body_style = rng.choice(["straight", "soda", "sport", "flask", "squareish"])
    z_overlap = 0.0015
    body_parts = []
    profile_spans = []

    def _add_taper(z0: float, z1: float, r0: float, r1: float, steps_min: int = 2, steps_max: int = 4):
        """Approximate a taper with stacked cylinders and slight overlap."""
        n = int(rng.integers(steps_min, steps_max + 1))
        for i in range(n):
            t0 = i / n
            t1 = (i + 1) / n
            za = z0 + (z1 - z0) * t0
            zb = z0 + (z1 - z0) * t1
            zr = 0.5 * (t0 + t1)
            r_mid = r0 + (r1 - r0) * zr
            h = max(0.0010, (zb - za) + z_overlap)
            body_parts.append(_cylinder(r_mid, h, [0, 0, 0.5 * (za + zb)], axis="z"))
        profile_spans.append((z0, z1, r0, r1))

    if body_style == "straight":
        z1 = body_h * rng.uniform(0.38, 0.55)
        z2 = body_h * rng.uniform(0.76, 0.90)
        r0 = base_r * rng.uniform(1.00, 1.08)
        r1 = base_r * rng.uniform(0.96, 1.04)
        r2 = base_r * rng.uniform(0.92, 1.00)
        _add_taper(0.0, z1, r0, r1)
        _add_taper(z1, z2, r1, r1 * rng.uniform(0.97, 1.01))
        _add_taper(z2, body_h, r1 * rng.uniform(0.97, 1.01), r2)
        body_top_r = r2
    elif body_style == "soda":
        z1 = body_h * rng.uniform(0.22, 0.36)
        z2 = body_h * rng.uniform(0.50, 0.64)
        z3 = body_h * rng.uniform(0.76, 0.88)
        r0 = base_r * rng.uniform(1.04, 1.16)
        r1 = base_r * rng.uniform(0.94, 1.04)
        r2 = base_r * rng.uniform(0.76, 0.90)
        r3 = base_r * rng.uniform(0.88, 0.98)
        r4 = base_r * rng.uniform(0.84, 0.95)
        _add_taper(0.0, z1, r0, r1)
        _add_taper(z1, z2, r1, r2)
        _add_taper(z2, z3, r2, r3)
        _add_taper(z3, body_h, r3, r4)
        body_top_r = r4
    elif body_style == "sport":
        z1 = body_h * rng.uniform(0.30, 0.46)
        z2 = body_h * rng.uniform(0.66, 0.82)
        r0 = base_r * rng.uniform(1.02, 1.12)
        r1 = base_r * rng.uniform(0.90, 1.00)
        r2 = base_r * rng.uniform(0.78, 0.90)
        r3 = base_r * rng.uniform(0.72, 0.84)
        _add_taper(0.0, z1, r0, r1)
        _add_taper(z1, z2, r1, r2)
        _add_taper(z2, body_h, r2, r3)
        body_top_r = r3
    elif body_style == "flask":
        z1 = body_h * rng.uniform(0.16, 0.30)
        z2 = body_h * rng.uniform(0.68, 0.84)
        r0 = base_r * rng.uniform(1.08, 1.22)
        r1 = base_r * rng.uniform(0.98, 1.10)
        r2 = base_r * rng.uniform(0.86, 0.98)
        _add_taper(0.0, z1, r0, r1)
        _add_taper(z1, z2, r1, r1 * rng.uniform(0.95, 1.00))
        _add_taper(z2, body_h, r1 * rng.uniform(0.95, 1.00), r2)
        body_top_r = r2
    else:
        # Square-ish silhouette: cylindrical core with side panels.
        z1 = body_h * rng.uniform(0.28, 0.42)
        z2 = body_h * rng.uniform(0.72, 0.86)
        core_r0 = base_r * rng.uniform(0.88, 0.96)
        core_r1 = core_r0 * rng.uniform(0.96, 1.02)
        core_r2 = core_r0 * rng.uniform(0.84, 0.94)
        _add_taper(0.0, z1, core_r0, core_r1)
        _add_taper(z1, z2, core_r1, core_r1 * rng.uniform(0.98, 1.02))
        _add_taper(z2, body_h, core_r1 * rng.uniform(0.98, 1.02), core_r2)
        panel_h = body_h * rng.uniform(0.42, 0.74)
        panel_z = body_h * rng.uniform(0.36, 0.58)
        panel_t = base_r * rng.uniform(0.10, 0.18)
        panel_w = base_r * rng.uniform(1.15, 1.55)
        x_off = core_r0 + panel_t / 2
        y_off = core_r0 + panel_t / 2
        body_parts.append(_box([panel_t, panel_w, panel_h], [x_off, 0.0, panel_z]))
        body_parts.append(_box([panel_t, panel_w, panel_h], [-x_off, 0.0, panel_z]))
        body_parts.append(_box([panel_w, panel_t, panel_h], [0.0, y_off, panel_z]))
        body_parts.append(_box([panel_w, panel_t, panel_h], [0.0, -y_off, panel_z]))
        body_top_r = core_r2

    def _radius_at(z: float) -> float:
        for z0, z1, r0, r1 in profile_spans:
            if z0 <= z <= z1:
                if z1 - z0 <= 1e-8:
                    return 0.5 * (r0 + r1)
                t = (z - z0) / (z1 - z0)
                return r0 + (r1 - r0) * t
        return body_top_r

    # Bottom stability ring.
    base_t = rng.uniform(0.0025, 0.0050)
    base_ring_r = max(_radius_at(0.0), base_r * 0.90) * rng.uniform(0.98, 1.05)
    body_parts.append(_cylinder(base_ring_r, base_t, [0, 0, base_t / 2], axis="z"))

    # Optional bands and vertical grip ribs to diversify silhouette.
    if rng.random() < 0.85:
        band_count = int(rng.integers(1, 5))
        band_h = rng.uniform(0.0012, 0.0032)
        for i in range(band_count):
            z = body_h * rng.uniform(0.12, 0.88)
            band_r = _radius_at(z) * rng.uniform(1.02, 1.10)
            body_parts.append(_cylinder(band_r, band_h, [0, 0, z], axis="z"))

    if rng.random() < 0.55:
        rib_count = int(rng.integers(3, 9))
        rib_h = body_h * rng.uniform(0.28, 0.68)
        rib_w = base_r * rng.uniform(0.16, 0.28)
        rib_d = base_r * rng.uniform(0.06, 0.12)
        rib_z = body_h * rng.uniform(0.30, 0.62)
        rib_radius = _radius_at(rib_z) * rng.uniform(0.92, 1.04)
        base_rib = _box([rib_w, rib_d, rib_h], [0.0, rib_radius + rib_d / 2, rib_z])
        for i in range(rib_count):
            t = 2.0 * np.pi * i / rib_count
            rib = base_rib.copy()
            rib.apply_transform(trimesh.transformations.rotation_matrix(t, [0, 0, 1], [0, 0, rib_z]))
            body_parts.append(rib)

    # Shoulder + neck transition, always at least one segment so no visual gap.
    neck_r = body_top_r * rng.uniform(0.46, 0.64)
    shoulder_h = rng.uniform(0.008, 0.024)
    shoulder_steps = int(rng.integers(1, 4))
    neck_base_z = body_h
    for i in range(shoulder_steps):
        t0 = i / shoulder_steps
        t1 = (i + 1) / shoulder_steps
        ra = body_top_r * (1.0 - t0) + neck_r * t0
        rb = body_top_r * (1.0 - t1) + neck_r * t1
        za = neck_base_z + shoulder_h * t0
        zb = neck_base_z + shoulder_h * t1
        h = max(0.0010, (zb - za) + z_overlap)
        body_parts.append(_cylinder(0.5 * (ra + rb), h, [0, 0, 0.5 * (za + zb)], axis="z"))
    neck_base_z += shoulder_h

    neck_h = rng.uniform(0.010, 0.028)
    neck = _cylinder(neck_r, neck_h, [0, 0, neck_base_z + neck_h / 2], axis="z")
    lip_h = rng.uniform(0.0020, 0.0048)
    lip_r = neck_r * rng.uniform(1.06, 1.18)
    lip = _cylinder(lip_r, lip_h, [0, 0, neck_base_z + neck_h + lip_h / 2], axis="z")

    if rng.random() < 0.65:
        tamper_h = rng.uniform(0.003, 0.009)
        tamper_r = lip_r * rng.uniform(1.01, 1.10)
        tamper_z = neck_base_z + tamper_h / 2
        body_parts.append(_cylinder(tamper_r, tamper_h, [0, 0, tamper_z], axis="z"))

    base_mesh = _merge(body_parts + [neck, lip])

    # Cap variants with more style diversity.
    cap_style = rng.choice(["short", "knurled", "tapered", "faceted", "winged", "dome_top"])
    if cap_style == "short":
        cap_h = rng.uniform(0.0085, 0.0155)
    elif cap_style == "knurled":
        cap_h = rng.uniform(0.013, 0.023)
    else:
        cap_h = rng.uniform(0.011, 0.026)

    cap_r = lip_r * rng.uniform(1.01, 1.15)
    cap_overlap = rng.uniform(0.0010, 0.0030)
    cap_bottom_z = neck_base_z + neck_h + lip_h - cap_overlap
    cap_center_z = cap_bottom_z + cap_h / 2
    cap_parts = []

    if cap_style == "tapered":
        h1 = cap_h * rng.uniform(0.45, 0.62)
        h2 = cap_h - h1
        r1 = cap_r * rng.uniform(1.00, 1.06)
        r2 = cap_r * rng.uniform(0.82, 0.95)
        cap_parts.append(_cylinder(r1, h1, [0, 0, cap_bottom_z + h1 / 2], axis="z"))
        cap_parts.append(_cylinder(r2, h2, [0, 0, cap_bottom_z + h1 + h2 / 2], axis="z"))
    elif cap_style == "faceted":
        cap_parts.append(_cylinder(cap_r * 0.92, cap_h, [0, 0, cap_center_z], axis="z"))
        facet_w = cap_h * rng.uniform(0.45, 0.75)
        facet_d = cap_r * rng.uniform(0.10, 0.17)
        facet_h = cap_h * rng.uniform(0.70, 0.96)
        panel = _box([facet_w, facet_d, facet_h], [0.0, cap_r + facet_d / 2, cap_center_z])
        facet_count = int(rng.integers(5, 9))
        for i in range(facet_count):
            t = 2.0 * np.pi * i / facet_count
            facet = panel.copy()
            facet.apply_transform(trimesh.transformations.rotation_matrix(t, [0, 0, 1], [0, 0, cap_center_z]))
            cap_parts.append(facet)
    else:
        cap_parts.append(_cylinder(cap_r, cap_h, [0, 0, cap_center_z], axis="z"))

    if cap_style in ("knurled", "short"):
        rib_count = int(rng.integers(8, 24))
        rib_w = rng.uniform(0.0012, 0.0032)
        rib_d = rng.uniform(0.0009, 0.0028)
        rib_h = cap_h * rng.uniform(0.58, 0.96)
        base_rib = _box([rib_w, rib_d, rib_h], [0, cap_r + rib_d / 2, cap_center_z])
        for i in range(rib_count):
            t = 2.0 * np.pi * i / rib_count
            rib = base_rib.copy()
            rib.apply_transform(trimesh.transformations.rotation_matrix(t, [0, 0, 1], [0, 0, cap_center_z]))
            cap_parts.append(rib)

    if cap_style == "winged":
        wing_w = cap_r * rng.uniform(0.26, 0.46)
        wing_d = rng.uniform(0.002, 0.005)
        wing_h = cap_h * rng.uniform(0.22, 0.48)
        x_off = cap_r * rng.uniform(0.72, 1.02)
        wing_y = cap_r * rng.uniform(0.18, 0.35)
        cap_parts.append(_box([wing_w, wing_d, wing_h], [-x_off, wing_y, cap_center_z]))
        cap_parts.append(_box([wing_w, wing_d, wing_h], [x_off, wing_y, cap_center_z]))

    if cap_style == "dome_top" or rng.random() < 0.35:
        dome_r = cap_r * rng.uniform(0.18, 0.38)
        cap_parts.append(_sphere(dome_r, [0, 0, cap_bottom_z + cap_h + dome_r * 0.25]))

    if rng.random() < 0.30:
        top_ring_h = rng.uniform(0.001, 0.0028)
        top_ring_r = cap_r * rng.uniform(0.88, 0.98)
        cap_parts.append(_cylinder(top_ring_r, top_ring_h, [0, 0, cap_bottom_z + cap_h - top_ring_h / 2], axis="z"))

    child_mesh = _merge(cap_parts)

    turns_to_open = rng.uniform(0.30, 1.90)
    open_angle = 2.0 * np.pi * turns_to_open

    return ArticulatedObject(
        name="bottle_cap",
        base_mesh=base_mesh,
        child_mesh=child_mesh,
        joint_type="revolute",
        joint_origin=np.array([0.0, 0.0, cap_center_z]),
        joint_axis=np.array([0.0, 0.0, 1.0]),
        joint_limits=(0.0, open_angle),
        joint_default=0.0,
        goal_joint_value=open_angle,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TEMPLATES: Dict[str, Callable] = {
    "scissors": generate_scissors,
    "book": generate_book,
    "laptop": generate_laptop,
    "tongs": generate_tongs,
    "drawer": generate_drawer,
    "door_handle": generate_door_handle,
    "bottle_cap": generate_bottle_cap,
}

def generate_random(template_name: str, seed=None) -> ArticulatedObject:
    if template_name not in TEMPLATES:
        valid = ", ".join(TEMPLATES.keys())
        raise KeyError(f"Unknown template '{template_name}'. Valid templates: {valid}")
    rng = np.random.default_rng(seed)
    return TEMPLATES[template_name](rng)

def generate_all(seed=None) -> Dict[str, ArticulatedObject]:
    rng = np.random.default_rng(seed)
    return {name: fn(rng) for name, fn in TEMPLATES.items()}
