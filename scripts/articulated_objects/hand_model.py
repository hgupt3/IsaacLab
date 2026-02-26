"""Cached LEAP hand model for efficient re-posing.

Parses the URDF once, caches per-link mesh data, and recomputes only FK
transforms when joint values change.  Designed for ~30 FPS animation.
"""

import os
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from generator import _URDF_PATH, _HAND_LINKS, _parse_origin, _make_schematic_hand


# ── Joint limits (from URDF) ────────────────────────────────────────────────
# index/middle/ring: joint_0 = splay, joint_1 = MCP flex, joint_2 = PIP, joint_3 = DIP
# thumb: joint_0..3 have unique ranges
JOINT_LIMITS = {
    "index_joint_0":  (-0.349, 0.349),
    "index_joint_1":  (-0.314, 2.23),
    "index_joint_2":  (-0.506, 1.885),
    "index_joint_3":  (-0.366, 2.042),
    "middle_joint_0": (-0.349, 0.349),
    "middle_joint_1": (-0.314, 2.23),
    "middle_joint_2": (-0.506, 1.885),
    "middle_joint_3": (-0.366, 2.042),
    "ring_joint_0":   (-0.349, 0.349),
    "ring_joint_1":   (-0.314, 2.23),
    "ring_joint_2":   (-0.506, 1.885),
    "ring_joint_3":   (-0.366, 2.042),
    "thumb_joint_0":  (-0.349, 2.094),
    "thumb_joint_1":  (-0.47, 2.443),
    "thumb_joint_2":  (-1.20, 1.90),
    "thumb_joint_3":  (-1.34, 1.88),
}

ALL_JOINT_NAMES = list(JOINT_LIMITS.keys())


# ── Finger presets ───────────────────────────────────────────────────────────

def _preset(index_splay=0.0, middle_splay=0.0, ring_splay=0.0,
            index_flex=None, middle_flex=None, ring_flex=None,
            thumb=None):
    """Build joint-value dict from compact specification.

    *_flex: (joint_1, joint_2, joint_3) tuple for each finger.
    thumb:  (joint_0, joint_1, joint_2, joint_3) tuple.
    """
    if index_flex is None:
        index_flex = (0.0, 0.0, 0.0)
    if middle_flex is None:
        middle_flex = (0.0, 0.0, 0.0)
    if ring_flex is None:
        ring_flex = (0.0, 0.0, 0.0)
    if thumb is None:
        thumb = (0.0, 0.0, 0.0, 0.0)

    return {
        "index_joint_0":  index_splay,
        "index_joint_1":  index_flex[0],
        "index_joint_2":  index_flex[1],
        "index_joint_3":  index_flex[2],
        "middle_joint_0": middle_splay,
        "middle_joint_1": middle_flex[0],
        "middle_joint_2": middle_flex[1],
        "middle_joint_3": middle_flex[2],
        "ring_joint_0":   ring_splay,
        "ring_joint_1":   ring_flex[0],
        "ring_joint_2":   ring_flex[1],
        "ring_joint_3":   ring_flex[2],
        "thumb_joint_0":  thumb[0],
        "thumb_joint_1":  thumb[1],
        "thumb_joint_2":  thumb[2],
        "thumb_joint_3":  thumb[3],
    }


FINGER_PRESETS = {
    "open": _preset(
        index_flex=(-0.1, -0.1, -0.1),
        middle_flex=(-0.1, -0.1, -0.1),
        ring_flex=(-0.1, -0.1, -0.1),
        thumb=(0.0, -0.1, 0.0, 0.0),
    ),
    "closed": _preset(
        index_flex=(1.5, 1.5, 1.5),
        middle_flex=(1.5, 1.5, 1.5),
        ring_flex=(1.5, 1.5, 1.5),
        thumb=(1.0, 1.5, 0.8, 1.0),
    ),
    "pinch": _preset(
        index_flex=(1.2, 1.0, 0.8),
        middle_flex=(-0.1, -0.1, -0.1),
        ring_flex=(-0.1, -0.1, -0.1),
        thumb=(0.8, 1.2, 0.4, 0.6),
    ),
    "power": _preset(
        index_flex=(1.0, 1.2, 1.0),
        middle_flex=(1.0, 1.2, 1.0),
        ring_flex=(1.0, 1.2, 1.0),
        index_splay=0.1, ring_splay=-0.1,
        thumb=(0.8, 1.0, 0.4, 0.8),
    ),
    "hook": _preset(
        index_flex=(0.3, 1.5, 1.5),
        middle_flex=(0.3, 1.5, 1.5),
        ring_flex=(0.3, 1.5, 1.5),
        thumb=(0.0, -0.1, 0.0, 0.0),
    ),
    "flat_push": _preset(
        index_flex=(0.0, 0.0, 0.0),
        middle_flex=(0.0, 0.0, 0.0),
        ring_flex=(0.0, 0.0, 0.0),
        thumb=(0.0, 0.0, 0.0, 0.0),
    ),
}


# ── HandModel ────────────────────────────────────────────────────────────────

class HandModel:
    """Singleton: parses URDF once, caches per-link mesh data, re-poses efficiently."""

    _instance = None

    @classmethod
    def get(cls) -> "HandModel":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._use_schematic = False
        self._link_visuals = {}   # link_name -> [(origin_T, trimesh.Trimesh)]
        self._joints = []         # [(jname, jtype, parent, child, origin_T, axis)]
        self._palm_frame_T = np.eye(4)

        if not os.path.isfile(_URDF_PATH):
            self._use_schematic = True
            return

        self._urdf_dir = os.path.dirname(os.path.abspath(_URDF_PATH))
        tree = ET.parse(_URDF_PATH)
        root = tree.getroot()

        # Parse link visuals and pre-load meshes
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
                if mesh_elem is None:
                    continue
                fname = mesh_elem.get("filename")
                scale_str = mesh_elem.get("scale")
                scale = np.ones(3)
                if scale_str:
                    scale = np.array([float(v) for v in scale_str.split()])
                fpath = os.path.join(self._urdf_dir, fname)
                if not os.path.isfile(fpath):
                    continue
                try:
                    m = trimesh.load(fpath, force="mesh")
                    if not hasattr(m, "vertices") or m.vertices.shape[0] == 0:
                        continue
                    m.vertices *= scale
                    visuals.append((origin_T, m))
                except Exception:
                    continue
            if visuals:
                self._link_visuals[name] = visuals

        if not self._link_visuals:
            self._use_schematic = True
            return

        # Parse joints
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
            self._joints.append((jname, jtype, parent, child, origin_T, axis))

            # Capture palm_frame fixed joint
            if jname == "palm_frame_joint":
                self._palm_frame_T = origin_T.copy()

    def get_palm_frame(self) -> np.ndarray:
        """Return the 4x4 palm_frame offset relative to palm_link (from URDF fixed joint)."""
        return self._palm_frame_T.copy()

    def get_mesh(self, joint_values: dict = None) -> trimesh.Trimesh:
        """Recompute FK from cached data, transform cached meshes, return merged.

        Args:
            joint_values: dict mapping joint_name -> angle (rad). Defaults to 0.
        """
        if self._use_schematic:
            return _make_schematic_hand()

        if joint_values is None:
            joint_values = {}

        # BFS forward kinematics from palm_link
        world_T = {"palm_link": np.eye(4)}
        resolved = {"palm_link"}
        for _ in range(10):
            for jname, jtype, parent, child, origin_T, axis in self._joints:
                if parent in resolved and child not in resolved and child in _HAND_LINKS:
                    T_joint = origin_T.copy()
                    if jtype == "revolute":
                        q = joint_values.get(jname, 0.0)
                        R_q = np.eye(4)
                        R_q[:3, :3] = Rotation.from_rotvec(q * axis).as_matrix()
                        T_joint = T_joint @ R_q
                    world_T[child] = world_T[parent] @ T_joint
                    resolved.add(child)

        # Transform cached meshes
        meshes = []
        for link_name, visuals in self._link_visuals.items():
            if link_name not in world_T:
                continue
            T_link = world_T[link_name]
            for vis_origin_T, cached_mesh in visuals:
                m = cached_mesh.copy()
                T_full = T_link @ vis_origin_T
                m.apply_transform(T_full)
                meshes.append(m)

        if not meshes:
            return _make_schematic_hand()

        return trimesh.util.concatenate(meshes)
