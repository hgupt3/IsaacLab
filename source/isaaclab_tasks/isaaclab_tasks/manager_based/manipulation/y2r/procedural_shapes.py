# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Procedural shape generation using grow-the-object algorithm.

Generates random objects by hierarchically attaching primitives:
1. Start with a random root primitive
2. For each additional primitive, pick a random existing one and attach to its surface
3. Boolean union all primitives into a single mesh
4. Export to USD with convex decomposition for collision

Usage:
    # Generate shapes (run once with Isaac Sim)
    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/y2r/scripts/generate_shapes.py
    
    # Then train as normal
    ./isaaclab.sh -p scripts/rl_games/train.py task=Isaac-Y2R-Kuka-Allegro-v0
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _remap_to_data_root(path: Path) -> Path:
    """If Y2R_DATA_ROOT is set, remap a repo path to the data root.

    Y2R_DATA_ROOT mirrors the repo root structure, e.g.:
        repo at /home/user/sam → Y2R_DATA_ROOT=/usr0/user/sam
    """
    if "Y2R_DATA_ROOT" not in os.environ:
        return path
    repo_root = Path(__file__).resolve().parents[7]
    try:
        rel = path.relative_to(repo_root)
        return Path(os.environ["Y2R_DATA_ROOT"]) / rel
    except ValueError:
        return path


class ProceduralShapesNotFoundError(Exception):
    """Raised when procedural shapes are required but not found."""
    pass


def get_procedural_shape_paths(cfg, y2r_dir: Path) -> list[Path]:
    """
    Get paths to procedural shapes, erroring if not enough exist.

    This function does NOT generate shapes - it only checks for existing ones.
    If shapes are missing, run the standalone generation script first:
        ./isaaclab.sh -p scripts/tools/generate_procedural_shapes.py

    Args:
        cfg: ProceduralObjectsConfig dataclass
        y2r_dir: Path to the y2r module directory

    Returns:
        List of paths to existing shape files (USD or OBJ)

    Raises:
        ProceduralShapesNotFoundError: If not enough shapes exist
    """
    if not cfg.enabled:
        return []

    asset_dir = _remap_to_data_root(y2r_dir / cfg.asset_dir)
    num_shapes = cfg.generation.num_shapes
    
    # Check for existing shapes (prefer USD, fallback to OBJ)
    existing_usd = sorted(asset_dir.glob("shape_*.usd"))
    existing_obj = sorted(asset_dir.glob("shape_*.obj"))
    existing = existing_usd if existing_usd else existing_obj
    
    if len(existing) < num_shapes:
        raise ProceduralShapesNotFoundError(
            f"Not enough procedural shapes found!\n"
            f"  Required: {num_shapes}\n"
            f"  Found: {len(existing)} in {asset_dir}\n"
            f"\n"
            f"Please generate shapes first by running:\n"
            f"  ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/y2r/scripts/generate_shapes.py\n"
            f"\n"
            f"Or disable procedural objects in configs/env.yaml:\n"
            f"  procedural_objects:\n"
            f"    enabled: false"
        )
    
    return existing[:num_shapes]


# ==============================================================================
# Generation functions (used by standalone script only)
# ==============================================================================

def _weighted_choice(weights: dict[str, float]) -> str:
    """Select a key from dict based on weights."""
    import numpy as np
    keys = list(weights.keys())
    values = [weights[k] for k in keys]
    total = sum(values)
    probs = [v / total for v in values]
    return np.random.choice(keys, p=probs)


def _make_primitive(ptype: str, size: float):
    """Create a primitive mesh of given type and approximate size."""
    import numpy as np
    import trimesh
    
    if ptype == "sphere":
        # Simple sphere with randomized radius
        radius = size * np.random.uniform(0.4, 1.0)
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    elif ptype == "box":
        # Random aspect ratios for box
        extents = np.random.uniform(0.1, 1.0, size=3)
        extents = extents / np.max(extents) * size * 2
        mesh = trimesh.creation.box(extents=extents)
    elif ptype == "capsule":
        # Random height/radius ratio
        radius = size * np.random.uniform(0.2, 1.0)
        height = size * np.random.uniform(0.2, 1.0)
        mesh = trimesh.creation.capsule(radius=radius, height=height)
    elif ptype == "cylinder":
        # Random height/radius ratio
        radius = size * np.random.uniform(0.2, 1.0)
        height = size * np.random.uniform(0.2, 1.0)
        mesh = trimesh.creation.cylinder(radius=radius, height=height)
    elif ptype == "annulus":
        # Ring/washer shape with random inner and outer radius
        r_min = size * np.random.uniform(0.3, 0.55)
        r_max = size * np.random.uniform(0.6, 1.0)
        height = size * np.random.uniform(0.2, 0.5)
        mesh = trimesh.creation.annulus(r_min=r_min, r_max=r_max, height=height)
    elif ptype == "cone":
        # Cone with randomized radius and height, centered at origin
        radius = size * np.random.uniform(0.3, 1.0)
        height = size * np.random.uniform(0.4, 1.0)
        mesh = trimesh.creation.cone(radius=radius, height=height)
        mesh.vertices -= mesh.centroid
    else:
        # Default to sphere
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=size)
    
    return mesh


def _sample_surface_point(mesh) -> tuple:
    """Sample a random point on mesh surface and return (point, normal)."""
    import trimesh
    points, face_indices = trimesh.sample.sample_surface(mesh, count=1)
    point = points[0]
    normal = mesh.face_normals[face_indices[0]]
    return point, normal


def _scale_to_max_extent(mesh, target_max_extent):
    """Scale mesh uniformly so its largest dimension equals target_max_extent."""
    import numpy as np
    current_max = np.max(mesh.extents)
    if current_max > 0:
        scale = target_max_extent / current_max
        mesh.apply_scale(scale)
    return mesh


def _grow_tool(cfg: dict[str, Any], rng):
    """
    Generate a tool-shaped object: elongated handle + wider head at one end.

    The handle is forced to be elongated (6-12x length/width ratio).
    The head is wider than the handle (1.6-2.8x) and attached at one end.
    Optional extras (collar, end-cap, second head) fill remaining primitive budget.

    Args:
        cfg: Generation config dict (same keys as grow_object)
        rng: numpy random generator

    Returns:
        Combined mesh, or None if boolean union fails
    """
    import numpy as np
    import trimesh

    # Sample primitive budget
    min_prims, max_prims = cfg["primitives_per_shape"]
    num_primitives = int(rng.integers(max(min_prims, 2), max_prims + 1))

    # Handle size from base_size config
    base_size_min, base_size_max = cfg["base_size"]
    handle_width = rng.uniform(base_size_min, base_size_max)

    # === HANDLE (1 primitive) ===
    elongation = rng.uniform(6.0, 12.0)
    handle_length = handle_width * elongation
    handle_radius = handle_width / 2

    handle_type = rng.choice(["cylinder", "capsule", "box"])
    if handle_type == "cylinder":
        handle = trimesh.creation.cylinder(radius=handle_radius, height=handle_length)
    elif handle_type == "capsule":
        handle = trimesh.creation.capsule(radius=handle_radius, height=handle_length)
    else:  # box
        # Slight width variation for box cross-section
        w2 = handle_width * rng.uniform(0.6, 1.0)
        handle = trimesh.creation.box(extents=(handle_width, w2, handle_length))

    # Use actual bounding box for endpoints (capsule total height includes caps)
    handle_top_z = handle.bounds[1][2]
    handle_bot_z = handle.bounds[0][2]

    primitives = [handle]
    budget_remaining = num_primitives - 1  # At least 1 for primary head

    # === PRIMARY HEAD (1 primitive) ===
    head_type = _weighted_choice(cfg["primitive_types"])
    head_width_ratio = rng.uniform(1.8, 3.2)
    head_target_width = handle_width * head_width_ratio
    head = _make_primitive(head_type, head_target_width)
    # Enforce size: scale so max extent matches desired width
    _scale_to_max_extent(head, head_target_width)

    # Place head at top end of handle with guaranteed overlap
    head_bounds = head.bounds
    head_half_extent_z = (head_bounds[1][2] - head_bounds[0][2]) / 2
    overlap_factor = rng.uniform(0.3, 0.6)
    head_z = handle_top_z + head_half_extent_z * (1.0 - overlap_factor)

    # Random rotation biased toward perpendicular to handle axis
    # Tilt angle from Z axis: 60-120 degrees
    tilt = rng.uniform(np.pi / 3, 2 * np.pi / 3)
    spin = rng.uniform(0, 2 * np.pi)
    rot_axis = np.array([np.cos(spin), np.sin(spin), 0.0])
    rot_matrix = trimesh.transformations.rotation_matrix(tilt, rot_axis)[:3, :3]
    head.apply_transform(
        np.vstack([
            np.hstack([rot_matrix, np.array([[0], [0], [head_z]])]),
            [0, 0, 0, 1]
        ])
    )
    primitives.append(head)
    budget_remaining -= 1

    # === OPTIONAL EXTRAS (use remaining budget) ===

    # Collar near head
    if budget_remaining > 0 and rng.random() < 0.3:
        collar_radius = handle_radius * rng.uniform(1.1, 1.3)
        collar_height = handle_width * rng.uniform(0.3, 0.6)
        collar = trimesh.creation.cylinder(radius=collar_radius, height=collar_height)
        collar_z = handle_top_z - collar_height
        collar.apply_translation([0, 0, collar_z])
        primitives.append(collar)
        budget_remaining -= 1

    # End-cap at butt end
    if budget_remaining > 0 and rng.random() < 0.2:
        cap_type = rng.choice(["sphere", "cylinder"])
        cap_size = handle_radius * rng.uniform(1.0, 1.3)
        if cap_type == "sphere":
            cap = trimesh.creation.icosphere(subdivisions=2, radius=cap_size)
        else:
            cap = trimesh.creation.cylinder(radius=cap_size, height=handle_width * 0.4)
        cap.apply_translation([0, 0, handle_bot_z])
        primitives.append(cap)
        budget_remaining -= 1

    # Second head at opposite end (smaller)
    if budget_remaining > 0 and rng.random() < 0.2:
        head2_type = _weighted_choice(cfg["primitive_types"])
        head2_target = head_target_width * rng.uniform(0.5, 0.8)
        head2 = _make_primitive(head2_type, head2_target)
        _scale_to_max_extent(head2, head2_target)

        head2_bounds = head2.bounds
        head2_half_z = (head2_bounds[1][2] - head2_bounds[0][2]) / 2
        head2_z = handle_bot_z - head2_half_z * (1.0 - overlap_factor)

        tilt2 = rng.uniform(np.pi / 3, 2 * np.pi / 3)
        spin2 = rng.uniform(0, 2 * np.pi)
        rot_axis2 = np.array([np.cos(spin2), np.sin(spin2), 0.0])
        rot_matrix2 = trimesh.transformations.rotation_matrix(tilt2, rot_axis2)[:3, :3]
        head2.apply_transform(
            np.vstack([
                np.hstack([rot_matrix2, np.array([[0], [0], [head2_z]])]),
                [0, 0, 0, 1]
            ])
        )
        primitives.append(head2)
        budget_remaining -= 1

    # === BOOLEAN UNION ===
    result = primitives[0]
    for mesh in primitives[1:]:
        try:
            result = result.union(mesh, engine="manifold")
        except (ValueError, Exception):
            return None

    # Clean up mesh
    result.merge_vertices()
    result.remove_unreferenced_vertices()
    result.fill_holes()
    result.process(validate=True)

    # Center at origin
    result.vertices -= result.centroid
    return result


def grow_object(cfg: dict[str, Any], rng=None):
    """
    Generate a random object using the grow-the-object algorithm.

    Dispatches to family-specific generators based on shape_families weights.

    Args:
        cfg: Generation config with keys:
            - primitives_per_shape: [min, max] number of primitives
            - base_size: [min, max] size of root primitive
            - size_decay: multiplier for child primitive sizes
            - primitive_types: dict of type -> weight
            - shape_families: dict of family -> weight
        rng: Optional numpy random generator for reproducibility

    Returns:
        Combined mesh of all primitives
    """
    import numpy as np
    import trimesh

    if rng is None:
        rng = np.random.default_rng()

    # Use numpy's random state
    old_state = np.random.get_state()
    np.random.seed(rng.integers(0, 2**31))

    # Select shape family (after RNG seeding for determinism)
    family = _weighted_choice(cfg["shape_families"])
    if family == "tool":
        result = _grow_tool(cfg, rng)
        np.random.set_state(old_state)
        return result
    elif family != "blob":
        raise ValueError(f"Unknown shape family: '{family}'. Expected 'blob' or 'tool'.")

    # === Blob family (original behavior) ===

    # Determine number of primitives
    min_prims, max_prims = cfg.get("primitives_per_shape", [3, 8])
    num_primitives = rng.integers(min_prims, max_prims + 1)
    
    # Create root primitive
    base_size_min, base_size_max = cfg.get("base_size", [0.03, 0.08])
    root_size = rng.uniform(base_size_min, base_size_max)
    root_type = _weighted_choice(cfg.get("primitive_types", {"sphere": 1.0}))
    
    primitives = [_make_primitive(root_type, root_size)]
    current_size = root_size
    
    # Grow by attaching children
    for _ in range(num_primitives - 1):
        # Pick random existing primitive to attach to
        parent_idx = rng.integers(0, len(primitives))
        parent = primitives[parent_idx]
        
        # Sample point on parent surface
        point, normal = _sample_surface_point(parent)
        
        # Create child primitive (smaller)
        current_size *= cfg.get("size_decay", 0.8)
        child_type = _weighted_choice(cfg.get("primitive_types", {"sphere": 1.0}))
        child = _make_primitive(child_type, current_size)
        
        # Get child's approximate radius for offset
        # Use MINIMUM dimension to ensure good overlap for elongated shapes
        child_bounds = child.bounds
        child_extents = child_bounds[1] - child_bounds[0]
        child_min_radius = np.min(child_extents) / 2
        
        # Position child so it overlaps with parent
        # Place center at point + normal * (min_radius * overlap_factor)
        # With factor < 1.0, the child penetrates into the parent
        overlap_factor = rng.uniform(0.2, 0.5)  # Random overlap depth
        offset = point + normal * child_min_radius * overlap_factor
        child.apply_translation(offset)
        
        primitives.append(child)
    
    # Boolean union all primitives
    if len(primitives) == 1:
        result = primitives[0]
    else:
        result = primitives[0]
        for mesh in primitives[1:]:
            try:
                result = result.union(mesh, engine="manifold")
            except (ValueError, Exception):
                # Boolean failed (non-volume mesh) - abort this shape
                np.random.set_state(old_state)
                return None
    
    # Clean up mesh
    result.merge_vertices()
    result.remove_unreferenced_vertices()
    result.fill_holes()
    result.process(validate=True)
    
    # Center at origin
    result.vertices -= result.centroid
    
    np.random.set_state(old_state)
    return result
    
def generate_procedural_shapes(cfg: dict[str, Any], y2r_dir: Path) -> list[Path]:
    """
    Generate procedural shapes and save to asset directory.
    
    This function should be called from the standalone generation script
    which runs with Isaac Sim initialized for proper USD conversion.
    
    Args:
        cfg: The procedural_objects config section from y2r_config.yaml
        y2r_dir: Path to the y2r module directory

    Returns:
        List of paths to generated shape files
    """
    import numpy as np

    asset_dir = _remap_to_data_root(y2r_dir / cfg.get("asset_dir", "assets/procedural"))
    gen_cfg = cfg.get("generation", {})
    num_shapes = gen_cfg.get("num_shapes", 100)
    
    print(f"Generating {num_shapes} procedural shapes in {asset_dir}...")
    asset_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up random seed
    seed = gen_cfg.get("seed")
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    generated = []
    i = 0
    failures = 0
    max_failures = num_shapes * 3  # Prevent infinite loop
    
    while len(generated) < num_shapes and failures < max_failures:
        # Generate mesh (may return None if boolean fails)
        mesh = grow_object(gen_cfg, rng)
        
        if mesh is None:
            failures += 1
            continue
        
        # Export - try USD first, fallback to OBJ
        output_path = asset_dir / f"shape_{i:03d}.usd"
        _export_mesh(mesh, output_path)
        
        # Track what was actually created
        if output_path.exists():
            generated.append(output_path)
            i += 1
        else:
            obj_path = output_path.with_suffix(".obj")
            if obj_path.exists():
                generated.append(obj_path)
                i += 1
            else:
                failures += 1
                continue
        
        if len(generated) % 10 == 0:
            print(f"  Generated {len(generated)}/{num_shapes} shapes (skipped {failures})")
    
    print(f"Generated {len(generated)} procedural shapes (skipped {failures} failed attempts)")
    return generated


def _export_mesh(mesh, output_path: Path) -> None:
    """Export mesh to USD (preferred) or OBJ (fallback)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
    from isaaclab.sim.schemas import (
        MassPropertiesCfg,
        RigidBodyPropertiesCfg,
        CollisionPropertiesCfg,
        ConvexDecompositionPropertiesCfg,
    )
    
    # Export to temporary OBJ
    temp_obj = output_path.with_suffix(".obj")
    mesh.export(str(temp_obj))
    
    # Convert to USD with convex decomposition
    converter_cfg = MeshConverterCfg(
        asset_path=str(temp_obj),
        usd_dir=str(output_path.parent),
        usd_file_name=output_path.name,
        force_usd_conversion=True,
        make_instanceable=False,
        mass_props=MassPropertiesCfg(mass=0.2),
        rigid_props=RigidBodyPropertiesCfg(),
        collision_props=CollisionPropertiesCfg(),
        mesh_collision_props=ConvexDecompositionPropertiesCfg(),
    )
    converter = MeshConverter(converter_cfg)
    
    # Clean up temp OBJ
    temp_obj.unlink(missing_ok=True)
        
