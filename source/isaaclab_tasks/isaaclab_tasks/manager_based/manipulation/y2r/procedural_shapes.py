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
    ./isaaclab.sh -p scripts/tools/generate_procedural_shapes.py
    
    # Then train as normal
    ./isaaclab.sh -p scripts/rl_games/train.py task=Isaac-Y2R-Kuka-Allegro-v0
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


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
    
    asset_dir = y2r_dir / cfg.asset_dir
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
            f"  ./isaaclab.sh -p scripts/tools/generate_procedural_shapes.py\n"
            f"\n"
            f"Or disable procedural objects in y2r_config.yaml:\n"
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


def grow_object(cfg: dict[str, Any], rng=None):
    """
    Generate a random object using the grow-the-object algorithm.
    
    Args:
        cfg: Generation config with keys:
            - primitives_per_shape: [min, max] number of primitives
            - base_size: [min, max] size of root primitive
            - size_decay: multiplier for child primitive sizes
            - primitive_types: dict of type -> weight
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
            result = result.union(mesh, engine="manifold")
    
    # Clean up mesh
    result.merge_vertices()
    result.remove_unreferenced_vertices()
    result.fill_holes()
    result.process(validate=True)
    
    # Center at origin
    result.vertices -= result.centroid
    
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
    
    asset_dir = y2r_dir / cfg.get("asset_dir", "assets/procedural")
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
    for i in range(num_shapes):
        # Generate mesh
        mesh = grow_object(gen_cfg, rng)
        
        # Export - try USD first, fallback to OBJ
        output_path = asset_dir / f"shape_{i:03d}.usd"
        _export_mesh(mesh, output_path)
        
        # Track what was actually created
        if output_path.exists():
            generated.append(output_path)
        else:
            obj_path = output_path.with_suffix(".obj")
            if obj_path.exists():
                generated.append(obj_path)
        
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{num_shapes} shapes")
    
    print(f"Generated {len(generated)} procedural shapes")
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
        
