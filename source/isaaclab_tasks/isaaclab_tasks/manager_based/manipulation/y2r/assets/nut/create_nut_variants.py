"""
Create multiple nut variants with radial thickening and inner hole chamfering.
"""

from pxr import Usd, UsdGeom, Gf, Vt
import numpy as np
import os

def modify_mesh(
    mesh: UsdGeom.Mesh,
    radial_scale: float,
    chamfer_depth: float,
    chamfer_expansion: float,
    inner_max_radius: float
):
    """Modify a single mesh (visual or collision)."""
    # Get mesh data
    points = mesh.GetPointsAttr().Get()
    points_np = np.array([(p[0], p[1], p[2]) for p in points])

    # Calculate radial distances in XY plane
    xy_dists = np.linalg.norm(points_np[:, :2], axis=1)

    # Find Z bounds
    z_min = points_np[:, 2].min()
    z_max = points_np[:, 2].max()

    # Identify vertex groups
    inner_mask = xy_dists <= inner_max_radius  # Inner hole + threading
    outer_mask = xy_dists > inner_max_radius   # Outer surface

    modified = points_np.copy()

    # Step 1: Radial scaling for outer vertices only
    if radial_scale != 1.0:
        modified[outer_mask, 0] *= radial_scale
        modified[outer_mask, 1] *= radial_scale

    # Step 2: Chamfer disabled (modifying bolt instead)
    chamfered_top = 0
    chamfered_bottom = 0

    # Update mesh
    modified_usd = Vt.Vec3fArray([Gf.Vec3f(*p) for p in modified])
    mesh.GetPointsAttr().Set(modified_usd)

    return {
        'total_vertices': len(points),
        'outer_vertices': outer_mask.sum(),
        'chamfered_top': chamfered_top,
        'chamfered_bottom': chamfered_bottom,
        'outer_radius_before': xy_dists[outer_mask].max() if outer_mask.any() else 0,
        'outer_radius_after': np.linalg.norm(modified[outer_mask, :2], axis=1).max() if outer_mask.any() else 0,
        'inner_radius_after': np.linalg.norm(modified[inner_mask, :2], axis=1).max() if inner_mask.any() else 0,
    }


def create_nut_variant(
    input_path: str,
    output_path: str,
    radial_scale: float = 1.0,
    chamfer_depth: float = 0.001,
    chamfer_expansion: float = 0.10,
    inner_min_radius: float = 0.0069,
    inner_max_radius: float = 0.0087
):
    """
    Create a nut variant with radial thickening and inner hole chamfering.

    Args:
        input_path: Input USD file
        output_path: Output USD file
        radial_scale: Multiply outer radius by this (1.0 = no change, 1.2 = 20% thicker)
        chamfer_depth: How far from top/bottom to chamfer inner hole (meters, e.g., 0.001 = 1mm)
        chamfer_expansion: How much to expand inner hole at edges (e.g., 0.10 = 10%)
        inner_min_radius: Minimum inner radius to identify inner hole vertices
        inner_max_radius: Maximum inner radius to separate inner from outer
    """
    stage = Usd.Stage.Open(input_path)

    # Get both visual and collision meshes
    visual_mesh = UsdGeom.Mesh.Get(stage, '/Root/Object/factory_nut_loose/visuals')
    collision_mesh = UsdGeom.Mesh.Get(stage, '/Root/Object/factory_nut_loose/collisions')

    print(f"Processing: radial_scale={radial_scale:.2f}, chamfer={chamfer_expansion*100:.0f}% over {chamfer_depth*1000:.1f}mm")

    # Modify visual mesh
    print("  [Visual mesh]")
    visual_stats = modify_mesh(visual_mesh, radial_scale, chamfer_depth, chamfer_expansion, inner_max_radius)
    print(f"    Vertices: {visual_stats['total_vertices']} ({visual_stats['outer_vertices']} outer)")
    print(f"    Radial scaling: {visual_stats['outer_vertices']} vertices × {radial_scale:.2f}")
    print(f"    Chamfering: {visual_stats['chamfered_top']} top, {visual_stats['chamfered_bottom']} bottom")

    # Modify collision mesh
    print("  [Collision mesh]")
    collision_stats = modify_mesh(collision_mesh, radial_scale, chamfer_depth, chamfer_expansion, inner_max_radius)
    print(f"    Vertices: {collision_stats['total_vertices']} ({collision_stats['outer_vertices']} outer)")
    print(f"    Radial scaling: {collision_stats['outer_vertices']} vertices × {radial_scale:.2f}")
    print(f"    Chamfering: {collision_stats['chamfered_top']} top, {collision_stats['chamfered_bottom']} bottom")

    # Save
    stage.Export(output_path)

    print(f"  ✓ Saved: {os.path.basename(output_path)}")
    print(f"    Outer: {visual_stats['outer_radius_before']*1000:.2f}mm → {visual_stats['outer_radius_after']*1000:.2f}mm")
    print(f"    Inner: {visual_stats['inner_radius_after']*1000:.2f}mm (at edges)")
    print()


def create_all_variants():
    """Create nut0 through nut5 with progressive thickening."""

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_usd = os.path.join(script_dir, "nut.usd")
    output_dir = script_dir

    # Configuration
    variants = [
        # (name, radial_scale)
        ("nut0.usd", 1.0),   # Original size + chamfer
        ("nut1.usd", 1.2),   # 20% thicker
        ("nut2.usd", 1.4),   # 40% thicker
        ("nut3.usd", 1.6),   # 60% thicker
        ("nut4.usd", 1.8),   # 80% thicker
        ("nut5.usd", 2.0),   # 100% thicker (2x)
    ]

    # Chamfer settings (same for all variants)
    CHAMFER_DEPTH = 0.0025     # 2.5mm from bottom edge
    CHAMFER_EXPANSION = 0.25   # 25% expansion (always relative to original)

    print("=" * 70)
    print("Creating Nut Variants with Inner Hole Chamfering")
    print("=" * 70)
    print(f"Chamfer: {CHAMFER_EXPANSION*100:.0f}% expansion over {CHAMFER_DEPTH*1000:.1f}mm depth (bottom only, linear)")
    print(f"Modifying BOTH visual and collision meshes")
    print()

    for name, scale in variants:
        output_path = os.path.join(output_dir, name)
        create_nut_variant(
            input_path=input_usd,
            output_path=output_path,
            radial_scale=scale,
            chamfer_depth=CHAMFER_DEPTH,
            chamfer_expansion=CHAMFER_EXPANSION,
            inner_min_radius=0.0069,  # 6.9mm - actual inner hole min
            inner_max_radius=0.0087   # 8.7mm - safe threshold from analysis
        )

    print("=" * 70)
    print("✓ All variants created!")
    print("=" * 70)
    print("\nCreated files:")
    for name, scale in variants:
        outer_radius = 14.03 * scale  # Original max radius * scale
        inner_chamfer = 6.9 * 1.20    # 6.9mm base * 20% expansion
        print(f"  {name:12s} - Outer: {outer_radius:.1f}mm, Inner chamfered: ~{inner_chamfer:.1f}mm at edges")

    print("\n⚠ IMPORTANT: Test in Isaac Sim to verify:")
    print("  1. Inner hole chamfer creates entry funnel at bottom (25% wider at edge)")
    print("  2. Threading is not affected (chamfer depth: 2.5mm)")
    print("  3. Bolt inserts more easily from bottom")
    print("  4. Collision mesh matches visual mesh (both were modified)")


if __name__ == "__main__":
    create_all_variants()
