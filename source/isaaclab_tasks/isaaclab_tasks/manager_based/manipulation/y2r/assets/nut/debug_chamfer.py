"""
Debug: Why isn't the chamfer visible?
"""

from pxr import Usd, UsdGeom
import numpy as np
import os

def debug():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    orig_path = os.path.join(base_dir, "nut.usd")
    var_path = os.path.join(base_dir, "nut0.usd")

    orig_stage = Usd.Stage.Open(orig_path)
    var_stage = Usd.Stage.Open(var_path)

    # Check VISUAL mesh (what you actually see)
    orig_mesh = UsdGeom.Mesh.Get(orig_stage, '/Root/Object/factory_nut_loose/visuals')
    var_mesh = UsdGeom.Mesh.Get(var_stage, '/Root/Object/factory_nut_loose/visuals')

    orig_points = np.array([(p[0], p[1], p[2]) for p in orig_mesh.GetPointsAttr().Get()])
    var_points = np.array([(p[0], p[1], p[2]) for p in var_mesh.GetPointsAttr().Get()])

    z_min = orig_points[:, 2].min()
    z_max = orig_points[:, 2].max()
    height = z_max - z_min

    print("="*70)
    print("CHAMFER DEBUG")
    print("="*70)
    print(f"\nNut dimensions:")
    print(f"  Z range: {z_min*1000:.2f}mm to {z_max*1000:.2f}mm")
    print(f"  Height: {height*1000:.2f}mm")
    print(f"  Chamfer depth: 2.5mm = {2.5/height/10:.1f}% of height")

    # Check if ANY vertices changed
    diff = np.linalg.norm(var_points - orig_points, axis=1)
    changed_mask = diff > 1e-6
    print(f"\nVertices changed: {changed_mask.sum()} / {len(orig_points)}")

    if changed_mask.sum() > 0:
        print(f"  Max displacement: {diff.max()*1000:.3f}mm")

        # Where are the changed vertices?
        changed_z = orig_points[changed_mask, 2]
        print(f"  Changed vertices Z range: {changed_z.min()*1000:.2f}mm to {changed_z.max()*1000:.2f}mm")

        # How much did they move radially?
        orig_r = np.linalg.norm(orig_points[changed_mask, :2], axis=1)
        var_r = np.linalg.norm(var_points[changed_mask, :2], axis=1)
        radial_change = var_r - orig_r
        print(f"  Radial change: {radial_change.min()*1000:.3f}mm to {radial_change.max()*1000:.3f}mm")

    # Check the INNER HOLE specifically at different Z levels
    orig_xy_r = np.linalg.norm(orig_points[:, :2], axis=1)
    var_xy_r = np.linalg.norm(var_points[:, :2], axis=1)

    INNER_MAX = 0.0087  # 8.7mm threshold from script

    print(f"\n" + "="*70)
    print("INNER HOLE PROFILE (minimum radius at each Z)")
    print("="*70)

    # Sample Z levels
    z_samples = np.linspace(z_min, z_max, 20)

    print(f"\n{'Z (mm)':>10} | {'Orig Min R':>12} | {'New Min R':>12} | {'Change':>10}")
    print("-"*50)

    for z in z_samples:
        z_mask = np.abs(orig_points[:, 2] - z) < 0.0005  # 0.5mm tolerance
        inner_mask = orig_xy_r < INNER_MAX
        combined = z_mask & inner_mask

        if combined.sum() > 0:
            orig_min = orig_xy_r[combined].min()
            var_min = var_xy_r[combined].min()
            change = (var_min - orig_min) * 1000

            marker = ""
            if z - z_min < 0.0025:
                marker = " <- CHAMFER ZONE"

            print(f"{z*1000:>10.2f} | {orig_min*1000:>12.2f} | {var_min*1000:>12.2f} | {change:>+9.2f}mm{marker}")

    # THE KEY QUESTION: What vertices are at the very bottom?
    print(f"\n" + "="*70)
    print("BOTTOM EDGE ANALYSIS")
    print("="*70)

    bottom_mask = orig_points[:, 2] < z_min + 0.001  # Within 1mm of bottom
    print(f"\nVertices within 1mm of bottom: {bottom_mask.sum()}")

    if bottom_mask.sum() > 0:
        bottom_r = orig_xy_r[bottom_mask]
        print(f"  Radii range: {bottom_r.min()*1000:.2f}mm to {bottom_r.max()*1000:.2f}mm")

        # How many are "inner" vs "outer"?
        inner_bottom = bottom_mask & (orig_xy_r < INNER_MAX)
        outer_bottom = bottom_mask & (orig_xy_r >= INNER_MAX)
        print(f"  Inner (r < {INNER_MAX*1000:.1f}mm): {inner_bottom.sum()}")
        print(f"  Outer (r >= {INNER_MAX*1000:.1f}mm): {outer_bottom.sum()}")

        # Check if inner bottom vertices actually changed
        if inner_bottom.sum() > 0:
            inner_diff = diff[inner_bottom]
            print(f"\n  Inner bottom vertex changes:")
            print(f"    Changed: {(inner_diff > 1e-6).sum()} / {inner_bottom.sum()}")
            print(f"    Max change: {inner_diff.max()*1000:.3f}mm")

            # Show actual expansion
            orig_inner_r = orig_xy_r[inner_bottom]
            var_inner_r = var_xy_r[inner_bottom]
            expansion = (var_inner_r / orig_inner_r - 1) * 100
            print(f"    Expansion range: {expansion.min():.1f}% to {expansion.max():.1f}%")


if __name__ == "__main__":
    debug()
