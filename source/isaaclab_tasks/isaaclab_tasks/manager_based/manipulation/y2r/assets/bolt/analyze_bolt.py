"""
Analyze the local bolt geometry.
"""

from pxr import Usd, UsdGeom
import numpy as np
import os

def analyze_bolt():
    bolt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bolt.usd")
    print(f"Loading bolt from: {bolt_path}")

    stage = Usd.Stage.Open(bolt_path)
    if not stage:
        print("ERROR: Could not open bolt USD")
        return

    print("\n" + "="*70)
    print("BOLT STRUCTURE")
    print("="*70)

    # List all prims
    print("\nPrim hierarchy:")
    for prim in stage.Traverse():
        indent = "  " * (len(str(prim.GetPath()).split("/")) - 1)
        prim_type = prim.GetTypeName()
        print(f"{indent}{prim.GetPath()} [{prim_type}]")

    # Find all meshes
    print("\n" + "="*70)
    print("MESH ANALYSIS")
    print("="*70)

    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            points = mesh.GetPointsAttr().Get()

            if points is None or len(points) == 0:
                print(f"\n{prim.GetPath()}: No points")
                continue

            points_np = np.array([(p[0], p[1], p[2]) for p in points])

            print(f"\n{prim.GetPath()}:")
            print(f"  Vertices: {len(points_np)}")

            # Bounds
            x_min, x_max = points_np[:, 0].min(), points_np[:, 0].max()
            y_min, y_max = points_np[:, 1].min(), points_np[:, 1].max()
            z_min, z_max = points_np[:, 2].min(), points_np[:, 2].max()

            print(f"  X range: {x_min*1000:.2f}mm to {x_max*1000:.2f}mm (width: {(x_max-x_min)*1000:.2f}mm)")
            print(f"  Y range: {y_min*1000:.2f}mm to {y_max*1000:.2f}mm (depth: {(y_max-y_min)*1000:.2f}mm)")
            print(f"  Z range: {z_min*1000:.2f}mm to {z_max*1000:.2f}mm (height: {(z_max-z_min)*1000:.2f}mm)")

            # Radial analysis (assuming bolt is cylindrical along Z)
            xy_radii = np.linalg.norm(points_np[:, :2], axis=1)
            print(f"  Radial range: {xy_radii.min()*1000:.2f}mm to {xy_radii.max()*1000:.2f}mm")

            # Analyze TOP section (where we'd add chamfer for insertion)
            print(f"\n  TOP SECTION (tip - where chamfer will go):")
            for z_offset in [0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5]:
                z_level = z_max - z_offset * 0.001  # Every 0.5-1mm from top
                z_mask = np.abs(points_np[:, 2] - z_level) < 0.0003
                if z_mask.sum() > 0:
                    radii_at_z = xy_radii[z_mask]
                    print(f"    Z={z_level*1000:.1f}mm (tip-{z_offset}mm): {z_mask.sum():3d} verts, r={radii_at_z.min()*1000:.2f}-{radii_at_z.max()*1000:.2f}mm")

            # Identify thread vs head
            print(f"\n  RADIUS DISTRIBUTION:")
            for r_threshold in [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]:
                r_mask = (xy_radii >= r_threshold/1000) & (xy_radii < (r_threshold+1)/1000)
                if r_mask.sum() > 0:
                    print(f"    r={r_threshold}-{r_threshold+1}mm: {r_mask.sum()} vertices")


if __name__ == "__main__":
    analyze_bolt()
