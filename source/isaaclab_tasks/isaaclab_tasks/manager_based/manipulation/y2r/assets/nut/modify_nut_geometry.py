"""
USD Mesh Geometry Editor - Thicken Nut Outer Surface
WARNING: This is a template - USD mesh editing is complex and may require
adjustments based on your specific mesh structure.
"""

from pxr import Usd, UsdGeom, Gf
import numpy as np

def thicken_nut_radially(usd_path: str, output_path: str, thickness_increase: float = 0.005):
    """
    Thickens the outer surface of a nut mesh by moving vertices radially outward.

    Args:
        usd_path: Path to input USD file
        output_path: Path to save modified USD
        thickness_increase: How much to thicken (meters) - default 5mm
    """
    # Open USD stage
    stage = Usd.Stage.Open(usd_path)

    # Find the mesh prim (you may need to adjust the path)
    # Common patterns: /Nut, /nut, /Mesh, /geometry
    mesh_prim = None
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            mesh_prim = UsdGeom.Mesh(prim)
            print(f"Found mesh: {prim.GetPath()}")
            break

    if not mesh_prim:
        raise ValueError("No mesh found in USD file")

    # Get vertices
    points_attr = mesh_prim.GetPointsAttr()
    points = np.array(points_attr.Get())

    print(f"Original mesh: {len(points)} vertices")

    # Identify hole center (assuming nut is centered at origin in XY)
    # Adjust if your nut has different orientation
    hole_center = np.array([0.0, 0.0])  # XY center

    # For each vertex, compute distance from hole center in XY plane
    xy_distances = np.linalg.norm(points[:, :2] - hole_center, axis=1)

    # Find inner hole radius (assume innermost vertices define the hole)
    inner_radius = np.percentile(xy_distances, 10)  # Bottom 10% are hole vertices

    print(f"Detected inner hole radius: {inner_radius:.4f}m")

    # Thicken only outer vertices (beyond hole + some margin)
    margin = 0.002  # 2mm safety margin to avoid affecting hole
    outer_mask = xy_distances > (inner_radius + margin)

    print(f"Thickening {outer_mask.sum()} outer vertices (leaving {(~outer_mask).sum()} inner vertices)")

    # Move outer vertices radially outward in XY plane
    modified_points = points.copy()
    for i in np.where(outer_mask)[0]:
        xy = points[i, :2]
        direction = (xy - hole_center) / np.linalg.norm(xy - hole_center)
        modified_points[i, :2] += direction * thickness_increase

    # Update mesh
    points_attr.Set(Gf.Vec3fArray.FromNumpy(modified_points))

    # Save modified stage
    stage.Export(output_path)
    print(f"Saved thickened nut to: {output_path}")

    # Print summary
    max_change = np.max(np.linalg.norm(modified_points - points, axis=1))
    print(f"Maximum vertex displacement: {max_change:.4f}m")


if __name__ == "__main__":
    # Example usage
    input_usd = "/home/harsh/y2r/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/y2r/assets/nut/nut.usd"
    output_usd = "/home/harsh/y2r/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/y2r/assets/nut/nut_thick.usd"

    # Thicken by 3mm on outer surface
    thicken_nut_radially(
        usd_path=input_usd,
        output_path=output_usd,
        thickness_increase=0.003  # 3mm in meters
    )

    print("\n✓ Done! Test the modified nut in simulation.")
    print("  If hole is affected, increase the 'margin' parameter.")
