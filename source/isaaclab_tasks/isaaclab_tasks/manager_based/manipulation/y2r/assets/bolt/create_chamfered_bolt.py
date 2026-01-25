"""
Create a chamfered bolt by EXTENDING the tip with a tapered cone.

This adds new geometry on top of the bolt, not shrinking existing geometry.

Run via: ./isaaclab.sh -p assets/bolt/create_chamfered_bolt.py
"""

from pxr import Usd, UsdGeom, Gf, Vt
import numpy as np
import os


def extend_bolt_tip(
    mesh: UsdGeom.Mesh,
    extension_height: float = 0.004,   # 4mm extension
    tip_radius_fraction: float = 0.3,  # Tip is 30% of thread radius
    num_taper_rings: int = 8,          # Number of rings in the taper
    thread_max_radius: float = 0.010   # Only extend thread part
):
    """
    Extend the bolt tip with a tapered cone.

    This ADDS new geometry on top of the existing bolt.

    Args:
        mesh: The bolt mesh to modify
        extension_height: How much to extend upward (meters)
        tip_radius_fraction: Radius at very tip as fraction of thread radius (0.3 = 30%)
        num_taper_rings: Number of vertex rings in the tapered extension
        thread_max_radius: Only extend vertices below this radius (thread, not head)

    Returns:
        Number of vertices added
    """
    points = mesh.GetPointsAttr().Get()
    face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get()
    face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()

    points_np = np.array([(p[0], p[1], p[2]) for p in points])
    face_counts = np.array(face_vertex_counts)
    face_indices = np.array(face_vertex_indices)

    z_max = points_np[:, 2].max()
    xy_radii = np.linalg.norm(points_np[:, :2], axis=1)

    # Find the top edge vertices (thread part only)
    top_mask = (np.abs(points_np[:, 2] - z_max) < 0.0005) & (xy_radii < thread_max_radius) & (xy_radii > 0.001)
    top_indices = np.where(top_mask)[0]

    if len(top_indices) == 0:
        print("    No top edge vertices found!")
        return 0

    # Get the top edge vertices
    top_vertices = points_np[top_indices]
    top_radii = xy_radii[top_indices]

    # Sort by angle for proper ring ordering
    angles = np.arctan2(top_vertices[:, 1], top_vertices[:, 0])
    sorted_order = np.argsort(angles)
    top_indices_sorted = top_indices[sorted_order]
    top_vertices_sorted = top_vertices[sorted_order]
    top_radii_sorted = top_radii[sorted_order]

    # Get average radius of top edge
    avg_radius = top_radii_sorted.mean()
    min_radius = tip_radius_fraction * avg_radius

    print(f"    Top edge: {len(top_indices)} vertices at r={avg_radius*1000:.2f}mm avg")
    print(f"    Extension: {extension_height*1000:.1f}mm, tip radius: {min_radius*1000:.2f}mm")

    # Create new tapered rings
    new_vertices = []
    new_faces = []
    n_top = len(top_indices_sorted)
    base_new_idx = len(points_np)

    for ring in range(num_taper_rings):
        # Height fraction (0 = at original top, 1 = at new tip)
        t = (ring + 1) / num_taper_rings
        z_new = z_max + t * extension_height

        # Radius interpolation (linear taper)
        ring_radius = avg_radius * (1 - t) + min_radius * t

        # Create vertices for this ring
        for i, (orig_idx, vert) in enumerate(zip(top_indices_sorted, top_vertices_sorted)):
            # Keep same angle, new radius and height
            angle = np.arctan2(vert[1], vert[0])
            x_new = ring_radius * np.cos(angle)
            y_new = ring_radius * np.sin(angle)

            new_vertices.append([x_new, y_new, z_new])

    # Add tip vertex (single point at top)
    tip_vertex_idx = base_new_idx + len(new_vertices)
    new_vertices.append([0.0, 0.0, z_max + extension_height])

    # Create faces connecting rings
    # Connect original top to first new ring
    for i in range(n_top):
        i_next = (i + 1) % n_top
        orig_i = top_indices_sorted[i]
        orig_next = top_indices_sorted[i_next]
        new_i = base_new_idx + i
        new_next = base_new_idx + i_next

        # Quad face (as two triangles for compatibility)
        new_faces.append([orig_i, orig_next, new_next, new_i])

    # Connect between new rings
    for ring in range(num_taper_rings - 1):
        ring_base = base_new_idx + ring * n_top
        next_ring_base = base_new_idx + (ring + 1) * n_top

        for i in range(n_top):
            i_next = (i + 1) % n_top
            v0 = ring_base + i
            v1 = ring_base + i_next
            v2 = next_ring_base + i_next
            v3 = next_ring_base + i

            new_faces.append([v0, v1, v2, v3])

    # Connect last ring to tip
    last_ring_base = base_new_idx + (num_taper_rings - 1) * n_top
    for i in range(n_top):
        i_next = (i + 1) % n_top
        v0 = last_ring_base + i
        v1 = last_ring_base + i_next

        new_faces.append([v0, v1, tip_vertex_idx])  # Triangle

    # Combine old and new geometry
    new_vertices_np = np.array(new_vertices)
    all_points = np.vstack([points_np, new_vertices_np])

    # Convert new faces to flat indices and counts
    new_face_counts = []
    new_face_indices = []
    for face in new_faces:
        new_face_counts.append(len(face))
        new_face_indices.extend(face)

    all_face_counts = np.concatenate([face_counts, new_face_counts])
    all_face_indices = np.concatenate([face_indices, new_face_indices])

    # Update mesh
    mesh.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*p) for p in all_points]))
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([int(c) for c in all_face_counts]))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray([int(i) for i in all_face_indices]))

    return len(new_vertices)


def analyze_mesh(mesh: UsdGeom.Mesh, name: str):
    """Print mesh info for verification."""
    points = mesh.GetPointsAttr().Get()
    points_np = np.array([(p[0], p[1], p[2]) for p in points])

    z_min, z_max = points_np[:, 2].min(), points_np[:, 2].max()
    xy_radii = np.linalg.norm(points_np[:, :2], axis=1)

    print(f"  {name}:")
    print(f"    Vertices: {len(points_np)}")
    print(f"    Height: {z_min*1000:.2f}mm to {z_max*1000:.2f}mm ({(z_max-z_min)*1000:.2f}mm total)")

    # Show top section
    for z_offset in [0, 1, 2, 3, 4, 5]:
        z_level = z_max - z_offset * 0.001
        z_mask = np.abs(points_np[:, 2] - z_level) < 0.0005
        if z_mask.sum() > 0:
            r_at_z = xy_radii[z_mask]
            print(f"    top-{z_offset}mm: r={r_at_z.min()*1000:.2f}-{r_at_z.max()*1000:.2f}mm ({z_mask.sum()} verts)")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "bolt.usd")
    output_path = os.path.join(script_dir, "bolt_chamfered.usd")

    # Extension parameters
    EXTENSION_HEIGHT = 0.005    # 5mm extension
    TIP_RADIUS_FRACTION = 0.2   # Tip is 20% of thread radius (~1.5mm)
    NUM_TAPER_RINGS = 10        # Smooth taper with 10 rings

    print("="*70)
    print("Creating Extended Chamfered Bolt")
    print("="*70)
    print(f"\nInput:  {input_path}")
    print(f"Output: {output_path}")
    print(f"\nExtension parameters:")
    print(f"  Extension height: {EXTENSION_HEIGHT*1000:.1f}mm")
    print(f"  Tip radius: {TIP_RADIUS_FRACTION*100:.0f}% of thread radius")
    print(f"  Taper rings: {NUM_TAPER_RINGS}")

    # Open stage
    stage = Usd.Stage.Open(input_path)
    if not stage:
        print("ERROR: Could not open bolt.usd")
        return

    # Get meshes
    visual_mesh = UsdGeom.Mesh.Get(stage, '/Root/Outline/factory_bolt_loose/visuals')
    collision_mesh = UsdGeom.Mesh.Get(stage, '/Root/Outline/factory_bolt_loose/collisions')

    print("\n" + "="*70)
    print("BEFORE EXTENSION")
    print("="*70)
    analyze_mesh(visual_mesh, "Visual mesh")
    analyze_mesh(collision_mesh, "Collision mesh")

    # Apply extension
    print("\n" + "="*70)
    print("ADDING TAPERED EXTENSION")
    print("="*70)

    print("\n  Visual mesh:")
    v_count = extend_bolt_tip(visual_mesh, EXTENSION_HEIGHT, TIP_RADIUS_FRACTION, NUM_TAPER_RINGS)
    print(f"    Added {v_count} vertices")

    print("\n  Collision mesh:")
    c_count = extend_bolt_tip(collision_mesh, EXTENSION_HEIGHT, TIP_RADIUS_FRACTION, NUM_TAPER_RINGS)
    print(f"    Added {c_count} vertices")

    print("\n" + "="*70)
    print("AFTER EXTENSION")
    print("="*70)
    analyze_mesh(visual_mesh, "Visual mesh")
    analyze_mesh(collision_mesh, "Collision mesh")

    # Save
    stage.Export(output_path)
    print("\n" + "="*70)
    print(f"SAVED: {output_path}")
    print("="*70)

    print(f"""
Extension effect:
  Original height: 35.0mm
  New height: {35.0 + EXTENSION_HEIGHT*1000:.1f}mm (+{EXTENSION_HEIGHT*1000:.1f}mm)

  Original top radius: ~7.5mm
  New tip radius: ~{7.5 * TIP_RADIUS_FRACTION:.1f}mm

Visual:
    Before:         After:
       ____            /\\      <- new pointed tip
      |    |          /  \\     <- new tapered section
      |    |         |    |
      |    |         |    |    <- original geometry unchanged
      |    |         |    |
""")


if __name__ == "__main__":
    main()
