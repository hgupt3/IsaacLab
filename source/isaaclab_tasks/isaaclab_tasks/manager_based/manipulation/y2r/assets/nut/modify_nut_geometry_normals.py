"""
USD Mesh Thickening - Normal-Based Vertex Displacement
Preserves surface curvature and threading detail.
"""

from pathlib import Path

from pxr import Usd, UsdGeom, Gf, Vt
import numpy as np

def compute_vertex_normals(points, face_vertex_counts, face_vertex_indices):
    """Compute smooth vertex normals from mesh topology."""
    points_np = np.array([(p[0], p[1], p[2]) for p in points])
    vertex_normals = np.zeros_like(points_np)

    # For each face, compute face normal and accumulate to vertices
    idx = 0
    for face_count in face_vertex_counts:
        if face_count == 3:  # Triangle
            v_indices = face_vertex_indices[idx:idx+3]
            v0, v1, v2 = points_np[v_indices]

            # Face normal via cross product
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm > 1e-8:
                normal /= norm
                # Accumulate to all vertices of this face
                for vi in v_indices:
                    vertex_normals[vi] += normal

        idx += face_count

    # Normalize
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    vertex_normals = np.divide(vertex_normals, norms,
                               where=norms > 1e-8, out=vertex_normals)

    return vertex_normals


def thicken_nut_normals(usd_path: str, output_path: str,
                        thickness: float = 0.003,
                        inner_radius_threshold: float = 0.008):
    """
    Thicken nut by displacing outer vertices along their surface normals.

    Args:
        usd_path: Input USD path
        output_path: Output USD path
        thickness: How much to displace (meters), default 3mm
        inner_radius_threshold: Radius below which vertices are NOT moved (protects hole)
    """
    stage = Usd.Stage.Open(usd_path)
    mesh = UsdGeom.Mesh.Get(stage, '/Root/Object/factory_nut_loose/visuals')

    # Get mesh data
    points = mesh.GetPointsAttr().Get()
    face_counts = mesh.GetFaceVertexCountsAttr().Get()
    face_indices = mesh.GetFaceVertexIndicesAttr().Get()

    print(f"Processing mesh: {len(points)} vertices, {len(face_counts)} faces")

    # Compute vertex normals
    print("Computing vertex normals...")
    normals = compute_vertex_normals(points, face_counts, face_indices)

    # Identify outer vertices (beyond inner radius)
    points_np = np.array([(p[0], p[1], p[2]) for p in points])
    xy_dists = np.linalg.norm(points_np[:, :2], axis=1)
    outer_mask = xy_dists > inner_radius_threshold

    print(f"Thickening {outer_mask.sum()} / {len(points)} vertices")
    print(f"  (Protecting {(~outer_mask).sum()} inner hole vertices)")

    # Displace outer vertices along normals
    modified = points_np.copy()
    modified[outer_mask] += normals[outer_mask] * thickness

    # Convert back to USD format
    modified_usd = Vt.Vec3fArray([Gf.Vec3f(*p) for p in modified])
    mesh.GetPointsAttr().Set(modified_usd)

    # Save
    stage.Export(output_path)
    print(f"\n✓ Saved to: {output_path}")

    # Stats
    displacements = np.linalg.norm(modified - points_np, axis=1)
    print(f"  Max displacement: {displacements.max():.4f}m")
    print(f"  Mean displacement (outer verts): {displacements[outer_mask].mean():.4f}m")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    input_usd = base_dir / "nut.usd"
    output_usd = base_dir / "nut_thick.usd"

    thicken_nut_normals(
        usd_path=str(input_usd),
        output_path=str(output_usd),
        thickness=0.003,  # 3mm thicker
        inner_radius_threshold=0.008  # Protect vertices within 8mm of center
    )

    print("\n⚠ IMPORTANT: Visually inspect the result in Isaac Sim or Blender")
    print("  Check that:")
    print("  - Inner hole is unchanged")
    print("  - No mesh artifacts or inversions")
    print("  - Threading detail is preserved")
