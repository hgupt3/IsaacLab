# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import hashlib
import logging
import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation
from trimesh.poses import compute_stable_poses
from trimesh.sample import sample_surface

import isaacsim.core.utils.prims as prim_utils
from pxr import Usd, UsdGeom

from isaaclab.sim.utils import get_all_matching_child_prims

# ---- module-scope caches ----
_PRIM_SAMPLE_CACHE: dict[tuple[str, int], np.ndarray] = {}  # (prim_hash, num_points) -> (N,3) in root frame
_FINAL_SAMPLE_CACHE: dict[str, np.ndarray] = {}  # env_hash -> (num_points,3) in root frame


def clear_pointcloud_caches():
    _PRIM_SAMPLE_CACHE.clear()
    _FINAL_SAMPLE_CACHE.clear()
    _POINT_CLOUD_CACHE.clear()

def _triangulate_faces(prim) -> np.ndarray:
    """Convert a USD Mesh prim into triangulated face indices (N, 3)."""
    mesh = UsdGeom.Mesh(prim)
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    faces = []
    it = iter(indices)
    for cnt in counts:
        poly = [next(it) for _ in range(cnt)]
        for k in range(1, cnt - 1):
            faces.append([poly[0], poly[k], poly[k + 1]])
    return np.asarray(faces, dtype=np.int64)


def create_primitive_mesh(prim) -> trimesh.Trimesh:
    """Create a trimesh mesh from a USD primitive (Cube, Sphere, Cylinder, etc.) or Mesh."""
    prim_type = prim.GetTypeName()
    if prim_type == "Cube":
        size = UsdGeom.Cube(prim).GetSizeAttr().Get()
        return trimesh.creation.box(extents=(size, size, size))
    elif prim_type == "Sphere":
        r = UsdGeom.Sphere(prim).GetRadiusAttr().Get()
        return trimesh.creation.icosphere(subdivisions=3, radius=r)
    elif prim_type == "Cylinder":
        c = UsdGeom.Cylinder(prim)
        return trimesh.creation.cylinder(radius=c.GetRadiusAttr().Get(), height=c.GetHeightAttr().Get())
    elif prim_type == "Capsule":
        c = UsdGeom.Capsule(prim)
        return trimesh.creation.capsule(radius=c.GetRadiusAttr().Get(), height=c.GetHeightAttr().Get())
    elif prim_type == "Cone":
        c = UsdGeom.Cone(prim)
        return trimesh.creation.cone(radius=c.GetRadiusAttr().Get(), height=c.GetHeightAttr().Get())
    elif prim_type == "Mesh":
        # Handle USD Mesh geometry
        mesh_prim = UsdGeom.Mesh(prim)
        points = np.array(mesh_prim.GetPointsAttr().Get())
        # Use existing triangulation function
        faces = _triangulate_faces(prim)
        return trimesh.Trimesh(vertices=points, faces=faces)
    else:
        raise KeyError(f"{prim_type} is not a valid primitive mesh type")


# === POINT CLOUD CACHE (built once for all envs) ===
# Key: (prim_path, pool_size) -> PointCloudCache
_POINT_CLOUD_CACHE: dict[tuple[str, int], "PointCloudCache"] = {}


class PointCloudCache:
    """Cache for point cloud sampling - built once, pure tensor indexing at runtime.
    
    Pre-computes:
    - geo_indices: env_id -> geo_idx
    - scales: env_id -> (sx, sy, sz)  
    - all_base_points: (num_geos, pool_size, 3) - stacked so we can index by geo_idx
    - all_base_normals: (num_geos, pool_size, 3) - surface normals for visibility filtering
    """
    def __init__(self, prim_path: str, num_envs: int, pool_size: int):
        from pxr import Sdf
        from isaaclab.sim.utils import get_current_stage
        
        self.prim_path = prim_path
        self.num_envs = num_envs
        self.pool_size = pool_size
        
        # Per-env data
        self.geo_indices = torch.zeros(num_envs, dtype=torch.long)
        self.scales = torch.ones(num_envs, 3, dtype=torch.float32)
        
        # Temporary for building
        geo_to_idx: dict[str, int] = {}
        geo_first_env: list[int] = []
        
        # Phase 1: USD traversal to get geo_indices and scales
        stage = get_current_stage()
        root_layer = stage.GetRootLayer()
        
        for env_id in range(num_envs):
            obj_path = prim_path.replace(".*", str(env_id))
            prims = get_all_matching_child_prims(
                obj_path, predicate=lambda p: p.GetTypeName() in ("Mesh", "Cube", "Sphere", "Cylinder", "Capsule", "Cone")
            )
            
            if not prims:
                self.geo_indices[env_id] = 0  # Fallback to first geo
                continue
            
            geo_key = _compute_geometry_group_key(prims)
            
            if geo_key not in geo_to_idx:
                geo_to_idx[geo_key] = len(geo_first_env)
                geo_first_env.append(env_id)
            
            self.geo_indices[env_id] = geo_to_idx[geo_key]
            
            prim_spec = root_layer.GetPrimAtPath(obj_path)
            if prim_spec:
                scale_attr = prim_spec.attributes.get("xformOp:scale")
                if scale_attr and scale_attr.default is not None:
                    self.scales[env_id] = torch.tensor(list(scale_attr.default), dtype=torch.float32)
        
        # Phase 2: Sample base points AND normals for each geometry and stack
        num_geos = len(geo_first_env)
        self.all_base_points = torch.zeros(num_geos, pool_size, 3, dtype=torch.float32)
        self.all_base_normals = torch.zeros(num_geos, pool_size, 3, dtype=torch.float32)
        
        for geo_idx, env_id in enumerate(geo_first_env):
            points, normals = _sample_base_points_and_normals_for_geometry(env_id, pool_size, prim_path, "cpu")
            self.all_base_points[geo_idx] = points
            self.all_base_normals[geo_idx] = normals

    def to(self, device):
        """Move all cached tensors to the given device."""
        self.all_base_points = self.all_base_points.to(device)
        self.all_base_normals = self.all_base_normals.to(device)
        self.geo_indices = self.geo_indices.to(device)
        self.scales = self.scales.to(device)
        return self

# === STABLE PLACEMENT CACHE (built once for all envs) ===
# Key: prim_path -> StablePlacementCache
_STABLE_PLACEMENT_CACHE: dict[str, "StablePlacementCache"] = {}

class StablePlacementCache:
    """Cache for stable placement data - built once, indexed many times.
    
    Pre-computes EVERYTHING so runtime is pure tensor indexing with NO loops.
    """
    def __init__(self, prim_path: str, num_envs: int):
        from pxr import Sdf
        from isaaclab.sim.utils import get_current_stage
        
        self.prim_path = prim_path
        self.num_envs = num_envs
        
        # Pre-computed per-env data (indexed by env_id) - FINAL VALUES
        self.scales_z = torch.ones(num_envs, dtype=torch.float32)
        self.z_offsets = torch.zeros(num_envs, dtype=torch.float32)  # z offset from stable pose
        self.quaternions = torch.zeros(num_envs, 4, dtype=torch.float32)
        self.quaternions[:, 0] = 1.0  # Default identity (w=1)
        self.valid = torch.zeros(num_envs, dtype=torch.bool)
        
        # Temporary: group envs by geometry for stable pose computation
        geo_to_envs: dict[str, list[int]] = {}  # geo_key -> [env_ids]
        geo_to_prim: dict[str, tuple] = {}  # geo_key -> (prim, prim_type, geom_scale_z)
        
        # Build cache - Phase 1: Group envs by geometry
        stage = get_current_stage()
        root_layer = stage.GetRootLayer()
        
        for env_id in range(num_envs):
            obj_path = prim_path.replace(".*", str(env_id))
            prims = get_all_matching_child_prims(
                obj_path, predicate=lambda p: p.GetTypeName() in ("Mesh", "Cube", "Sphere", "Cylinder", "Capsule", "Cone")
            )
            
            if not prims:
                continue
            
            geo_key = _compute_geometry_group_key(prims)
            
            if geo_key not in geo_to_envs:
                geo_to_envs[geo_key] = []
                prim = prims[0]
                prim_type = prim.GetTypeName()
                geom_scale_attr = prim.GetAttribute("xformOp:scale")
                geom_scale_val = geom_scale_attr.Get() if geom_scale_attr else None
                geom_scale_z = geom_scale_val[2] if geom_scale_val else 1.0
                geo_to_prim[geo_key] = (prim, prim_type, geom_scale_z)
            
            geo_to_envs[geo_key].append(env_id)
            
            # Read root scale
            prim_spec = root_layer.GetPrimAtPath(obj_path)
            if prim_spec:
                scale_attr = prim_spec.attributes.get("xformOp:scale")
                if scale_attr and scale_attr.default is not None:
                    self.scales_z[env_id] = float(scale_attr.default[2])
        
        # Build cache - Phase 2: Compute stable poses per geometry, assign to envs
        for geo_key, env_list in geo_to_envs.items():
            prim, prim_type, geom_scale_z = geo_to_prim[geo_key]
            
            pose_data = _get_stable_pose_data(prim, prim_type)
            if pose_data is None:
                continue
            
            transforms, probabilities, mesh_tm = pose_data
            
            # Use most stable pose (randomization happens at runtime via index selection)
            pose_idx = np.argmax(probabilities)
            stable_transform = transforms[pose_idx]
            rotation_matrix = stable_transform[:3, :3]
            
            # Quaternion
            quat = Rotation.from_matrix(rotation_matrix).as_quat()  # [x,y,z,w]
            quat_wxyz = torch.tensor([quat[3], quat[0], quat[1], quat[2]], dtype=torch.float32)
            
            # Transformed mesh for z-offset
            transformed_mesh = mesh_tm.copy()
            transformed_mesh.apply_transform(stable_transform)
            z_center = (transformed_mesh.bounds[0, 2] + transformed_mesh.bounds[1, 2]) / 2
            z_bottom = transformed_mesh.bounds[0, 2]
            z_offset_unscaled = z_center - z_bottom
            
            # Assign to all envs with this geometry (vectorized)
            env_tensor = torch.tensor(env_list, dtype=torch.long)
            self.z_offsets[env_tensor] = z_offset_unscaled * geom_scale_z
            self.quaternions[env_tensor] = quat_wxyz
            self.valid[env_tensor] = True


def read_object_scales_from_usd(prim_path: str, num_envs: int) -> torch.Tensor:
    """Read per-environment object scales from USD prims.
    
    This reads the xformOp:scale attribute that was set by randomize_rigid_body_scale
    during prestartup. Scales are stored on USD prims and don't change during simulation.
    
    Args:
        prim_path: Prim path pattern with .* for env_id (e.g., "/World/envs/env_.*/Object")
        num_envs: Number of environments
        
    Returns:
        Tensor of shape (num_envs,) with z-scale for each environment
    """
    from pxr import Sdf
    from isaaclab.sim.utils import get_current_stage
    
    scales_z = torch.ones(num_envs, dtype=torch.float32)
    
    stage = get_current_stage()
    root_layer = stage.GetRootLayer()
    
    for env_id in range(num_envs):
        obj_path = prim_path.replace(".*", str(env_id))
        prim_spec = root_layer.GetPrimAtPath(obj_path)
        if prim_spec:
            scale_attr = prim_spec.attributes.get("xformOp:scale")
            if scale_attr and scale_attr.default is not None:
                scales_z[env_id] = float(scale_attr.default[2])
    
    return scales_z


def clear_random_pointcloud_cache():
    """Clear the random point cloud caches."""
    _POINT_CLOUD_CACHE.clear()
    _STABLE_PLACEMENT_CACHE.clear()


def _compute_geometry_group_key(prims: list) -> str:
    """Compute a key for grouping envs by geometry type (shape + dimensions, excluding scale).
    
    For Mesh types, includes bounding box dimensions to distinguish meshes with the same
    vertex/face count but different sizes (e.g., different procedural shapes).
    """
    parts = []
    for prim in prims:
        prim_type = prim.GetTypeName()
        if prim_type == "Mesh":
            mesh = UsdGeom.Mesh(prim)
            verts = mesh.GetPointsAttr().Get()
            faces = mesh.GetFaceVertexIndicesAttr().Get()
            # Include bounding box to distinguish meshes with same vert/face count but different sizes
            verts_np = np.asarray(verts, dtype=np.float32)
            bbox_min = tuple(np.round(verts_np.min(axis=0), 4))
            bbox_max = tuple(np.round(verts_np.max(axis=0), 4))
            parts.append(f"Mesh:{len(verts)}:{len(faces)}:{bbox_min}:{bbox_max}")
        elif prim_type == "Cube":
            size = UsdGeom.Cube(prim).GetSizeAttr().Get()
            parts.append(f"Cube:{size:.6f}")
        elif prim_type == "Sphere":
            r = UsdGeom.Sphere(prim).GetRadiusAttr().Get()
            parts.append(f"Sphere:{r:.6f}")
        elif prim_type == "Cylinder":
            c = UsdGeom.Cylinder(prim)
            parts.append(f"Cylinder:{c.GetRadiusAttr().Get():.6f}:{c.GetHeightAttr().Get():.6f}")
        elif prim_type == "Capsule":
            c = UsdGeom.Capsule(prim)
            parts.append(f"Capsule:{c.GetRadiusAttr().Get():.6f}:{c.GetHeightAttr().Get():.6f}")
        elif prim_type == "Cone":
            c = UsdGeom.Cone(prim)
            parts.append(f"Cone:{c.GetRadiusAttr().Get():.6f}:{c.GetHeightAttr().Get():.6f}")
    return "|".join(sorted(parts))


def _sample_base_points_and_normals_for_geometry(
    env_id: int,
    pool_size: int,
    prim_path: str,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample an unscaled pool of points AND surface normals for an environment's geometry.
    
    Returns:
        Tuple of (points, normals) each with shape (pool_size, 3).
        Points and normals are in object-local frame (without scale).
    """
    obj_path = prim_path.replace(".*", str(env_id))
    xform_cache = UsdGeom.XformCache()
    
    prims = get_all_matching_child_prims(
        obj_path, predicate=lambda p: p.GetTypeName() in ("Mesh", "Cube", "Sphere", "Cylinder", "Capsule", "Cone")
    )
    if not prims:
        raise KeyError(f"No valid prims under {obj_path}")

    object_prim = prim_utils.get_prim_at_path(obj_path)
    world_root = xform_cache.GetLocalToWorldTransform(object_prim)

    all_samples: list[torch.Tensor] = []
    all_normals: list[torch.Tensor] = []
    
    for prim in prims:
        prim_type = prim.GetTypeName()

        # Create trimesh
        if prim_type == "Mesh":
            mesh = UsdGeom.Mesh(prim)
            verts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
            faces = _triangulate_faces(prim)
            mesh_tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        else:
            mesh_tm = create_primitive_mesh(prim)

        # Random surface sampling (area-weighted) - get face indices for normals
        face_weights = mesh_tm.area_faces
        samples_np, face_indices = sample_surface(mesh_tm, pool_size, face_weight=face_weights)
        local_pts = torch.from_numpy(samples_np.astype(np.float32)).to(device)
        
        # Get normals from face indices
        local_normals_np = mesh_tm.face_normals[face_indices]
        local_normals = torch.from_numpy(local_normals_np.astype(np.float32)).to(device)

        # Transform from prim local frame to object root frame
        rel = xform_cache.GetLocalToWorldTransform(prim) * world_root.GetInverse()
        mat_np = np.array([[rel[r][c] for c in range(4)] for r in range(4)], dtype=np.float32)
        mat_t = torch.from_numpy(mat_np).to(device)
        
        # Extract rotation matrix (3x3) for normals
        rot_mat = mat_t[:3, :3]

        # Transform points (position + rotation)
        ones = torch.ones((pool_size, 1), device=device)
        pts_h = torch.cat([local_pts, ones], dim=1)
        root_h = pts_h @ mat_t
        samples = root_h[:, :3]
        
        # Transform normals (rotation only, then normalize)
        normals = local_normals @ rot_mat.T
        normals = normals / (normals.norm(dim=-1, keepdim=True) + 1e-8)

        # Cone height adjustment
        if prim_type == "Cone":
            samples[:, 2] -= UsdGeom.Cone(prim).GetHeightAttr().Get() / 2

        all_samples.append(samples)
        all_normals.append(normals)

    # Combine samples
    if len(all_samples) == 1:
        points = all_samples[0][:pool_size]
        normals = all_normals[0][:pool_size]
    else:
        combined_pts = torch.cat(all_samples, dim=0)
        combined_norms = torch.cat(all_normals, dim=0)
        perm = torch.randperm(combined_pts.shape[0], device=device)[:pool_size]
        points = combined_pts[perm]
        normals = combined_norms[perm]
    
    # Points and normals are now in object-local space at geometry_size (WITHOUT object's scale)
    return points, normals


def sample_object_point_cloud_random(
    env_ids: list[int],
    num_points: int,
    prim_path: str,
    device: str = "cpu",
    pool_size: int | None = None,
    filter_config: dict | None = None,
) -> torch.Tensor:
    """
    Sample random points on object surfaces for specified environments.

    OPTIMIZED: Pure tensor indexing, NO LOOPS. Cache built once on first call.

    Args:
        env_ids: List of environment indices to sample for.
        num_points: Number of points to sample per environment.
        prim_path: USD prim path pattern with ".*" placeholder for env index.
        device: Device to place the output tensor on.
        pool_size: Size of point pool to sample from. If None, uses 4x num_points.
        filter_config: Optional dict with keys:
                      - "enabled" (bool): Enable filtering
                      - "type" (str): "axis" or "radial" (default: "axis")
                      For axis type: "axis" ("x"/"y"/"z"), "min" (float|None), "max" (float|None)
                      For radial type: "origin" ([x,y,z]), "min_radius" (float|None), "max_radius" (float|None)

    Returns:
        torch.Tensor: Shape (len(env_ids), num_points, 3) on `device`.
    """
    global _POINT_CLOUD_CACHE
    
    # Default pool size is 4x num_points to ensure enough points for visibility filtering
    if pool_size is None:
        pool_size = max(256, num_points * 4)
    pool_size = max(pool_size, num_points)
    cache_key = (prim_path, pool_size)
    
    # === Build cache on first call (one-time cost) ===
    if cache_key not in _POINT_CLOUD_CACHE:
        num_envs = max(env_ids) + 1
        _POINT_CLOUD_CACHE[cache_key] = PointCloudCache(prim_path, num_envs, pool_size)
        _POINT_CLOUD_CACHE[cache_key].to(device)

    cache = _POINT_CLOUD_CACHE[cache_key]
    n_envs = len(env_ids)

    # === Pure tensor indexing - NO LOOPS ===
    env_ids_t = torch.tensor(env_ids, dtype=torch.long, device=device)
    geo_indices = cache.geo_indices[env_ids_t]  # (n,)
    scales = cache.scales[env_ids_t]  # (n, 3)

    # Get base points for each env by geo_idx indexing: (n, pool_size, 3)
    all_points = cache.all_base_points  # (num_geos, pool_size, 3) - already on device
    env_base_points = all_points[geo_indices]  # (n, pool_size, 3)

    # Apply per-env scale BEFORE filtering: (n, pool_size, 3)
    env_scaled_points = scales.unsqueeze(1) * env_base_points  # (n, pool_size, 3)

    # Apply regional filter if specified
    if filter_config and filter_config.get("enabled", False):
        filter_type = filter_config.get("type", "axis")  # Default to axis for backward compat

        if filter_type == "radial":
            # Radial distance filter from origin point
            origin = filter_config.get("origin", [0, 0, 0])
            origin_t = torch.tensor(origin, dtype=torch.float32, device=device)  # (3,)

            # Compute distance from origin for all points: (n, pool_size)
            distances = torch.norm(env_scaled_points - origin_t, dim=2)

            # Create mask for valid points based on radius bounds
            valid_mask = torch.ones(n_envs, pool_size, dtype=torch.bool, device=device)
            min_radius = filter_config.get("min_radius")
            max_radius = filter_config.get("max_radius")

            if min_radius is not None:
                valid_mask &= distances >= min_radius
            if max_radius is not None:
                valid_mask &= distances <= max_radius

        else:  # axis filter (existing logic)
            axis_map = {"x": 0, "y": 1, "z": 2}
            axis = axis_map.get(filter_config.get("axis", "z"), 2)
            axis_min = filter_config.get("min")
            axis_max = filter_config.get("max")

            # Create mask for valid points: (n, pool_size)
            valid_mask = torch.ones(n_envs, pool_size, dtype=torch.bool, device=device)
            if axis_min is not None:
                valid_mask &= env_scaled_points[:, :, axis] >= axis_min
            if axis_max is not None:
                valid_mask &= env_scaled_points[:, :, axis] <= axis_max

        # Use valid mask to bias random sampling (invalid points get -inf random value)
        rand_vals = torch.rand(n_envs, pool_size, device=device)
        rand_vals[~valid_mask] = -1.0  # Invalid points won't be selected
    else:
        # No filter - standard random sampling
        rand_vals = torch.rand(n_envs, pool_size, device=device)

    # Random subsample: argsort gives us random permutations (valid points first)
    rand_indices = rand_vals.argsort(dim=1, descending=True)[:, :num_points]  # (n, num_points)

    # Gather random points: (n, num_points, 3)
    batch_idx = torch.arange(n_envs, device=device).unsqueeze(1).expand(-1, num_points)
    points = env_scaled_points[batch_idx, rand_indices]  # (n, num_points, 3)

    return points


def get_point_cloud_cache(
    env_ids: list[int],
    prim_path: str,
    pool_size: int,
    device: str | None = None,
) -> PointCloudCache:
    """Get or create a PointCloudCache for the given prim path and pool size.

    Args:
        env_ids: List of environment indices.
        prim_path: USD prim path pattern with ".*" placeholder for env index.
        pool_size: Size of point pool.
        device: If given, move cache tensors to this device.

    Returns:
        PointCloudCache with points and normals.
    """
    global _POINT_CLOUD_CACHE

    cache_key = (prim_path, pool_size)

    if cache_key not in _POINT_CLOUD_CACHE:
        num_envs = max(env_ids) + 1
        cache = PointCloudCache(prim_path, num_envs, pool_size)
        if device is not None:
            cache.to(device)
        _POINT_CLOUD_CACHE[cache_key] = cache

    cache = _POINT_CLOUD_CACHE[cache_key]
    if device is not None and cache.all_base_points.device != torch.device(device):
        cache.to(device)

    return cache


# ---- Stable Pose Computation ----

# Cache for stable poses and aligned mesh per object geometry (to avoid recomputation)
# Key: geometry hash -> (transforms, probabilities, aligned_mesh)
_STABLE_POSE_CACHE: dict[str, tuple[np.ndarray, np.ndarray, trimesh.Trimesh]] = {}


def clear_stable_pose_cache():
    """Clear the stable pose cache."""
    _STABLE_POSE_CACHE.clear()
    _STABLE_PLACEMENT_CACHE.clear()


def _get_stable_pose_data(prim, prim_type: str) -> tuple[np.ndarray, np.ndarray, trimesh.Trimesh] | None:
    """Get stable pose transforms, probabilities, and mesh for a geometry prim."""
    cache_key = _compute_geometry_hash(prim, prim_type)
    
    if cache_key in _STABLE_POSE_CACHE:
        return _STABLE_POSE_CACHE[cache_key]
    
    # Get bounding box
    imageable = UsdGeom.Imageable(prim)
    bbox = imageable.ComputeLocalBound(Usd.TimeCode.Default(), purpose1="default")
    bbox_range = bbox.GetRange()
    usd_min = np.array(bbox_range.GetMin())
    usd_max = np.array(bbox_range.GetMax())
    
    # Create trimesh
    mesh_tm = create_primitive_mesh(prim)
    
    # Align centers
    tm_center = (mesh_tm.bounds[0] + mesh_tm.bounds[1]) / 2
    usd_center = (usd_min + usd_max) / 2
    offset = usd_center - tm_center
    mesh_tm.apply_translation(offset)

    # Compute stable poses
    try:
        transforms, probabilities = compute_stable_poses(mesh_tm)
    except Exception as e:
        logging.warning(f"Failed to compute stable poses: {e}")
        return None

    if len(transforms) == 0:
        return None
        
    _STABLE_POSE_CACHE[cache_key] = (transforms, probabilities, mesh_tm)
    return transforms, probabilities, mesh_tm


def get_stable_object_placement(
    env_ids: list[int],
    prim_path: str,
    table_surface_z: float = 0.255,
    randomize_pose: bool = True,
    z_offset: float = 0.005,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute stable z-positions and orientations for placing objects on table.
    
    OPTIMIZED: Uses global cache built once on first call. Subsequent calls
    just index into cached data - no USD traversal or for loops.

    Args:
        env_ids: List of environment indices to compute placements for.
        prim_path: USD prim path pattern with ".*" placeholder for env index.
        table_surface_z: Z-coordinate of the table surface.
        randomize_pose: If True, randomly sample from stable poses weighted by
                       probability. If False, always use the most stable pose.
        z_offset: Safety margin above table to prevent physics depenetration (meters).

    Returns:
        Tuple of:
            - z_positions: (len(env_ids),) tensor of z-coordinates for object centers
            - quaternions: (len(env_ids), 4) tensor of quaternions (w, x, y, z)
    """
    global _STABLE_PLACEMENT_CACHE
    
    # === Build cache on first call (one-time cost) ===
    if prim_path not in _STABLE_PLACEMENT_CACHE:
        # Total envs = max env_id + 1 (env_ids are 0-indexed)
        num_envs = max(env_ids) + 1
        _STABLE_PLACEMENT_CACHE[prim_path] = StablePlacementCache(prim_path, num_envs)
    
    cache = _STABLE_PLACEMENT_CACHE[prim_path]
    
    # === Pure tensor indexing - NO LOOPS! ===
    env_ids_t = torch.tensor(env_ids, dtype=torch.long)
    
    # Index into pre-computed cache
    scales_z = cache.scales_z[env_ids_t]
    z_offsets = cache.z_offsets[env_ids_t]
    quaternions = cache.quaternions[env_ids_t].clone()
    valid = cache.valid[env_ids_t]
    
    # Compute final z positions: table + (z_offset * scale) + safety margin
    z_positions = table_surface_z + (z_offsets * scales_z) + z_offset
    
    # Apply fallback for invalid envs
    z_positions = torch.where(valid, z_positions, torch.full_like(z_positions, table_surface_z + 0.1))
    
    # Safety clamp: ensure z-position is never below table surface + minimum clearance
    # This protects against any numerical errors or edge cases in stable pose computation
    min_z = table_surface_z + max(z_offset, 0.003)  # At least 3mm above table
    z_positions = torch.maximum(z_positions, torch.full_like(z_positions, min_z))
    
    return z_positions, quaternions


def compute_z_offset_from_usd(
    usd_path: str,
    rotation_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    scale: float = 1.0,
) -> float:
    """
    Compute the z-offset (origin-to-bottom distance) for a USD mesh in a given orientation.
    
    Uses USD's built-in BBoxCache on every boundable prim, unions the bounds in
    world space, applies rotation + scale, and returns origin->bottom distance.
    
    Args:
        usd_path: Path to the USD file.
        rotation_wxyz: Quaternion (w, x, y, z) for the desired orientation.
        scale: Uniform scale factor to apply.
        
    Returns:
        Distance from object origin to its lowest point after rotation and scale.
        Add this to table_surface_z + safety_margin to get correct spawn z.
    """
    from pxr import Usd, UsdGeom
    from scipy.spatial.transform import Rotation
    
    stage = Usd.Stage.Open(str(usd_path))
    assert stage is not None, f"Failed to open USD stage: {usd_path}"
    
    # Include common purposes for bbox computation
    purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy]
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes)
    
    # Collect bboxes from all boundable prims
    mins = []
    maxs = []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Boundable):
            world_bbox = bbox_cache.ComputeWorldBound(prim)
            aligned = world_bbox.ComputeAlignedBox()
            mn = aligned.GetMin()
            mx = aligned.GetMax()
            mins.append(np.array([mn[0], mn[1], mn[2]]))
            maxs.append(np.array([mx[0], mx[1], mx[2]]))
    
    assert len(mins) > 0, f"No boundable prims found in USD: {usd_path}"
    
    # Union all bboxes
    min_pt = np.min(np.stack(mins, axis=0), axis=0)
    max_pt = np.max(np.stack(maxs, axis=0), axis=0)
    
    # Build 8 corners for rotation
    corners = np.array([
        [min_pt[0], min_pt[1], min_pt[2]],
        [min_pt[0], min_pt[1], max_pt[2]],
        [min_pt[0], max_pt[1], min_pt[2]],
        [min_pt[0], max_pt[1], max_pt[2]],
        [max_pt[0], min_pt[1], min_pt[2]],
        [max_pt[0], min_pt[1], max_pt[2]],
        [max_pt[0], max_pt[1], min_pt[2]],
        [max_pt[0], max_pt[1], max_pt[2]],
    ])
    
    # Apply rotation (convert wxyz to xyzw for scipy)
    quat_xyzw = [rotation_wxyz[1], rotation_wxyz[2], rotation_wxyz[3], rotation_wxyz[0]]
    rotation = Rotation.from_quat(quat_xyzw)
    rotated = rotation.apply(corners)
    
    # Apply scale
    scaled = rotated * scale
    
    # Origin-to-bottom distance = -z_min (z_min is the lowest point relative to origin)
    z_min = scaled[:, 2].min()
    return float(-z_min)


def _compute_geometry_hash(prim, prim_type: str) -> str:
    """Compute a hash for a geometry prim to use as cache key."""
    hasher = hashlib.sha256()
    hasher.update(prim_type.encode())

    if prim_type == "Mesh":
        mesh = UsdGeom.Mesh(prim)
        verts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
        hasher.update(verts.tobytes())
    elif prim_type == "Cube":
        size = UsdGeom.Cube(prim).GetSizeAttr().Get()
        hasher.update(np.float32(size).tobytes())
    elif prim_type == "Sphere":
        r = UsdGeom.Sphere(prim).GetRadiusAttr().Get()
        hasher.update(np.float32(r).tobytes())
    elif prim_type == "Cylinder":
        c = UsdGeom.Cylinder(prim)
        hasher.update(np.float32(c.GetRadiusAttr().Get()).tobytes())
        hasher.update(np.float32(c.GetHeightAttr().Get()).tobytes())
    elif prim_type == "Capsule":
        c = UsdGeom.Capsule(prim)
        hasher.update(np.float32(c.GetRadiusAttr().Get()).tobytes())
        hasher.update(np.float32(c.GetHeightAttr().Get()).tobytes())
    elif prim_type == "Cone":
        c = UsdGeom.Cone(prim)
        hasher.update(np.float32(c.GetRadiusAttr().Get()).tobytes())
        hasher.update(np.float32(c.GetHeightAttr().Get()).tobytes())

    return hasher.hexdigest()
