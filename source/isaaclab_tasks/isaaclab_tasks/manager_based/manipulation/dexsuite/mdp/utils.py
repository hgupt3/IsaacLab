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


def sample_object_point_cloud(num_envs: int, num_points: int, prim_path: str, device: str = "cpu") -> torch.Tensor:
    """
    Samples point clouds for each environment instance by collecting points
    from all matching USD prims under `prim_path`, then downsamples to
    exactly `num_points` per env using farthest-point sampling.

    Caching is in-memory within this module:
      - per-prim raw samples:   _PRIM_SAMPLE_CACHE[(prim_hash, num_points)]
      - final downsampled env:  _FINAL_SAMPLE_CACHE[env_hash]

    Returns:
        torch.Tensor: Shape (num_envs, num_points, 3) on `device`.
    """
    points = torch.zeros((num_envs, num_points, 3), dtype=torch.float32, device=device)
    xform_cache = UsdGeom.XformCache()

    for i in range(num_envs):
        # Resolve prim path
        obj_path = prim_path.replace(".*", str(i))

        # Gather prims
        prims = get_all_matching_child_prims(
            obj_path, predicate=lambda p: p.GetTypeName() in ("Mesh", "Cube", "Sphere", "Cylinder", "Capsule", "Cone")
        )
        if not prims:
            raise KeyError(f"No valid prims under {obj_path}")

        object_prim = prim_utils.get_prim_at_path(obj_path)
        world_root = xform_cache.GetLocalToWorldTransform(object_prim)

        # hash each child prim by its rel transform + geometry
        prim_hashes = []
        for prim in prims:
            prim_type = prim.GetTypeName()
            hasher = hashlib.sha256()

            rel = world_root.GetInverse() * xform_cache.GetLocalToWorldTransform(prim)  # prim -> root
            mat_np = np.array([[rel[r][c] for c in range(4)] for r in range(4)], dtype=np.float32)
            hasher.update(mat_np.tobytes())

            if prim_type == "Mesh":
                mesh = UsdGeom.Mesh(prim)
                verts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
                hasher.update(verts.tobytes())
            else:
                if prim_type == "Cube":
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

            prim_hashes.append(hasher.hexdigest())

        # scale on root (default to 1 if missing)
        attr = object_prim.GetAttribute("xformOp:scale")
        scale_val = attr.Get() if attr else None
        if scale_val is None:
            base_scale = torch.ones(3, dtype=torch.float32, device=device)
        else:
            base_scale = torch.tensor(scale_val, dtype=torch.float32, device=device)

        # env-level cache key (includes num_points)
        env_key = "_".join(sorted(prim_hashes)) + f"_{num_points}"
        env_hash = hashlib.sha256(env_key.encode()).hexdigest()

        # load from env-level in-memory cache
        if env_hash in _FINAL_SAMPLE_CACHE:
            arr = _FINAL_SAMPLE_CACHE[env_hash]  # (num_points,3) in root frame
            points[i] = torch.from_numpy(arr).to(device) * base_scale.unsqueeze(0)
            continue

        # otherwise build per-prim samples (with per-prim cache)
        all_samples_np: list[np.ndarray] = []
        for prim, ph in zip(prims, prim_hashes):
            key = (ph, num_points)
            if key in _PRIM_SAMPLE_CACHE:
                samples = _PRIM_SAMPLE_CACHE[key]
            else:
                prim_type = prim.GetTypeName()
                if prim_type == "Mesh":
                    mesh = UsdGeom.Mesh(prim)
                    verts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
                    faces = _triangulate_faces(prim)
                    mesh_tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
                else:
                    mesh_tm = create_primitive_mesh(prim)

                face_weights = mesh_tm.area_faces
                samples_np, _ = sample_surface(mesh_tm, num_points * 2, face_weight=face_weights)

                # FPS to num_points on chosen device
                tensor_pts = torch.from_numpy(samples_np.astype(np.float32)).to(device)
                prim_idxs = farthest_point_sampling(tensor_pts, num_points)
                local_pts = tensor_pts[prim_idxs]

                # prim -> root transform
                rel = xform_cache.GetLocalToWorldTransform(prim) * world_root.GetInverse()
                mat_np = np.array([[rel[r][c] for c in range(4)] for r in range(4)], dtype=np.float32)
                mat_t = torch.from_numpy(mat_np).to(device)

                ones = torch.ones((num_points, 1), device=device)
                pts_h = torch.cat([local_pts, ones], dim=1)
                root_h = pts_h @ mat_t
                samples = root_h[:, :3].detach().cpu().numpy()

                if prim_type == "Cone":
                    samples[:, 2] -= UsdGeom.Cone(prim).GetHeightAttr().Get() / 2

                _PRIM_SAMPLE_CACHE[key] = samples  # cache in root frame @ num_points

            all_samples_np.append(samples)

        # combine & env-level FPS (if needed)
        if len(all_samples_np) == 1:
            samples_final = torch.from_numpy(all_samples_np[0]).to(device)
        else:
            combined = torch.from_numpy(np.concatenate(all_samples_np, axis=0)).to(device)
            idxs = farthest_point_sampling(combined, num_points)
            samples_final = combined[idxs]

        # store env-level cache in root frame (CPU)
        _FINAL_SAMPLE_CACHE[env_hash] = samples_final.detach().cpu().numpy()

        # apply root scale and write out
        points[i] = samples_final * base_scale.unsqueeze(0)

    return points


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
    """Create a trimesh mesh from a USD primitive (Cube, Sphere, Cylinder, etc.)."""
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
    elif prim_type == "Cone":  # Cone
        c = UsdGeom.Cone(prim)
        return trimesh.creation.cone(radius=c.GetRadiusAttr().Get(), height=c.GetHeightAttr().Get())
    else:
        raise KeyError(f"{prim_type} is not a valid primitive mesh type")


def farthest_point_sampling(
    points: torch.Tensor, n_samples: int, memory_threashold=2 * 1024**3
) -> torch.Tensor:  # 2 GiB
    """
    Farthest Point Sampling (FPS) for point sets.

    Selects `n_samples` points such that each new point is farthest from the
    already chosen ones. Uses a full pairwise distance matrix if memory allows,
    otherwise falls back to an iterative version.

    Args:
        points (torch.Tensor): Input points of shape (N, D).
        n_samples (int): Number of samples to select.
        memory_threashold (int): Max allowed bytes for distance matrix. Default 2 GiB.

    Returns:
        torch.Tensor: Indices of sampled points (n_samples,).
    """
    device = points.device
    N = points.shape[0]
    elem_size = points.element_size()
    bytes_needed = N * N * elem_size
    if bytes_needed <= memory_threashold:
        dist_mat = torch.cdist(points, points)
        sampled_idx = torch.zeros(n_samples, dtype=torch.long, device=device)
        min_dists = torch.full((N,), float("inf"), device=device)
        farthest = torch.randint(0, N, (1,), device=device)
        for j in range(n_samples):
            sampled_idx[j] = farthest
            min_dists = torch.minimum(min_dists, dist_mat[farthest].view(-1))
            farthest = torch.argmax(min_dists)
        return sampled_idx
    logging.warning(f"FPS fallback to iterative (needed {bytes_needed} > {memory_threashold})")
    sampled_idx = torch.zeros(n_samples, dtype=torch.long, device=device)
    distances = torch.full((N,), float("inf"), device=device)
    farthest = torch.randint(0, N, (1,), device=device)
    for j in range(n_samples):
        sampled_idx[j] = farthest
        dist = torch.norm(points - points[farthest], dim=1)
        distances = torch.minimum(distances, dist)
        farthest = torch.argmax(distances)
    return sampled_idx


def sample_object_point_cloud_random(
    env_ids: list[int],
    num_points: int,
    prim_path: str,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Sample random points on object surfaces for specified environments.
    
    OPTIMIZED: Sample mesh ONCE from env 0, then apply per-env scales.
    All envs share the same mesh geometry, only scales differ.

    Args:
        env_ids: List of environment indices to sample for.
        num_points: Number of points to sample per environment.
        prim_path: USD prim path pattern with ".*" placeholder for env index.
        device: Device to place the output tensor on.

    Returns:
        torch.Tensor: Shape (len(env_ids), num_points, 3) on `device`.
    """
    n_envs = len(env_ids)
    xform_cache = UsdGeom.XformCache()
    
    # === STEP 1: Sample points ONCE from env 0's mesh (unscaled) ===
    obj_path_0 = prim_path.replace(".*", "0")
    prims = get_all_matching_child_prims(
        obj_path_0, predicate=lambda p: p.GetTypeName() in ("Mesh", "Cube", "Sphere", "Cylinder", "Capsule", "Cone")
    )
    if not prims:
        raise KeyError(f"No valid prims under {obj_path_0}")

    object_prim_0 = prim_utils.get_prim_at_path(obj_path_0)
    world_root_0 = xform_cache.GetLocalToWorldTransform(object_prim_0)

    # Sample from all prims in env 0
    all_samples: list[torch.Tensor] = []
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

        # Random surface sampling (area-weighted)
        face_weights = mesh_tm.area_faces
        samples_np, _ = sample_surface(mesh_tm, num_points, face_weight=face_weights)
        local_pts = torch.from_numpy(samples_np.astype(np.float32)).to(device)

        # Transform from prim local frame to object root frame
        rel = xform_cache.GetLocalToWorldTransform(prim) * world_root_0.GetInverse()
        mat_np = np.array([[rel[r][c] for c in range(4)] for r in range(4)], dtype=np.float32)
        mat_t = torch.from_numpy(mat_np).to(device)

        ones = torch.ones((num_points, 1), device=device)
        pts_h = torch.cat([local_pts, ones], dim=1)
        root_h = pts_h @ mat_t
        samples = root_h[:, :3]

        # Cone height adjustment
        if prim_type == "Cone":
            samples[:, 2] -= UsdGeom.Cone(prim).GetHeightAttr().Get() / 2

        all_samples.append(samples)

    # Combine samples
    if len(all_samples) == 1:
        base_points = all_samples[0][:num_points]
    else:
        combined = torch.cat(all_samples, dim=0)
        perm = torch.randperm(combined.shape[0], device=device)[:num_points]
        base_points = combined[perm]

    # === STEP 2: Get per-env scales (fast loop - just attribute reads) ===
    scales = torch.ones((n_envs, 3), dtype=torch.float32, device=device)
    for idx, env_id in enumerate(env_ids):
        obj_path = prim_path.replace(".*", str(env_id))
        object_prim = prim_utils.get_prim_at_path(obj_path)
        attr = object_prim.GetAttribute("xformOp:scale")
        scale_val = attr.Get() if attr else None
        if scale_val is not None:
            scales[idx] = torch.tensor(scale_val, dtype=torch.float32, device=device)

    # === STEP 3: Apply scales to all envs (vectorized) ===
    # base_points: (P, 3), scales: (N, 3) -> points: (N, P, 3)
    points = base_points.unsqueeze(0) * scales.unsqueeze(1)  # Broadcasting

    return points


# ---- Stable Pose Computation ----

# Cache for stable poses and aligned mesh per object geometry (to avoid recomputation)
# Key: geometry hash -> (transforms, probabilities, aligned_mesh)
_STABLE_POSE_CACHE: dict[str, tuple[np.ndarray, np.ndarray, trimesh.Trimesh]] = {}


def clear_stable_pose_cache():
    """Clear the stable pose cache."""
    _STABLE_POSE_CACHE.clear()


def get_stable_object_placement(
    env_ids: list[int],
    prim_path: str,
    table_surface_z: float = 0.255,
    randomize_pose: bool = True,
    z_offset: float = 0.005,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute stable z-positions and orientations for placing objects on table.
    
    OPTIMIZED: Compute stable pose ONCE from env 0, then apply per-env scales.

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
    n_envs = len(env_ids)
    z_positions = torch.zeros(n_envs, dtype=torch.float32)
    quaternions = torch.zeros(n_envs, 4, dtype=torch.float32)
    quaternions[:, 0] = 1.0  # Default to identity quaternion (w=1)

    xform_cache = UsdGeom.XformCache()

    # === STEP 1: Get stable pose and rotated mesh bounds from env 0 ===
    obj_path_0 = prim_path.replace(".*", "0")
    prims = get_all_matching_child_prims(
        obj_path_0, predicate=lambda p: p.GetTypeName() in ("Mesh", "Cube", "Sphere", "Cylinder", "Capsule", "Cone")
    )
    if not prims:
        logging.warning(f"No valid prims under {obj_path_0}, using default placement")
        z_positions[:] = table_surface_z + 0.1
        return z_positions, quaternions

    prim = prims[0]
    prim_type = prim.GetTypeName()
    cache_key = _compute_geometry_hash(prim, prim_type)

    # Check cache first
    if cache_key in _STABLE_POSE_CACHE:
        transforms, probabilities, mesh_tm = _STABLE_POSE_CACHE[cache_key]
    else:
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
            z_positions[:] = table_surface_z + 0.1
            return z_positions, quaternions

        _STABLE_POSE_CACHE[cache_key] = (transforms, probabilities, mesh_tm)

    if len(transforms) == 0:
        logging.warning("No stable poses found")
        z_positions[:] = table_surface_z + 0.1
        return z_positions, quaternions

    # Select stable pose (same for all envs if not randomizing)
    if randomize_pose and len(transforms) > 1:
        pose_idx = np.random.choice(len(transforms), p=probabilities / probabilities.sum())
    else:
        pose_idx = np.argmax(probabilities)

    stable_transform = transforms[pose_idx]
    rotation_matrix = stable_transform[:3, :3]
    
    # Convert to quaternion (same for all envs)
    rot = Rotation.from_matrix(rotation_matrix)
    quat_xyzw = rot.as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    quaternions[:] = torch.tensor(quat_wxyz, dtype=torch.float32)

    # Create rotated mesh ONCE (unscaled)
    mesh_rotated = mesh_tm.copy()
    rotation_only_transform = np.eye(4)
    rotation_only_transform[:3, :3] = rotation_matrix
    mesh_rotated.apply_transform(rotation_only_transform)
    
    # Get unscaled min_z (the z-bound scales linearly with scale_z)
    unscaled_min_z = mesh_rotated.bounds[0, 2]

    # Get geometric scale from env 0 (same for all envs)
    geom_scale_attr = prim.GetAttribute("xformOp:scale")
    geom_scale_val = geom_scale_attr.Get() if geom_scale_attr else None
    geom_scale_z = geom_scale_val[2] if geom_scale_val else 1.0

    # === STEP 2: Get per-env root scales (fast attribute reads) ===
    root_scales_z = np.ones(n_envs, dtype=np.float32)
    for idx, env_id in enumerate(env_ids):
        obj_path = prim_path.replace(".*", str(env_id))
        object_prim = prim_utils.get_prim_at_path(obj_path)
        root_scale_attr = object_prim.GetAttribute("xformOp:scale")
        root_scale_val = root_scale_attr.Get() if root_scale_attr else None
        if root_scale_val is not None:
            root_scales_z[idx] = root_scale_val[2]

    # === STEP 3: Compute z positions vectorized ===
    # scaled_min_z = unscaled_min_z * geom_scale_z * root_scale_z
    # z_position = table_surface_z - scaled_min_z + z_offset
    combined_scales_z = geom_scale_z * root_scales_z
    scaled_min_z = unscaled_min_z * combined_scales_z
    z_positions = torch.tensor(table_surface_z - scaled_min_z + z_offset, dtype=torch.float32)

    return z_positions, quaternions


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
