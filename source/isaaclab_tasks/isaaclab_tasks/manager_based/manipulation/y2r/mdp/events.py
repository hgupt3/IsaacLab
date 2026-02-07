# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom event functions for Y2R trajectory task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

from .utils import get_stable_object_placement

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_object_on_table_stable(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    xy_position_range: dict[str, tuple[float, float]],
    table_surface_z: float = 0.255,
    randomize_stable_pose: bool = True,
    z_offset: float = 0.005,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """Reset object to a stable resting pose on the table.

    This function uses trimesh.poses.compute_stable_poses() to find physically
    stable orientations for the object, then places it on the table at that
    orientation with randomized x, y position.

    Args:
        env: The environment instance.
        env_ids: Environment indices to reset.
        xy_position_range: Dictionary with "x" and "y" keys specifying the
            (min, max) range for position randomization relative to default.
        table_surface_z: Z-coordinate of the table surface.
        randomize_stable_pose: If True, randomly sample from stable poses
            weighted by probability. If False, use most stable pose.
        z_offset: Safety margin above table to prevent physics depenetration (meters).
        asset_cfg: Configuration for the object asset.
    """
    # Get the asset
    asset: RigidObject = env.scene[asset_cfg.name]

    # Get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # Compute stable placements for each environment
    env_ids_list = env_ids.tolist()
    z_positions, quaternions = get_stable_object_placement(
        env_ids=env_ids_list,
        prim_path=asset.cfg.prim_path,
        table_surface_z=table_surface_z,
        randomize_pose=randomize_stable_pose,
        z_offset=z_offset,
    )

    # Move tensors to correct device
    z_positions = z_positions.to(asset.device)
    quaternions = quaternions.to(asset.device)

    # Randomize x, y positions
    x_range = xy_position_range.get("x", (0.0, 0.0))
    y_range = xy_position_range.get("y", (0.0, 0.0))
    
    ranges = torch.tensor([x_range, y_range], device=asset.device)
    rand_xy = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 2), device=asset.device)

    # Build positions: default x,y + random offset, computed stable z
    positions = torch.zeros(len(env_ids), 3, device=asset.device)
    positions[:, 0] = root_states[:, 0] + env.scene.env_origins[env_ids, 0] + rand_xy[:, 0]
    positions[:, 1] = root_states[:, 1] + env.scene.env_origins[env_ids, 1] + rand_xy[:, 1]
    positions[:, 2] = z_positions + env.scene.env_origins[env_ids, 2]

    # Apply optional yaw randomization in WORLD frame (around world Z axis)
    # This rotates the object on the table regardless of its stable orientation
    yaw_range = xy_position_range.get("yaw", (0.0, 0.0))
    if yaw_range[0] != 0.0 or yaw_range[1] != 0.0:
        yaw_rand = math_utils.sample_uniform(
            yaw_range[0],
            yaw_range[1],
            (len(env_ids),),
            device=asset.device,
        )
        yaw_quat = math_utils.quat_from_euler_xyz(
            torch.zeros_like(yaw_rand),
            torch.zeros_like(yaw_rand),
            yaw_rand,
        )
        # yaw_quat * stable_quat = apply yaw in world frame, then stable orientation
        quaternions = math_utils.quat_mul(yaw_quat, quaternions)

    # Zero velocities
    velocities = torch.zeros(len(env_ids), 6, device=asset.device)

    # Write to simulation
    asset.write_root_pose_to_sim(torch.cat([positions, quaternions], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def reset_push_t_object(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """Reset push_t object with world-frame yaw randomization.

    Unlike reset_root_state_uniform which applies yaw in local frame, this function
    applies yaw in the WORLD frame (around world Z axis). This is necessary because
    the push_t object has a non-identity default rotation to lay flat on the table,
    and we want yaw to rotate it on the table surface.

    Args:
        env: The environment instance.
        env_ids: Environment indices to reset.
        pose_range: Dictionary with "x", "y", and "yaw" keys specifying
            (min, max) ranges. x/y are offsets from default position.
        asset_cfg: Configuration for the object asset.
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    # Get default root state (includes configured object_rotation)
    root_states = asset.data.default_root_state[env_ids].clone()

    # Randomize x, y positions (offsets from default)
    x_range = pose_range.get("x", (0.0, 0.0))
    y_range = pose_range.get("y", (0.0, 0.0))

    ranges = torch.tensor([x_range, y_range], device=asset.device)
    rand_xy = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 2), device=asset.device)

    # Build positions: default + env_origin + random offset
    positions = torch.zeros(len(env_ids), 3, device=asset.device)
    positions[:, 0] = root_states[:, 0] + env.scene.env_origins[env_ids, 0] + rand_xy[:, 0]
    positions[:, 1] = root_states[:, 1] + env.scene.env_origins[env_ids, 1] + rand_xy[:, 1]
    positions[:, 2] = root_states[:, 2] + env.scene.env_origins[env_ids, 2]  # Keep default z

    # Get default orientation (the configured object_rotation)
    quaternions = root_states[:, 3:7].clone()

    # Apply yaw in WORLD frame (around world Z axis)
    # This rotates the object on the table regardless of its default orientation
    yaw_range = pose_range.get("yaw", (0.0, 0.0))
    if yaw_range[0] != 0.0 or yaw_range[1] != 0.0:
        yaw_rand = math_utils.sample_uniform(
            yaw_range[0],
            yaw_range[1],
            (len(env_ids),),
            device=asset.device,
        )
        yaw_quat = math_utils.quat_from_euler_xyz(
            torch.zeros_like(yaw_rand),
            torch.zeros_like(yaw_rand),
            yaw_rand,
        )
        # yaw_quat * default_quat = apply yaw in world frame first
        quaternions = math_utils.quat_mul(yaw_quat, quaternions)

    # Zero velocities
    velocities = torch.zeros(len(env_ids), 6, device=asset.device)

    # Write to simulation
    asset.write_root_pose_to_sim(torch.cat([positions, quaternions], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def reset_robot_joints_above_table(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    wrist_position_range: tuple[float, float],
    table_surface_z: float,
    min_clearance: float,
    hand_body_regex: str,
    wrist_joint_name: str,
    arm_joint_regex: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset robot joints with table penetration check.

    Randomizes all joints with position_range, overrides wrist with
    wrist_position_range, then checks FK via update_articulations_kinematic()
    to ensure all hand bodies are above the table. Envs where any hand body
    penetrates get their arm+wrist joints reset to defaults (finger
    randomization is preserved).

    Args:
        env: The environment instance.
        env_ids: Environment indices to reset.
        position_range: (min, max) offset for all joints.
        wrist_position_range: (min, max) offset for wrist joint (overrides position_range).
        table_surface_z: Table surface z in local env frame.
        min_clearance: Min clearance above table (meters).
        hand_body_regex: Regex for hand body names to check.
        wrist_joint_name: Wrist joint name.
        arm_joint_regex: Regex matching all arm joints (including wrist).
        asset_cfg: Robot asset config.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    device = asset.device
    n = len(env_ids)
    if n == 0:
        return

    # Cache body/joint indices on first call
    cache_key = "_above_table_cache"
    if not hasattr(asset, cache_key):
        import re
        hand_body_ids, _ = asset.find_bodies(hand_body_regex)
        wrist_joint_ids, _ = asset.find_joints(wrist_joint_name)
        arm_joint_ids = [
            i for i, name in enumerate(asset.joint_names)
            if re.match(arm_joint_regex, name) and name != wrist_joint_name
        ]
        setattr(asset, cache_key, {
            "hand_body_ids": hand_body_ids,
            "arm_joint_ids": arm_joint_ids,
            "wrist_joint_ids": wrist_joint_ids,
        })
    cache = getattr(asset, cache_key)
    hand_body_ids = cache["hand_body_ids"]
    arm_joint_ids = cache["arm_joint_ids"]
    wrist_joint_ids = cache["wrist_joint_ids"]

    # Sample all joints: default + uniform offset
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()
    joint_pos += math_utils.sample_uniform(*position_range, joint_pos.shape, device)

    # Override wrist with wider range
    wrist_offset = math_utils.sample_uniform(
        *wrist_position_range, (n, len(wrist_joint_ids)), device
    )
    for i, jid in enumerate(wrist_joint_ids):
        joint_pos[:, jid] = asset.data.default_joint_pos[env_ids, jid] + wrist_offset[:, i]

    # Clamp to limits, zero velocity
    soft_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos.clamp_(soft_limits[..., 0], soft_limits[..., 1])
    joint_vel.zero_()

    # Write to sim (invalidates body_link_pose_w cache)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    # FK check: body_link_pose_w calls update_articulations_kinematic() then get_link_transforms()
    # Shape: (num_all_envs, num_bodies, 7) — positions are [:, :, :3]
    hand_z = asset.data.body_link_pose_w[env_ids][:, hand_body_ids, 2]
    min_hand_z = hand_z.min(dim=1).values
    table_z_thresh = table_surface_z + min_clearance + env.scene.env_origins[env_ids, 2]
    violating = min_hand_z < table_z_thresh

    # Fallback: reset arm+wrist to defaults for violating envs (keep finger randomization)
    if violating.any():
        viol_local = violating.nonzero(as_tuple=False).squeeze(-1)
        viol_env_ids = env_ids[viol_local]
        default_pos = asset.data.default_joint_pos[viol_env_ids]
        for jid in arm_joint_ids + wrist_joint_ids:
            joint_pos[viol_local, jid] = default_pos[:, jid]
        asset.write_joint_state_to_sim(
            joint_pos[viol_local], joint_vel[viol_local], env_ids=viol_env_ids
        )


def reset_camera_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    base_offset_pos: tuple[float, float, float],
    base_offset_rot_euler: tuple[float, float, float],
    perturbation_ranges: dict[str, tuple[float, float]],
    parent_body_name: str,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("wrist_camera"),
):
    """Randomize wrist camera position offset from parent body at each reset.

    Samples a position perturbation in camera frame (forward/lateral/vertical),
    transforms it to the parent body frame via the offset rotation, and applies
    the new offset via set_world_poses. Only position is perturbed — rotation
    is left unchanged.

    The perturbation ranges use camera-intuitive directions:
        forward  = camera looking direction (OpenGL -Z)
        lateral  = camera right direction   (OpenGL +X)
        vertical = camera up direction      (OpenGL +Y)

    Note: Uses stale body data from the previous step, but this cancels out
    because set_world_poses decomposes local = inv(parent_world) × desired_world
    using the same stale parent transform from the USD stage.

    Args:
        env: The environment instance.
        env_ids: Environment indices to reset.
        base_offset_pos: Default camera offset position in parent body frame (meters).
        base_offset_rot_euler: Euler angles [rx, ry, rz] degrees for the offset
            rotation (same format as wrist_camera.offset.rot in config).
        perturbation_ranges: Dict with "forward", "lateral", "vertical" keys,
            each a (min, max) range in meters.
        parent_body_name: Name of the parent body (e.g. "palm_link").
        sensor_cfg: Configuration for the camera sensor.
    """
    from scipy.spatial.transform import Rotation

    camera = env.scene[sensor_cfg.name]
    robot: Articulation = env.scene["robot"]
    device = robot.device
    n = len(env_ids)
    if n == 0:
        return

    # Cache parent body index on first call
    if not hasattr(camera, "_parent_body_idx"):
        body_ids = robot.find_bodies(parent_body_name)[0]
        camera._parent_body_idx = body_ids[0]
    parent_idx = camera._parent_body_idx

    # Compute opengl→palm rotation matrix (same Euler swap as env cfg mixin)
    re = base_offset_rot_euler
    r_palm_to_opengl = Rotation.from_euler("xyz", (re[0], re[2], -re[1]), degrees=True)
    R_opengl_to_palm = torch.tensor(
        r_palm_to_opengl.inv().as_matrix(), dtype=torch.float32, device=device
    )

    # Sample perturbation in OpenGL camera frame
    fwd = perturbation_ranges.get("forward", (0.0, 0.0))
    lat = perturbation_ranges.get("lateral", (0.0, 0.0))
    vert = perturbation_ranges.get("vertical", (0.0, 0.0))
    # OpenGL: +X = right (lateral), +Y = up (vertical), -Z = forward
    ranges_t = torch.tensor([lat, vert, [-fwd[1], -fwd[0]]], device=device)
    pert_opengl = math_utils.sample_uniform(
        ranges_t[:, 0], ranges_t[:, 1], (n, 3), device=device
    )

    # Rotate perturbation from OpenGL frame to palm frame
    pert_palm = (R_opengl_to_palm @ pert_opengl.unsqueeze(-1)).squeeze(-1)

    # New camera local offset in palm frame = base + perturbation
    base_pos = torch.tensor(base_offset_pos, dtype=torch.float32, device=device)
    new_local_pos = base_pos.unsqueeze(0) + pert_palm

    # Get parent body world pose (stale — cancels with set_world_poses)
    palm_pos_w = robot.data.body_pos_w[env_ids, parent_idx]
    palm_quat_w = robot.data.body_quat_w[env_ids, parent_idx]

    # Camera world position = palm_world + palm_rot @ local_offset
    new_cam_pos_w = palm_pos_w + math_utils.quat_apply(palm_quat_w, new_local_pos)

    # Set position only — orientation stays unchanged (preserves spawn-time OpenGL convention)
    camera.set_world_poses(positions=new_cam_pos_w, env_ids=env_ids)
