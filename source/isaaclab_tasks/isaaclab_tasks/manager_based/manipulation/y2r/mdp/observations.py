# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import (
    combine_frame_transforms,
    quat_apply,
    quat_apply_inverse,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    subtract_frame_transforms,
)

from .utils import sample_object_point_cloud_random, get_point_cloud_cache
from .actions import ALLEGRO_PCA_MATRIX, ALLEGRO_HAND_JOINT_NAMES, get_allegro_hand_joint_ids

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from ..trajectory_manager import TrajectoryManager
    from ..config_loader import Y2RConfig


# ---------------------------------------------------------------------------
# Fused broadcast quaternion utilities (avoid expand + contiguous copies)
# ---------------------------------------------------------------------------

@torch.jit.script
def _quat_rotate_vec(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate vec by quat with broadcasting over the points dimension.

    Uses torch.cross with broadcast views — only 2 intermediates (t, result)
    instead of 12+ scalar slices + cat.

    Args:
        quat: (N, 4) quaternion (w, x, y, z).
        vec:  (N, P, 3) vectors to rotate.

    Returns:
        Rotated vectors (N, P, 3).
    """
    xyz = quat[:, 1:].unsqueeze(1)  # (N, 1, 3) view — no alloc
    w = quat[:, 0:1].unsqueeze(1)   # (N, 1, 1) view — no alloc
    t = torch.cross(xyz.expand_as(vec), vec, dim=-1) * 2  # (N, P, 3)
    return vec + w * t + torch.cross(xyz.expand_as(t), t, dim=-1)


@torch.jit.script
def _points_to_frame(frame_pos: torch.Tensor, frame_quat: torch.Tensor, points_w: torch.Tensor) -> torch.Tensor:
    """Transform world-space points into a reference frame (fused inverse rotate + translate).

    Replaces the expand + subtract_frame_transforms pattern with a single
    broadcast operation. Only allocates delta, t, and result tensors.

    Args:
        frame_pos:  (N, 3) reference frame position.
        frame_quat: (N, 4) reference frame quaternion (w, x, y, z).
        points_w:   (N, P, 3) points in world frame.

    Returns:
        Points in the reference frame (N, P, 3).
    """
    # Conjugate = inverse for unit quaternions: negate xyz
    inv_xyz = -frame_quat[:, 1:].unsqueeze(1)  # (N, 1, 3) view
    w = frame_quat[:, 0:1].unsqueeze(1)         # (N, 1, 1) view

    delta = points_w - frame_pos.unsqueeze(1)    # (N, P, 3)
    t = torch.cross(inv_xyz.expand_as(delta), delta, dim=-1) * 2
    return delta + w * t + torch.cross(inv_xyz.expand_as(t), t, dim=-1)


# ==============================================================================
# PALM FRAME OFFSET HELPER
# ==============================================================================
# Cached tensors for palm frame offset (initialized on first call)
_PALM_OFFSET_POS: torch.Tensor | None = None
_PALM_OFFSET_QUAT: torch.Tensor | None = None


def _ensure_palm_offset_cache(y2r_cfg: Y2RConfig, device: torch.device) -> None:
    """Initialize cached palm frame offset tensors (once per process)."""
    global _PALM_OFFSET_POS, _PALM_OFFSET_QUAT
    if _PALM_OFFSET_POS is not None:
        return
    off = y2r_cfg.robot.palm_frame_offset
    if off is None:
        return
    _PALM_OFFSET_POS = torch.tensor(off.pos, dtype=torch.float32, device=device).unsqueeze(0)  # (1, 3)
    # Convert euler XYZ (degrees) to quaternion
    r, p, y = [x * 3.14159265358979 / 180.0 for x in off.rot_euler]
    _PALM_OFFSET_QUAT = quat_from_euler_xyz(
        torch.tensor([r], device=device),
        torch.tensor([p], device=device),
        torch.tensor([y], device=device),
    )  # (1, 4)


def get_palm_frame_pose_w(
    robot: Articulation,
    y2r_cfg: Y2RConfig,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Get palm frame world pose, applying offset if configured.

    Args:
        robot: The robot articulation.
        y2r_cfg: Y2R configuration.

    Returns:
        Tuple of (pos_w, quat_w, palm_body_idx) where pos_w and quat_w
        are the palm frame pose in world coordinates.
    """
    global _PALM_OFFSET_POS, _PALM_OFFSET_QUAT
    _ensure_palm_offset_cache(y2r_cfg, robot.device)

    palm_idx = robot.find_bodies(y2r_cfg.robot.palm_body_name)[0][0]
    pos_w = robot.data.body_pos_w[:, palm_idx]
    quat_w = robot.data.body_quat_w[:, palm_idx]

    if y2r_cfg.robot.palm_frame_offset is not None:
        N = pos_w.shape[0]
        pos_w, quat_w = combine_frame_transforms(
            pos_w, quat_w,
            _PALM_OFFSET_POS.expand(N, 3),
            _PALM_OFFSET_QUAT.expand(N, 4),
        )
    return pos_w, quat_w, palm_idx


def zeros_placeholder(
    env: ManagerBasedRLEnv,
    size: int = 7,
) -> torch.Tensor:
    """Placeholder observation that returns zeros.
    
    Useful for maintaining checkpoint compatibility when observations are removed.
    
    Args:
        env: The environment.
        size: Size of the zeros tensor. Defaults to 7 (pose size).
    
    Returns:
        Tensor of shape (num_envs, size) filled with zeros.
    """
    return torch.zeros(env.num_envs, size, device=env.device)


def _batched_slerp(
    q0: torch.Tensor,
    q1: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Batched SLERP between quaternions.

    Args:
        q0: (N, 4) start quaternions
        q1: (N, 4) end quaternions
        t: (N,) interpolation parameters in [0, 1]

    Returns:
        (N, 4) interpolated quaternions
    """
    dot = (q0 * q1).sum(dim=-1)
    q1 = torch.where(dot.unsqueeze(-1) < 0, -q1, q1)
    dot = dot.abs().clamp(max=1.0)

    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    near_parallel = sin_theta < 1e-6

    w0 = torch.sin((1 - t) * theta) / sin_theta
    w1 = torch.sin(t * theta) / sin_theta
    w0 = torch.where(near_parallel, 1 - t, w0)
    w1 = torch.where(near_parallel, t, w1)

    result = w0.unsqueeze(-1) * q0 + w1.unsqueeze(-1) * q1
    return result / result.norm(dim=-1, keepdim=True)


def _compute_aligned_hand_target(
    trajectory_manager: TrajectoryManager,
    path_mode: bool,
    timing_aware: bool,
    palm_pos_w: torch.Tensor,
) -> torch.Tensor:
    """Compute aligned hand target pose (pos+quat) for the current segment.

    Uses trajectory.path_mode + trajectory.timing_aware to optionally interpolate
    between the current and next hand target. Falls back to current target if
    the window is too small.
    """
    hand_window = trajectory_manager.get_hand_window_targets()  # (N, W, 7)
    if not path_mode or hand_window.shape[1] < 2:
        return hand_window[:, 0, :]

    p0 = hand_window[:, 0, :3]
    p1 = hand_window[:, 1, :3]
    q0 = hand_window[:, 0, 3:7]
    q1 = hand_window[:, 1, 3:7]

    if timing_aware:
        t = trajectory_manager.get_segment_progress()
    else:
        v = p1 - p0
        v_len_sq = (v ** 2).sum(dim=-1, keepdim=True).clamp(min=1e-8)
        t = ((palm_pos_w - p0) * v).sum(dim=-1, keepdim=True) / v_len_sq
        t = t.clamp(0.0, 1.0).squeeze(-1)

    interp_pos = p0 + t.unsqueeze(-1) * (p1 - p0)
    interp_quat = _batched_slerp(q0, q1, t)
    return torch.cat([interp_pos, interp_quat], dim=-1)


def object_pos_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """Object position in the robot's root frame.

    Args:
        env: The environment.
        robot_cfg: Scene entity for the robot (reference frame). Defaults to ``SceneEntityCfg("robot")``.
        object_cfg: Scene entity for the object. Defaults to ``SceneEntityCfg("object")``.

    Returns:
        Tensor of shape ``(num_envs, 3)``: object position [x, y, z] expressed in the robot root frame.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    return quat_apply_inverse(robot.data.root_quat_w, object.data.root_pos_w - robot.data.root_pos_w)


def object_quat_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object orientation in the robot's root frame.

    Args:
        env: The environment.
        robot_cfg: Scene entity for the robot (reference frame). Defaults to ``SceneEntityCfg("robot")``.
        object_cfg: Scene entity for the object. Defaults to ``SceneEntityCfg("object")``.

    Returns:
        Tensor of shape ``(num_envs, 4)``: object quaternion ``(w, x, y, z)`` in the robot root frame.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    return quat_mul(quat_inv(robot.data.root_quat_w), object.data.root_quat_w)


def object_pose_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object pose (position + quaternion) in the robot's root frame.

    Combines object_pos_b and object_quat_b into a single 7-dim observation.
    Used in pose mode instead of point cloud observations.

    Args:
        env: The environment.
        robot_cfg: Scene entity for the robot (reference frame). Defaults to ``SceneEntityCfg("robot")``.
        object_cfg: Scene entity for the object. Defaults to ``SceneEntityCfg("object")``.

    Returns:
        Tensor of shape ``(num_envs, 7)``: [x, y, z, qw, qx, qy, qz] in robot root frame.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    
    # Position in robot frame
    pos_b = quat_apply_inverse(robot.data.root_quat_w, object.data.root_pos_w - robot.data.root_pos_w)
    
    # Orientation in robot frame
    quat_b = quat_mul(quat_inv(robot.data.root_quat_w), object.data.root_quat_w)
    
    return torch.cat([pos_b, quat_b], dim=-1)


def hand_pose_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Hand (palm) pose in the robot's root frame.

    Extracts the palm_link body pose and expresses it in the robot base frame.
    Returns position + quaternion (7D) matching hand_pose_targets format.

    Args:
        env: The environment.
        robot_cfg: Scene entity for the robot. Defaults to ``SceneEntityCfg("robot")``.

    Returns:
        Tensor of shape ``(num_envs, 7)``: [x, y, z, qw, qx, qy, qz] in robot root frame.
    """
    robot: Articulation = env.scene[robot_cfg.name]

    # Get palm frame pose in world frame (with offset if configured)
    palm_pos_w, palm_quat_w, _ = get_palm_frame_pose_w(robot, env.cfg.y2r_cfg)

    # Get robot base frame
    root_pos_w = robot.data.root_link_pos_w  # (N, 3)
    root_quat_w = robot.data.root_link_quat_w  # (N, 4)

    # Transform to robot base frame
    palm_pos_b, palm_quat_b = subtract_frame_transforms(
        root_pos_w, root_quat_w, palm_pos_w, palm_quat_w
    )

    return torch.cat([palm_pos_b, palm_quat_b], dim=-1)


def object_pose_palm_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object pose in the palm frame (7D: pos + quat).

    Args:
        env: The environment.
        robot_cfg: Scene entity for the robot. Defaults to ``SceneEntityCfg("robot")``.
        object_cfg: Scene entity for the object. Defaults to ``SceneEntityCfg("object")``.

    Returns:
        Tensor of shape ``(num_envs, 7)``: [x, y, z, qw, qx, qy, qz] in palm frame.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    palm_pos_w, palm_quat_w, _ = get_palm_frame_pose_w(robot, env.cfg.y2r_cfg)
    obj_pos_palm, obj_quat_palm = subtract_frame_transforms(
        palm_pos_w, palm_quat_w, object.data.root_pos_w, object.data.root_quat_w
    )
    return torch.cat([obj_pos_palm, obj_quat_palm], dim=-1)


def hand_pose_error_palm(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Delta from current palm pose to next palm target, in current palm frame (7D).

    Args:
        env: The environment.
        robot_cfg: Scene entity for the robot. Defaults to ``SceneEntityCfg("robot")``.

    Returns:
        Tensor of shape ``(num_envs, 7)``: [dx, dy, dz, dqw, dqx, dqy, dqz] in palm frame.
        Returns zeros if hand trajectory is disabled.
    """
    cfg = env.cfg.y2r_cfg
    if not cfg.hand_trajectory.enabled:
        return torch.zeros(env.num_envs, 7, device=env.device)

    robot: Articulation = env.scene[robot_cfg.name]
    trajectory_manager = env.trajectory_manager

    palm_pos_w, palm_quat_w, _ = get_palm_frame_pose_w(robot, cfg)
    hand_targets = trajectory_manager.get_hand_window_targets()  # (N, W, 7)
    next_pos_w = hand_targets[:, 0, :3]
    next_quat_w = hand_targets[:, 0, 3:7]

    delta_pos, delta_quat = subtract_frame_transforms(
        palm_pos_w, palm_quat_w, next_pos_w, next_quat_w
    )
    return torch.cat([delta_pos, delta_quat], dim=-1)


def object_pose_error(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Delta from current object pose to next object target, in current object frame (7D).

    Args:
        env: The environment.
        object_cfg: Scene entity for the object. Defaults to ``SceneEntityCfg("object")``.

    Returns:
        Tensor of shape ``(num_envs, 7)``: [dx, dy, dz, dqw, dqx, dqy, dqz] in object frame.
    """
    object: RigidObject = env.scene[object_cfg.name]
    trajectory_manager = env.trajectory_manager

    obj_pos_w = object.data.root_pos_w
    obj_quat_w = object.data.root_quat_w
    obj_targets = trajectory_manager.get_window_targets()  # (N, W, 7)
    next_pos_w = obj_targets[:, 0, :3]
    next_quat_w = obj_targets[:, 0, 3:7]

    delta_pos, delta_quat = subtract_frame_transforms(
        obj_pos_w, obj_quat_w, next_pos_w, next_quat_w
    )
    return torch.cat([delta_pos, delta_quat], dim=-1)


def body_state_b(
    env: ManagerBasedRLEnv,
    body_asset_cfg: SceneEntityCfg,
    base_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Body state (pos, quat, lin vel, ang vel) in the base asset's root frame.

    The state for each body is stacked horizontally as
    ``[position(3), quaternion(4)(wxyz), linvel(3), angvel(3)]`` and then concatenated over bodies.

    Args:
        env: The environment.
        body_asset_cfg: Scene entity for the articulated body whose links are observed.
        base_asset_cfg: Scene entity providing the reference (root) frame.

    Returns:
        Tensor of shape ``(num_envs, num_bodies * 13)`` with per-body states expressed in the base root frame.
    """
    body_asset: Articulation = env.scene[body_asset_cfg.name]
    base_asset: Articulation = env.scene[base_asset_cfg.name]
    # get world pose of bodies
    body_pos_w = body_asset.data.body_pos_w[:, body_asset_cfg.body_ids].view(-1, 3)
    body_quat_w = body_asset.data.body_quat_w[:, body_asset_cfg.body_ids].view(-1, 4)
    body_lin_vel_w = body_asset.data.body_lin_vel_w[:, body_asset_cfg.body_ids].view(-1, 3)
    body_ang_vel_w = body_asset.data.body_ang_vel_w[:, body_asset_cfg.body_ids].view(-1, 3)
    num_bodies = int(body_pos_w.shape[0] / env.num_envs)
    # get world pose of base frame
    root_pos_w = base_asset.data.root_link_pos_w.unsqueeze(1).repeat_interleave(num_bodies, dim=1).view(-1, 3)
    root_quat_w = base_asset.data.root_link_quat_w.unsqueeze(1).repeat_interleave(num_bodies, dim=1).view(-1, 4)
    # transform from world body pose to local body pose
    body_pos_b, body_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, body_pos_w, body_quat_w)
    body_lin_vel_b = quat_apply_inverse(root_quat_w, body_lin_vel_w)
    body_ang_vel_b = quat_apply_inverse(root_quat_w, body_ang_vel_w)
    # concate and return
    out = torch.cat((body_pos_b, body_quat_b, body_lin_vel_b, body_ang_vel_b), dim=1)
    return out.view(env.num_envs, -1)


# Cached tip offset tensor (initialized on first call)
_TIP_OFFSETS_TENSOR: torch.Tensor | None = None


def _ensure_tip_offsets_cache(y2r_cfg: Y2RConfig, device: torch.device) -> None:
    """Initialize cached tip offset tensor (once per process)."""
    global _TIP_OFFSETS_TENSOR
    if _TIP_OFFSETS_TENSOR is not None:
        return
    tip_cfg = y2r_cfg.robot.tip_offsets
    if tip_cfg is None:
        return
    # Order matches regex ".*_link_3": index, middle, ring, thumb
    _TIP_OFFSETS_TENSOR = torch.tensor(
        [tip_cfg["index"], tip_cfg["middle"], tip_cfg["ring"], tip_cfg["thumb"]],
        dtype=torch.float32,
        device=device,
    )  # (4, 3)


def hand_tips_state_with_offsets_b(
    env: ManagerBasedRLEnv,
    body_asset_cfg: SceneEntityCfg,
    base_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Hand tips body state with tip offsets applied.

    Same as body_state_b but applies configured tip offsets to link_3 positions.
    Body order: [palm_link, index_link_3, middle_link_3, ring_link_3, thumb_link_3].
    Tip offsets are applied to the last 4 bodies (link_3) if configured.

    Returns:
        Tensor of shape ``(num_envs, 5 * 13)`` with per-body states in base root frame.
    """
    y2r_cfg = env.cfg.y2r_cfg
    _ensure_tip_offsets_cache(y2r_cfg, env.device)

    # Get base body states
    out = body_state_b(env, body_asset_cfg, base_asset_cfg)  # (N, 5*13)

    # Apply tip offsets if configured
    global _TIP_OFFSETS_TENSOR
    if _TIP_OFFSETS_TENSOR is not None:
        N = env.num_envs
        out_reshaped = out.view(N, 5, 13)  # (N, 5, 13)
        # Bodies 1-4 are link_3 bodies; apply offset in each body's local frame
        for i in range(4):
            body_quat_b = out_reshaped[:, 1 + i, 3:7]  # (N, 4) quaternion in base frame
            offset_b = quat_apply(body_quat_b, _TIP_OFFSETS_TENSOR[i].unsqueeze(0).expand(N, 3))
            out_reshaped[:, 1 + i, 0:3] = out_reshaped[:, 1 + i, 0:3] + offset_b
        out = out_reshaped.view(N, -1)

    return out


class object_point_cloud_b(ManagerTermBase):
    """Object surface point cloud expressed in a reference asset's root frame.

    Uses RANDOM surface sampling and caches points on env._object_points_local
    for sharing with target_sequence_point_clouds_b.

    Args (from ``cfg.params``):
        object_cfg: Scene entity for the object to sample. Defaults to ``SceneEntityCfg("object")``.
        ref_asset_cfg: Scene entity providing the reference frame. Defaults to ``SceneEntityCfg("robot")``.
        num_points: Number of points to sample on the object surface. Defaults to ``10``.

    Returns (from ``__call__``):
        If ``flatten=False``: tensor of shape ``(num_envs, num_points, 3)``.
        If ``flatten=True``: tensor of shape ``(num_envs, 3 * num_points)``.

    Note:
        Caches points_local on env._object_points_local for use by target_sequence_point_clouds_b.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.object_cfg: SceneEntityCfg = cfg.params.get("object_cfg", SceneEntityCfg("object"))
        self.ref_asset_cfg: SceneEntityCfg = cfg.params.get("ref_asset_cfg", SceneEntityCfg("robot"))
        self.num_points: int = cfg.params.get("num_points", 10)
        self.object: RigidObject = env.scene[self.object_cfg.name]
        self.ref_asset: Articulation = env.scene[self.ref_asset_cfg.name]

        # Sample points using RANDOM sampling (not FPS) for consistency with targets
        all_env_ids = list(range(env.num_envs))

        # Get filter config from Y2R config (already a dict)
        filter_config = None
        if hasattr(env, 'cfg') and hasattr(env.cfg, 'y2r_cfg'):
            y2r_cfg = env.cfg.y2r_cfg
            filter_config = y2r_cfg.observations.point_cloud_filter

        self.points_local = sample_object_point_cloud_random(
            all_env_ids, self.num_points, self.object.cfg.prim_path,
            device=env.device, filter_config=filter_config
        )
        self.points_w = torch.zeros_like(self.points_local)

        # Cache on env for target_sequence_point_clouds_b to use the SAME points
        env._object_points_local = self.points_local

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        ref_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        num_points: int = 10,
        flatten: bool = False,
    ):
        """Compute the object point cloud in the reference asset's root frame.

        Args:
            env: The environment.
            ref_asset_cfg: Reference frame provider (root). Defaults to ``SceneEntityCfg("robot")``.
            object_cfg: Object to sample. Defaults to ``SceneEntityCfg("object")``.
            num_points: Number of points (set at init from cfg.params).
            flatten: If ``True``, return a flattened tensor ``(num_envs, 3 * num_points)``.

        Returns:
            Tensor of shape ``(num_envs, num_points, 3)`` or flattened if requested.
        """
        # Transform local points to world (broadcast, no expand)
        self.points_w = _quat_rotate_vec(self.object.data.root_quat_w, self.points_local) + self.object.data.root_pos_w.unsqueeze(1)

        # Transform to robot base frame (fused inverse rotate + translate)
        object_point_cloud_pos_b = _points_to_frame(self.ref_asset.data.root_pos_w, self.ref_asset.data.root_quat_w, self.points_w)

        return object_point_cloud_pos_b.view(env.num_envs, -1) if flatten else object_point_cloud_pos_b


class visible_object_point_cloud_b(ManagerTermBase):
    """Visible object point cloud from a pseudo-camera viewpoint.
    
    Samples points on object surface and filters to only include points
    visible from the pseudo-camera position (using self-occlusion / back-face culling).
    Points whose surface normals face away from the camera are hidden.
    
    Args (from ``cfg.params``):
        object_cfg: Scene entity for the object. Defaults to ``SceneEntityCfg("object")``.
        ref_asset_cfg: Scene entity for reference frame. Defaults to ``SceneEntityCfg("robot")``.
        num_points: Number of visible points to return. Defaults to config value.
        
    Returns (from ``__call__``):
        Flattened tensor of shape ``(num_envs, num_points * 3)``.
    """
    
    # Blue color for visible point cloud visualization
    VISIBLE_POINT_COLOR = (0.0, 0.0, 1.0)  # Bright blue
    POINT_RADIUS = 0.003
    
    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        
        self.object_cfg: SceneEntityCfg = cfg.params.get("object_cfg", SceneEntityCfg("object"))
        self.ref_asset_cfg: SceneEntityCfg = cfg.params.get("ref_asset_cfg", SceneEntityCfg("robot"))
        
        # Get config - use student_num_points for student observations
        y2r_cfg = env.cfg.y2r_cfg
        self.num_points: int = cfg.params.get("num_points", y2r_cfg.observations.student_num_points)
        self.pool_size: int = y2r_cfg.observations.point_pool_size
        self.pseudo_camera_pos: tuple = y2r_cfg.pseudo_camera.position
        self.visualize: bool = y2r_cfg.visualization.student_visible
        self.visualize_env_ids: list | None = y2r_cfg.visualization.env_ids
        
        self.object: RigidObject = env.scene[self.object_cfg.name]
        self.ref_asset: Articulation = env.scene[self.ref_asset_cfg.name]
        self.env = env
        
        # Compute resample interval: resample every window_size target updates
        # This means visibility is resampled when the entire target window advances
        target_hz = y2r_cfg.trajectory.target_hz
        window_size = y2r_cfg.trajectory.window_size
        policy_dt = env.step_dt  # Time per policy step
        steps_per_target = 1.0 / (target_hz * policy_dt)  # Steps between target updates
        self.resample_interval: int = max(1, int(window_size * steps_per_target))
        
        # Get the point cloud cache with normals and move to GPU
        all_env_ids = list(range(env.num_envs))
        cache = get_point_cloud_cache(all_env_ids, self.object.cfg.prim_path, self.pool_size)
        
        # Move cache tensors to GPU once (avoid CPU-GPU transfer every frame)
        self.geo_indices = cache.geo_indices.to(env.device)
        self.scales = cache.scales.to(env.device)
        self.all_base_points = cache.all_base_points.to(env.device)
        self.all_base_normals = cache.all_base_normals.to(env.device)
        
        # Pre-compute camera position tensor (relative to env origins)
        camera_offset = torch.tensor(self.pseudo_camera_pos, device=env.device, dtype=torch.float32)
        self.camera_offset = camera_offset.unsqueeze(0)  # (1, 3) for broadcasting
        
        # Pre-allocate output buffer
        self.output = torch.zeros(env.num_envs, self.num_points, 3, device=env.device)
        
        # For visualization - store world-space points
        self.selected_points_w = None
        
        # Cached visibility sampling (resample at target_hz rate)
        self.cached_indices = None  # (N, num_points) selected point indices
        self.cached_points_local = None  # (N, num_points, 3) selected points in local frame
        self._step_counter = 0  # CPU-side counter to avoid GPU sync for resample check
        
        # Initialize visualization if enabled
        self.markers = None
        if self.visualize:
            self._init_visualizer()
        
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        ref_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        num_points: int = 64,
        flatten: bool = True,
    ) -> torch.Tensor:
        """Compute visible point cloud from pseudo-camera viewpoint.
        
        Visibility sampling happens at target_hz rate (not every step) for efficiency.
        Between resamples, cached indices are used with updated object poses.
        
        Args:
            env: The environment.
            ref_asset_cfg: Reference frame (robot).
            object_cfg: Object to sample.
            num_points: Number of points (from init).
            flatten: If True, return flattened tensor.
            
        Returns:
            Tensor of shape (num_envs, num_points * 3) if flattened.
        """
        N = env.num_envs
        P = self.pool_size
        num_pts = self.num_points
        device = env.device
        
        # Get per-env geometry data (already on GPU)
        geo_indices = self.geo_indices[:N]  # (N,)
        scales = self.scales[:N]  # (N, 3)
        
        # Index into cached points/normals (all on GPU)
        pool_points_local = self.all_base_points[geo_indices]   # (N, P, 3)
        pool_normals_local = self.all_base_normals[geo_indices]  # (N, P, 3)
        
        # Apply per-env scale to points (normals don't scale)
        pool_points_local = pool_points_local * scales.unsqueeze(1)  # (N, P, 3)
        
        # Determine if resampling is needed using a CPU-side counter to avoid
        # GPU sync (.any().item() forces a pipeline stall every step).
        # Resets are handled by always resampling at the interval — any env that
        # reset within the last interval steps will get fresh samples.
        needs_resample = self.cached_indices is None

        if not needs_resample:
            self._step_counter += 1
            needs_resample = self._step_counter >= self.resample_interval
            if needs_resample:
                self._step_counter = 0
        
        if needs_resample:
            # Full visibility resampling
            object_pos_w = self.object.data.root_pos_w  # (N, 3)
            object_quat_w = self.object.data.root_quat_w  # (N, 4)

            # Transform to world (broadcast, no expand)
            pool_points_w = _quat_rotate_vec(object_quat_w, pool_points_local) + object_pos_w.unsqueeze(1)  # (N, P, 3)
            pool_normals_w = _quat_rotate_vec(object_quat_w, pool_normals_local)  # (N, P, 3)
            pool_normals_w = pool_normals_w / (pool_normals_w.norm(dim=-1, keepdim=True) + 1e-8)

            # Camera position in world
            env_origins = env.scene.env_origins  # (N, 3)
            camera_pos_w = env_origins + self.camera_offset  # (N, 3)

            # Compute visibility
            view_dirs = camera_pos_w.unsqueeze(1) - pool_points_w  # (N, P, 3)
            view_dirs = view_dirs / (view_dirs.norm(dim=-1, keepdim=True) + 1e-8)
            visibility = (pool_normals_w * view_dirs).sum(dim=-1)  # (N, P)

            # Sample visible points
            selected_indices = self._sample_visible_batched(visibility, num_pts)  # (N, num_pts)

            # Cache indices and local points
            self.cached_indices = selected_indices
            batch_idx = torch.arange(N, device=device).unsqueeze(1).expand(-1, num_pts)
            self.cached_points_local = pool_points_local[batch_idx, selected_indices]  # (N, num_pts, 3)

        # Use cached local points and transform to current world frame (broadcast, no expand)
        object_pos_w = self.object.data.root_pos_w  # (N, 3)
        object_quat_w = self.object.data.root_quat_w  # (N, 4)

        selected_points_w = _quat_rotate_vec(object_quat_w, self.cached_points_local) + object_pos_w.unsqueeze(1)

        # Store visible points in object-local frame for visible_target_sequence_obs_b
        env._visible_points_local = self.cached_points_local  # (N, num_points, 3)

        # Transform to robot base frame (fused inverse rotate + translate)
        points_b = _points_to_frame(self.ref_asset.data.root_pos_w, self.ref_asset.data.root_quat_w, selected_points_w)

        # Store for visualization and visualize if enabled
        self.selected_points_w = selected_points_w
        if self.visualize and self.markers is not None:
            self._visualize(selected_points_w, N)

        return points_b.view(N, -1) if flatten else points_b
    
    def _init_visualizer(self):
        """Initialize blue sphere markers for visible point cloud."""
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        
        markers_dict = {
            "visible": sim_utils.SphereCfg(
                radius=self.POINT_RADIUS,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=self.VISIBLE_POINT_COLOR),
            ),
        }
        
        self.markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/StudentVisiblePointCloud",
                markers=markers_dict,
            )
        )
    
    def _visualize(self, points_w: torch.Tensor, num_envs: int):
        """Visualize visible point cloud with blue markers.
        
        Args:
            points_w: (N, num_points, 3) selected visible points in world space.
            num_envs: Number of environments.
        """
        # Determine which envs to visualize
        if self.visualize_env_ids is None:
            env_ids = list(range(num_envs))
        elif len(self.visualize_env_ids) == 0:
            return  # Empty list means no visualization
        else:
            env_ids = [i for i in self.visualize_env_ids if i < num_envs]
        
        if not env_ids:
            return
        
        P = points_w.shape[1]
        E = len(env_ids)
        device = points_w.device
        
        # Gather points for selected envs
        selected_pts = points_w[env_ids]  # (E, P, 3)
        
        # Flatten: (E, P, 3) -> (E*P, 3)
        positions = selected_pts.reshape(-1, 3)
        
        # All points use the same marker (index 0 = "visible")
        marker_indices = torch.zeros(E * P, dtype=torch.long, device=device)
        
        self.markers.visualize(
            translations=positions,
            marker_indices=marker_indices,
        )
    
    def _sample_visible_batched(self, visibility: torch.Tensor, num_points: int) -> torch.Tensor:
        """Randomly sample N points from visible set, fully batched.
        
        Args:
            visibility: (N, pool) - dot product scores, >0 means visible.
            num_points: Number of points to select.
            
        Returns:
            indices: (N, num_points) - selected point indices per env.
        """
        N, P = visibility.shape
        device = visibility.device
        
        visible_mask = visibility > 0  # (N, P)
        
        # Assign random priority to visible, -inf to hidden
        random_scores = torch.where(
            visible_mask,
            torch.rand(N, P, device=device),
            torch.full((N, P), float('-inf'), device=device)
        )
        
        # topk is O(N*k) vs sort's O(N*P*logP) — ~8x faster for k=32, P=256
        _, selected = random_scores.topk(num_points, dim=-1)  # (N, num_points)

        # Handle edge case: if < num_points visible, repeat from visible ones
        num_visible = visible_mask.sum(dim=-1, keepdim=True)  # (N, 1)
        needs_repeat = num_visible < num_points

        if needs_repeat.any():
            # topk puts visible (positive scores) first, hidden (-inf) last.
            # Wrap indices to cycle through the visible portion of selected.
            repeat_idx = torch.arange(num_points, device=device).unsqueeze(0)  # (1, num_points)
            wrapped_idx = (repeat_idx % num_visible.clamp(min=1)).long()  # (N, num_points)
            repeated = selected.gather(1, wrapped_idx)

            selected = torch.where(
                needs_repeat.expand(-1, num_points),
                repeated,
                selected
            )
        
        return selected


class visible_target_sequence_obs_b(ManagerTermBase):
    """Student target observation: transforms visible points through target trajectory.
    
    Takes the visible points selected by visible_object_point_cloud_b and transforms
    them through the target poses from the trajectory manager. This provides the
    student with a filtered view of where visible surface points will be at each
    future target pose.
    
    Prerequisites:
        - visible_object_point_cloud_b must be initialized first (stores env._visible_points_local)
        - target_sequence_obs_b must be initialized first (creates env.trajectory_manager)
    
    Args (from ``cfg.params``):
        ref_asset_cfg: Scene entity for reference frame. Defaults to ``SceneEntityCfg("robot")``.
        
    Returns (from ``__call__``):
        Flattened tensor of shape ``(num_envs, window_size * num_points * 3)``.
    """
    
    # Same inferno colormap as target_sequence_obs_b
    INFERNO_COLORS = [
        (0.07, 0.04, 0.11),  # 0: nearly black
        (0.32, 0.06, 0.38),  # 1: dark purple
        (0.55, 0.09, 0.38),  # 2: magenta
        (0.75, 0.19, 0.27),  # 3: red-purple
        (0.89, 0.35, 0.13),  # 4: orange-red
        (0.96, 0.55, 0.04),  # 5: orange
        (0.99, 0.77, 0.17),  # 6: yellow-orange
        (0.99, 0.99, 0.64),  # 7: bright yellow
    ]
    POINT_RADIUS = 0.0025
    
    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        
        self.ref_asset_cfg: SceneEntityCfg = cfg.params.get("ref_asset_cfg", SceneEntityCfg("robot"))
        
        # Get config - use student_num_points for student observations
        y2r_cfg = env.cfg.y2r_cfg
        self.num_points: int = y2r_cfg.observations.student_num_points
        self.window_size: int = y2r_cfg.trajectory.window_size
        self.visualize: bool = y2r_cfg.visualization.student_target
        self.visualize_env_ids: list | None = y2r_cfg.visualization.env_ids
        
        self.ref_asset: Articulation = env.scene[self.ref_asset_cfg.name]
        self.env = env
        
        # Verify prerequisites
        if not hasattr(env, 'trajectory_manager'):
            raise RuntimeError(
                "visible_target_sequence_obs_b requires target_sequence_obs_b to be initialized first. "
                "Ensure 'targets' observation group is defined before 'student_targets'."
            )
        
        # Initialize visualization if enabled
        self.markers = None
        if self.visualize:
            self._init_visualizer()
    
    def _init_visualizer(self):
        """Initialize visualization markers with inferno colormap."""
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        
        markers_dict = {
            f"visible_target_{i}": sim_utils.SphereCfg(
                radius=self.POINT_RADIUS,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=self.INFERNO_COLORS[i]),
            )
            for i in range(self.window_size)
        }
        
        self.markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/StudentVisibleTargetSequence",
                markers=markers_dict,
            )
        )
    
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        ref_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        """Compute visible target sequence observations.
        
        Returns:
            Flattened tensor (num_envs, window_size * num_points * 3).
        """
        N = env.num_envs
        W = self.window_size
        P = self.num_points
        device = env.device
        
        # Get visible points in object-local frame (set by visible_object_point_cloud_b)
        if not hasattr(env, '_visible_points_local'):
            # Fallback: return zeros if visible points not available yet
            return torch.zeros(N, W * P * 3, device=device)
        
        visible_points_local = env._visible_points_local  # (N, P, 3)
        
        # Get target poses from trajectory manager
        window_targets = env.trajectory_manager.get_window_targets()  # (N, W, 7)
        target_pos_w = window_targets[:, :, :3]   # (N, W, 3)
        target_quat_w = window_targets[:, :, 3:7]  # (N, W, 4)
        
        # Transform visible points to each target pose
        # Expand local points for all windows: (N, P, 3) -> (N, W, P, 3)
        local_exp = visible_points_local.unsqueeze(1).expand(N, W, P, 3)
        
        # Expand target poses for all points
        target_pos = target_pos_w.unsqueeze(2)   # (N, W, 1, 3)
        # Flatten (N, W) → (N*W) to use broadcast utilities, avoiding (N, W, P) expand
        target_quat_flat = target_quat_w.reshape(-1, 4)  # (N*W, 4)
        target_pos_flat = target_pos_w.reshape(-1, 3)    # (N*W, 3)
        local_flat = local_exp.reshape(N * W, P, 3)      # (N*W, P, 3)

        all_points_w_flat = _quat_rotate_vec(target_quat_flat, local_flat) + target_pos_flat.unsqueeze(1)
        all_points_w = all_points_w_flat.reshape(N, W, P, 3)

        # Transform to robot base frame (expand only (N)→(N*W), not (N)→(N*W*P))
        ref_pos_flat = self.ref_asset.data.root_pos_w.unsqueeze(1).expand(N, W, 3).reshape(-1, 3)
        ref_quat_flat = self.ref_asset.data.root_quat_w.unsqueeze(1).expand(N, W, 4).reshape(-1, 4)

        all_points_b = _points_to_frame(ref_pos_flat, ref_quat_flat, all_points_w_flat).reshape(N, W, P, 3)
        
        # Visualize if enabled
        if self.visualize and self.markers is not None:
            self._visualize(all_points_w, N)
        
        # Flatten: (N, W * P * 3)
        return all_points_b.reshape(N, -1)
    
    def _visualize(self, target_points_w: torch.Tensor, num_envs: int):
        """Visualize visible target point clouds with inferno colormap.
        
        Args:
            target_points_w: (N, W, P, 3) visible points at each target pose in world space.
            num_envs: Number of environments.
        """
        # Determine which envs to visualize
        if self.visualize_env_ids is None:
            env_ids = list(range(num_envs))
        elif len(self.visualize_env_ids) == 0:
            return  # Empty list means no visualization
        else:
            env_ids = [i for i in self.visualize_env_ids if i < num_envs]
        
        if not env_ids:
            return
        
        W = target_points_w.shape[1]
        P = target_points_w.shape[2]
        E = len(env_ids)
        device = target_points_w.device
        
        # Gather points for selected envs
        target_pts = target_points_w[env_ids]  # (E, W, P, 3)
        
        # Flatten: (E, W, P, 3) -> (E*W*P, 3)
        positions = target_pts.reshape(-1, 3)
        
        # Marker indices: [0,0,...,1,1,...,W-1,W-1,...] repeated E times
        single_env_indices = torch.arange(W, device=device).repeat_interleave(P)  # (W*P,)
        marker_indices = single_env_indices.repeat(E)  # (E*W*P,)
        
        self.markers.visualize(
            translations=positions,
            marker_indices=marker_indices,
        )


def fingers_contact_force_b(
    env: ManagerBasedRLEnv,
    contact_sensor_names: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """base-frame contact forces from listed sensors, concatenated per env.

    Args:
        env: The environment.
        contact_sensor_names: Names of contact sensors in ``env.scene.sensors`` to read.

    Returns:
        Tensor of shape ``(num_envs, 3 * num_sensors)`` with forces stacked horizontally as
        ``[fx, fy, fz]`` per sensor.
    """
    force_w = [env.scene.sensors[name].data.force_matrix_w.view(env.num_envs, 3) for name in contact_sensor_names]
    force_w = torch.stack(force_w, dim=1)
    robot: Articulation = env.scene[asset_cfg.name]
    forces_b = quat_apply_inverse(robot.data.root_link_quat_w.unsqueeze(1).repeat(1, force_w.shape[1], 1), force_w)
    return forces_b


def allegro_hand_eigen_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    arm_joint_count: int = 7,
    hand_joint_count: int = 16,
    eigen_dim: int = 5,
    use_default_delta: bool = True,
) -> torch.Tensor:
    """Project current Allegro hand joints into eigen-grasp coefficients.

    We compute eigen coefficients from the current hand joint *delta* (relative to default pose),
    by least-squares projection onto the first ``eigen_dim`` PCA components:

        hand_delta ≈ eigen @ PCA

    where ``PCA`` is ``ALLEGRO_PCA_MATRIX[:eigen_dim]`` with shape (eigen_dim, 16).

    Args:
        env: The environment.
        asset_cfg: Scene entity for the robot articulation. Defaults to ``SceneEntityCfg("robot")``.
        arm_joint_count: Number of arm joints at the front of the joint array. Defaults to 7.
        hand_joint_count: Number of Allegro hand joints after the arm joints. Defaults to 16.
        eigen_dim: Number of eigen dimensions to use (1..5). Defaults to 5.
        use_default_delta: If True, subtract default joint positions before projection. Defaults to True.

    Returns:
        Tensor of shape (num_envs, eigen_dim) with eigen coefficients.
    """
    if not (1 <= eigen_dim <= int(ALLEGRO_PCA_MATRIX.shape[0])):
        raise ValueError(f"Invalid eigen_dim={eigen_dim}. Must be in [1, {int(ALLEGRO_PCA_MATRIX.shape[0])}].")
    if hand_joint_count != int(ALLEGRO_PCA_MATRIX.shape[1]):
        raise ValueError(
            f"hand_joint_count={hand_joint_count} does not match PCA width={int(ALLEGRO_PCA_MATRIX.shape[1])}."
        )

    robot: Articulation = env.scene[asset_cfg.name]
    
    # Use explicit joint IDs to ensure correct order (not PhysX native order)
    joint_ids = get_allegro_hand_joint_ids(env, robot)
    hand = robot.data.joint_pos[:, joint_ids]  # (N, 16) in canonical order
    if use_default_delta:
        hand = hand - robot.data.default_joint_pos[:, joint_ids]

    # Cache the right-inverse per eigen_dim on the env instance.
    cache_attr = f"_allegro_pca_right_pinv_{eigen_dim}"
    right_pinv = getattr(env, cache_attr, None)
    if right_pinv is None or right_pinv.device != env.device:
        A = ALLEGRO_PCA_MATRIX[:eigen_dim].to(env.device)  # (eigen_dim, 16)
        # right_pinv = A.T @ (A @ A.T)^-1  -> (16, eigen_dim)
        right_pinv = A.T @ torch.linalg.inv(A @ A.T)
        setattr(env, cache_attr, right_pinv)

    return hand @ right_pinv


class target_sequence_obs_b(ManagerTermBase):
    """Trajectory target observations expressed in robot's base frame.
    
    Supports two modes based on config mode.use_point_cloud:
    - Point cloud mode (default): Transforms object point cloud to each target pose.
      Returns (num_envs, window_size * num_points * 3).
    - Pose mode: Returns target poses directly.
      Returns (num_envs, window_size * 7).
    
    Also caches errors for reward/termination functions:
    - Raw window errors: env._cached_mean_errors (N, W) and env._cached_pose_errors
    - Aligned current errors: env._cached_aligned_mean_error and env._cached_aligned_pose_errors
    
    Args (from ``cfg.params``):
        object_cfg: Scene entity for the object. Defaults to ``SceneEntityCfg("object")``.
        ref_asset_cfg: Reference frame (robot). Defaults to ``SceneEntityCfg("robot")``.
    
    Returns:
        Point cloud mode: Tensor of shape ``(num_envs, window_size * num_points * 3)``.
        Pose mode: Tensor of shape ``(num_envs, window_size * 7)``.
    """
    
    # Inferno colormap (dark purple → bright yellow)
    INFERNO_COLORS = [
        (0.07, 0.04, 0.11),  # 0: nearly black
        (0.32, 0.06, 0.38),  # 1: dark purple
        (0.55, 0.09, 0.38),  # 2: magenta
        (0.75, 0.19, 0.27),  # 3: red-purple
        (0.89, 0.35, 0.13),  # 4: orange-red
        (0.96, 0.55, 0.04),  # 5: orange
        (0.99, 0.77, 0.17),  # 6: yellow-orange
        (0.99, 0.99, 0.64),  # 7: bright yellow
    ]
    CURRENT_OBJECT_COLOR = (1.0, 0.0, 0.0)  # Full red for current object
    POINT_RADIUS = 0.0025
    
    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        
        # Import here to avoid circular import
        from ..trajectory_manager import TrajectoryManager
        
        # Get configuration from env.cfg.y2r_cfg (set in TrajectoryEnvCfg.__post_init__)
        y2r_cfg = env.cfg.y2r_cfg
        
        self.object_cfg: SceneEntityCfg = cfg.params.get("object_cfg", SceneEntityCfg("object"))
        self.ref_asset_cfg: SceneEntityCfg = cfg.params.get("ref_asset_cfg", SceneEntityCfg("robot"))
        
        # Use values from YAML config
        self.y2r_cfg = y2r_cfg
        self.use_point_cloud = y2r_cfg.mode.use_point_cloud
        self.num_points = y2r_cfg.observations.num_points
        self.window_size = y2r_cfg.trajectory.window_size
        self.visualize = y2r_cfg.visualization.targets
        self.visualize_current = y2r_cfg.visualization.current_object
        self.visualize_waypoint_region = y2r_cfg.visualization.waypoint_region
        self.visualize_goal_region = y2r_cfg.visualization.goal_region
        self.visualize_pose_axes = y2r_cfg.visualization.pose_axes
        self.visualize_hand_pose_targets = y2r_cfg.visualization.hand_pose_targets
        self.visualize_grasp_surface_point = y2r_cfg.visualization.grasp_surface_point
        self.visualize_contact_forces = y2r_cfg.visualization.contact_forces
        self.visualize_env_ids = y2r_cfg.visualization.env_ids
        
        self.object: RigidObject = env.scene[self.object_cfg.name]
        self.ref_asset: Articulation = env.scene[self.ref_asset_cfg.name]
        self.env = env
        self.dt = env.step_dt
        
        # Create trajectory manager and attach to env for other components to access
        self.trajectory_manager = TrajectoryManager(
            cfg=y2r_cfg,
            num_envs=env.num_envs,
            device=env.device,
            table_height=y2r_cfg.workspace.table_surface_z,
            object_prim_path=self.object.cfg.prim_path,
        )
        env.trajectory_manager = self.trajectory_manager
        
        # Point cloud initialization (always needed for visualization)
        self.points_local = None
        # Use cached points from object_point_cloud_b (must be initialized first)
        # This ensures SAME local points are used for both current object and targets
        if hasattr(env, '_object_points_local'):
            self.points_local = env._object_points_local
        else:
            # Fallback if object_point_cloud_b not initialized yet
            all_env_ids = list(range(env.num_envs))

            # Get filter config from Y2R config (already a dict)
            filter_config = y2r_cfg.observations.point_cloud_filter

            self.points_local = sample_object_point_cloud_random(
                all_env_ids, self.num_points, self.object.cfg.prim_path,
                device=env.device, filter_config=filter_config
            )
            env._object_points_local = self.points_local
        
        # Pass point cloud to trajectory manager for penetration checking
        self.trajectory_manager.points_local = self.points_local
        
        # Visualizers
        self.markers = None
        self.grasp_point_marker = None
        self._debug_draw = None
        self._visualizer_initialized = False
        if self.visualize or self.visualize_current:
            self._init_visualizer()
        if self.visualize_grasp_surface_point:
            self._init_grasp_point_visualizer()
        if self.visualize_waypoint_region or self.visualize_goal_region or self.visualize_pose_axes or self.visualize_hand_pose_targets or self.visualize_contact_forces:
            self._init_region_visualizer()
        
        # Cache palm body index for hand trajectory
        _ensure_palm_offset_cache(y2r_cfg, env.device)
        palm_ids = self.ref_asset.find_bodies(y2r_cfg.robot.palm_body_name)[0]
        self._palm_body_idx = palm_ids[0] if len(palm_ids) > 0 else None

        # Expose render_debug on the env so external scripts (keyboard, etc.) can call it
        env.render_debug = self.render_debug

        # --- Episode metric accumulators (current episode, reset each episode) ---
        N = env.num_envs
        device = env.device
        self._ep_step_count = torch.zeros(N, device=device)
        self._ep_pos_error_sum = torch.zeros(N, device=device)
        self._ep_rot_error_sum = torch.zeros(N, device=device)
        self._ep_last_pos_error = torch.zeros(N, device=device)
        self._ep_last_rot_error = torch.zeros(N, device=device)
        self._ep_last_phase = torch.zeros(N, device=device)
        self._ep_reached_release = torch.zeros(N, dtype=torch.bool, device=device)
        self._ep_success_ever = torch.zeros(N, dtype=torch.bool, device=device)
        self._ep_grasp_contact_steps = torch.zeros(N, device=device)
        self._ep_grasp_total_steps = torch.zeros(N, device=device)

        # --- Last-episode snapshots (one per env, mean over all envs for logging) ---
        # Same approach as termination_manager._last_episode_dones:
        # updated at reset, reported as global mean for consistent denominators.
        self._last_success = torch.zeros(N, dtype=torch.bool, device=device)
        self._last_time_out = torch.zeros(N, dtype=torch.bool, device=device)
        self._last_reached_release = torch.zeros(N, dtype=torch.bool, device=device)
        self._last_phase = torch.zeros(N, device=device)
        self._last_mean_pos_error = torch.zeros(N, device=device)
        self._last_mean_rot_error = torch.zeros(N, device=device)
        self._last_final_pos_error = torch.zeros(N, device=device)
        self._last_final_rot_error = torch.zeros(N, device=device)
        self._last_grasp_contact_rate = torch.zeros(N, device=device)

    def reset(self, env_ids=None):
        """Compute episode metrics for resetting envs and reset accumulators.

        Called by ObservationManager.reset(). Returns a dict that flows into
        extras["log"] and gets logged to wandb automatically.

        Uses the same approach as TerminationManager._last_episode_dones:
        snapshot per-env metrics at reset, report mean over ALL envs so the
        denominator is consistent with Episode_Termination/* stats.
        """
        if env_ids is None:
            env_ids = torch.arange(self.env.num_envs, device=self.env.device)

        ids = env_ids
        if len(ids) > 0:
            mask = self._ep_step_count[ids] > 0
            if mask.any():
                valid = ids[mask]
                sc = self._ep_step_count[valid].clamp(min=1)

                # Snapshot finished-episode metrics into last-episode buffers
                # "True success" = succeeded and episode ended via timeout (successful termination).
                # This keeps success_rate semantics aligned with termination outcomes.
                if hasattr(self.env, "reset_time_outs") and self.env.reset_time_outs is not None:
                    time_out_valid = self.env.reset_time_outs[valid]
                else:
                    time_out_valid = torch.zeros_like(self._ep_success_ever[valid])
                self._last_time_out[valid] = time_out_valid
                self._last_success[valid] = self._ep_success_ever[valid] & time_out_valid
                self._last_reached_release[valid] = self._ep_reached_release[valid]
                self._last_phase[valid] = self._ep_last_phase[valid]
                self._last_mean_pos_error[valid] = self._ep_pos_error_sum[valid] / sc
                self._last_mean_rot_error[valid] = self._ep_rot_error_sum[valid] / sc
                self._last_final_pos_error[valid] = self._ep_last_pos_error[valid]
                self._last_final_rot_error[valid] = self._ep_last_rot_error[valid]
                grasp_total = self._ep_grasp_total_steps[valid].clamp(min=1)
                self._last_grasp_contact_rate[valid] = (
                    self._ep_grasp_contact_steps[valid] / grasp_total
                )

            # Reset accumulators for these env_ids
            self._ep_step_count[ids] = 0
            self._ep_pos_error_sum[ids] = 0
            self._ep_rot_error_sum[ids] = 0
            self._ep_last_pos_error[ids] = 0
            self._ep_last_rot_error[ids] = 0
            self._ep_last_phase[ids] = 0
            self._ep_reached_release[ids] = False
            self._ep_success_ever[ids] = False
            self._ep_grasp_contact_steps[ids] = 0
            self._ep_grasp_total_steps[ids] = 0

        # Report global mean over all envs (like Episode_Termination/*)
        metrics = {}
        metrics["Episode_Metric/success_rate"] = self._last_success.float().mean().item()
        metrics["Episode_Metric/reached_release"] = self._last_reached_release.float().mean().item()
        metrics["Episode_Metric/phase_at_termination"] = self._last_phase.mean().item()
        metrics["Episode_Metric/mean_pos_error"] = self._last_mean_pos_error.mean().item()
        metrics["Episode_Metric/mean_rot_error"] = self._last_mean_rot_error.mean().item()
        metrics["Episode_Metric/final_pos_error"] = self._last_final_pos_error.mean().item()
        metrics["Episode_Metric/final_rot_error"] = self._last_final_rot_error.mean().item()
        metrics["Episode_Metric/grasp_contact_rate"] = self._last_grasp_contact_rate.mean().item()

        return metrics

    def render_debug(self):
        """Draw debug visualizations (pose axes, regions) using current scene state.

        Can be called independently of __call__ — e.g., from keyboard debug scripts
        that bypass env.step(). Reads all data from scene/trajectory_manager directly.
        """
        N = self.env.num_envs

        # Debug draw (pose axes + regions + contact forces)
        if self._debug_draw is not None and (
            self.visualize_pose_axes or self.visualize_hand_pose_targets
            or self.visualize_waypoint_region or self.visualize_goal_region
            or self.visualize_contact_forces
        ):
            self._debug_draw.clear_lines()

            if self.visualize_pose_axes or self.visualize_hand_pose_targets:
                object_pos_w = self.object.data.root_pos_w
                object_quat_w = self.object.data.root_quat_w
                window_targets = self.trajectory_manager.get_window_targets()
                target_pos_w = window_targets[:, :, :3]
                target_quat_w = window_targets[:, :, 3:7]
                self._visualize_pose_axes(object_pos_w, object_quat_w, target_pos_w, target_quat_w, N)

            if self.visualize_waypoint_region or self.visualize_goal_region:
                self._visualize_regions(self.env.scene.env_origins, self.trajectory_manager.start_poses[:, :3])

            if self.visualize_contact_forces:
                self._visualize_contact_forces(N)

        # Grasp surface point (marker-based)
        if self.visualize_grasp_surface_point and self.grasp_point_marker is not None:
            self._visualize_grasp_surface_point(N)

    def _init_visualizer(self):
        """Initialize visualization markers for targets + current object."""
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        
        # Target markers (Inferno colors) + current object marker (red)
        markers_dict = {
            f"target_{i}": sim_utils.SphereCfg(
                radius=self.POINT_RADIUS,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=self.INFERNO_COLORS[i]),
            )
            for i in range(self.window_size)
        }
        markers_dict["current"] = sim_utils.SphereCfg(
            radius=self.POINT_RADIUS,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=self.CURRENT_OBJECT_COLOR,
            ),
        )
        
        self.markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/TargetSequencePointClouds",
                markers=markers_dict,
            )
        )
    
    def _init_grasp_point_visualizer(self):
        """Initialize visualization markers for grasp and keypoint surface points."""
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        # Colors: green for initial grasp, yellow/orange for keypoints
        keypoint_colors = [
            (1.0, 0.8, 0.0),  # Yellow - keypoint 0
            (1.0, 0.5, 0.0),  # Orange - keypoint 1
        ]

        markers_dict = {
            "grasp_point": sim_utils.SphereCfg(
                radius=self.POINT_RADIUS * 2.0,  # Double size
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 0.0),  # Green
                ),
            ),
        }

        # Add keypoint markers
        for i, color in enumerate(keypoint_colors):
            markers_dict[f"keypoint_{i}"] = sim_utils.SphereCfg(
                radius=self.POINT_RADIUS * 1.5,  # Slightly smaller than grasp
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=color,
                ),
            )

        self.grasp_point_marker = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/GraspSurfacePoint",
                markers=markers_dict,
            )
        )
    
    def _init_region_visualizer(self):
        """Initialize debug draw for waypoint/goal region visualization."""
        try:
            # Try different import paths for different Isaac Sim versions
            try:
                import isaacsim.util.debug_draw._debug_draw as omni_debug_draw
            except ModuleNotFoundError:
                try:
                    from omni.isaac.debug_draw import _debug_draw as omni_debug_draw
                except ModuleNotFoundError:
                    from omni.debugdraw import get_debug_draw_interface
                    self._debug_draw = get_debug_draw_interface()
                    return
            self._debug_draw = omni_debug_draw.acquire_debug_draw_interface()
        except Exception as e:
            print(f"[WARN] Could not initialize debug draw: {e}. Region visualization disabled.")
            self._debug_draw = None
    
    def _visualize_regions(self, env_origins: torch.Tensor, start_pos: torch.Tensor):
        """Visualize waypoint and goal sampling regions using debug lines."""
        if self._debug_draw is None:
            return
        
        cfg = self.y2r_cfg
        device = env_origins.device
        N = env_origins.shape[0]
        
        # Select envs to visualize
        if self.visualize_env_ids is None:
            env_ids = torch.arange(N, device=device)
        else:
            env_ids = torch.tensor([i for i in self.visualize_env_ids if i < N], device=device)
        E = len(env_ids)
        if E == 0:
            return
        
        origins = env_origins[env_ids]  # (E, 3)
        starts = start_pos[env_ids]     # (E, 3)
        
        # Note: clear_lines is done in _visualize_pose_axes which runs every frame
        
        # Waypoint region: 3D box (12 edges)
        if self.visualize_waypoint_region:
            # Find random_waypoint segments for position_range
            random_wp_segments = [seg for seg in cfg.trajectory.segments if seg.type == "random_waypoint"]
            if not random_wp_segments:
                # No random waypoints, skip visualization
                return

            wp_cfg = random_wp_segments[0]  # Use first random_waypoint segment
            start_local = starts - origins

            # X bounds (None = workspace, else offset from start)
            if wp_cfg.position_range.x is None:
                x_min = torch.full((E,), cfg.workspace.x[0], device=device)
                x_max = torch.full((E,), cfg.workspace.x[1], device=device)
            else:
                x_min = torch.clamp(start_local[:, 0] + wp_cfg.position_range.x[0], min=cfg.workspace.x[0])
                x_max = torch.clamp(start_local[:, 0] + wp_cfg.position_range.x[1], max=cfg.workspace.x[1])
            
            # Y bounds
            if wp_cfg.position_range.y is None:
                y_min = torch.full((E,), cfg.workspace.y[0], device=device)
                y_max = torch.full((E,), cfg.workspace.y[1], device=device)
            else:
                y_min = torch.clamp(start_local[:, 1] + wp_cfg.position_range.y[0], min=cfg.workspace.y[0])
                y_max = torch.clamp(start_local[:, 1] + wp_cfg.position_range.y[1], max=cfg.workspace.y[1])
            
            # Z bounds
            if wp_cfg.position_range.z is None:
                z_min = torch.full((E,), cfg.workspace.z[0], device=device)
                z_max = torch.full((E,), cfg.workspace.z[1], device=device)
            else:
                z_min = torch.full((E,), max(wp_cfg.position_range.z[0], cfg.workspace.z[0]), device=device)
                z_max = torch.full((E,), min(wp_cfg.position_range.z[1], cfg.workspace.z[1]), device=device)
            
            wp_min = torch.stack([x_min, y_min, z_min], dim=-1) + origins
            wp_max = torch.stack([x_max, y_max, z_max], dim=-1) + origins
            
            starts_wp, ends_wp = self._box_edges_lines(wp_min, wp_max)
            color_wp = (0.2, 0.9, 0.2, 1.0)  # Green
            self._draw_lines(starts_wp, ends_wp, color_wp)
        
        # Goal region: 2D rectangle on table (4 edges)
        if self.visualize_goal_region:
            x_range = cfg.goal.x_range if cfg.goal.x_range is not None else cfg.workspace.x
            y_range = cfg.goal.y_range if cfg.goal.y_range is not None else cfg.workspace.y
            
            g_min = origins[:, :2] + torch.tensor([[x_range[0], y_range[0]]], device=device)
            g_max = origins[:, :2] + torch.tensor([[x_range[1], y_range[1]]], device=device)
            g_z = cfg.workspace.table_surface_z + origins[:, 2] + 0.01
            
            starts_g, ends_g = self._rect_edges_lines(g_min, g_max, g_z)
            color_g = (0.2, 0.9, 0.9, 1.0)  # Cyan
            self._draw_lines(starts_g, ends_g, color_g)
    
    def _draw_lines(self, starts: torch.Tensor, ends: torch.Tensor, color: tuple):
        """Draw lines using debug draw interface."""
        # Non-blocking transfer + direct tolist (skip numpy intermediate)
        starts_list = starts.cpu().tolist()
        ends_list = ends.cpu().tolist()
        n = len(starts_list)
        self._debug_draw.draw_lines(starts_list, ends_list, [color] * n, [2.0] * n)
    
    def _box_edges_lines(self, mins: torch.Tensor, maxs: torch.Tensor):
        """Get start/end points for 12 edges of 3D boxes. Returns (E*12, 3) each."""
        E = mins.shape[0]
        x0, y0, z0 = mins[:, 0], mins[:, 1], mins[:, 2]
        x1, y1, z1 = maxs[:, 0], maxs[:, 1], maxs[:, 2]
        
        # 8 vertices of box: (E, 8, 3)
        verts = torch.stack([
            torch.stack([x0, y0, z0], dim=-1),  # 0: min corner
            torch.stack([x1, y0, z0], dim=-1),  # 1
            torch.stack([x0, y1, z0], dim=-1),  # 2
            torch.stack([x1, y1, z0], dim=-1),  # 3
            torch.stack([x0, y0, z1], dim=-1),  # 4
            torch.stack([x1, y0, z1], dim=-1),  # 5
            torch.stack([x0, y1, z1], dim=-1),  # 6
            torch.stack([x1, y1, z1], dim=-1),  # 7: max corner
        ], dim=1)  # (E, 8, 3)
        
        # 12 edges: pairs of vertex indices
        edge_pairs = [(0,1), (2,3), (4,5), (6,7),  # X edges
                      (0,2), (1,3), (4,6), (5,7),  # Y edges
                      (0,4), (1,5), (2,6), (3,7)]  # Z edges
        
        starts = torch.cat([verts[:, i] for i, j in edge_pairs], dim=0)  # (E*12, 3)
        ends = torch.cat([verts[:, j] for i, j in edge_pairs], dim=0)    # (E*12, 3)
        return starts, ends
    
    def _rect_edges_lines(self, mins: torch.Tensor, maxs: torch.Tensor, z: torch.Tensor):
        """Get start/end points for 4 edges of 2D rectangles. Returns (E*4, 3) each."""
        E = mins.shape[0]
        x0, y0 = mins[:, 0], mins[:, 1]
        x1, y1 = maxs[:, 0], maxs[:, 1]
        
        # 4 vertices: (E, 4, 3)
        verts = torch.stack([
            torch.stack([x0, y0, z], dim=-1),  # 0: bottom-left
            torch.stack([x1, y0, z], dim=-1),  # 1: bottom-right
            torch.stack([x0, y1, z], dim=-1),  # 2: top-left
            torch.stack([x1, y1, z], dim=-1),  # 3: top-right
        ], dim=1)  # (E, 4, 3)
        
        # 4 edges
        edge_pairs = [(0,1), (2,3), (0,2), (1,3)]
        starts = torch.cat([verts[:, i] for i, j in edge_pairs], dim=0)
        ends = torch.cat([verts[:, j] for i, j in edge_pairs], dim=0)
        return starts, ends
    
    def _visualize_contact_forces(self, num_envs: int):
        """Visualize contact force direction (cyan) and pad normal (yellow) on finger links.

        Reads debug data stored by _get_finger_contact_mags() in rewards.py.
        Only draws arrows for links with contact magnitude > 0.1N.
        """
        if self._debug_draw is None:
            return
        debug_data = getattr(self.env, '_contact_debug_data', None)
        if debug_data is None:
            return

        # Select envs to visualize
        if self.visualize_env_ids is None:
            env_ids = list(range(num_envs))
        else:
            env_ids = [i for i in self.visualize_env_ids if i < num_envs]
        if not env_ids:
            return

        ARROW_LEN = 0.04  # 4cm
        MIN_FORCE = 0.1   # Only draw if contact > 0.1N

        starts_list = []
        ends_list = []
        all_colors = []

        for link_name, link_pos, force_w, pad_w in debug_data:
            for ei in env_ids:
                mag = force_w[ei].norm().item()
                if mag < MIN_FORCE:
                    continue
                pos = link_pos[ei]
                # Cyan arrow: normalized force direction
                force_dir = force_w[ei] / mag
                force_end = pos + force_dir * ARROW_LEN
                starts_list.append(pos.cpu().tolist())
                ends_list.append(force_end.cpu().tolist())
                all_colors.append((0.0, 1.0, 1.0, 1.0))  # Cyan
                # Yellow arrow: pad normal direction
                pad_end = pos + pad_w[ei] * ARROW_LEN
                starts_list.append(pos.cpu().tolist())
                ends_list.append(pad_end.cpu().tolist())
                all_colors.append((1.0, 1.0, 0.0, 1.0))  # Yellow

        if starts_list:
            n = len(starts_list)
            self._debug_draw.draw_lines(starts_list, ends_list, all_colors, [5.0] * n)

    def _visualize_pose_axes(
        self,
        object_pos_w: torch.Tensor,
        object_quat_w: torch.Tensor,
        target_pos_w: torch.Tensor,
        target_quat_w: torch.Tensor,
        num_envs: int,
    ):
        """Visualize pose axes (XYZ arrows) for current object, targets, and palm.
        
        Args:
            object_pos_w: (N, 3) current object position in world frame.
            object_quat_w: (N, 4) current object quaternion in world frame.
            target_pos_w: (N, W, 3) target positions in world frame.
            target_quat_w: (N, W, 4) target quaternions in world frame.
            num_envs: Number of environments.
        """
        if self._debug_draw is None:
            return
        
        # Axis arrow lengths
        AXIS_LENGTH = 0.08        # Object and target axes
        PALM_AXIS_LENGTH = 0.10   # Palm axes (medium size)
        
        # Select envs to visualize
        if self.visualize_env_ids is None:
            env_ids = list(range(num_envs))
        else:
            env_ids = [i for i in self.visualize_env_ids if i < num_envs]
        
        if not env_ids:
            return
        
        E = len(env_ids)
        W = target_pos_w.shape[1]
        device = object_pos_w.device
        
        # Local axis directions
        x_axis = torch.tensor([1.0, 0.0, 0.0], device=device)
        y_axis = torch.tensor([0.0, 1.0, 0.0], device=device)
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=device)
        
        all_starts = []
        all_ends = []
        all_colors = []
        
        # === Current object axes (bright colors) ===
        obj_pos = object_pos_w[env_ids]  # (E, 3)
        obj_quat = object_quat_w[env_ids]  # (E, 4)
        
        # Transform local axes to world
        for axis, color in [(x_axis, (1.0, 0.0, 0.0, 1.0)),    # Red for X
                            (y_axis, (0.0, 1.0, 0.0, 1.0)),    # Green for Y
                            (z_axis, (0.0, 0.0, 1.0, 1.0))]:   # Blue for Z
            axis_exp = axis.unsqueeze(0).expand(E, 3)
            axis_world = quat_apply(obj_quat, axis_exp) * AXIS_LENGTH
            
            all_starts.append(obj_pos)
            all_ends.append(obj_pos + axis_world)
            all_colors.extend([color] * E)
        
        # === Target axes (dimmer colors, only first target for clarity) ===
        # Using first target in window (current target)
        tgt_pos = target_pos_w[env_ids, 0]  # (E, 3) - first target
        tgt_quat = target_quat_w[env_ids, 0]  # (E, 4)
        
        for axis, color in [(x_axis, (0.5, 0.0, 0.0, 0.8)),    # Dark red for X
                            (y_axis, (0.0, 0.5, 0.0, 0.8)),    # Dark green for Y
                            (z_axis, (0.0, 0.0, 0.5, 0.8))]:   # Dark blue for Z
            axis_exp = axis.unsqueeze(0).expand(E, 3)
            axis_world = quat_apply(tgt_quat, axis_exp) * AXIS_LENGTH
            
            all_starts.append(tgt_pos)
            all_ends.append(tgt_pos + axis_world)
            all_colors.extend([color] * E)
        
        # === Palm frame axes (standard RGB, medium size) ===
        if self._palm_body_idx is not None:
            palm_pos_w, palm_quat_w, _ = get_palm_frame_pose_w(self.ref_asset, self.y2r_cfg)

            palm_pos = palm_pos_w[env_ids]  # (E, 3)
            palm_quat = palm_quat_w[env_ids]  # (E, 4)
            
            # Standard RGB axes (10cm)
            # Based on observation: X = finger direction, Z = palm normal (out of palm)
            for axis, color in [(x_axis, (1.0, 0.0, 0.0, 1.0)),    # Red for X (finger direction)
                                (y_axis, (0.0, 1.0, 0.0, 1.0)),    # Green for Y
                                (z_axis, (0.0, 0.0, 1.0, 1.0))]:   # Blue for Z (palm normal)
                axis_exp = axis.unsqueeze(0).expand(E, 3)
                axis_world = quat_apply(palm_quat, axis_exp) * PALM_AXIS_LENGTH
                
                all_starts.append(palm_pos)
                all_ends.append(palm_pos + axis_world)
                all_colors.extend([color] * E)
        
        # === Hand pose TARGET axes (dim RGB, similar to object target) ===
        # Show ALL targets in the window, with decreasing brightness for future targets
        if self.visualize_hand_pose_targets:
            cfg = self.env.cfg.y2r_cfg
            if cfg.hand_trajectory.enabled:
                HAND_TARGET_AXIS_LENGTH = 0.10
                W = cfg.trajectory.window_size
                
                # Get all hand pose targets in window
                hand_targets = self.trajectory_manager.get_hand_window_targets()  # (N, W, 7)
                
                for w in range(W):
                    hand_pos = hand_targets[env_ids, w, :3]  # (E, 3)
                    hand_quat = hand_targets[env_ids, w, 3:7]  # (E, 4)
                    
                    # Brightness decreases for future targets (1.0 -> 0.3)
                    brightness = 0.6 - 0.4 * (w / max(W - 1, 1))
                    
                    # Dim RGB axes
                    for axis, base_color in [(x_axis, (1.0, 0.0, 0.0)),    # Red for X
                                             (y_axis, (0.0, 1.0, 0.0)),    # Green for Y
                                             (z_axis, (0.0, 0.0, 1.0))]:   # Blue for Z
                        color = (base_color[0] * brightness, base_color[1] * brightness, base_color[2] * brightness, 0.8)
                        axis_exp = axis.unsqueeze(0).expand(E, 3)
                        axis_world = quat_apply(hand_quat, axis_exp) * HAND_TARGET_AXIS_LENGTH
                        
                        all_starts.append(hand_pos)
                        all_ends.append(hand_pos + axis_world)
                        all_colors.extend([color] * E)
        
        # Concatenate and draw
        starts = torch.cat(all_starts, dim=0)  # (9*E, 3) with palm axes
        ends = torch.cat(all_ends, dim=0)
        
        starts_list = starts.cpu().tolist()
        ends_list = ends.cpu().tolist()
        n = len(starts_list)
        self._debug_draw.draw_lines(starts_list, ends_list, all_colors, [4.0] * n)
    
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        ref_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        """Compute target sequence observations in robot base frame.
        
        Also manages trajectory manager: resets on env reset, steps each call.
        
        Returns:
            Point cloud mode: Flattened tensor (num_envs, window_size * num_points * 3).
            Pose mode: Flattened tensor (num_envs, window_size * 7).
        """
        N = env.num_envs
        W = self.window_size
        
        # Handle resets: check which envs just reset (episode_length_buf == 0)
        just_reset = env.episode_length_buf == 0
        reset_ids = just_reset.nonzero(as_tuple=True)[0]
        
        if len(reset_ids) > 0:
            # Get current object poses for resetting envs
            object_pos = self.object.data.root_pos_w[reset_ids]
            object_quat = self.object.data.root_quat_w[reset_ids]
            start_poses = torch.cat([object_pos, object_quat], dim=-1)
            
            # Get environment origins
            env_origins = env.scene.env_origins[reset_ids]
            
            # Get current palm poses for hand trajectory (if enabled)
            start_palm_poses = None
            if self.y2r_cfg.hand_trajectory.enabled and self._palm_body_idx is not None:
                palm_pos_all, palm_quat_all, _ = get_palm_frame_pose_w(self.ref_asset, self.y2r_cfg)
                palm_pos = palm_pos_all[reset_ids]
                palm_quat = palm_quat_all[reset_ids]
                start_palm_poses = torch.cat([palm_pos, palm_quat], dim=-1)
            
            difficulties = None
            curriculum_manager = getattr(env, "curriculum_manager", None)
            if curriculum_manager is not None:
                adr_cfg = getattr(curriculum_manager, "cfg", None)
                adr = getattr(adr_cfg, "adr", None) if adr_cfg is not None else None
                scheduler = getattr(adr, "func", None) if adr is not None else None
                if scheduler is not None and hasattr(scheduler, "current_adr_difficulties"):
                    difficulties = scheduler.current_adr_difficulties[reset_ids]

            # Scales are read once at trajectory_manager init from USD prims
            self.trajectory_manager.reset(
                reset_ids,
                start_poses,
                env_origins,
                start_palm_poses,
                difficulties,
            )
        
        # Step trajectory (advance time)
        # For push-T mode: pass current object poses for replanning
        cfg = env.cfg.y2r_cfg
        if cfg.push_t.enabled:
            object_pos = self.object.data.root_pos_w
            object_quat = self.object.data.root_quat_w
            object_poses = torch.cat([object_pos, object_quat], dim=-1)
            self.trajectory_manager.step(self.dt, object_poses=object_poses)
        else:
            self.trajectory_manager.step(self.dt)
        
        # Get window targets: (N, W, 7) - pos(3) + quat(4)
        window_targets = self.trajectory_manager.get_window_targets()
        
        # Current object pose
        object_pos_w = self.object.data.root_pos_w  # (N, 3)
        object_quat_w = self.object.data.root_quat_w  # (N, 4)
        
        # Reference frame
        ref_pos_w = self.ref_asset.data.root_pos_w   # (N, 3)
        ref_quat_w = self.ref_asset.data.root_quat_w  # (N, 4)
        
        # Always compute point cloud observations and cache BOTH error types.
        # Rewards/terminations use aligned errors (path-aware if enabled).
        obs = self._compute_observations(
            env, N, W, window_targets, object_pos_w, object_quat_w, ref_pos_w, ref_quat_w
        )

        # --- Accumulate episode metrics (uses cached errors from _compute_observations) ---
        pos_error = env._cached_aligned_pose_errors['pos']  # (N,)
        rot_error = env._cached_aligned_pose_errors['rot']  # (N,)
        phase = self.trajectory_manager.get_phase().float()  # (N,) 0=grasp,1=manip,2=release

        self._ep_step_count += 1
        self._ep_pos_error_sum += pos_error
        self._ep_rot_error_sum += rot_error
        self._ep_last_pos_error = pos_error
        self._ep_last_rot_error = rot_error
        self._ep_last_phase = phase

        # Track if ever reached release phase
        self._ep_reached_release |= (phase >= 2)

        # Success detection: object at goal + in release phase
        cfg = env.cfg.y2r_cfg
        success_pos_thresh = getattr(cfg, '_ep_metric_pos_thresh', 0.05)
        success_rot_thresh = getattr(cfg, '_ep_metric_rot_thresh', 0.3)
        # Check reward config for trajectory_success thresholds
        if not hasattr(self, '_success_thresholds_cached'):
            self._success_thresholds_cached = True
            reward_cfg = getattr(env.cfg, 'rewards', None)
            if reward_cfg is not None:
                ts_cfg = getattr(reward_cfg, 'trajectory_success', None)
                if ts_cfg is not None:
                    params = getattr(ts_cfg, 'params', {})
                    success_pos_thresh = params.get('pos_threshold', 0.05)
                    success_rot_thresh = params.get('rot_threshold', 0.3)
            self._success_pos_thresh = success_pos_thresh
            self._success_rot_thresh = success_rot_thresh
        in_release = (phase >= 2)
        pos_ok = (pos_error < self._success_pos_thresh)
        rot_ok = (rot_error < self._success_rot_thresh)
        self._ep_success_ever |= (in_release & pos_ok & rot_ok)

        # Contact tracking during grasp phase
        in_grasp = (phase < 1)
        self._ep_grasp_total_steps += in_grasp.float()
        try:
            from .rewards import contact_factor
            cfac = contact_factor(env)
            self._ep_grasp_contact_steps += (in_grasp.float() * cfac)
        except Exception:
            pass  # Contact sensors may not be available

        return obs
    
    def _compute_observations(
        self,
        env: ManagerBasedRLEnv,
        N: int,
        W: int,
        window_targets: torch.Tensor,
        object_pos_w: torch.Tensor,
        object_quat_w: torch.Tensor,
        ref_pos_w: torch.Tensor,
        ref_quat_w: torch.Tensor,
    ) -> torch.Tensor:
        """Compute point cloud observations and cache errors used by rewards/terminations.
        
        Always outputs point clouds. Caches both point cloud and pose errors, plus a
        shared aligned current error (path-aware if enabled).
        """
        P = self.num_points
        
        # Target poses in world frame
        target_pos_w = window_targets[:, :, :3]   # (N, W, 3)
        target_quat_w = window_targets[:, :, 3:7]  # (N, W, 4)
        
        # === Point cloud computation ===
        target_pos = target_pos_w.unsqueeze(2)   # (N, W, 1, 3)
        target_quat = target_quat_w.unsqueeze(2)  # (N, W, 1, 4)
        
        local_exp = self.points_local.unsqueeze(1).expand(N, W, P, 3)  # (N, W, P, 3)

        # Flatten (N, W) → (N*W) to use broadcast utilities, avoiding (N, W, P) expand
        target_quat_flat = target_quat_w.reshape(-1, 4)  # (N*W, 4)
        target_pos_flat = target_pos_w.reshape(-1, 3)    # (N*W, 3)
        local_flat = local_exp.reshape(N * W, P, 3)      # (N*W, P, 3)

        all_points_w_flat = _quat_rotate_vec(target_quat_flat, local_flat) + target_pos_flat.unsqueeze(1)
        all_points_w = all_points_w_flat.reshape(N, W, P, 3)

        # Transform to robot base frame (expand only (N)→(N*W), not (N)→(N*W*P))
        ref_pos_flat = ref_pos_w.unsqueeze(1).expand(N, W, 3).reshape(-1, 3)
        ref_quat_flat = ref_quat_w.unsqueeze(1).expand(N, W, 4).reshape(-1, 4)

        all_points_b = _points_to_frame(ref_pos_flat, ref_quat_flat, all_points_w_flat).reshape(N, W, P, 3)

        # Current object points (world space, broadcast)
        current_points_w = _quat_rotate_vec(object_quat_w, self.points_local) + object_pos_w.unsqueeze(1)
        
        # === Cache point cloud errors ===
        current_exp = current_points_w.unsqueeze(1).expand(N, W, P, 3)
        point_errors = (current_exp - all_points_w).norm(dim=-1)  # (N, W, P)
        env._cached_mean_errors = point_errors.mean(dim=-1)  # (N, W)
        
        # === Cache pose errors ===
        # Position error: (N, W)
        object_pos_exp_w = object_pos_w.unsqueeze(1).expand(N, W, 3)
        pos_errors = (object_pos_exp_w - target_pos_w).norm(dim=-1)  # (N, W)
        
        # Rotation error using quat_error_magnitude: (N, W)
        object_quat_exp_w = object_quat_w.unsqueeze(1).expand(N, W, 4)
        rot_errors = quat_error_magnitude(
            object_quat_exp_w.reshape(-1, 4),
            target_quat_w.reshape(-1, 4)
        ).reshape(N, W)  # (N, W)
        
        env._cached_pose_errors = {
            'pos': pos_errors,  # (N, W) position error in meters
            'rot': rot_errors,  # (N, W) rotation error in radians
        }

        # === Cache aligned current errors (shared across rewards/terminations) ===
        cfg: Y2RConfig = env.cfg.y2r_cfg
        path_mode = getattr(cfg.trajectory, "path_mode", False)
        timing_aware = getattr(cfg.trajectory, "timing_aware", True)

        if path_mode and W >= 2:
            # Current segment: target[0] -> target[1]
            p0 = target_pos_w[:, 0, :]  # (N, 3)
            p1 = target_pos_w[:, 1, :]  # (N, 3)
            q0 = target_quat_w[:, 0, :]  # (N, 4)
            q1 = target_quat_w[:, 1, :]  # (N, 4)

            if timing_aware:
                t = self.trajectory_manager.get_segment_progress()  # (N,) in [0,1)
            else:
                v = p1 - p0
                v_len_sq = (v ** 2).sum(dim=-1, keepdim=True).clamp(min=1e-8)
                t = ((object_pos_w - p0) * v).sum(dim=-1, keepdim=True) / v_len_sq
                t = t.clamp(0.0, 1.0).squeeze(-1)

            interp_pos = p0 + t.unsqueeze(-1) * (p1 - p0)
            interp_quat = _batched_slerp(q0, q1, t)

            aligned_pos_error = (object_pos_w - interp_pos).norm(dim=-1)
            aligned_rot_error = quat_error_magnitude(object_quat_w, interp_quat)
            env._cached_aligned_pose_errors = {
                "pos": aligned_pos_error,
                "rot": aligned_rot_error,
            }

            points_local = self.points_local
            if points_local.dim() == 2:
                points_local = points_local.unsqueeze(0).expand(N, -1, -1)
            elif points_local.shape[0] != N:
                points_local = points_local.expand(N, -1, -1)

            target_points_w = quat_apply(
                interp_quat.unsqueeze(1).expand(-1, P, -1),
                points_local,
            ) + interp_pos.unsqueeze(1)
            aligned_point_errors = (current_points_w - target_points_w).norm(dim=-1)
            env._cached_aligned_mean_error = aligned_point_errors.mean(dim=-1)
        else:
            env._cached_aligned_pose_errors = {
                "pos": pos_errors[:, 0],
                "rot": rot_errors[:, 0],
            }
            env._cached_aligned_mean_error = env._cached_mean_errors[:, 0]

        # Cache aligned hand target (pose in world frame)
        if self._palm_body_idx is not None:
            palm_pos_w, _, _ = get_palm_frame_pose_w(self.ref_asset, self.y2r_cfg)
        else:
            palm_pos_w = self.ref_asset.data.root_pos_w  # Fallback (N, 3)
        env._cached_aligned_hand_target = _compute_aligned_hand_target(
            self.trajectory_manager,
            path_mode=path_mode,
            timing_aware=timing_aware,
            palm_pos_w=palm_pos_w,
        )
        
        # Visualization (point clouds every frame — depends on computed data)
        if (self.visualize or self.visualize_current) and self.markers is not None:
            self._visualize(all_points_w, current_points_w, N)

        # Debug draw visualizations (pose axes, regions, grasp point)
        self.render_debug()
        
        # Flatten: (N, W * P * 3)
        return all_points_b.reshape(N, -1)
    
    def _visualize(
        self,
        target_points_w: torch.Tensor,
        current_points_w: torch.Tensor,
        num_envs: int,
    ):
        """Visualize target and current point clouds.
        
        Args:
            target_points_w: (N, W, P, 3) all target point clouds in world space.
            current_points_w: (N, P, 3) current object points in world space.
            num_envs: Number of environments.
        """
        # Determine which envs to visualize
        if self.visualize_env_ids is None:
            env_ids = list(range(num_envs))
        else:
            env_ids = [i for i in self.visualize_env_ids if i < num_envs]
        
        if not env_ids:
            return
        
        W = target_points_w.shape[1]
        P = target_points_w.shape[2]
        E = len(env_ids)
        device = target_points_w.device
        
        # Gather points for selected envs (stay on GPU)
        target_pts = target_points_w[env_ids]  # (E, W, P, 3)
        current_pts = current_points_w[env_ids]  # (E, P, 3)
        
        # Build positions and indices based on what to visualize
        positions_list = []
        indices_list = []
        
        # Targets (if enabled)
        if self.visualize:
            target_flat = target_pts.reshape(-1, 3)  # (E*W*P, 3)
            single_env_indices = torch.arange(W, device=device).repeat_interleave(P)  # (W*P,)
            target_indices = single_env_indices.repeat(E)  # (E*W*P,)
            positions_list.append(target_flat)
            indices_list.append(target_indices)
        
        # Current object (if enabled)
        if self.visualize_current:
            current_flat = current_pts.reshape(-1, 3)  # (E*P, 3)
            current_indices = torch.full((E * P,), W, dtype=torch.long, device=device)
            positions_list.append(current_flat)
            indices_list.append(current_indices)
        
        if not positions_list:
            return
        
        all_positions = torch.cat(positions_list, dim=0)
        all_indices = torch.cat(indices_list)
        
        self.markers.visualize(
            translations=all_positions,
            marker_indices=all_indices,
        )
    
    def _visualize_grasp_surface_point(self, num_envs: int):
        """Visualize grasp and keypoint surface points.

        - Green sphere: initial grasp surface point
        - Yellow/Orange spheres: keypoint surface points

        Points are stored in object-local frame and transformed to world
        frame each frame so they move with the object.

        Args:
            num_envs: Number of environments.
        """
        # Determine which envs to visualize
        if self.visualize_env_ids is None:
            env_ids = list(range(num_envs))
        else:
            env_ids = [i for i in self.visualize_env_ids if i < num_envs]

        if not env_ids:
            return

        env_ids_t = torch.tensor(env_ids, device=self.object.device)
        E = len(env_ids)

        # Get current object pose
        obj_pos = self.object.data.root_pos_w[env_ids_t]  # (E, 3)
        obj_quat = self.object.data.root_quat_w[env_ids_t]  # (E, 4)

        # Get grasp surface point in LOCAL frame
        grasp_points_local = self.trajectory_manager.debug_surface_point_local[env_ids_t]  # (E, 3)
        grasp_points_w = quat_apply(obj_quat, grasp_points_local) + obj_pos  # (E, 3)

        # Get keypoint surface points in LOCAL frame
        keypoint_points_local = self.trajectory_manager.debug_keypoint_points_local[env_ids_t]  # (E, max_kp, 3)
        num_kp = self.trajectory_manager.num_grasp_keypoints[env_ids_t]  # (E,)
        max_kp = keypoint_points_local.shape[1]

        # Collect all points to visualize
        all_points = [grasp_points_w]
        all_indices = [torch.zeros(E, dtype=torch.long, device=self.object.device)]  # marker index 0 = grasp_point

        for kp_idx in range(max_kp):
            # Only include keypoints that exist for each env
            kp_mask = num_kp > kp_idx  # (E,)
            if kp_mask.any():
                kp_local = keypoint_points_local[:, kp_idx]  # (E, 3)
                kp_world = quat_apply(obj_quat, kp_local) + obj_pos  # (E, 3)

                # Only add points for envs that have this keypoint
                valid_envs = kp_mask.nonzero(as_tuple=True)[0]
                all_points.append(kp_world[valid_envs])
                # Marker index 1 = keypoint_0, index 2 = keypoint_1, etc.
                all_indices.append(torch.full((len(valid_envs),), kp_idx + 1, dtype=torch.long, device=self.object.device))

        # Concatenate all points and indices
        translations = torch.cat(all_points, dim=0)
        marker_indices = torch.cat(all_indices, dim=0)

        self.grasp_point_marker.visualize(
            translations=translations,
            marker_indices=marker_indices,
        )


def target_sequence_poses_b(
    env: ManagerBasedRLEnv,
    ref_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Target sequence poses in robot's base frame.
    
    Returns the window of target poses from the trajectory manager.
    Must be called AFTER target_sequence_obs_b which initializes the trajectory manager.
    
    Args:
        env: The environment (must have trajectory_manager attached).
        ref_asset_cfg: Reference frame (robot). Defaults to ``SceneEntityCfg("robot")``.
    
    Returns:
        Flattened tensor (num_envs, window_size * 7) - target poses in robot frame.
    """
    trajectory_manager = env.trajectory_manager
    ref_asset: Articulation = env.scene[ref_asset_cfg.name]
    
    # Get window size from env config
    cfg = env.cfg.y2r_cfg
    
    N = env.num_envs
    W = cfg.trajectory.window_size
    
    # Get window targets: (N, W, 7) - pos(3) + quat(4)
    window_targets = trajectory_manager.get_window_targets()
    target_pos_w = window_targets[:, :, :3]   # (N, W, 3)
    target_quat_w = window_targets[:, :, 3:7]  # (N, W, 4)
    
    # Reference frame
    ref_pos_w = ref_asset.data.root_pos_w   # (N, 3)
    ref_quat_w = ref_asset.data.root_quat_w  # (N, 4)
    
    # Transform target poses to robot base frame
    target_pos_rel = target_pos_w - ref_pos_w.unsqueeze(1)  # (N, W, 3)
    ref_quat_inv = quat_inv(ref_quat_w)  # (N, 4)
    
    # Rotate each target position to robot frame
    target_pos_b = quat_apply(
        ref_quat_inv.unsqueeze(1).expand(N, W, 4).reshape(-1, 4),
        target_pos_rel.reshape(-1, 3)
    ).reshape(N, W, 3)
    
    # Orientation: multiply by inverse robot quat
    target_quat_b = quat_mul(
        ref_quat_inv.unsqueeze(1).expand(N, W, 4).reshape(-1, 4),
        target_quat_w.reshape(-1, 4)
    ).reshape(N, W, 4)
    
    # Combine to poses in robot frame: (N, W, 7)
    target_poses_b = torch.cat([target_pos_b, target_quat_b], dim=-1)
    
    # Flatten: (N, W * 7)
    return target_poses_b.reshape(N, -1)


def trajectory_timing(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Current segment progress (0-1) within target interval.

    Provides temporal context for the policy: where it should be in the
    current target interval. Use with trajectory.timing_aware path tracking.

    0.0 = just reached current target (target[0])
    ~1.0 = about to reach next target (target[1])

    Must be called AFTER target_sequence_obs_b which initializes the trajectory manager.

    Returns:
        (num_envs, 1) tensor with segment progress in [0, 1).
    """
    return env.trajectory_manager.get_segment_progress().unsqueeze(-1)


def hand_pose_targets_b(
    env: ManagerBasedRLEnv,
    ref_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Hand (palm) pose targets in robot's base frame.
    
    Returns the window of target palm poses from the trajectory manager.
    Only active when hand_trajectory.enabled is True.
    
    Args:
        env: The environment (must have trajectory_manager attached).
        ref_asset_cfg: Reference frame (robot). Defaults to ``SceneEntityCfg("robot")``.
    
    Returns:
        Flattened tensor (num_envs, window_size * 7) - target palm poses in robot frame.
        Returns zeros if hand trajectory is disabled.
    """
    cfg = env.cfg.y2r_cfg
    N = env.num_envs
    W = cfg.trajectory.window_size
    
    # Return zeros if hand trajectory is disabled
    if not cfg.hand_trajectory.enabled:
        return torch.zeros(N, W * 7, device=env.device)
    
    trajectory_manager = env.trajectory_manager
    ref_asset: Articulation = env.scene[ref_asset_cfg.name]
    
    # Get hand window targets: (N, W, 7) - pos(3) + quat(4)
    hand_targets = trajectory_manager.get_hand_window_targets()
    target_pos_w = hand_targets[:, :, :3]   # (N, W, 3)
    target_quat_w = hand_targets[:, :, 3:7]  # (N, W, 4)
    
    # Reference frame
    ref_pos_w = ref_asset.data.root_pos_w   # (N, 3)
    ref_quat_w = ref_asset.data.root_quat_w  # (N, 4)
    ref_quat_inv = quat_inv(ref_quat_w)  # (N, 4)
    
    # Transform target positions to robot base frame
    target_pos_rel = target_pos_w - ref_pos_w.unsqueeze(1)  # (N, W, 3)
    target_pos_b = quat_apply(
        ref_quat_inv.unsqueeze(1).expand(N, W, 4).reshape(-1, 4),
        target_pos_rel.reshape(-1, 3)
    ).reshape(N, W, 3)
    
    # Transform orientations to robot base frame
    target_quat_b = quat_mul(
        ref_quat_inv.unsqueeze(1).expand(N, W, 4).reshape(-1, 4),
        target_quat_w.reshape(-1, 4)
    ).reshape(N, W, 4)
    
    # Combine to poses in robot frame: (N, W, 7)
    target_poses_b = torch.cat([target_pos_b, target_quat_b], dim=-1)
    
    # Flatten: (N, W * 7)
    return target_poses_b.reshape(N, -1)


# ==============================================================================
# Wrist Camera Observations
# ==============================================================================

# Global web viewer state
_web_viewer_app = None
_web_viewer_thread = None
_web_viewer_latest_image = None


def wrist_depth_image(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("wrist_camera"),
) -> torch.Tensor:
    """Wrist-mounted depth camera observation using TiledCamera.
    
    Returns 32x32 depth image normalized to [0, 1] based on clipping range.
    Optionally serves images via web viewer for remote visualization.
    
    Args:
        env: Environment instance
        sensor_cfg: Camera sensor configuration
    
    Returns:
        Flattened depth tensor (num_envs, resolution * resolution)
    """
    cfg = env.cfg.y2r_cfg.wrist_camera
    camera = env.scene.sensors[sensor_cfg.name]
    
    # Get depth: TiledCamera returns (num_envs, H, W, 1)
    depth = camera.data.output["distance_to_image_plane"]
    
    # Remove channel dimension and normalize
    depth = depth.squeeze(-1)  # (num_envs, H, W)
    near, far = cfg.clipping_range
    depth_normalized = (depth.clamp(near, far) - near) / (far - near)
    
    # Web viewer update
    if cfg.web_viewer:
        _update_web_viewer(env, depth_normalized, cfg)
    
    # Flatten: (N, resolution * resolution)
    return depth_normalized.reshape(env.num_envs, -1)


def _update_web_viewer(env, depth: torch.Tensor, cfg):
    """Update web viewer with latest depth image."""
    global _web_viewer_app, _web_viewer_thread, _web_viewer_latest_image
    
    # Start web viewer on first call
    if _web_viewer_app is None and cfg.web_viewer:
        _start_web_viewer(cfg.viewer_port)
    
    # Update at specified rate
    update_interval = int((1.0 / cfg.viewer_update_hz) / env.step_dt)
    if env.common_step_counter % max(1, update_interval) == 0:
        # Convert first env's depth to numpy
        depth_np = depth[0].cpu().numpy()
        
        # Convert to RGB for display using colormap
        import matplotlib.cm as cm
        colored = cm.viridis(depth_np)[:, :, :3]  # Drop alpha
        colored = (colored * 255).astype('uint8')
        
        # Update global image buffer
        _web_viewer_latest_image = {
            'image': colored,
            'step': env.common_step_counter,
            'min_depth': depth[0].min().item() * (cfg.clipping_range[1] - cfg.clipping_range[0]) + cfg.clipping_range[0],
            'max_depth': depth[0].max().item() * (cfg.clipping_range[1] - cfg.clipping_range[0]) + cfg.clipping_range[0],
        }


def _start_web_viewer(port: int):
    """Start Flask web server in background thread."""
    global _web_viewer_app, _web_viewer_thread
    
    from flask import Flask, Response
    import threading
    import io
    from PIL import Image
    
    app = Flask(__name__)
    
    @app.route('/')
    def index():
        return '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Wrist Camera Viewer</title>
            <style>
                body { 
                    background: #1e1e1e; 
                    color: #ffffff; 
                    font-family: monospace; 
                    text-align: center;
                    padding: 20px;
                }
                img { 
                    image-rendering: pixelated; 
                    width: 512px; 
                    height: 512px; 
                    border: 2px solid #444;
                }
                .info {
                    margin-top: 10px;
                    font-size: 14px;
                }
            </style>
        </head>
        <body>
            <h1>Wrist Camera - Depth View</h1>
            <img src="/stream" id="stream">
            <div class="info" id="info">Waiting for data...</div>
            <script>
                // Auto-refresh image (add timestamp to bust cache)
                setInterval(() => {
                    document.getElementById('stream').src = '/stream?' + Date.now();
                }, 100);
                // Auto-refresh info
                setInterval(() => {
                    fetch('/info')
                        .then(r => r.json())
                        .then(data => {
                            document.getElementById('info').innerHTML = 
                                `Step: ${data.step} | Depth: ${data.min_depth.toFixed(3)}m - ${data.max_depth.toFixed(3)}m`;
                        });
                }, 100);
            </script>
        </body>
        </html>
        '''
    
    @app.route('/stream')
    def stream():
        if _web_viewer_latest_image is None:
            # Return black image if no data yet
            img = Image.new('RGB', (32, 32), color='black')
        else:
            img = Image.fromarray(_web_viewer_latest_image['image'])
        
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        
        return Response(buf.getvalue(), mimetype='image/png')
    
    @app.route('/info')
    def info():
        if _web_viewer_latest_image is None:
            return {'step': 0, 'min_depth': 0.0, 'max_depth': 0.0}
        # Only return JSON-serializable fields (not the image array)
        return {
            'step': int(_web_viewer_latest_image['step']),
            'min_depth': float(_web_viewer_latest_image['min_depth']),
            'max_depth': float(_web_viewer_latest_image['max_depth']),
        }
    
    def run_server():
        app.run(host='0.0.0.0', port=port, threaded=True, use_reloader=False, debug=False)
    
    _web_viewer_app = app
    _web_viewer_thread = threading.Thread(target=run_server, daemon=True)
    _web_viewer_thread.start()
    
    print(f"\n{'='*60}")
    print(f"Wrist Camera Web Viewer started!")
    print(f"View at: http://localhost:{port}")
    print(f"Or forward port via SSH: ssh -L {port}:localhost:{port} user@remote")
    print(f"{'='*60}\n")
