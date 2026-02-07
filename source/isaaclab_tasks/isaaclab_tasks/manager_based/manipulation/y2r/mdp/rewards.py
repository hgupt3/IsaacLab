# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for Y2R trajectory following task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
import isaaclab.utils.math as math_utils
from isaaclab.utils.math import quat_apply_inverse, quat_inv, quat_mul, quat_error_magnitude

from .observations import get_palm_frame_pose_w

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Phase name to index mapping
PHASE_MAP = {"grasp": 0, "manipulation": 1, "release": 2}


def _get_cached_phase(env) -> torch.Tensor:
    """Return trajectory_manager.get_phase(), cached per step."""
    step = env.common_step_counter
    cached = getattr(env, '_cached_phase_data', None)
    if cached is not None and cached[0] == step:
        return cached[1]
    phase = env.trajectory_manager.get_phase()
    env._cached_phase_data = (step, phase)
    return phase


def _apply_phase_filter(
    env: ManagerBasedRLEnv,
    reward: torch.Tensor,
    phases: list[str] | None = None,
) -> torch.Tensor:
    """Zero out reward for envs not in specified phases.
    
    Args:
        env: The environment.
        reward: The reward tensor to filter.
        phases: List of phase names to be active in, e.g. ["grasp", "manipulation"].
                If None or empty, reward is active in all phases.
    
    Returns:
        Filtered reward tensor (num_envs,).
    """
    if phases is None or len(phases) == 0:
        return reward  # All phases active
    
    if not hasattr(env, 'trajectory_manager') or env.trajectory_manager is None:
        return reward

    phase = _get_cached_phase(env)  # (N,) - 0=grasp, 1=manip, 2=release

    active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for p in phases:
        if p in PHASE_MAP:
            active = active | (phase == PHASE_MAP[p])

    return torch.where(active, reward, torch.zeros_like(reward))


def _apply_hand_pose_gate(
    env: ManagerBasedRLEnv,
    reward: torch.Tensor,
    use_hand_pose_gate: bool = False,
) -> torch.Tensor:
    """Apply hand pose gate to reward if enabled.
    
    The gate is computed by hand_pose_following reward and stored on env.
    During grasp and release phases, it ramps from 1→0 as hand deviates from target.
    
    Args:
        env: The environment.
        reward: The reward tensor to gate.
        use_hand_pose_gate: If True, apply the gate. If False, return unchanged.
    
    Returns:
        Gated reward tensor (num_envs,).
    """
    if not use_hand_pose_gate:
        return reward
    
    if hasattr(env, 'hand_pose_gate'):
        return reward * env.hand_pose_gate
    return reward


def _get_hand_pose_gate(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get hand pose gate factor, defaulting to 1.0 if not set.
    
    The gate is computed by hand_pose_following reward and stored on env.
    During grasp and release phases, it ramps from 1→0 as hand deviates from target.
    """
    if hasattr(env, 'hand_pose_gate'):
        return env.hand_pose_gate
    return torch.ones(env.num_envs, device=env.device)


def _get_finger_release_gate(
    env: ManagerBasedRLEnv,
    threshold: float = 0.5,
    ramp: float = 2.0,
    floor: float = 0.2,
) -> torch.Tensor:
    """Get finger release gate factor for gating rewards during release.
    
    Returns a factor in [floor, 1.0] based on finger-object contact force (INVERTED):
    - Strong finger contact: returns floor (e.g., 0.2)
    - No finger contact: returns 1.0
    - Smooth ramp as fingers release
    
    This gates rewards so "holding object at goal" doesn't get full credit -
    must actually release the object.
    
    Uses max(all finger forces) to detect if ANY finger is still touching.
    
    Args:
        env: The environment.
        threshold: Contact force below which gate starts rising (Newtons).
        ramp: Range over which gate ramps from floor to 1.0 (Newtons).
        floor: Minimum gate value when fingers are touching (0-1).
    
    Returns:
        Gate tensor (num_envs,) in range [floor, 1.0].
    """
    
    # Get max finger contact force
    thumb_mag = env.scene.sensors["thumb_link_3_object_s"].data.force_matrix_w.view(env.num_envs, 3).norm(dim=-1)
    index_mag = env.scene.sensors["index_link_3_object_s"].data.force_matrix_w.view(env.num_envs, 3).norm(dim=-1)
    middle_mag = env.scene.sensors["middle_link_3_object_s"].data.force_matrix_w.view(env.num_envs, 3).norm(dim=-1)
    ring_mag = env.scene.sensors["ring_link_3_object_s"].data.force_matrix_w.view(env.num_envs, 3).norm(dim=-1)
    
    max_finger_force = torch.stack([thumb_mag, index_mag, middle_mag, ring_mag], dim=-1).max(dim=-1).values
    
    # INVERTED ramp: high contact = low gate, no contact = high gate
    # When max_finger_force > threshold: raw_factor = 0 (gate = floor)
    # When max_finger_force < threshold - ramp: raw_factor = 1 (gate = 1.0)
    raw_factor = ((threshold - max_finger_force) / ramp).clamp(0.0, 1.0)
    
    # Apply floor: gate ranges from floor to 1.0
    gate = floor + (1.0 - floor) * raw_factor

    return gate


def _get_cached_finger_release_gate(env, threshold=0.5, ramp=2.0, floor=0.2):
    """Return _get_finger_release_gate(), cached per step."""
    step = env.common_step_counter
    cached = getattr(env, '_cached_finger_gate_data', None)
    if cached is not None and cached[0] == step:
        return cached[1]
    gate = _get_finger_release_gate(env, threshold, ramp, floor)
    env._cached_finger_gate_data = (step, gate)
    return gate


def action_rate_l2_clamped(
    env: ManagerBasedRLEnv,
    arm_joint_count: int = 7,
    finger_scale: float = 0.1,
) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel.
    
    Applies different scaling to arm vs hand actions to encourage finger manipulation.
    Hand-related actions (eigen coefficients + residuals in eigen mode, or hand joints
    in full mode) receive reduced penalty.
    
    Args:
        env: The environment.
        arm_joint_count: Number of arm joints (first N actions). Default 7 for Kuka.
        finger_scale: Scale factor for hand action penalties. Default 0.1 = 10x less penalty.
    
    Returns:
        Weighted sum of squared action rate for each environment.
    """
    action_diff = env.action_manager.action - env.action_manager.prev_action
    
    # Split into arm and hand actions
    arm_diff = action_diff[:, :arm_joint_count]
    hand_diff = action_diff[:, arm_joint_count:]
    
    # Apply different weights: full penalty on arm, reduced on hand
    arm_penalty = torch.sum(torch.square(arm_diff), dim=1)
    hand_penalty = torch.sum(torch.square(hand_diff), dim=1) * finger_scale
    
    return (arm_penalty + hand_penalty).clamp(-1000, 1000)


def action_l2_clamped(
    env: ManagerBasedRLEnv,
    arm_joint_count: int = 7,
    finger_scale: float = 0.1,
) -> torch.Tensor:
    """Penalize the actions using L2 squared kernel.
    
    Applies different scaling to arm vs hand actions to encourage finger manipulation.
    Hand-related actions (eigen coefficients + residuals in eigen mode, or hand joints
    in full mode) receive reduced penalty.
    
    Args:
        env: The environment.
        arm_joint_count: Number of arm joints (first N actions). Default 7 for Kuka.
        finger_scale: Scale factor for hand action penalties. Default 0.1 = 10x less penalty.
    
    Returns:
        Weighted sum of squared actions for each environment.
    """
    action = env.action_manager.action
    
    # Split into arm and hand actions
    arm_action = action[:, :arm_joint_count]
    hand_action = action[:, arm_joint_count:]
    
    # Apply different weights: full penalty on arm, reduced on hand
    arm_penalty = torch.sum(torch.square(arm_action), dim=1)
    hand_penalty = torch.sum(torch.square(hand_action), dim=1) * finger_scale
    
    return (arm_penalty + hand_penalty).clamp(-1000, 1000)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    phases: list[str] | None = None,
    use_hand_pose_gate: bool = False,
    error_gate_pos_threshold: float | None = None,
    error_gate_pos_slope: float = 0.02,
    error_gate_rot_threshold: float | None = None,
    error_gate_rot_slope: float = 0.5,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward reaching the object using a tanh-kernel on end-effector distance.

    The reward is close to 1 when the maximum distance between the object and any end-effector body is small.
    
    Args:
        phases: List of phases to be active in, e.g. ["grasp", "manipulation"]. None = all phases.
        use_hand_pose_gate: If True, apply hand pose gate from hand_pose_following.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    asset_pos = asset.data.body_pos_w[:, asset_cfg.body_ids].clone()  # (N, B, 3)

    # Apply tip offsets if configured (bodies 1+ are link_3, offsets are in body-local frame)
    tip_cfg = env.cfg.y2r_cfg.robot.tip_offsets
    if tip_cfg is not None and asset_pos.shape[1] > 1:
        from . import observations as _obs_mod
        from isaaclab.utils.math import quat_apply as _quat_apply
        _obs_mod._ensure_tip_offsets_cache(env.cfg.y2r_cfg, env.device)
        tip_offsets = _obs_mod._TIP_OFFSETS_TENSOR
        if tip_offsets is not None:
            asset_quat = asset.data.body_quat_w[:, asset_cfg.body_ids]  # (N, B, 4)
            for i in range(min(4, asset_pos.shape[1] - 1)):
                # Rotate local offset into world frame and add to position
                offset_w = _quat_apply(
                    asset_quat[:, 1 + i],
                    tip_offsets[i].unsqueeze(0).expand(env.num_envs, 3),
                )
                asset_pos[:, 1 + i] = asset_pos[:, 1 + i] + offset_w

    object_pos = object.data.root_pos_w
    object_ee_distance = torch.norm(asset_pos - object_pos[:, None, :], dim=-1).max(dim=-1).values
    reward = 1 - torch.tanh(object_ee_distance / std)

    # Error-dependent gate: when tracking is clearly bad, reduce this reward so that
    # "freeze + squeeze to keep contact" is not a good local optimum.
    # (Only active when thresholds are provided.)
    if error_gate_pos_threshold is not None:
        cfg = env.cfg.y2r_cfg
        if not cfg.mode.use_point_cloud:
            pos_err = env._cached_aligned_pose_errors["pos"]  # (N,)
            # gate in [0,1], ~1 when pos_err << threshold, ~0 when pos_err >> threshold
            gate = torch.sigmoid((error_gate_pos_threshold - pos_err) / max(error_gate_pos_slope, 1e-6))
            if error_gate_rot_threshold is not None:
                rot_err = env._cached_aligned_pose_errors["rot"]
                gate_rot = torch.sigmoid((error_gate_rot_threshold - rot_err) / max(error_gate_rot_slope, 1e-6))
                gate = gate * gate_rot
        else:
            mean_err = env._cached_aligned_mean_error  # (N,)
            gate = torch.sigmoid((error_gate_pos_threshold - mean_err) / max(error_gate_pos_slope, 1e-6))

        reward = reward * gate
    
    # Apply phase filter and optional hand pose gate
    reward = _apply_phase_filter(env, reward, phases)
    reward = _apply_hand_pose_gate(env, reward, use_hand_pose_gate)
    
    return reward


def _contacts_bool(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    """Check if good finger contact is present (thumb + at least one other finger).
    
    Internal helper that returns boolean tensor.
    """
    thumb_contact_sensor: ContactSensor = env.scene.sensors["thumb_link_3_object_s"]
    index_contact_sensor: ContactSensor = env.scene.sensors["index_link_3_object_s"]
    middle_contact_sensor: ContactSensor = env.scene.sensors["middle_link_3_object_s"]
    ring_contact_sensor: ContactSensor = env.scene.sensors["ring_link_3_object_s"]
    # check if contact force is above threshold
    thumb_contact = thumb_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    index_contact = index_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    middle_contact = middle_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    ring_contact = ring_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)

    thumb_contact_mag = torch.norm(thumb_contact, dim=-1)
    index_contact_mag = torch.norm(index_contact, dim=-1)
    middle_contact_mag = torch.norm(middle_contact, dim=-1)
    ring_contact_mag = torch.norm(ring_contact, dim=-1)
    good_contact_cond1 = (thumb_contact_mag > threshold) & (
        (index_contact_mag > threshold) | (middle_contact_mag > threshold) | (ring_contact_mag > threshold)
    )
    return good_contact_cond1


def contacts(
    env: ManagerBasedRLEnv, 
    threshold: float, 
    phases: list[str] | None = None,
    use_hand_pose_gate: bool = True,
    invert_in_release: bool = False,
) -> torch.Tensor:
    """Reward for good finger contact (thumb + at least one other finger).
    
    Returns a gated float reward (0.0 or 1.0 * gate).
    
    Args:
        env: The environment.
        threshold: Contact force threshold (Newtons).
        phases: List of phases to be active in. None = all phases.
        use_hand_pose_gate: If True, apply hand pose gate from hand_pose_following.
        invert_in_release: If True, reward NO contact during release (encourages letting go).
    """
    good_contact = _contacts_bool(env, threshold)
    
    # During release phase, optionally invert (reward no contact instead)
    if invert_in_release and hasattr(env, 'trajectory_manager') and env.trajectory_manager is not None:
        in_release = env.trajectory_manager.is_in_release_phase()
        # Invert: reward when NOT in contact during release
        contact_value = torch.where(in_release, (~good_contact).float(), good_contact.float())
    else:
        contact_value = good_contact.float()
    
    reward = contact_value
    
    # Apply phase filter and optional hand pose gate
    reward = _apply_phase_filter(env, reward, phases)
    reward = _apply_hand_pose_gate(env, reward, use_hand_pose_gate)
    return reward


def contact_factor(
    env: ManagerBasedRLEnv,
    threshold: float,
    ramp: float = 1.0,
    min_factor: float = 0.05,
) -> torch.Tensor:
    """Continuous [0, 1] contact factor (soft version of `contacts`).

    Uses thumb + max(other finger) contact magnitudes and converts them into a smooth factor:

        strength = min(|F_thumb|, max(|F_other|))
        factor   = clamp((strength - threshold) / ramp, 0, 1)
        factor   = min_factor + (1 - min_factor) * factor

    This prevents the reward from collapsing to exactly 0 when contact is weak, which
    helps recovery behaviors.
    """
    if ramp <= 0.0:
        raise ValueError(f"ramp must be > 0, got {ramp}.")
    min_factor = float(min(max(min_factor, 0.0), 1.0))

    thumb_contact_sensor: ContactSensor = env.scene.sensors["thumb_link_3_object_s"]
    index_contact_sensor: ContactSensor = env.scene.sensors["index_link_3_object_s"]
    middle_contact_sensor: ContactSensor = env.scene.sensors["middle_link_3_object_s"]
    ring_contact_sensor: ContactSensor = env.scene.sensors["ring_link_3_object_s"]

    thumb_mag = torch.norm(thumb_contact_sensor.data.force_matrix_w.view(env.num_envs, 3), dim=-1)
    index_mag = torch.norm(index_contact_sensor.data.force_matrix_w.view(env.num_envs, 3), dim=-1)
    middle_mag = torch.norm(middle_contact_sensor.data.force_matrix_w.view(env.num_envs, 3), dim=-1)
    ring_mag = torch.norm(ring_contact_sensor.data.force_matrix_w.view(env.num_envs, 3), dim=-1)

    other_mag = torch.stack([index_mag, middle_mag, ring_mag], dim=-1).max(dim=-1).values
    strength = torch.minimum(thumb_mag, other_mag)

    factor = ((strength - threshold) / ramp).clamp(0.0, 1.0)
    return min_factor + (1.0 - min_factor) * factor


def joint_pos_limits_margin(
    env: ManagerBasedRLEnv,
    threshold: float = 0.95,
    power: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for operating too close to joint position soft limits.

    We map each joint position into a normalized range [-1, 1] using the soft limits:
        -1 -> lower soft limit, +1 -> upper soft limit.

    Then penalize joints whose |normalized| exceeds `threshold` (e.g. 0.95),
    with a smooth ramp from 0 to 1:

        x = clamp((|q_norm| - threshold) / (1 - threshold), 0, 1)
        penalty = sum(x^power) over joints

    Positions beyond the limit are clamped to the limit (max penalty = 1^power per joint).

    Args:
        env: The environment.
        threshold: Normalized proximity threshold in (0, 1). Larger = only penalize very near limits.
        power: Shape of penalty (>= 1). 2.0 gives quadratic growth near limits.
        asset_cfg: Which articulation / joints to penalize. Defaults to all robot joints.

    Returns:
        Tensor of shape (num_envs,) with per-env penalties (max = num_joints when all at limit).
    """
    if not (0.0 < threshold < 1.0):
        raise ValueError(f"threshold must be in (0, 1), got {threshold}.")
    if power < 1.0:
        raise ValueError(f"power must be >= 1, got {power}.")

    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids if asset_cfg.joint_ids is not None else slice(None)
    limits = asset.data.soft_joint_pos_limits[:, joint_ids]  # (N, J, 2)
    q = asset.data.joint_pos[:, joint_ids]  # (N, J)

    # Normalize to [-1, 1], clamp to handle positions beyond limits
    q_norm = math_utils.scale_transform(q, limits[..., 0], limits[..., 1]).clamp(-1.0, 1.0)
    
    # Ramp from 0 (at threshold) to 1 (at limit)
    x = ((q_norm.abs() - threshold) / (1.0 - threshold)).clamp(min=0.0)
    return torch.sum(x**power, dim=1)


# ==================== Trajectory-based rewards ====================


def lookahead_tracking(
    env: ManagerBasedRLEnv,
    std: float = 0.1,
    decay: float = 0.5,
    phases: list[str] | None = None,
    use_hand_pose_gate: bool = True,
    use_contact_gating: bool = True,
    contact_threshold: float = 0.05,
    contact_ramp: float = 1.0,
    contact_min_factor: float = 0.05,
    rot_std: float = 0.5,
    neg_threshold: float = 0.08,
    neg_std: float = 0.1,
    neg_scale: float = 0.5,
    rot_neg_threshold: float = 0.6,
    rot_neg_std: float = 0.4,
    grasp_scale: float = 1.0,
) -> torch.Tensor:
    """
    Reward tracking trajectory targets with positive and negative zones.

    Supports two tracking strategies based on trajectory config:
    - Waypoint mode (trajectory.path_mode=False): Rewards proximity to discrete waypoints with decay
    - Path mode (trajectory.path_mode=True): Rewards staying on current path segment (target[0]→target[1])

    In path mode, timing is controlled by trajectory.timing_aware:
    - True: time-based interpolation using segment progress
    - False: spatial projection to closest point on segment

    And two error computation modes:
    - Point cloud mode: Uses aligned mean error (point-to-point distance)
    - Pose mode: Uses aligned position + rotation errors

    Two-kernel reward structure:
    - Positive reward: exp(-error/std) for being close to target/path
    - Negative penalty: tanh((error - neg_threshold)/neg_std) when error > neg_threshold
    - Combined: reward = pos_reward - neg_scale * neg_penalty

    Args:
        env: The environment.
        std: Standard deviation for positive position reward exp kernel.
        decay: Exponential decay factor (0-1). Lower = focus on current target. (Ignored in path_mode)
        phases: List of phases to be active in, e.g. ["manipulation"]. None = all phases.
        use_hand_pose_gate: If True, apply hand pose gate from hand_pose_following.
        use_contact_gating: If True, scale reward by finger contact factor.
        contact_threshold: Force threshold for contact detection (N).
        rot_std: Standard deviation for rotation exp kernel (pose mode only).
        neg_threshold: Position error threshold above which negative penalty activates (m).
        neg_std: How fast the position penalty grows beyond threshold.
        neg_scale: Maximum magnitude of negative penalty (0.5 = can go to -0.5).
        rot_neg_threshold: Rotation error threshold above which negative penalty activates (rad).
        rot_neg_std: How fast the rotation penalty grows beyond threshold.

    Returns:
        Reward tensor (num_envs,).
    """
    # Check which mode via env config
    cfg = env.cfg.y2r_cfg
    use_point_cloud = cfg.mode.use_point_cloud
    path_mode = getattr(cfg.trajectory, "path_mode", False)

    # DEBUG: Print all rewards for env 0 every 10 steps (controlled by config visualization.debug_print_rewards)
    step = env.episode_length_buf[0].item()
    if cfg.visualization.debug_print_rewards and step % 10 == 0:
        rm = env.reward_manager
        print(f"\n=== Episode Step {step} | Env 0 Rewards ===")
        # Use get_active_iterable_terms to get per-term values
        for name, values in rm.get_active_iterable_terms(0):
            term_cfg = rm.get_term_cfg(name)
            print(f"  {name:20s}: weighted={values[0]:+.4f} (w={term_cfg.weight:+.2f})")

    if path_mode:
        # ===== PATH MODE: Reward staying on current segment =====
        if not use_point_cloud:
            # Pose mode: use aligned current errors
            pos_error = env._cached_aligned_pose_errors["pos"]  # (N,)
            rot_error = env._cached_aligned_pose_errors["rot"]  # (N,)

            # Positive reward: exp kernel (peaks at 1.0 when error=0)
            pos_pos_reward = torch.exp(-pos_error / std)
            pos_rot_reward = torch.exp(-rot_error / rot_std)
            pos_reward = (pos_pos_reward + pos_rot_reward) / 2.0

            # Soft contact gating (prevents reward collapse when contact is weak)
            if use_contact_gating:
                cfac = contact_factor(env, threshold=contact_threshold, ramp=contact_ramp, min_factor=contact_min_factor)
                pos_reward = pos_reward * cfac

            # Negative penalty: separate for position and rotation
            pos_excess_error = torch.clamp(pos_error - neg_threshold, min=0.0)
            pos_neg_penalty = torch.tanh(pos_excess_error / neg_std)

            rot_excess_error = torch.clamp(rot_error - rot_neg_threshold, min=0.0)
            rot_neg_penalty = torch.tanh(rot_excess_error / rot_neg_std)

            combined_neg_penalty = (pos_neg_penalty + rot_neg_penalty) / 2.0

            reward = pos_reward - neg_scale * combined_neg_penalty
        else:
            # Point cloud mode: use aligned current errors
            mean_error = env._cached_aligned_mean_error  # (N,)

            # Positive reward: exp kernel
            pos_reward = torch.exp(-mean_error / std)

            # Soft contact gating
            if use_contact_gating:
                cfac = contact_factor(env, threshold=contact_threshold, ramp=contact_ramp, min_factor=contact_min_factor)
                pos_reward = pos_reward * cfac

            # Negative penalty
            excess_error = torch.clamp(mean_error - neg_threshold, min=0.0)
            neg_penalty = torch.tanh(excess_error / neg_std)

            reward = pos_reward - neg_scale * neg_penalty

    else:
        # ===== WAYPOINT MODE: Reward proximity to discrete waypoints with decay =====
        if not use_point_cloud:
            # Pose mode: combine position and rotation rewards
            pos_errors = env._cached_pose_errors['pos']  # (N, W)
            rot_errors = env._cached_pose_errors['rot']  # (N, W)
            W = pos_errors.shape[1]

            # Exponential decay weights for lookahead
            weights = decay ** torch.arange(W, device=env.device, dtype=torch.float32)
            weights = weights / weights.sum()

            # Weighted average errors
            weighted_pos_error = (pos_errors * weights.unsqueeze(0)).sum(dim=-1)  # (N,)
            weighted_rot_error = (rot_errors * weights.unsqueeze(0)).sum(dim=-1)  # (N,)

            # Positive reward: exp kernel (peaks at 1.0 when error=0)
            pos_pos_reward = torch.exp(-weighted_pos_error / std)
            pos_rot_reward = torch.exp(-weighted_rot_error / rot_std)
            pos_reward = (pos_pos_reward + pos_rot_reward) / 2.0

            # Soft contact gating (prevents reward collapse when contact is weak)
            if use_contact_gating:
                cfac = contact_factor(env, threshold=contact_threshold, ramp=contact_ramp, min_factor=contact_min_factor)
                pos_reward = pos_reward * cfac

            # Negative penalty: separate for position and rotation
            # Position penalty
            pos_excess_error = torch.clamp(weighted_pos_error - neg_threshold, min=0.0)
            pos_neg_penalty = torch.tanh(pos_excess_error / neg_std)

            # Rotation penalty
            rot_excess_error = torch.clamp(weighted_rot_error - rot_neg_threshold, min=0.0)
            rot_neg_penalty = torch.tanh(rot_excess_error / rot_neg_std)

            # Combined: position penalty weighted same as position reward
            combined_neg_penalty = (pos_neg_penalty + rot_neg_penalty) / 2.0

            reward = pos_reward - neg_scale * combined_neg_penalty
        else:
            # Point cloud mode: use mean errors
            mean_errors = env._cached_mean_errors  # (N, W)
            W = mean_errors.shape[1]

            # Exponential decay weights for lookahead
            weights = decay ** torch.arange(W, device=env.device, dtype=torch.float32)
            weights = weights / weights.sum()

            # Weighted average error
            weighted_error = (mean_errors * weights.unsqueeze(0)).sum(dim=-1)  # (N,)

            # Positive reward: exp kernel
            pos_reward = torch.exp(-weighted_error / std)

            # Soft contact gating (prevents reward collapse when contact is weak)
            if use_contact_gating:
                cfac = contact_factor(env, threshold=contact_threshold, ramp=contact_ramp, min_factor=contact_min_factor)
                pos_reward = pos_reward * cfac

            # Negative penalty: activates when error > neg_threshold
            excess_error = torch.clamp(weighted_error - neg_threshold, min=0.0)
            neg_penalty = torch.tanh(excess_error / neg_std)

            reward = pos_reward - neg_scale * neg_penalty

    # Apply per-phase scaling with grasp_scale
    if phases is not None and len(phases) > 0 and hasattr(env, 'trajectory_manager') and env.trajectory_manager is not None:
        phase = _get_cached_phase(env)  # (N,) - 0=grasp, 1=manip, 2=release
        scale = torch.zeros(env.num_envs, dtype=reward.dtype, device=env.device)
        for p in phases:
            if p in PHASE_MAP:
                mask = phase == PHASE_MAP[p]
                if p == "grasp":
                    scale = torch.where(mask, torch.full_like(scale, grasp_scale), scale)
                else:
                    scale = torch.where(mask, torch.ones_like(scale), scale)
        reward = reward * scale
    elif phases is not None and len(phases) > 0:
        # No trajectory manager — fall through without filtering
        pass

    reward = _apply_hand_pose_gate(env, reward, use_hand_pose_gate)

    return reward


def tracking_progress(
    env: ManagerBasedRLEnv,
    pos_weight: float = 1.0,
    rot_weight: float = 0.5,
    positive_only: bool = False,
    clip: float = 1.0,
    phases: list[str] | None = None,
    use_hand_pose_gate: bool = True,
) -> torch.Tensor:
    """Reward improvement in tracking error over ~1/target_hz seconds (relative).

    Uses cached aligned errors produced by the target observation term:
    - Pose mode: combines aligned position + rotation error.
    - Point cloud mode: uses aligned mean point error.

    This is meant to be **off by default** (weight=0) and turned on only if needed.
    
    Args:
        phases: List of phases to be active in. None = all phases.
        use_hand_pose_gate: If True, apply hand pose gate from hand_pose_following.
    """
    cfg = env.cfg.y2r_cfg
    if not cfg.mode.use_point_cloud:
        pos_err = env._cached_aligned_pose_errors["pos"]
        rot_err = env._cached_aligned_pose_errors["rot"]
        cur_err = pos_weight * pos_err + rot_weight * rot_err
    else:
        cur_err = env._cached_aligned_mean_error

    # Compare against the error from approximately 1/target_hz seconds ago
    # (same cadence as the trajectory targets / finger_manipulation).
    target_interval = 1.0 / float(cfg.trajectory.target_hz)
    interval_steps = max(1, int(target_interval / float(env.step_dt)))

    # Maintain a small per-env circular buffer so we can look back interval_steps.
    # History length = interval_steps + 1 to index the value from exactly that many steps ago.
    hist_len = interval_steps + 1
    hist = getattr(env, "_tracking_error_hist", None)
    idx = getattr(env, "_tracking_error_hist_idx", None)
    if hist is None or hist.shape != (env.num_envs, hist_len) or hist.device != cur_err.device:
        hist = cur_err.detach().unsqueeze(1).repeat(1, hist_len).clone()
        idx = 0
        env._tracking_error_hist = hist
        env._tracking_error_hist_idx = idx

    # Reset histories for envs that just reset (episode_length == 0).
    if hasattr(env, "episode_length_buf"):
        reset_mask = env.episode_length_buf == 0
        if torch.any(reset_mask):
            hist[reset_mask] = cur_err.detach()[reset_mask].unsqueeze(1).repeat(1, hist_len)

    # Write current error at idx, read error from interval_steps ago (the next slot).
    idx = int(env._tracking_error_hist_idx)
    hist[:, idx] = cur_err.detach()
    old_idx = (idx + 1) % hist_len
    prev_err = hist[:, old_idx]
    env._tracking_error_hist_idx = (idx + 1) % hist_len

    # improvement: positive when error decreases
    progress = prev_err - cur_err

    # Fractional improvement: (prev - cur) / max(prev, eps)
    progress = progress / torch.clamp(prev_err, min=1e-6)

    # Optionally ignore regressions (prevents this term from acting like a penalty).
    if positive_only:
        progress = progress.clamp(min=0.0)

    if clip is not None:
        progress = progress.clamp(min=-float(clip), max=float(clip))
    
    # Apply phase filter and optional hand pose gate
    progress = _apply_phase_filter(env, progress, phases)
    progress = _apply_hand_pose_gate(env, progress, use_hand_pose_gate)
    
    return progress


def trajectory_success(
    env: ManagerBasedRLEnv,
    pos_threshold: float = 0.05,
    rot_threshold: float = 0.3,
    pos_std: float = 0.1,
    rot_std: float = 0.5,
    sparse_weight: float = 0.7,
    phases: list[str] | None = None,
    use_hand_pose_gate: bool = True,
    use_finger_release_gate: bool = True,
    contact_gate_threshold: float = 0.5,
    contact_gate_ramp: float = 2.0,
    contact_gate_floor: float = 0.2,
) -> torch.Tensor:
    """
    Dense + Sparse success reward: combines shaping gradient with binary bonus.
    
    Dense component: Always provides gradient toward goal (exp kernel).
    Sparse component: Binary bonus when BOTH pos AND rot are within thresholds,
                     optionally GATED by finger release (must actually release, not just hold).
    
    Combined: reward = (1 - sparse_weight) * dense + sparse_weight * sparse * contact_gate
    
    This gives:
    - Smooth gradient to guide toward goal (dense shaping, always active)
    - Big reward only when actually placed (sparse bonus optionally gated by finger release)
    - Prevents "hold object at goal" exploit - must release onto table
    
    Supports two modes based on cached aligned errors:
    - Point cloud mode: Uses aligned mean error (position only)
    - Pose mode: Uses aligned position + rotation errors
    
    Args:
        env: The environment.
        pos_threshold: Position threshold for sparse bonus (meters).
        rot_threshold: Rotation threshold for sparse bonus (radians).
        pos_std: Position std for dense shaping exp kernel (meters).
        rot_std: Rotation std for dense shaping exp kernel (radians).
        sparse_weight: Weight of sparse bonus (0-1). Dense weight = 1 - sparse_weight.
        phases: List of phases to be active in. Default ["release"] for success reward.
        use_hand_pose_gate: If True, apply hand pose gate from hand_pose_following.
        use_finger_release_gate: If True, gate sparse bonus by finger release.
        contact_gate_threshold: Contact force threshold to start gating (Newtons).
        contact_gate_ramp: Range over which contact gate ramps from floor to 1.0 (Newtons).
        contact_gate_floor: Minimum gate value when no table contact (0-1).
    
    Returns:
        Reward tensor (num_envs,) in range [0, 1].
    """
    # Default to release phase if not specified
    if phases is None:
        phases = ["release"]
    
    cfg = env.cfg.y2r_cfg
    use_point_cloud = cfg.mode.use_point_cloud
    
    if not use_point_cloud:
        # Pose mode: position + rotation
        pos_error = env._cached_aligned_pose_errors['pos']  # (N,)
        rot_error = env._cached_aligned_pose_errors['rot']  # (N,)
        
        # Dense component: exp kernel shaping (always provides gradient)
        pos_dense = torch.exp(-pos_error / pos_std)
        rot_dense = torch.exp(-rot_error / rot_std)
        dense = (pos_dense + rot_dense) / 2.0
        
        # Sparse component: binary bonus when in bounds
        pos_ok = pos_error < pos_threshold
        rot_ok = rot_error < rot_threshold
        sparse = (pos_ok & rot_ok).float()
    else:
        # Point cloud mode: position only
        current_error = env._cached_aligned_mean_error  # (N,)
        
        # Dense component
        dense = torch.exp(-current_error / pos_std)
        
        # Sparse component
        sparse = (current_error < pos_threshold).float()
    
    # Optionally gate sparse component by finger release (must actually release, not hold)
    if use_finger_release_gate:
        contact_gate = _get_cached_finger_release_gate(
            env,
            threshold=contact_gate_threshold,
            ramp=contact_gate_ramp,
            floor=contact_gate_floor
        )
        sparse_gated = sparse * contact_gate
    else:
        sparse_gated = sparse
    
    # Combine: dense shaping (ungated) + sparse bonus (optionally contact-gated)
    reward = (1.0 - sparse_weight) * dense + sparse_weight * sparse_gated
    
    # Apply phase filter and optional hand pose gate
    reward = _apply_phase_filter(env, reward, phases)
    reward = _apply_hand_pose_gate(env, reward, use_hand_pose_gate)
    
    return reward


def arm_table_binary_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    table_z: float = 0.255,
    threshold_mid: float = 0.08,
    threshold_distal: float = 0.04,
) -> torch.Tensor:
    """
    Binary penalty for arm links below safe height thresholds.
    
    Checks body_ids from asset_cfg (expected to be links 3-7 + palm_link).
    Links are split into two groups by their index in body_ids:
    - First 3 bodies (links 3,4,5): threshold = table_z + threshold_mid
    - Remaining bodies (links 6,7,palm): threshold = table_z + threshold_distal
    
    Returns fraction of links below their threshold (0.0 to 1.0).
    
    Args:
        env: The environment.
        asset_cfg: SceneEntityCfg with body_ids for links to check.
        table_z: Base table height (without env origin offset).
        threshold_mid: Safe height above table for mid-arm links (m).
        threshold_distal: Safe height above table for distal links + palm (m).
    
    Returns:
        Penalty tensor (num_envs,) - fraction of links below threshold.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    
    # Get Z positions for selected bodies: (N, L)
    link_z = robot.data.body_pos_w[:, asset_cfg.body_ids, 2]
    
    # Account for env origins
    env_table_z = table_z + env.scene.env_origins[:, 2]  # (N,)
    
    num_bodies = len(asset_cfg.body_ids)
    
    # Split into mid (first 3) and distal (remaining)
    # Assumes body_ids are ordered: [link3, link4, link5, link6, link7, palm]
    num_mid = 3
    
    # Mid links (3,4,5): table_z + threshold_mid
    mid_threshold = env_table_z.unsqueeze(1) + threshold_mid  # (N, 1)
    mid_violations = (link_z[:, :num_mid] < mid_threshold).float()  # (N, 3)
    
    # Distal links (6,7,palm): table_z + threshold_distal
    distal_threshold = env_table_z.unsqueeze(1) + threshold_distal  # (N, 1)
    distal_violations = (link_z[:, num_mid:] < distal_threshold).float()  # (N, remaining)
    
    # Concatenate and compute fraction
    all_violations = torch.cat([mid_violations, distal_violations], dim=1)  # (N, L)
    fraction_below = all_violations.mean(dim=1)  # (N,)
    
    return fraction_below


def finger_manipulation(
    env: ManagerBasedRLEnv,
    pos_std: float = 0.01,
    rot_std: float = 0.1,
    phases: list[str] | None = None,
    use_hand_pose_gate: bool = True,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """
    Reward finger manipulation - object movement relative to palm.
    
    Distinguishes finger-driven manipulation from arm movement by tracking
    object pose in palm frame. When the object moves relative to the palm,
    that indicates finger manipulation (in-hand adjustment).
    
    GATED BY ERROR IMPROVEMENT: Only gives reward when tracking error is
    decreasing, ensuring palm-relative movement is actually helpful.
    
    Compares current pose/error to values from 1/target_hz seconds ago (same
    frequency as trajectory targets), capturing meaningful changes over time
    rather than per-step jitter.
    
    Uses exponential kernel: reward = 1 - exp(-delta / std) to reward
    significant palm-relative movement while ignoring tiny changes.
    
    Args:
        env: The environment.
        pos_std: Position change threshold in meters (default 0.01 = 1cm).
        rot_std: Rotation change threshold in radians (default 0.1 ≈ 6°).
        phases: List of phases to be active in. None = all phases.
        use_hand_pose_gate: If True, apply hand pose gate from hand_pose_following.
        robot_cfg: Scene entity config for robot (must have palm_link body).
        object_cfg: Scene entity config for the manipulated object.
    
    Returns:
        Reward tensor (num_envs,) in range [0, 1].
    """
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    cfg = env.cfg.y2r_cfg
    
    # Get palm frame pose in world frame (with offset if configured)
    palm_pos_w, palm_quat_w, _ = get_palm_frame_pose_w(robot, env.cfg.y2r_cfg)

    # Get object pose in world frame
    object_pos_w = obj.data.root_pos_w  # (N, 3)
    object_quat_w = obj.data.root_quat_w  # (N, 4)
    
    # Transform object pose to palm frame
    # Position: rotate (object_pos - palm_pos) by inverse palm rotation
    object_pos_palm = quat_apply_inverse(palm_quat_w, object_pos_w - palm_pos_w)  # (N, 3)
    
    # Orientation: object_quat in palm frame = inv(palm_quat) * object_quat
    object_quat_palm = quat_mul(quat_inv(palm_quat_w), object_quat_w)  # (N, 4)
    
    # Get current tracking error (use cached aligned errors from observation computation)
    use_point_cloud = cfg.mode.use_point_cloud
    if use_point_cloud:
        # Point cloud mode: use mean error of current target
        current_error = env._cached_aligned_mean_error  # (N,)
    else:
        # Pose mode: combine position and rotation errors
        pos_error = env._cached_aligned_pose_errors['pos']  # (N,)
        rot_error = env._cached_aligned_pose_errors['rot']  # (N,)
        current_error = pos_error + rot_error * 0.1  # Weighted combination
    
    # Initialize on first call: compute update interval and cache
    if not hasattr(env, '_finger_manip_update_interval'):
        control_dt = cfg.simulation.physics_dt * cfg.simulation.decimation
        target_interval = 1.0 / cfg.trajectory.target_hz
        env._finger_manip_update_interval = max(1, int(target_interval / control_dt))
        env._finger_manip_step_counter = 0
        env._finger_manip_old_pos = object_pos_palm.clone()
        env._finger_manip_old_quat = object_quat_palm.clone()
        env._finger_manip_old_error = current_error.clone()
        return torch.zeros(env.num_envs, device=env.device)
    
    # Compute change in palm-relative pose (compared to 1/target_hz seconds ago)
    pos_delta = (object_pos_palm - env._finger_manip_old_pos).norm(dim=-1)  # (N,)
    rot_delta = quat_error_magnitude(object_quat_palm, env._finger_manip_old_quat)  # (N,)
    
    # Check if error improved (decreased) since last update
    error_improved = env._finger_manip_old_error > current_error  # (N,) bool
    
    # Detect resets: if delta is huge (>0.5m or >π rad), it's likely a reset
    reset_mask = (pos_delta > 0.5) | (rot_delta > 3.14)
    
    # Update cached values every update_interval steps
    env._finger_manip_step_counter += 1
    if env._finger_manip_step_counter >= env._finger_manip_update_interval:
        env._finger_manip_step_counter = 0
        env._finger_manip_old_pos = object_pos_palm.clone()
        env._finger_manip_old_quat = object_quat_palm.clone()
        env._finger_manip_old_error = current_error.clone()
    
    # On reset, immediately update the cache to avoid spurious rewards
    if reset_mask.any():
        env._finger_manip_old_pos[reset_mask] = object_pos_palm[reset_mask]
        env._finger_manip_old_quat[reset_mask] = object_quat_palm[reset_mask]
        env._finger_manip_old_error[reset_mask] = current_error[reset_mask]
    
    # Compute reward using exponential kernel
    # Reward is ~0 for no movement, approaches 1 for significant movement
    pos_reward = 1.0 - torch.exp(-pos_delta / pos_std)
    rot_reward = 1.0 - torch.exp(-rot_delta / rot_std)
    
    # Average position and rotation rewards
    reward = (pos_reward + rot_reward) / 2.0
    
    # Gate by error improvement: only reward when tracking is getting better
    reward = torch.where(error_improved, reward, torch.zeros_like(reward))
    reward = torch.where(_contacts_bool(env, 1.0), reward, torch.zeros_like(reward))
    
    # Zero out reward for reset frames
    reward = torch.where(reset_mask, torch.zeros_like(reward), reward)
    
    # Apply phase filter and optional hand pose gate
    reward = _apply_phase_filter(env, reward, phases)
    reward = _apply_hand_pose_gate(env, reward, use_hand_pose_gate)
    
    return reward


def palm_velocity_penalty(
    env: ManagerBasedRLEnv,
    angular_std: float = 0.5,
    linear_std: float = 0.3,
    linear_scale: float = 0.2,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Penalize palm velocity during manipulation phase.
    
    Encourages using fingers for fine adjustments rather than moving the whole arm.
    Only active during manipulation phase (not pickup or release).
    Angular velocity is penalized more heavily than linear velocity.
    
    Args:
        env: The environment.
        angular_std: Std for angular velocity penalty (rad/s). Lower = stricter.
        linear_std: Std for linear velocity penalty (m/s). Lower = stricter.
        linear_scale: Scale factor for linear penalty (default 0.2 = 5x more lenient than angular).
        robot_cfg: Scene entity config for robot (must have palm_link body).
    
    Returns:
        Penalty tensor (num_envs,) in range [0, 1]. Higher = more movement = more penalty.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    
    # Get palm body index (velocities use palm_link directly, offset is rigid)
    _, _, palm_idx = get_palm_frame_pose_w(robot, env.cfg.y2r_cfg)

    # Get palm velocities in world frame
    palm_lin_vel_w = robot.data.body_lin_vel_w[:, palm_idx]  # (N, 3)
    palm_ang_vel_w = robot.data.body_ang_vel_w[:, palm_idx]  # (N, 3)
    
    # Compute velocity magnitudes
    lin_vel_mag = palm_lin_vel_w.norm(dim=-1)  # (N,)
    ang_vel_mag = palm_ang_vel_w.norm(dim=-1)  # (N,)
    
    # Compute penalties using tanh kernel
    # Higher velocity = higher penalty (closer to 1)
    angular_penalty = torch.tanh(ang_vel_mag / angular_std)
    linear_penalty = torch.tanh(lin_vel_mag / linear_std) * linear_scale
    
    # Combined penalty (angular is dominant due to linear_scale < 1)
    penalty = angular_penalty + linear_penalty
    
    # Gate by manipulation phase (phase == 1)
    in_manipulation = _get_cached_phase(env) == 1  # (N,) bool
    
    # Only penalize during manipulation phase
    penalty = torch.where(in_manipulation, penalty, torch.zeros_like(penalty))
    
    return penalty


def palm_orientation_penalty(
    env: ManagerBasedRLEnv,
    target_euler: tuple[float, float, float] = (0.0, -3.14159, 0.0),
    std: float = 0.5,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Penalize palm orientation deviation from preferred orientation.
    
    Encourages the palm to stay near a specific orientation (default: 0°, -180°, 0°)
    to avoid awkward wrist configurations. Weak penalty to allow flexibility.
    
    Args:
        env: The environment.
        target_euler: Target orientation as (roll, pitch, yaw) in radians.
                     Default (0, -π, 0) = upright but flipped.
        std: Angular error std for tanh kernel (rad). Larger = more lenient.
        robot_cfg: Scene entity config for robot (must have palm_link body).
    
    Returns:
        Penalty tensor (num_envs,) in range [0, 1].
    """
    robot: Articulation = env.scene[robot_cfg.name]
    
    # Get palm frame pose (with offset if configured)
    _, palm_quat_w, _ = get_palm_frame_pose_w(robot, env.cfg.y2r_cfg)
    
    # Convert target euler to quaternion (cached on first call)
    if not hasattr(env, '_target_palm_quat'):
        from isaaclab.utils.math import quat_from_euler_xyz
        roll, pitch, yaw = target_euler
        target_quat = quat_from_euler_xyz(
            torch.tensor([roll], device=env.device),
            torch.tensor([pitch], device=env.device),
            torch.tensor([yaw], device=env.device),
        )  # (1, 4)
        env._target_palm_quat = target_quat.squeeze(0)  # (4,)
    
    target_quat = env._target_palm_quat  # (4,)
    
    # Compute angular error between current and target orientation
    angular_error = quat_error_magnitude(palm_quat_w, target_quat.unsqueeze(0).expand(env.num_envs, 4))  # (N,)
    
    # Penalize using tanh kernel
    penalty = torch.tanh(angular_error / std)
    
    return penalty


def distal_joint3_penalty(
    env: ManagerBasedRLEnv,
    std: float = 1.0,
    joint_name_regex: str = ".*_joint_3",
    only_when_contact: bool = True,
    contact_threshold: float = 1.0,
    phases: list[str] | None = None,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Penalize magnitude of distal Allegro joints (joint_3) to discourage over-curling.

    This targets the posture that tends to produce "nail-side" / folded-in grasps.
    The penalty is bounded using a tanh kernel and can be gated:
    - by contact (thumb + at least one other finger) and/or
    - by phase.

    Args:
        env: The environment.
        std: Scale (rad) for tanh kernel. Larger = more lenient.
        joint_name_regex: Regex to select distal joints. Defaults to ".*_joint_3".
        only_when_contact: If True, only apply penalty when grasp contact is present.
        contact_threshold: Threshold (N) for contact detection used by `contacts()`.
        phases: List of phases to be active in. None = all phases.
        robot_cfg: Scene entity config for robot.

    Returns:
        Penalty tensor (num_envs,) in range [0, 1]. Higher = more distal curl.
    """
    robot: Articulation = env.scene[robot_cfg.name]

    # Resolve and cache joint_3 ids once (regex over joint names)
    cache_key = "_distal_joint3_ids"
    if not hasattr(env, cache_key):
        joint_ids, joint_names = robot.find_joints([joint_name_regex], preserve_order=False)
        if len(joint_ids) == 0:
            setattr(env, cache_key, None)
        else:
            setattr(env, cache_key, torch.tensor(joint_ids, device=env.device, dtype=torch.long))
            # Optional: keep names for debugging if needed later
            setattr(env, "_distal_joint3_names", joint_names)

    joint3_ids = getattr(env, cache_key)
    if joint3_ids is None:
        return torch.zeros(env.num_envs, device=env.device)

    # Joint positions: (N, K)
    q = robot.data.joint_pos[:, joint3_ids]

    # Bounded penalty per joint, then mean across joints -> (N,)
    per_joint = torch.tanh(torch.abs(q) / std)
    penalty = per_joint.mean(dim=-1)

    # Gate by contact if requested
    if only_when_contact:
        has_contact = _contacts_bool(env, contact_threshold)
        penalty = torch.where(has_contact, penalty, torch.zeros_like(penalty))

    # Apply phase filter
    penalty = _apply_phase_filter(env, penalty, phases)

    return penalty


# ==================== Hand Pose Following ====================


def hand_pose_following(
    env: ManagerBasedRLEnv,
    grasp_pos_tol: float = 0.06,
    grasp_rot_tol: float = 0.4,
    manipulation_pos_tol: float = 0.10,
    manipulation_rot_tol: float = 0.5,
    release_pos_tol: float = 0.06,
    release_rot_tol: float = 0.4,
    gate_in_grasp: bool = True,
    gate_in_manipulation: bool = False,
    gate_in_release: bool = True,
    gate_pos_threshold: tuple[float, float] = (0.03, 0.06),
    gate_rot_threshold: tuple[float, float] = (0.6, 0.9),
    manipulation_gate_pos_threshold: tuple[float, float] | None = None,
    manipulation_gate_rot_threshold: tuple[float, float] | None = None,
    gate_floor: float = 0.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Reward for following hand (palm) pose trajectory.
    
    Uses exp(-error/tol) kernel with phase-varying tolerances:
    - Grasp phase: tight tolerance (learn structured approach)
    - Manipulation phase: loose tolerance (allow in-hand adjustment)
    - Release phase: tight tolerance (ensure hand leaves properly)
    
    Also computes a gate factor stored on env.hand_pose_gate for gating other rewards.
    The gate can be enabled per-phase and smoothly ramps from 1→gate_floor
    as position OR rotation error increases beyond thresholds.
    
    Args:
        env: The environment.
        grasp_pos_tol: Position tolerance during grasp phase (meters).
        grasp_rot_tol: Rotation tolerance during grasp phase (radians).
        manipulation_pos_tol: Position tolerance during manipulation phase (meters).
        manipulation_rot_tol: Rotation tolerance during manipulation phase (radians).
        release_pos_tol: Position tolerance during release phase (meters).
        release_rot_tol: Rotation tolerance during release phase (radians).
        gate_in_grasp: If True, apply gate during grasp phase.
        gate_in_manipulation: If True, apply gate during manipulation phase.
        gate_in_release: If True, apply gate during release phase.
        gate_pos_threshold: Position gate thresholds [start_dropping, reach_minimum] (meters).
            Used for grasp and release phases.
        gate_rot_threshold: Rotation gate thresholds [start_dropping, reach_minimum] (radians).
            Used for grasp and release phases.
        manipulation_gate_pos_threshold: Position gate thresholds for manipulation phase (meters).
            If None, uses gate_pos_threshold.
        manipulation_gate_rot_threshold: Rotation gate thresholds for manipulation phase (radians).
            If None, uses gate_rot_threshold.
        gate_floor: Minimum gate value (0.0-1.0). Prevents complete reward collapse.
        robot_cfg: Scene entity config for robot.
    
    Returns:
        Reward tensor (num_envs,) in range [0, 1].
    """
    cfg = env.cfg.y2r_cfg
    N = env.num_envs
    
    # If hand trajectory is disabled, return zeros and set gate to 1
    if not cfg.hand_trajectory.enabled:
        env.hand_pose_gate = torch.ones(N, device=env.device)
        return torch.zeros(N, device=env.device)
    
    robot: Articulation = env.scene[robot_cfg.name]
    trajectory_manager = env.trajectory_manager
    
    # Get palm frame pose in world frame (with offset if configured)
    palm_pos_w, palm_quat_w, _ = get_palm_frame_pose_w(robot, env.cfg.y2r_cfg)
    
    # Get aligned hand target pose (cached during observations)
    if not hasattr(env, "_cached_aligned_hand_target"):
        raise RuntimeError("Aligned hand target cache missing; ensure observations run before rewards.")
    hand_target = env._cached_aligned_hand_target  # (N, 7)
    target_pos_w = hand_target[:, :3]  # (N, 3)
    target_quat_w = hand_target[:, 3:7]  # (N, 4)
    
    # Compute errors
    pos_error = (palm_pos_w - target_pos_w).norm(dim=-1)  # (N,)
    rot_error = quat_error_magnitude(palm_quat_w, target_quat_w)  # (N,)
    
    # Get current phase (0=grasp, 1=manipulation, 2=release)
    phase = _get_cached_phase(env)  # (N,)

    # Phase-varying tolerances
    pos_tol = torch.where(
        phase == 0, 
        torch.full((N,), grasp_pos_tol, device=env.device),
        torch.where(
            phase == 1,
            torch.full((N,), manipulation_pos_tol, device=env.device),
            torch.full((N,), release_pos_tol, device=env.device)
        )
    )
    rot_tol = torch.where(
        phase == 0,
        torch.full((N,), grasp_rot_tol, device=env.device),
        torch.where(
            phase == 1,
            torch.full((N,), manipulation_rot_tol, device=env.device),
            torch.full((N,), release_rot_tol, device=env.device)
        )
    )
    
    # Compute reward: exp(-error/tol) for position and rotation
    pos_reward = torch.exp(-pos_error / pos_tol)
    rot_reward = torch.exp(-rot_error / rot_tol)
    reward = (pos_reward + rot_reward) / 2.0
    
    # Compute gate factor for enabled phases
    any_gating = gate_in_grasp or gate_in_manipulation or gate_in_release
    if any_gating:
        # Determine which phases have gating enabled
        in_grasp = (phase == 0)
        in_manipulation = (phase == 1)
        in_release = (phase == 2)
        
        # Build mask of which envs are in a gated phase
        in_gated_phase = torch.zeros(N, dtype=torch.bool, device=env.device)
        if gate_in_grasp:
            in_gated_phase = in_gated_phase | in_grasp
        if gate_in_manipulation:
            in_gated_phase = in_gated_phase | in_manipulation
        if gate_in_release:
            in_gated_phase = in_gated_phase | in_release
        
        # Unpack threshold tuples: [start_dropping, reach_minimum]
        # Default thresholds (used for grasp and release phases)
        default_pos_start, default_pos_end = gate_pos_threshold
        default_rot_start, default_rot_end = gate_rot_threshold

        # Manipulation-specific thresholds (fall back to default if not provided)
        if manipulation_gate_pos_threshold is not None:
            manip_pos_start, manip_pos_end = manipulation_gate_pos_threshold
        else:
            manip_pos_start, manip_pos_end = default_pos_start, default_pos_end

        if manipulation_gate_rot_threshold is not None:
            manip_rot_start, manip_rot_end = manipulation_gate_rot_threshold
        else:
            manip_rot_start, manip_rot_end = default_rot_start, default_rot_end

        # Select thresholds per-env based on phase (manipulation vs grasp/release)
        pos_start = torch.where(in_manipulation, manip_pos_start, default_pos_start)
        pos_end = torch.where(in_manipulation, manip_pos_end, default_pos_end)
        rot_start = torch.where(in_manipulation, manip_rot_start, default_rot_start)
        rot_end = torch.where(in_manipulation, manip_rot_end, default_rot_end)

        # Position gate: linear ramp 1.0 → 0.0 as error exceeds threshold
        pos_gate_range = pos_end - pos_start
        pos_gate = 1.0 - (pos_error - pos_start) / pos_gate_range
        pos_gate = pos_gate.clamp(0.0, 1.0)

        # Rotation gate: linear ramp 1.0 → 0.0 as error exceeds threshold
        rot_gate_range = rot_end - rot_start
        rot_gate = 1.0 - (rot_error - rot_start) / rot_gate_range
        rot_gate = rot_gate.clamp(0.0, 1.0)
        
        # Combined gate: minimum of position and rotation gates
        # If either error is too large, the gate drops
        gate = torch.min(pos_gate, rot_gate)
        
        # Apply floor so gate never goes below gate_floor
        gate = gate * (1.0 - gate_floor) + gate_floor
        
        # Only apply gate during enabled phases
        gate = torch.where(in_gated_phase, gate, torch.ones_like(gate))
        
        env.hand_pose_gate = gate
    else:
        env.hand_pose_gate = torch.ones(N, device=env.device)
    
    return reward


# ==================== Finger Release ====================


from .actions import ALLEGRO_PCA_MATRIX, ALLEGRO_HAND_JOINT_NAMES, get_allegro_hand_joint_ids


def finger_release(
    env: ManagerBasedRLEnv,
    scale: float = 1.0,
    phases: list[str] | None = None,
    arm_joint_count: int = 7,
    hand_joint_count: int = 16,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Reward for opening the hand during release phase.
    
    Uses the first eigengrasp component (primary open/close synergy) to measure
    hand openness. The eigengrasp coefficient indicates:
    - Positive = fingers curled more than default (closing)
    - Zero = at default position  
    - Negative = fingers extended more than default (opening)
    
    Reward uses sigmoid(-coeff * scale) so:
    - Default pose (coeff=0): reward ≈ 0.5
    - Hand opening (coeff negative): reward → 1.0
    - Hand closing (coeff positive): reward → 0.0
    
    Args:
        env: The environment.
        scale: Controls gradient steepness. Higher = stronger push toward open.
        phases: List of phases to be active in. Default ["release"].
        arm_joint_count: Number of arm joints (default 7 for Kuka).
        hand_joint_count: Number of hand joints (default 16 for Allegro).
        robot_cfg: Scene entity config for robot.
    
    Returns:
        Reward tensor (num_envs,) in range (0, 1).
    """
    # Default to release phase if not specified
    cfg = env.cfg.y2r_cfg
    N = env.num_envs
    device = env.device
    
    # If hand trajectory is disabled, return zeros
    if not cfg.hand_trajectory.enabled:
        return torch.zeros(N, device=device)
    
    robot: Articulation = env.scene[robot_cfg.name]
    
    # Use explicit joint IDs to ensure correct order (not PhysX native order)
    joint_ids = get_allegro_hand_joint_ids(env, robot)
    hand_pos = robot.data.joint_pos[:, joint_ids]  # (N, 16) in canonical order
    default_pos = robot.data.default_joint_pos[:, joint_ids]  # (N, 16)
    hand_delta = hand_pos - default_pos  # (N, 16)
    
    # Project onto first eigengrasp basis (open/close synergy)
    # Negative coefficient = more open than default
    # Positive coefficient = more closed than default
    eigen_basis = ALLEGRO_PCA_MATRIX[0].to(device)  # (16,)
    eigen_coeff = torch.matmul(hand_delta, eigen_basis)  # (N,)
    
    # Sigmoid reward: more negative coeff (more open) = higher reward
    # sigmoid(-coeff * scale) gives smooth gradient in both directions
    reward = torch.sigmoid(-eigen_coeff * scale)
    
    # Apply phase filter
    reward = _apply_phase_filter(env, reward, phases)
    
    return reward


# ==================== Stability Penalties ====================


def finger_regularizer(
    env: ManagerBasedRLEnv,
    default_joints: list[float],
    std: float = 0.5,
    phases: list[str] | None = None,
    arm_joint_count: int = 7,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for finger joint deviation from default neutral pose.
    
    Returns a penalty in [0, 1] based on L2 distance from default pose.
    Penalty is 0 at default pose, approaches 1 as fingers deviate.
    
    Args:
        env: The environment.
        default_joints: Default finger joint positions (16 values for Allegro).
            Must be in canonical order: index(4), middle(4), ring(4), thumb(4).
        std: How fast penalty grows with deviation (radians).
        phases: List of phases to be active in. None = all phases.
        arm_joint_count: Number of arm joints (default 7 for Kuka).
        robot_cfg: Scene entity config for robot.
    
    Returns:
        Penalty tensor (num_envs,) in range [0, 1]. Multiply by negative weight.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    device = env.device
    
    # Use explicit joint IDs to ensure correct order (not PhysX native order)
    joint_ids = get_allegro_hand_joint_ids(env, robot)
    finger_pos = robot.data.joint_pos[:, joint_ids]  # (N, 16) in canonical order
    
    # Default joints tensor (must be in canonical order: index, middle, ring, thumb)
    default_tensor = torch.tensor(default_joints, device=device, dtype=torch.float32)  # (16,)
    
    # L2 distance from default
    finger_error = (finger_pos - default_tensor).norm(dim=-1)  # (N,)
    
    # Penalty: 0 at default, approaches 1 as error grows
    penalty = 1.0 - torch.exp(-finger_error / std)
    
    # Apply phase filter
    penalty = _apply_phase_filter(env, penalty, phases)
    
    return penalty


def object_stillness(
    env: ManagerBasedRLEnv,
    lin_std: float = 0.05,
    ang_std: float = 0.2,
    phases: list[str] | None = None,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Penalty for object movement (velocity) during grasp/release phases.
    
    Returns a penalty in [0, 1] based on object linear and angular velocity.
    Penalty is 0 when object is still, approaches 1 with more movement.
    
    Args:
        env: The environment.
        lin_std: Scaling for linear velocity penalty (m/s).
        ang_std: Scaling for angular velocity penalty (rad/s).
        phases: List of phases to be active in. None = all phases.
        object_cfg: Scene entity config for object.
    
    Returns:
        Penalty tensor (num_envs,) in range [0, 1]. Multiply by negative weight.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    
    # Get object velocities
    lin_vel = obj.data.root_lin_vel_w.norm(dim=-1)  # (N,)
    ang_vel = obj.data.root_ang_vel_w.norm(dim=-1)  # (N,)
    
    # Penalty using tanh (smooth, bounded)
    lin_penalty = torch.tanh(lin_vel / lin_std)
    ang_penalty = torch.tanh(ang_vel / ang_std)
    
    # Combined penalty (average)
    penalty = (lin_penalty + ang_penalty) / 2.0
    
    # Apply phase filter
    penalty = _apply_phase_filter(env, penalty, phases)
    
    return penalty
