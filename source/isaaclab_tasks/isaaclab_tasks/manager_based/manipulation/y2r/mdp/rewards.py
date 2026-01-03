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

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_hand_pose_gate(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get hand pose gate factor, defaulting to 1.0 if not set.
    
    The gate is computed by hand_pose_following reward and stored on env.
    During grasp and release phases, it ramps from 1→0 as hand deviates from target.
    """
    if hasattr(env, 'hand_pose_gate'):
        return env.hand_pose_gate
    return torch.ones(env.num_envs, device=env.device)


def _get_release_mask(env: ManagerBasedRLEnv, disable_in_release: bool) -> torch.Tensor:
    """Get mask for release phase disabling.
    
    Returns a tensor that is 0 during release phase (if disable_in_release=True),
    and 1 otherwise. Multiply reward by this mask to disable during release.
    
    Args:
        env: The environment.
        disable_in_release: If True, mask is 0 during release phase.
    
    Returns:
        Mask tensor (num_envs,) - 0 if in release and disabled, 1 otherwise.
    """
    if not disable_in_release:
        return torch.ones(env.num_envs, device=env.device)
    
    if not hasattr(env, 'trajectory_manager') or env.trajectory_manager is None:
        return torch.ones(env.num_envs, device=env.device)
    
    in_release = env.trajectory_manager.is_in_release_phase()  # (N,) bool
    return (~in_release).float()  # 1 if NOT in release, 0 if in release


def _get_table_contact_gate(
    env: ManagerBasedRLEnv,
    threshold: float = 0.5,
    ramp: float = 2.0,
    floor: float = 0.2,
) -> torch.Tensor:
    """Get table contact gate factor for gating rewards during release.
    
    Returns a factor in [floor, 1.0] based on object-table contact force:
    - No contact: returns floor (e.g., 0.2)
    - Good contact: returns 1.0
    - Smooth ramp between threshold and threshold+ramp
    
    This gates rewards so "holding object at goal" doesn't get full credit -
    must actually place on table.
    
    Args:
        env: The environment.
        threshold: Contact force threshold to start ramping (Newtons).
        ramp: Range over which gate ramps from floor to 1.0 (Newtons).
        floor: Minimum gate value when no contact (0-1).
    
    Returns:
        Gate tensor (num_envs,) in range [floor, 1.0].
    """
    # Check if sensor exists
    if "object_table_contact" not in env.scene.sensors:
        return torch.ones(env.num_envs, device=env.device)
    
    contact_sensor: ContactSensor = env.scene.sensors["object_table_contact"]
    contact_force = contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    contact_mag = contact_force.norm(dim=-1)  # (N,)
    
    # Smooth ramp: 0 at threshold, 1 at threshold+ramp
    raw_factor = ((contact_mag - threshold) / ramp).clamp(0.0, 1.0)
    
    # Apply floor: gate ranges from floor to 1.0
    gate = floor + (1.0 - floor) * raw_factor
    
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
    error_gate_pos_threshold: float | None = None,
    error_gate_pos_slope: float = 0.02,
    error_gate_rot_threshold: float | None = None,
    error_gate_rot_slope: float = 0.5,
    disable_in_release: bool = False,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward reaching the object using a tanh-kernel on end-effector distance.

    The reward is close to 1 when the maximum distance between the object and any end-effector body is small.
    Can be disabled during release phase via disable_in_release flag.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    asset_pos = asset.data.body_pos_w[:, asset_cfg.body_ids]
    object_pos = object.data.root_pos_w
    object_ee_distance = torch.norm(asset_pos - object_pos[:, None, :], dim=-1).max(dim=-1).values
    reward = 1 - torch.tanh(object_ee_distance / std)

    # Error-dependent gate: when tracking is clearly bad, reduce this reward so that
    # "freeze + squeeze to keep contact" is not a good local optimum.
    # (Only active when thresholds are provided.)
    if error_gate_pos_threshold is not None:
        cfg = env.cfg.y2r_cfg
        if not cfg.mode.use_point_cloud:
            pos_err = env._cached_pose_errors["pos"][:, 0]  # (N,)
            # gate in [0,1], ~1 when pos_err << threshold, ~0 when pos_err >> threshold
            gate = torch.sigmoid((error_gate_pos_threshold - pos_err) / max(error_gate_pos_slope, 1e-6))
            if error_gate_rot_threshold is not None:
                rot_err = env._cached_pose_errors["rot"][:, 0]
                gate_rot = torch.sigmoid((error_gate_rot_threshold - rot_err) / max(error_gate_rot_slope, 1e-6))
                gate = gate * gate_rot
        else:
            mean_err = env._cached_mean_errors[:, 0]  # (N,)
            gate = torch.sigmoid((error_gate_pos_threshold - mean_err) / max(error_gate_pos_slope, 1e-6))

        reward = reward * gate
    
    # Apply release phase mask
    reward = reward * _get_release_mask(env, disable_in_release)
    
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


def contacts(env: ManagerBasedRLEnv, threshold: float, disable_in_release: bool = False) -> torch.Tensor:
    """Reward for good finger contact (thumb + at least one other finger).
    
    Returns a gated float reward (0.0 or 1.0 * gate).
    Gate reduces reward during release if hand is not following target trajectory.
    Can be disabled during release phase via disable_in_release flag.
    """
    good_contact = _contacts_bool(env, threshold)
    reward = good_contact.float() * _get_hand_pose_gate(env)
    # Apply release phase mask
    reward = reward * _get_release_mask(env, disable_in_release)
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
    contact_threshold: float = 0.05,
    contact_ramp: float = 1.0,
    contact_min_factor: float = 0.05,
    rot_std: float = 0.5,
    neg_threshold: float = 0.08,
    neg_std: float = 0.1,
    neg_scale: float = 0.5,
    rot_neg_threshold: float = 0.6,
    rot_neg_std: float = 0.4,
    disable_in_release: bool = False,
) -> torch.Tensor:
    """
    Reward tracking future trajectory targets with positive and negative zones.
    
    Supports two modes based on cached errors:
    - Point cloud mode: Uses _cached_mean_errors (point-to-point distance)
    - Pose mode: Uses _cached_pose_errors (position + rotation errors)
    
    Two-kernel reward structure:
    - Positive reward: exp(-error/std) for being close to target
    - Negative penalty: tanh((error - neg_threshold)/neg_std) when error > neg_threshold
    - Combined: reward = pos_reward - neg_scale * neg_penalty
    
    In pose mode, separate negative zones for position and rotation errors.
    
    Args:
        env: The environment.
        std: Standard deviation for positive position reward exp kernel.
        decay: Exponential decay factor (0-1). Lower = focus on current target.
        contact_threshold: Force threshold for contact detection (N) - currently unused.
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
    
    # DEBUG: Print all rewards for env 0 every 10 steps (controlled by config visualization.debug_print_rewards)
    step = env.episode_length_buf[0].item()
    if cfg.visualization.debug_print_rewards and step % 10 == 0:
        rm = env.reward_manager
        print(f"\n=== Episode Step {step} | Env 0 Rewards ===")
        # Use get_active_iterable_terms to get per-term values
        for name, values in rm.get_active_iterable_terms(0):
            term_cfg = rm.get_term_cfg(name)
            print(f"  {name:20s}: weighted={values[0]:+.4f} (w={term_cfg.weight:+.2f})")
    
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
        pos_reward = (pos_pos_reward + pos_rot_reward * 2) / 3.0

        # Soft contact gating (prevents reward collapse when contact is weak)
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
        cfac = contact_factor(env, threshold=contact_threshold, ramp=contact_ramp, min_factor=contact_min_factor)
        pos_reward = pos_reward * cfac
        
        # Negative penalty: activates when error > neg_threshold
        excess_error = torch.clamp(weighted_error - neg_threshold, min=0.0)
        neg_penalty = torch.tanh(excess_error / neg_std)
        
        reward = pos_reward - neg_scale * neg_penalty
    
    # Apply hand pose gate (reduces reward during release if hand not following target)
    reward = reward * _get_hand_pose_gate(env)
    
    # Apply release phase mask
    reward = reward * _get_release_mask(env, disable_in_release)
    
    return reward


def tracking_progress(
    env: ManagerBasedRLEnv,
    pos_weight: float = 1.0,
    rot_weight: float = 0.5,
    positive_only: bool = False,
    clip: float = 1.0,
    disable_in_release: bool = False,
) -> torch.Tensor:
    """Reward improvement in tracking error over ~1/target_hz seconds (relative).

    Uses cached errors produced by the target observation term:
    - Pose mode: combines position + rotation error of the *current* target (index 0).
    - Point cloud mode: uses mean point error of the current target (index 0).

    This is meant to be **off by default** (weight=0) and turned on only if needed.
    Can be disabled during release phase via disable_in_release flag.
    """
    cfg = env.cfg.y2r_cfg
    if not cfg.mode.use_point_cloud:
        pos_err = env._cached_pose_errors["pos"][:, 0]
        rot_err = env._cached_pose_errors["rot"][:, 0]
        cur_err = pos_weight * pos_err + rot_weight * rot_err
    else:
        cur_err = env._cached_mean_errors[:, 0]

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
    
    # Apply hand pose gate
    progress = progress * _get_hand_pose_gate(env)
    
    # Apply release phase mask
    progress = progress * _get_release_mask(env, disable_in_release)
    
    return progress


def trajectory_success(
    env: ManagerBasedRLEnv,
    pos_threshold: float = 0.05,
    rot_threshold: float = 0.3,
    pos_std: float = 0.1,
    rot_std: float = 0.5,
    sparse_weight: float = 0.7,
    contact_gate_threshold: float = 0.5,
    contact_gate_ramp: float = 2.0,
    contact_gate_floor: float = 0.2,
) -> torch.Tensor:
    """
    Dense + Sparse success reward: combines shaping gradient with binary bonus.
    
    Dense component: Always provides gradient toward goal (exp kernel).
    Sparse component: Binary bonus when BOTH pos AND rot are within thresholds,
                     GATED by table contact (must actually place, not just hold).
    
    Combined: reward = (1 - sparse_weight) * dense + sparse_weight * sparse * contact_gate
    
    This gives:
    - Smooth gradient to guide toward goal (dense shaping, always active)
    - Big reward only when actually placed on table (sparse bonus gated by contact)
    - Prevents "hold object at goal" exploit - must release onto table
    
    Supports two modes based on cached errors:
    - Point cloud mode: Uses _cached_mean_errors (position only)
    - Pose mode: Uses _cached_pose_errors (position + rotation)
    
    Args:
        env: The environment.
        pos_threshold: Position threshold for sparse bonus (meters).
        rot_threshold: Rotation threshold for sparse bonus (radians).
        pos_std: Position std for dense shaping exp kernel (meters).
        rot_std: Rotation std for dense shaping exp kernel (radians).
        sparse_weight: Weight of sparse bonus (0-1). Dense weight = 1 - sparse_weight.
        contact_gate_threshold: Contact force threshold to start gating (Newtons).
        contact_gate_ramp: Range over which contact gate ramps from floor to 1.0 (Newtons).
        contact_gate_floor: Minimum gate value when no table contact (0-1).
    
    Returns:
        Reward tensor (num_envs,) in range [0, 1].
    """
    cfg = env.cfg.y2r_cfg
    use_point_cloud = cfg.mode.use_point_cloud
    
    trajectory_manager = env.trajectory_manager
    
    # Only active in release phase
    in_release = trajectory_manager.is_in_release_phase()  # (N,) bool
    
    if not use_point_cloud:
        # Pose mode: position + rotation
        pos_error = env._cached_pose_errors['pos'][:, 0]  # (N,)
        rot_error = env._cached_pose_errors['rot'][:, 0]  # (N,)
        
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
        mean_errors = env._cached_mean_errors  # (N, W)
        current_error = mean_errors[:, 0]  # (N,)
        
        # Dense component
        dense = torch.exp(-current_error / pos_std)
        
        # Sparse component
        sparse = (current_error < pos_threshold).float()
    
    # Gate sparse component by table contact (must actually place, not hold)
    contact_gate = _get_table_contact_gate(
        env, 
        threshold=contact_gate_threshold, 
        ramp=contact_gate_ramp, 
        floor=contact_gate_floor
    )
    sparse_gated = sparse * contact_gate
    
    # Combine: dense shaping (ungated) + sparse bonus (contact-gated)
    reward = (1.0 - sparse_weight) * dense + sparse_weight * sparse_gated
    
    # Only give reward in release phase
    reward = torch.where(in_release, reward, torch.zeros_like(reward))
    
    # Apply hand pose gate
    reward = reward * _get_hand_pose_gate(env)
    
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
    disable_in_release: bool = False,
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
    
    Can be disabled during release phase via disable_in_release flag.
    
    Args:
        env: The environment.
        pos_std: Position change threshold in meters (default 0.01 = 1cm).
        rot_std: Rotation change threshold in radians (default 0.1 ≈ 6°).
        disable_in_release: If True, reward returns 0 during release phase.
        robot_cfg: Scene entity config for robot (must have palm_link body).
        object_cfg: Scene entity config for the manipulated object.
    
    Returns:
        Reward tensor (num_envs,) in range [0, 1].
    """
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    cfg = env.cfg.y2r_cfg
    
    # Get palm body index (cache on first call)
    if not hasattr(env, '_palm_body_idx'):
        palm_ids = robot.find_bodies("palm_link")[0]
        if len(palm_ids) == 0:
            # Fallback: no palm_link found, return zero reward
            return torch.zeros(env.num_envs, device=env.device)
        env._palm_body_idx = palm_ids[0]
    
    palm_idx = env._palm_body_idx
    
    # Get palm pose in world frame
    palm_pos_w = robot.data.body_pos_w[:, palm_idx]  # (N, 3)
    palm_quat_w = robot.data.body_quat_w[:, palm_idx]  # (N, 4)
    
    # Get object pose in world frame
    object_pos_w = obj.data.root_pos_w  # (N, 3)
    object_quat_w = obj.data.root_quat_w  # (N, 4)
    
    # Transform object pose to palm frame
    # Position: rotate (object_pos - palm_pos) by inverse palm rotation
    object_pos_palm = quat_apply_inverse(palm_quat_w, object_pos_w - palm_pos_w)  # (N, 3)
    
    # Orientation: object_quat in palm frame = inv(palm_quat) * object_quat
    object_quat_palm = quat_mul(quat_inv(palm_quat_w), object_quat_w)  # (N, 4)
    
    # Get current tracking error (use cached errors from observation computation)
    use_point_cloud = cfg.mode.use_point_cloud
    if use_point_cloud:
        # Point cloud mode: use mean error of current target
        current_error = env._cached_mean_errors[:, 0]  # (N,)
    else:
        # Pose mode: combine position and rotation errors
        pos_error = env._cached_pose_errors['pos'][:, 0]  # (N,)
        rot_error = env._cached_pose_errors['rot'][:, 0]  # (N,)
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
    
    # Apply hand pose gate
    reward = reward * _get_hand_pose_gate(env)
    
    # Apply release phase mask
    reward = reward * _get_release_mask(env, disable_in_release)
    
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
    
    # Get palm body index (reuse cached value if available from finger_manipulation)
    if not hasattr(env, '_palm_body_idx'):
        palm_ids = robot.find_bodies("palm_link")[0]
        if len(palm_ids) == 0:
            return torch.zeros(env.num_envs, device=env.device)
        env._palm_body_idx = palm_ids[0]
    
    palm_idx = env._palm_body_idx
    
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
    trajectory_manager = env.trajectory_manager
    in_manipulation = trajectory_manager.get_phase() == 1  # (N,) bool
    
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
    
    # Get palm body index (reuse cached if available)
    if not hasattr(env, '_palm_body_idx'):
        palm_ids = robot.find_bodies("palm_link")[0]
        if len(palm_ids) == 0:
            return torch.zeros(env.num_envs, device=env.device)
        env._palm_body_idx = palm_ids[0]
    
    palm_idx = env._palm_body_idx
    
    # Get current palm orientation in world frame
    palm_quat_w = robot.data.body_quat_w[:, palm_idx]  # (N, 4)
    
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
    only_in_manipulation: bool = True,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Penalize magnitude of distal Allegro joints (joint_3) to discourage over-curling.

    This targets the posture that tends to produce "nail-side" / folded-in grasps.
    The penalty is bounded using a tanh kernel and can be gated:
    - by contact (thumb + at least one other finger) and/or
    - by manipulation phase (phase == 1).

    Args:
        env: The environment.
        std: Scale (rad) for tanh kernel. Larger = more lenient.
        joint_name_regex: Regex to select distal joints. Defaults to ".*_joint_3".
        only_when_contact: If True, only apply penalty when grasp contact is present.
        contact_threshold: Threshold (N) for contact detection used by `contacts()`.
        only_in_manipulation: If True, only apply penalty in manipulation phase.
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
        has_contact = contacts(env, contact_threshold)
        penalty = torch.where(has_contact, penalty, torch.zeros_like(penalty))

    # Gate by manipulation phase if requested
    if only_in_manipulation and hasattr(env, "trajectory_manager") and env.trajectory_manager is not None:
        in_manip = env.trajectory_manager.get_phase() == 1
        penalty = torch.where(in_manip, penalty, torch.zeros_like(penalty))

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
    gate_in_release: bool = True,
    gate_start_threshold: float = 0.06,
    gate_end_threshold: float = 0.12,
    gate_rot_start_threshold: float = 0.4,
    gate_rot_end_threshold: float = 0.8,
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
    The gate is active during grasp and release phases and smoothly ramps from 1→gate_floor
    as position OR rotation error increases beyond thresholds.
    
    Args:
        env: The environment.
        grasp_pos_tol: Position tolerance during grasp phase (meters).
        grasp_rot_tol: Rotation tolerance during grasp phase (radians).
        manipulation_pos_tol: Position tolerance during manipulation phase (meters).
        manipulation_rot_tol: Rotation tolerance during manipulation phase (radians).
        release_pos_tol: Position tolerance during release phase (meters).
        release_rot_tol: Rotation tolerance during release phase (radians).
        gate_in_release: If True, compute gate factor for other rewards.
        gate_start_threshold: Position error where gate starts dropping (meters).
        gate_end_threshold: Position error where gate reaches minimum (meters).
        gate_rot_start_threshold: Rotation error where gate starts dropping (radians).
        gate_rot_end_threshold: Rotation error where gate reaches minimum (radians).
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
    
    # Get palm body index (cache it)
    if not hasattr(env, '_palm_body_idx'):
        palm_ids = robot.find_bodies("palm_link")[0]
        if len(palm_ids) == 0:
            env.hand_pose_gate = torch.ones(N, device=env.device)
            return torch.zeros(N, device=env.device)
        env._palm_body_idx = palm_ids[0]
    
    palm_idx = env._palm_body_idx
    
    # Get actual palm pose in world frame
    palm_pos_w = robot.data.body_pos_w[:, palm_idx]  # (N, 3)
    palm_quat_w = robot.data.body_quat_w[:, palm_idx]  # (N, 4)
    
    # Get target palm pose from trajectory
    hand_target = trajectory_manager.get_current_hand_target()  # (N, 7)
    target_pos_w = hand_target[:, :3]  # (N, 3)
    target_quat_w = hand_target[:, 3:7]  # (N, 4)
    
    # Compute errors
    pos_error = (palm_pos_w - target_pos_w).norm(dim=-1)  # (N,)
    rot_error = quat_error_magnitude(palm_quat_w, target_quat_w)  # (N,)
    
    # Get current phase (0=grasp, 1=manipulation, 2=release)
    phase = trajectory_manager.get_phase()  # (N,)
    
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
    
    # Compute gate factor for grasp and release phases
    if gate_in_release:
        in_grasp = (phase == 0)
        in_release = (phase == 2)
        in_gated_phase = in_grasp | in_release
        
        # Position gate: linear ramp 1.0 → gate_floor as error exceeds threshold
        pos_gate_range = gate_end_threshold - gate_start_threshold
        pos_gate = 1.0 - (pos_error - gate_start_threshold) / pos_gate_range
        pos_gate = pos_gate.clamp(0.0, 1.0)
        
        # Rotation gate: linear ramp 1.0 → gate_floor as error exceeds threshold
        rot_gate_range = gate_rot_end_threshold - gate_rot_start_threshold
        rot_gate = 1.0 - (rot_error - gate_rot_start_threshold) / rot_gate_range
        rot_gate = rot_gate.clamp(0.0, 1.0)
        
        # Combined gate: minimum of position and rotation gates
        # If either error is too large, the gate drops
        gate = torch.min(pos_gate, rot_gate)
        
        # Apply floor so gate never goes below gate_floor
        gate = gate * (1.0 - gate_floor) + gate_floor
        
        # Only apply gate during grasp and release phases
        gate = torch.where(in_gated_phase, gate, torch.ones_like(gate))
        
        env.hand_pose_gate = gate
    else:
        env.hand_pose_gate = torch.ones(N, device=env.device)
    
    return reward


# ==================== Finger Release ====================


from .actions import ALLEGRO_PCA_MATRIX


def finger_release(
    env: ManagerBasedRLEnv,
    scale: float = 1.0,
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
    
    Only active during release phase.
    
    Args:
        env: The environment.
        scale: Controls gradient steepness. Higher = stronger push toward open.
        arm_joint_count: Number of arm joints (default 7 for Kuka).
        hand_joint_count: Number of hand joints (default 16 for Allegro).
        robot_cfg: Scene entity config for robot.
    
    Returns:
        Reward tensor (num_envs,) in range (0, 1).
    """
    cfg = env.cfg.y2r_cfg
    N = env.num_envs
    device = env.device
    
    # If hand trajectory is disabled, return zeros
    if not cfg.hand_trajectory.enabled:
        return torch.zeros(N, device=device)
    
    trajectory_manager = env.trajectory_manager
    
    # Only active in release phase
    in_release = trajectory_manager.is_in_release_phase()  # (N,) bool
    if not in_release.any():
        return torch.zeros(N, device=device)
    
    robot: Articulation = env.scene[robot_cfg.name]
    
    # Get hand joint positions (relative to default)
    start_idx = arm_joint_count
    end_idx = arm_joint_count + hand_joint_count
    hand_pos = robot.data.joint_pos[:, start_idx:end_idx]  # (N, 16)
    default_pos = robot.data.default_joint_pos[:, start_idx:end_idx]  # (N, 16)
    hand_delta = hand_pos - default_pos  # (N, 16)
    
    # Project onto first eigengrasp basis (open/close synergy)
    # Negative coefficient = more open than default
    # Positive coefficient = more closed than default
    eigen_basis = ALLEGRO_PCA_MATRIX[0].to(device)  # (16,)
    eigen_coeff = torch.matmul(hand_delta, eigen_basis)  # (N,)
    
    # Sigmoid reward: more negative coeff (more open) = higher reward
    # sigmoid(-coeff * scale) gives smooth gradient in both directions
    reward = torch.sigmoid(-eigen_coeff * scale)
    
    # Only give reward in release phase
    reward = torch.where(in_release, reward, torch.zeros_like(reward))
    
    return reward
