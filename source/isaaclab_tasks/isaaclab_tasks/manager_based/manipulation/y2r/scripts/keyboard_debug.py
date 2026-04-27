# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2024, Harsh Gupta
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Keyboard Debug Script for Palm Frame Exploration
=================================================

Allows manual control of the robot arm via keyboard to explore palm orientations.
Uses the full y2r environment with all visualization features.

Controls:
    WASDQE      - Move palm (X/Y/Z)
    ZXTGCV      - Rotate palm (roll/pitch/yaw)
    J/K         - Close/Open selected joint (or eigengrasp)
    Up/Down     - Cycle through joint modes
    R           - Reset pose
    ESC         - Quit

Joint modes (cycle with Up/Down):
    [0] Eigengrasp  - All fingers via PCA synergy
    [1..16]         - Individual finger joints

Usage:
    ./y2r_sim/run/keyboard.sh
"""

import argparse
import sys

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Keyboard debug for palm frame exploration.")
parser.add_argument("--task", type=str, default="Isaac-Trajectory-UR5e-Leap-v0", help="Task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--depth", action="store_true", help="Enable wrist depth camera. Press P to save images.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Clear sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# Launch simulation
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows after simulation launch."""

import gymnasium as gym
import torch
import time
from scipy.spatial.transform import Rotation as R

import carb
import omni

# Isaac Lab imports
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices.keyboard import Se3Keyboard, Se3KeyboardCfg
from isaaclab.utils.math import (
    matrix_from_quat, quat_inv, 
    subtract_frame_transforms, apply_delta_pose
)

# Import tasks to register them
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

from isaaclab_tasks.manager_based.manipulation.y2r.mdp.actions import ALLEGRO_PCA_MATRIX, ALLEGRO_HAND_JOINT_NAMES

from isaaclab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg, DirectMARLEnvCfg


def quat_to_euler_deg(quat: torch.Tensor) -> tuple[float, float, float]:
    """Convert quaternion (w, x, y, z) to Euler angles (roll, pitch, yaw) in degrees."""
    # quat shape: (4,) - [w, x, y, z]
    w, x, y, z = quat.cpu().numpy()
    r = R.from_quat([x, y, z, w])  # scipy uses [x, y, z, w]
    euler = r.as_euler('xyz', degrees=True)
    return float(euler[0]), float(euler[1]), float(euler[2])


def compute_finger_regularizer(
    finger_pos: torch.Tensor,
    default_joints: list,
    std: float,
) -> tuple[float, float]:
    """Mirror of mdp.finger_regularizer (RMS form), no phase filter.

    Returns (penalty in [0, 1], RMS deviation in radians).
    """
    device = finger_pos.device
    default_tensor = torch.tensor(default_joints, device=device, dtype=torch.float32)
    finger_error = (finger_pos - default_tensor).pow(2).mean().sqrt().item()
    penalty = 1.0 - float(torch.exp(torch.tensor(-finger_error / std)))
    return penalty, finger_error


@hydra_task_config(args_cli.task, "rl_games_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Main keyboard control loop."""
    
    # Force single environment for keyboard control
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device if args_cli.device else "cuda:0"

    # Inject wrist depth camera if --depth requested
    wrist_camera_enabled = args_cli.depth
    if wrist_camera_enabled:
        from scipy.spatial.transform import Rotation as RotScipy
        from isaaclab.sensors import TiledCameraCfg
        from isaaclab.sim import PinholeCameraCfg

        wrist_cfg = env_cfg.y2r_cfg.wrist_camera
        rot_euler = wrist_cfg.offset.rot
        # Isaac Lab opengl convention: swap Y/Z and negate
        rot_swapped = (rot_euler[0], rot_euler[2], -rot_euler[1])
        r_cam = RotScipy.from_euler('xyz', rot_swapped, degrees=True)
        quat_xyzw = r_cam.as_quat()
        quat_wxyz = (float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2]))
        pos = tuple(float(x) for x in wrist_cfg.offset.pos)

        # Detect prim path based on robot type
        palm_body = env_cfg.y2r_cfg.robot.palm_body_name
        if "ur5e" in palm_body:
            cam_prim = "{ENV_REGEX_NS}/Robot/ur5e_link_6/wrist_camera"
        else:
            cam_prim = "{ENV_REGEX_NS}/Robot/ee_link/palm_link/wrist_camera"

        env_cfg.scene.wrist_camera = TiledCameraCfg(
            prim_path=cam_prim,
            offset=TiledCameraCfg.OffsetCfg(pos=pos, rot=quat_wxyz, convention="opengl"),
            data_types=["distance_to_image_plane"],
            spawn=PinholeCameraCfg(
                focal_length=wrist_cfg.focal_length,
                horizontal_aperture=wrist_cfg.horizontal_aperture,
                clipping_range=wrist_cfg.clipping_range,
            ),
            width=wrist_cfg.width,
            height=wrist_cfg.height,
        )
        print(f"[INFO] Wrist depth camera enabled: {wrist_cfg.width}x{wrist_cfg.height}, "
              f"clip=[{wrist_cfg.clipping_range[0]}, {wrist_cfg.clipping_range[1]}]m")

    # ==================== DEBUG: finger_regularizer config (live, from env_cfg) ====================
    fr_cfg = env_cfg.y2r_cfg.rewards.finger_regularizer
    fr_weight = float(fr_cfg.weight)
    fr_default_joints = list(fr_cfg.params["default_joints"])
    fr_std = float(fr_cfg.params["std"])
    fr_phases = fr_cfg.params.get("phases")
    print("\n" + "=" * 80)
    print("FINGER_REGULARIZER CONFIG (live from env_cfg)")
    print("=" * 80)
    print(f"  weight  : {fr_weight}")
    print(f"  std     : {fr_std} rad")
    print(f"  phases  : {fr_phases}")
    print(f"  default_joints (16):")
    for i, val in enumerate(fr_default_joints):
        print(f"    [{i:2d}] {val:+.4f} rad ({val * 57.2958:+7.2f} deg)")
    print("=" * 80 + "\n")

    # Create environment
    print("[INFO] Creating environment...")
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    # Get unwrapped environment and robot
    unwrapped_env = env.unwrapped
    robot = unwrapped_env.scene["robot"]
    device = unwrapped_env.device
    
    # ==================== DEBUG: Print PhysX joint ordering ====================
    print("\n" + "=" * 80)
    print("DEBUG: JOINT ORDERING ANALYSIS")
    print("=" * 80)

    # Get PhysX native joint names (this is the order of robot.data.joint_pos columns)
    physx_joint_names = robot.data.joint_names
    print(f"\n[PhysX Native Order] Total joints: {len(physx_joint_names)}")
    for i, name in enumerate(physx_joint_names):
        print(f"  [{i:2d}] {name}")

    # Detect arm type from joint names
    if any("ur5e" in name for name in physx_joint_names):
        arm_joint_names = [f"ur5e_joint_{i+1}" for i in range(6)]
        arm_type = "ur5e"
    else:
        arm_joint_names = [f"iiwa7_joint_{i+1}" for i in range(7)]
        arm_type = "kuka"
    print(f"\n[Detected arm: {arm_type}] {len(arm_joint_names)} arm joints")

    arm_joint_ids, _ = robot.find_joints(arm_joint_names, preserve_order=True)
    arm_joint_ids = list(arm_joint_ids)
    
    print(f"\n[Arm Joint Mapping] (preserve_order=True)")
    for i, (name, physx_idx) in enumerate(zip(arm_joint_names, arm_joint_ids)):
        print(f"  Expected[{i}] {name:20s} -> PhysX index {physx_idx}")
    
    # Find hand joint IDs (16 joints for Allegro)
    hand_joint_names = ALLEGRO_HAND_JOINT_NAMES.copy()
    hand_joint_ids, _ = robot.find_joints(hand_joint_names, preserve_order=True)
    hand_joint_ids = list(hand_joint_ids)
    
    print(f"\n[Hand Joint Mapping] (preserve_order=True)")
    print("  Expected index -> Joint name -> PhysX index -> PhysX name")
    for i, (name, physx_idx) in enumerate(zip(hand_joint_names, hand_joint_ids)):
        physx_name = physx_joint_names[physx_idx]
        match = "✓" if name == physx_name else "✗ MISMATCH!"
        print(f"  [{i:2d}] {name:16s} -> PhysX[{physx_idx:2d}] = {physx_name:16s} {match}")
    
    
    # Print config default_joints (live from env_cfg)
    print(f"\n[Config default_joints from finger_regularizer (live)]")
    for i, (name, val) in enumerate(zip(hand_joint_names, fr_default_joints)):
        print(f"  [{i:2d}] {name:16s}: {val:+.4f} rad ({val * 57.2958:+7.2f} deg)")

    # Print robot's default joint positions for hand
    robot_default_hand = robot.data.default_joint_pos[0, hand_joint_ids].cpu().tolist()
    print(f"\n[Robot default_joint_pos (via find_joints)]")
    for i, (name, val) in enumerate(zip(hand_joint_names, robot_default_hand)):
        config_val = fr_default_joints[i]
        diff = abs(val - config_val)
        match = "✓" if diff < 0.01 else f"Δ={diff:.3f}"
        print(f"  [{i:2d}] {name:16s}: {val:+.4f} rad ({val * 57.2958:+7.2f} deg) vs config {config_val:+.4f} {match}")
    
    print("=" * 80 + "\n")
    
    # Find palm body index
    palm_name = unwrapped_env.cfg.y2r_cfg.robot.palm_body_name
    palm_ids = robot.find_bodies(palm_name)[0]
    if len(palm_ids) == 0:
        raise RuntimeError(f"Could not find '{palm_name}' body on robot!")
    palm_body_idx = palm_ids[0]
    
    # Correct Jacobian index depends on fixed vs floating base
    jacobi_body_idx = palm_body_idx - 1 if robot.is_fixed_base else palm_body_idx
    
    # Get first eigengrasp basis (primary open/close synergy)
    eigen_basis = ALLEGRO_PCA_MATRIX[0].to(device)  # Shape: (16,)
    
    # Eigengrasp coefficient state (positive = close, negative = open)
    eigen_coeff = torch.zeros(1, device=device)
    eigen_speed = 0.02  # Coefficient change per frame when key held
    eigen_range = [-2.0, 2.0]  # Min/max coefficient range
    
    # Setup IK controller - ABSOLUTE mode (tracks fixed target)
    ik_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,  # ABSOLUTE mode - track fixed target!
        ik_method="dls",
        ik_params={"lambda_val": 0.01},
    )
    ik_controller = DifferentialIKController(ik_cfg, num_envs=1, device=device)
    
    # Setup keyboard
    se3_keyboard_cfg = Se3KeyboardCfg(
        pos_sensitivity=0.005,   # 5mm per frame while key held
        rot_sensitivity=0.01,    # ~0.5 degrees per frame while key held
        gripper_term=True,
        sim_device=device,
    )
    se3_keyboard = Se3Keyboard(se3_keyboard_cfg)
    
    # ==================== OBSERVATION ORDER DIAGNOSTIC ====================
    print("\n" + "=" * 80)
    print("OBSERVATION ORDER DIAGNOSTIC")
    print("=" * 80)

    obs_mgr = unwrapped_env.observation_manager
    print(f"\nObservation groups: {list(obs_mgr.group_obs_term_dim.keys())}")
    for group_name, term_dims in obs_mgr.group_obs_term_dim.items():
        print(f"\n  [{group_name}] term_dims={term_dims}")

    # Print raw joint_pos from the default SceneEntityCfg (what mdp.joint_pos returns)
    raw_jp = robot.data.joint_pos[0].cpu().tolist()
    print(f"\n[mdp.joint_pos raw output] robot.data.joint_pos[0] (PhysX native order):")
    for i, val in enumerate(raw_jp):
        name = physx_joint_names[i]
        print(f"  [{i:2d}] {name:16s}: {val:+.4f}")

    # Print hand eigen (canonical order via get_allegro_hand_joint_ids)
    from isaaclab_tasks.manager_based.manipulation.y2r.mdp.actions import get_allegro_hand_joint_ids
    eigen_ids = get_allegro_hand_joint_ids(unwrapped_env, robot)
    hand_canonical = robot.data.joint_pos[0, eigen_ids].cpu().tolist()
    default_canonical = robot.data.default_joint_pos[0, eigen_ids].cpu().tolist()
    print(f"\n[allegro_hand_eigen_b] hand joints in CANONICAL order (via get_allegro_hand_joint_ids):")
    for i, (name, val, dflt) in enumerate(zip(hand_joint_names, hand_canonical, default_canonical)):
        print(f"  [{i:2d}] {name:16s}: pos={val:+.4f}  default={dflt:+.4f}  delta={val-dflt:+.4f}")

    # Reset environment
    print("[INFO] Resetting environment...")
    obs, info = env.reset()

    # Set unique joint positions so we can identify ordering
    # Assign value = (joint_index + 1) * 0.01 for each PhysX joint
    unique_pos = torch.zeros(1, 22, device=device)
    for i in range(22):
        unique_pos[0, i] = (i + 1) * 0.01  # 0.01, 0.02, ..., 0.22
    robot.write_joint_state_to_sim(unique_pos, torch.zeros_like(unique_pos))
    # Step physics so it takes effect
    unwrapped_env.sim.step(render=False)
    unwrapped_env.scene.update(dt=unwrapped_env.physics_dt)

    print(f"\n[Set unique positions] PhysX joint i = (i+1)*0.01:")
    for i, name in enumerate(physx_joint_names):
        val = robot.data.joint_pos[0, i].item()
        print(f"  PhysX[{i:2d}] {name:16s} = {val:.4f}")

    all_obs = obs_mgr.compute()
    for gname, gdata in all_obs.items():
        if isinstance(gdata, dict):
            continue
        flat = gdata[0].cpu()
        if "proprio" not in gname:
            continue
        print(f"\n[{gname} obs] shape={gdata.shape}")
        jp = flat[:22].tolist()
        print(f"  joint_pos (first 22 values) — match by value to PhysX index:")
        for i, val in enumerate(jp):
            # val ≈ (physx_idx+1)*0.01, so physx_idx ≈ round(val/0.01) - 1
            physx_idx = round(val / 0.01) - 1
            if 0 <= physx_idx < 22:
                name = physx_joint_names[physx_idx]
                print(f"    obs[{i:2d}] = {val:.6f} → PhysX[{physx_idx:2d}] = {name}")
            else:
                print(f"    obs[{i:2d}] = {val:.6f} → ???")
        if flat.numel() >= 27:
            eigen = flat[22:27].tolist()
            print(f"  hand_eigen (22-26): {[f'{v:+.4f}' for v in eigen]}")

    print("=" * 80 + "\n")
    
    # Initialize target pose (will be set after first sim step)
    target_pos_b = None  # (1, 3) - target position in base frame
    target_quat_b = None  # (1, 4) - target orientation in base frame
    default_pos_b = None  # Store initial pose as default for reset
    default_quat_b = None
    
    # Key state for reset (use list to allow modification in nested function)
    reset_state = {"pressed": False}
    
    def on_reset_key():
        reset_state["pressed"] = True
    
    # Add callback for 'R' key using Se3Keyboard's built-in mechanism
    se3_keyboard.add_callback("R", on_reset_key)
    
    # Joint mode: 0 = eigengrasp, 1..16 = individual joints
    # Mode names for display
    joint_mode_names = ["Eigengrasp"] + hand_joint_names
    joint_mode = [0]  # Mutable container: 0 = eigengrasp, 1-16 = individual joint index
    num_modes = len(joint_mode_names)  # 17 total (eigen + 16 joints)

    # Per-joint position offsets (added to default pos for individual joint control)
    joint_offsets = torch.zeros(16, device=device)
    joint_speed = 0.02  # rad per frame when key held

    # Gripper key states (J = close, K = open) - track press/release for continuous control
    gripper_delta = [0.0]  # Mutable container for closure access

    # Depth save state
    save_depth_state = {"requested": False, "count": 0}

    if wrist_camera_enabled:
        import os
        import numpy as np
        from pathlib import Path as _Path
        # keyboard.sh cds into IsaacLab/, so repo root is one level up from cwd
        _repo_root = str(_Path.cwd().parent)
        data_root = os.environ.get("Y2R_DATA_ROOT", _repo_root)
        depth_save_dir = os.path.join(data_root, "debug", "sim_depth")
        os.makedirs(depth_save_dir, exist_ok=True)
        print(f"[INFO] Depth images will be saved to: {depth_save_dir}")

    def _save_wrist_depth():
        """Read wrist camera and save normalized depth as 16-bit PNG + colorized PNG."""
        import numpy as np
        import cv2

        camera = unwrapped_env.scene.sensors["wrist_camera"]
        depth_raw = camera.data.output["distance_to_image_plane"]  # (1, H, W, 1)
        depth = depth_raw[0].squeeze(-1).cpu().numpy()  # (H, W) in meters

        wrist_cfg = unwrapped_env.cfg.y2r_cfg.wrist_camera
        near, far = wrist_cfg.clipping_range
        depth_clamped = np.clip(depth, near, far)
        depth_norm = (depth_clamped - near) / (far - near)  # [0, 1]

        idx = save_depth_state["count"]

        # Save raw normalized as 16-bit PNG (lossless, full precision)
        depth_u16 = (depth_norm * 65535).astype(np.uint16)
        raw_path = os.path.join(depth_save_dir, f"sim_depth_{idx:04d}_raw.png")
        cv2.imwrite(raw_path, depth_u16)

        # Save colorized version for easy viewing
        import matplotlib.cm as cm
        colored = (cm.viridis(depth_norm)[:, :, :3] * 255).astype(np.uint8)
        colored_bgr = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)
        vis_path = os.path.join(depth_save_dir, f"sim_depth_{idx:04d}_vis.png")
        cv2.imwrite(vis_path, colored_bgr)

        # Also save the raw float32 for exact numerical comparison
        npy_path = os.path.join(depth_save_dir, f"sim_depth_{idx:04d}.npy")
        np.save(npy_path, depth_norm)

        save_depth_state["count"] += 1
        print(f"  [DEPTH SAVED] #{idx}")
        print(f"    Raw 16-bit : {raw_path}")
        print(f"    Colorized  : {vis_path}")
        print(f"    Float32 npy: {npy_path}")
        print(f"    Shape: {depth_norm.shape}, range: [{depth_norm.min():.4f}, {depth_norm.max():.4f}]")
        print(f"    Meters range: [{depth_clamped.min():.4f}, {depth_clamped.max():.4f}]")

    snap_to_reg_state = {"pressed": False}

    def on_keyboard_event(event, *args):
        """Handle J/K for joint control and Up/Down for mode cycling."""
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "J":
                gripper_delta[0] += 1.0  # Close / positive direction
            elif event.input.name == "K":
                gripper_delta[0] -= 1.0  # Open / negative direction
            elif event.input.name == "UP":
                joint_mode[0] = (joint_mode[0] + 1) % num_modes
                mode_name = joint_mode_names[joint_mode[0]]
                print(f"  >> Joint mode: [{joint_mode[0]}] {mode_name}")
            elif event.input.name == "DOWN":
                joint_mode[0] = (joint_mode[0] - 1) % num_modes
                mode_name = joint_mode_names[joint_mode[0]]
                print(f"  >> Joint mode: [{joint_mode[0]}] {mode_name}")
            elif event.input.name == "B":
                snap_to_reg_state["pressed"] = True
            elif event.input.name == "P" and wrist_camera_enabled:
                save_depth_state["requested"] = True
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name == "J":
                gripper_delta[0] -= 1.0
            elif event.input.name == "K":
                gripper_delta[0] += 1.0
        return True

    # Subscribe to keyboard events
    appwindow = omni.appwindow.get_default_app_window()
    kb_input = carb.input.acquire_input_interface()
    keyboard = appwindow.get_keyboard()
    gripper_kb_sub = kb_input.subscribe_to_keyboard_events(
        keyboard,
        lambda event, *args: on_keyboard_event(event, *args),
    )
    
    # Timing for periodic prints
    last_print_time = time.time()
    print_interval = 2.0  # Print every 2 seconds (more time for debug output)
    
    print("\n" + "=" * 60)
    print("KEYBOARD DEBUG MODE ACTIVE")
    print("=" * 60)
    print("Controls:")
    print("  W/S     - Move forward/backward (X)")
    print("  A/D     - Move left/right (Y)")
    print("  Q/E     - Move up/down (Z)")
    print("  Z/X     - Roll (around X)")
    print("  T/G     - Pitch (around Y)")
    print("  C/V     - Yaw (around Z)")
    print("  J/K     - Close/Open (hold)")
    print("  Up/Down - Cycle joint mode")
    print("  R       - Reset to default pose")
    print("  B       - Snap hand to finger_regularizer default (penalty -> 0)")
    if wrist_camera_enabled:
        print("  P       - Save wrist depth image")
    print("  ESC     - Quit")
    print("")
    print("Joint modes (Up/Down to cycle):")
    for i, name in enumerate(joint_mode_names):
        marker = " >>" if i == 0 else "   "
        print(f"  {marker} [{i:2d}] {name}")
    print("=" * 60 + "\n")
    
    step_count = 0
    
    while simulation_app.is_running():
        # Get current palm pose (world frame)
        palm_pos_w = robot.data.body_pos_w[:, palm_body_idx]  # (1, 3)
        palm_quat_w = robot.data.body_quat_w[:, palm_body_idx]  # (1, 4)
        
        # Get root pose for frame conversion
        root_pos_w = robot.data.root_pos_w  # (1, 3)
        root_quat_w = robot.data.root_quat_w  # (1, 4)
        
        # Convert palm pose to robot base frame
        palm_pos_b, palm_quat_b = subtract_frame_transforms(
            root_pos_w, root_quat_w, palm_pos_w, palm_quat_w
        )  # (1, 3), (1, 4)
        
        # Initialize target on first step (also store as default for reset)
        if target_pos_b is None:
            target_pos_b = palm_pos_b.clone()
            target_quat_b = palm_quat_b.clone()
            default_pos_b = palm_pos_b.clone()
            default_quat_b = palm_quat_b.clone()
        
        # Handle reset key press
        if reset_state["pressed"]:
            target_pos_b = default_pos_b.clone()
            target_quat_b = default_quat_b.clone()
            eigen_coeff.zero_()  # Reset gripper to default
            joint_offsets.zero_()  # Reset individual joint offsets
            reset_state["pressed"] = False
            print("[INFO] Reset to default pose")

        # Handle B: snap hand target to finger_regularizer default_joints (penalty -> 0)
        if snap_to_reg_state["pressed"]:
            fr_default_tensor = torch.tensor(fr_default_joints, device=device, dtype=torch.float32)
            robot_default_hand_vec = robot.data.default_joint_pos[0, hand_joint_ids]  # (16,)
            joint_offsets[:] = fr_default_tensor - robot_default_hand_vec
            eigen_coeff.zero_()
            snap_to_reg_state["pressed"] = False
            print("[INFO] Snapped hand target to finger_regularizer default_joints (penalty target)")
        
        # Get keyboard command: [dx, dy, dz, rx, ry, rz, gripper]
        kb_cmd = se3_keyboard.advance()
        delta_pose = kb_cmd[:6]  # Position and rotation deltas
        gripper_cmd = kb_cmd[6].item() if len(kb_cmd) > 6 else 1.0
        
        # Update TARGET (not current pose!) based on keyboard input
        # This is the key fix: target accumulates, doesn't chase current position
        if delta_pose.abs().sum() > 0.001:
            target_pos_b, target_quat_b = apply_delta_pose(
                target_pos_b, target_quat_b, delta_pose.unsqueeze(0)
            )
        
        # Set IK command with ABSOLUTE target pose
        # In absolute mode, command is (N, 7) = [x, y, z, qw, qx, qy, qz]
        command = torch.cat([target_pos_b, target_quat_b], dim=-1)  # (1, 7)
        ik_controller.set_command(command)
        
        # Get Jacobian and transform to base frame
        jacobian = robot.root_physx_view.get_jacobians()[:, jacobi_body_idx, :, arm_joint_ids]  # (1, 6, N_arm)
        
        # Transform Jacobian from world to base frame
        base_rot_matrix = matrix_from_quat(quat_inv(root_quat_w))  # (1, 3, 3)
        jacobian_b = jacobian.clone()
        jacobian_b[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian_b[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
        
        # Get current arm joint positions
        arm_joint_pos = robot.data.joint_pos[:, arm_joint_ids]  # (1, N_arm)
        
        # Compute IK: finds joint positions that move current pose toward target
        arm_target = ik_controller.compute(
            ee_pos=palm_pos_b,
            ee_quat=palm_quat_b,
            jacobian=jacobian_b,
            joint_pos=arm_joint_pos
        )  # (1, N_arm)
        
        # Compute error from current to target
        pos_error = (target_pos_b - palm_pos_b).norm().item()
        
        # Set arm joint position targets
        robot.set_joint_position_target(arm_target, joint_ids=arm_joint_ids)
        
        # Update hand targets based on current joint mode
        mode = joint_mode[0]
        default_hand_pos = robot.data.default_joint_pos[:, hand_joint_ids]  # (1, 16)

        if mode == 0:
            # Eigengrasp mode: J/K control eigen coefficient
            eigen_coeff += gripper_delta[0] * eigen_speed
            eigen_coeff.clamp_(eigen_range[0], eigen_range[1])
            hand_delta = eigen_coeff * eigen_basis  # (16,)
            hand_target = default_hand_pos + joint_offsets.unsqueeze(0) + hand_delta.unsqueeze(0)
        else:
            # Individual joint mode: J/K control single joint
            joint_idx = mode - 1  # 0-15
            joint_offsets[joint_idx] += gripper_delta[0] * joint_speed
            hand_delta = eigen_coeff * eigen_basis
            hand_target = default_hand_pos + joint_offsets.unsqueeze(0) + hand_delta.unsqueeze(0)
        
        # Set hand joint position targets
        robot.set_joint_position_target(hand_target, joint_ids=hand_joint_ids)
        
        # Step simulation
        unwrapped_env.scene.write_data_to_sim()
        unwrapped_env.sim.step(render=True)
        unwrapped_env.scene.update(unwrapped_env.sim.get_physics_dt())
        if hasattr(unwrapped_env, 'render_debug'):
            unwrapped_env.render_debug()
        step_count += 1

        # Save wrist depth on P keypress (after sim step so camera has fresh data)
        if wrist_camera_enabled and save_depth_state["requested"]:
            _save_wrist_depth()
            save_depth_state["requested"] = False

        # Periodic status print
        current_time = time.time()
        if current_time - last_print_time >= print_interval:
            roll, pitch, yaw = quat_to_euler_deg(palm_quat_w[0])
            mode = joint_mode[0]
            mode_name = joint_mode_names[mode]
            if mode == 0:
                mode_info = f"Eigen: {eigen_coeff.item():.2f}"
            else:
                joint_idx = mode - 1
                cur_pos = robot.data.joint_pos[0, hand_joint_ids[joint_idx]].item()
                mode_info = f"{mode_name}: {cur_pos:.3f} rad ({cur_pos*57.296:.1f} deg)"
            print(f"[Step {step_count:6d}] Palm: ({palm_pos_w[0,0]:.3f}, {palm_pos_w[0,1]:.3f}, {palm_pos_w[0,2]:.3f}) | "
                  f"Euler: ({roll:.1f}, {pitch:.1f}, {yaw:.1f}) | [{mode}] {mode_info}")

            # finger_regularizer live print (uses live env_cfg values, ignores phase filter)
            finger_pos = robot.data.joint_pos[0, hand_joint_ids]  # (16,) canonical order
            penalty, rms_err = compute_finger_regularizer(finger_pos, fr_default_joints, fr_std)
            weighted_reward = fr_weight * penalty
            default_tensor = torch.tensor(fr_default_joints, device=device, dtype=torch.float32)
            per_joint_dev = (finger_pos - default_tensor).cpu().tolist()
            print(f"  [finger_regularizer] rms_err={rms_err:.4f} rad  penalty={penalty:.4f}  "
                  f"weighted={weighted_reward:+.4f}  (weight={fr_weight}, std={fr_std}, phases={fr_phases})")
            print(f"  [finger_regularizer] per-joint deviation (current - default) rad:")
            for i, name in enumerate(hand_joint_names):
                cur = finger_pos[i].item()
                dflt = fr_default_joints[i]
                print(f"    [{i:2d}] {name:16s} cur={cur:+.4f}  default={dflt:+.4f}  delta={per_joint_dev[i]:+.4f}")
            last_print_time = current_time
    
    # Cleanup
    env.close()
    print("[INFO] Keyboard debug ended.")


if __name__ == "__main__":
    main()
    simulation_app.close()
