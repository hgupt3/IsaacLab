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
    WASDQE  - Move palm (X/Y/Z)
    ZXTGCV  - Rotate palm (roll/pitch/yaw)
    J/K     - Close/Open gripper (eigengrasp)
    R       - Reset pose
    ESC     - Quit

Usage:
    ./isaac_scripts/keyboard.sh
"""

import argparse
import sys

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Keyboard debug for palm frame exploration.")
parser.add_argument("--task", type=str, default="Isaac-Trajectory-UR5e-Leap-v0", help="Task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
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


# ==================== DEBUG: default_joints from base.yaml ====================
# These are the values used in finger_regularizer reward
CONFIG_DEFAULT_JOINTS = [
    0.0315, 0.0764, 0.0196, 0.2944,   # index (0-3)
    -0.0010, 0.1109, 0.0309, 0.2541,  # middle (0-3)
    -0.0067, 0.1260, 0.0697, 0.2933,  # ring (0-3)
    1.5165, 0.5420, 0.3112, 0.3816    # thumb (0-3)
]


def debug_finger_regularizer(finger_pos: torch.Tensor, default_joints: list, std: float = 1.5) -> tuple[float, float]:
    """
    Compute finger_regularizer penalty exactly as the reward function does.
    
    Args:
        finger_pos: Current finger joint positions (16,)
        default_joints: Default joint positions from config (16,)
        std: Standard deviation for penalty computation
    
    Returns:
        Tuple of (penalty value in [0, 1], L2 error)
    """
    device = finger_pos.device
    default_tensor = torch.tensor(default_joints, device=device, dtype=torch.float32)
    finger_error = (finger_pos - default_tensor).norm().item()
    penalty = 1.0 - torch.exp(torch.tensor(-finger_error / std)).item()
    return penalty, finger_error


@hydra_task_config(args_cli.task, "rl_games_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Main keyboard control loop."""
    
    # Force single environment for keyboard control
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device if args_cli.device else "cuda:0"
    
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
    
    
    # Print config default_joints
    print(f"\n[Config default_joints from base.yaml]")
    for i, (name, val) in enumerate(zip(hand_joint_names, CONFIG_DEFAULT_JOINTS)):
        print(f"  [{i:2d}] {name:16s}: {val:+.4f} rad ({val * 57.2958:+7.2f} deg)")
    
    # Print robot's default joint positions for hand
    robot_default_hand = robot.data.default_joint_pos[0, hand_joint_ids].cpu().tolist()
    print(f"\n[Robot default_joint_pos (via find_joints)]")
    for i, (name, val) in enumerate(zip(hand_joint_names, robot_default_hand)):
        config_val = CONFIG_DEFAULT_JOINTS[i]
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
    
    # Reset environment
    print("[INFO] Resetting environment...")
    obs, info = env.reset()
    
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
    
    # Gripper key states (J = close, K = open) - track press/release for continuous control
    # Note: Using K for open since L is used for reset in Se3Keyboard
    gripper_delta = [0.0]  # Mutable container for closure access
    
    def on_gripper_keyboard_event(event, *args):
        """Handle J/K keys for continuous gripper control (eigengrasp)."""
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "J":
                gripper_delta[0] += eigen_speed  # Close (positive eigen coeff)
            elif event.input.name == "K":
                gripper_delta[0] -= eigen_speed  # Open (negative eigen coeff)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name == "J":
                gripper_delta[0] -= eigen_speed  # Stop closing
            elif event.input.name == "K":
                gripper_delta[0] += eigen_speed  # Stop opening
        return True
    
    # Subscribe to keyboard events for gripper control
    appwindow = omni.appwindow.get_default_app_window()
    kb_input = carb.input.acquire_input_interface()
    keyboard = appwindow.get_keyboard()
    gripper_kb_sub = kb_input.subscribe_to_keyboard_events(
        keyboard,
        lambda event, *args: on_gripper_keyboard_event(event, *args),
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
    print("  J       - Close gripper (hold)")
    print("  K       - Open gripper (hold)")
    print("  R       - Reset to default pose")
    print("  ESC     - Quit")
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
            reset_state["pressed"] = False
            print("[INFO] Reset to default pose")
        
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
        
        # Update eigengrasp coefficient based on gripper delta
        eigen_coeff += gripper_delta[0]
        eigen_coeff.clamp_(eigen_range[0], eigen_range[1])
        
        # Compute hand joint targets from eigengrasp
        # hand_target = default_pos + eigen_coeff * eigen_basis
        default_hand_pos = robot.data.default_joint_pos[:, hand_joint_ids]  # (1, 16)
        hand_delta = eigen_coeff * eigen_basis  # (1,) * (16,) = (16,)
        hand_target = default_hand_pos + hand_delta.unsqueeze(0)  # (1, 16)
        
        # Set hand joint position targets
        robot.set_joint_position_target(hand_target, joint_ids=hand_joint_ids)
        
        # Step simulation
        unwrapped_env.scene.write_data_to_sim()
        unwrapped_env.sim.step(render=True)
        unwrapped_env.scene.update(unwrapped_env.sim.get_physics_dt())
        if hasattr(unwrapped_env, 'render_debug'):
            unwrapped_env.render_debug()
        step_count += 1
        
        # Periodic status print
        current_time = time.time()
        if current_time - last_print_time >= print_interval:
            roll, pitch, yaw = quat_to_euler_deg(palm_quat_w[0])
            print(f"[Step {step_count:6d}] Palm: ({palm_pos_w[0,0]:.3f}, {palm_pos_w[0,1]:.3f}, {palm_pos_w[0,2]:.3f}) | "
                  f"Euler: ({roll:.1f}, {pitch:.1f}, {yaw:.1f}) | Error: {pos_error*100:.1f}cm | "
                  f"Grip: {eigen_coeff.item():.2f}")
            last_print_time = current_time
    
    # Cleanup
    env.close()
    print("[INFO] Keyboard debug ended.")


if __name__ == "__main__":
    main()
    simulation_app.close()
