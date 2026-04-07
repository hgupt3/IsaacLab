# Copyright (c) 2024, Harsh Gupta
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Dump Sim Observations & Actions (Pure One-Hot)
================================================

For each of 27 action dimensions, sends a one-hot action (action[i]=1.0, rest 0)
through the sim for N ticks and records the resulting joint positions, observations,
and eigen values. This reveals exactly what each action channel does.

Action layout: [arm(6) | eigen(5) | hand_raw(16)] = 27D

Usage:
    ./y2r_sim/run/dump_obs.sh
    ./y2r_sim/run/dump_obs.sh --interactive --tick-delay 0.2
"""

import argparse
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Dump sim observations (pure one-hot actions).")
parser.add_argument("--task", type=str, default="Isaac-Trajectory-UR5e-Leap-v0", help="Task name.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--out", type=str, default=None, help="Output npz path. Default: debug/sim_obs_actions.npz")
parser.add_argument("--interactive", action="store_true", help="Pause between tests, print per-tick details.")
parser.add_argument("--tick-delay", type=float, default=0.0, help="Seconds to sleep between ticks (slow down for viewer).")
parser.add_argument("--skip-to", type=int, default=0, help="Skip to test N.")
parser.add_argument("--ticks-per-test", type=int, default=10, help="Number of one-hot ticks per test.")
parser.add_argument("--settle-ticks", type=int, default=5, help="Zero-action ticks between tests.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows after simulation launch."""

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg, DirectMARLEnvCfg

from isaaclab_tasks.manager_based.manipulation.y2r.mdp.actions import (
    ALLEGRO_PCA_MATRIX, ALLEGRO_HAND_JOINT_NAMES, get_allegro_hand_joint_ids,
)


@hydra_task_config(args_cli.task, "rl_games_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device if args_cli.device else "cuda:0"
    env_cfg.episode_length_s = 9999.0

    print("[INFO] Creating environment...")
    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    device = unwrapped.device

    y2r_cfg = env_cfg.y2r_cfg

    # ── Resolve joint orderings ──────────────────────────────────────────────
    physx_joint_names = list(robot.data.joint_names)
    num_physx_joints = len(physx_joint_names)

    arm_regex = y2r_cfg.robot.arm_joint_regex
    arm_count = y2r_cfg.robot.arm_joint_count
    if "ur5e" in arm_regex:
        arm_joint_names = [f"ur5e_joint_{i+1}" for i in range(arm_count)]
    else:
        arm_joint_names = [f"iiwa7_joint_{i+1}" for i in range(7)]
    arm_ids, _ = robot.find_joints(arm_joint_names, preserve_order=True)
    arm_ids = list(arm_ids)

    hand_ids = get_allegro_hand_joint_ids(unwrapped, robot)
    all_canonical_ids = arm_ids + hand_ids.cpu().tolist()
    all_canonical_names = arm_joint_names + list(ALLEGRO_HAND_JOINT_NAMES)
    num_canonical = len(all_canonical_names)

    # ── Action manager info ─────────────────────────────────────────────────
    action_mgr = unwrapped.action_manager
    action_term = action_mgr._terms["action"]
    pca_matrix = action_term._pca_matrix.cpu().numpy()
    action_scale = float(action_term._scale) if isinstance(action_term._scale, float) else action_term._scale.cpu().numpy()
    eigen_dim = action_term._eigen_dim
    arm_dim = action_term._arm_dim
    hand_dim = action_term._hand_dim
    action_dim = arm_dim + eigen_dim + hand_dim  # 27

    default_joint_pos = robot.data.default_joint_pos[0].clone()

    # ── Observation manager info ─────────────────────────────────────────────
    obs_mgr = unwrapped.observation_manager
    print(f"\n[Obs groups] {list(obs_mgr.group_obs_term_dim.keys())}")
    for gname, tdims in obs_mgr.group_obs_term_dim.items():
        print(f"  {gname}: {tdims}")

    if "student_proprio" not in obs_mgr.group_obs_term_dim:
        print("\n[FATAL] student_proprio group not present — mode.use_student_mode must be true.")
        env.close()
        return

    # ── PCA pseudo-inverse ───────────────────────────────────────────────────
    A = torch.tensor(pca_matrix, dtype=torch.float32)
    right_pinv = A.T @ torch.linalg.inv(A @ A.T)

    # ── Helpers ──────────────────────────────────────────────────────────────
    zero_action = torch.zeros(1, action_dim, device=device)
    interactive = args_cli.interactive
    tick_delay = args_cli.tick_delay
    ticks_per_test = args_cli.ticks_per_test
    settle_ticks = args_cli.settle_ticks

    obs, info = env.reset()

    def force_to_defaults():
        """Snap all joints to default positions via direct state write."""
        robot.write_joint_state_to_sim(
            default_joint_pos.unsqueeze(0),
            torch.zeros(1, num_physx_joints, device=device),
        )
        unwrapped.scene.write_data_to_sim()
        # Step a few ticks with zero action to let physics/obs settle
        for _ in range(settle_ticks):
            env.step(zero_action)

    def record_state():
        """Record current state and return dict."""
        jp_physx = robot.data.joint_pos[0].cpu().numpy().copy()
        jp_canonical = robot.data.joint_pos[0, all_canonical_ids].cpu().numpy().copy()

        hand_canon = robot.data.joint_pos[0, hand_ids.tolist()].cpu()
        hand_default = default_joint_pos[hand_ids.tolist()].cpu()
        eigen_val = ((hand_canon - hand_default) @ right_pinv).numpy()

        all_obs = obs_mgr.compute()
        sp = all_obs["student_proprio"][0].cpu().numpy().copy()

        return {
            "jp_physx": jp_physx,
            "jp_canonical": jp_canonical,
            "eigen": eigen_val,
            "student_proprio": sp,
        }

    # ── Build test list: pure one-hot for each of 27 action dims ─────────
    num_tests = action_dim  # 27
    print(f"\n[INFO] Action layout: arm({arm_dim}) + eigen({eigen_dim}) + hand_raw({hand_dim}) = {action_dim}")
    print(f"[INFO] Running {num_tests} one-hot tests, {ticks_per_test} ticks each")

    # Label each action dimension
    action_labels = []
    for i in range(arm_dim):
        action_labels.append(f"arm[{i}] ({arm_joint_names[i]})")
    for i in range(eigen_dim):
        action_labels.append(f"eigen[{i}]")
    for i in range(hand_dim):
        action_labels.append(f"hand_raw[{i}] ({ALLEGRO_HAND_JOINT_NAMES[i]})")

    # ── Prepare recordings ───────────────────────────────────────────────────
    records = {
        "joint_pos_physx": [],
        "joint_pos_canonical": [],
        "eigen": [],
        "student_proprio": [],
        "action_raw": [],
        "test_idx": [],
        "tick_in_test": [],  # -1 = at defaults before test, 0..N-1 = one-hot ticks
    }

    # ── Run tests ────────────────────────────────────────────────────────────
    for test_i in range(num_tests):
        if test_i < args_cli.skip_to:
            continue

        print(f"\n{'='*70}")
        print(f"Test {test_i}/{num_tests-1}: action[{test_i}] = 1.0  ({action_labels[test_i]})")
        print('='*70)

        # Reset to defaults
        force_to_defaults()

        # Record at defaults (tick_in_test = -1)
        state = record_state()
        records["joint_pos_physx"].append(state["jp_physx"])
        records["joint_pos_canonical"].append(state["jp_canonical"])
        records["eigen"].append(state["eigen"])
        records["student_proprio"].append(state["student_proprio"])
        records["action_raw"].append(np.zeros(action_dim))
        records["test_idx"].append(test_i)
        records["tick_in_test"].append(-1)

        jp_default = state["jp_canonical"].copy()

        if interactive:
            input(f"  [At defaults] Press ENTER to apply one-hot action...")

        # Apply one-hot action for N ticks
        one_hot = torch.zeros(1, action_dim, device=device)
        one_hot[0, test_i] = 1.0

        for tick in range(ticks_per_test):
            env.step(one_hot)
            if tick_delay > 0:
                time.sleep(tick_delay)

            state = record_state()
            records["joint_pos_physx"].append(state["jp_physx"])
            records["joint_pos_canonical"].append(state["jp_canonical"])
            records["eigen"].append(state["eigen"])
            records["student_proprio"].append(state["student_proprio"])
            records["action_raw"].append(one_hot[0].cpu().numpy().copy())
            records["test_idx"].append(test_i)
            records["tick_in_test"].append(tick)

        # Report: what moved?
        jp_after = state["jp_canonical"]
        delta_deg = (jp_after - jp_default) * 57.2958
        top_idx = np.argsort(np.abs(delta_deg))[::-1]
        print(f"  After {ticks_per_test} ticks. Top movers:")
        for rank, j in enumerate(top_idx[:5]):
            if abs(delta_deg[j]) < 0.01:
                break
            print(f"    {all_canonical_names[j]:20s}: {delta_deg[j]:+7.2f} deg")

        # Eigen change
        eigen_default = records["eigen"][-ticks_per_test - 1]  # at defaults
        eigen_after = state["eigen"]
        eigen_delta = eigen_after - eigen_default
        print(f"  Eigen delta: {' '.join(f'e{i}={eigen_delta[i]:+.4f}' for i in range(eigen_dim))}")

        if interactive:
            input("  Press ENTER to continue...")

    # ── Save ─────────────────────────────────────────────────────────────────
    repo_root = Path.cwd().parent  # keyboard.sh cds into IsaacLab/
    data_root = os.environ.get("Y2R_DATA_ROOT", str(repo_root))
    out_path = args_cli.out or os.path.join(data_root, "debug", "sim_obs_actions.npz")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    save_dict = {}
    for k, v in records.items():
        save_dict[k] = np.array(v)

    # Metadata
    save_dict["joint_names_physx"] = np.array(physx_joint_names)
    save_dict["joint_names_canonical"] = np.array(all_canonical_names)
    save_dict["action_labels"] = np.array(action_labels)
    save_dict["pca_matrix"] = pca_matrix
    if isinstance(action_scale, float):
        save_dict["action_scale"] = np.array([action_scale])
    else:
        save_dict["action_scale"] = action_scale
    save_dict["default_joint_pos"] = default_joint_pos.cpu().numpy()
    save_dict["arm_dim"] = np.array([arm_dim])
    save_dict["eigen_dim"] = np.array([eigen_dim])
    save_dict["hand_dim"] = np.array([hand_dim])
    save_dict["action_dim"] = np.array([action_dim])
    save_dict["num_tests"] = np.array([num_tests])
    save_dict["ticks_per_test"] = np.array([ticks_per_test])

    np.savez(out_path, **save_dict)
    print(f"\n[SAVED] {out_path}")
    print(f"  Tests: {num_tests} (one-hot action dims 0..{num_tests-1})")
    print(f"  Ticks per test: {ticks_per_test}")
    print(f"  Records per test: 1 (defaults) + {ticks_per_test} (one-hot) = {ticks_per_test + 1}")
    print(f"  Total snapshots: {len(records['test_idx'])}")
    print(f"  Keys: {list(save_dict.keys())}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
