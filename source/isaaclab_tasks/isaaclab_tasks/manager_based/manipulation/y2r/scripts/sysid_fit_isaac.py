"""
SysID Fitting via Isaac Lab Parallel Environments
==================================================

Runs N parallel envs with different actuator parameters (stiffness, damping, friction),
replays real step response data, and uses CMA-ES to find parameters that match reality.

Usage:
    Y2R_MODE=sysid ./IsaacLab/isaaclab.sh -p \\
        source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/y2r/scripts/sysid_fit_isaac.py \\
        --task Isaac-Trajectory-UR5e-Leap-v0 \\
        --data /path/to/hand_*.npz \\
        --headless --num_envs 256
"""

import argparse
import glob as glob_mod
import os
import re
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="SysID fitting via Isaac Lab parallel envs.")
parser.add_argument("--task", type=str, default="Isaac-Trajectory-UR5e-Leap-v0")
parser.add_argument("--num_envs", type=int, default=None, help="Override num_envs (default: from sysid.yaml)")
parser.add_argument("--data", type=str, nargs="+", required=True, help=".npz files from sysid_collect.py")
parser.add_argument("--hand", action="store_true", help="Fit hand joints (default if no flag given)")
parser.add_argument("--arm", action="store_true", help="Fit arm joints")
parser.add_argument("--output", type=str, default=None, help="Output dir. Default: same dir as data.")
parser.add_argument("--maxiter", type=int, default=50, help="CMA-ES max iterations.")
parser.add_argument("--plot-only", action="store_true", help="Skip optimization, just plot with current sim params.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows after simulation launch."""

import cma
import gymnasium as gym
import numpy as np
import torch
import yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg, DirectMARLEnvCfg
from isaaclab_tasks.manager_based.manipulation.y2r.mdp.actions import ALLEGRO_HAND_JOINT_NAMES

# =========================================================================
# Joint groups
# =========================================================================
HAND_GROUPS = {
    'splay': ['index_joint_0', 'middle_joint_0', 'ring_joint_0'],
    'mcp': ['index_joint_1', 'middle_joint_1', 'ring_joint_1'],
    'pip': ['index_joint_2', 'middle_joint_2', 'ring_joint_2'],
    'dip': ['index_joint_3', 'middle_joint_3', 'ring_joint_3'],
    'thumb': ['thumb_joint_0', 'thumb_joint_1', 'thumb_joint_2', 'thumb_joint_3'],
}

ARM_GROUPS = {
    'ur5e_joint_1': ['ur5e_joint_1'],
    'ur5e_joint_2': ['ur5e_joint_2'],
    'ur5e_joint_3': ['ur5e_joint_3'],
    'ur5e_joint_4': ['ur5e_joint_4'],
    'ur5e_joint_5': ['ur5e_joint_5'],
    'ur5e_joint_6': ['ur5e_joint_6'],
}

# =========================================================================
# Load real data
# =========================================================================

def load_real_data(npz_paths):
    """Load trials from .npz files. Returns dict: joint_name → list of trial dicts."""
    all_trials = {}
    trial_pattern = re.compile(r'^(.+)_t(\d{2})_(.+)$')

    for path in npz_paths:
        data = dict(np.load(path, allow_pickle=True))
        trial_keys = {}
        for key in data.keys():
            m = trial_pattern.match(key)
            if m:
                joint_name, trial_idx, field = m.group(1), int(m.group(2)), m.group(3)
                tk = (joint_name, trial_idx)
                if tk not in trial_keys:
                    trial_keys[tk] = {}
                trial_keys[tk][field] = data[key]

        for (joint_name, _), fields in sorted(trial_keys.items()):
            if 'timestamps' not in fields or 'positions' not in fields:
                continue
            # Normalize key names (arm collector used 'center'/'target' before fix)
            if 'center_sim' not in fields and 'center' in fields:
                fields['center_sim'] = fields['center']
            if 'target_sim' not in fields and 'target' in fields:
                fields['target_sim'] = fields['target']
            if joint_name not in all_trials:
                all_trials[joint_name] = []
            all_trials[joint_name].append(fields)

    return all_trials


def resample_trial_to_sim(trial, sim_dt):
    """Resample a real trial to sim timesteps. Returns (positions_resampled, num_sim_steps)."""
    ts = trial['timestamps']
    pos = trial['positions']
    onset = float(trial['step_onset'])

    # Only use post-step data
    mask = ts >= onset
    if mask.sum() < 5:
        return None, 0

    ts_post = ts[mask] - onset  # time from step onset
    pos_post = pos[mask]

    # Positions should be in sim convention (from sysid_collect.py)

    duration = ts_post[-1]
    n_steps = int(duration / sim_dt)
    if n_steps < 5:
        return None, 0

    sim_times = np.arange(n_steps) * sim_dt
    pos_resampled = np.interp(sim_times, ts_post, pos_post)
    return pos_resampled, n_steps


# =========================================================================
# Evaluation
# =========================================================================

def evaluate_params(robot, env, sim_dt, joint_ids, trials_resampled, param_matrix, device):
    """Run all trials for all envs with given parameters. Returns cost per env.

    Args:
        robot: Articulation object
        env: unwrapped env
        sim_dt: control dt (physics_dt * decimation)
        joint_ids: list of PhysX joint indices to evaluate
        trials_resampled: list of (target_sim, positions_resampled, n_steps, center_sim)
        param_matrix: (num_envs, 3) tensor — [stiffness, damping, friction] per env
        device: torch device
    """
    num_envs = param_matrix.shape[0]
    num_joints = robot.num_joints

    # Write actuator params for the target joints
    stiffness_full = robot.data.joint_stiffness.clone()
    damping_full = robot.data.joint_damping.clone()

    for i, jid in enumerate(joint_ids):
        stiffness_full[:, jid] = param_matrix[:, 0]
        damping_full[:, jid] = param_matrix[:, 1]

    robot.write_joint_stiffness_to_sim(stiffness_full)
    robot.write_joint_damping_to_sim(damping_full)

    # Friction: write per joint
    friction_full = torch.zeros(num_envs, num_joints, device=device)
    for jid in joint_ids:
        friction_full[:, jid] = param_matrix[:, 2]
    robot.write_joint_friction_coefficient_to_sim(friction_full)

    total_cost = torch.zeros(num_envs, device=device)

    for trial_idx, (target_sim, pos_real, n_steps, center_sim) in enumerate(trials_resampled):
        # Reset all envs to the trial's starting position
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = torch.zeros_like(joint_pos)

        # Set the test joints to the trial's center position
        for jid in joint_ids:
            joint_pos[:, jid] = center_sim

        # Direct state write + settle
        # Re-write state every step to fight env managers
        for _ in range(50):
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.set_joint_position_target(joint_pos)
            robot.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(env.physics_dt)

        # Final state write
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        robot.set_joint_position_target(joint_pos)
        robot.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(env.physics_dt)

        # Command the step target
        target_pos = joint_pos.clone()
        for jid in joint_ids:
            target_pos[:, jid] = target_sim

        # Record trajectory
        sim_positions = torch.zeros(n_steps, num_envs, device=device)

        for step in range(n_steps):
            robot.set_joint_position_target(target_pos)
            robot.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(env.physics_dt)

            # Record position of first test joint (they should all behave similarly within a group)
            sim_positions[step] = robot.data.joint_pos[:, joint_ids[0]]

        # Compare with real data
        pos_real_tensor = torch.tensor(pos_real[:n_steps], device=device, dtype=torch.float32)
        # pos_real is shape (n_steps,), sim_positions is (n_steps, num_envs)
        error = (sim_positions - pos_real_tensor.unsqueeze(1)) ** 2
        total_cost += error.sum(dim=0)

    return total_cost


# =========================================================================
# Plotting
# =========================================================================

def plot_joint_comparison(joint_name, joint_ids, robot, env, sim_dt, trials_resampled,
                          params, output_dir, device, label="fitted"):
    """Plot real vs sim trajectories for one joint with given params."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    num_joints = robot.num_joints
    stiffness, damping, friction = params

    # Write params to env 0 only
    stiffness_full = robot.data.joint_stiffness.clone()
    damping_full = robot.data.joint_damping.clone()
    friction_full = torch.zeros(1, num_joints, device=device)

    for jid in joint_ids:
        stiffness_full[0, jid] = stiffness
        damping_full[0, jid] = damping
        friction_full[0, jid] = friction

    robot.write_joint_stiffness_to_sim(stiffness_full[:1], env_ids=torch.tensor([0], device=device))
    robot.write_joint_damping_to_sim(damping_full[:1], env_ids=torch.tensor([0], device=device))
    robot.write_joint_friction_coefficient_to_sim(friction_full, env_ids=torch.tensor([0], device=device))

    n_trials = len(trials_resampled)
    fig, axes = plt.subplots(1, n_trials, figsize=(4 * n_trials, 3.5), squeeze=False)
    fig.suptitle(f'{joint_name} [{label}]  K={stiffness:.3f}  D={damping:.4f}  F={friction:.4f}', fontsize=11)

    for t_idx, (target_sim, pos_real, n_steps, center_sim) in enumerate(trials_resampled):
        ax = axes[0, t_idx]

        # Reset env 0
        env0 = torch.tensor([0], device=device)
        joint_pos = robot.data.default_joint_pos[:1].clone()
        joint_vel = torch.zeros_like(joint_pos)
        for jid in joint_ids:
            joint_pos[0, jid] = center_sim

        # Direct state write + settle
        # Re-write state every step to fight env managers that may override
        for _ in range(50):
            robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env0)
            robot.set_joint_position_target(joint_pos, env_ids=env0)
            robot.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(env.physics_dt)

        # Final state write to ensure we're exactly at center
        robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env0)
        robot.set_joint_position_target(joint_pos, env_ids=env0)
        robot.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(env.physics_dt)

        target_pos = joint_pos.clone()
        for jid in joint_ids:
            target_pos[0, jid] = target_sim

        sim_traj = []
        for step in range(n_steps):
            robot.set_joint_position_target(target_pos, env_ids=env0)
            robot.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(env.physics_dt)
            sim_traj.append(robot.data.joint_pos[0, joint_ids[0]].item())

        sim_traj = np.array(sim_traj)
        t_ms = np.arange(n_steps) * sim_dt * 1000

        amp_deg = np.degrees(target_sim - center_sim)
        ax.plot(t_ms, np.degrees(pos_real[:n_steps]), 'b-', linewidth=1.2, label='Real')
        ax.plot(t_ms, np.degrees(sim_traj), 'r--', linewidth=1.2, label='Sim')
        ax.axhline(np.degrees(target_sim), color='gray', linestyle=':', alpha=0.4)
        ax.set_title(f'{amp_deg:+.0f}°', fontsize=9)
        ax.set_xlabel('ms', fontsize=8)
        if t_idx == 0:
            ax.set_ylabel('deg', fontsize=8)
            ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f'{joint_name}_{label}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    return path


# =========================================================================
# Main
# =========================================================================

@hydra_task_config(args_cli.task, "rl_games_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    # Override num_envs (use 1 for plot-only mode, otherwise from config or CLI)
    if args_cli.plot_only:
        env_cfg.scene.num_envs = 1
    elif args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    # else: use whatever sysid.yaml set (128)
    env_cfg.sim.device = args_cli.device if args_cli.device else "cuda:0"

    # Expand data globs
    npz_paths = []
    for pattern in args_cli.data:
        npz_paths.extend(sorted(glob_mod.glob(pattern)))
    if not npz_paths:
        print("ERROR: No .npz files found")
        return

    print(f"Loading real data from {len(npz_paths)} file(s)...")
    all_trials = load_real_data(npz_paths)
    print(f"Joints: {list(all_trials.keys())}")

    # Create environment
    print("Creating environment...")
    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    device = unwrapped.device

    physics_dt = unwrapped.physics_dt
    decimation = unwrapped.cfg.decimation if hasattr(unwrapped.cfg, 'decimation') else 2
    sim_dt = physics_dt * decimation
    print(f"Sim: physics_dt={physics_dt}, decimation={decimation}, control_dt={sim_dt}")

    # PhysX joint names
    physx_names = robot.data.joint_names
    print(f"Robot joints: {len(physx_names)}")

    # Output dir
    output_dir = args_cli.output or os.path.dirname(npz_paths[0])
    plot_dir = os.path.join(output_dir, "plots")

    # Reset env once to initialize
    obs, info = env.reset()

    # Determine which groups to fit
    groups_to_fit = {}
    if args_cli.arm:
        groups_to_fit.update(ARM_GROUPS)
    if args_cli.hand:
        groups_to_fit.update(HAND_GROUPS)
    if not groups_to_fit:
        # Default: fit everything
        groups_to_fit.update(HAND_GROUPS)
        groups_to_fit.update(ARM_GROUPS)

    fitted_params = {}

    for group_name, joint_names in groups_to_fit.items():
        # Filter to joints that have real data
        available = [j for j in joint_names if j in all_trials]
        if not available:
            print(f"\n[{group_name}] No real data — skipping")
            continue

        # Find PhysX joint IDs
        joint_ids_found, _ = robot.find_joints(available, preserve_order=True)
        joint_ids = list(joint_ids_found)
        if not joint_ids:
            print(f"\n[{group_name}] Joints not found in robot — skipping")
            continue

        # Prepare resampled trials (use first joint's data as representative for the group)
        rep_joint = available[0]
        trials_resampled = []
        for trial in all_trials[rep_joint]:
            pos_resampled, n_steps = resample_trial_to_sim(trial, sim_dt)
            if pos_resampled is not None:
                target_sim = float(trial['target_sim'])
                center_sim = float(trial['center_sim'])
                trials_resampled.append((target_sim, pos_resampled, n_steps, center_sim))

        if not trials_resampled:
            print(f"\n[{group_name}] No valid trials after resampling — skipping")
            continue

        print(f"\n{'='*60}")
        print(f"[{group_name}] Joints: {available}, {len(trials_resampled)} trials")
        print(f"  PhysX IDs: {joint_ids}")

        # Get current (default) sim params for this joint
        default_stiffness = robot.data.joint_stiffness[0, joint_ids[0]].item()
        default_damping = robot.data.joint_damping[0, joint_ids[0]].item()
        # Friction isn't directly readable from data, use 0.01 as default
        default_friction = 0.01

        print(f"  Default: K={default_stiffness:.3f}, D={default_damping:.4f}, F={default_friction:.4f}")

        # Plot with default params first
        print(f"  Plotting default params...")
        plot_joint_comparison(
            group_name, joint_ids, robot, unwrapped, sim_dt, trials_resampled,
            (default_stiffness, default_damping, default_friction),
            plot_dir, device, label="default"
        )

        if args_cli.plot_only:
            print(f"  [plot-only mode — skipping optimization]")
            continue

        # CMA-ES optimization
        num_envs = env_cfg.scene.num_envs
        print(f"  Optimizing with CMA-ES ({num_envs} parallel envs, maxiter={args_cli.maxiter})...")

        # Initial guess = current sim params (log scale for stiffness/damping)
        x0 = [default_stiffness, default_damping, default_friction]

        def cma_cost(params_list):
            """Evaluate a batch of parameter vectors. CMA-ES calls this with a list of candidates."""
            n_candidates = len(params_list)

            # Pad or chunk to fit num_envs
            if n_candidates > num_envs:
                # Shouldn't happen with default CMA-ES popsize, but handle it
                costs = []
                for i in range(0, n_candidates, num_envs):
                    chunk = params_list[i:i+num_envs]
                    c = _eval_chunk(chunk)
                    costs.extend(c)
                return costs[:n_candidates]
            else:
                return _eval_chunk(params_list)

        def _eval_chunk(params_list):
            n = len(params_list)
            # Build param matrix (n, 3), pad to num_envs
            pm = torch.zeros(num_envs, 3, device=device)
            for i, p in enumerate(params_list):
                if is_arm:
                    # Only friction varies; K,D fixed
                    pm[i, 0] = default_stiffness
                    pm[i, 1] = default_damping
                    pm[i, 2] = max(p[0], 0.0)
                else:
                    pm[i, 0] = max(p[0], 0.01)  # stiffness > 0
                    pm[i, 1] = max(p[1], 0.001)  # damping > 0
                    pm[i, 2] = max(p[2], 0.0)  # friction >= 0
            # Fill unused envs with default
            for i in range(n, num_envs):
                pm[i, 0] = default_stiffness
                pm[i, 1] = default_damping
                pm[i, 2] = default_friction

            costs = evaluate_params(
                robot, unwrapped, sim_dt, joint_ids, trials_resampled, pm, device
            )
            return [costs[i].item() for i in range(n)]

        # Optimize — arm: 1D grid search for friction. Hand: CMA-ES for K,D,F.
        is_arm = group_name.startswith('ur5e_')

        if is_arm:
            # 1D optimization: sweep friction values across all envs in one shot
            friction_values = torch.linspace(0.0, 20.0, num_envs, device=device)
            pm = torch.zeros(num_envs, 3, device=device)
            pm[:, 0] = default_stiffness
            pm[:, 1] = default_damping
            pm[:, 2] = friction_values

            print(f"  Grid search: {num_envs} friction values in [0, 20] Nm...")
            costs = evaluate_params(
                robot, unwrapped, sim_dt, joint_ids, trials_resampled, pm, device
            )
            best_idx = costs.argmin().item()
            best_cost = costs[best_idx].item()
            friction = friction_values[best_idx].item()
            stiffness, damping = default_stiffness, default_damping
            print(f"  Best: F={friction:.4f} (cost={best_cost:.4f})")

        else:
            # CMA-ES for hand joints (3 params)
            bounds = [[0.01, 0.001, 0.0], [50.0, 10.0, 2.0]]
            sigma0 = max(default_stiffness * 0.3, 0.1)
            opts = {
                'maxiter': args_cli.maxiter,
                'popsize': num_envs,
                'bounds': bounds,
                'verbose': 1,
                'seed': 42,
            }
            es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

            while not es.stop():
                candidates = es.ask()
                costs = cma_cost(candidates)
                es.tell(candidates, costs)
                bp = es.result.xbest
                print(f"    Gen {es.countiter}: best_cost={es.result.fbest:.4f} "
                      f"K={bp[0]:.4f} D={bp[1]:.4f} F={bp[2]:.4f}")

            best_params = es.result.xbest
            best_cost = es.result.fbest
            stiffness, damping, friction = best_params[0], best_params[1], max(best_params[2], 0.0)

        print(f"\n  FITTED: K={stiffness:.4f}, D={damping:.4f}, F={friction:.4f} (cost={best_cost:.4f})")

        fitted_params[group_name] = {
            'stiffness': round(float(stiffness), 4),
            'damping': round(float(damping), 4),
            'friction': round(float(friction), 4),
            'cost': round(float(best_cost), 6),
            'joints': available,
        }

        # Plot with fitted params
        print(f"  Plotting fitted params...")
        # Need to use 1 env for plotting — resize param write to env 0
        plot_joint_comparison(
            group_name, joint_ids, robot, unwrapped, sim_dt, trials_resampled,
            (stiffness, damping, friction),
            plot_dir, device, label="fitted"
        )

    # Save results
    if fitted_params:
        out_path = os.path.join(output_dir, 'fitted_params_isaac.yaml')
        with open(out_path, 'w') as f:
            yaml.dump(fitted_params, f, default_flow_style=False, sort_keys=False)
        print(f"\nFitted params saved to: {out_path}")

        print(f"\n{'='*60}")
        print("UPDATE FOR ur5e_leap.py")
        print(f"{'='*60}")
        for gname, p in fitted_params.items():
            joints_str = ', '.join(p['joints'])
            print(f"\n# {gname} ({joints_str}):")
            print(f"#   stiffness: {p['stiffness']}")
            print(f"#   damping:   {p['damping']}")
            print(f"#   friction:  {p['friction']}")

    env.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
    simulation_app.close()
