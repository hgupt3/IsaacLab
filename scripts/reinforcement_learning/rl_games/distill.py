# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# Copyright (c) 2024, Harsh Gupta
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run teacher-student distillation with RL-Games infrastructure."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from distutils.util import strtobool

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Distill a teacher policy into a student policy.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to student model checkpoint to resume from.")
parser.add_argument(
    "--teacher-checkpoint", type=str, required=True,
    help="Path to teacher policy checkpoint (required)."
)
parser.add_argument(
    "--teacher-agent", type=str, default="rl_games_teacher_cfg_entry_point",
    help="Teacher agent config entry point (default: rl_games_cfg_entry_point)."
)
parser.add_argument("--max_iterations", type=int, default=None, help="Distillation training iterations.")
parser.add_argument("--wandb-project-name", type=str, default=None, help="the wandb's project name")
parser.add_argument("--wandb-entity", type=str, default=None, help="the entity (team) of wandb's project")
parser.add_argument("--wandb-name", type=str, default=None, help="the name of wandb's run")
parser.add_argument(
    "--track",
    type=lambda x: bool(strtobool(x)),
    default=False,
    nargs="?",
    const=True,
    help="if toggled, this experiment will be tracked with Weights and Biases",
)
# Distillation-specific arguments
parser.add_argument("--beta", type=float, default=0.5, help="Fraction of envs using teacher actions (0-1).")
parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate for distillation.")
parser.add_argument("--grad-norm", type=float, default=1.0, help="Gradient clipping norm.")
parser.add_argument("--save-frequency", type=int, default=5000, help="Checkpoint save frequency (iterations).")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import logging
import math
import os
import random
import torch
from datetime import datetime

from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_tasks.utils import load_cfg_from_registry

# Register custom networks for y2r tasks
import isaaclab_tasks.manager_based.manipulation.y2r.networks  # noqa: F401

# Import and register distillation agent
from isaaclab_tasks.manager_based.manipulation.y2r.distillation import DistillAgent

# import logger
logger = logging.getLogger(__name__)


class RlGamesVecEnvWrapperWithCamera(RlGamesVecEnvWrapper):
    """Extended wrapper that passes additional observation groups (like camera)."""
    
    def _process_obs(self, obs_dict):
        """Process observations including additional groups beyond 'obs' and 'states'."""
        result = super()._process_obs(obs_dict)
        
        # Pass through any additional groups defined in obs_groups
        for group_key, group_names in self._obs_groups.items():
            if group_key not in ['obs', 'states'] and len(group_names) > 0:
                # Concatenate all obs in this group
                group_obs = [obs_dict[name] for name in group_names if name in obs_dict]
                if group_obs:
                    result[group_key] = torch.cat(group_obs, dim=-1) if len(group_obs) > 1 else group_obs[0]
        
        return result


def register_distill_agent_with_runner(runner: Runner):
    """Register DistillAgent with the RL-Games runner."""
    runner.algo_factory.register_builder(
        'distill_a2c_continuous',
        lambda **kwargs: DistillAgent(**kwargs)
    )


@hydra_task_config(args_cli.task, "rl_games_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Run teacher-student distillation."""
    
    # Override configurations with CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    
    # Check for invalid CPU + distributed combo
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported with CPU device. "
            "Use GPU device (e.g., --device cuda) for distributed training."
        )
    
    # Update agent device
    if args_cli.device is not None:
        agent_cfg["params"]["config"]["device"] = args_cli.device
        agent_cfg["params"]["config"]["device_name"] = args_cli.device
    
    # Handle seed
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    
    # Multi-GPU config
    if args_cli.distributed:
        agent_cfg["params"]["seed"] += app_launcher.global_rank
        agent_cfg["params"]["config"]["device"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["device_name"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["multi_gpu"] = True
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    
    env_cfg.seed = agent_cfg["params"]["seed"]
    
    # ===== Load teacher config =====
    # Use registry to find teacher config (allows overriding via --teacher-agent)
    teacher_cfg = load_cfg_from_registry(args_cli.task, args_cli.teacher_agent)
    
    # ===== Inject distillation config into agent_cfg =====
    teacher_ckpt = retrieve_file_path(args_cli.teacher_checkpoint)
    
    # Start with YAML distillation config (if exists), then override with CLI args
    yaml_distill_cfg = agent_cfg["params"].get("distillation", {})
    
    # Merge: YAML values as base, CLI args override
    agent_cfg["params"]["distillation"] = {
        **yaml_distill_cfg,  # YAML values first (save_best_after, depth_aug, etc.)
        "teacher_cfg": teacher_cfg,  # Full teacher config dict
        "teacher_ckpt": teacher_ckpt,
        "beta": args_cli.beta,
        "learning_rate": args_cli.learning_rate,
        "grad_norm": args_cli.grad_norm,
        "max_iterations": args_cli.max_iterations if args_cli.max_iterations else yaml_distill_cfg.get("max_iterations", 1_000_000),
        "save_frequency": args_cli.save_frequency if args_cli.save_frequency != 5000 else yaml_distill_cfg.get("save_frequency", 5000),
    }
    
    # Override the algo name to use our distillation agent
    agent_cfg["params"]["algo"]["name"] = "distill_a2c_continuous"
    
    # ===== Setup logging directory =====
    config_name = agent_cfg["params"]["config"].get("name", "distillation")
    log_root_path = os.path.join("logs", "rl_games", config_name)
    log_root_path = os.path.abspath(log_root_path)
    
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    
    log_dir = agent_cfg["params"]["config"].get(
        "full_experiment_name", 
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    agent_cfg["params"]["config"]["train_dir"] = log_root_path
    agent_cfg["params"]["config"]["full_experiment_name"] = log_dir
    
    wandb_project = args_cli.wandb_project_name or config_name
    experiment_name = args_cli.wandb_name or log_dir
    
    # Dump configs
    os.makedirs(os.path.join(log_root_path, log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "agent.yaml"), agent_cfg)
    
    print(f"[INFO] Experiment directory: {os.path.join(log_root_path, log_dir)}")
    print(f"[INFO] Teacher checkpoint: {teacher_ckpt}")
    print(f"[INFO] Beta (teacher env ratio): {args_cli.beta}")
    
    # ===== Read agent configurations =====
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)
    
    # Set log directory for environment
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.log_dir = os.path.join(log_root_path, log_dir)
    
    # ===== Create environment =====
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    
    # Video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    
    # Wrap for RL-Games (use extended wrapper to pass camera obs)
    env = RlGamesVecEnvWrapperWithCamera(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)
    
    # Register environment
    vecenv.register(
        "IsaacRlgWrapper", 
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})
    
    # Set num actors
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    
    # ===== Create Runner and register distill agent =====
    runner = Runner(IsaacAlgoObserver())
    register_distill_agent_with_runner(runner)
    
    runner.load(agent_cfg)
    runner.reset()
    
    # ===== WandB tracking =====
    global_rank = int(os.getenv("RANK", "0"))
    if args_cli.track and global_rank == 0:
        if args_cli.wandb_entity is None:
            raise ValueError("Weights and Biases entity must be specified for tracking.")
        import wandb
        
        wandb.init(
            project=wandb_project,
            entity=args_cli.wandb_entity,
            name=experiment_name,
            sync_tensorboard=True,
            config={
                "beta": args_cli.beta,
                "learning_rate": args_cli.learning_rate,
                "grad_norm": args_cli.grad_norm,
                "teacher_checkpoint": args_cli.teacher_checkpoint,
            }
        )
        if not wandb.run.resumed:
            wandb.config.update({"env_cfg": env_cfg.to_dict()})
            wandb.config.update({"agent_cfg": agent_cfg})
    
    # ===== Run distillation =====
    print("\n" + "="*60)
    print("Starting Distillation")
    print("="*60 + "\n")
    
    runner.run({"train": True, "play": False})
    
    # Close
    env.close()
    
    if args_cli.track and global_rank == 0:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
    simulation_app.close()
