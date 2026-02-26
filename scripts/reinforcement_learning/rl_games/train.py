# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
from distutils.util import strtobool

# Apply GPU offset for multi-GPU on specific devices (e.g. --multi_gpu 2,3)
# This shifts LOCAL_RANK so torchrun process 0 maps to physical GPU 2, etc.
_gpu_offset = int(os.environ.get("Y2R_GPU_OFFSET", "0"))
if _gpu_offset and "LOCAL_RANK" in os.environ:
    os.environ["LOCAL_RANK"] = str(int(os.environ["LOCAL_RANK"]) + _gpu_offset)

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rl_games_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--sigma", type=str, default=None, help="The policy's initial standard deviation.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
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
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
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
import subprocess
import torch
from datetime import datetime
from pathlib import Path

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

from isaaclab_rl.rl_games import MultiObserver, PbtAlgoObserver, RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_tasks.manager_based.manipulation.y2r.rl_games_checkpoint import Y2RCheckpointObserver

# Register custom networks for y2r tasks (must be before Runner.load)
import isaaclab_tasks.manager_based.manipulation.y2r.networks  # noqa: F401

# import logger
logger = logging.getLogger(__name__)

# PLACEHOLDER: Extension template (do not remove this comment)


def get_git_info():
    """Get git repository metadata for reproducibility."""
    repo_root = Path(__file__).parent.parent.parent
    commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'],
        cwd=repo_root
    ).decode('ascii').strip()
    dirty_files = subprocess.check_output(
        ['git', 'status', '--porcelain'],
        cwd=repo_root
    ).decode('ascii').strip()
    branch = subprocess.check_output(
        ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
        cwd=repo_root
    ).decode('ascii').strip()
    return {
        "git_commit": commit,
        "git_dirty": bool(dirty_files),
        "git_branch": branch,
    }


def log_y2r_config_files():
    """Log Y2R YAML config files to wandb for full reproducibility.

    Uses get_config_file_paths() to ensure logged files match actual config composition.
    Reads Y2R_MODE, Y2R_TASK, Y2R_ROBOT from environment variables.
    """
    import wandb
    from isaaclab_tasks.manager_based.manipulation.y2r.config_loader import get_config_file_paths

    # Get list of files that will actually be loaded (single source of truth)
    config_files = get_config_file_paths()

    # Base path for relative paths in wandb
    isaaclab_tasks_root = Path(__file__).parent.parent.parent / "source" / "isaaclab_tasks"

    # Upload all config files that are part of this run
    for yaml_path in config_files:
        # Convert to Path and ensure it's relative to base
        yaml_path = Path(yaml_path)
        if yaml_path.is_relative_to(isaaclab_tasks_root):
            wandb.save(str(yaml_path), base_path=str(isaaclab_tasks_root), policy="now")
        else:
            # File outside base path, upload without base path (will use absolute path)
            wandb.save(str(yaml_path), policy="now")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with RL-Games agent."""
    y2r_mode = os.environ["Y2R_MODE"]
    y2r_task = os.environ["Y2R_TASK"]

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # update agent device to match simulation device
    if args_cli.device is not None:
        agent_cfg["params"]["config"]["device"] = args_cli.device
        agent_cfg["params"]["config"]["device_name"] = args_cli.device

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    agent_cfg["params"]["config"]["max_epochs"] = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg["params"]["config"]["max_epochs"]
    )
    if args_cli.checkpoint is not None:
        resume_path = retrieve_file_path(args_cli.checkpoint)
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = resume_path
        print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")
    train_sigma = float(args_cli.sigma) if args_cli.sigma is not None else None

    # multi-gpu training config
    if args_cli.distributed:
        agent_cfg["params"]["seed"] += app_launcher.global_rank
        agent_cfg["params"]["config"]["device"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["device_name"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["multi_gpu"] = True
        # update env config device
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"

    # set the environment seed (after multi-gpu config for updated rank from agent seed)
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["params"]["seed"]

    # specify directory for logging experiments
    # Y2R_DATA_ROOT controls where all large outputs go; defaults to repo root
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    _data_root = os.environ.get("Y2R_DATA_ROOT", _repo_root)
    config_name = agent_cfg["params"]["config"]["name"]
    log_root_path = os.path.join(_data_root, "IsaacLab", "logs", "rl_games", config_name)
    if "pbt" in agent_cfg and agent_cfg["pbt"]["directory"] != ".":
        log_root_path = os.path.join(agent_cfg["pbt"]["directory"], log_root_path)

    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs
    log_dir = agent_cfg["params"]["config"].get("full_experiment_name", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    # set directory into agent config
    # logging directory path: <train_dir>/<full_experiment_name>
    agent_cfg["params"]["config"]["train_dir"] = log_root_path
    agent_cfg["params"]["config"]["full_experiment_name"] = log_dir
    wandb_project = config_name if args_cli.wandb_project_name is None else args_cli.wandb_project_name
    experiment_name = log_dir if args_cli.wandb_name is None else args_cli.wandb_name

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "agent.yaml"), agent_cfg)
    print(f"Exact experiment name requested from command line: {os.path.join(log_root_path, log_dir)}")

    # read configurations about the agent-training
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)
    half_precision_obs = agent_cfg["params"]["config"].get("mixed_precision", False)
    if agent_cfg["params"]["config"].get("expl_type", "none").startswith("mixed_expl"):
        # Keep rollout observations in FP32 for mixed_expl to preserve coefficient-id fidelity.
        half_precision_obs = False

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups, half_precision_obs)

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    # create runner from rl-games

    if "pbt" in agent_cfg and agent_cfg["pbt"]["enabled"]:
        runner = Runner(MultiObserver([IsaacAlgoObserver(), PbtAlgoObserver(agent_cfg, args_cli), Y2RCheckpointObserver()]))
    else:
        runner = Runner(MultiObserver([IsaacAlgoObserver(), Y2RCheckpointObserver()]))

    runner.load(agent_cfg)

    # monkey-patch: fused Adam + FP32 upcast before RunningMeanStd
    _orig_create = runner.algo_factory.create
    def _create_with_patches(*args, **kwargs):
        agent = _orig_create(*args, **kwargs)
        # fused optimizer (single CUDA kernel per param update)
        if hasattr(agent, 'optimizer'):
            agent.optimizer = torch.optim.Adam(agent.optimizer.param_groups, foreach=True)
        # upcast FP16 obs to FP32 before RunningMeanStd for stable normalization statistics
        _orig_preproc = agent._preproc_obs
        def _preproc_obs_f32(obs_batch):
            obs_batch = _orig_preproc(obs_batch)
            if isinstance(obs_batch, torch.Tensor) and obs_batch.dtype == torch.float16:
                obs_batch = obs_batch.float()
            elif isinstance(obs_batch, dict):
                obs_batch = {k: v.float() if v.dtype == torch.float16 else v for k, v in obs_batch.items()}
            return obs_batch
        agent._preproc_obs = _preproc_obs_f32
        return agent
    runner.algo_factory.create = _create_with_patches

    # reset the agent and env
    runner.reset()
    # train the agent

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
            monitor_gym=True,
            save_code=True,
        )
        if not wandb.run.resumed:
            # Log source YAML files (reads Y2R_MODE/Y2R_TASK/Y2R_ROBOT from env)
            log_y2r_config_files()

            # Structured config logging - maximizes signal, minimizes noise
            wandb.config.update({
                "run_metadata": {
                    "y2r_mode": y2r_mode,
                    "y2r_task": y2r_task,
                    "config_layers": [
                        "base.yaml",
                        *([] if y2r_mode == "train" else [f"layers/{y2r_mode}.yaml"]),
                        *([] if y2r_task == "base" else [f"layers/tasks/{y2r_task}.yaml"])
                    ],
                    "num_envs": env_cfg.scene.num_envs,
                    "device": str(env_cfg.sim.device),
                    "seed": agent_cfg["params"]["seed"],
                    "command_line": " ".join(sys.argv),
                    **get_git_info(),
                },
                "simulation": {
                    "physics_dt": env_cfg.sim.dt,
                    "decimation": getattr(env_cfg.sim, 'render_interval', 2),
                    "replicate_physics": getattr(env_cfg.sim, 'replicate_physics', False),
                },
                "training": {
                    "max_epochs": agent_cfg["params"]["config"]["max_epochs"],
                    "learning_rate": agent_cfg["params"]["config"].get("learning_rate"),
                    "batch_size": agent_cfg["params"]["config"].get("minibatch_size"),
                    "horizon_length": agent_cfg["params"]["config"].get("horizon_length"),
                    "gamma": agent_cfg["params"]["config"].get("gamma"),
                    "tau": agent_cfg["params"]["config"].get("tau"),
                },
                # Full configs as reference (secondary to structured data above)
                "env_cfg_full": env_cfg.to_dict(),
                "agent_cfg_full": agent_cfg,
            }, allow_val_change=True)

    if args_cli.checkpoint is not None:
        runner.run({"train": True, "play": False, "sigma": train_sigma, "checkpoint": resume_path})
    else:
        runner.run({"train": True, "play": False, "sigma": train_sigma})

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
