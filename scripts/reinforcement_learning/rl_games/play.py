# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rl_games_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--record", type=str, default=None, help="Record obs/actions/joint_pos to npz file for 1 episode.")
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
import math
import os
import random
import time
import torch

import carb
import numpy as np
import omni

from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
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
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Play with RL-Games agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # update agent device to match simulation device
    if args_cli.device is not None:
        agent_cfg["params"]["config"]["device"] = args_cli.device
        agent_cfg["params"]["config"]["device_name"] = args_cli.device

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    # set the environment seed (after multi-gpu config for updated rank from agent seed)
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["params"]["seed"]

    # specify directory for logging experiments
    # Y2R_DATA_ROOT controls where all large outputs go; defaults to repo root
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    _data_root = os.environ.get("Y2R_DATA_ROOT", _repo_root)
    log_root_path = os.path.join(_data_root, "IsaacLab", "logs", "rl_games", agent_cfg["params"]["config"]["name"])
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # find checkpoint
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rl_games", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint is None:
        # specify directory for logging runs
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        # specify name of checkpoint
        if args_cli.use_last_checkpoint:
            checkpoint_file = ".*"
        else:
            # this loads the best checkpoint
            checkpoint_file = f"{agent_cfg['params']['config']['name']}.pth"
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # wrap around environment for rl-games
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)
    half_precision_obs = agent_cfg["params"]["config"].get("mixed_precision", False)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
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

    # load previously trained model
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    # create runner from rl-games
    runner = Runner()
    runner.load(agent_cfg)
    # obtain the agent from the runner
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()
    # Upcast FP16 observations to FP32 before model preprocessing.
    # Keeps inference numerics and student network assumptions consistent.
    _orig_preproc = agent._preproc_obs
    def _preproc_obs_f32(obs_batch):
        obs_batch = _orig_preproc(obs_batch)
        if isinstance(obs_batch, torch.Tensor) and obs_batch.dtype == torch.float16:
            obs_batch = obs_batch.float()
        elif isinstance(obs_batch, dict):
            obs_batch = {k: v.float() if isinstance(v, torch.Tensor) and v.dtype == torch.float16 else v for k, v in obs_batch.items()}
        return obs_batch
    agent._preproc_obs = _preproc_obs_f32

    # Explicit runtime switches for inference: disable train-time stochasticity/stat updates.
    a2c_network = getattr(getattr(agent, "model", None), "a2c_network", None)
    if a2c_network is not None:
        if hasattr(a2c_network, "apply_depth_aug"):
            a2c_network.apply_depth_aug = False
            print("[INFO] Disabled student depth augmentation for play.")
        if hasattr(a2c_network, "apply_obs_rms_update"):
            a2c_network.apply_obs_rms_update = False
            print("[INFO] Disabled student obs RMS updates for play.")
        rms = getattr(a2c_network, "running_mean_std", None)
        if rms is not None:
            rms.eval()

    dt = env.unwrapped.step_dt

    # ── Recording setup ──────────────────────────────────────────────────
    recording = args_cli.record is not None
    if recording:
        recorded_obs = []
        recorded_actions = []
        recorded_joint_pos = []

        # Extract metadata from env for the npz
        unwrapped = env.unwrapped
        robot = unwrapped.scene["robot"]
        y2r_cfg = env_cfg.y2r_cfg

        # Action term metadata
        action_term = unwrapped.action_manager._terms["action"]
        rec_pca_matrix = action_term._pca_matrix.cpu().numpy()
        rec_action_scale = float(action_term._scale) if isinstance(action_term._scale, float) else action_term._scale.cpu().numpy()
        rec_arm_dim = action_term._arm_dim
        rec_eigen_dim = action_term._eigen_dim
        rec_hand_dim = action_term._hand_dim

        # Default joint positions
        rec_default_joint_pos = robot.data.default_joint_pos[0].cpu().numpy()
        rec_joint_names = list(robot.data.joint_names)

        # Obs group info from the wrapper
        obs_group_list = obs_groups["obs"] if obs_groups else ["policy"]
        obs_space = unwrapped.single_observation_space
        obs_group_names = []
        obs_group_sizes = []
        for grp in obs_group_list:
            obs_group_names.append(grp)
            obs_group_sizes.append(obs_space[grp].shape[0])

        # Control Hz
        rec_control_hz = 1.0 / dt

        print(f"[INFO] Recording enabled → {args_cli.record}")
        print(f"[INFO]   obs groups: {list(zip(obs_group_names, obs_group_sizes))}")
        print(f"[INFO]   action_dim: {rec_arm_dim + rec_eigen_dim + rec_hand_dim}, action_scale: {rec_action_scale}")
        print(f"[INFO]   control_hz: {rec_control_hz:.1f}")

    # reset environment (use enable_grad to allow tensor modifications)
    with torch.enable_grad():
        obs = env.reset()
        if isinstance(obs, dict):
            obs = obs["obs"]
    timestep = 0
    # required: enables the flag for batched observations
    _ = agent.get_batch_size(obs, 1)
    # initialize RNN states if used
    if agent.is_rnn:
        agent.init_rnn()

    # Setup keyboard event listener for manual reset
    reset_requested = {"flag": False}

    def on_keyboard_event(event, *args):
        """Handle keyboard events for manual reset."""
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "KEY_1" or event.input.name == "NUMPAD_1":
                reset_requested["flag"] = True
                print("[INFO] Manual reset requested (Key: 1)")
        return True

    # Subscribe to keyboard events
    appwindow = omni.appwindow.get_default_app_window()
    kb_input = carb.input.acquire_input_interface()
    keyboard = appwindow.get_keyboard()
    kb_sub = kb_input.subscribe_to_keyboard_events(keyboard, on_keyboard_event)

    print("[INFO] Keyboard controls enabled: Press '1' to reset environments")

    # simulate environment
    # note: We simplified the logic in rl-games player.py (:func:`BasePlayer.run()`) function in an
    #   attempt to have complete control over environment stepping. However, this removes other
    #   operations such as masking that is used for multi-agent learning by RL-Games.
    while simulation_app.is_running():
        start_time = time.time()

        # Check if manual reset was requested via keyboard
        if reset_requested["flag"]:
            # Force reset by setting episode length to max for all environments
            env.unwrapped.episode_length_buf[:] = env.unwrapped.max_episode_length
            reset_requested["flag"] = False
            print("[INFO] Manual reset triggered")

        # run everything in inference mode
        with torch.inference_mode():
            # convert obs to agent format
            obs = agent.obs_to_torch(obs)
            # agent stepping
            actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)

            # Record BEFORE env.step — captures pre-reset state
            if recording:
                # obs["obs"] is the concatenated observation vector after obs_to_torch
                obs_tensor = obs["obs"] if isinstance(obs, dict) else obs
                recorded_obs.append(obs_tensor[0].cpu().numpy().copy())
                recorded_actions.append(actions[0].cpu().numpy().copy())
                recorded_joint_pos.append(
                    env.unwrapped.scene["robot"].data.joint_pos[0].cpu().numpy().copy()
                )

            # env stepping
            obs, _, dones, _ = env.step(actions)

            # Check for episode termination during recording
            if recording and dones.numel() > 0 and dones[0]:
                print(f"[INFO] Episode done after {len(recorded_actions)} steps. Saving recording...")
                _save_recording(
                    args_cli.record,
                    recorded_obs, recorded_actions, recorded_joint_pos,
                    rec_pca_matrix, rec_action_scale, rec_arm_dim, rec_eigen_dim, rec_hand_dim,
                    rec_default_joint_pos, rec_joint_names,
                    obs_group_names, obs_group_sizes, rec_control_hz,
                    clip_actions,
                )
                break

            # perform operations for terminated episodes
            if len(dones) > 0:
                # reset rnn state for terminated episodes
                if agent.is_rnn and agent.states is not None:
                    for s in agent.states:
                        s[:, dones, :] = 0.0
        if args_cli.video:
            timestep += 1
            # exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # cleanup keyboard subscription
    kb_input.unsubscribe_to_keyboard_events(keyboard, kb_sub)

    # close the simulator
    env.close()


def _save_recording(
    out_path, obs_list, actions_list, joint_pos_list,
    pca_matrix, action_scale, arm_dim, eigen_dim, hand_dim,
    default_joint_pos, joint_names,
    obs_group_names, obs_group_sizes, control_hz,
    clip_actions,
):
    """Save recorded episode data to npz."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    save_dict = {
        "obs": np.array(obs_list),
        "actions": np.array(actions_list),
        "joint_pos": np.array(joint_pos_list),
        "pca_matrix": pca_matrix,
        "action_scale": np.array([action_scale]) if isinstance(action_scale, float) else action_scale,
        "arm_dim": np.array([arm_dim]),
        "eigen_dim": np.array([eigen_dim]),
        "hand_dim": np.array([hand_dim]),
        "default_joint_pos": default_joint_pos,
        "joint_names": np.array(joint_names),
        "obs_group_names": np.array(obs_group_names),
        "obs_group_sizes": np.array(obs_group_sizes),
        "control_hz": np.array([control_hz]),
        "clip_actions": np.array([clip_actions]),
    }

    np.savez(out_path, **save_dict)
    print(f"[SAVED] {out_path}")
    print(f"  Steps: {len(actions_list)}")
    print(f"  obs shape: {save_dict['obs'].shape}")
    print(f"  actions shape: {save_dict['actions'].shape}")
    print(f"  joint_pos shape: {save_dict['joint_pos'].shape}")
    print(f"  obs groups: {list(zip(obs_group_names, obs_group_sizes))}")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
