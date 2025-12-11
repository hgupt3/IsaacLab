# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Y2R Kuka Allegro trajectory following environments.
"""

import gymnasium as gym

from . import agents

##
# Trajectory Following Environments
##

gym.register(
    id="Isaac-Trajectory-Kuka-Allegro-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.trajectory_kuka_allegro_env_cfg:TrajectoryKukaAllegroEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_trajectory_ppo_cfg.yaml",
        "rl_games_point_transformer_cfg_entry_point": f"{agents.__name__}:rl_games_point_transformer_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Y2RKukaAllegroPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Trajectory-Kuka-Allegro-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.trajectory_kuka_allegro_env_cfg:TrajectoryKukaAllegroEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_trajectory_ppo_cfg.yaml",
        "rl_games_point_transformer_cfg_entry_point": f"{agents.__name__}:rl_games_point_transformer_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Y2RKukaAllegroPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Trajectory-Kuka-Allegro-PushT-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.trajectory_kuka_allegro_env_cfg:TrajectoryKukaAllegroEnvCfg_PUSH",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_trajectory_ppo_cfg.yaml",
        "rl_games_point_transformer_cfg_entry_point": f"{agents.__name__}:rl_games_point_transformer_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Y2RKukaAllegroPPORunnerCfg",
    },
)
