# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Y2R Kuka Allegro trajectory following environment.

Single registration with config variant selection via Y2R_VARIANT env var:
    Y2R_VARIANT=push ./scripts/push.sh --continue

Available variants:
    - base: Teacher training (default)
    - play: Teacher evaluation
    - push: Push-T task evaluation
    - cup: Cup task evaluation
    - student: Student training/distillation
    - student_play: Student evaluation
"""

import gymnasium as gym

from ...configs import agents as shared_agents
from . import agents

##
# Trajectory Following Environment
##

gym.register(
    id="Isaac-Trajectory-Kuka-Allegro-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.trajectory_kuka_allegro_env_cfg:TrajectoryKukaAllegroEnvCfg",
        # Teacher agent config (default for train.py, play.py)
        "rl_games_cfg_entry_point": f"{shared_agents.__name__}:teacher_ppo.yaml",
        # Student agent configs (distill.py uses rl_games_student_cfg_entry_point by default)
        "rl_games_student_cfg_entry_point": f"{shared_agents.__name__}:student_depth_ppo.yaml",
        "rl_games_student_pt_cfg_entry_point": f"{shared_agents.__name__}:student_pt_ppo.yaml",
        # Teacher config for distillation (used by distill.py --teacher-agent)
        "rl_games_teacher_cfg_entry_point": f"{shared_agents.__name__}:teacher_ppo.yaml",
        # Other frameworks
        "rl_games_point_transformer_cfg_entry_point": f"{shared_agents.__name__}:point_transformer_ppo.yaml",
        "rl_games_pointnet_tnet_cfg_entry_point": f"{shared_agents.__name__}:pointnet_tnet_ppo.yaml",
        # Robot-specific (experiment name)
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Y2RKukaAllegroPPORunnerCfg",
    },
)
