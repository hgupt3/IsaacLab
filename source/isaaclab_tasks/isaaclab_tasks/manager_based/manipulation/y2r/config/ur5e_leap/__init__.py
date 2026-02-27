"""
Y2R UR5e + LEAP Hand trajectory following environment.

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

from .. import agents as shared_agents

##
# Trajectory Following Environment
##

gym.register(
    id="Isaac-Trajectory-UR5e-Leap-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.trajectory_ur5e_leap_env_cfg:TrajectoryUR5eLeapEnvCfg",
        # Teacher agent config (default for train.py, play.py)
        "rl_games_cfg_entry_point": f"{shared_agents.__name__}:rl_games_trajectory_ppo_cfg.yaml",
        # Student agent configs (distill.py uses rl_games_student_cfg_entry_point by default)
        "rl_games_student_cfg_entry_point": f"{shared_agents.__name__}:rl_games_student_depth_ppo_cfg.yaml",
        "rl_games_student_pt_cfg_entry_point": f"{shared_agents.__name__}:rl_games_student_depth_pt_ppo_cfg.yaml",
        # Teacher config for distillation (used by distill.py --teacher-agent)
        "rl_games_teacher_cfg_entry_point": f"{shared_agents.__name__}:rl_games_trajectory_ppo_cfg.yaml",
        # Other frameworks
        "rl_games_point_transformer_cfg_entry_point": f"{shared_agents.__name__}:rl_games_point_transformer_cfg.yaml",
        "rl_games_pointnet_tnet_cfg_entry_point": f"{shared_agents.__name__}:rl_games_pointnet_tnet_cfg.yaml",
    },
)
