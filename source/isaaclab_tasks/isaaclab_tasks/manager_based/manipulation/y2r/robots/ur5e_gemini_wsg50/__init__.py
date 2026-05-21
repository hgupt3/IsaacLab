"""Y2R UR5e + Gemini 305 + WSG50 trajectory following environment."""

import gymnasium as gym

from ...configs import agents as shared_agents


gym.register(
    id="Isaac-Trajectory-UR5e-Gemini-WSG50-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.trajectory_ur5e_gemini_wsg50_env_cfg:TrajectoryUR5eGeminiWSG50EnvCfg",
        "rl_games_cfg_entry_point": f"{shared_agents.__name__}:teacher_ppo.yaml",
        "rl_games_student_cfg_entry_point": f"{shared_agents.__name__}:student_depth_ppo.yaml",
        "rl_games_student_pt_cfg_entry_point": f"{shared_agents.__name__}:student_pt_ppo.yaml",
        "rl_games_student_pt_dagger_cfg_entry_point": f"{shared_agents.__name__}:student_pt_dagger_ppo.yaml",
        "rl_games_student_pt_no_depth_cfg_entry_point": f"{shared_agents.__name__}:student_pt_no_depth_ppo.yaml",
        "rl_games_student_pt_patch_cfg_entry_point": f"{shared_agents.__name__}:student_pt_patch_ppo.yaml",
        "rl_games_student_pt_patch_dagger_cfg_entry_point": f"{shared_agents.__name__}:student_pt_patch_dagger_ppo.yaml",
        "rl_games_student_pt_patch_no_depth_cfg_entry_point": f"{shared_agents.__name__}:student_pt_patch_no_depth_ppo.yaml",
        "rl_games_teacher_cfg_entry_point": f"{shared_agents.__name__}:teacher_ppo.yaml",
        "rl_games_point_transformer_cfg_entry_point": f"{shared_agents.__name__}:point_transformer_ppo.yaml",
        "rl_games_pointnet_tnet_cfg_entry_point": f"{shared_agents.__name__}:pointnet_tnet_ppo.yaml",
    },
)
