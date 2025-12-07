# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from . import mdp
from .trajectory_cfg import TRAJECTORY_PARAMS


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # adr stands for automatic/adaptive domain randomization
    adr = CurrTerm(
        func=mdp.DifficultyScheduler, params={"init_difficulty": 0, "min_difficulty": 0, "max_difficulty": 10}
    )

    joint_pos_unoise_min_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "observations.proprio.joint_pos.noise.n_min",
            "modify_fn": mdp.initial_final_interpolate_fn,
            "modify_params": {"initial_value": 0.0, "final_value": -0.1, "difficulty_term_str": "adr"},
        },
    )

    joint_pos_unoise_max_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "observations.proprio.joint_pos.noise.n_max",
            "modify_fn": mdp.initial_final_interpolate_fn,
            "modify_params": {"initial_value": 0.0, "final_value": 0.1, "difficulty_term_str": "adr"},
        },
    )

    joint_vel_unoise_min_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "observations.proprio.joint_vel.noise.n_min",
            "modify_fn": mdp.initial_final_interpolate_fn,
            "modify_params": {"initial_value": 0.0, "final_value": -0.2, "difficulty_term_str": "adr"},
        },
    )

    joint_vel_unoise_max_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "observations.proprio.joint_vel.noise.n_max",
            "modify_fn": mdp.initial_final_interpolate_fn,
            "modify_params": {"initial_value": 0.0, "final_value": 0.2, "difficulty_term_str": "adr"},
        },
    )

    hand_tips_pos_unoise_min_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "observations.proprio.hand_tips_state_b.noise.n_min",
            "modify_fn": mdp.initial_final_interpolate_fn,
            "modify_params": {"initial_value": 0.0, "final_value": -0.01, "difficulty_term_str": "adr"},
        },
    )

    hand_tips_pos_unoise_max_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "observations.proprio.hand_tips_state_b.noise.n_max",
            "modify_fn": mdp.initial_final_interpolate_fn,
            "modify_params": {"initial_value": 0.0, "final_value": 0.01, "difficulty_term_str": "adr"},
        },
    )

    object_quat_unoise_min_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "observations.policy.object_quat_b.noise.n_min",
            "modify_fn": mdp.initial_final_interpolate_fn,
            "modify_params": {"initial_value": 0.0, "final_value": -0.03, "difficulty_term_str": "adr"},
        },
    )

    object_quat_unoise_max_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "observations.policy.object_quat_b.noise.n_max",
            "modify_fn": mdp.initial_final_interpolate_fn,
            "modify_params": {"initial_value": 0.0, "final_value": 0.03, "difficulty_term_str": "adr"},
        },
    )

    object_obs_unoise_min_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "observations.perception.object_point_cloud.noise.n_min",
            "modify_fn": mdp.initial_final_interpolate_fn,
            "modify_params": {"initial_value": 0.0, "final_value": -0.01, "difficulty_term_str": "adr"},
        },
    )

    object_obs_unoise_max_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "observations.perception.object_point_cloud.noise.n_max",
            "modify_fn": mdp.initial_final_interpolate_fn,
            "modify_params": {"initial_value": 0.0, "final_value": -0.01, "difficulty_term_str": "adr"},
        },
    )

    gravity_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "events.variable_gravity.params.gravity_distribution_params",
            "modify_fn": mdp.initial_final_interpolate_fn,
            "modify_params": {
                "initial_value": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                "final_value": ((0.0, 0.0, -9.81), (0.0, 0.0, -9.81)),
                "difficulty_term_str": "adr",
            },
        },
    )


@configclass
class TrajectoryCurriculumCfg:
    """Curriculum terms for trajectory following task."""

    # Difficulty scheduler using point-to-point error
    adr = CurrTerm(
        func=mdp.DifficultyScheduler,
        params={
            "init_difficulty": 0,
            "min_difficulty": 0,
            "max_difficulty": 10,
            "error_tol": TRAJECTORY_PARAMS.curriculum_error_tol,
            "use_trajectory": True,
        },
    )

    # Termination threshold: start lenient, become stricter
    termination_threshold_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "terminations.trajectory_deviation.params.threshold",
            "modify_fn": mdp.initial_final_interpolate_fn,
            "modify_params": {
                "initial_value": TRAJECTORY_PARAMS.termination_threshold_initial,
                "final_value": TRAJECTORY_PARAMS.termination_threshold,
                "difficulty_term_str": "adr",
            },
        },
    )

    # Success threshold: start lenient, become stricter
    success_threshold_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.trajectory_success.params.error_threshold",
            "modify_fn": mdp.initial_final_interpolate_fn,
            "modify_params": {
                "initial_value": TRAJECTORY_PARAMS.success_threshold_initial,
                "final_value": TRAJECTORY_PARAMS.success_threshold,
                "difficulty_term_str": "adr",
            },
        },
    )

    # Joint position noise
    joint_pos_unoise_min_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "observations.proprio.joint_pos.noise.n_min",
            "modify_fn": mdp.initial_final_interpolate_fn,
            "modify_params": {"initial_value": 0.0, "final_value": -0.1, "difficulty_term_str": "adr"},
        },
    )

    joint_pos_unoise_max_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "observations.proprio.joint_pos.noise.n_max",
            "modify_fn": mdp.initial_final_interpolate_fn,
            "modify_params": {"initial_value": 0.0, "final_value": 0.1, "difficulty_term_str": "adr"},
        },
    )

    # Joint velocity noise
    joint_vel_unoise_min_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "observations.proprio.joint_vel.noise.n_min",
            "modify_fn": mdp.initial_final_interpolate_fn,
            "modify_params": {"initial_value": 0.0, "final_value": -0.2, "difficulty_term_str": "adr"},
        },
    )

    joint_vel_unoise_max_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "observations.proprio.joint_vel.noise.n_max",
            "modify_fn": mdp.initial_final_interpolate_fn,
            "modify_params": {"initial_value": 0.0, "final_value": 0.2, "difficulty_term_str": "adr"},
        },
    )

    # Hand tips noise
    hand_tips_pos_unoise_min_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "observations.proprio.hand_tips_state_b.noise.n_min",
            "modify_fn": mdp.initial_final_interpolate_fn,
            "modify_params": {"initial_value": 0.0, "final_value": -0.01, "difficulty_term_str": "adr"},
        },
    )

    hand_tips_pos_unoise_max_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "observations.proprio.hand_tips_state_b.noise.n_max",
            "modify_fn": mdp.initial_final_interpolate_fn,
            "modify_params": {"initial_value": 0.0, "final_value": 0.01, "difficulty_term_str": "adr"},
        },
    )

    # Current object point cloud noise
    object_obs_unoise_min_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "observations.perception.object_point_cloud.noise.n_min",
            "modify_fn": mdp.initial_final_interpolate_fn,
            "modify_params": {"initial_value": 0.0, "final_value": -0.01, "difficulty_term_str": "adr"},
        },
    )

    object_obs_unoise_max_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "observations.perception.object_point_cloud.noise.n_max",
            "modify_fn": mdp.initial_final_interpolate_fn,
            "modify_params": {"initial_value": 0.0, "final_value": 0.01, "difficulty_term_str": "adr"},
        },
    )

    # Target point clouds noise
    target_obs_unoise_min_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "observations.targets.target_point_clouds.noise.n_min",
            "modify_fn": mdp.initial_final_interpolate_fn,
            "modify_params": {"initial_value": 0.0, "final_value": -0.01, "difficulty_term_str": "adr"},
        },
    )

    target_obs_unoise_max_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "observations.targets.target_point_clouds.noise.n_max",
            "modify_fn": mdp.initial_final_interpolate_fn,
            "modify_params": {"initial_value": 0.0, "final_value": 0.01, "difficulty_term_str": "adr"},
        },
    )

    # Gravity scheduling (0 -> -9.81)
    gravity_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "events.variable_gravity.params.gravity_distribution_params",
            "modify_fn": mdp.initial_final_interpolate_fn,
            "modify_params": {
                "initial_value": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                "final_value": ((0.0, 0.0, -9.81), (0.0, 0.0, -9.81)),
                "difficulty_term_str": "adr",
            },
        },
    )
