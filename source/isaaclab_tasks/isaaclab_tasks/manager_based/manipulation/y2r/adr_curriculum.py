# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from . import mdp
from .config_loader import Y2RConfig


def build_curriculum_cfg(cfg: Y2RConfig):
    """Build TrajectoryCurriculumCfg from config.
    
    This function creates the curriculum configuration dynamically using
    values from the provided Y2RConfig, avoiding import-time config loading.
    """
    
    @configclass
    class TrajectoryCurriculumCfg:
        """Curriculum terms for trajectory following task."""

        adr = CurrTerm(
            func=mdp.DifficultyScheduler,
            params={
                "init_difficulty": cfg.curriculum.difficulty.initial,
                "min_difficulty": cfg.curriculum.difficulty.min,
                "max_difficulty": cfg.curriculum.difficulty.max,
                "use_trajectory": True,
                "pos_tol": cfg.curriculum.advancement_tolerances.position,
                "rot_tol": cfg.curriculum.advancement_tolerances.rotation,
                "pc_tol": cfg.curriculum.advancement_tolerances.point_cloud,
                "step_interval": cfg.curriculum.scheduler.step_interval,
                "use_performance": cfg.curriculum.scheduler.use_performance,
            },
        )

        termination_threshold_adr = CurrTerm(
            func=mdp.modify_term_cfg,
            params={
                "address": "terminations.trajectory_deviation.params.threshold",
                "modify_fn": mdp.initial_final_interpolate_fn,
                "modify_params": {
                    "initial_value": cfg.terminations.trajectory_deviation.position_threshold[0],
                    "final_value": cfg.terminations.trajectory_deviation.position_threshold[1],
                    "difficulty_term_str": "adr",
                },
            },
        )

        # Sparse trajectory_success reward: curriculum for pos_threshold (tightens from initial to final)
        success_pos_threshold_adr = CurrTerm(
            func=mdp.modify_term_cfg,
            params={
                "address": "rewards.trajectory_success.params.pos_threshold",
                "modify_fn": mdp.initial_final_interpolate_fn,
                "modify_params": {
                    "initial_value": cfg.rewards.trajectory_success.params["pos_threshold"][0],
                    "final_value": cfg.rewards.trajectory_success.params["pos_threshold"][1],
                    "difficulty_term_str": "adr",
                },
            },
        )

        termination_rot_threshold_adr = CurrTerm(
            func=mdp.modify_term_cfg,
            params={
                "address": "terminations.trajectory_deviation.params.rot_threshold",
                "modify_fn": mdp.initial_final_interpolate_fn,
                "modify_params": {
                    "initial_value": cfg.terminations.trajectory_deviation.rotation_threshold[0],
                    "final_value": cfg.terminations.trajectory_deviation.rotation_threshold[1],
                    "difficulty_term_str": "adr",
                },
            },
        )

        # Hand pose deviation termination: curriculum for position threshold
        hand_pose_term_pos_adr = CurrTerm(
            func=mdp.modify_term_cfg,
            params={
                "address": "terminations.hand_pose_deviation.params.pos_threshold",
                "modify_fn": mdp.initial_final_interpolate_fn,
                "modify_params": {
                    "initial_value": cfg.terminations.hand_pose_deviation.position_threshold[0],
                    "final_value": cfg.terminations.hand_pose_deviation.position_threshold[1],
                    "difficulty_term_str": "adr",
                },
            },
        )

        # Hand pose deviation termination: curriculum for rotation threshold
        hand_pose_term_rot_adr = CurrTerm(
            func=mdp.modify_term_cfg,
            params={
                "address": "terminations.hand_pose_deviation.params.rot_threshold",
                "modify_fn": mdp.initial_final_interpolate_fn,
                "modify_params": {
                    "initial_value": cfg.terminations.hand_pose_deviation.rotation_threshold[0],
                    "final_value": cfg.terminations.hand_pose_deviation.rotation_threshold[1],
                    "difficulty_term_str": "adr",
                },
            },
        )

        # Sparse trajectory_success reward: curriculum for rot_threshold (tightens from initial to final)
        success_rot_threshold_adr = CurrTerm(
            func=mdp.modify_term_cfg,
            params={
                "address": "rewards.trajectory_success.params.rot_threshold",
                "modify_fn": mdp.initial_final_interpolate_fn,
                "modify_params": {
                    "initial_value": cfg.rewards.trajectory_success.params["rot_threshold"][0],
                    "final_value": cfg.rewards.trajectory_success.params["rot_threshold"][1],
                    "difficulty_term_str": "adr",
                },
            },
        )

        joint_pos_unoise_min_adr = CurrTerm(
            func=mdp.modify_term_cfg,
            params={
                "address": "observations.proprio.joint_pos.noise.n_min",
                "modify_fn": mdp.initial_final_interpolate_fn,
                "modify_params": {
                    "initial_value": 0.0,
                    "final_value": -cfg.curriculum.noise.joint_pos[1],
                    "difficulty_term_str": "adr",
                },
            },
        )

        joint_pos_unoise_max_adr = CurrTerm(
            func=mdp.modify_term_cfg,
            params={
                "address": "observations.proprio.joint_pos.noise.n_max",
                "modify_fn": mdp.initial_final_interpolate_fn,
                "modify_params": {
                    "initial_value": 0.0,
                    "final_value": cfg.curriculum.noise.joint_pos[1],
                    "difficulty_term_str": "adr",
                },
            },
        )

        joint_vel_unoise_min_adr = CurrTerm(
            func=mdp.modify_term_cfg,
            params={
                "address": "observations.proprio.joint_vel.noise.n_min",
                "modify_fn": mdp.initial_final_interpolate_fn,
                "modify_params": {
                    "initial_value": 0.0,
                    "final_value": -cfg.curriculum.noise.joint_vel[1],
                    "difficulty_term_str": "adr",
                },
            },
        )

        joint_vel_unoise_max_adr = CurrTerm(
            func=mdp.modify_term_cfg,
            params={
                "address": "observations.proprio.joint_vel.noise.n_max",
                "modify_fn": mdp.initial_final_interpolate_fn,
                "modify_params": {
                    "initial_value": 0.0,
                    "final_value": cfg.curriculum.noise.joint_vel[1],
                    "difficulty_term_str": "adr",
                },
            },
        )

        hand_tips_pos_unoise_min_adr = CurrTerm(
            func=mdp.modify_term_cfg,
            params={
                "address": "observations.proprio.hand_tips_state_b.noise.n_min",
                "modify_fn": mdp.initial_final_interpolate_fn,
                "modify_params": {
                    "initial_value": 0.0,
                    "final_value": -cfg.curriculum.noise.hand_tips[1],
                    "difficulty_term_str": "adr",
                },
            },
        )

        hand_tips_pos_unoise_max_adr = CurrTerm(
            func=mdp.modify_term_cfg,
            params={
                "address": "observations.proprio.hand_tips_state_b.noise.n_max",
                "modify_fn": mdp.initial_final_interpolate_fn,
                "modify_params": {
                    "initial_value": 0.0,
                    "final_value": cfg.curriculum.noise.hand_tips[1],
                    "difficulty_term_str": "adr",
                },
            },
        )

        hand_eigen_unoise_min_adr = CurrTerm(
            func=mdp.modify_term_cfg,
            params={
                "address": "observations.proprio.hand_eigen.noise.n_min",
                "modify_fn": mdp.initial_final_interpolate_fn,
                "modify_params": {
                    "initial_value": 0.0,
                    "final_value": -cfg.curriculum.noise.hand_eigen[1],
                    "difficulty_term_str": "adr",
                },
            },
        )

        hand_eigen_unoise_max_adr = CurrTerm(
            func=mdp.modify_term_cfg,
            params={
                "address": "observations.proprio.hand_eigen.noise.n_max",
                "modify_fn": mdp.initial_final_interpolate_fn,
                "modify_params": {
                    "initial_value": 0.0,
                    "final_value": cfg.curriculum.noise.hand_eigen[1],
                    "difficulty_term_str": "adr",
                },
            },
        )

        object_obs_unoise_min_adr = CurrTerm(
            func=mdp.modify_term_cfg,
            params={
                "address": "observations.current_pc.object_point_cloud.noise.n_min",
                "modify_fn": mdp.initial_final_interpolate_fn,
                "modify_params": {
                    "initial_value": 0.0,
                    "final_value": -cfg.curriculum.noise.object_point_cloud[1],
                    "difficulty_term_str": "adr",
                },
            },
        )

        object_obs_unoise_max_adr = CurrTerm(
            func=mdp.modify_term_cfg,
            params={
                "address": "observations.current_pc.object_point_cloud.noise.n_max",
                "modify_fn": mdp.initial_final_interpolate_fn,
                "modify_params": {
                    "initial_value": 0.0,
                    "final_value": cfg.curriculum.noise.object_point_cloud[1],
                    "difficulty_term_str": "adr",
                },
            },
        )

        target_obs_unoise_min_adr = CurrTerm(
            func=mdp.modify_term_cfg,
            params={
                "address": "observations.future_pc.target_point_clouds.noise.n_min",
                "modify_fn": mdp.initial_final_interpolate_fn,
                "modify_params": {
                    "initial_value": 0.0,
                    "final_value": -cfg.curriculum.noise.target_point_clouds[1],
                    "difficulty_term_str": "adr",
                },
            },
        )

        target_obs_unoise_max_adr = CurrTerm(
            func=mdp.modify_term_cfg,
            params={
                "address": "observations.future_pc.target_point_clouds.noise.n_max",
                "modify_fn": mdp.initial_final_interpolate_fn,
                "modify_params": {
                    "initial_value": 0.0,
                    "final_value": cfg.curriculum.noise.target_point_clouds[1],
                    "difficulty_term_str": "adr",
                },
            },
        )

        object_pose_unoise_min_adr = CurrTerm(
            func=mdp.modify_term_cfg,
            params={
                "address": "observations.current_poses.object_pose.noise.n_min",
                "modify_fn": mdp.initial_final_interpolate_fn,
                "modify_params": {
                    "initial_value": 0.0,
                    "final_value": -cfg.curriculum.noise.object_pose[1],
                    "difficulty_term_str": "adr",
                },
            },
        )

        object_pose_unoise_max_adr = CurrTerm(
            func=mdp.modify_term_cfg,
            params={
                "address": "observations.current_poses.object_pose.noise.n_max",
                "modify_fn": mdp.initial_final_interpolate_fn,
                "modify_params": {
                    "initial_value": 0.0,
                    "final_value": cfg.curriculum.noise.object_pose[1],
                    "difficulty_term_str": "adr",
                },
            },
        )

        hand_pose_unoise_min_adr = CurrTerm(
            func=mdp.modify_term_cfg,
            params={
                "address": "observations.current_poses.hand_pose.noise.n_min",
                "modify_fn": mdp.initial_final_interpolate_fn,
                "modify_params": {
                    "initial_value": 0.0,
                    "final_value": -cfg.curriculum.noise.hand_pose[1],
                    "difficulty_term_str": "adr",
                },
            },
        )

        hand_pose_unoise_max_adr = CurrTerm(
            func=mdp.modify_term_cfg,
            params={
                "address": "observations.current_poses.hand_pose.noise.n_max",
                "modify_fn": mdp.initial_final_interpolate_fn,
                "modify_params": {
                    "initial_value": 0.0,
                    "final_value": cfg.curriculum.noise.hand_pose[1],
                    "difficulty_term_str": "adr",
                },
            },
        )

        target_poses_unoise_min_adr = CurrTerm(
            func=mdp.modify_term_cfg,
            params={
                "address": "observations.future_poses.target_poses.noise.n_min",
                "modify_fn": mdp.initial_final_interpolate_fn,
                "modify_params": {
                    "initial_value": 0.0,
                    "final_value": -cfg.curriculum.noise.target_poses[1],
                    "difficulty_term_str": "adr",
                },
            },
        )

        target_poses_unoise_max_adr = CurrTerm(
            func=mdp.modify_term_cfg,
            params={
                "address": "observations.future_poses.target_poses.noise.n_max",
                "modify_fn": mdp.initial_final_interpolate_fn,
                "modify_params": {
                    "initial_value": 0.0,
                    "final_value": cfg.curriculum.noise.target_poses[1],
                    "difficulty_term_str": "adr",
                },
            },
        )

        hand_pose_targets_unoise_min_adr = CurrTerm(
            func=mdp.modify_term_cfg,
            params={
                "address": "observations.future_poses.hand_pose_targets.noise.n_min",
                "modify_fn": mdp.initial_final_interpolate_fn,
                "modify_params": {
                    "initial_value": 0.0,
                    "final_value": -cfg.curriculum.noise.hand_pose_targets[1],
                    "difficulty_term_str": "adr",
                },
            },
        )

        hand_pose_targets_unoise_max_adr = CurrTerm(
            func=mdp.modify_term_cfg,
            params={
                "address": "observations.future_poses.hand_pose_targets.noise.n_max",
                "modify_fn": mdp.initial_final_interpolate_fn,
                "modify_params": {
                    "initial_value": 0.0,
                    "final_value": cfg.curriculum.noise.hand_pose_targets[1],
                    "difficulty_term_str": "adr",
                },
            },
        )

        gravity_adr = CurrTerm(
            func=mdp.modify_term_cfg,
            params={
                "address": "events.variable_gravity.params.gravity_distribution_params",
                "modify_fn": mdp.initial_final_interpolate_fn,
                "modify_params": {
                    "initial_value": (tuple(cfg.curriculum.gravity.initial), tuple(cfg.curriculum.gravity.initial)),
                    "final_value": (tuple(cfg.curriculum.gravity.final), tuple(cfg.curriculum.gravity.final)),
                    "difficulty_term_str": "adr",
                },
            },
        )

        gate_floor_adr = CurrTerm(
            func=mdp.modify_term_cfg,
            params={
                "address": "rewards.hand_pose_following.params.gate_floor",
                "modify_fn": mdp.initial_final_interpolate_fn,
                "modify_params": {
                    "initial_value": cfg.curriculum.gate_floor[0],
                    "final_value": cfg.curriculum.gate_floor[1],
                    "difficulty_term_str": "adr",
                },
            },
        )

        lookahead_grasp_scale_adr = CurrTerm(
            func=mdp.modify_term_cfg,
            params={
                "address": "rewards.lookahead_tracking.params.grasp_scale",
                "modify_fn": mdp.initial_final_interpolate_fn,
                "modify_params": {
                    "initial_value": cfg.curriculum.grasp_scale[0],
                    "final_value": cfg.curriculum.grasp_scale[1],
                    "difficulty_term_str": "adr",
                },
            },
        )

    return TrajectoryCurriculumCfg()


# For backwards compatibility - this is now just a type alias
# The actual class is created by build_curriculum_cfg()
TrajectoryCurriculumCfg = type("TrajectoryCurriculumCfg", (), {})

