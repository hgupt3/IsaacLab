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
                "level_overrides": cfg.curriculum.scheduler.level_overrides,
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

        # --- Teacher noise: Gaussian std curriculum ---
        # joint_pos split into arm/hand
        joint_pos_arm_noise_adr = CurrTerm(
            func=mdp.modify_term_cfg,
            params={
                "address": "observations.proprio.joint_pos_arm.noise.std",
                "modify_fn": mdp.initial_final_interpolate_fn,
                "modify_params": {
                    "initial_value": cfg.curriculum.teacher_noise.joint_pos_arm[0],
                    "final_value": cfg.curriculum.teacher_noise.joint_pos_arm[1],
                    "difficulty_term_str": "adr",
                },
            },
        )
        joint_pos_hand_noise_adr = CurrTerm(
            func=mdp.modify_term_cfg,
            params={
                "address": "observations.proprio.joint_pos_hand.noise.std",
                "modify_fn": mdp.initial_final_interpolate_fn,
                "modify_params": {
                    "initial_value": cfg.curriculum.teacher_noise.joint_pos_hand[0],
                    "final_value": cfg.curriculum.teacher_noise.joint_pos_hand[1],
                    "difficulty_term_str": "adr",
                },
            },
        )
        # joint_vel split into arm/hand
        joint_vel_arm_noise_adr = CurrTerm(
            func=mdp.modify_term_cfg,
            params={
                "address": "observations.proprio.joint_vel_arm.noise.std",
                "modify_fn": mdp.initial_final_interpolate_fn,
                "modify_params": {
                    "initial_value": cfg.curriculum.teacher_noise.joint_vel_arm[0],
                    "final_value": cfg.curriculum.teacher_noise.joint_vel_arm[1],
                    "difficulty_term_str": "adr",
                },
            },
        )
        joint_vel_hand_noise_adr = CurrTerm(
            func=mdp.modify_term_cfg,
            params={
                "address": "observations.proprio.joint_vel_hand.noise.std",
                "modify_fn": mdp.initial_final_interpolate_fn,
                "modify_params": {
                    "initial_value": cfg.curriculum.teacher_noise.joint_vel_hand[0],
                    "final_value": cfg.curriculum.teacher_noise.joint_vel_hand[1],
                    "difficulty_term_str": "adr",
                },
            },
        )
        joint_pos_targets_noise_adr = CurrTerm(
            func=mdp.modify_term_cfg,
            params={
                "address": "observations.proprio.joint_pos_targets.noise.std",
                "modify_fn": mdp.initial_final_interpolate_fn,
                "modify_params": {
                    "initial_value": cfg.curriculum.teacher_noise.joint_pos_targets[0],
                    "final_value": cfg.curriculum.teacher_noise.joint_pos_targets[1],
                    "difficulty_term_str": "adr",
                },
            },
        )
        hand_tips_noise_adr = CurrTerm(
            func=mdp.modify_term_cfg,
            params={
                "address": "observations.proprio.hand_tips_state_b.noise.std",
                "modify_fn": mdp.initial_final_interpolate_fn,
                "modify_params": {
                    "initial_value": cfg.curriculum.teacher_noise.hand_tips[0],
                    "final_value": cfg.curriculum.teacher_noise.hand_tips[1],
                    "difficulty_term_str": "adr",
                },
            },
        )
        hand_eigen_noise_adr = CurrTerm(
            func=mdp.modify_term_cfg,
            params={
                "address": "observations.proprio.hand_eigen.noise.std",
                "modify_fn": mdp.initial_final_interpolate_fn,
                "modify_params": {
                    "initial_value": cfg.curriculum.teacher_noise.hand_eigen[0],
                    "final_value": cfg.curriculum.teacher_noise.hand_eigen[1],
                    "difficulty_term_str": "adr",
                },
            },
        )
        object_pc_noise_adr = CurrTerm(
            func=mdp.modify_term_cfg,
            params={
                "address": "observations.current_pc.object_point_cloud.noise.std",
                "modify_fn": mdp.initial_final_interpolate_fn,
                "modify_params": {
                    "initial_value": cfg.curriculum.teacher_noise.object_point_cloud[0],
                    "final_value": cfg.curriculum.teacher_noise.object_point_cloud[1],
                    "difficulty_term_str": "adr",
                },
            },
        )
        target_pc_noise_adr = CurrTerm(
            func=mdp.modify_term_cfg,
            params={
                "address": "observations.future_pc.target_point_clouds.noise.std",
                "modify_fn": mdp.initial_final_interpolate_fn,
                "modify_params": {
                    "initial_value": cfg.curriculum.teacher_noise.target_point_clouds[0],
                    "final_value": cfg.curriculum.teacher_noise.target_point_clouds[1],
                    "difficulty_term_str": "adr",
                },
            },
        )
        object_pose_noise_adr = CurrTerm(
            func=mdp.modify_term_cfg,
            params={
                "address": "observations.current_poses.object_pose.noise.std",
                "modify_fn": mdp.initial_final_interpolate_fn,
                "modify_params": {
                    "initial_value": cfg.curriculum.teacher_noise.object_pose[0],
                    "final_value": cfg.curriculum.teacher_noise.object_pose[1],
                    "difficulty_term_str": "adr",
                },
            },
        )
        hand_pose_noise_adr = CurrTerm(
            func=mdp.modify_term_cfg,
            params={
                "address": "observations.current_poses.hand_pose.noise.std",
                "modify_fn": mdp.initial_final_interpolate_fn,
                "modify_params": {
                    "initial_value": cfg.curriculum.teacher_noise.hand_pose[0],
                    "final_value": cfg.curriculum.teacher_noise.hand_pose[1],
                    "difficulty_term_str": "adr",
                },
            },
        )
        target_poses_noise_adr = CurrTerm(
            func=mdp.modify_term_cfg,
            params={
                "address": "observations.future_poses.target_poses.noise.std",
                "modify_fn": mdp.initial_final_interpolate_fn,
                "modify_params": {
                    "initial_value": cfg.curriculum.teacher_noise.target_poses[0],
                    "final_value": cfg.curriculum.teacher_noise.target_poses[1],
                    "difficulty_term_str": "adr",
                },
            },
        )
        hand_pose_targets_noise_adr = CurrTerm(
            func=mdp.modify_term_cfg,
            params={
                "address": "observations.future_poses.hand_pose_targets.noise.std",
                "modify_fn": mdp.initial_final_interpolate_fn,
                "modify_params": {
                    "initial_value": cfg.curriculum.teacher_noise.hand_pose_targets[0],
                    "final_value": cfg.curriculum.teacher_noise.hand_pose_targets[1],
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

    result = TrajectoryCurriculumCfg()

    # Student noise curriculum (Gaussian std) for student observation groups only.
    # Only added when student mode is active (student obs groups are None otherwise).
    if cfg.mode.use_student_mode:
        _student_noise_entries = [
            ("student_jp_arm",  "observations.student_proprio.joint_pos_arm",             cfg.curriculum.student_noise.joint_pos_arm),
            ("student_jp_hand", "observations.student_proprio.joint_pos_hand",            cfg.curriculum.student_noise.joint_pos_hand),
            ("student_eigen",   "observations.student_proprio.hand_eigen",                cfg.curriculum.student_noise.hand_eigen),
            ("student_jp_tgt",  "observations.student_proprio.joint_pos_targets",         cfg.curriculum.student_noise.joint_pos_targets),
            ("student_hpose",   "observations.student_current_poses.hand_pose",           cfg.curriculum.student_noise.hand_pose),
            ("student_hpose_t", "observations.student_future_poses.hand_pose_targets",    cfg.curriculum.student_noise.hand_pose_targets),
            ("student_vpc",     "observations.student_current_pc.visible_point_cloud",    cfg.curriculum.student_noise.object_point_cloud),
            ("student_vtgt",    "observations.student_future_pc.visible_target_sequence", cfg.curriculum.student_noise.target_point_clouds),
        ]
        for name, address, noise_cfg in _student_noise_entries:
            setattr(result, f"{name}_noise_adr", CurrTerm(
                func=mdp.modify_term_cfg,
                params={
                    "address": f"{address}.noise.std",
                    "modify_fn": mdp.initial_final_interpolate_fn,
                    "modify_params": {
                        "initial_value": noise_cfg[0],
                        "final_value": noise_cfg[1],
                        "difficulty_term_str": "adr",
                    },
                },
            ))

    return result


# For backwards compatibility - this is now just a type alias
# The actual class is created by build_curriculum_cfg()
TrajectoryCurriculumCfg = type("TrajectoryCurriculumCfg", (), {})
