# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Persist common_step_counter and ADR curriculum state across RL-Games checkpoint save/resume.

The environment's common_step_counter resets to 0 whenever the env is
re-created (e.g. on --continue). This observer patches the algo's
save/restore methods to include the counter in .pth checkpoints, so
step-based schedules (like ADR's step_based_floor) resume correctly.

Additionally persists per-env ADR difficulties so that resume doesn't
snap all environments back to init_difficulty.
"""

from __future__ import annotations

import torch

from rl_games.common.algo_observer import AlgoObserver

from .mdp.curriculums import _compute_step_based_floor


def _find_difficulty_scheduler(env):
    """Find the DifficultyScheduler instance from the env's curriculum manager.

    Uses duck-typing (get_state/set_state) to avoid importing the class directly.
    """
    curriculum_mgr = getattr(env, "curriculum_manager", None)
    if curriculum_mgr is None:
        return None
    cfg = getattr(curriculum_mgr, "cfg", None)
    if cfg is None:
        return None
    adr_cfg = getattr(cfg, "adr", None)
    if adr_cfg is None:
        return None
    func = getattr(adr_cfg, "func", None)
    if func is not None and hasattr(func, "get_state") and hasattr(func, "set_state"):
        return func
    return None


class Y2RCheckpointObserver(AlgoObserver):
    """Observer that persists common_step_counter and ADR state in rl_games checkpoints."""

    def after_init(self, algo):
        env = getattr(getattr(getattr(algo, "vec_env", None), "env", None), "unwrapped", None)
        if env is None or not hasattr(env, "common_step_counter"):
            return

        # Y2R config lives on the env cfg object at runtime (env.cfg.y2r_cfg), not on the env itself.
        # Keep a direct-env fallback for any legacy/debug wrappers that may attach it there.
        y2r_cfg = getattr(env, "y2r_cfg", None)
        if y2r_cfg is None:
            env_cfg = getattr(env, "cfg", None)
            y2r_cfg = getattr(env_cfg, "y2r_cfg", None) if env_cfg is not None else None
        resume_step = y2r_cfg is not None and y2r_cfg.curriculum.scheduler.resume_step

        # Patch save: inject counter + ADR state into checkpoint dict
        _orig_get = algo.get_full_state_weights

        def _get_with_state():
            state = _orig_get()
            state["common_step_counter"] = int(env.common_step_counter)
            scheduler = _find_difficulty_scheduler(env)
            if scheduler is not None:
                state["adr_difficulties"] = scheduler.get_state().cpu()
            return state

        algo.get_full_state_weights = _get_with_state

        # Patch restore: pull counter + ADR state back out of checkpoint dict
        _orig_set = algo.set_full_state_weights

        def _set_with_state(weights, set_epoch=True):
            # Strip custom keys before handing off to rl_games internals.
            weights = dict(weights)
            restored_counter = None

            if "common_step_counter" in weights:
                restored_counter = int(weights.pop("common_step_counter"))
                if resume_step:
                    env.common_step_counter = restored_counter

            adr_difficulties = weights.pop("adr_difficulties", None)

            if resume_step and y2r_cfg is not None:
                scheduler = _find_difficulty_scheduler(env)
                if scheduler is not None:
                    max_difficulty = y2r_cfg.curriculum.difficulty.max

                    # Compute floor for logging (and fallback)
                    counter = restored_counter if restored_counter is not None else 0
                    floor = _compute_step_based_floor(
                        common_step_counter=counter,
                        init_difficulty=y2r_cfg.curriculum.difficulty.initial,
                        max_difficulty=max_difficulty,
                        step_interval=y2r_cfg.curriculum.scheduler.step_interval,
                        level_overrides=y2r_cfg.curriculum.scheduler.level_overrides,
                    )

                    if adr_difficulties is not None and adr_difficulties.shape[0] == env.num_envs:
                        # Exact match: restore per-env difficulties
                        scheduler.set_state(adr_difficulties)
                        num_envs_match = True
                    else:
                        # Missing or num_envs mismatch: fall back to step-based floor
                        scheduler.current_adr_difficulties[:] = floor
                        scheduler._restored = True  # Skip first demotion
                        num_envs_match = False

                    # Recompute difficulty_frac from restored state
                    mean_diff = torch.mean(scheduler.current_adr_difficulties).item()
                    scheduler.difficulty_frac = mean_diff / max(max_difficulty, 1)
                    print(f"[INFO] Restored ADR: mean_difficulty={mean_diff:.2f}, floor={floor}, num_envs_match={num_envs_match}")

            _orig_set(weights, set_epoch=set_epoch)

        algo.set_full_state_weights = _set_with_state
