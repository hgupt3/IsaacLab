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


def _resolve_y2r_cfg(env):
    """Return the Y2R config attached to either the env or its cfg, else None."""
    y2r_cfg = getattr(env, "y2r_cfg", None)
    if y2r_cfg is None:
        env_cfg = getattr(env, "cfg", None)
        y2r_cfg = getattr(env_cfg, "y2r_cfg", None) if env_cfg is not None else None
    return y2r_cfg


def restore_curriculum_state(env, weights, *, force: bool = False) -> dict:
    """Apply persisted common_step_counter + ADR scheduler state to the env in-place.

    Shared between the rl_games AlgoObserver (for train/distill) and play/eval scripts,
    which load checkpoints via BasePlayer.restore() and never go through the observer.

    Pops the custom keys from ``weights`` so the dict is safe to hand to rl_games' own
    set_full_state_weights afterwards. Returns a small dict describing what happened so
    the caller can print or assert.

    Args:
        env: ManagerBasedRLEnv (the unwrapped env). Must expose ``common_step_counter``,
            ``num_envs``, and a ``curriculum_manager`` with an ``adr`` term.
        weights: Loaded checkpoint dict (as produced by torch.load on the .pth file).
            Modified in place — custom keys are popped.
        force: If True, ignore ``y2r_cfg.curriculum.scheduler.resume_step`` and apply the
            restoration regardless. Used by play/eval where the user explicitly opted into
            ``--continue`` and the resume_step toggle is for train-time semantics.

    Returns:
        dict with keys: ``restored_counter``, ``mean_difficulty``, ``floor``, ``num_envs_match``.
        Any may be None if not applied.
    """
    info: dict = {
        "restored_counter": None,
        "mean_difficulty": None,
        "floor": None,
        "num_envs_match": None,
    }

    y2r_cfg = _resolve_y2r_cfg(env)
    apply = force or (y2r_cfg is not None and y2r_cfg.curriculum.scheduler.resume_step)

    restored_counter = None
    if "common_step_counter" in weights:
        restored_counter = int(weights.pop("common_step_counter"))
        if apply:
            env.common_step_counter = restored_counter
            info["restored_counter"] = restored_counter

    adr_difficulties = weights.pop("adr_difficulties", None)

    if apply and y2r_cfg is not None:
        scheduler = _find_difficulty_scheduler(env)
        if scheduler is not None:
            max_difficulty = y2r_cfg.curriculum.difficulty.max
            counter = restored_counter if restored_counter is not None else 0
            floor = _compute_step_based_floor(
                common_step_counter=counter,
                init_difficulty=y2r_cfg.curriculum.difficulty.initial,
                max_difficulty=max_difficulty,
                step_interval=y2r_cfg.curriculum.scheduler.step_interval,
                level_overrides=y2r_cfg.curriculum.scheduler.level_overrides,
            )

            if adr_difficulties is not None and adr_difficulties.shape[0] == env.num_envs:
                scheduler.set_state(adr_difficulties)
                num_envs_match = True
            elif adr_difficulties is not None:
                # Shape mismatch (e.g. play with --num_envs 1 vs trained at 2048).
                # The saved tensor IS the ground truth — broadcast its rounded mean
                # to all current envs so play matches the curriculum the trainer
                # left off at, instead of dropping to the step-based floor.
                pinned = round(float(adr_difficulties.float().mean().item()))
                scheduler.current_adr_difficulties[:] = pinned
                scheduler._restored = True  # Skip first demotion
                num_envs_match = False
            else:
                scheduler.current_adr_difficulties[:] = floor
                scheduler._restored = True  # Skip first demotion
                num_envs_match = False

            mean_diff = torch.mean(scheduler.current_adr_difficulties).item()
            scheduler.difficulty_frac = mean_diff / max(max_difficulty, 1)
            info.update(
                mean_difficulty=mean_diff,
                floor=floor,
                num_envs_match=num_envs_match,
            )

    return info


def restore_adr_from_teacher(env, teacher_ckpt_path: str) -> dict:
    """Seed the student env's DifficultyScheduler with the teacher checkpoint's ADR state.

    Used by distillation: the standard rl_games restore path reads the STUDENT checkpoint,
    not the teacher's, so teacher ADR has to be plumbed in separately. The teacher and
    student typically have different num_envs, so we broadcast the teacher's mean per-env
    difficulty (rounded) to all student envs and recompute difficulty_frac.

    Note on student ``--continue``: distill.py calls this every launch, but rl_games' patched
    ``set_full_state_weights`` runs ``restore_curriculum_state`` later in the load path. On
    resume, the student checkpoint's saved ADR (which was already a frozen copy) overwrites
    the teacher pin. By design — the student-checkpoint-wins-on-resume keeps long-running
    distillations resumable without re-coupling to a possibly-moved teacher .pth.

    Fails fast (raises) if:
      - teacher .pth has no ``adr_difficulties`` (would otherwise pin frozen distillation at 0)
      - the env has no ``y2r_cfg`` attached (we need ``curriculum.difficulty.max``)
      - the curriculum manager has no DifficultyScheduler (no place to restore to)

    Args:
        env: ManagerBasedRLEnv (unwrapped). Must expose ``curriculum_manager`` and ``y2r_cfg``.
        teacher_ckpt_path: Path to the teacher's .pth file.

    Returns:
        dict with keys: ``mean_difficulty`` (float), ``num_teacher_envs`` (int),
        ``num_student_envs`` (int).
    """
    weights = torch.load(teacher_ckpt_path, map_location="cpu", weights_only=False)
    adr_difficulties = weights.get("adr_difficulties")
    if adr_difficulties is None:
        raise RuntimeError(
            f"Teacher checkpoint {teacher_ckpt_path} has no 'adr_difficulties' key. "
            "Distillation with frozen ADR requires teacher state to pin to. "
            "Re-train the teacher with the current checkpoint observer or pick a newer .pth."
        )

    scheduler = _find_difficulty_scheduler(env)
    if scheduler is None:
        raise RuntimeError(
            "Curriculum manager has no DifficultyScheduler — cannot restore teacher ADR. "
            "Verify the env was constructed with the y2r curriculum config."
        )

    y2r_cfg = _resolve_y2r_cfg(env)
    if y2r_cfg is None:
        raise RuntimeError(
            "Env has no y2r_cfg attached — cannot resolve curriculum.difficulty.max. "
            "This function must be called after the env config is fully attached."
        )
    max_difficulty = y2r_cfg.curriculum.difficulty.max

    # Broadcast teacher's mean to every student env. Round to int because difficulty
    # is treated as integer levels by downstream logic.
    teacher_mean = float(adr_difficulties.float().mean().item())
    pinned = round(teacher_mean)
    scheduler.current_adr_difficulties[:] = pinned
    scheduler._restored = True  # Skip first demotion after env reset (no-op while freeze=True)
    scheduler.difficulty_frac = torch.mean(scheduler.current_adr_difficulties) / max(max_difficulty, 1)

    return {
        "mean_difficulty": teacher_mean,
        "num_teacher_envs": int(adr_difficulties.shape[0]),
        "num_student_envs": env.num_envs,
    }


class Y2RCheckpointObserver(AlgoObserver):
    """Observer that persists common_step_counter and ADR state in rl_games checkpoints."""

    def after_init(self, algo):
        env = getattr(getattr(getattr(algo, "vec_env", None), "env", None), "unwrapped", None)
        if env is None or not hasattr(env, "common_step_counter"):
            return

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
            weights = dict(weights)  # local copy — we strip custom keys
            info = restore_curriculum_state(env, weights)
            if info["mean_difficulty"] is not None:
                print(
                    f"[INFO] Restored ADR: mean_difficulty={info['mean_difficulty']:.2f}, "
                    f"floor={info['floor']}, num_envs_match={info['num_envs_match']}"
                )
            _orig_set(weights, set_epoch=set_epoch)

        algo.set_full_state_weights = _set_with_state
