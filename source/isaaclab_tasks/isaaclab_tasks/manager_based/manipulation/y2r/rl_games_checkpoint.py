# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Persist common_step_counter across RL-Games checkpoint save/resume.

The environment's common_step_counter resets to 0 whenever the env is
re-created (e.g. on --continue). This observer patches the algo's
save/restore methods to include the counter in .pth checkpoints, so
step-based schedules (like ADR's step_based_floor) resume correctly.
"""

from __future__ import annotations

from rl_games.common.algo_observer import AlgoObserver


class Y2RCheckpointObserver(AlgoObserver):
    """Observer that persists common_step_counter in rl_games checkpoints."""

    def after_init(self, algo):
        env = getattr(getattr(getattr(algo, "vec_env", None), "env", None), "unwrapped", None)
        if env is None or not hasattr(env, "common_step_counter"):
            return

        # Patch save: inject counter into checkpoint dict
        _orig_get = algo.get_full_state_weights

        def _get_with_counter():
            state = _orig_get()
            state["common_step_counter"] = int(env.common_step_counter)
            return state

        algo.get_full_state_weights = _get_with_counter

        # Patch restore: pull counter back out of checkpoint dict
        _orig_set = algo.set_full_state_weights

        def _set_with_counter(weights, set_epoch=True):
            if "common_step_counter" in weights:
                # Strip custom key before handing off to rl_games internals.
                weights = dict(weights)
                env.common_step_counter = int(weights.pop("common_step_counter"))
            _orig_set(weights, set_epoch=set_epoch)

        algo.set_full_state_weights = _set_with_counter
