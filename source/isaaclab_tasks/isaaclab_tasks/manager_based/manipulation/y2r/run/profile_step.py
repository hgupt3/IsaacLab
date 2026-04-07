"""Profile env.step() breakdown with CUDA-synced timing.

Monkey-patches ManagerBasedRLEnv.step() to measure each section independently,
and observation_manager.compute() to measure each obs group.

Launched via ./y2r_sim/run/benchmark.sh — see that script for usage.
"""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

# ── Args (matches train.py / play.py pattern) ───────────────────────────────
parser = argparse.ArgumentParser(description="Profile env.step() timing breakdown.")
parser.add_argument("--task", type=str, required=True, help="Registered task name.")
parser.add_argument("--num_envs", type=int, default=None, help="Override num_envs from config.")
parser.add_argument("--agent", type=str, default="rl_games_cfg_entry_point",
                    help="Agent config entry point (controls obs groups / network).")
parser.add_argument("--warmup_steps", type=int, default=50, help="Steps before profiling starts.")
parser.add_argument("--profile_steps", type=int, default=200, help="Steps to profile.")
parser.add_argument("--torch_profile", action="store_true",
                    help="Run torch.profiler and save chrome trace to ./profile_traces/.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Clear sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Post-launch imports ─────────────────────────────────────────────────────
import types  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg  # noqa: E402

import isaaclab_tasks  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


# ── CUDA timer ──────────────────────────────────────────────────────────────
class CudaTimer:
    """Accumulates CUDA-synced elapsed times for a named section."""

    def __init__(self, name: str):
        self.name = name
        self.times_ms: list[float] = []
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)

    def start(self):
        self._start.record()

    def stop(self):
        self._end.record()

    def sync_and_log(self):
        self.times_ms.append(self._start.elapsed_time(self._end))

    @property
    def mean_ms(self):
        return sum(self.times_ms) / len(self.times_ms) if self.times_ms else 0.0

    @property
    def std_ms(self):
        if len(self.times_ms) < 2:
            return 0.0
        m = self.mean_ms
        return (sum((t - m) ** 2 for t in self.times_ms) / (len(self.times_ms) - 1)) ** 0.5


# ── Timer registries ────────────────────────────────────────────────────────
step_section_names = [
    "total_step",
    "action_process",
    "physics_loop",
    "  sim.step",
    "  scene.update",
    "  render",
    "termination",
    "reward",
    "obs_compute",
    "reset_handling",
]
step_timers = {name: CudaTimer(name) for name in step_section_names}

# Per-obs-group timers — populated after env creation
obs_group_timers: dict[str, CudaTimer] = {}


def profiled_step(self, action):
    """Drop-in replacement for ManagerBasedRLEnv.step() with CUDA timers."""
    t = step_timers

    t["total_step"].start()

    # 1. Action processing
    t["action_process"].start()
    self.action_manager.process_action(action.to(self.device))
    self.recorder_manager.record_pre_step()
    t["action_process"].stop()

    # 2. Physics loop
    is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
    t["physics_loop"].start()
    for _ in range(self.cfg.decimation):
        self._sim_step_counter += 1
        self.action_manager.apply_action()
        self.scene.write_data_to_sim()

        t["  sim.step"].start()
        self.sim.step(render=False)
        t["  sim.step"].stop()

        self.recorder_manager.record_post_physics_decimation_step()

        t["  render"].start()
        if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
            self.sim.render()
        t["  render"].stop()

        t["  scene.update"].start()
        self.scene.update(dt=self.physics_dt)
        t["  scene.update"].stop()
    t["physics_loop"].stop()

    # 3. Post-step accounting
    self.episode_length_buf += 1
    self.common_step_counter += 1

    t["termination"].start()
    self.reset_buf = self.termination_manager.compute()
    self.reset_terminated = self.termination_manager.terminated
    self.reset_time_outs = self.termination_manager.time_outs
    t["termination"].stop()

    t["reward"].start()
    self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
    t["reward"].stop()

    if len(self.recorder_manager.active_terms) > 0:
        self.obs_buf = self.observation_manager.compute()
        self.recorder_manager.record_post_step()

    # 4. Resets
    t["reset_handling"].start()
    reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    if len(reset_env_ids) > 0:
        self.recorder_manager.record_pre_reset(reset_env_ids)
        self._reset_idx(reset_env_ids)
        if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
            for _ in range(self.cfg.num_rerenders_on_reset):
                self.sim.render()
        self.recorder_manager.record_post_reset(reset_env_ids)
    t["reset_handling"].stop()

    # 5. Observations (per-group timing inside patched compute)
    self.command_manager.compute(dt=self.step_dt)
    if "interval" in self.event_manager.available_modes:
        self.event_manager.apply(mode="interval", dt=self.step_dt)

    t["obs_compute"].start()
    self.obs_buf = self.observation_manager.compute(update_history=True)
    t["obs_compute"].stop()

    t["total_step"].stop()

    return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras


def make_profiled_compute_group(original_compute_group):
    """Wrap ObservationManager.compute_group to time each group."""
    def profiled_compute_group(group_name, update_history=False):
        t = obs_group_timers.get(group_name)
        if t is not None:
            t.start()
        result = original_compute_group(group_name, update_history=update_history)
        if t is not None:
            t.stop()
        return result
    return profiled_compute_group


# ── Main (uses hydra_task_config like train.py / play.py) ────────────────────
@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Profile env.step() with CUDA-synced section timing."""
    # CLI overrides (only num_envs — everything else comes from config)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    # Create environment (identical to train.py)
    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped

    y2r_cfg = unwrapped.cfg.y2r_cfg
    mode = os.environ.get("Y2R_MODE", "?")
    task = os.environ.get("Y2R_TASK", "?")

    # Discover obs groups and create per-group timers
    obs_mgr = unwrapped.observation_manager
    group_names = list(obs_mgr._group_obs_term_names.keys())
    for gn in group_names:
        obs_group_timers[gn] = CudaTimer(gn)

    print(f"\n{'='*60}")
    print(f"  BENCHMARK: {args_cli.task}")
    print(f"{'='*60}")
    print(f"  mode         = {mode}")
    print(f"  task         = {task}")
    print(f"  num_envs     = {unwrapped.num_envs}")
    print(f"  decimation   = {unwrapped.cfg.decimation}")
    print(f"  physics_dt   = {unwrapped.physics_dt}")
    print(f"  step_dt      = {unwrapped.step_dt}")
    print(f"  device       = {unwrapped.device}")
    print(f"  student_mode = {y2r_cfg.mode.use_student_mode}")
    print(f"  point_cloud  = {y2r_cfg.mode.use_point_cloud}")
    print(f"  obs_groups   = {group_names}")
    print(f"  warmup_steps = {args_cli.warmup_steps}")
    print(f"  profile_steps= {args_cli.profile_steps}")
    print(f"{'='*60}\n")

    action_dim = unwrapped.action_manager.total_action_dim
    device = unwrapped.device

    # ── Warmup ───────────────────────────────────────────────────────────────
    print(f"Warming up ({args_cli.warmup_steps} steps)...")
    env.reset()
    for _ in range(args_cli.warmup_steps):
        action = torch.randn(unwrapped.num_envs, action_dim, device=device)
        env.step(action)
    torch.cuda.synchronize()
    print("Warmup done.\n")

    # ── Patch step + obs compute ─────────────────────────────────────────────
    original_step = unwrapped.step.__func__
    unwrapped.step = types.MethodType(profiled_step, unwrapped)

    original_compute_group = obs_mgr.compute_group
    obs_mgr.compute_group = make_profiled_compute_group(original_compute_group)

    # ── Profiled run ─────────────────────────────────────────────────────────
    print(f"Profiling ({args_cli.profile_steps} steps)...")
    for _ in range(args_cli.profile_steps):
        action = torch.randn(unwrapped.num_envs, action_dim, device=device)
        env.step(action)
        torch.cuda.synchronize()
        for t in step_timers.values():
            t.sync_and_log()
        for t in obs_group_timers.values():
            t.sync_and_log()

    # ── Results ──────────────────────────────────────────────────────────────
    total_mean = step_timers["total_step"].mean_ms
    obs_mean = step_timers["obs_compute"].mean_ms

    print(f"\n{'='*60}")
    print(f"  ENV STEP PROFILING RESULTS  ({args_cli.profile_steps} steps)")
    print(f"{'='*60}")
    print(f"  {'Section':<28} {'Mean (ms)':>10} {'Std (ms)':>10} {'% of step':>10}")
    print(f"  {'-'*28} {'-'*10} {'-'*10} {'-'*10}")

    for name in step_section_names:
        t = step_timers[name]
        pct = (t.mean_ms / total_mean * 100) if total_mean > 0 else 0
        indent = "  " if name.startswith("  ") else ""
        label = name.strip()
        print(f"  {indent}{label:<26} {t.mean_ms:>10.2f} {t.std_ms:>10.2f} {pct:>9.1f}%")

    # Per-obs-group breakdown
    print(f"\n  {'Obs Group Breakdown':<28} {'Mean (ms)':>10} {'Std (ms)':>10} {'% of obs':>10}")
    print(f"  {'-'*28} {'-'*10} {'-'*10} {'-'*10}")
    for gn in group_names:
        t = obs_group_timers[gn]
        pct_obs = (t.mean_ms / obs_mean * 100) if obs_mean > 0 else 0
        print(f"    {gn:<26} {t.mean_ms:>10.2f} {t.std_ms:>10.2f} {pct_obs:>9.1f}%")

    fps = 1000.0 / total_mean if total_mean > 0 else 0
    sps = fps * unwrapped.num_envs
    print(f"\n  Step rate: {fps:.1f} Hz  |  {sps:,.0f} samples/sec")
    print(f"{'='*60}\n")

    # ── Optional: torch.profiler chrome trace ────────────────────────────────
    if args_cli.torch_profile:
        trace_steps = 20
        print(f"Running torch.profiler ({trace_steps} steps)...")
        unwrapped.step = types.MethodType(original_step, unwrapped)
        obs_mgr.compute_group = original_compute_group

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=True,
            schedule=torch.profiler.schedule(wait=2, warmup=3, active=trace_steps - 5),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./profile_traces"),
        ) as prof:
            for _ in range(trace_steps):
                action = torch.randn(unwrapped.num_envs, action_dim, device=device)
                env.step(action)
                prof.step()

        print("\nTop 20 CUDA kernels by total time:")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        print("\nChrome trace saved to ./profile_traces/")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
