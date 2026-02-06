#!/usr/bin/env python3
"""Benchmark inference speed: MLP vs Point Transformer networks.

Usage:
    python benchmark_networks.py --num_envs 16384 --num_iters 100
"""

import argparse
import time
import torch
import numpy as np
from typing import Dict
import sys
import os
import importlib.util
import copy

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../' * 6))

# Import network builders
from rl_games.algos_torch.network_builder import A2CBuilder

# Direct file import to bypass isaaclab package
spec = importlib.util.spec_from_file_location(
    "pt_module",
    os.path.join(os.path.dirname(__file__), "../networks/point_transformer.py")
)
pt_module = importlib.util.module_from_spec(spec)
sys.modules['pt_module'] = pt_module
spec.loader.exec_module(pt_module)
PointTransformerBuilder = pt_module.PointTransformerBuilder


class TNet(torch.nn.Module):
    """PointNet T-Net for per-point feature alignment."""

    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(k, 16),
            torch.nn.ELU(),
        )
        self.fc = torch.nn.Linear(16, k * k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, P, k)
        B, _, _ = x.shape
        feat = self.mlp(x).max(dim=1).values  # (B, 32)
        trans = self.fc(feat).view(B, self.k, self.k)
        eye = torch.eye(self.k, device=x.device).unsqueeze(0)
        return trans + eye


class PTPointNetWrapper(torch.nn.Module):
    """PointNet-style encoder with T-Nets and global pooling."""

    def __init__(self, base: torch.nn.Module):
        super().__init__()
        self.base = base
        self.tnet1 = TNet(30)
        self.tnet2 = TNet(64)
        self.mlp1 = torch.nn.Sequential(
            torch.nn.Linear(30, 64),
            torch.nn.ELU(),
        )
        self.mlp2 = torch.nn.Sequential(
            torch.nn.Linear(64, 64),
            torch.nn.ELU(),
        )
        input_dim = self.base.proprio_dim + 64
        self.actor_trunk = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.ELU(),
            torch.nn.Linear(512, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
        )
        self.critic_trunk = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.ELU(),
            torch.nn.Linear(512, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
        )
        self.actor_output = torch.nn.Linear(128, self.base.actions_num)
        self.critic_output = torch.nn.Linear(128, self.base.value_size)
        self.fixed_sigma = self.base.fixed_sigma
        if self.fixed_sigma:
            self.sigma = self.base.sigma
        else:
            self.sigma_head = torch.nn.Linear(128, self.base.actions_num)

    def forward(self, obs_dict: Dict[str, torch.Tensor]):
        obs = obs_dict["obs"]
        B = obs.shape[0]
        point_obs, proprio_obs = self.base._split_obs(obs)

        # Flatten point trajectories: (B, P, T, 3) -> (B, P, 30)
        point_feat = point_obs.reshape(B, self.base.num_points, -1)
        trans1 = self.tnet1(point_feat)
        point_feat = torch.bmm(point_feat, trans1)

        point_feat = self.mlp1(point_feat)  # (B, P, 64)
        trans2 = self.tnet2(point_feat)
        point_feat = torch.bmm(point_feat, trans2)

        point_feat = self.mlp2(point_feat)  # (B, P, 256)
        global_feat = point_feat.max(dim=1).values  # (B, 256)

        trunk_input = torch.cat([proprio_obs, global_feat], dim=-1)
        actor_latent = self.actor_trunk(trunk_input)
        critic_latent = self.critic_trunk(trunk_input)
        mu = self.actor_output(actor_latent)
        value = self.critic_output(critic_latent)

        if self.fixed_sigma:
            sigma = self.sigma.expand(B, -1)
        else:
            sigma = self.sigma_head(actor_latent)

        return mu, sigma, value, None


def load_network_config(config_path: str) -> dict:
    """Load rl_games network config YAML."""
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['params']


def create_mlp_network(obs_dim: int, action_dim: int, config: dict) -> torch.nn.Module:
    """Create standard MLP actor-critic network."""
    # Use standard A2CBuilder
    builder = A2CBuilder()
    builder.load(config)

    network = builder.build(
        'actor_critic',
        input_shape=(obs_dim,),
        actions_num=action_dim,
        value_size=1,
    )
    return network


def create_point_transformer(obs_dim: int, action_dim: int, config: dict) -> torch.nn.Module:
    """Create Point Transformer network."""
    # Manually set params to avoid config_loader dependency
    config['num_points'] = 32
    config['num_timesteps'] = 6

    builder = PointTransformerBuilder()
    builder.params = config  # Skip load() to avoid config_loader

    network = builder.build(
        'point_transformer',
        input_shape=(obs_dim,),
        actions_num=action_dim,
        value_size=1,
    )
    return network


def benchmark_network(
    network: torch.nn.Module,
    obs: torch.Tensor,
    num_iters: int = 100,
    warmup_iters: int = 10,
    device: str = 'cuda:0',
) -> Dict[str, float]:
    """Benchmark network inference speed.

    Returns:
        dict with 'mean_ms', 'std_ms', 'throughput_envs_per_sec', 'memory_mb'
    """
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    if device.startswith("cuda") and not use_cuda:
        device = "cpu"
    network = network.to(device).eval()
    obs = obs.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = network({'obs': obs})

    if use_cuda:
        torch.cuda.synchronize()

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_iters):
            start = time.perf_counter()
            _ = network({'obs': obs})
            if use_cuda:
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

    # Memory
    memory_mb = torch.cuda.max_memory_allocated(device) / 1024**2 if use_cuda else 0.0

    mean_ms = np.mean(times)
    std_ms = np.std(times)
    throughput = (obs.shape[0] / mean_ms) * 1000  # envs/sec

    return {
        'mean_ms': mean_ms,
        'std_ms': std_ms,
        'throughput_envs_per_sec': throughput,
        'memory_mb': memory_mb,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_envs', type=int, default=16384, help='Batch size (parallel environments)')
    parser.add_argument('--num_iters', type=int, default=100, help='Benchmark iterations')
    parser.add_argument('--obs_dim', type=int, default=2890, help='Observation dimension')
    parser.add_argument('--action_dim', type=int, default=23, help='Action dimension')
    parser.add_argument('--mlp_config', type=str,
                        default='IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/y2r/config/agents/rl_games_trajectory_ppo_cfg.yaml')
    parser.add_argument('--pt_config', type=str,
                        default='IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/y2r/config/agents/rl_games_point_transformer_cfg.yaml')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Inference Speed Benchmark: MLP vs Point Transformer")
    print(f"{'='*60}")
    print(f"Batch size: {args.num_envs} environments")
    print(f"Iterations: {args.num_iters}")
    print(f"Observation dim: {args.obs_dim}")
    print(f"Action dim: {args.action_dim}")
    print(f"Device: {args.device}\n")

    # Generate random observations
    obs = torch.randn(args.num_envs, args.obs_dim)

    # Load configs
    mlp_cfg = load_network_config(args.mlp_config)
    pt_cfg = load_network_config(args.pt_config)

    # Create networks
    print("Creating MLP network...")
    mlp_network = create_mlp_network(args.obs_dim, args.action_dim, mlp_cfg['network'])
    mlp_params = sum(p.numel() for p in mlp_network.parameters())
    print(f"  Parameters: {mlp_params:,}")

    # Point Transformer variants
    base_pt_cfg = pt_cfg['network']
    hidden_dim = base_pt_cfg.get("hidden_dim", 64)

    def _pt_cfg(**overrides):
        cfg = copy.deepcopy(base_pt_cfg)
        cfg.update(overrides)
        return cfg

    print("Creating Point Transformer networks...")
    pt_baseline = create_point_transformer(args.obs_dim, args.action_dim, _pt_cfg())
    pt_ffn1 = create_point_transformer(args.obs_dim, args.action_dim, _pt_cfg(ffn_ratio=1.0))
    pt_ffn2 = create_point_transformer(args.obs_dim, args.action_dim, _pt_cfg(ffn_ratio=2.0))
    pt_ffn2_enc2 = create_point_transformer(
        args.obs_dim,
        args.action_dim,
        _pt_cfg(ffn_ratio=2.0, point_encoder_layers=[hidden_dim, hidden_dim]),
    )

    pt_baseline_params = sum(p.numel() for p in pt_baseline.parameters())
    pt_ffn1_params = sum(p.numel() for p in pt_ffn1.parameters())
    pt_ffn2_params = sum(p.numel() for p in pt_ffn2.parameters())
    pt_ffn2_enc2_params = sum(p.numel() for p in pt_ffn2_enc2.parameters())

    print(f"  PT baseline parameters: {pt_baseline_params:,}")
    print(f"  PT ffn1 parameters:     {pt_ffn1_params:,}")
    print(f"  PT ffn2 parameters:     {pt_ffn2_params:,}")
    print(f"  PT ffn2+enc2 params:    {pt_ffn2_enc2_params:,}\n")

    use_cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    if args.device.startswith("cuda") and not use_cuda:
        print("⚠️  CUDA not available; running on CPU.")

    # Benchmark MLP
    print("Benchmarking MLP network...")
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    mlp_stats = benchmark_network(mlp_network, obs, args.num_iters, device=args.device)

    # Benchmark Point Transformer variants
    print("Benchmarking PT baseline...")
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    pt_baseline_stats = benchmark_network(pt_baseline, obs, args.num_iters, device=args.device)

    print("Benchmarking PT ffn1...")
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    pt_ffn1_stats = benchmark_network(pt_ffn1, obs, args.num_iters, device=args.device)

    print("Benchmarking PT ffn2...")
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    pt_ffn2_stats = benchmark_network(pt_ffn2, obs, args.num_iters, device=args.device)

    print("Benchmarking PT ffn2 + 2-layer encoder...")
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    pt_ffn2_enc2_stats = benchmark_network(pt_ffn2_enc2, obs, args.num_iters, device=args.device)

    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}\n")

    print(f"{'Metric':<30} {'MLP':<16} {'PT base':<16} {'PT ffn1':<16} {'PT ffn2':<16} {'PT ffn2+enc2':<16}")
    print(f"{'-'*75}")

    print(
        f"{'Parameters':<30} {mlp_params:<16,} {pt_baseline_params:<16,} {pt_ffn1_params:<16,} "
        f"{pt_ffn2_params:<16,} {pt_ffn2_enc2_params:<16,}"
    )
    print(
        f"{'Mean Latency (ms)':<30} {mlp_stats['mean_ms']:<16.3f} {pt_baseline_stats['mean_ms']:<16.3f} "
        f"{pt_ffn1_stats['mean_ms']:<16.3f} {pt_ffn2_stats['mean_ms']:<16.3f} {pt_ffn2_enc2_stats['mean_ms']:<16.3f}"
    )
    print(
        f"{'Std Latency (ms)':<30} {mlp_stats['std_ms']:<16.3f} {pt_baseline_stats['std_ms']:<16.3f} "
        f"{pt_ffn1_stats['std_ms']:<16.3f} {pt_ffn2_stats['std_ms']:<16.3f} {pt_ffn2_enc2_stats['std_ms']:<16.3f}"
    )
    print(
        f"{'Throughput (envs/sec)':<30} {mlp_stats['throughput_envs_per_sec']:<16,.0f} "
        f"{pt_baseline_stats['throughput_envs_per_sec']:<16,.0f} {pt_ffn1_stats['throughput_envs_per_sec']:<16,.0f} "
        f"{pt_ffn2_stats['throughput_envs_per_sec']:<16,.0f} {pt_ffn2_enc2_stats['throughput_envs_per_sec']:<16,.0f}"
    )
    print(
        f"{'GPU Memory (MB)':<30} {mlp_stats['memory_mb']:<16.1f} {pt_baseline_stats['memory_mb']:<16.1f} "
        f"{pt_ffn1_stats['memory_mb']:<16.1f} {pt_ffn2_stats['memory_mb']:<16.1f} {pt_ffn2_enc2_stats['memory_mb']:<16.1f}"
    )

    print(f"\n{'='*60}\n")

    # Speedup summary
    def _speed_msg(label: str, stats: Dict[str, float]):
        speedup = mlp_stats['mean_ms'] / stats['mean_ms']
        if speedup > 1:
            print(f"✅ {label} is {speedup:.2f}x FASTER than MLP")
        else:
            print(f"⚠️  {label} is {1/speedup:.2f}x SLOWER than MLP")

        if mlp_stats['memory_mb'] > 0:
            memory_ratio = stats['memory_mb'] / mlp_stats['memory_mb']
            if memory_ratio < 1:
                print(f"✅ {label} uses {(1-memory_ratio)*100:.1f}% LESS memory")
            else:
                print(f"⚠️  {label} uses {(memory_ratio-1)*100:.1f}% MORE memory")
        else:
            print(f"ℹ️  {label} memory comparison skipped (CPU run)")

    _speed_msg("PT baseline", pt_baseline_stats)
    _speed_msg("PT ffn1", pt_ffn1_stats)
    _speed_msg("PT ffn2", pt_ffn2_stats)
    _speed_msg("PT ffn2+enc2", pt_ffn2_enc2_stats)

    print()


if __name__ == '__main__':
    main()
