#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark comparison: MLP vs Cross-Attention Point Network.

Compares forward pass latency, parameter count, and throughput between
the standard MLP policy and the Cross-Attention Point Network.

Usage:
    cd /home/harsh/sam/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/y2r/networks
    conda activate y2r
    python benchmark_comparison.py
"""

import torch
import torch.nn as nn
import time
from typing import Dict, Any


class MLPNetwork(nn.Module):
    """Standard MLP network matching rl_games actor_critic.
    
    Architecture: [512, 256, 128] with ELU activation (separate actor/critic).
    """
    
    def __init__(self, obs_dim: int, actions_num: int, hidden_units: list = [512, 256, 128]):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.actions_num = actions_num
        
        # Actor MLP
        actor_layers = []
        in_dim = obs_dim
        for units in hidden_units:
            actor_layers.extend([nn.Linear(in_dim, units), nn.ELU()])
            in_dim = units
        actor_layers.append(nn.Linear(in_dim, actions_num))
        self.actor = nn.Sequential(*actor_layers)
        
        # Critic MLP (separate)
        critic_layers = []
        in_dim = obs_dim
        for units in hidden_units:
            critic_layers.extend([nn.Linear(in_dim, units), nn.ELU()])
            in_dim = units
        critic_layers.append(nn.Linear(in_dim, 1))
        self.critic = nn.Sequential(*critic_layers)
        
        # Sigma
        self.sigma = nn.Parameter(torch.zeros(actions_num))
    
    def forward(self, obs_dict: Dict[str, Any]):
        obs = obs_dict['obs']
        mu = self.actor(obs)
        value = self.critic(obs)
        sigma = self.sigma.expand(obs.shape[0], -1)
        return mu, sigma, value, None


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def benchmark_model(model: nn.Module, obs_dim: int, batch_sizes: list, device: torch.device, 
                    warmup_iters: int = 20, bench_iters: int = 100) -> dict:
    """Benchmark a model across different batch sizes.
    
    Returns dict with timing and throughput for each batch size.
    """
    model = model.to(device).eval()
    results = {}
    
    for batch_size in batch_sizes:
        obs = torch.randn(batch_size, obs_dim, device=device)
        obs_dict = {'obs': obs}
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_iters):
                _ = model(obs_dict)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(bench_iters):
                _ = model(obs_dict)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = time.perf_counter() - start
        avg_ms = (elapsed / bench_iters) * 1000
        throughput = (batch_size * bench_iters) / elapsed
        
        results[batch_size] = {
            'time_ms': avg_ms,
            'throughput': throughput,
        }
    
    return results


def main():
    from point_transformer import PointTransformerBuilder
    
    print("=" * 70)
    print("MLP vs Cross-Attention Point Network Benchmark")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Configuration
    num_points = 32
    num_timesteps = 10
    hidden_dim = 64
    num_heads = 4
    actions_num = 28
    
    # Observation dimensions
    num_point_tokens = num_points * num_timesteps  # 320
    point_cloud_dim = num_point_tokens * 3  # 960
    proprio_dim = 815  # Approximate
    obs_dim = point_cloud_dim + proprio_dim
    
    print(f"\nObservation dim: {obs_dim}")
    print(f"  - Point cloud: {point_cloud_dim} ({num_point_tokens} tokens × 3)")
    print(f"  - Proprio: {proprio_dim}")
    print(f"Action dim: {actions_num}")
    
    # Build MLP model
    mlp_model = MLPNetwork(obs_dim, actions_num, hidden_units=[512, 256, 128])
    mlp_params = count_parameters(mlp_model)
    
    # Build Cross-Attention model
    params = {
        'num_points': num_points,
        'num_timesteps': num_timesteps,
        'hidden_dim': hidden_dim,
        'num_heads': num_heads,
        'separate': True,
        'space': {'continuous': {'fixed_sigma': True, 'sigma_init': {'val': 0}}}
    }
    
    builder = PointTransformerBuilder()
    builder.params = params
    cross_attn_model = builder.build(
        name='cross_attn',
        input_shape=(obs_dim,),
        actions_num=actions_num,
        value_size=1,
    )
    cross_attn_params = count_parameters(cross_attn_model)
    
    print(f"\n{'Model':<25} {'Parameters':>15}")
    print("-" * 42)
    print(f"{'MLP [512, 256, 128]':<25} {mlp_params:>15,}")
    print(f"{'Cross-Attention (64d)':<25} {cross_attn_params:>15,}")
    print(f"{'Ratio (Cross/MLP)':<25} {cross_attn_params/mlp_params:>15.2f}x")
    
    # Benchmark
    batch_sizes = [64, 256, 1024, 4096, 8192, 16384]
    
    print(f"\n{'='*70}")
    print("Forward Pass Latency (ms)")
    print("=" * 70)
    print(f"\n{'Batch':<10} {'MLP':>12} {'CrossAttn':>12} {'Speedup':>12}")
    print("-" * 50)
    
    mlp_results = benchmark_model(mlp_model, obs_dim, batch_sizes, device)
    cross_attn_results = benchmark_model(cross_attn_model, obs_dim, batch_sizes, device)
    
    for bs in batch_sizes:
        mlp_time = mlp_results[bs]['time_ms']
        ca_time = cross_attn_results[bs]['time_ms']
        speedup = mlp_time / ca_time
        print(f"{bs:<10} {mlp_time:>12.3f} {ca_time:>12.3f} {speedup:>11.2f}x")
    
    print(f"\n{'='*70}")
    print("Throughput (samples/second)")
    print("=" * 70)
    print(f"\n{'Batch':<10} {'MLP':>15} {'CrossAttn':>15}")
    print("-" * 45)
    
    for bs in batch_sizes:
        mlp_tp = mlp_results[bs]['throughput']
        ca_tp = cross_attn_results[bs]['throughput']
        print(f"{bs:<10} {mlp_tp:>15,.0f} {ca_tp:>15,.0f}")
    
    # Memory usage
    if device.type == 'cuda':
        print(f"\n{'='*70}")
        print("GPU Memory Usage")
        print("=" * 70)
        
        torch.cuda.reset_peak_memory_stats()
        
        # MLP memory
        mlp_model = mlp_model.to(device)
        obs = torch.randn(4096, obs_dim, device=device)
        with torch.no_grad():
            _ = mlp_model({'obs': obs})
        mlp_mem = torch.cuda.max_memory_allocated() / 1024**2
        
        torch.cuda.reset_peak_memory_stats()
        
        # Cross-attention memory
        cross_attn_model = cross_attn_model.to(device)
        with torch.no_grad():
            _ = cross_attn_model({'obs': obs})
        ca_mem = torch.cuda.max_memory_allocated() / 1024**2
        
        print(f"\nBatch size 4096:")
        print(f"  MLP:         {mlp_mem:>8.1f} MB")
        print(f"  CrossAttn:   {ca_mem:>8.1f} MB")
    
    print(f"\n{'='*70}")
    print("Summary")
    print("=" * 70)
    print(f"""
Hybrid Self+Cross Attention Point Network:
  - Single Linear encoders: point (30→64), proprio (815→64)
  - Layer 1: Self-attention among 32 point tokens (points share info)
  - Layer 2: Cross-attention (proprio queries enriched points)
  - Output: [raw_proprio ({proprio_dim}D) + attn_out ({hidden_dim}D)] → heads
  - {cross_attn_params:,} parameters vs MLP's {mlp_params:,} ({100*cross_attn_params/mlp_params:.1f}%)
    """)


if __name__ == '__main__':
    main()

