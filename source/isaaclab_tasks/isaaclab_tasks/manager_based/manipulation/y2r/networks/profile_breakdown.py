#!/usr/bin/env python3
"""Profile Hybrid Self+Cross Attention Point Network.

Architecture:
  - Layer 1: Self-attention among 32 point tokens
  - Layer 2: Cross-attention (proprio queries enriched points)
"""

import torch
import torch.nn as nn
import time


def profile_component(name, fn, warmup=10, iters=100, device='cuda'):
    """Profile a function and return average time in ms."""
    for _ in range(warmup):
        _ = fn()
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(iters):
        _ = fn()
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    return (time.perf_counter() - start) / iters * 1000


def main():
    from point_transformer import PointTransformerBuilder, SelfAttentionBlock, CrossAttentionBlock
    
    device = torch.device('cuda')
    batch_size = 4096
    
    num_points = 32
    num_timesteps = 10
    hidden_dim = 64
    num_heads = 4
    proprio_dim = 815
    
    print("=" * 60)
    print(f"Profiling Hybrid Self+Cross Attention (batch={batch_size})")
    print("=" * 60)
    
    # Create components
    point_encoder = nn.Linear(num_timesteps * 3, hidden_dim).to(device)
    proprio_encoder = nn.Linear(proprio_dim, hidden_dim).to(device)
    
    point_self_attn = SelfAttentionBlock(hidden_dim, num_heads).to(device)
    cross_attn = CrossAttentionBlock(hidden_dim, num_heads).to(device)
    
    actor_head = nn.Sequential(
        nn.Linear(proprio_dim + hidden_dim, 256),
        nn.ELU(),
        nn.Linear(256, 28),
    ).to(device)
    
    # Create inputs
    point_obs_4d = torch.randn(batch_size, num_points, num_timesteps, 3, device=device)
    proprio_obs = torch.randn(batch_size, proprio_dim, device=device)
    
    # Pre-compute intermediates
    with torch.no_grad():
        point_flat = point_obs_4d.reshape(batch_size * num_points, num_timesteps * 3)
        point_tokens = point_encoder(point_flat).view(batch_size, num_points, hidden_dim)
        proprio_token = proprio_encoder(proprio_obs).unsqueeze(1)
        enriched_points = point_self_attn(point_tokens)
    
    print(f"\nInput shapes:")
    print(f"  point_obs_4d:     {tuple(point_obs_4d.shape)}")
    print(f"  proprio_obs:      {tuple(proprio_obs.shape)}")
    print(f"  point_tokens:     {tuple(point_tokens.shape)} (32 point tokens)")
    print(f"  proprio_token:    {tuple(proprio_token.shape)} (1 proprio token)")
    print(f"  enriched_points:  {tuple(enriched_points.shape)} (after self-attn)")
    
    print(f"\n{'Component':<40} {'Time (ms)':>12} {'% of Total':>12}")
    print("-" * 66)
    
    times = {}
    
    with torch.no_grad():
        # Point encoder
        def encode_points():
            x = point_obs_4d.reshape(batch_size * num_points, num_timesteps * 3)
            return point_encoder(x).view(batch_size, num_points, hidden_dim)
        times['point_encoder (Linear 30→64)'] = profile_component('', encode_points)
        
        # Proprio encoder
        times['proprio_encoder (Linear 815→64)'] = profile_component(
            '', lambda: proprio_encoder(proprio_obs))
        
        # Self-attention among 32 points
        times['point_self_attn (32×32)'] = profile_component(
            '', lambda: point_self_attn(point_tokens))
        
        # Cross-attention (1 query, 32 keys)
        times['cross_attn (1→32)'] = profile_component(
            '', lambda: cross_attn(proprio_token, enriched_points))
        
        # Actor head
        attn_out = cross_attn(proprio_token, enriched_points)[:, 0]
        actor_input = torch.cat([proprio_obs, attn_out], dim=-1)
        times['actor_head'] = profile_component('', lambda: actor_head(actor_input))
    
    total = sum(times.values())
    
    for name, t in times.items():
        pct = 100 * t / total
        print(f"  {name:<38} {t:>10.3f} ms {pct:>10.1f}%")
    
    print("-" * 66)
    print(f"  {'TOTAL':<38} {total:>10.3f} ms {100.0:>10.1f}%")
    
    # Full model
    print(f"\n{'='*60}")
    print("Full Model Comparison")
    print("=" * 60)
    
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
    obs_dim = num_points * num_timesteps * 3 + proprio_dim
    
    full_model = builder.build(
        name='test',
        input_shape=(obs_dim,),
        actions_num=28,
        value_size=1,
    ).to(device).eval()
    
    obs = torch.randn(batch_size, obs_dim, device=device)
    obs_dict = {'obs': obs}
    
    with torch.no_grad():
        full_time = profile_component('Full Model', lambda: full_model(obs_dict))
    
    print(f"\n  Full model forward: {full_time:.3f} ms")
    print(f"  Sum of components:  {total:.3f} ms")
    print(f"  Overhead:           {full_time - total:.3f} ms")
    
    # MLP comparison
    print(f"\n{'='*60}")
    print("MLP Comparison")
    print("=" * 60)
    
    mlp = nn.Sequential(
        nn.Linear(obs_dim, 512),
        nn.ELU(),
        nn.Linear(512, 256),
        nn.ELU(),
        nn.Linear(256, 128),
        nn.ELU(),
        nn.Linear(128, 28),
    ).to(device)
    
    with torch.no_grad():
        mlp_time = profile_component('MLP', lambda: mlp(obs))
    
    print(f"\n  MLP forward:        {mlp_time:.3f} ms")
    print(f"  Point Net forward:  {full_time:.3f} ms")
    print(f"  Ratio:              {full_time/mlp_time:.1f}x slower")
    
    # Attention analysis
    print(f"\n{'='*60}")
    print("Attention Complexity Analysis")
    print("=" * 60)
    print(f"""
    Layer 1 (Self-attention): 32×32 = 1,024 attention ops per sample
    Layer 2 (Cross-attention): 1×32 = 32 attention ops per sample
    Total: 1,056 attention ops per sample
    
    For batch {batch_size}: {batch_size * 1056:,} attention ops total
    """)


if __name__ == '__main__':
    main()
