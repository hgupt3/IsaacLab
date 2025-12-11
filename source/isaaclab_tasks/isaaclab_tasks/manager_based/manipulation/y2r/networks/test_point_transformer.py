#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Test script for Cross-Attention Point Network.

Benchmarks forward pass latency and validates output shapes.

Usage:
    cd /home/harsh/sam/IsaacLab
    conda activate y2r
    python source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/y2r/networks/test_point_transformer.py
"""

import torch
import time


def test_sinusoidal_embedding():
    """Test SinusoidalTimeEmbedding class."""
    from point_transformer import SinusoidalTimeEmbedding
    
    print("=" * 60)
    print("SinusoidalTimeEmbedding Test")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dim = 64
    num_timesteps = 10
    
    embed = SinusoidalTimeEmbedding(dim, num_timesteps).to(device)
    
    # Check cached embeddings shape
    print(f"Cached embeddings shape: {embed.pe.shape}")
    assert embed.pe.shape == (num_timesteps, dim), f"Expected ({num_timesteps}, {dim})"
    
    # Test lookup
    timesteps = torch.arange(num_timesteps, device=device)
    out = embed(timesteps)
    print(f"Output shape for all timesteps: {out.shape}")
    assert out.shape == (num_timesteps, dim)
    
    # Test batched lookup
    batch_timesteps = torch.tensor([0, 5, 9, 0, 5], device=device)
    out = embed(batch_timesteps)
    print(f"Output shape for batch: {out.shape}")
    assert out.shape == (5, dim)
    
    # Test that same timestep gives same embedding
    assert torch.allclose(out[0], out[3]), "Same timestep should give same embedding"
    assert torch.allclose(out[1], out[4]), "Same timestep should give same embedding"
    
    print("✓ SinusoidalTimeEmbedding tests passed!")


def test_cross_attention_block():
    """Test CrossAttentionBlock in isolation."""
    from point_transformer import CrossAttentionBlock
    
    print("\n" + "=" * 60)
    print("CrossAttentionBlock Unit Test")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dim = 64
    num_heads = 4
    
    block = CrossAttentionBlock(dim, num_heads).to(device)
    
    # Test forward pass
    B = 256
    query = torch.randn(B, 1, dim, device=device)  # proprio token
    context = torch.randn(B, 320, dim, device=device)  # point tokens
    
    out = block(query, context)
    
    assert out.shape == query.shape, f"Shape mismatch: {out.shape} vs {query.shape}"
    print(f"✓ Query shape: {query.shape} → Output shape: {out.shape}")
    
    # Check residual connection (output should be different from input)
    assert not torch.allclose(out, query), "Output should differ from input after attention"
    print("✓ Residual connection working")


def test_cross_attention_network():
    """Test full Cross-Attention Point Network with synthetic data."""
    from point_transformer import PointTransformerBuilder
    
    print("\n" + "=" * 60)
    print("Cross-Attention Point Network Test")
    print("=" * 60)
    
    # Configuration matching y2r environment
    num_points = 32
    num_timesteps = 10  # 5 history + 5 targets
    hidden_dim = 64
    num_heads = 4
    actions_num = 28  # Eigen grasp action space
    
    # Observation dimensions
    # Point cloud: 32 points × 10 timesteps × 3 xyz = 960
    # Proprio: everything else ≈ 815
    num_point_tokens = num_points * num_timesteps  # 320
    point_cloud_dim = num_point_tokens * 3  # 960
    proprio_dim = 815
    obs_dim = point_cloud_dim + proprio_dim
    
    print(f"\nConfiguration:")
    print(f"  num_points: {num_points}")
    print(f"  num_timesteps: {num_timesteps}")
    print(f"  num_point_tokens: {num_point_tokens}")
    print(f"  hidden_dim: {hidden_dim}")
    print(f"  num_heads: {num_heads}")
    print(f"  actions_num: {actions_num}")
    print(f"  obs_dim: {obs_dim} (point_cloud: {point_cloud_dim}, proprio: {proprio_dim})")
    
    # Build network
    params = {
        'num_points': num_points,
        'num_timesteps': num_timesteps,
        'hidden_dim': hidden_dim,
        'num_heads': num_heads,
        'separate': True,
        'space': {
            'continuous': {
                'fixed_sigma': True,
                'sigma_init': {'val': 0},
            }
        }
    }
    
    builder = PointTransformerBuilder()
    builder.params = params
    
    network = builder.build(
        name='point_transformer',
        input_shape=(obs_dim,),
        actions_num=actions_num,
        value_size=1,
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = network.to(device)
    network.eval()
    
    print(f"\nDevice: {device}")
    
    # Count parameters
    total_params = sum(p.numel() for p in network.parameters())
    trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test batch sizes
    batch_sizes = [64, 256, 1024, 4096, 8192]
    
    print(f"\n{'Batch Size':<12} {'Forward (ms)':<15} {'Throughput (samples/s)':<25}")
    print("-" * 55)
    
    for batch_size in batch_sizes:
        # Create synthetic observation
        obs = torch.randn(batch_size, obs_dim, device=device)
        obs_dict = {'obs': obs}
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = network(obs_dict)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        num_iters = 100
        start = time.perf_counter()
        
        with torch.no_grad():
            for _ in range(num_iters):
                mu, sigma, value, states = network(obs_dict)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = time.perf_counter() - start
        avg_ms = (elapsed / num_iters) * 1000
        throughput = (batch_size * num_iters) / elapsed
        
        print(f"{batch_size:<12} {avg_ms:<15.3f} {throughput:<25,.0f}")
        
        # Validate output shapes
        assert mu.shape == (batch_size, actions_num), f"mu shape mismatch: {mu.shape}"
        assert sigma.shape == (batch_size, actions_num), f"sigma shape mismatch: {sigma.shape}"
        assert value.shape == (batch_size, 1), f"value shape mismatch: {value.shape}"
        assert states is None, f"states should be None for non-RNN: {states}"
    
    print("\n✓ All shape validations passed!")
    
    # Test separate vs unified
    print("\n" + "=" * 60)
    print("Separate vs Unified Cross-Attention Comparison")
    print("=" * 60)
    
    for separate in [True, False]:
        params['separate'] = separate
        builder.params = params
        net = builder.build(
            name='test',
            input_shape=(obs_dim,),
            actions_num=actions_num,
            value_size=1,
        ).to(device).eval()
        
        params_count = sum(p.numel() for p in net.parameters())
        
        obs = torch.randn(4096, obs_dim, device=device)
        obs_dict = {'obs': obs}
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = net(obs_dict)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(100):
                _ = net(obs_dict)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        mode = "Separate" if separate else "Unified"
        print(f"{mode:<10} - Params: {params_count:>10,} - Time: {(elapsed/100)*1000:.3f} ms")
    
    print("\n✓ Cross-Attention Point Network test completed successfully!")


if __name__ == '__main__':
    test_sinusoidal_embedding()
    test_cross_attention_block()
    test_cross_attention_network()
