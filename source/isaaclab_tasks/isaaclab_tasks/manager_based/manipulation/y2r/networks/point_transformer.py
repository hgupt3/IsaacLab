# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Cross-Attention Point Network for rl_games.

This module implements a cross-attention policy that processes point cloud
observations across time with sinusoidal time embeddings.

Architecture:
    - 320 point tokens: 32 points × 10 timesteps (5 history + 5 targets)
    - Each point: xyz → MLP → 64-dim → + sinusoidal_time_embed
    - 1 proprio token: all other features → MLP → 64-dim (query)
    - Cross-attention: proprio queries point tokens
    - Policy/Value heads from updated proprio token
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from typing import Dict, Any

from rl_games.algos_torch.network_builder import A2CBuilder


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional encoding for timesteps (cached).
    
    Precomputes embeddings for fixed number of timesteps and stores them
    as a buffer for efficient lookup during forward pass.
    """
    
    def __init__(self, dim: int, num_timesteps: int):
        """Initialize sinusoidal time embeddings.
        
        Args:
            dim: Embedding dimension (must be even)
            num_timesteps: Number of timesteps to precompute (e.g., 10)
        """
        super().__init__()
        assert dim % 2 == 0, f"Embedding dim must be even, got {dim}"
        
        self.dim = dim
        self.num_timesteps = num_timesteps
        
        # Precompute embeddings: (num_timesteps, dim)
        pe = self._compute_embeddings(num_timesteps, dim)
        self.register_buffer('pe', pe)  # Cached, no gradients
    
    def _compute_embeddings(self, num_timesteps: int, dim: int) -> torch.Tensor:
        """Compute sinusoidal embeddings for all timesteps.
        
        Args:
            num_timesteps: Number of timesteps
            dim: Embedding dimension
            
        Returns:
            Tensor of shape (num_timesteps, dim)
        """
        position = torch.arange(num_timesteps).unsqueeze(1).float()  # (T, 1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )  # (dim/2,)
        
        pe = torch.zeros(num_timesteps, dim)
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        
        return pe
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Look up precomputed embeddings for given timesteps.
        
        Args:
            timesteps: Tensor of timestep indices (any shape, values 0 to num_timesteps-1)
            
        Returns:
            Embeddings with same shape as timesteps + (dim,)
        """
        return self.pe[timesteps]


class SelfAttentionBlock(nn.Module):
    """Self-attention block using Flash Attention.
    
    All tokens attend to all other tokens. Uses pre-norm and residual connections.
    """
    
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        
        self.norm = nn.LayerNorm(dim)
        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Self-attention: (B, S, dim) → (B, S, dim)"""
        B, S, _ = x.shape
        
        x_norm = self.norm(x)
        qkv = self.qkv_proj(x_norm).view(B, S, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, S, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        dropout_p = self.dropout if self.training else 0.0
        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, self.dim)
        
        return x + self.out_proj(attn_out)


class CrossAttentionBlock(nn.Module):
    """Cross-attention block using Flash Attention.
    
    Query tokens attend to context tokens. Uses pre-norm and residual connections.
    """
    
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
    
    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Cross-attention: query (B, Q, dim) attends to context (B, C, dim)"""
        B, Q, _ = query.shape
        _, C, _ = context.shape
        
        q = self.q_proj(self.norm_q(query))
        kv = self.norm_kv(context)
        k = self.k_proj(kv)
        v = self.v_proj(kv)
        
        q = q.view(B, Q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, C, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, C, self.num_heads, self.head_dim).transpose(1, 2)
        
        dropout_p = self.dropout if self.training else 0.0
        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, Q, self.dim)
        
        return query + self.out_proj(attn_out)


class PointTransformerBuilder(A2CBuilder):
    """Builder for Hybrid Self+Cross Attention Point Network.
    
    Architecture:
        1. Self-attention among 32 point tokens (points share spatial info)
        2. Cross-attention: proprio queries the enriched point tokens
    
    YAML Config Example:
        network:
            name: point_transformer
            separate: True
            num_points: 32
            num_timesteps: 10
            hidden_dim: 64
            num_heads: 4
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    class Network(A2CBuilder.Network):
        """Hybrid Self+Cross Attention Point Network for actor-critic RL.
        
        Architecture:
            1. Split obs → point cloud (B, 32, 10, 3) and proprio (B, D)
            2. Point encoder: (B*32, 30) → Linear → 32 tokens (B, 32, 64)
            3. Proprio encoder: (B, D) → Linear → 1 token (B, 1, 64)
            4. Self-attention: 32 point tokens share spatial info
            5. Cross-attention: proprio queries enriched points
            6. Concat [raw_proprio, attn_out] → heads
        """

        def __init__(self, params, **kwargs):
            """Initialize Cross-Attention Point Network.
            
            Args:
                params: Network parameters from YAML config
                kwargs: Additional arguments including:
                    - input_shape: Total observation dimension
                    - actions_num: Number of action dimensions
                    - value_size: Value output size (usually 1)
            """
            nn.Module.__init__(self)
            
            # Extract config
            self.params = params
            self.actions_num = kwargs.get('actions_num')
            self.value_size = kwargs.get('value_size', 1)
            input_shape = kwargs.get('input_shape')
            self.obs_dim = input_shape[0] if isinstance(input_shape, (tuple, list)) else input_shape
            
            # Architecture params
            self.num_points = params.get('num_points', 32)
            self.num_timesteps = params.get('num_timesteps', 10)  # 5 history + 5 targets
            self.hidden_dim = params.get('hidden_dim', 64)
            self.num_heads = params.get('num_heads', 4)
            self.separate = params.get('separate', True)
            
            # Total point tokens: num_points × num_timesteps
            self.num_point_tokens = self.num_points * self.num_timesteps  # 320
            
            # Point cloud dimension: each point is xyz (3D)
            self.point_cloud_dim = self.num_point_tokens * 3  # 960
            self.proprio_dim = self.obs_dim - self.point_cloud_dim
            
            # Validate dimensions
            if self.proprio_dim < 0:
                raise ValueError(
                    f"Observation dim ({self.obs_dim}) is smaller than point cloud dim "
                    f"({self.point_cloud_dim} = {self.num_point_tokens} tokens × 3). "
                    f"Check num_points and num_timesteps settings."
                )
            
            # Build components
            # Point encoder: single linear layer per point
            self.point_encoder = self._build_point_encoder()
            # Proprio encoder: single linear layer  
            self.proprio_encoder = self._build_proprio_encoder()
            
            # Layer 1: Self-attention among 32 point tokens (shared for actor/critic)
            self.point_self_attn = SelfAttentionBlock(self.hidden_dim, self.num_heads)
            
            # Layer 2: Cross-attention (proprio queries enriched points)
            if self.separate:
                self.actor_cross_attn = CrossAttentionBlock(self.hidden_dim, self.num_heads)
                self.critic_cross_attn = CrossAttentionBlock(self.hidden_dim, self.num_heads)
            else:
                self.shared_cross_attn = CrossAttentionBlock(self.hidden_dim, self.num_heads)
            
            # Build heads
            self.actor_head = self._build_actor_head()
            self.critic_head = self._build_critic_head()
            
            # Sigma (log std) for continuous actions
            space_config = params.get('space', {}).get('continuous', {})
            self.fixed_sigma = space_config.get('fixed_sigma', True)
            
            if self.fixed_sigma:
                sigma_init_val = space_config.get('sigma_init', {}).get('val', 0)
                self.sigma = nn.Parameter(
                    torch.zeros(self.actions_num) + sigma_init_val,
                    requires_grad=True
                )
            else:
                self.sigma_head = nn.Linear(self.proprio_dim + self.hidden_dim, self.actions_num)
            
            # Initialize weights
            self._init_weights()

        def _build_point_encoder(self) -> nn.Module:
            """Build single linear encoder for each point's flattened timeseries.
            
            Input: (B * num_points, num_timesteps * 3) = (B*32, 30)
            Output: (B * num_points, hidden_dim) = (B*32, 64)
            """
            input_dim = self.num_timesteps * 3  # 10 × 3 = 30
            return nn.Linear(input_dim, self.hidden_dim)

        def _build_proprio_encoder(self) -> nn.Module:
            """Build single linear encoder for proprioceptive features."""
            if self.proprio_dim == 0:
                return nn.Identity()
            return nn.Linear(self.proprio_dim, self.hidden_dim)

        def _build_actor_head(self) -> nn.Module:
            """Build policy head: [raw_proprio + attn_out] → actions.
            
            Input is concat of raw proprio (proprio_dim) + cross-attn output (hidden_dim).
            MLP: (proprio_dim + hidden_dim) → 256 → actions.
            """
            input_dim = self.proprio_dim + self.hidden_dim
            return nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ELU(),
                nn.Linear(256, self.actions_num),
            )

        def _build_critic_head(self) -> nn.Module:
            """Build value head: [raw_proprio + attn_out] → value.
            
            Input is concat of raw proprio (proprio_dim) + cross-attn output (hidden_dim).
            MLP: (proprio_dim + hidden_dim) → 256 → value.
            """
            input_dim = self.proprio_dim + self.hidden_dim
            return nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ELU(),
                nn.Linear(256, self.value_size),
            )

        def _init_weights(self):
            """Initialize network weights."""
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=1.0)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        def _split_obs(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Split flat observation into point cloud and proprio.
            
            The observation is structured as:
                [policy | proprio | perception | targets]
            
            We extract:
                - Point cloud: xyz across all timesteps (perception + targets)
                - Proprio: everything else (policy actions, joint pos/vel, etc.)
            
            Args:
                obs: Flat observation tensor (B, obs_dim)
                
            Returns:
                point_obs: (B, num_points, num_timesteps, 3) - 32 points × 10 timesteps
                proprio_obs: (B, proprio_dim)
            """
            B = obs.shape[0]
            
            # Split: proprio is first, point cloud is last
            proprio_obs = obs[:, :self.proprio_dim]  # (B, proprio_dim)
            point_flat = obs[:, self.proprio_dim:]   # (B, num_point_tokens * 3)
            
            # Reshape: (B, 32 points, 10 timesteps, 3 xyz)
            # Data is organized as [t0_p0, t0_p1, ..., t0_p31, t1_p0, ..., t9_p31]
            # We want (B, points, timesteps, xyz) for conv over time
            point_obs = point_flat.view(B, self.num_timesteps, self.num_points, 3)
            point_obs = point_obs.permute(0, 2, 1, 3)  # (B, 32, 10, 3)
            
            return point_obs, proprio_obs

        def forward(self, obs_dict: Dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            """Forward pass through Hybrid Self+Cross Attention Point Network.
            
            Args:
                obs_dict: Dictionary with 'obs' key containing observations
                
            Returns:
                mu: Action means (B, actions_num)
                sigma: Action stds (B, actions_num) 
                value: Value estimate (B, value_size)
                states: RNN states (None for non-recurrent)
            """
            obs = obs_dict['obs']
            B = obs.shape[0]
            
            # Split observation
            point_obs, proprio_obs = self._split_obs(obs)  # (B, 32, 10, 3), (B, proprio_dim)
            
            # Flatten timeseries: (B, 32, 10, 3) → (B*32, 30)
            point_flat = point_obs.reshape(B * self.num_points, self.num_timesteps * 3)
            
            # Encode points: (B*32, 30) → Linear → (B*32, hidden_dim)
            point_encoded = self.point_encoder(point_flat)
            
            # Reshape to 32 tokens: (B*32, hidden_dim) → (B, 32, hidden_dim)
            point_tokens = point_encoded.view(B, self.num_points, self.hidden_dim)
            
            # Encode proprio: (B, proprio_dim) → (B, 1, hidden_dim)
            if self.proprio_dim > 0:
                proprio_encoded = self.proprio_encoder(proprio_obs)  # (B, hidden_dim)
                proprio_token = proprio_encoded.unsqueeze(1)  # (B, 1, hidden_dim)
            else:
                proprio_token = torch.zeros(B, 1, self.hidden_dim, device=obs.device)
            
            # Layer 1: Self-attention among 32 point tokens (shared)
            # Points share spatial information with each other
            enriched_points = self.point_self_attn(point_tokens)  # (B, 32, hidden_dim)
            
            # Layer 2: Cross-attention (proprio queries enriched points)
            if self.separate:
                actor_out = self.actor_cross_attn(proprio_token, enriched_points)   # (B, 1, hidden_dim)
                critic_out = self.critic_cross_attn(proprio_token, enriched_points)  # (B, 1, hidden_dim)
                
                actor_attn = actor_out[:, 0]   # (B, hidden_dim)
                critic_attn = critic_out[:, 0]  # (B, hidden_dim)
            else:
                out = self.shared_cross_attn(proprio_token, enriched_points)
                actor_attn = out[:, 0]
                critic_attn = out[:, 0]
            
            # Concat RAW proprio + attention output for heads
            actor_latent = torch.cat([proprio_obs, actor_attn], dim=-1)  # (B, proprio_dim + hidden_dim)
            critic_latent = torch.cat([proprio_obs, critic_attn], dim=-1)  # (B, proprio_dim + hidden_dim)
            
            # Heads
            mu = self.actor_head(actor_latent)
            value = self.critic_head(critic_latent)
            
            # Sigma
            if self.fixed_sigma:
                sigma = self.sigma.expand(B, -1)
            else:
                sigma = self.sigma_head(actor_latent)
            
            return mu, sigma, value, None

        def is_rnn(self) -> bool:
            """Return False - this is not a recurrent network."""
            return False

        def load(self, params):
            """Load parameters (called by rl_games)."""
            self.params = params

    def build(self, name: str, **kwargs) -> Network:
        """Build and return the Cross-Attention Point Network.
        
        Args:
            name: Network name (unused)
            **kwargs: Arguments passed to Network.__init__
            
        Returns:
            Configured Network instance
        """
        return self.Network(self.params, **kwargs)
