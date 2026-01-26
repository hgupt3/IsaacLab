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


class PointNetTNet(nn.Module):
    """PointNet-style T-Net with a small hidden layer."""

    def __init__(self, k: int, hidden_dim: int = 16):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Linear(k, hidden_dim),
            nn.ELU(),
        )
        self.fc = nn.Linear(hidden_dim, k * k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, P, k)
        B, _, _ = x.shape
        feat = self.mlp(x).max(dim=1).values  # (B, hidden_dim)
        trans = self.fc(feat).view(B, self.k, self.k)
        eye = torch.eye(self.k, device=x.device).unsqueeze(0)
        return trans + eye


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
        self.params = None  # Will be set by load()

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
            
            # Architecture params (from base.yaml via builder)
            self.num_points = params['num_points']
            self.num_timesteps = params['num_timesteps']
            self.hidden_dim = params['hidden_dim']
            self.num_heads = params['num_heads']

            # Point cloud dimension: num_points × num_timesteps × 3
            self.point_cloud_dim = self.num_points * self.num_timesteps * 3
            self.proprio_dim = self.obs_dim - self.point_cloud_dim

            # Validate dimensions
            if self.proprio_dim < 0:
                raise ValueError(
                    f"Observation dim ({self.obs_dim}) is smaller than point cloud dim "
                    f"({self.point_cloud_dim}). Check num_points and num_timesteps."
                )
            
            # Build components
            # Point encoder: Conv1d temporal encoder
            self.point_encoder = self._build_point_encoder()
            # Proprio encoder: single linear layer
            self.proprio_encoder = self._build_proprio_encoder()

            # Cross-attention: shared between actor/critic (separate heads after)
            self.cross_attn = CrossAttentionBlock(self.hidden_dim, self.num_heads)

            # Build separate trunks and output layers (match MLP style)
            self.actor_trunk = self._build_actor_trunk()
            self.critic_trunk = self._build_critic_trunk()
            self.actor_output = self._build_actor_output()
            self.critic_output = self._build_critic_output()
            
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
                self.sigma_head = nn.Linear(128, self.actions_num)
            
            # Initialize weights
            self._init_weights()

        def _build_point_encoder(self) -> nn.Module:
            """Build linear encoder for point timeseries.

            Input: (B * num_points, num_timesteps * 3)
            Output: (B * num_points, hidden_dim)
            """
            input_dim = self.num_timesteps * 3
            return nn.Linear(input_dim, self.hidden_dim)

        def _build_proprio_encoder(self) -> nn.Module:
            """Build single linear encoder for proprioceptive features."""
            if self.proprio_dim == 0:
                return nn.Identity()
            return nn.Linear(self.proprio_dim, self.hidden_dim)

        def _build_actor_trunk(self) -> nn.Module:
            """Build actor trunk: [raw_proprio + attn_out] → hidden.

            Input is concat of raw proprio (proprio_dim) + cross-attn output (hidden_dim).
            MLP: (proprio_dim + hidden_dim) → 512 → 256 → 128.
            If no proprio features, input is just (hidden_dim).
            """
            input_dim = (self.proprio_dim + self.hidden_dim) if self.proprio_dim > 0 else self.hidden_dim
            return nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ELU(),
                nn.Linear(512, 256),
                nn.ELU(),
                nn.Linear(256, 128),
                nn.ELU(),
            )

        def _build_critic_trunk(self) -> nn.Module:
            """Build critic trunk: [raw_proprio + attn_out] → hidden.

            Input is concat of raw proprio (proprio_dim) + cross-attn output (hidden_dim).
            MLP: (proprio_dim + hidden_dim) → 512 → 256 → 128.
            If no proprio features, input is just (hidden_dim).
            """
            input_dim = (self.proprio_dim + self.hidden_dim) if self.proprio_dim > 0 else self.hidden_dim
            return nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ELU(),
                nn.Linear(512, 256),
                nn.ELU(),
                nn.Linear(256, 128),
                nn.ELU(),
            )

        def _build_actor_output(self) -> nn.Module:
            """Build actor output layer: 128 → actions."""
            return nn.Linear(128, self.actions_num)

        def _build_critic_output(self) -> nn.Module:
            """Build critic output layer: 128 → value."""
            return nn.Linear(128, self.value_size)

        def _init_weights(self):
            """Initialize network weights."""
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=1.0)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        def _split_obs(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Split flat observation into point clouds and extended proprio.

            Observation structure (from obs_groups):
                [policy | proprio | current_poses | future_poses | current_pc | future_pc]

            Network grouping:
                - Extended proprio: policy + proprio + current_poses + future_poses
                - Point clouds: current_pc + future_pc (concatenated along time dimension)

            Args:
                obs: Flat observation tensor (B, obs_dim)

            Returns:
                point_obs: (B, num_points, num_timesteps, 3)
                extended_proprio: (B, proprio_dim) - includes policy, proprio, and all poses
            """
            B = obs.shape[0]
            obs_dim = obs.shape[1]
            if obs_dim != self.proprio_dim + self.point_cloud_dim:
                raise ValueError(
                    "PointTransformer obs split mismatch: expected obs_dim="
                    f"{self.proprio_dim + self.point_cloud_dim} but got {obs_dim}. "
                    "Check obs_groups ordering so point clouds are last and sized "
                    f"{self.point_cloud_dim}."
                )

            # Split: extended proprio first, point clouds last
            extended_proprio = obs[:, :self.proprio_dim]  # (B, proprio_dim)
            point_flat = obs[:, self.proprio_dim:]        # (B, point_cloud_dim)

            # Reshape point clouds to (B, num_timesteps, num_points, 3)
            # Then permute to (B, num_points, num_timesteps, 3)
            point_obs = point_flat.view(B, self.num_timesteps, self.num_points, 3)
            point_obs = point_obs.permute(0, 2, 1, 3)  # (B, num_points, num_timesteps, 3)

            return point_obs, extended_proprio

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
            point_obs, proprio_obs = self._split_obs(obs)  # (B, num_points, num_timesteps, 3), (B, proprio_dim)

            # Flatten point timeseries: (B, num_points, num_timesteps, 3) → (B*num_points, num_timesteps*3)
            point_flat = point_obs.reshape(B * self.num_points, self.num_timesteps * 3)

            # Encode points: (B*num_points, num_timesteps*3) → Linear → (B*num_points, hidden_dim)
            point_encoded = self.point_encoder(point_flat)

            # Reshape to point tokens: (B*num_points, hidden_dim) → (B, num_points, hidden_dim)
            point_tokens = point_encoded.view(B, self.num_points, self.hidden_dim)
            
            # Encode proprio: (B, proprio_dim) → (B, 1, hidden_dim)
            if self.proprio_dim > 0:
                proprio_encoded = self.proprio_encoder(proprio_obs)  # (B, hidden_dim)
                proprio_token = proprio_encoded.unsqueeze(1)  # (B, 1, hidden_dim)
            else:
                proprio_token = torch.zeros(B, 1, self.hidden_dim, device=obs.device)
            
            # Cross-attention: shared between actor/critic
            attn_out = self.cross_attn(proprio_token, point_tokens)  # (B, 1, hidden_dim)
            attn_features = attn_out[:, 0]  # (B, hidden_dim)

            # Concat raw proprio + attention output
            if self.proprio_dim > 0:
                trunk_input = torch.cat([proprio_obs, attn_features], dim=-1)  # (B, proprio_dim + hidden_dim)
            else:
                trunk_input = attn_features

            # Separate trunks
            actor_latent = self.actor_trunk(trunk_input)   # (B, 128)
            critic_latent = self.critic_trunk(trunk_input)  # (B, 128)

            # Separate output layers
            mu = self.actor_output(actor_latent)    # (B, actions_num)
            value = self.critic_output(critic_latent)  # (B, value_size)

            # Sigma
            if self.fixed_sigma:
                sigma = self.sigma.expand(B, -1)
            else:
                sigma = self.sigma_head(actor_latent)
            
            return mu, sigma, value, None

        def is_rnn(self) -> bool:
            """Return False - this is not a recurrent network."""
            return False

    def _inject_params(self, params):
        """Inject num_points and num_timesteps from base.yaml into params.

        Args:
            params: Network parameters dict to inject into
        """
        from ..config_loader import get_config
        import os

        # Get Y2R config to extract observation params
        mode = os.getenv('Y2R_MODE')
        y2r_cfg = get_config(mode)

        # Inject observation params into network params
        params['num_points'] = y2r_cfg.observations.num_points

        # Calculate num_timesteps: current history + future trajectory
        num_current = y2r_cfg.observations.history.perception
        num_future = y2r_cfg.trajectory.window_size
        params['num_timesteps'] = num_current + num_future

    def load(self, params):
        """Load parameters (called by rl_games).

        Injects num_points and num_timesteps from environment config (base.yaml).
        """
        self._inject_params(params)
        self.params = params

    def build(self, name: str, **kwargs) -> Network:
        """Build and return the Cross-Attention Point Network.

        Args:
            name: Network name (unused)
            **kwargs: Arguments passed to Network.__init__

        Returns:
            Configured Network instance
        """
        # Ensure params are loaded (in case build() called before load())
        if self.params is None:
            self.params = {}
            self._inject_params(self.params)

        return self.Network(self.params, **kwargs)


class PointNetTNetBuilder(A2CBuilder):
    """Builder for PointNet+TNet encoder with standard actor/critic trunks."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.params = None

    class Network(A2CBuilder.Network):
        """PointNet-style network with two T-Nets and global max pooling."""

        def __init__(self, params, **kwargs):
            nn.Module.__init__(self)
            self.params = params
            self.actions_num = kwargs.get("actions_num")
            self.value_size = kwargs.get("value_size", 1)
            input_shape = kwargs.get("input_shape")
            self.obs_dim = input_shape[0] if isinstance(input_shape, (tuple, list)) else input_shape

            self.num_points = params["num_points"]
            self.num_timesteps = params["num_timesteps"]

            self.point_cloud_dim = self.num_points * self.num_timesteps * 3
            self.proprio_dim = self.obs_dim - self.point_cloud_dim
            if self.proprio_dim < 0:
                raise ValueError(
                    f"Observation dim ({self.obs_dim}) is smaller than point cloud dim "
                    f"({self.point_cloud_dim}). Check num_points and num_timesteps."
                )

            self.tnet1 = PointNetTNet(30, hidden_dim=16)
            self.tnet2 = PointNetTNet(32, hidden_dim=16)
            self.mlp1 = nn.Sequential(
                nn.Linear(30, 32),
                nn.ELU(),
            )
            self.mlp2 = nn.Sequential(
                nn.Linear(32, 64),
                nn.ELU(),
            )

            trunk_input_dim = self.proprio_dim + 64
            self.actor_trunk = nn.Sequential(
                nn.Linear(trunk_input_dim, 512),
                nn.ELU(),
                nn.Linear(512, 256),
                nn.ELU(),
                nn.Linear(256, 128),
                nn.ELU(),
            )
            self.critic_trunk = nn.Sequential(
                nn.Linear(trunk_input_dim, 512),
                nn.ELU(),
                nn.Linear(512, 256),
                nn.ELU(),
                nn.Linear(256, 128),
                nn.ELU(),
            )
            self.actor_output = nn.Linear(128, self.actions_num)
            self.critic_output = nn.Linear(128, self.value_size)

            space_config = params.get("space", {}).get("continuous", {})
            self.fixed_sigma = space_config.get("fixed_sigma", True)
            if self.fixed_sigma:
                sigma_init_val = space_config.get("sigma_init", {}).get("val", 0)
                self.sigma = nn.Parameter(torch.zeros(self.actions_num) + sigma_init_val, requires_grad=True)
            else:
                self.sigma_head = nn.Linear(128, self.actions_num)

            self._init_weights()

        def _init_weights(self):
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=1.0)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        def _split_obs(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            B = obs.shape[0]
            obs_dim = obs.shape[1]
            if obs_dim != self.proprio_dim + self.point_cloud_dim:
                raise ValueError(
                    "PointNet obs split mismatch: expected obs_dim="
                    f"{self.proprio_dim + self.point_cloud_dim} but got {obs_dim}."
                )
            proprio_obs = obs[:, :self.proprio_dim]
            point_flat = obs[:, self.proprio_dim:]
            point_obs = point_flat.view(B, self.num_points, self.num_timesteps, 3)
            return point_obs, proprio_obs

        def forward(self, obs_dict: Dict[str, Any]):
            obs = obs_dict["obs"]
            B = obs.shape[0]
            point_obs, proprio_obs = self._split_obs(obs)

            point_feat = point_obs.reshape(B, self.num_points, -1)  # (B, P, 30)
            trans1 = self.tnet1(point_feat)
            point_feat = torch.bmm(point_feat, trans1)

            point_feat = self.mlp1(point_feat)  # (B, P, 64)
            trans2 = self.tnet2(point_feat)
            point_feat = torch.bmm(point_feat, trans2)

            point_feat = self.mlp2(point_feat)  # (B, P, 64)
            global_feat = point_feat.max(dim=1).values  # (B, 64)

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

        def is_rnn(self) -> bool:
            return False

    def _inject_params(self, params):
        from ..config_loader import get_config
        import os

        mode = os.getenv("Y2R_MODE")
        y2r_cfg = get_config(mode)
        params["num_points"] = y2r_cfg.observations.num_points
        num_current = y2r_cfg.observations.history.perception
        num_future = y2r_cfg.trajectory.window_size
        params["num_timesteps"] = num_current + num_future

    def load(self, params):
        self._inject_params(params)
        self.params = params

    def build(self, name: str, **kwargs) -> Network:
        if self.params is None:
            self.params = {}
            self._inject_params(self.params)
        return self.Network(self.params, **kwargs)
