# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Cross-Attention Point Network for rl_games.

This module implements a self+cross-attention policy that processes point cloud
observations across time.

Architecture:
    - 320 point tokens: 32 points × 10 timesteps (5 history + 5 targets)
    - Each point: xyz → MLP → 64-dim
    - 1 proprio token: all other features → MLP → 64-dim (query)
    - Self-attention on point tokens
    - Cross-attention: proprio queries point tokens
    - Optional lightweight FFN on tokens
    - Policy/Value heads from updated proprio token
"""

from __future__ import annotations

import contextlib
import torch
import torch.nn as nn
from typing import Dict, Any

from rl_games.algos_torch.network_builder import A2CBuilder, NetworkBuilder


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


class FeedForwardBlock(nn.Module):
    """Lightweight FFN block with pre-norm and residual."""

    def __init__(self, dim: int, ratio: float, dropout: float = 0.0):
        super().__init__()
        self.ratio = ratio
        if ratio <= 0:
            self.enabled = False
            return

        self.enabled = True
        hidden_dim = int(dim * ratio)
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.act = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        y = self.fc1(self.norm(x))
        y = self.act(y)
        y = self.dropout(y)
        y = self.fc2(y)
        return x + self.dropout(y)


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

        def load(self, params: Dict[str, Any]):
            """Load network configuration from params."""
            self.separate = params.get("separate", False)
            mlp_cfg = params.get("mlp", {})
            self.units = mlp_cfg.get("units", [])
            self.activation = mlp_cfg.get("activation", "elu")
            self.initializer = mlp_cfg.get("initializer", {"name": "default"})
            self.is_d2rl = mlp_cfg.get("d2rl", False)
            self.norm_only_first_layer = mlp_cfg.get("norm_only_first_layer", False)
            self.normalization = params.get("normalization", None)
            self.value_activation = params.get("value_activation", "None")
            self.attn_dropout = params.get("attn_dropout", 0.0)
            self.use_self_attention = params.get("self_attention", True)
            self.use_flash_attention = params.get("flash_attention", True)
            self.ffn_ratio = params.get("ffn_ratio", 0.0)
            self.ffn_dropout = params.get("ffn_dropout", 0.0)
            self.point_encoder_layers = params.get("point_encoder_layers", None)
            self.point_encoder_norm = params.get("point_encoder_norm", False)
            self.proprio_encoder_norm = params.get("proprio_encoder_norm", False)
            self.proprio_encoder_layers = params.get("proprio_encoder_layers", None)
            self.partial_separate = params.get("partial_separate", False)
            if self.separate and self.partial_separate:
                self.partial_separate = False

            self.has_space = "space" in params
            if self.has_space:
                self.is_multi_discrete = "multi_discrete" in params["space"]
                self.is_discrete = "discrete" in params["space"]
                self.is_continuous = "continuous" in params["space"]
                if self.is_continuous:
                    self.space_config = params["space"]["continuous"]
                    fixed_sigma = self.space_config.get("fixed_sigma", True)
                    if fixed_sigma is True:
                        fixed_sigma = "fixed"
                    elif fixed_sigma is False:
                        fixed_sigma = "obs_cond"
                    self.fixed_sigma = fixed_sigma
                elif self.is_discrete:
                    self.space_config = params["space"]["discrete"]
                elif self.is_multi_discrete:
                    self.space_config = params["space"]["multi_discrete"]
            else:
                self.is_continuous = True
                self.is_discrete = False
                self.is_multi_discrete = False
                self.fixed_sigma = "fixed"
                self.space_config = {
                    "mu_activation": "None",
                    "sigma_activation": "None",
                    "mu_init": {"name": "default"},
                    "sigma_init": {"name": "const_initializer", "val": 0},
                }

        def __init__(self, params, **kwargs):
            """Initialize Cross-Attention Point Network.
            
            Args:
                params: Network parameters from YAML config
                kwargs: Additional arguments including:
                    - input_shape: Total observation dimension
                    - actions_num: Number of action dimensions
                    - value_size: Value output size (usually 1)
            """
            NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)

            self.params = params
            self.actions_num = kwargs.get("actions_num")
            self.value_size = kwargs.get("value_size", 1)
            input_shape = kwargs.get("input_shape")
            self.input_obs_dim = input_shape[0] if isinstance(input_shape, (tuple, list)) else input_shape
            self.net_type = kwargs.get("type", "simple")
            self.coef_id_idx = kwargs.get("coef_id_idx", None)

            # Architecture params (from base.yaml via builder)
            self.num_points = params["num_points"]
            self.num_timesteps = params["num_timesteps"]
            self.hidden_dim = params["hidden_dim"]
            self.num_heads = params["num_heads"]

            # Point cloud dimension: num_points × num_timesteps × 3
            self.point_cloud_dim = self.num_points * self.num_timesteps * 3
            self.base_obs_dim = self.input_obs_dim
            self.extra_tail_dim = 0
            if self.coef_id_idx is not None:
                self.base_obs_dim = self.coef_id_idx
                self.extra_tail_dim = self.input_obs_dim - self.coef_id_idx
            self.base_proprio_dim = self.base_obs_dim - self.point_cloud_dim

            # Validate dimensions
            if self.base_proprio_dim < 0:
                raise ValueError(
                    f"Observation dim ({self.base_obs_dim}) is smaller than point cloud dim "
                    f"({self.point_cloud_dim}). Check num_points and num_timesteps."
                )

            if self.net_type == "extra_param":
                self.param_ids = kwargs["coef_ids"]
                self.pid_idx = kwargs["coef_id_idx"]
                self.param_size = kwargs.get("param_size", 32)
                self.extra_params = nn.Parameter(
                    torch.randn((len(self.param_ids), self.param_size), dtype=torch.float32, requires_grad=True),
                    requires_grad=True,
                )
                self.proprio_dim = self.base_proprio_dim + self.param_size
            else:
                self.proprio_dim = self.base_proprio_dim + self.extra_tail_dim
            self.obs_dim = self.proprio_dim + self.point_cloud_dim
            
            # Build components
            # Point encoder: Conv1d temporal encoder
            self.point_encoder = self._build_point_encoder()
            # Proprio encoder: single linear layer
            self.proprio_encoder = self._build_proprio_encoder()
            self.proprio_norm = nn.LayerNorm(self.hidden_dim) if self.proprio_encoder_norm else None

            # Self-attention on point tokens (optional), then cross-attention
            if self.use_self_attention:
                self.self_attn = SelfAttentionBlock(self.hidden_dim, self.num_heads, dropout=self.attn_dropout)
                self.self_ffn = FeedForwardBlock(self.hidden_dim, self.ffn_ratio, dropout=self.ffn_dropout)
            self.cross_attn = CrossAttentionBlock(self.hidden_dim, self.num_heads, dropout=self.attn_dropout)

            # If separate, also build separate encoders/attention for critic (A2C parity)
            if self.separate:
                self.critic_point_encoder = self._build_point_encoder()
                self.critic_proprio_encoder = self._build_proprio_encoder()
                if self.use_self_attention:
                    self.critic_self_attn = SelfAttentionBlock(self.hidden_dim, self.num_heads, dropout=self.attn_dropout)
                    self.critic_self_ffn = FeedForwardBlock(self.hidden_dim, self.ffn_ratio, dropout=self.ffn_dropout)
                self.critic_cross_attn = CrossAttentionBlock(self.hidden_dim, self.num_heads, dropout=self.attn_dropout)

            # Build trunks and output layers
            trunk_input_dim = (self.proprio_dim + self.hidden_dim) if self.proprio_dim > 0 else self.hidden_dim
            self.actor_trunk, trunk_output_dim = self._build_trunk(trunk_input_dim)
            if self.separate or self.partial_separate:
                self.critic_trunk, _ = self._build_trunk(trunk_input_dim)
            else:
                self.critic_trunk = self.actor_trunk

            self.actor_output = nn.Linear(trunk_output_dim, self.actions_num)
            self.critic_output = nn.Linear(trunk_output_dim, self.value_size)
            self.value_act = self.activations_factory.create(self.value_activation)

            # Sigma (log std) for continuous actions
            self.mu_act = self.activations_factory.create(self.space_config.get("mu_activation", "None"))
            self.sigma_act = self.activations_factory.create(self.space_config.get("sigma_activation", "None"))

            if self.fixed_sigma == "fixed":
                self.sigma = nn.Parameter(
                    torch.zeros(self.actions_num, requires_grad=True, dtype=torch.float32),
                    requires_grad=True,
                )
            elif self.fixed_sigma == "coef_cond":
                self.sigma_ids = kwargs["coef_ids"]
                self.sigma_id_idx = kwargs["coef_id_idx"]
                self.sigma = nn.Parameter(
                    torch.zeros(len(self.sigma_ids), self.actions_num, requires_grad=True, dtype=torch.float32),
                    requires_grad=True,
                )
            else:
                self.sigma_head = nn.Linear(trunk_output_dim, self.actions_num)

            # Initialize weights
            self._init_weights()

        def _build_point_encoder(self) -> nn.Module:
            """Build linear encoder for point timeseries.

            Input: (B, num_points, num_timesteps * 3)
            Output: (B, num_points, hidden_dim)
            """
            input_dim = self.num_timesteps * 3
            layers = self.point_encoder_layers
            if not layers:
                layers = [self.hidden_dim]
            if layers[-1] != self.hidden_dim:
                raise ValueError(
                    f"point_encoder_layers must end with hidden_dim ({self.hidden_dim}), got {layers[-1]}"
                )

            modules = []
            in_dim = input_dim
            for i, out_dim in enumerate(layers):
                modules.append(nn.Linear(in_dim, out_dim))
                if i < len(layers) - 1:
                    modules.append(self.activations_factory.create(self.activation))
                in_dim = out_dim
            if self.point_encoder_norm:
                modules.append(nn.LayerNorm(self.hidden_dim))
            if len(modules) == 1:
                return modules[0]
            return nn.Sequential(*modules)

        def _build_proprio_encoder(self) -> nn.Module:
            """Build single linear encoder for proprioceptive features."""
            if self.proprio_dim == 0:
                return nn.Identity()
            layers = self.proprio_encoder_layers
            if not layers:
                layers = [self.hidden_dim]
            if layers[-1] != self.hidden_dim:
                raise ValueError(
                    f"proprio_encoder_layers must end with hidden_dim ({self.hidden_dim}), got {layers[-1]}"
                )

            modules = []
            in_dim = self.proprio_dim
            for i, out_dim in enumerate(layers):
                modules.append(nn.Linear(in_dim, out_dim))
                if i < len(layers) - 1:
                    modules.append(self.activations_factory.create(self.activation))
                in_dim = out_dim
            if len(modules) == 1:
                return modules[0]
            return nn.Sequential(*modules)

        def _build_trunk(self, input_dim: int) -> tuple[nn.Module, int]:
            """Build MLP trunk and return (module, output_dim)."""
            if len(self.units) == 0:
                return nn.Identity(), input_dim
            if self.is_d2rl:
                return (
                    self._build_mlp(
                        input_size=input_dim,
                        units=self.units,
                        activation=self.activation,
                        dense_func=nn.Linear,
                        norm_only_first_layer=self.norm_only_first_layer,
                        norm_func_name=self.normalization,
                        d2rl=True,
                    ),
                    self.units[-1],
                )

            layers = []
            in_size = input_dim
            need_norm = True
            for unit in self.units:
                layers.append(nn.Linear(in_size, unit))
                layers.append(self.activations_factory.create(self.activation))
                if need_norm and self.normalization:
                    if self.normalization == "layer_norm":
                        layers.append(nn.LayerNorm(unit))
                    elif self.normalization == "batch_norm":
                        layers.append(nn.BatchNorm1d(unit))
                    if self.norm_only_first_layer:
                        need_norm = False
                in_size = unit
            return nn.Sequential(*layers), self.units[-1]

        def _init_weights(self):
            """Initialize network weights."""
            mlp_init = self.init_factory.create(**self.initializer)
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    mlp_init(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

            if self.is_continuous:
                mu_init = self.init_factory.create(**self.space_config.get("mu_init", {"name": "default"}))
                sigma_init = self.init_factory.create(
                    **self.space_config.get("sigma_init", {"name": "const_initializer", "val": 0})
                )
                mu_init(self.actor_output.weight)
                if self.fixed_sigma in ["fixed", "coef_cond"]:
                    sigma_init(self.sigma)
                else:
                    sigma_init(self.sigma_head.weight)

        def _match_coef_indices(self, obs_ids: torch.Tensor, coef_ids: torch.Tensor) -> torch.Tensor:
            if obs_ids.dtype != coef_ids.dtype:
                obs_ids = obs_ids.to(coef_ids.dtype)
            distances = torch.abs(obs_ids.unsqueeze(1) - coef_ids.unsqueeze(0))
            return torch.argmin(distances, dim=1)

        def _prepare_obs(self, obs: torch.Tensor) -> torch.Tensor:
            # Keep point-cloud features contiguous at the tail for deterministic splitting.
            if self.net_type == "extra_param":
                coef_indices = self._match_coef_indices(obs[:, self.pid_idx], self.param_ids)
                base_obs = obs[:, :self.pid_idx]
                base_proprio = base_obs[:, :self.base_proprio_dim]
                point_flat = base_obs[:, self.base_proprio_dim:]
                return torch.cat([base_proprio, self.extra_params[coef_indices], point_flat], dim=1)
            if self.extra_tail_dim > 0:
                base_obs = obs[:, :self.base_obs_dim]
                tail = obs[:, self.base_obs_dim:]
                base_proprio = base_obs[:, :self.base_proprio_dim]
                point_flat = base_obs[:, self.base_proprio_dim:]
                return torch.cat([base_proprio, tail, point_flat], dim=1)
            return obs

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
                point_obs: (B, num_points, num_timesteps * 3) — ready for encoder
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

            # Reshape to (B, num_points, num_timesteps * 3) — no 4D reshape needed,
            # nn.Linear works on arbitrary leading dims so encoder handles (B, P, F) directly.
            point_obs = point_flat.reshape(B, self.num_points, self.num_timesteps * 3)

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
            raw_obs = obs_dict['obs']
            obs = self._prepare_obs(raw_obs)
            B = obs.shape[0]
            
            # Split observation
            point_obs, proprio_obs = self._split_obs(obs)  # (B, num_points, num_timesteps*3), (B, proprio_dim)

            # Encode points: (B, num_points, num_timesteps*3) → (B, num_points, hidden_dim)
            # nn.Linear works on arbitrary leading dims — no reshape needed.
            point_tokens = self.point_encoder(point_obs)
            
            # Encode proprio: (B, proprio_dim) → (B, 1, hidden_dim)
            if self.proprio_dim > 0:
                proprio_encoded = self.proprio_encoder(proprio_obs)  # (B, hidden_dim)
                if self.proprio_norm is not None:
                    proprio_encoded = self.proprio_norm(proprio_encoded)
                proprio_token = proprio_encoded.unsqueeze(1)  # (B, 1, hidden_dim)
            else:
                proprio_token = torch.zeros(B, 1, self.hidden_dim, device=obs.device)
            
            # Self-attention over point tokens (if enabled), then cross-attention from proprio
            # flash_attention: False uses math backend to bypass CUDA grid dim limit (batch > 65535)
            attn_ctx = (torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH)
                        if not self.use_flash_attention
                        else contextlib.nullcontext())
            with attn_ctx:
                if self.use_self_attention:
                    point_tokens = self.self_attn(point_tokens)
                    point_tokens = self.self_ffn(point_tokens)
                attn_out = self.cross_attn(proprio_token, point_tokens)  # (B, 1, hidden_dim)
            attn_features = attn_out[:, 0]  # (B, hidden_dim)

            # Concat raw proprio + attention output
            if self.proprio_dim > 0:
                trunk_input = torch.cat([proprio_obs, attn_features], dim=-1)  # (B, proprio_dim + hidden_dim)
            else:
                trunk_input = attn_features

            # Trunks
            actor_latent = self.actor_trunk(trunk_input)
            if self.separate:
                # Recompute critic features with separate encoders/attention
                critic_point_tokens = self.critic_point_encoder(point_obs)
                with attn_ctx:
                    if self.use_self_attention:
                        critic_point_tokens = self.critic_self_attn(critic_point_tokens)
                        critic_point_tokens = self.critic_self_ffn(critic_point_tokens)
                    if self.proprio_dim > 0:
                        critic_proprio_encoded = self.critic_proprio_encoder(proprio_obs)
                        if self.proprio_norm is not None:
                            critic_proprio_encoded = self.proprio_norm(critic_proprio_encoded)
                        critic_proprio_token = critic_proprio_encoded.unsqueeze(1)
                    else:
                        critic_proprio_token = torch.zeros(B, 1, self.hidden_dim, device=obs.device)
                    critic_attn_out = self.critic_cross_attn(critic_proprio_token, critic_point_tokens)
                critic_attn_features = critic_attn_out[:, 0]
                if self.proprio_dim > 0:
                    critic_trunk_input = torch.cat([proprio_obs, critic_attn_features], dim=-1)
                else:
                    critic_trunk_input = critic_attn_features
                critic_latent = self.critic_trunk(critic_trunk_input)
            elif self.partial_separate:
                if self.proprio_dim > 0:
                    critic_trunk_input = torch.cat([proprio_obs, attn_features.detach()], dim=-1)
                else:
                    critic_trunk_input = attn_features.detach()
                critic_latent = self.critic_trunk(critic_trunk_input)
            else:
                critic_latent = actor_latent

            # Outputs
            mu = self.mu_act(self.actor_output(actor_latent))
            value = self.value_act(self.critic_output(critic_latent))

            # Sigma
            if self.fixed_sigma == "fixed":
                sigma = self.sigma_act(self.sigma)
            elif self.fixed_sigma == "coef_cond":
                sigma_indices = self._match_coef_indices(raw_obs[:, self.sigma_id_idx], self.sigma_ids)
                sigma = self.sigma_act(self.sigma[sigma_indices])
            else:
                sigma = self.sigma_act(self.sigma_head(actor_latent))

            return mu, mu * 0 + sigma, value, None

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

        # Calculate num_timesteps: current history + future trajectory history
        num_current = y2r_cfg.observations.history.object_pc
        num_future = y2r_cfg.trajectory.window_size * y2r_cfg.observations.history.targets
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

        # Enable TF32 for matmuls outside AMP autocast (observation pipeline, etc.)
        torch.backends.cuda.matmul.allow_tf32 = True

        net = self.Network(self.params, **kwargs)
        if self.params.get("torch_compile", False):
            # torch.compile fuses Linear, LayerNorm, attention, and reshapes into
            # optimized Triton kernels. "reduce-overhead" uses CUDA graphs for minimal
            # launch overhead; dynamic=True handles varying batch sizes.
            net = torch.compile(net, mode="reduce-overhead", dynamic=True)
        return net
