# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2024, Harsh Gupta
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Hybrid Depth ResNet + Point Transformer Student Network for rl_games.

This module implements a student policy that combines:
1. SmallResNet depth encoder for wrist camera images
2. Point Transformer (self-attention + cross-attention) for visible point clouds
3. MLP trunk for action/value outputs

Architecture:
    - Depth (last H×W of obs) → SmallResNet → depth_features
    - Point clouds (before depth) → reshape (B, num_points, num_timesteps*3)
        → point_encoder MLP → self-attention → FFN
    - Query input = [proprio | depth_features] → proprio_encoder → 1 token
        → cross-attention (queries enriched point tokens) → attn_out
    - Trunk: [proprio | depth_features | attn_out] → MLP → mu, sigma, value

Depth handling:
    Depth is ALWAYS packed at the END of the obs tensor.
    Point clouds are right before depth.
    Both dimensions are computed dynamically from Y2R config.

Usage in YAML config:
    network:
      name: depth_point_transformer_student
      hidden_dim: 64
      num_heads: 2
      self_attention: True
      fuse_depth_in_query: True  # concatenate depth_features into query token input
      point_encoder_layers: [64, 64]
      point_encoder_norm: True
      proprio_encoder_layers: [64, 64]
      proprio_encoder_norm: True
      depth_encoder:
        channels: [32, 128, 256]
      mlp:
        units: [512, 256, 128]
        activation: elu
"""

from __future__ import annotations

import contextlib
import torch
import torch.nn as nn
from typing import Dict, Any

from rl_games.algos_torch.network_builder import NetworkBuilder
from rl_games.algos_torch.running_mean_std import RunningMeanStd

from .point_transformer import SelfAttentionBlock, CrossAttentionBlock, FeedForwardBlock
from .depth_resnet_student import DepthCNN

try:
    _compile_disable = torch.compiler.disable
except Exception:
    try:
        _compile_disable = torch._dynamo.disable
    except Exception:
        def _compile_disable(fn):
            return fn


class DepthPointTransformerStudentBuilder(NetworkBuilder):
    """Network builder for student policy with depth encoder + point transformer.

    Combines:
    - SmallResNet for wrist depth images
    - Self+Cross attention for visible point cloud processing
    - MLP trunk for action/value heads
    """

    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)
        self.params = None

    def _inject_params(self, params):
        """Inject student observation params from Y2R config (single source of truth).

        Reads student_num_points, history lengths, window_size, and camera dims
        from base.yaml via config_loader.
        """
        from ..config_loader import get_config
        import os

        mode = os.getenv('Y2R_MODE')
        cfg = get_config(mode)

        params['student_num_points'] = cfg.observations.student_num_points
        params['student_pc_history'] = cfg.observations.history.student_pc
        params['student_targets_history'] = cfg.observations.history.student_targets
        params['window_size'] = cfg.trajectory.window_size
        params['depth_height'] = cfg.wrist_camera.height
        params['depth_width'] = cfg.wrist_camera.width

    def load(self, params: Dict[str, Any]):
        """Load parameters (called by rl_games). Injects Y2R config params."""
        self._inject_params(params)
        self.params = params

    def build(self, name: str, **kwargs) -> nn.Module:
        """Build and return the hybrid network."""
        if self.params is None:
            self.params = {}
            self._inject_params(self.params)

        # Enable TF32 for matmuls outside AMP autocast
        torch.backends.cuda.matmul.allow_tf32 = True

        net = DepthPointTransformerStudentBuilder.Network(self.params, **kwargs)
        if self.params.get("torch_compile", False):
            net = torch.compile(net, mode="reduce-overhead", dynamic=True)
        return net

    class Network(NetworkBuilder.BaseNetwork):
        """Hybrid Depth + Point Transformer student policy network."""

        def __init__(self, params: Dict[str, Any], **kwargs):
            NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)

            self.params = params
            self.actions_num = kwargs.pop('actions_num')
            self.value_size = kwargs.pop('value_size', 1)
            input_shape = kwargs.pop('input_shape')
            self.num_seqs = kwargs.pop('num_seqs', 1)
            # RL-Games normalize_input is disabled (depth in obs). We normalize inside model.
            _ = kwargs.pop('normalize_input', False)

            self.input_obs_dim = input_shape[0] if isinstance(input_shape, (tuple, list)) else input_shape

            # =====================================================================
            # Dimensions from Y2R config (injected by builder)
            # =====================================================================
            self.num_points = params['student_num_points']
            student_pc_history = params['student_pc_history']
            student_targets_history = params['student_targets_history']
            window_size = params['window_size']
            depth_height = params['depth_height']
            depth_width = params['depth_width']

            self.num_timesteps = student_pc_history + window_size * student_targets_history
            self.point_cloud_dim = self.num_points * self.num_timesteps * 3
            self.depth_dim = depth_height * depth_width
            self.depth_height = depth_height
            self.depth_width = depth_width

            # Proprio = everything except point clouds and depth
            self.proprio_dim = self.input_obs_dim - self.point_cloud_dim - self.depth_dim

            if self.proprio_dim < 0:
                raise ValueError(
                    f"Observation dim ({self.input_obs_dim}) too small for "
                    f"point_cloud_dim ({self.point_cloud_dim}) + depth_dim ({self.depth_dim}). "
                    f"Check student obs groups."
                )

            # =====================================================================
            # Transformer params
            # =====================================================================
            self.hidden_dim = params.get("hidden_dim", 64)
            self.num_heads = params.get("num_heads", 2)
            self.use_self_attention = params.get("self_attention", True)
            self.use_flash_attention = params.get("flash_attention", True)
            attn_dropout = params.get("attn_dropout", 0.0)
            ffn_ratio = params.get("ffn_ratio", 0.0)
            ffn_dropout = params.get("ffn_dropout", 0.0)

            # =====================================================================
            # Normalization: RunningMeanStd on non-depth obs (proprio + point clouds)
            # This matches teacher behavior (rl_games normalize_input: True normalizes
            # entire obs including PCs). Student can't use global normalize_input because
            # depth is packed in obs. Running-stats updates are explicitly gated by
            # apply_obs_rms_update (toggled by train/distill/play scripts).
            # =====================================================================
            self.normalize_non_depth = params.get("normalize_base_obs", True)
            self.apply_obs_rms_update = bool(params.get("apply_obs_rms_update", False))
            non_depth_dim = self.proprio_dim + self.point_cloud_dim
            if self.normalize_non_depth:
                self.running_mean_std = RunningMeanStd(non_depth_dim)

            # =====================================================================
            # Depth encoder (3-layer CNN)
            # =====================================================================
            depth_cfg = params.get('depth_encoder', {})
            self.depth_channels = depth_cfg.get('channels', [32, 64, 128])
            self.depth_encoder = DepthCNN(in_channels=1, channels=self.depth_channels)
            self.depth_features_dim = self.depth_encoder.out_features

            # Depth augmentation (GPU-accelerated, explicitly controlled)
            self.use_depth_aug = params.get('depth_augmentation', False)
            self.apply_depth_aug = bool(params.get("apply_depth_aug", False))
            self.depth_aug_config = params.get('depth_aug_config', None)
            self.depth_aug = None  # Lazy init on first forward

            # =====================================================================
            # Point cloud encoder
            # =====================================================================
            point_encoder_layers = params.get("point_encoder_layers", [self.hidden_dim])
            point_encoder_norm = params.get("point_encoder_norm", False)
            self.point_encoder = self._build_encoder(
                self.num_timesteps * 3, point_encoder_layers, point_encoder_norm
            )

            # =====================================================================
            # Proprio encoder
            # =====================================================================
            self.fuse_depth_in_query = params.get("fuse_depth_in_query", True)
            query_input_dim = self.proprio_dim + self.depth_features_dim if self.fuse_depth_in_query else self.proprio_dim
            proprio_encoder_layers = params.get("proprio_encoder_layers", [self.hidden_dim])
            proprio_encoder_norm = params.get("proprio_encoder_norm", False)
            self.proprio_encoder = self._build_encoder(
                query_input_dim, proprio_encoder_layers, proprio_encoder_norm
            )

            # =====================================================================
            # Attention blocks
            # =====================================================================
            if self.use_self_attention:
                self.self_attn = SelfAttentionBlock(self.hidden_dim, self.num_heads, dropout=attn_dropout)
                self.self_ffn = FeedForwardBlock(self.hidden_dim, ffn_ratio, dropout=ffn_dropout)
            self.cross_attn = CrossAttentionBlock(self.hidden_dim, self.num_heads, dropout=attn_dropout)

            # =====================================================================
            # Trunk MLP: [proprio | depth_features | attn_out] → hidden
            # =====================================================================
            trunk_input_dim = self.proprio_dim + self.depth_features_dim + self.hidden_dim
            mlp_units = self.units
            if len(mlp_units) == 0:
                out_size = trunk_input_dim
                self.trunk = nn.Identity()
            else:
                out_size = mlp_units[-1]
                self.trunk = self._build_trunk(trunk_input_dim, mlp_units)

            # =====================================================================
            # Output heads
            # =====================================================================
            self.value = nn.Linear(out_size, self.value_size)
            self.value_act = self.activations_factory.create(self.value_activation)

            if self.is_continuous:
                self.mu = nn.Linear(out_size, self.actions_num)
                self.mu_act = self.activations_factory.create(
                    self.space_config['mu_activation']
                )

                if self.fixed_sigma == "fixed":
                    self.sigma = nn.Parameter(
                        torch.zeros(self.actions_num, requires_grad=True, dtype=torch.float32),
                        requires_grad=True
                    )
                else:
                    self.sigma_head = nn.Linear(out_size, self.actions_num)

                self.sigma_act = self.activations_factory.create(
                    self.space_config['sigma_activation']
                )

            # Initialize all weights
            self._init_weights()

            # Initialize output heads
            if self.is_continuous:
                mu_init = self.init_factory.create(**self.space_config['mu_init'])
                sigma_init = self.init_factory.create(**self.space_config['sigma_init'])
                mu_init(self.mu.weight)
                if self.fixed_sigma == "fixed":
                    sigma_init(self.sigma)
                else:
                    sigma_init(self.sigma_head.weight)

            # Store last depth for video recording
            self.last_depth_img = None

            print(f"[DepthPointTransformerStudent] Initialized:")
            print(f"  input_obs_dim: {self.input_obs_dim}")
            print(f"  proprio_dim: {self.proprio_dim}")
            print(f"  point_cloud_dim: {self.point_cloud_dim} "
                  f"({self.num_points} pts × {self.num_timesteps} ts × 3)")
            print(f"  depth_dim: {self.depth_dim} ({self.depth_height}×{self.depth_width})")
            print(f"  depth_features: {self.depth_features_dim}")
            print(f"  hidden_dim: {self.hidden_dim}, num_heads: {self.num_heads}")
            print(f"  trunk_input: {trunk_input_dim} → mlp {mlp_units} → {out_size}")
            print(f"  normalize_non_depth: {self.normalize_non_depth}")
            print(f"  fuse_depth_in_query: {self.fuse_depth_in_query}")

        def _build_encoder(self, input_dim: int, layers: list, use_norm: bool) -> nn.Module:
            """Build MLP encoder (reusable for point and proprio encoders)."""
            if layers[-1] != self.hidden_dim:
                raise ValueError(
                    f"Encoder layers must end with hidden_dim ({self.hidden_dim}), got {layers[-1]}"
                )
            modules = []
            in_dim = input_dim
            for i, out_dim in enumerate(layers):
                modules.append(nn.Linear(in_dim, out_dim))
                if i < len(layers) - 1:
                    modules.append(self.activations_factory.create(self.activation))
                in_dim = out_dim
            if use_norm:
                modules.append(nn.LayerNorm(self.hidden_dim))
            return nn.Sequential(*modules) if len(modules) > 1 else modules[0]

        def _build_trunk(self, input_dim: int, units: list) -> nn.Sequential:
            """Build MLP trunk."""
            layers = []
            in_size = input_dim
            for unit in units:
                layers.append(nn.Linear(in_size, unit))
                layers.append(self.activations_factory.create(self.activation))
                if self.normalization == 'layer_norm':
                    layers.append(nn.LayerNorm(unit))
                elif self.normalization == 'batch_norm':
                    layers.append(nn.BatchNorm1d(unit))
                in_size = unit
            return nn.Sequential(*layers)

        def _init_weights(self):
            """Initialize network weights."""
            mlp_init = self.init_factory.create(**self.initializer)
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    # Policy output heads are initialized explicitly after this pass.
                    if module is getattr(self, "mu", None) or module is getattr(self, "sigma_head", None):
                        continue
                    mlp_init(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        def _init_depth_aug(self, device: str):
            """Lazy initialization of DepthAug (needs device info)."""
            if self.depth_aug is None and self.use_depth_aug:
                import warp as wp
                wp.init()
                from isaaclab_tasks.manager_based.manipulation.y2r.distillation.depth_augs import DepthAug
                self.depth_aug = DepthAug(device, self.depth_aug_config)
                print(f"[DepthPointTransformerStudent] DepthAug initialized on {device}")

        @_compile_disable
        def _apply_depth_aug_eager(self, depth_img: torch.Tensor) -> torch.Tensor:
            """Run Warp depth augmentation outside torch.compile tracing."""
            depth_3d = depth_img.squeeze(1)
            depth_3d = self.depth_aug.augment(depth_3d)
            return depth_3d.unsqueeze(1)

        @_compile_disable
        def _apply_obs_rms_eager(self, non_depth: torch.Tensor) -> torch.Tensor:
            """Run obs RMS outside torch.compile tracing to avoid CUDAGraph overwrite issues."""
            return self.running_mean_std(non_depth)

        def forward(self, obs_dict: Dict[str, Any]):
            """Forward pass through hybrid depth + point transformer network.

            Args:
                obs_dict: Dictionary with 'obs' key containing flat observations.
                    Layout: [proprio | point_clouds | depth]

            Returns:
                mu, sigma, value, states (None for non-recurrent)
            """
            full_obs = obs_dict['obs']
            B = full_obs.shape[0]

            # =================================================================
            # Split observation: [proprio | point_clouds | depth]
            # =================================================================
            depth_flat = full_obs[:, -self.depth_dim:]
            non_depth = full_obs[:, :-self.depth_dim]

            # Normalize non-depth (proprio + point clouds), matching teacher behavior.
            # Running stats update is explicitly controlled via apply_obs_rms_update.
            if self.normalize_non_depth:
                non_depth_dtype = non_depth.dtype
                if self.apply_obs_rms_update:
                    self.running_mean_std.train()
                else:
                    self.running_mean_std.eval()
                if non_depth.dtype != torch.float32:
                    non_depth = non_depth.float()
                non_depth = self._apply_obs_rms_eager(non_depth)
                if non_depth.dtype != non_depth_dtype:
                    non_depth = non_depth.to(non_depth_dtype)

            proprio = non_depth[:, :self.proprio_dim]
            pc_flat = non_depth[:, self.proprio_dim:]

            # Reshape point cloud to (B, num_points, num_timesteps * 3)
            point_obs = pc_flat.reshape(B, self.num_points, self.num_timesteps * 3)

            # =================================================================
            # Depth encoder
            # =================================================================
            depth_img = depth_flat.view(B, 1, self.depth_height, self.depth_width)

            # Apply depth augmentation when explicitly enabled.
            if self.apply_depth_aug and self.use_depth_aug:
                self._init_depth_aug(str(depth_img.device))
                if self.depth_aug is not None:
                    depth_img = self._apply_depth_aug_eager(depth_img)

            self.last_depth_img = depth_img.detach()
            depth_features = self.depth_encoder(depth_img)  # (B, depth_features_dim)

            # =================================================================
            # Point transformer
            # =================================================================
            # Encode points: (B, num_points, num_timesteps*3) → (B, num_points, hidden_dim)
            point_tokens = self.point_encoder(point_obs)

            # Encode query: (B, proprio_dim [+ depth_features_dim]) → (B, 1, hidden_dim)
            if self.fuse_depth_in_query:
                query_input = torch.cat([proprio, depth_features], dim=-1)
            else:
                query_input = proprio
            proprio_encoded = self.proprio_encoder(query_input)
            proprio_token = proprio_encoded.unsqueeze(1)

            # Self-attention on point tokens, then cross-attention from proprio
            attn_ctx = (torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH)
                        if not self.use_flash_attention
                        else contextlib.nullcontext())
            with attn_ctx:
                if self.use_self_attention:
                    point_tokens = self.self_attn(point_tokens)
                    point_tokens = self.self_ffn(point_tokens)
                attn_out = self.cross_attn(proprio_token, point_tokens)  # (B, 1, hidden_dim)
            attn_features = attn_out[:, 0]  # (B, hidden_dim)

            # =================================================================
            # Trunk: [proprio | depth_features | attn_out]
            # =================================================================
            trunk_input = torch.cat([proprio, depth_features, attn_features], dim=-1)
            out = self.trunk(trunk_input)

            # =================================================================
            # Output heads
            # =================================================================
            value = self.value_act(self.value(out))

            if self.is_continuous:
                mu = self.mu_act(self.mu(out))
                if self.fixed_sigma == "fixed":
                    sigma = self.sigma_act(self.sigma)
                else:
                    sigma = self.sigma_act(self.sigma_head(out))
                return mu, mu * 0 + sigma, value, None

            return None, None, value, None

        def is_rnn(self):
            return False

        def get_default_rnn_state(self):
            return None

        def load(self, params: Dict[str, Any]):
            """Load network configuration from params."""
            self.separate = params.get('separate', False)
            self.units = params['mlp']['units']
            self.activation = params['mlp']['activation']
            self.initializer = params['mlp']['initializer']
            self.normalization = params.get('normalization', None)
            self.value_activation = params.get('value_activation', 'None')

            self.has_space = 'space' in params
            if self.has_space:
                self.is_continuous = 'continuous' in params['space']
                self.is_discrete = 'discrete' in params['space']
                self.is_multi_discrete = 'multi_discrete' in params['space']

                if self.is_continuous:
                    self.space_config = params['space']['continuous']
                    fixed_sigma = self.space_config.get('fixed_sigma', True)
                    if fixed_sigma is True:
                        fixed_sigma = "fixed"
                    elif fixed_sigma is False:
                        fixed_sigma = "obs_cond"
                    self.fixed_sigma = fixed_sigma
            else:
                self.is_continuous = True
                self.is_discrete = False
                self.is_multi_discrete = False
                self.fixed_sigma = "fixed"
                self.space_config = {
                    'mu_activation': 'None',
                    'sigma_activation': 'None',
                    'mu_init': {'name': 'default'},
                    'sigma_init': {'name': 'const_initializer', 'val': 0},
                }
