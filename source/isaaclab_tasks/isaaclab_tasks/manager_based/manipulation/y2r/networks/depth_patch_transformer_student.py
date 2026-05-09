# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2024, Harsh Gupta
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""ViT-style Depth Patch Tokenizer + Point Transformer Student Network for rl_games.

Variant of depth_point_transformer_student where the depth pathway is
redesigned around AME/ViT-style patch tokens instead of a CNN+GAP encoder.
Depth becomes a small set of spatial tokens that join the point cloud tokens
in the cross-attention's K/V set, instead of a single 128-d feature dumped
into the trunk concat.

Architecture:
    - Depth (last H×W of obs) → ConvStem (4× stride-2 conv stages) → (B, hidden, H', W')
        → flatten → (B, num_depth_tokens, hidden) + learnable 2D positional embedding
    - Point clouds → reshape (B, num_points, num_timesteps*3)
        → point_encoder MLP → point tokens (B, N, hidden)
    - Combined K/V = concat([point_tokens, depth_tokens], dim=1)
        with optional learnable modality embedding (PC vs depth) added
    - Global feature (AME-2 style): amax over the COMBINED K/V → (B, hidden)
    - Query = proprio_encoder([proprio, global_feat?]) → 1 token (depth NOT in query)
    - Optional self-attention on combined K/V (off by default)
    - Cross-attention: query attends to combined K/V → attn_out (B, hidden)
    - Trunk: [proprio, global_feat?, attn_out] → MLP → mu, sigma, value
        (NO depth_features concat — depth is in K/V, not in trunk)

Depth handling:
    Depth is ALWAYS packed at the END of the obs tensor.
    Point clouds are right before depth.
    Both dimensions are computed dynamically from Y2R config.

Usage in YAML config:
    network:
      name: depth_patch_transformer_student
      hidden_dim: 128
      num_heads: 16
      self_attention: False
      use_global_feature: True
      point_encoder_layers: [128, 128]
      point_encoder_norm: True
      proprio_encoder_layers: [128, 128]
      proprio_encoder_norm: True
      depth_tokenizer:
        channels: [32, 64, 128, 128]   # 4 stride-2 stages; last channel = hidden_dim
        pe_init_std: 0.02
        modality_embedding: True       # add learnable PC vs depth embedding
      mlp:
        units: [1024, 512, 256, 128]
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

try:
    _compile_disable = torch.compiler.disable
except Exception:
    try:
        _compile_disable = torch._dynamo.disable
    except Exception:
        def _compile_disable(fn):
            return fn


class ConvStemTokenizer(nn.Module):
    """Patch tokenizer for the wrist depth image (ViT-style).

    Stack of stride-2 conv layers (each Conv3x3 + GroupNorm + GELU) producing
    a small grid of dense tokens. Each spatial cell becomes one token of
    `hidden_dim` dimensions, then a learnable 2D positional embedding is added.

    Default channels=[32, 64, 128, 128] on a 60x80 input gives a 4x5 = 20
    token grid. Output shape (B, num_tokens, hidden_dim).

    Args:
        in_channels: input channels (1 for depth)
        channels: list of output channels per stride-2 stage. Last channel is
            projected to hidden_dim if it doesn't match.
        hidden_dim: token embedding dim (matches the point transformer hidden_dim)
        in_height, in_width: input depth image shape (used to compute num_tokens)
        pe_init_std: stddev for initializing the learnable positional embedding
    """

    def __init__(
        self,
        in_channels: int,
        channels: list,
        hidden_dim: int,
        in_height: int,
        in_width: int,
        pe_init_std: float = 0.02,
    ):
        super().__init__()
        layers = []
        in_c = in_channels
        for c in channels:
            layers += [
                nn.Conv2d(in_c, c, kernel_size=3, stride=2, padding=1, bias=False),
                nn.GroupNorm(1, c),
                nn.GELU(),
            ]
            in_c = c
        self.stem = nn.Sequential(*layers)

        # Compute output spatial size: each stride-2 conv halves (with ceil for odd)
        h, w = in_height, in_width
        for _ in channels:
            h = (h + 1) // 2
            w = (w + 1) // 2
        self.out_h, self.out_w = h, w
        self.num_tokens = h * w

        # Project final channels to hidden_dim if mismatched
        last_c = channels[-1]
        self.proj = nn.Linear(last_c, hidden_dim) if last_c != hidden_dim else nn.Identity()

        # Learnable 2D positional embedding (one per grid cell)
        self.pe = nn.Parameter(torch.randn(1, self.num_tokens, hidden_dim) * pe_init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_channels, H, W)
        x = self.stem(x)                          # (B, last_c, out_h, out_w)
        B = x.shape[0]
        x = x.flatten(2).transpose(1, 2)          # (B, num_tokens, last_c)
        x = self.proj(x)                          # (B, num_tokens, hidden_dim)
        x = x + self.pe                           # broadcast PE
        return x


class DepthPatchTransformerStudentBuilder(NetworkBuilder):
    """Network builder for student policy with patch-tokenized depth + point transformer.

    Combines:
    - ViT-style ConvStem patch tokenizer for wrist depth (no GAP — depth becomes K/V tokens)
    - Self+Cross attention for visible point cloud + depth tokens (combined K/V)
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
        params['use_depth_camera'] = cfg.mode.use_depth_camera
        params['depth_height'] = cfg.wrist_camera.height if cfg.mode.use_depth_camera else 0
        params['depth_width'] = cfg.wrist_camera.width if cfg.mode.use_depth_camera else 0

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

        net = DepthPatchTransformerStudentBuilder.Network(self.params, **kwargs)
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
            # use_depth_camera=False zeroes depth_dim so the obs split / encoder / aug
            # branches all collapse cleanly; saves the wrist depth camera too (handled
            # in trajectory_*_env_cfg.py). Required key — _inject_params reads it from
            # y2r_cfg, so a missing value means plumbing is broken; crash explicitly.
            self.use_depth_camera = bool(params['use_depth_camera'])
            self.depth_dim = depth_height * depth_width if self.use_depth_camera else 0
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
            # AME-2 global feature: max-pooled point tokens are concatenated into the query.
            self.use_global_feature = params["use_global_feature"]

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
            # Depth tokenizer (ConvStem patches) — skipped when use_depth_camera=False
            # =====================================================================
            if self.use_depth_camera:
                tok_cfg = params['depth_tokenizer']
                self.depth_channels = tok_cfg['channels']
                self.depth_tokenizer = ConvStemTokenizer(
                    in_channels=1,
                    channels=self.depth_channels,
                    hidden_dim=self.hidden_dim,
                    in_height=depth_height,
                    in_width=depth_width,
                    pe_init_std=float(tok_cfg.get('pe_init_std', 0.02)),
                )
                self.num_depth_tokens = self.depth_tokenizer.num_tokens

                # Optional learnable modality embedding so the model can distinguish
                # PC tokens from depth tokens in the combined K/V set.
                self.use_modality_embed = bool(tok_cfg.get('modality_embedding', True))
                if self.use_modality_embed:
                    # 2 × hidden: [pc_embed, depth_embed]
                    self.modality_embed = nn.Parameter(
                        torch.randn(2, self.hidden_dim) * float(tok_cfg.get('pe_init_std', 0.02))
                    )

                # Depth augmentation (GPU-accelerated, explicitly controlled)
                self.use_depth_aug = params.get('depth_augmentation', False)
                self.apply_depth_aug = bool(params.get("apply_depth_aug", False))
                self.depth_aug_config = params.get('depth_aug_config', None)
                self.depth_aug = None  # Lazy init on first forward
            else:
                self.depth_channels = []
                self.depth_tokenizer = None
                self.num_depth_tokens = 0
                self.use_modality_embed = False
                self.use_depth_aug = False
                self.apply_depth_aug = False
                self.depth_aug_config = None
                self.depth_aug = None

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
            # Depth no longer enters the query — it lives in the cross-attention K/V.
            query_input_dim = self.proprio_dim + (self.hidden_dim if self.use_global_feature else 0)
            proprio_encoder_layers = params.get("proprio_encoder_layers", [self.hidden_dim])
            proprio_encoder_norm = params.get("proprio_encoder_norm", False)
            self.proprio_encoder = self._build_encoder(
                query_input_dim, proprio_encoder_layers, proprio_encoder_norm
            )

            # =====================================================================
            # Attention blocks (operate on combined PC+depth K/V set)
            # =====================================================================
            if self.use_self_attention:
                self.self_attn = SelfAttentionBlock(self.hidden_dim, self.num_heads, dropout=attn_dropout)
                self.self_ffn = FeedForwardBlock(self.hidden_dim, ffn_ratio, dropout=ffn_dropout)
            self.cross_attn = CrossAttentionBlock(self.hidden_dim, self.num_heads, dropout=attn_dropout)

            # =====================================================================
            # Trunk MLP: [proprio | global_feat? | attn_out] → hidden
            # AME-2 fuses the global feature again at the trunk input (not just the query).
            # No depth_features concat — depth lives in the cross-attention K/V now.
            # =====================================================================
            trunk_input_dim = (
                self.proprio_dim
                + (self.hidden_dim if self.use_global_feature else 0)
                + self.hidden_dim
            )
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

            # Attention-capture path (visualization only — set capture_attn=True
            # before forward to populate last_cross_attn_weights with shape
            # (B, num_heads, 1, num_pc + num_depth_tokens). Off by default; the
            # train path stays on SDPA and returns no weights.
            self.capture_attn = False
            self.last_cross_attn_weights = None
            self.last_num_pc_tokens = None

            print(f"[DepthPatchTransformerStudent] Initialized:")
            print(f"  input_obs_dim: {self.input_obs_dim}")
            print(f"  proprio_dim: {self.proprio_dim}")
            print(f"  point_cloud_dim: {self.point_cloud_dim} "
                  f"({self.num_points} pts × {self.num_timesteps} ts × 3)")
            print(f"  depth_dim: {self.depth_dim} ({self.depth_height}×{self.depth_width})")
            if self.use_depth_camera:
                print(f"  depth_tokens: {self.num_depth_tokens} "
                      f"({self.depth_tokenizer.out_h}×{self.depth_tokenizer.out_w} × {self.hidden_dim})")
                print(f"  modality_embedding: {self.use_modality_embed}")
            else:
                print(f"  depth_tokens: 0 (use_depth_camera=False)")
            print(f"  hidden_dim: {self.hidden_dim}, num_heads: {self.num_heads}")
            print(f"  trunk_input: {trunk_input_dim} → mlp {mlp_units} → {out_size}")
            print(f"  normalize_non_depth: {self.normalize_non_depth}")
            print(f"  use_global_feature: {self.use_global_feature}")

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
                print(f"[DepthPatchTransformerStudent] DepthAug initialized on {device}")

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
            # Split observation: [proprio | point_clouds | depth?]
            # When use_depth_camera=False, depth_dim==0 and full_obs has no depth tail.
            # =================================================================
            if self.use_depth_camera:
                depth_flat = full_obs[:, -self.depth_dim:]
                non_depth = full_obs[:, :-self.depth_dim]
            else:
                depth_flat = None
                non_depth = full_obs

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

            # The flat layout is time-major: [t0_p0..pN, t1_p0..pN, ...].
            # Reshape to 4D time-major, then permute to point-centric so each
            # row is one physical point tracked across all timesteps.
            point_obs = (pc_flat
                         .reshape(B, self.num_timesteps, self.num_points, 3)
                         .permute(0, 2, 1, 3)
                         .reshape(B, self.num_points, self.num_timesteps * 3))

            # =================================================================
            # Depth tokenizer — only runs when use_depth_camera=True
            # Produces (B, num_depth_tokens, hidden_dim) instead of a single
            # GAP'd feature; tokens join the cross-attention K/V set below.
            # =================================================================
            if self.use_depth_camera:
                depth_img = depth_flat.view(B, 1, self.depth_height, self.depth_width)

                # Apply depth augmentation when explicitly enabled.
                if self.apply_depth_aug and self.use_depth_aug:
                    self._init_depth_aug(str(depth_img.device))
                    if self.depth_aug is not None:
                        depth_img = self._apply_depth_aug_eager(depth_img)

                self.last_depth_img = depth_img.detach()
                depth_tokens = self.depth_tokenizer(depth_img)  # (B, num_depth_tokens, hidden_dim)
            else:
                depth_tokens = None

            # =================================================================
            # Point transformer — combine PC + depth tokens into one K/V set
            # =================================================================
            # Encode points: (B, num_points, num_timesteps*3) → (B, num_points, hidden_dim)
            point_tokens = self.point_encoder(point_obs)

            # Add learnable modality embedding so the model can distinguish
            # PC tokens from depth tokens in the combined set.
            if depth_tokens is not None and self.use_modality_embed:
                point_tokens = point_tokens + self.modality_embed[0].view(1, 1, -1)
                depth_tokens = depth_tokens + self.modality_embed[1].view(1, 1, -1)

            # Combined K/V tokens for cross-attention. When depth is disabled,
            # this is just point_tokens (matches the no_depth ablation pathway).
            kv_tokens = (
                torch.cat([point_tokens, depth_tokens], dim=1)
                if depth_tokens is not None
                else point_tokens
            )

            # Global feature (AME-2): max-pool over POINT tokens only.
            # Pooling over kv_tokens (PC + depth) would smuggle depth into both
            # the proprio query AND the trunk concat, defeating the design goal
            # of "depth only via cross-attention K/V". Keep depth strictly inside
            # the attention pathway.
            if self.use_global_feature:
                global_feat = point_tokens.amax(dim=1)  # (B, hidden_dim)

            # Encode query: depth no longer enters here — only proprio (+ optional global).
            # query_input shape: (B, proprio_dim [+ hidden_dim]) → (B, hidden_dim)
            query_parts = [proprio]
            if self.use_global_feature:
                query_parts.append(global_feat)
            query_input = torch.cat(query_parts, dim=-1) if len(query_parts) > 1 else query_parts[0]
            proprio_encoded = self.proprio_encoder(query_input)
            proprio_token = proprio_encoded.unsqueeze(1)

            # Self-attention on combined K/V tokens, then cross-attention from proprio.
            attn_ctx = (torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH)
                        if not self.use_flash_attention
                        else contextlib.nullcontext())
            with attn_ctx:
                if self.use_self_attention:
                    kv_tokens = self.self_attn(kv_tokens)
                    kv_tokens = self.self_ffn(kv_tokens)
                if self.capture_attn:
                    attn_out, attn_weights = self.cross_attn(
                        proprio_token, kv_tokens, return_attn=True
                    )
                    self.last_cross_attn_weights = attn_weights.detach()
                    self.last_num_pc_tokens = point_tokens.shape[1]
                else:
                    attn_out = self.cross_attn(proprio_token, kv_tokens)  # (B, 1, hidden_dim)
            attn_features = attn_out[:, 0]  # (B, hidden_dim)

            # =================================================================
            # Trunk: [proprio | global_feat? | attn_out]
            # No depth_features concat — depth lives in the cross-attention K/V.
            # =================================================================
            trunk_parts = [proprio]
            if self.use_global_feature:
                trunk_parts.append(global_feat)
            trunk_parts.append(attn_features)
            trunk_input = torch.cat(trunk_parts, dim=-1)
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

        def get_depth_attention_grid(self, reduce: str = "mean"):
            """Return the cross-attention weights placed on depth tokens, reshaped
            to the spatial (out_h, out_w) grid produced by the ConvStem tokenizer.

            Requires capture_attn=True before the most recent forward. Returns
            None when capture is off, depth is disabled, or no weights have been
            captured yet — callers can safely no-op on None.

            Args:
                reduce: "mean" or "max" over heads, or "none" to keep heads.

            Returns:
                Tensor of shape (B, out_h, out_w) for reduce in {mean,max}, or
                (B, num_heads, out_h, out_w) for reduce="none".
            """
            if (
                not self.capture_attn
                or self.last_cross_attn_weights is None
                or not self.use_depth_camera
                or self.last_num_pc_tokens is None
            ):
                return None
            # last_cross_attn_weights: (B, heads, 1, num_pc + num_depth)
            depth_attn = self.last_cross_attn_weights[:, :, 0, self.last_num_pc_tokens:]
            B, H, N = depth_attn.shape
            out_h, out_w = self.depth_tokenizer.out_h, self.depth_tokenizer.out_w
            grid = depth_attn.view(B, H, out_h, out_w)
            if reduce == "mean":
                return grid.mean(dim=1)
            if reduce == "max":
                return grid.amax(dim=1)
            if reduce == "none":
                return grid
            raise ValueError(f"reduce must be one of 'mean'|'max'|'none', got {reduce!r}")

        def get_attention_mass_split(self):
            """Return the split of total cross-attention mass between PC and depth
            tokens for the most recent forward — sanity check for whether depth is
            actually used. Returns (pc_mass, depth_mass) tensors of shape (B,) each.
            Sums to 1.0 per batch element since softmax weights are normalized.
            Returns (None, None) when capture is off or no weights have been seen.
            """
            if (
                not self.capture_attn
                or self.last_cross_attn_weights is None
                or self.last_num_pc_tokens is None
            ):
                return None, None
            w = self.last_cross_attn_weights[:, :, 0, :]            # (B, heads, S_kv)
            w = w.mean(dim=1)                                       # (B, S_kv) — mean over heads
            pc_mass = w[:, :self.last_num_pc_tokens].sum(dim=-1)
            depth_mass = w[:, self.last_num_pc_tokens:].sum(dim=-1)
            return pc_mass, depth_mass

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
