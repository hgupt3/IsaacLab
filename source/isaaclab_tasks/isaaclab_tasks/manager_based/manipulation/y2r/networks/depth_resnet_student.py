# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""3-Layer CNN Depth Encoder for Student Policy Distillation and PPO Fine-tuning.

This module implements a student policy network that processes wrist camera
depth images through a simple 3-layer CNN encoder and concatenates features
with proprioceptive and point cloud observations.

Architecture:
    - Depth encoder: 3x Conv2d(k=3,s=2) + GroupNorm + ReLU → GAP
    - Feature fusion: depth_features (C) + base_obs (proprio + student_perception + student_targets)
      where C is the last entry of depth_encoder.channels
    - Policy MLP: [512, 256, 128] → actions

Depth handling:
    Depth is ALWAYS packed at the END of the obs tensor (last H*W floats).
    The model:
    1. Slices depth from end of obs
    2. Applies RunningMeanStd normalization ONLY to base_obs (not depth)
    3. Applies optional DepthAug when explicitly enabled via apply_depth_aug
    4. Encodes depth via DepthCNN
    5. Concatenates normalized base_obs + depth_features

Usage in YAML config:
    network:
      name: depth_resnet_student
      depth_augmentation: True
      depth_aug_config:
        correlated_noise:
          sigma_s: 0.3
          sigma_d: 0.1
        ...
      depth_encoder:
        # Optional explicit override. If omitted, uses active Y2R wrist camera size.
        # height: 64
        # width: 64
        channels: [32, 64, 128]
      mlp:
        units: [512, 256, 128]
        activation: elu
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

from rl_games.algos_torch.network_builder import NetworkBuilder
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd

try:
    _compile_disable = torch.compiler.disable
except Exception:
    try:
        _compile_disable = torch._dynamo.disable
    except Exception:
        def _compile_disable(fn):
            return fn


def _get_wrist_camera_shape_from_y2r_config() -> tuple[int, int] | None:
    """Get wrist camera (height, width) from active Y2R config, if available."""
    try:
        import os
        from ..config_loader import get_config

        mode = os.getenv("Y2R_MODE")
        y2r_cfg = get_config(mode)
        return int(y2r_cfg.wrist_camera.height), int(y2r_cfg.wrist_camera.width)
    except Exception:
        return None


# =============================================================================
# 3-Layer CNN Depth Encoder
# =============================================================================

class DepthCNN(nn.Module):
    """Simple 3-layer CNN encoder for depth images.

    Each layer: Conv2d(k=3, s=2, p=1) → GroupNorm → ReLU, followed by GAP.

    Example for 64x64 input with channels=[32, 64, 128]:
        Input:  (B, 1, 64, 64)
        Conv1:  (B, 32, 32, 32)
        Conv2:  (B, 64, 16, 16)
        Conv3:  (B, 128, 8, 8)
        GAP:    (B, 128, 1, 1)
        Output: (B, 128)

    Args:
        in_channels: Number of input channels (1 for depth)
        channels: List of channel sizes [32, 64, 128]
    """

    def __init__(self, in_channels: int = 1, channels: list = None):
        super().__init__()

        if channels is None:
            channels = [32, 64, 128]

        assert len(channels) == 3, f"Expected 3 channel values, got {len(channels)}"
        c1, c2, c3 = channels

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(1, c1),
            nn.ReLU(inplace=True),

            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(1, c2),
            nn.ReLU(inplace=True),

            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(1, c3),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.out_features = c3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        x = self.pool(x)
        x = x.flatten(1)
        return x


# =============================================================================
# A2C Network Builder for rl_games
# =============================================================================

class DepthResNetStudentBuilder(NetworkBuilder):
    """Network builder for student policy with depth encoder.
    
    This builder creates a network that:
    1. Slices depth from END of obs tensor (last H*W floats)
    2. Normalizes ONLY base_obs with RunningMeanStd (depth is already [0,1])
    3. Applies optional DepthAug when explicitly enabled via apply_depth_aug
    4. Encodes depth through SmallResNet
    5. Concatenates features and feeds through MLP to produce actions
    """
    
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)
        self.params = None
    
    def load(self, params: Dict[str, Any]):
        self.params = params
    
    def build(self, name: str, **kwargs) -> nn.Module:
        if self.params is None:
            raise RuntimeError(
                "DepthResNetStudentBuilder.build() called before load(). "
                "Ensure RL-Games has loaded network params from agent config."
            )
        # Enable TF32 for matmuls outside AMP autocast.
        torch.backends.cuda.matmul.allow_tf32 = True

        net = DepthResNetStudentBuilder.Network(self.params, **kwargs)
        if self.params.get("torch_compile", False):
            net = torch.compile(net, mode="reduce-overhead", dynamic=True)
        return net
    
    class Network(NetworkBuilder.BaseNetwork):
        """Student policy network with depth encoder."""
        
        def __init__(self, params: Dict[str, Any], **kwargs):
            NetworkBuilder.BaseNetwork.__init__(self)
            
            # Extract kwargs
            actions_num = kwargs.pop('actions_num')
            input_shape = kwargs.pop('input_shape')  # Shape of full 'obs' (base_obs + depth)
            self.value_size = kwargs.pop('value_size', 1)
            self.num_seqs = kwargs.pop('num_seqs', 1)
            # RL-Games `config.normalize_input` applies a global RMS *before* the model.
            # Since we pack depth into obs, global RMS would incorrectly normalize depth.
            # We therefore disable RL-Games normalize_input in the YAML and do base_obs-only
            # normalization inside this model via `network.normalize_base_obs`.
            _ = kwargs.pop('normalize_input', False)  # intentionally ignored
            
            # Load params
            self.load(params)

            # Model-side normalization for base_obs only (independent of RL-Games config.normalize_input)
            self.normalize_base_obs = params.get("normalize_base_obs", True)
            
            # Depth encoder config
            depth_cfg = params.get('depth_encoder', {})
            camera_hw = _get_wrist_camera_shape_from_y2r_config()
            resolution = depth_cfg.get('resolution', None)
            depth_h = depth_cfg.get('height', None)
            depth_w = depth_cfg.get('width', None)

            if depth_h is not None or depth_w is not None:
                if depth_h is None or depth_w is None:
                    raise ValueError("depth_encoder.height and depth_encoder.width must be set together.")
                self.depth_height = int(depth_h)
                self.depth_width = int(depth_w)
            elif camera_hw is not None:
                self.depth_height, self.depth_width = camera_hw
                if resolution is not None:
                    print(
                        f"[DepthResNetStudent] Ignoring depth_encoder.resolution={resolution} "
                        f"and using wrist_camera size {self.depth_height}x{self.depth_width} from Y2R config."
                    )
            else:
                # Fallback for standalone usage outside Y2R config system.
                self.depth_height = int(resolution) if resolution is not None else 64
                self.depth_width = int(resolution) if resolution is not None else 64

            self.depth_channels = depth_cfg.get('channels', [32, 64, 128])
            self.depth_dim = self.depth_height * self.depth_width
            
            # Runtime control flags (explicitly toggled by train/distill/play scripts)
            self.apply_obs_rms_update = bool(params.get("apply_obs_rms_update", False))

            # Depth augmentation config
            self.use_depth_aug = params.get('depth_augmentation', False)
            self.apply_depth_aug = bool(params.get("apply_depth_aug", False))
            self.depth_aug_config = params.get('depth_aug_config', None)
            self.depth_aug = None  # Lazy init on first forward (needs device)
            
            # Build depth encoder
            self.depth_encoder = DepthCNN(
                in_channels=1,
                channels=self.depth_channels
            )
            depth_features = self.depth_encoder.out_features  # 128
            
            # Calculate base_obs dimension (full obs minus depth)
            full_obs_dim = input_shape[0] if isinstance(input_shape, tuple) else input_shape
            self.base_obs_dim = full_obs_dim - self.depth_dim
            if self.base_obs_dim <= 0:
                raise ValueError(
                    f"Invalid student obs layout: full_obs_dim={full_obs_dim}, depth_dim={self.depth_dim} "
                    f"(depth={self.depth_height}x{self.depth_width}). "
                    "Expected depth to be packed at tail of obs with base_obs before it."
                )
            
            # RunningMeanStd for base_obs ONLY (not depth)
            if self.normalize_base_obs:
                self.running_mean_std = RunningMeanStd(self.base_obs_dim)
            
            # MLP input: normalized base_obs + depth_features
            mlp_input_size = self.base_obs_dim + depth_features
            
            # Build MLP
            mlp_units = self.units
            if len(mlp_units) == 0:
                out_size = mlp_input_size
            else:
                out_size = mlp_units[-1]
            
            self.actor_mlp = self._build_mlp(
                input_size=mlp_input_size,
                units=mlp_units,
                activation=self.activation,
                dense_func=nn.Linear,
                norm_func_name=self.normalization,
            )
            
            # Value head
            self.value = nn.Linear(out_size, self.value_size)
            self.value_act = self.activations_factory.create(self.value_activation)
            
            # Policy head (continuous actions)
            if self.is_continuous:
                self.mu = nn.Linear(out_size, actions_num)
                self.mu_act = self.activations_factory.create(
                    self.space_config['mu_activation']
                )
                
                if self.fixed_sigma == "fixed":
                    self.sigma = nn.Parameter(
                        torch.zeros(actions_num, requires_grad=True, dtype=torch.float32),
                        requires_grad=True
                    )
                else:
                    self.sigma_head = nn.Linear(out_size, actions_num)
                
                self.sigma_act = self.activations_factory.create(
                    self.space_config['sigma_activation']
                )
                
                # Initialize
                mu_init = self.init_factory.create(**self.space_config['mu_init'])
                sigma_init = self.init_factory.create(**self.space_config['sigma_init'])
                mu_init(self.mu.weight)
                if self.fixed_sigma == "fixed":
                    sigma_init(self.sigma)
                else:
                    sigma_init(self.sigma_head.weight)
            
            # Initialize MLP weights
            mlp_init = self.init_factory.create(**self.initializer)
            for m in self.actor_mlp.modules():
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
            
            # Store last post-augmentation depth for video recording
            self.last_depth_img = None  # Will be (B, 1, H, W) after forward
            
            print(f"[DepthResNetStudent] Initialized:")
            print(f"  full_obs_dim: {full_obs_dim}")
            print(f"  base_obs_dim: {self.base_obs_dim}")
            print(f"  depth_dim: {self.depth_dim} ({self.depth_height}x{self.depth_width})")
            print(f"  depth_features: {depth_features}")
            print(f"  mlp_input_size: {mlp_input_size}")
            print(f"  normalize_base_obs: {self.normalize_base_obs}")
            print(f"  depth_augmentation: {self.use_depth_aug}")
        
        def _init_depth_aug(self, device: str):
            """Lazy initialization of DepthAug (needs device info)."""
            if self.depth_aug is None and self.use_depth_aug:
                import warp as wp
                wp.init()
                from isaaclab_tasks.manager_based.manipulation.y2r.distillation.depth_augs import DepthAug
                self.depth_aug = DepthAug(device, self.depth_aug_config)
                print(f"[DepthResNetStudent] DepthAug initialized on {device}")

        @_compile_disable
        def _apply_depth_aug_eager(self, depth_img: torch.Tensor) -> torch.Tensor:
            """Run Warp depth augmentation outside torch.compile tracing."""
            depth_3d = depth_img.squeeze(1)
            depth_3d = self.depth_aug.augment(depth_3d)
            return depth_3d.unsqueeze(1)

        @_compile_disable
        def _apply_obs_rms_eager(self, base_obs: torch.Tensor) -> torch.Tensor:
            """Run obs RMS outside torch.compile tracing to avoid CUDAGraph overwrite issues."""
            return self.running_mean_std(base_obs)
        
        def forward(self, obs_dict: Dict[str, torch.Tensor]):
            """Forward pass.
            
            Args:
                obs_dict: Dictionary containing:
                    - 'obs': Full observation tensor with depth packed at end
                             (B, base_obs_dim + depth_dim)
                    
            Returns:
                mu, sigma, value, states (for continuous actions)
            """
            full_obs = obs_dict['obs']
            B = full_obs.shape[0]
            
            # Slice depth from end of obs (depth is ALWAYS at end)
            base_obs = full_obs[:, :-self.depth_dim]
            depth_flat = full_obs[:, -self.depth_dim:]
            
            # Normalize base_obs only (depth is already normalized to [0,1]).
            # Running stats update is explicitly controlled via apply_obs_rms_update.
            if self.normalize_base_obs:
                base_obs_dtype = base_obs.dtype
                if self.apply_obs_rms_update:
                    self.running_mean_std.train()
                else:
                    self.running_mean_std.eval()
                if base_obs.dtype != torch.float32:
                    base_obs = base_obs.float()
                base_obs = self._apply_obs_rms_eager(base_obs)
                if base_obs.dtype != base_obs_dtype:
                    base_obs = base_obs.to(base_obs_dtype)
            
            # Reshape depth to image: (B, H*W) → (B, 1, H, W)
            depth_img = depth_flat.view(B, 1, self.depth_height, self.depth_width)
            
            # Apply depth augmentation when explicitly enabled.
            if self.apply_depth_aug and self.use_depth_aug:
                self._init_depth_aug(str(depth_img.device))
                if self.depth_aug is not None:
                    depth_img = self._apply_depth_aug_eager(depth_img)
            
            # Store post-augmentation depth for video recording
            self.last_depth_img = depth_img.detach()
            
            # Encode depth
            depth_features = self.depth_encoder(depth_img)  # (B, depth_features)
            
            # Concatenate: normalized base_obs + depth_features
            combined = torch.cat([base_obs, depth_features], dim=-1)
            
            # MLP forward
            out = self.actor_mlp(combined)
            
            # Value
            value = self.value_act(self.value(out))
            
            # Policy (continuous)
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
                self.is_continuous = True  # Default to continuous
                self.is_discrete = False
                self.is_multi_discrete = False
                self.fixed_sigma = "fixed"
                self.space_config = {
                    'mu_activation': 'None',
                    'sigma_activation': 'None',
                    'mu_init': {'name': 'default'},
                    'sigma_init': {'name': 'const_initializer', 'val': 0},
                }
        
        def _build_mlp(
            self,
            input_size: int,
            units: list,
            activation: str,
            dense_func,
            norm_func_name: str = None,
        ) -> nn.Sequential:
            """Build MLP layers."""
            layers = []
            in_size = input_size
            
            for unit in units:
                layers.append(dense_func(in_size, unit))
                layers.append(self.activations_factory.create(activation))
                if norm_func_name == 'layer_norm':
                    layers.append(nn.LayerNorm(unit))
                elif norm_func_name == 'batch_norm':
                    layers.append(nn.BatchNorm1d(unit))
                in_size = unit
            
            return nn.Sequential(*layers)
