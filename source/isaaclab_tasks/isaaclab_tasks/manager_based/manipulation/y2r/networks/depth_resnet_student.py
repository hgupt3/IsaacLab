# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Small ResNet-style Depth Encoder for Student Policy Distillation.

This module implements a student policy network that processes wrist camera
depth images through a small ResNet encoder and concatenates features with
proprioceptive and point cloud observations.

Architecture:
    - Depth encoder: Conv3x3 s=2 → ResBlock(32) → ResBlock(64,s=2) → ResBlock(128,s=2) → AvgPool
    - Feature fusion: depth_features (128) + proprio + student_perception + student_targets
    - Policy MLP: [512, 256, 128] → actions

Usage in YAML config:
    network:
      name: depth_resnet_student
      depth_encoder:
        resolution: 64
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


# =============================================================================
# Residual Block
# =============================================================================

class ResBlock(nn.Module):
    """Residual block with optional downsampling and channel change.
    
    Structure:
        x → Conv3x3 → LayerNorm → ReLU → Conv3x3 → LayerNorm → (+) → ReLU
            └─────────────── skip connection (1x1 conv if needed) ─┘
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Stride for first conv (1 = same size, 2 = downsample)
    """
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, 
            stride=stride, padding=1, bias=False
        )
        self.ln1 = nn.GroupNorm(1, out_channels)  # GroupNorm(1) = LayerNorm for conv
        
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.ln2 = nn.GroupNorm(1, out_channels)
        
        # Skip connection with projection if dimensions change
        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(1, out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        
        out = self.conv1(x)
        out = self.ln1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.ln2(out)
        
        out = out + identity
        out = F.relu(out)
        
        return out


# =============================================================================
# Small ResNet Encoder
# =============================================================================

class SmallResNet(nn.Module):
    """Small ResNet encoder for 64x64 depth images.
    
    Architecture:
        Input: (B, 1, 64, 64) depth image
        Conv 3x3, s=2: 1 → 32 channels, 64x64 → 32x32
        ResBlock: 32 → 32 channels, 32x32
        ResBlock s=2: 32 → 64 channels, 32x32 → 16x16
        ResBlock s=2: 64 → 128 channels, 16x16 → 8x8
        AdaptiveAvgPool: 8x8 → 1x1
        Output: (B, 128) features
    
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
        
        # Initial conv: 1 → 32, stride 2 (64x64 → 32x32)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(1, c1),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks
        self.block1 = ResBlock(c1, c1, stride=1)   # 32x32, 32ch
        self.block2 = ResBlock(c1, c2, stride=2)   # 32x32 → 16x16, 64ch
        self.block3 = ResBlock(c2, c3, stride=2)   # 16x16 → 8x8, 128ch
        
        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.out_features = c3
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Depth image tensor (B, 1, H, W)
            
        Returns:
            Feature tensor (B, out_features)
        """
        x = self.stem(x)      # (B, 32, 32, 32)
        x = self.block1(x)    # (B, 32, 32, 32)
        x = self.block2(x)    # (B, 64, 16, 16)
        x = self.block3(x)    # (B, 128, 8, 8)
        x = self.pool(x)      # (B, 128, 1, 1)
        x = x.flatten(1)      # (B, 128)
        return x


# =============================================================================
# A2C Network Builder for rl_games
# =============================================================================

class DepthResNetStudentBuilder(NetworkBuilder):
    """Network builder for student policy with depth encoder.
    
    This builder creates a network that:
    1. Encodes depth images through SmallResNet
    2. Concatenates with proprioceptive observations
    3. Feeds through MLP to produce actions
    
    The network expects observations to be passed in obs_dict with:
    - 'obs': Concatenated proprio + student_perception + student_targets
    - 'camera': Flattened depth image from student_camera group
    """
    
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)
    
    def load(self, params: Dict[str, Any]):
        self.params = params
    
    def build(self, name: str, **kwargs) -> nn.Module:
        return DepthResNetStudentBuilder.Network(self.params, **kwargs)
    
    class Network(NetworkBuilder.BaseNetwork):
        """Student policy network with depth encoder."""
        
        def __init__(self, params: Dict[str, Any], **kwargs):
            NetworkBuilder.BaseNetwork.__init__(self)
            
            # Extract kwargs
            actions_num = kwargs.pop('actions_num')
            input_shape = kwargs.pop('input_shape')  # Shape of 'obs' (proprio + pc)
            self.value_size = kwargs.pop('value_size', 1)
            self.num_seqs = kwargs.pop('num_seqs', 1)
            
            # Load params
            self.load(params)
            
            # Depth encoder config
            depth_cfg = params.get('depth_encoder', {})
            self.depth_resolution = depth_cfg.get('resolution', 64)
            self.depth_channels = depth_cfg.get('channels', [32, 64, 128])
            
            # Build depth encoder
            self.depth_encoder = SmallResNet(
                in_channels=1,
                channels=self.depth_channels
            )
            depth_features = self.depth_encoder.out_features  # 128
            
            
            # Adjust input shape to include depth features
            obs_dim = input_shape[0] if isinstance(input_shape, tuple) else input_shape
            mlp_input_size = obs_dim + depth_features
            
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
                
                if self.fixed_sigma:
                    self.sigma = nn.Parameter(
                        torch.zeros(actions_num, requires_grad=True, dtype=torch.float32),
                        requires_grad=True
                    )
                else:
                    self.sigma = nn.Linear(out_size, actions_num)
                
                self.sigma_act = self.activations_factory.create(
                    self.space_config['sigma_activation']
                )
                
                # Initialize
                mu_init = self.init_factory.create(**self.space_config['mu_init'])
                sigma_init = self.init_factory.create(**self.space_config['sigma_init'])
                mu_init(self.mu.weight)
                if self.fixed_sigma:
                    sigma_init(self.sigma)
                else:
                    sigma_init(self.sigma.weight)
            
            # Initialize MLP weights
            mlp_init = self.init_factory.create(**self.initializer)
            for m in self.actor_mlp.modules():
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
        
        def forward(self, obs_dict: Dict[str, torch.Tensor]):
            """Forward pass.
            
            Args:
                obs_dict: Dictionary containing:
                    - 'obs': Proprio + student_perception + student_targets (B, obs_dim)
                    - 'camera': Flattened depth image (B, resolution*resolution)
                    
            Returns:
                mu, sigma, value, states (for continuous actions)
            """
            obs = obs_dict['obs']
            
            # Process depth if available
            if 'camera' in obs_dict:
                depth_flat = obs_dict['camera']
                B = depth_flat.shape[0]
                
                # Reshape to image: (B, res*res) → (B, 1, res, res)
                depth_img = depth_flat.view(B, 1, self.depth_resolution, self.depth_resolution)
                
                # Encode
                depth_features = self.depth_encoder(depth_img)  # (B, 128)
                
                # Concatenate with observations
                obs = torch.cat([obs, depth_features], dim=-1)
            
            # MLP forward
            out = self.actor_mlp(obs)
            
            # Value
            value = self.value_act(self.value(out))
            
            # Policy (continuous)
            if self.is_continuous:
                mu = self.mu_act(self.mu(out))
                if self.fixed_sigma:
                    sigma = self.sigma_act(self.sigma)
                else:
                    sigma = self.sigma_act(self.sigma(out))
                
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
                    self.fixed_sigma = self.space_config['fixed_sigma']
            else:
                self.is_continuous = True  # Default to continuous
                self.is_discrete = False
                self.is_multi_discrete = False
                self.fixed_sigma = True
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

