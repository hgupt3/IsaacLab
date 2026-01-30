#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""PointNet+TNet encoder network for rl_games."""

from __future__ import annotations

from typing import Dict, Any

import torch
import torch.nn as nn

from rl_games.algos_torch.network_builder import A2CBuilder, NetworkBuilder


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


class PointNetTNetBuilder(A2CBuilder):
    """Builder for PointNet+TNet encoder with standard actor/critic trunks."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.params = None

    class Network(A2CBuilder.Network):
        """PointNet-style network with two T-Nets and global max pooling."""

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

            self.has_space = "space" in params
            if self.has_space:
                self.is_multi_discrete = "multi_discrete" in params["space"]
                self.is_discrete = "discrete" in params["space"]
                self.is_continuous = "continuous" in params["space"]
                if self.is_continuous:
                    self.space_config = params["space"]["continuous"]
                    self.fixed_sigma = self.space_config.get("fixed_sigma", True)
                elif self.is_discrete:
                    self.space_config = params["space"]["discrete"]
                elif self.is_multi_discrete:
                    self.space_config = params["space"]["multi_discrete"]
            else:
                self.is_continuous = True
                self.is_discrete = False
                self.is_multi_discrete = False
                self.fixed_sigma = True
                self.space_config = {
                    "mu_activation": "None",
                    "sigma_activation": "None",
                    "mu_init": {"name": "default"},
                    "sigma_init": {"name": "const_initializer", "val": 0},
                }

        def __init__(self, params, **kwargs):
            NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)

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

            point_dim = self.num_timesteps * 3
            self.tnet1 = PointNetTNet(point_dim, hidden_dim=16)
            self.tnet2 = PointNetTNet(32, hidden_dim=16)
            self.mlp1 = nn.Sequential(
                nn.Linear(point_dim, 32),
                nn.ELU(),
            )
            self.mlp2 = nn.Sequential(
                nn.Linear(32, 64),
                nn.ELU(),
            )

            if self.separate:
                self.critic_tnet1 = PointNetTNet(point_dim, hidden_dim=16)
                self.critic_tnet2 = PointNetTNet(32, hidden_dim=16)
                self.critic_mlp1 = nn.Sequential(
                    nn.Linear(point_dim, 32),
                    nn.ELU(),
                )
                self.critic_mlp2 = nn.Sequential(
                    nn.Linear(32, 64),
                    nn.ELU(),
                )

            trunk_input_dim = self.proprio_dim + 64
            self.actor_trunk, trunk_output_dim = self._build_trunk(trunk_input_dim)
            if self.separate:
                self.critic_trunk, _ = self._build_trunk(trunk_input_dim)
            else:
                self.critic_trunk = self.actor_trunk

            self.actor_output = nn.Linear(trunk_output_dim, self.actions_num)
            self.critic_output = nn.Linear(trunk_output_dim, self.value_size)
            self.value_act = self.activations_factory.create(self.value_activation)

            self.mu_act = self.activations_factory.create(self.space_config.get("mu_activation", "None"))
            self.sigma_act = self.activations_factory.create(self.space_config.get("sigma_activation", "None"))

            if self.fixed_sigma:
                self.sigma = nn.Parameter(
                    torch.zeros(self.actions_num, requires_grad=True, dtype=torch.float32),
                    requires_grad=True,
                )
            else:
                self.sigma_head = nn.Linear(trunk_output_dim, self.actions_num)

            self._init_weights()

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
                if self.fixed_sigma:
                    sigma_init(self.sigma)
                else:
                    sigma_init(self.sigma_head.weight)

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

            point_feat = point_obs.reshape(B, self.num_points, -1)  # (B, P, num_timesteps * 3)
            trans1 = self.tnet1(point_feat)
            point_feat = torch.bmm(point_feat, trans1)

            point_feat = self.mlp1(point_feat)  # (B, P, 64)
            trans2 = self.tnet2(point_feat)
            point_feat = torch.bmm(point_feat, trans2)

            point_feat = self.mlp2(point_feat)  # (B, P, 64)
            global_feat = point_feat.max(dim=1).values  # (B, 64)

            trunk_input = torch.cat([proprio_obs, global_feat], dim=-1)
            actor_latent = self.actor_trunk(trunk_input)
            if self.separate:
                critic_point_feat = point_obs.reshape(B, self.num_points, -1)
                critic_trans1 = self.critic_tnet1(critic_point_feat)
                critic_point_feat = torch.bmm(critic_point_feat, critic_trans1)

                critic_point_feat = self.critic_mlp1(critic_point_feat)
                critic_trans2 = self.critic_tnet2(critic_point_feat)
                critic_point_feat = torch.bmm(critic_point_feat, critic_trans2)

                critic_point_feat = self.critic_mlp2(critic_point_feat)
                critic_global_feat = critic_point_feat.max(dim=1).values

                critic_trunk_input = torch.cat([proprio_obs, critic_global_feat], dim=-1)
                critic_latent = self.critic_trunk(critic_trunk_input)
            else:
                critic_latent = actor_latent

            mu = self.mu_act(self.actor_output(actor_latent))
            value = self.value_act(self.critic_output(critic_latent))

            if self.fixed_sigma:
                sigma = self.sigma_act(self.sigma)
            else:
                sigma = self.sigma_act(self.sigma_head(actor_latent))

            return mu, mu * 0 + sigma, value, None

        def is_rnn(self) -> bool:
            return False

    def _inject_params(self, params):
        from ..config_loader import get_config
        import os

        mode = os.getenv("Y2R_MODE")
        y2r_cfg = get_config(mode)
        params["num_points"] = y2r_cfg.observations.num_points
        num_current = y2r_cfg.observations.history.object_pc
        num_future = y2r_cfg.trajectory.window_size * y2r_cfg.observations.history.targets
        params["num_timesteps"] = num_current + num_future

    def load(self, params):
        self._inject_params(params)
        self.params = params

    def build(self, name: str, **kwargs) -> Network:
        if self.params is None:
            self.params = {}
            self._inject_params(self.params)
        return self.Network(self.params, **kwargs)
