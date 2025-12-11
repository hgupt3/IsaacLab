# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Custom network architectures for Y2R trajectory task.

This module registers custom networks with rl_games when imported.
Import this module before creating the rl_games Runner to enable custom networks.
"""

from rl_games.algos_torch.model_builder import register_network

from .point_transformer import PointTransformerBuilder

# Register Point-Transformer network with rl_games
# This allows YAML config to use: network.name: point_transformer
register_network('point_transformer', PointTransformerBuilder)

__all__ = ["PointTransformerBuilder"]

