# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Custom network architectures for Y2R trajectory task.

This module registers custom networks with rl_games when imported.
Import this module before creating the rl_games Runner to enable custom networks.
"""

from rl_games.algos_torch.model_builder import register_network

from .point_transformer import PointTransformerBuilder
from .pointnet_tnet import PointNetTNetBuilder
from .depth_resnet_student import DepthResNetStudentBuilder
from .depth_point_transformer_student import DepthPointTransformerStudentBuilder

# Register Point-Transformer network with rl_games
# This allows YAML config to use: network.name: point_transformer
register_network('point_transformer', PointTransformerBuilder)
register_network('pointnet_tnet', PointNetTNetBuilder)

# Register Depth ResNet Student network for distillation
# This allows YAML config to use: network.name: depth_resnet_student
register_network('depth_resnet_student', DepthResNetStudentBuilder)

# Register Depth + Point Transformer Student network for distillation
# This allows YAML config to use: network.name: depth_point_transformer_student
register_network('depth_point_transformer_student', DepthPointTransformerStudentBuilder)

__all__ = ["PointTransformerBuilder", "PointNetTNetBuilder", "DepthResNetStudentBuilder", "DepthPointTransformerStudentBuilder"]

