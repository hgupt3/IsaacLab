# Copyright (c) 2024, Harsh Gupta
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Distillation module for teacher-student policy transfer."""

from .distill_agent import DistillAgent, register_distill_agent
from .depth_augs import DepthAug

__all__ = ["DistillAgent", "register_distill_agent", "DepthAug"]
