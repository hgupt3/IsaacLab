# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Y2R: Trajectory following environments for dexterous manipulation.

This module implements trajectory following tasks where the robot must pick up an object,
follow a smooth trajectory through waypoints, and place it at a goal location.
"""

from .trajectory_manager import TrajectoryManager
