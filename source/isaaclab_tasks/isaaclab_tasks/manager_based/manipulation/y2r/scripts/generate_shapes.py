#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Generate procedural shapes for Y2R trajectory task.

This script generates random "grown" objects by hierarchically attaching
primitives. The shapes are exported as USD files with convex decomposition
for proper collision detection.

Usage:
    # Generate shapes (reads num_shapes from base.yaml config)
    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/y2r/scripts/generate_shapes.py
    
    # Force regeneration even if shapes exist
    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/y2r/scripts/generate_shapes.py --regenerate
    
    # Generate specific number of shapes
    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/y2r/scripts/generate_shapes.py --num-shapes 200
    
    # Use specific seed for reproducibility
    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/y2r/scripts/generate_shapes.py --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Generate procedural shapes for Y2R task",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--regenerate", "-r",
        action="store_true",
        help="Force regeneration even if shapes already exist",
    )
    parser.add_argument(
        "--num-shapes", "-n",
        type=int,
        default=None,
        help="Number of shapes to generate (default: from base.yaml config)",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()
    
    # Import after parsing to avoid slow imports if just showing help
    print("Initializing Isaac Sim...")
    
    # Must instantiate SimulationApp BEFORE importing any Isaac Lab modules
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": True})
    
    # Now we can import Isaac Lab modules
    from dataclasses import asdict
    
    # Import from y2r package (works because we're in the package)
    from isaaclab_tasks.manager_based.manipulation.y2r.config_loader import get_config
    from isaaclab_tasks.manager_based.manipulation.y2r.procedural_shapes import generate_procedural_shapes
    
    # Get y2r directory (this script is in y2r/scripts/)
    y2r_dir = Path(__file__).parent.parent
    
    # Load config from base.yaml
    cfg = get_config()
    proc_cfg = asdict(cfg.procedural_objects)
    
    # Override with command line args
    if args.regenerate:
        proc_cfg["regenerate"] = True
    if args.num_shapes is not None:
        proc_cfg["generation"]["num_shapes"] = args.num_shapes
    if args.seed is not None:
        proc_cfg["generation"]["seed"] = args.seed
    
    # Check if shapes already exist
    asset_dir = y2r_dir / proc_cfg.get("asset_dir", "assets/procedural")
    existing = list(asset_dir.glob("shape_*.usd")) + list(asset_dir.glob("shape_*.obj"))
    num_shapes = proc_cfg.get("generation", {}).get("num_shapes", 100)
    
    if existing and len(existing) >= num_shapes and not proc_cfg.get("regenerate", False):
        print(f"Found {len(existing)} existing shapes in {asset_dir}")
        print("Use --regenerate to force regeneration")
        simulation_app.close()
        return
    
    # Generate shapes
    generated = generate_procedural_shapes(proc_cfg, y2r_dir)
    
    print(f"\nDone! Generated {len(generated)} shapes.")
    print(f"Location: {asset_dir}")
    print("\nYou can now run training:")
    print("  ./isaaclab.sh -p scripts/rl_games/train.py task=Isaac-Y2R-Kuka-Allegro-v0")
    
    # Clean up
    simulation_app.close()


if __name__ == "__main__":
    main()

