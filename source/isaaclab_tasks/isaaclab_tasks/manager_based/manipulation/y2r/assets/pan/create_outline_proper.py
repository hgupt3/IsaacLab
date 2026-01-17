#!/usr/bin/env python3
"""Generate cylinder outline for pan task using Isaac Lab's MeshConverter."""

from pathlib import Path

# Must instantiate SimulationApp BEFORE importing Isaac Lab
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import trimesh
from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from isaaclab.sim.schemas import (
    MassPropertiesCfg,
    RigidBodyPropertiesCfg,
    CollisionPropertiesCfg,
)

# Create cylinder mesh
# Trimesh cylinder is along Z axis by default
cylinder = trimesh.creation.cylinder(
    radius=0.04,      # 4cm radius
    height=0.001,     # 1mm height
    sections=32,      # Smooth circular cross-section
)

# Set color to black (RGB: 0, 0, 0)
cylinder.visual.vertex_colors = [0, 0, 0, 255]  # Black, fully opaque

# Export to temporary OBJ
output_dir = Path(__file__).parent
temp_obj = output_dir / "pan_outline_temp.obj"
cylinder.export(str(temp_obj))

print(f"Exporting cylinder to {output_dir / 'pan_outline.usd'}...")

# Convert to USD with Isaac Lab's converter
converter_cfg = MeshConverterCfg(
    asset_path=str(temp_obj),
    usd_dir=str(output_dir),
    usd_file_name="pan_outline.usd",
    force_usd_conversion=True,
    make_instanceable=False,
    mass_props=MassPropertiesCfg(mass=0.01),  # Very light (10g)
    rigid_props=RigidBodyPropertiesCfg(
        disable_gravity=False,
        kinematic_enabled=False,
    ),
    collision_props=CollisionPropertiesCfg(
        collision_enabled=False,  # Outline shouldn't collide
    ),
)
converter = MeshConverter(converter_cfg)

# Clean up temp OBJ
temp_obj.unlink(missing_ok=True)

print("Created pan_outline.usd")
print("  - Radius: 4cm")
print("  - Height: 1mm")
print("  - Color: Black")
print("  - Collision: Disabled")

# Clean up
simulation_app.close()
