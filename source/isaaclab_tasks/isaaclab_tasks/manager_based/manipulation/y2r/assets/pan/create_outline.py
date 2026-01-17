#!/usr/bin/env python3
"""Generate a simple cylinder outline for pan task visualization."""

from pxr import Usd, UsdGeom

# Create USD stage
stage = Usd.Stage.CreateNew("pan_outline.usd")

# Create cylinder
cylinder_prim = UsdGeom.Cylinder.Define(stage, "/cylinder")

# Set dimensions
cylinder_prim.GetRadiusAttr().Set(0.04)  # 4cm radius
cylinder_prim.GetHeightAttr().Set(0.001)  # 1mm height
cylinder_prim.GetAxisAttr().Set("Z")  # Cylinder along Z axis

# Set display color (black)
color_attr = cylinder_prim.GetDisplayColorAttr()
if not color_attr:
    color_attr = cylinder_prim.CreateDisplayColorAttr()
color_attr.Set([(0.0, 0.0, 0.0)])  # Black (RGB)

# Optional: Set opacity
opacity_attr = cylinder_prim.GetDisplayOpacityAttr()
if not opacity_attr:
    opacity_attr = cylinder_prim.CreateDisplayOpacityAttr()
opacity_attr.Set([1.0])  # Fully opaque

# Set default prim (required for referencing)
stage.SetDefaultPrim(cylinder_prim.GetPrim())

# Save
stage.GetRootLayer().Save()

print("Created pan_outline.usd")
print("  - Radius: 4cm")
print("  - Height: 1mm")
print("  - Color: Black")
