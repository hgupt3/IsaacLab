# UR5e Gemini WSG50 Asset

This asset assembles the existing Y2R UR5e/Gemini 305 arm with a WSG50 gripper,
the custom WSG mount, the custom Gemini mount, and duplicated soft gripper
fingers. The editable source of truth for placement is `calibration.yaml`; the
URDF is generated from that file.

The WSG mount and finger-holder CAD pieces are visual-only links attached below
the already calibrated WSG base and soft-finger links. The mount is stored as
STL and the finger holders are stored as OBJ/MTL exports. They intentionally do
not define collision geometry, so moving them in the editor does not change the
functional gripper chain or contact bodies.

`calibration.yaml` also owns the default UR5e/WSG pose, contact marker normals,
and coarse primitive collision geometry. Visual meshes remain detailed, while
the custom WSG/Gemini collision shapes are generated as boxes for IsaacLab.

## Build

```bash
python build_ur5e_gemini_wsg50.py
```

## Calibration Editor

```bash
python serve_calibration_tool.py
```

Then open `http://127.0.0.1:8765/`.

## Attribution

The WSG50 source model files come from
`caelan/pybullet-planning/models/drake/wsg_50_description`. The original README,
license, and URDF are preserved in `wsg50/source/`.
