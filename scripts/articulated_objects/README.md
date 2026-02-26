# Articulated Objects

Procedural generation of hand-scale articulated objects + interactive viewer for dialing in hand grasp poses per template. Used to produce training data for the RL policy.

## Files

| File | Purpose |
|------|---------|
| `generator.py` | Procedural object generation. Each template (scissors, book, laptop, etc.) builds base + child meshes from primitives with randomized dimensions. `ArticulatedObject` dataclass holds meshes, joint params, and `get_child_transform()` / `get_meshes_at()`. |
| `hand_model.py` | LEAP hand URDF parser. Singleton `HandModel` caches link meshes, does FK, returns posed hand mesh. `FINGER_PRESETS` dict has named poses (open, closed, pinch, power, hook, flat_push). |
| `viewer.py` | Interactive viser web viewer (http://localhost:8080). Used to set hand pose per template via sliders, preview coupling, and play animations. Saves settings to `hand_defaults.json`. |
| `hand_defaults.json` | Auto-generated. Persists per-template hand settings (anchor, offset, orientation, finger preset, coupling mode). |

## Templates

| Template | Joint | Description |
|----------|-------|-------------|
| scissors | revolute | Two blades with handles, pivot in the middle |
| book | revolute | Cover hinged to spine |
| laptop | revolute | Screen hinged to base |
| tongs | revolute | Two arms joined at one end |
| drawer | prismatic | Drawer slides out of cabinet |
| door_handle | revolute | Lever handle on a door panel |
| bottle_cap | revolute | Cap twists on bottle neck |

## Viewer Usage

```bash
cd articulated_objects && conda run -n y2r python viewer.py
```

### Controls

**Object**: Template dropdown, joint slider, seed for random dimensions.

**Anchor** (normalized to child bbox, range [-1.5, 1.5]): Picks a point on the articulated part of the object. `[0,0,0]` = child centroid, `[1,0,0]` = right edge, etc. This point moves with the child as the joint articulates. The red sphere shows where it is.

**Hand Pose** (offset from anchor): Finger preset, XYZ offset in meters, Roll/Pitch/Yaw orientation. The hand is positioned at `anchor + offset`.

**Coupling Mode**: How the hand follows the child during articulation.
- **full**: Hand is rigidly attached to the child. Anchor, offset, and orientation all rotate with it.
- **position_only**: Hand position tracks the anchor on the child, but offset direction and orientation stay in world frame.
- **none**: Hand stays frozen at the default-joint position.

**Animation** (Play button): 5-phase sequence with random start/end poses.

For **liftable** objects (scissors, tongs):
1. **Approach**: Hand moves from random pose to contact pose
2. **Lift**: Object + hand lift to random air pose
3. **Manipulate**: Joint cycles open/close x2, hand follows per coupling
4. **Lower**: Object + hand return to table
5. **Retreat**: Hand moves to random end pose

For **table** objects (book, laptop, drawer, door_handle, bottle_cap):
1. **Approach**: Hand moves from random pose to contact pose
2. **Manipulate**: Joint cycles open/close x2 on the table, hand follows per coupling
3. **Retreat**: Hand moves to random end pose

Settings auto-save to `hand_defaults.json` (debounced 0.5s) and persist across sessions.
