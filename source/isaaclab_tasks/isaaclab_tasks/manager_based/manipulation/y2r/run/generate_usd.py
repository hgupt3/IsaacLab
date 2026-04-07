"""Test URDF → USD conversion and inspect the UR5e + LEAP Hand robot.

Usage:
    # Headless: print joint/body info only
    python y2r_sim/run/generate_usd.py

    # Visual: open livestream viewer with debug axes
    python y2r_sim/run/generate_usd.py --view

Viewer legend (--view mode):
    ur5e_link_6     — dim RGB axes (R=X, G=Y, B=Z)
    palm_frame      — bright RGB axes (COMPUTED from ur5e_link_6 + offset)
    camera_frame    — bright RGB axes, 8cm (COMPUTED from ur5e_link_6 + camera offset)
    fingertips (4×) — small bright axes (COMPUTED from link_3 + offset)
    pad normals     — yellow arrows on link_3 + link_2 (direction finger pad faces)
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test UR5e + LEAP Hand URDF conversion.")
parser.add_argument("--view", action="store_true", help="Open livestream viewer to visually inspect.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.view:
    args.livestream = 2

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import math
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.utils.math import combine_frame_transforms, quat_apply, quat_from_euler_xyz
from isaaclab_assets.robots import UR5E_LEAP_CFG

# =============================================================================
# Offsets from URDF joint origins (virtual frames merged away by merge_fixed_joints)
# All offsets are relative to ur5e_link_6 (palm_link is merged into it).
# =============================================================================

# palm_frame offset: ur5e_link_6 → virtual palm frame
# Computed: R_flange × old_palm_offset + p_palm_in_link6
PALM_FRAME_OFFSET_POS = torch.tensor([-0.008, -0.0345, 0.11])
PALM_FRAME_OFFSET_QUAT = quat_from_euler_xyz(
    torch.tensor([0.0]),
    torch.tensor([-math.pi / 2]),
    torch.tensor([math.pi / 2]),
).squeeze(0)  # (4,)

# Camera offset: ur5e_link_6 → camera optical frame
# Computed: R_flange × old_camera_offset + p_palm_in_link6
# Rotation is Ry(180°): camera -Z (look) → +Z in link_6, +Y (up) → +Y in link_6
CAMERA_OFFSET_POS = torch.tensor([0.009, -0.0705, -0.0274])
CAMERA_OFFSET_QUAT = quat_from_euler_xyz(
    torch.tensor([0.0]),
    torch.tensor([math.pi]),
    torch.tensor([0.0]),
).squeeze(0)  # (4,)

TIP_OFFSETS = {
    "index_tip":  ("index_link_3",  torch.tensor([0.0, -0.048, 0.015])),
    "middle_tip": ("middle_link_3", torch.tensor([0.0, -0.048, 0.015])),
    "ring_tip":   ("ring_link_3",   torch.tensor([0.0, -0.048, 0.015])),
    "thumb_tip":  ("thumb_link_3",  torch.tensor([0.0, -0.06, -0.015])),
}

# Body used for palm/camera frame computation (palm_link merged into this)
PALM_BODY_NAME = "ur5e_link_6"

# Create simulation
sim_cfg = sim_utils.SimulationCfg(dt=1 / 120.0, device="cpu")
sim = sim_utils.SimulationContext(sim_cfg)
sim.set_camera_view(eye=(1.5, 1.5, 1.0), target=(0.0, 0.0, 0.3))

# Spawn the robot
robot_cfg = UR5E_LEAP_CFG.replace(prim_path="/World/Robot")
robot = Articulation(robot_cfg)

# Initialize physics
sim.reset()

# Apply initial joint positions
robot.reset()

print("\n" + "=" * 60)
print("UR5e + LEAP Hand — URDF Conversion Report")
print("=" * 60)

print(f"\nNumber of joints:  {robot.num_joints}")
print(f"Number of bodies:  {robot.num_bodies}")

print("\n--- Joint names ---")
for i, name in enumerate(robot.joint_names):
    print(f"  [{i:2d}] {name}")

print("\n--- Body names ---")
for i, name in enumerate(robot.body_names):
    print(f"  [{i:2d}] {name}")

print("\n--- Joint limits (degrees) ---")
limits = robot.root_physx_view.get_dof_limits().squeeze(0)
for i, name in enumerate(robot.joint_names):
    lo = torch.rad2deg(limits[i, 0]).item()
    hi = torch.rad2deg(limits[i, 1]).item()
    print(f"  {name:25s}  [{lo:8.2f}, {hi:8.2f}]")

print("\n--- Initial joint positions (degrees) ---")
pos = robot.data.joint_pos.squeeze(0)
for i, name in enumerate(robot.joint_names):
    deg = torch.rad2deg(pos[i]).item()
    print(f"  {name:25s}  {deg:8.2f}")

print("\n--- USD Prim Hierarchy ---")
stage = sim.stage
root_prim = stage.GetPrimAtPath("/World/Robot")
def print_tree(prim, indent=0):
    typ = prim.GetTypeName()
    if typ in ("Xform", ""):
        print(f"  {'  ' * indent}{prim.GetName()}  -> {prim.GetPath()}")
        for child in prim.GetChildren():
            print_tree(child, indent + 1)
print_tree(root_prim)

# =============================================================================
# Frame poses — print world poses for key frames
# =============================================================================

# Helper: look up a body by name, returns (index, found) — safe for optional bodies
def find_body(name):
    ids = robot.find_bodies(name)[0]
    if len(ids) == 0:
        return None
    return ids[0]

robot.update(sim.cfg.dt)

# Unit vectors for axis computation
X_AXIS = torch.tensor([1.0, 0.0, 0.0])
Y_AXIS = torch.tensor([0.0, 1.0, 0.0])
Z_AXIS = torch.tensor([0.0, 0.0, 1.0])

def quat_apply_single(quat, vec):
    """Apply quaternion rotation to a single vector."""
    return quat_apply(quat.unsqueeze(0), vec.unsqueeze(0)).squeeze(0)


# --- Real body: ur5e_link_6 (palm_link merged into this) ---
palm_body_idx = find_body(PALM_BODY_NAME)

print("\n--- Frame poses (world) ---")

if palm_body_idx is not None:
    palm_pos = robot.data.body_pos_w[0, palm_body_idx]
    palm_quat = robot.data.body_quat_w[0, palm_body_idx]
    print(f"\n  {PALM_BODY_NAME} (real body — palm_link merged here):")
    print(f"    pos:  ({palm_pos[0]:.5f}, {palm_pos[1]:.5f}, {palm_pos[2]:.5f})")
    print(f"    quat: ({palm_quat[0]:.5f}, {palm_quat[1]:.5f}, {palm_quat[2]:.5f}, {palm_quat[3]:.5f})  [wxyz]")

    # Compute palm_frame from offset
    pf_pos, pf_quat = combine_frame_transforms(
        palm_pos.unsqueeze(0), palm_quat.unsqueeze(0),
        PALM_FRAME_OFFSET_POS.unsqueeze(0), PALM_FRAME_OFFSET_QUAT.unsqueeze(0),
    )
    pf_pos = pf_pos.squeeze(0)
    pf_quat = pf_quat.squeeze(0)
    print(f"\n  palm_frame (COMPUTED from {PALM_BODY_NAME} + palm offset):")
    print(f"    pos:  ({pf_pos[0]:.5f}, {pf_pos[1]:.5f}, {pf_pos[2]:.5f})")
    print(f"    quat: ({pf_quat[0]:.5f}, {pf_quat[1]:.5f}, {pf_quat[2]:.5f}, {pf_quat[3]:.5f})  [wxyz]")

    # Compute camera_frame from offset
    cf_pos, cf_quat = combine_frame_transforms(
        palm_pos.unsqueeze(0), palm_quat.unsqueeze(0),
        CAMERA_OFFSET_POS.unsqueeze(0), CAMERA_OFFSET_QUAT.unsqueeze(0),
    )
    cf_pos = cf_pos.squeeze(0)
    cf_quat = cf_quat.squeeze(0)
    print(f"\n  camera_frame (COMPUTED from {PALM_BODY_NAME} + camera offset):")
    print(f"    pos:  ({cf_pos[0]:.5f}, {cf_pos[1]:.5f}, {cf_pos[2]:.5f})")
    print(f"    quat: ({cf_quat[0]:.5f}, {cf_quat[1]:.5f}, {cf_quat[2]:.5f}, {cf_quat[3]:.5f})  [wxyz]")
else:
    pf_pos = pf_quat = cf_pos = cf_quat = None
    print(f"\n  {PALM_BODY_NAME}: NOT FOUND")

# --- Computed tip positions ---
tip_poses = {}
for tip_name, (parent_name, offset) in TIP_OFFSETS.items():
    parent_idx = find_body(parent_name)
    if parent_idx is not None:
        parent_pos = robot.data.body_pos_w[0, parent_idx]
        parent_quat = robot.data.body_quat_w[0, parent_idx]
        tip_pos = parent_pos + quat_apply_single(parent_quat, offset)
        tip_poses[tip_name] = (tip_pos, parent_quat, parent_idx)
        print(f"\n  {tip_name} (COMPUTED from {parent_name} + offset):")
        print(f"    pos:  ({tip_pos[0]:.5f}, {tip_pos[1]:.5f}, {tip_pos[2]:.5f})")
    else:
        print(f"\n  {tip_name}: parent {parent_name} NOT FOUND")

print("\n" + "=" * 60)
print("Conversion successful!")
print("=" * 60)

# =============================================================================
# Visual debug draw (--view mode)
# =============================================================================

if args.view:
    # Initialize debug draw
    debug_draw = None
    try:
        import isaacsim.util.debug_draw._debug_draw as omni_debug_draw
    except ModuleNotFoundError:
        try:
            from omni.isaac.debug_draw import _debug_draw as omni_debug_draw
        except ModuleNotFoundError:
            from omni.debugdraw import get_debug_draw_interface
            debug_draw = get_debug_draw_interface()
            omni_debug_draw = None
    if omni_debug_draw is not None:
        debug_draw = omni_debug_draw.acquire_debug_draw_interface()

    # -- Drawing config --
    FRAME_AXIS_LEN = 0.12  # 12cm
    TIP_AXIS_LEN = 0.04    # 4cm for fingertips
    CAM_AXIS_LEN = 0.08    # 8cm for camera

    print("\nViewer running at http://localhost:8211")
    print("Frame axes:")
    print(f"  {PALM_BODY_NAME:15s} = dim   RGB  (R=X, G=Y, B=Z)")
    print("  palm_frame      = bright RGB  (COMPUTED from ur5e_link_6 + palm offset)")
    print("  camera_frame    = bright RGB  (COMPUTED from ur5e_link_6 + camera offset, 8cm axes)")
    print("  fingertips (4×) = small bright axes (COMPUTED from link_3 + offset)")
    print("  pad normals     = YELLOW arrows on link_3 + link_2 (direction finger pad faces)")
    print("Press Ctrl+C to exit.\n")

    def draw_axes_at(pos, quat, axis_len, line_width, dim=False, color_override=None):
        """Append RGB axis lines at a given pose to the draw lists."""
        brightness = 0.5 if dim else 1.0
        alpha = 0.7 if dim else 1.0
        if color_override is not None:
            # Use same color for all 3 axes
            for axis in [X_AXIS, Y_AXIS, Z_AXIS]:
                end = pos + quat_apply_single(quat, axis) * axis_len
                starts.append(pos.tolist())
                ends.append(end.tolist())
                colors.append((*color_override, alpha))
                sizes.append(line_width)
        else:
            for axis, rgb in [(X_AXIS, (brightness, 0.0, 0.0)),
                              (Y_AXIS, (0.0, brightness, 0.0)),
                              (Z_AXIS, (0.0, 0.0, brightness))]:
                end = pos + quat_apply_single(quat, axis) * axis_len
                starts.append(pos.tolist())
                ends.append(end.tolist())
                colors.append((*rgb, alpha))
                sizes.append(line_width)

    while simulation_app.is_running():
        sim.step()
        robot.update(sim.cfg.dt)

        if debug_draw is not None:
            debug_draw.clear_lines()
            starts = []
            ends = []
            colors = []
            sizes = []

            # ur5e_link_6 (dim RGB axes)
            if palm_body_idx is not None:
                pl_pos = robot.data.body_pos_w[0, palm_body_idx]
                pl_quat = robot.data.body_quat_w[0, palm_body_idx]
                draw_axes_at(pl_pos, pl_quat, FRAME_AXIS_LEN, 3.0, dim=True)

                # palm_frame (bright RGB axes, computed from ur5e_link_6 + palm offset)
                pf_p, pf_q = combine_frame_transforms(
                    pl_pos.unsqueeze(0), pl_quat.unsqueeze(0),
                    PALM_FRAME_OFFSET_POS.unsqueeze(0), PALM_FRAME_OFFSET_QUAT.unsqueeze(0),
                )
                draw_axes_at(pf_p.squeeze(0), pf_q.squeeze(0), FRAME_AXIS_LEN, 5.0, dim=False)

                # camera_frame (RGB axes, computed from ur5e_link_6 + camera offset)
                cf_p, cf_q = combine_frame_transforms(
                    pl_pos.unsqueeze(0), pl_quat.unsqueeze(0),
                    CAMERA_OFFSET_POS.unsqueeze(0), CAMERA_OFFSET_QUAT.unsqueeze(0),
                )
                draw_axes_at(cf_p.squeeze(0), cf_q.squeeze(0), CAM_AXIS_LEN, 4.0, dim=False)

            # Fingertips (small bright axes, computed from link_3 + offset)
            for tip_name, (parent_name, offset) in TIP_OFFSETS.items():
                parent_idx = find_body(parent_name)
                if parent_idx is not None:
                    p_pos = robot.data.body_pos_w[0, parent_idx]
                    p_quat = robot.data.body_quat_w[0, parent_idx]
                    t_pos = p_pos + quat_apply_single(p_quat, offset)
                    draw_axes_at(t_pos, p_quat, TIP_AXIS_LEN, 3.0, dim=False)

            # Pad-normal arrows on link_3 and link_2 bodies
            # From URDF analysis: pad faces local -X for all links,
            # EXCEPT thumb_link_2 where 180° yaw in thumb_joint_3 flips it to +X.
            PAD_NORMAL_LINKS = {
                "index_link_3":  torch.tensor([-1.0, 0.0, 0.0]),
                "index_link_2":  torch.tensor([-1.0, 0.0, 0.0]),
                "middle_link_3": torch.tensor([-1.0, 0.0, 0.0]),
                "middle_link_2": torch.tensor([-1.0, 0.0, 0.0]),
                "ring_link_3":   torch.tensor([-1.0, 0.0, 0.0]),
                "ring_link_2":   torch.tensor([-1.0, 0.0, 0.0]),
                "thumb_link_3":  torch.tensor([-1.0, 0.0, 0.0]),
                "thumb_link_2":  torch.tensor([1.0, 0.0, 0.0]),
            }
            PAD_ARROW_LEN = 0.04  # 4cm
            for link_name, pad_dir_local in PAD_NORMAL_LINKS.items():
                link_idx = find_body(link_name)
                if link_idx is not None:
                    l_pos = robot.data.body_pos_w[0, link_idx]
                    l_quat = robot.data.body_quat_w[0, link_idx]
                    pad_dir_w = quat_apply_single(l_quat, pad_dir_local)
                    end = l_pos + pad_dir_w * PAD_ARROW_LEN
                    starts.append(l_pos.tolist())
                    ends.append(end.tolist())
                    colors.append((1.0, 1.0, 0.0, 1.0))  # Yellow arrow = pad normal
                    sizes.append(5.0)

            debug_draw.draw_lines(starts, ends, colors, sizes)
else:
    simulation_app.close()
