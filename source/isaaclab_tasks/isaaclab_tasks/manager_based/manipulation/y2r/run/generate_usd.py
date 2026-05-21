"""Generate/inspect a Y2R robot USD conversion with computed runtime frames."""

from __future__ import annotations

import argparse
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Inspect Y2R robot URDF/USD conversion.")
parser.add_argument("--robot", default=os.environ.get("Y2R_ROBOT", "ur5e_leap"), help="Robot config name.")
parser.add_argument("--view", action="store_true", help="Open livestream viewer with debug axes.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.view:
    args.livestream = 2

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.utils.math import combine_frame_transforms, quat_apply, quat_from_euler_xyz, quat_mul
from isaaclab_assets.robots import KUKA_ALLEGRO_CFG, UR5E_GEMINI_WSG50_CFG, UR5E_LEAP_CFG

from isaaclab_tasks.manager_based.manipulation.y2r.config_loader import get_config


ROBOT_ASSETS = {
    "ur5e_leap": UR5E_LEAP_CFG,
    "kuka_allegro": KUKA_ALLEGRO_CFG,
    "ur5e_gemini_wsg50": UR5E_GEMINI_WSG50_CFG,
}


def _quat_from_rpy_deg(rpy_deg) -> torch.Tensor:
    roll, pitch, yaw = [math.radians(float(value)) for value in rpy_deg]
    return quat_from_euler_xyz(torch.tensor([roll]), torch.tensor([pitch]), torch.tensor([yaw])).squeeze(0)


def _camera_quat_from_config_rpy(rpy_deg) -> torch.Tensor:
    swapped = (float(rpy_deg[0]), float(rpy_deg[2]), -float(rpy_deg[1]))
    return _quat_from_rpy_deg(swapped)


def _quat_apply_single(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    return quat_apply(quat.unsqueeze(0), vec.unsqueeze(0)).squeeze(0)


def _find_body(robot: Articulation, name: str):
    ids = robot.find_bodies(name)[0]
    return ids[0] if len(ids) > 0 else None


def _draw_axes(draw, starts, ends, colors, sizes, pos, quat, axis_len, line_width, dim=False):
    brightness = 0.45 if dim else 1.0
    alpha = 0.65 if dim else 1.0
    axes = [
        (torch.tensor([1.0, 0.0, 0.0]), (brightness, 0.0, 0.0, alpha)),
        (torch.tensor([0.0, 1.0, 0.0]), (0.0, brightness, 0.0, alpha)),
        (torch.tensor([0.0, 0.0, 1.0]), (0.0, 0.0, brightness, alpha)),
    ]
    for axis, color in axes:
        end = pos + _quat_apply_single(quat, axis.to(pos.device)) * axis_len
        starts.append(pos.tolist())
        ends.append(end.tolist())
        colors.append(color)
        sizes.append(line_width)


if args.robot not in ROBOT_ASSETS:
    raise ValueError(f"Unknown robot '{args.robot}'. Available: {sorted(ROBOT_ASSETS)}")

y2r_cfg = get_config(mode="play", task="base", robot=args.robot)

sim_cfg = sim_utils.SimulationCfg(dt=1 / 120.0, device="cpu")
sim = sim_utils.SimulationContext(sim_cfg)
sim.set_camera_view(eye=(1.5, 1.5, 1.0), target=(0.0, 0.0, 0.3))

robot_cfg = ROBOT_ASSETS[args.robot].replace(prim_path="/World/Robot")
robot = Articulation(robot_cfg)

sim.reset()
robot.reset()
robot.update(sim.cfg.dt)

print("\n" + "=" * 72)
print(f"{args.robot} USD Conversion Report")
print("=" * 72)
print(f"\nNumber of joints:  {robot.num_joints}")
print(f"Number of bodies:  {robot.num_bodies}")

print("\n--- Joint names ---")
for i, name in enumerate(robot.joint_names):
    print(f"  [{i:2d}] {name}")

print("\n--- Body names ---")
for i, name in enumerate(robot.body_names):
    print(f"  [{i:2d}] {name}")

print("\n--- Joint limits ---")
limits = robot.root_physx_view.get_dof_limits().squeeze(0)
for i, name in enumerate(robot.joint_names):
    lo = limits[i, 0].item()
    hi = limits[i, 1].item()
    print(f"  {name:28s} [{lo:+.5f}, {hi:+.5f}]")

palm_body_idx = _find_body(robot, y2r_cfg.robot.palm_body_name)
if palm_body_idx is None:
    raise RuntimeError(f"Palm body not found: {y2r_cfg.robot.palm_body_name}")

palm_body_pos = robot.data.body_pos_w[0, palm_body_idx]
palm_body_quat = robot.data.body_quat_w[0, palm_body_idx]
palm_offset = y2r_cfg.robot.palm_frame_offset
if palm_offset is None:
    palm_frame_pos = palm_body_pos
    palm_frame_quat = palm_body_quat
else:
    palm_frame_pos, palm_frame_quat = combine_frame_transforms(
        palm_body_pos.unsqueeze(0),
        palm_body_quat.unsqueeze(0),
        torch.tensor(palm_offset.pos, dtype=torch.float32).unsqueeze(0),
        _quat_from_rpy_deg(palm_offset.rot_euler).unsqueeze(0),
    )
    palm_frame_pos = palm_frame_pos.squeeze(0)
    palm_frame_quat = palm_frame_quat.squeeze(0)

camera_offset = y2r_cfg.wrist_camera.offset
camera_pos, camera_quat = combine_frame_transforms(
    palm_body_pos.unsqueeze(0),
    palm_body_quat.unsqueeze(0),
    torch.tensor(camera_offset.pos, dtype=torch.float32).unsqueeze(0),
    _camera_quat_from_config_rpy(camera_offset.rot).unsqueeze(0),
)
camera_pos = camera_pos.squeeze(0)
camera_quat = camera_quat.squeeze(0)

print("\n--- Computed runtime frames ---")
print(f"  palm parent body: {y2r_cfg.robot.palm_body_name}")
print(f"  palm_frame pos:   ({palm_frame_pos[0]:+.5f}, {palm_frame_pos[1]:+.5f}, {palm_frame_pos[2]:+.5f})")
print(f"  palm_frame quat:  ({palm_frame_quat[0]:+.5f}, {palm_frame_quat[1]:+.5f}, {palm_frame_quat[2]:+.5f}, {palm_frame_quat[3]:+.5f})")
print(f"  camera pos:       ({camera_pos[0]:+.5f}, {camera_pos[1]:+.5f}, {camera_pos[2]:+.5f})")
print(f"  camera quat:      ({camera_quat[0]:+.5f}, {camera_quat[1]:+.5f}, {camera_quat[2]:+.5f}, {camera_quat[3]:+.5f})")

tip_poses = []
print("\n--- Tip/contact frames ---")
for body_name in y2r_cfg.robot.tip_state_body_names:
    body_idx = _find_body(robot, body_name)
    if body_idx is None:
        print(f"  {body_name}: NOT FOUND")
        continue
    body_pos = robot.data.body_pos_w[0, body_idx]
    body_quat = robot.data.body_quat_w[0, body_idx]
    offset = torch.tensor(y2r_cfg.robot.tip_state_offsets[body_name], dtype=torch.float32)
    rot_offset = _quat_from_rpy_deg(y2r_cfg.robot.tip_state_rot_euler[body_name])
    tip_pos = body_pos + _quat_apply_single(body_quat, offset)
    tip_quat = quat_mul(body_quat.unsqueeze(0), rot_offset.unsqueeze(0)).squeeze(0)
    tip_poses.append((body_name, body_idx, tip_pos, tip_quat))
    print(f"  {body_name:26s} offset=({offset[0]:+.5f}, {offset[1]:+.5f}, {offset[2]:+.5f}) "
          f"rot=({y2r_cfg.robot.tip_state_rot_euler[body_name][0]:+.2f}, "
          f"{y2r_cfg.robot.tip_state_rot_euler[body_name][1]:+.2f}, "
          f"{y2r_cfg.robot.tip_state_rot_euler[body_name][2]:+.2f}) "
          f"tip=({tip_pos[0]:+.5f}, {tip_pos[1]:+.5f}, {tip_pos[2]:+.5f})")

print("\n--- Contact pad normals ---")
for body_name, normal in y2r_cfg.robot.contact_layout.pad_normals.items():
    print(f"  {body_name:26s} normal=({normal[0]:+.4f}, {normal[1]:+.4f}, {normal[2]:+.4f})")

print("\n" + "=" * 72)
print("Conversion report complete.")
print("=" * 72)

if args.view:
    debug_draw = None
    try:
        import isaacsim.util.debug_draw._debug_draw as omni_debug_draw
        debug_draw = omni_debug_draw.acquire_debug_draw_interface()
    except ModuleNotFoundError:
        try:
            from omni.isaac.debug_draw import _debug_draw as omni_debug_draw
            debug_draw = omni_debug_draw.acquire_debug_draw_interface()
        except ModuleNotFoundError:
            from omni.debugdraw import get_debug_draw_interface
            debug_draw = get_debug_draw_interface()

    print("\nViewer running at http://localhost:8211")
    print("RGB axes: dim parent body, bright palm/camera/tips. Yellow lines are pad normals.")
    print("Press Ctrl+C to exit.\n")

    while simulation_app.is_running():
        sim.step()
        robot.update(sim.cfg.dt)

        if debug_draw is not None:
            debug_draw.clear_lines()
            starts, ends, colors, sizes = [], [], [], []

            palm_body_pos = robot.data.body_pos_w[0, palm_body_idx]
            palm_body_quat = robot.data.body_quat_w[0, palm_body_idx]
            _draw_axes(debug_draw, starts, ends, colors, sizes, palm_body_pos, palm_body_quat, 0.12, 3.0, dim=True)

            if palm_offset is None:
                pf_pos, pf_quat = palm_body_pos, palm_body_quat
            else:
                pf_pos, pf_quat = combine_frame_transforms(
                    palm_body_pos.unsqueeze(0),
                    palm_body_quat.unsqueeze(0),
                    torch.tensor(palm_offset.pos, dtype=torch.float32).unsqueeze(0),
                    _quat_from_rpy_deg(palm_offset.rot_euler).unsqueeze(0),
                )
                pf_pos, pf_quat = pf_pos.squeeze(0), pf_quat.squeeze(0)
            _draw_axes(debug_draw, starts, ends, colors, sizes, pf_pos, pf_quat, 0.12, 5.0)

            cf_pos, cf_quat = combine_frame_transforms(
                palm_body_pos.unsqueeze(0),
                palm_body_quat.unsqueeze(0),
                torch.tensor(camera_offset.pos, dtype=torch.float32).unsqueeze(0),
                _camera_quat_from_config_rpy(camera_offset.rot).unsqueeze(0),
            )
            _draw_axes(debug_draw, starts, ends, colors, sizes, cf_pos.squeeze(0), cf_quat.squeeze(0), 0.08, 4.0)

            for body_name in y2r_cfg.robot.tip_state_body_names:
                body_idx = _find_body(robot, body_name)
                if body_idx is None:
                    continue
                body_pos = robot.data.body_pos_w[0, body_idx]
                body_quat = robot.data.body_quat_w[0, body_idx]
                offset = torch.tensor(y2r_cfg.robot.tip_state_offsets[body_name], dtype=torch.float32)
                rot_offset = _quat_from_rpy_deg(y2r_cfg.robot.tip_state_rot_euler[body_name])
                tip_pos = body_pos + _quat_apply_single(body_quat, offset)
                tip_quat = quat_mul(body_quat.unsqueeze(0), rot_offset.unsqueeze(0)).squeeze(0)
                _draw_axes(debug_draw, starts, ends, colors, sizes, tip_pos, tip_quat, 0.04, 3.0)

            for body_name, normal in y2r_cfg.robot.contact_layout.pad_normals.items():
                body_idx = _find_body(robot, body_name)
                if body_idx is None:
                    continue
                body_pos = robot.data.body_pos_w[0, body_idx]
                body_quat = robot.data.body_quat_w[0, body_idx]
                if body_name in y2r_cfg.robot.tip_state_offsets:
                    offset = torch.tensor(y2r_cfg.robot.tip_state_offsets[body_name], dtype=torch.float32)
                    origin = body_pos + _quat_apply_single(body_quat, offset)
                else:
                    origin = body_pos
                normal_w = _quat_apply_single(body_quat, torch.tensor(normal, dtype=torch.float32))
                end = origin + normal_w * 0.05
                starts.append(origin.tolist())
                ends.append(end.tolist())
                colors.append((1.0, 1.0, 0.0, 1.0))
                sizes.append(5.0)

            debug_draw.draw_lines(starts, ends, colors, sizes)
else:
    simulation_app.close()
