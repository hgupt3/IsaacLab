"""Interactive web viewer for articulated objects with hand pose editor.

Run:
    cd articulated_objects && conda run -n y2r python viewer.py

Then open http://localhost:8080 in your browser.
"""

import json
import os
import time

import numpy as np
import trimesh
import viser
from scipy.spatial.transform import Rotation, Slerp

from generator import TEMPLATES, generate_random
from hand_model import HandModel, FINGER_PRESETS

BASE_COLOR = (0.55, 0.58, 0.65)
CHILD_COLOR = (0.85, 0.55, 0.25)
HAND_COLOR = (0.25, 0.65, 0.85)
ANCHOR_COLOR = (1.0, 0.2, 0.2)

COUPLING_MODES = ["full", "position_only", "none"]
PRESET_NAMES = list(FINGER_PRESETS.keys())

DEFAULTS_PATH = os.path.join(os.path.dirname(__file__), "hand_defaults.json")

# Templates that get picked up, manipulated in the air, then put back
LIFTABLE = {"scissors", "tongs"}

PHASE_LIFT = {
    "approach":   (0.00, 0.15),
    "lift":       (0.15, 0.30),
    "manipulate": (0.30, 0.70),
    "lower":      (0.70, 0.85),
    "retreat":    (0.85, 1.00),
}
PHASE_TABLE = {
    "approach":   (0.00, 0.15),
    "manipulate": (0.15, 0.85),
    "retreat":    (0.85, 1.00),
}
NUM_JOINT_CYCLES = 2


def _load_defaults():
    if os.path.isfile(DEFAULTS_PATH):
        with open(DEFAULTS_PATH) as f:
            return json.load(f)
    return {}


def _save_defaults(data):
    with open(DEFAULTS_PATH, "w") as f:
        json.dump(data, f, indent=2)


def _smoothstep(t):
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def _lerp_transform(T_a, T_b, t):
    """Interpolate two 4x4 transforms (slerp rotation, lerp position)."""
    s = _smoothstep(t)
    pos = T_a[:3, 3] + (T_b[:3, 3] - T_a[:3, 3]) * s
    R_a = Rotation.from_matrix(T_a[:3, :3])
    R_b = Rotation.from_matrix(T_b[:3, :3])
    slerp = Slerp([0.0, 1.0], Rotation.concatenate([R_a, R_b]))
    T = np.eye(4)
    T[:3, :3] = slerp(s).as_matrix()
    T[:3, 3] = pos
    return T


def _oscillate(t, num_cycles=2):
    """Triangle wave: 0→1→0→1→0 over t∈[0,1]."""
    x = t * num_cycles * 2
    return 1.0 - abs((x % 2) - 1.0)


def _random_hand_transform(rng, center, spread=0.15):
    T = np.eye(4)
    offset = rng.uniform(-spread, spread, size=3)
    offset[2] = abs(offset[2]) + 0.05
    T[:3, 3] = center + offset
    T[:3, :3] = Rotation.random(random_state=rng).as_matrix()
    return T


def _random_air_transform(rng):
    T = np.eye(4)
    T[:3, 3] = [
        rng.uniform(-0.05, 0.05),
        rng.uniform(-0.05, 0.05),
        rng.uniform(0.10, 0.20),
    ]
    angle = np.radians(rng.uniform(10, 60))
    axis = rng.standard_normal(3)
    axis /= np.linalg.norm(axis) + 1e-9
    T[:3, :3] = Rotation.from_rotvec(angle * axis).as_matrix()
    return T


def _get_phase(t, bounds):
    for name, (start, end) in bounds.items():
        if t <= end or name == "retreat":
            local = (t - start) / (end - start) if end > start else 0.0
            return name, float(np.clip(local, 0.0, 1.0))
    return "retreat", 1.0


def main():
    server = viser.ViserServer(host="0.0.0.0", port=8080)
    print("Viewer running at http://localhost:8080")

    template_names = list(TEMPLATES.keys())
    hand_model = HandModel.get()
    all_defaults = _load_defaults()

    # ── GUI: Object controls ──────────────────────────────────────────────
    template_dropdown = server.gui.add_dropdown(
        "Template", options=template_names, initial_value=template_names[0],
    )
    joint_slider = server.gui.add_slider(
        "Joint Value", min=0.0, max=1.0, step=0.001, initial_value=0.0,
    )
    seed_slider = server.gui.add_slider(
        "Seed", min=0, max=999, step=1, initial_value=42,
    )
    randomize_button = server.gui.add_button("Randomize (new seed)")
    info_text = server.gui.add_markdown("")

    # ── GUI: Anchor + Hand pose ───────────────────────────────────────────
    server.gui.add_markdown("---\n**Anchor** *(normalized to child bbox)*")
    anchor_x = server.gui.add_slider("Anchor X", min=-1.5, max=1.5, step=0.01, initial_value=0.0)
    anchor_y = server.gui.add_slider("Anchor Y", min=-1.5, max=1.5, step=0.01, initial_value=0.0)
    anchor_z = server.gui.add_slider("Anchor Z", min=-1.5, max=1.5, step=0.01, initial_value=0.0)

    server.gui.add_markdown("---\n**Hand Pose** *(offset from anchor)*")
    finger_preset = server.gui.add_dropdown(
        "Finger Preset", options=PRESET_NAMES, initial_value="open",
    )
    offset_x = server.gui.add_slider("Offset X", min=-0.15, max=0.15, step=0.001, initial_value=0.0)
    offset_y = server.gui.add_slider("Offset Y", min=-0.15, max=0.15, step=0.001, initial_value=0.0)
    offset_z = server.gui.add_slider("Offset Z", min=-0.15, max=0.15, step=0.001, initial_value=0.0)
    palm_roll = server.gui.add_slider("Palm Roll", min=-180.0, max=180.0, step=1.0, initial_value=0.0)
    palm_pitch = server.gui.add_slider("Palm Pitch", min=-180.0, max=180.0, step=1.0, initial_value=0.0)
    palm_yaw = server.gui.add_slider("Palm Yaw", min=-180.0, max=180.0, step=1.0, initial_value=0.0)

    # ── GUI: Animation controls ───────────────────────────────────────────
    server.gui.add_markdown("---\n**Animation**")
    coupling_dropdown = server.gui.add_dropdown(
        "Coupling Mode", options=COUPLING_MODES, initial_value="full",
    )
    play_button = server.gui.add_button("Play / Pause")
    reset_button = server.gui.add_button("Reset")
    save_button = server.gui.add_button("Save")
    speed_slider = server.gui.add_slider(
        "Speed", min=0.25, max=4.0, step=0.25, initial_value=1.0,
    )

    state = {
        "obj": None,
        "suppress": False,
        "playing": False,
        "anim_time": 0.0,
        "last_tick": None,
        "save_timer": None,
        "T_approach_start": None,
        "T_retreat_end": None,
        "T_air": None,
    }

    # ── Helpers ───────────────────────────────────────────────────────────

    def _anchor_at_default(obj):
        """Anchor position on the child mesh at joint_default (world coords)."""
        _, child = obj.get_meshes_at(obj.joint_default)
        center = (child.bounds[0] + child.bounds[1]) / 2.0
        half_ext = (child.bounds[1] - child.bounds[0]) / 2.0
        norm = np.array([anchor_x.value, anchor_y.value, anchor_z.value])
        return center + norm * half_ext

    def _anchor_at_joint(obj, joint_value):
        """Anchor position on the child mesh at any joint value."""
        anchor_def = _anchor_at_default(obj)
        T_ref = obj.get_child_transform(obj.joint_default)
        T_cur = obj.get_child_transform(joint_value)
        T_delta = T_cur @ np.linalg.inv(T_ref)
        return (T_delta @ np.array([*anchor_def, 1.0]))[:3]

    def _child_delta(obj, joint_value):
        """4x4 delta transform of child from joint_default to joint_value."""
        T_ref = obj.get_child_transform(obj.joint_default)
        T_cur = obj.get_child_transform(joint_value)
        return T_cur @ np.linalg.inv(T_ref)

    def _get_hand_transform(obj, joint_value=None):
        """Hand 4x4 at a given joint value, using anchor-based coupling.

        The anchor is a point on the child mesh. As the child articulates,
        the anchor moves with it. Coupling mode controls the hand:
          full:          offset + orientation rotate with the child (rigid)
          position_only: position follows anchor, offset/orient stay in world frame
          none:          everything frozen at the default-joint position
        """
        if joint_value is None:
            joint_value = obj.joint_default

        anchor_def = _anchor_at_default(obj)
        offset = np.array([offset_x.value, offset_y.value, offset_z.value])
        R_hand = Rotation.from_euler(
            "xyz", [palm_roll.value, palm_pitch.value, palm_yaw.value], degrees=True,
        ).as_matrix()

        mode = coupling_dropdown.value
        T = np.eye(4)

        if mode == "full":
            # Everything transforms rigidly with the child
            Td = _child_delta(obj, joint_value)
            anchor_cur = (Td @ np.array([*anchor_def, 1.0]))[:3]
            T[:3, :3] = Td[:3, :3] @ R_hand
            T[:3, 3] = anchor_cur + Td[:3, :3] @ offset

        elif mode == "position_only":
            # Anchor tracks child, offset + orientation stay in world frame
            anchor_cur = _anchor_at_joint(obj, joint_value)
            T[:3, :3] = R_hand
            T[:3, 3] = anchor_cur + offset

        else:  # none
            T[:3, :3] = R_hand
            T[:3, 3] = anchor_def + offset

        return T

    # ── Rendering ─────────────────────────────────────────────────────────

    def _render_meshes(obj, joint_value, T_world=None):
        base_mesh, child_mesh = obj.get_meshes_at(joint_value)
        if T_world is not None:
            base_mesh.apply_transform(T_world)
            child_mesh.apply_transform(T_world)
        server.scene.add_mesh_simple(
            "/base", vertices=base_mesh.vertices.astype(np.float32),
            faces=base_mesh.faces.astype(np.uint32), color=BASE_COLOR, flat_shading=True,
        )
        server.scene.add_mesh_simple(
            "/child", vertices=child_mesh.vertices.astype(np.float32),
            faces=child_mesh.faces.astype(np.uint32), color=CHILD_COLOR, flat_shading=True,
        )

    def _render_grid():
        size = 0.3
        segs = []
        n = 20
        for i in range(n + 1):
            t = -size / 2 + i * size / n
            segs.append([[-size / 2, t, 0], [size / 2, t, 0]])
            segs.append([[t, -size / 2, 0], [t, size / 2, 0]])
        pts = np.array(segs, dtype=np.float32)
        cols = np.full_like(pts, 180, dtype=np.uint8)
        server.scene.add_line_segments("/grid", points=pts, colors=cols)

    def _render_axis(obj, T_world=None):
        origin = obj.joint_origin.copy()
        axis = obj.joint_axis.copy()
        half = 0.025
        p0 = origin - half * axis
        p1 = origin + half * axis
        if T_world is not None:
            p0 = T_world[:3, :3] @ p0 + T_world[:3, 3]
            p1 = T_world[:3, :3] @ p1 + T_world[:3, 3]
        server.scene.add_line_segments(
            "/joint_axis",
            points=np.array([[p0.astype(np.float32), p1.astype(np.float32)]]),
            colors=np.array([[[255, 50, 50], [255, 50, 50]]], dtype=np.uint8),
        )

    def _render_anchor(obj, joint_value=None, T_world=None):
        """Show a small red sphere at the anchor point on the child."""
        if joint_value is None:
            joint_value = obj.joint_default
        pos = _anchor_at_joint(obj, joint_value)
        if T_world is not None:
            pos = T_world[:3, :3] @ pos + T_world[:3, 3]
        marker = trimesh.creation.icosphere(radius=0.004)
        marker.apply_translation(pos)
        server.scene.add_mesh_simple(
            "/anchor", vertices=marker.vertices.astype(np.float32),
            faces=marker.faces.astype(np.uint32),
            color=ANCHOR_COLOR, flat_shading=True,
        )

    def _render_scene(obj, joint_value, T_world=None):
        _render_meshes(obj, joint_value, T_world)
        _render_grid()
        _render_axis(obj, T_world)
        _render_anchor(obj, joint_value, T_world)

    def _render_hand(transform=None):
        joints = FINGER_PRESETS[finger_preset.value]
        h_mesh = hand_model.get_mesh(joints)
        T = transform if transform is not None else _get_hand_transform(state["obj"])
        h_mesh.apply_transform(T)
        server.scene.add_mesh_simple(
            "/hand", vertices=h_mesh.vertices.astype(np.float32),
            faces=h_mesh.faces.astype(np.uint32),
            color=HAND_COLOR, flat_shading=True, opacity=0.85,
        )

    # ── Info / Save ───────────────────────────────────────────────────────

    def _update_info(obj, jv, phase=None):
        jtype = obj.joint_type
        if jtype == "revolute":
            val_str = f"{np.degrees(jv):.1f} deg"
            lim_str = (f"[{np.degrees(obj.joint_limits[0]):.1f}, "
                       f"{np.degrees(obj.joint_limits[1]):.1f}] deg")
            def_str = f"{np.degrees(obj.joint_default):.1f} deg"
            goal_str = (f"{np.degrees(obj.goal_joint_value):.1f} deg"
                        if obj.goal_joint_value is not None else "n/a")
        else:
            val_str = f"{jv * 1000:.1f} mm"
            lim_str = (f"[{obj.joint_limits[0] * 1000:.1f}, "
                       f"{obj.joint_limits[1] * 1000:.1f}] mm")
            def_str = f"{obj.joint_default * 1000:.1f} mm"
            goal_str = (f"{obj.goal_joint_value * 1000:.1f} mm"
                        if obj.goal_joint_value is not None else "n/a")

        combined = trimesh.util.concatenate([obj.base_mesh, obj.child_mesh])
        ext = (combined.bounds[1] - combined.bounds[0]) * 1000
        bbox_str = f"{ext[0]:.0f} x {ext[1]:.0f} x {ext[2]:.0f} mm"

        phase_str = f" | **{phase}**" if phase else ""
        anim_str = (f" | t={state['anim_time']:.2f}"
                    if state["playing"] or state["anim_time"] > 0 else "")

        info_text.content = (
            f"**{obj.name}** | {jtype} | seed {int(seed_slider.value)}\n\n"
            f"Size: {bbox_str} | Value: **{val_str}** | Limits: {lim_str} | "
            f"Default: {def_str} | Goal: {goal_str} | "
            f"Coupling: {coupling_dropdown.value}{phase_str}{anim_str}"
        )

    def _save_current():
        name = template_dropdown.value
        all_defaults[name] = {
            "anchor_xyz": [anchor_x.value, anchor_y.value, anchor_z.value],
            "offset_xyz": [offset_x.value, offset_y.value, offset_z.value],
            "palm_rpy": [palm_roll.value, palm_pitch.value, palm_yaw.value],
            "finger_preset": finger_preset.value,
            "coupling_mode": coupling_dropdown.value,
        }
        _save_defaults(all_defaults)

    def _schedule_autosave():
        state["save_timer"] = time.time() + 0.5

    # ── Load / reset ──────────────────────────────────────────────────────

    def _render_full(obj, joint_value):
        _render_scene(obj, joint_value)
        _render_hand()
        _update_info(obj, joint_value)

    def load_object():
        state["suppress"] = True
        state["playing"] = False
        state["anim_time"] = 0.0
        for name in ("/base", "/child", "/grid", "/joint_axis", "/hand", "/anchor"):
            try:
                server.scene.remove_by_name(name)
            except Exception:
                pass

        tpl_name = template_dropdown.value
        seed = int(seed_slider.value)
        obj = generate_random(tpl_name, seed=seed)
        state["obj"] = obj

        joint_slider.min = obj.joint_limits[0]
        joint_slider.max = obj.joint_limits[1]
        joint_slider.value = obj.joint_default

        saved = all_defaults.get(tpl_name)
        if saved:
            anchor_x.value = saved["anchor_xyz"][0]
            anchor_y.value = saved["anchor_xyz"][1]
            anchor_z.value = saved["anchor_xyz"][2]
            offset_x.value = saved["offset_xyz"][0]
            offset_y.value = saved["offset_xyz"][1]
            offset_z.value = saved["offset_xyz"][2]
            palm_roll.value = saved["palm_rpy"][0]
            palm_pitch.value = saved["palm_rpy"][1]
            palm_yaw.value = saved["palm_rpy"][2]
            finger_preset.value = saved["finger_preset"]
            coupling_dropdown.value = saved["coupling_mode"]
        else:
            # Defaults: anchor at child centroid, small offset
            anchor_x.value = 0.0
            anchor_y.value = 0.0
            anchor_z.value = 0.0
            offset_x.value = 0.06
            offset_y.value = 0.0
            offset_z.value = 0.02
            palm_roll.value = 0.0
            palm_pitch.value = 0.0
            palm_yaw.value = 0.0
            finger_preset.value = "open"
            coupling_dropdown.value = "full"

        state["suppress"] = False
        _render_full(obj, obj.joint_default)

    # ── Event handlers ────────────────────────────────────────────────────

    @template_dropdown.on_update
    def _on_template(event):
        if not state["suppress"]:
            load_object()

    @joint_slider.on_update
    def _on_joint(event):
        if not state["suppress"] and state["obj"] is not None:
            if state["playing"]:
                state["playing"] = False
            obj = state["obj"]
            jv = joint_slider.value
            _render_meshes(obj, jv)
            _render_anchor(obj, jv)
            _render_hand(_get_hand_transform(obj, jv))
            _update_info(obj, jv)

    @seed_slider.on_update
    def _on_seed(event):
        if not state["suppress"]:
            load_object()

    @randomize_button.on_click
    def _on_randomize(event):
        seed_slider.value = int(np.random.randint(0, 1000))

    # Anchor + hand pose sliders — update hand + anchor marker live
    for slider in [anchor_x, anchor_y, anchor_z,
                   offset_x, offset_y, offset_z,
                   palm_roll, palm_pitch, palm_yaw]:
        @slider.on_update
        def _on_pose_change(event):
            if not state["suppress"] and state["obj"] is not None:
                obj = state["obj"]
                jv = joint_slider.value
                _render_anchor(obj, jv)
                _render_hand(_get_hand_transform(obj, jv))
                _schedule_autosave()

    @finger_preset.on_update
    def _on_preset(event):
        if not state["suppress"] and state["obj"] is not None:
            _render_hand(_get_hand_transform(state["obj"], joint_slider.value))
            _schedule_autosave()

    @coupling_dropdown.on_update
    def _on_coupling(event):
        if not state["suppress"]:
            _schedule_autosave()

    def _generate_random_poses():
        obj = state["obj"]
        rng = np.random.default_rng(int(time.time() * 1000) % (2**31))
        _, child = obj.get_meshes_at(obj.joint_default)
        center = child.centroid
        state["T_approach_start"] = _random_hand_transform(rng, center)
        state["T_retreat_end"] = _random_hand_transform(rng, center)
        state["T_air"] = _random_air_transform(rng)

    @play_button.on_click
    def _on_play_pause(event):
        if state["obj"] is None:
            return
        state["playing"] = not state["playing"]
        if state["playing"]:
            state["last_tick"] = time.time()
            if state["anim_time"] >= 1.0 or state["anim_time"] == 0.0:
                state["anim_time"] = 0.0
                _generate_random_poses()
            elif state["T_approach_start"] is None:
                _generate_random_poses()

    @reset_button.on_click
    def _on_reset(event):
        state["playing"] = False
        state["anim_time"] = 0.0
        if state["obj"] is not None:
            state["suppress"] = True
            joint_slider.value = state["obj"].joint_default
            state["suppress"] = False
            _render_full(state["obj"], state["obj"].joint_default)

    @save_button.on_click
    def _on_save(event):
        _save_current()

    load_object()

    # ── Main loop: ~30 FPS ────────────────────────────────────────────────
    try:
        while True:
            if state["save_timer"] is not None and time.time() >= state["save_timer"]:
                state["save_timer"] = None
                _save_current()

            if (state["playing"] and state["obj"] is not None
                    and state["T_approach_start"] is not None):
                obj = state["obj"]
                now = time.time()
                if state["last_tick"] is not None:
                    dt = now - state["last_tick"]
                    state["anim_time"] += dt * speed_slider.value / 3.0
                    if state["anim_time"] >= 1.0:
                        state["anim_time"] = 1.0
                        state["playing"] = False
                state["last_tick"] = now

                t = state["anim_time"]
                liftable = obj.name in LIFTABLE
                bounds = PHASE_LIFT if liftable else PHASE_TABLE
                phase, pt = _get_phase(t, bounds)

                start_jv = obj.joint_default
                if obj.goal_joint_value is not None:
                    end_jv = obj.goal_joint_value
                else:
                    d_lo = abs(obj.joint_default - obj.joint_limits[0])
                    d_hi = abs(obj.joint_default - obj.joint_limits[1])
                    end_jv = obj.joint_limits[0] if d_lo >= d_hi else obj.joint_limits[1]
                # Hand at default joint (contact pose)
                T_contact = _get_hand_transform(obj, start_jv)
                T_air = state["T_air"]
                I4 = np.eye(4)

                if phase == "approach":
                    T_world = I4
                    current_jv = start_jv
                    T_hand = _lerp_transform(state["T_approach_start"], T_contact, pt)

                elif phase == "lift":
                    T_world = _lerp_transform(I4, T_air, pt)
                    current_jv = start_jv
                    T_hand = T_world @ T_contact

                elif phase == "manipulate":
                    T_world = T_air if liftable else I4
                    osc = _oscillate(pt, NUM_JOINT_CYCLES)
                    current_jv = start_jv + (end_jv - start_jv) * osc
                    T_hand = T_world @ _get_hand_transform(obj, current_jv)

                elif phase == "lower":
                    T_world = _lerp_transform(T_air, I4, pt)
                    current_jv = start_jv
                    T_hand = T_world @ T_contact

                else:  # retreat
                    T_world = I4
                    current_jv = start_jv
                    T_hand = _lerp_transform(T_contact, state["T_retreat_end"], pt)

                _render_meshes(obj, current_jv, T_world)
                _render_axis(obj, T_world)
                _render_anchor(obj, current_jv, T_world)
                _render_hand(T_hand)

                state["suppress"] = True
                joint_slider.value = np.clip(
                    current_jv, obj.joint_limits[0], obj.joint_limits[1],
                )
                state["suppress"] = False
                _update_info(obj, current_jv, phase=phase)

            time.sleep(1 / 30)
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()
