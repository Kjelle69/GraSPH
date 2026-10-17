"""GraSPH scene-authoring and simulation controls for Blender.

Install the complete ZIP as a legacy add-on, or run this file in the Text Editor.
The numerical solver is launched as a separate process; Blender remains the
scene editor and result viewer.
"""

ADDON_VERSION = (0, 15, 0)


bl_info = {
    "name": "GraSPH Controls",
    "author": "GraSPH",
    "version": ADDON_VERSION,
    "blender": (4, 5, 0),
    "location": "3D View > Sidebar > GraSPH",
    "description": "Tag, validate, export and simulate GraSPH USD scenes",
    "category": "Physics",
}

import os
from pathlib import Path
import importlib.util
import math
import re
import subprocess
import sys
import time

import bpy
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import Operator, Panel, PropertyGroup, UIList


ROLE_ITEMS = (
    ("NONE", "Unassigned", "No explicit GraSPH role"),
    ("FLUID", "Fluid Volume", "Initial SPH fluid volume"),
    ("GRANULAR_SOURCE", "Granular Source", "Initial volume of rigid spherical grains"),
    ("SINK", "Sink / Drain", "Remove particles entering this volume"),
    ("STATIC", "Static Collider", "Fixed tank or obstacle"),
    ("KINEMATIC", "Kinematic Collider", "Animated collider that drives the fluid"),
    ("DYNAMIC_HINGE", "Dynamic Hinge", "Collider driven by fluid torque around one axle"),
    ("DYNAMIC_BODY", "Dynamic Body", "Freely translating and rotating rigid collider"),
    ("JOINT", "Joint", "Separate constraint between two bodies or a body and the world"),
    ("IGNORE", "Visual Only", "Export visually but exclude from simulation"),
)

ROLE_TO_USD = {
    "FLUID": "fluid",
    "GRANULAR_SOURCE": "granular_source",
    "SINK": "sink",
    "STATIC": "collider",
    "KINEMATIC": "collider",
    "DYNAMIC_HINGE": "collider",
    "DYNAMIC_BODY": "collider",
    "JOINT": "joint",
    "IGNORE": "ignore",
}

JOINT_TYPE_ITEMS = (
    ("REVOLUTE", "Revolute / Hinge", "One rotational degree of freedom"),
    ("PRISMATIC", "Prismatic / Linear", "One translational degree of freedom"),
    ("FREE", "Free", "No joint constraint; body remains fully dynamic"),
    ("FIXED", "Fixed", "Weld Body A and Body B; empty Body A means world"),
)

QUALITY_PRESETS = {
    "DRAFT": {"dp": 0.05, "granular_dt": 0.001, "granular_iterations": 4, "output_fps": 24.0},
    "BALANCED": {"dp": 0.025, "granular_dt": 0.0005, "granular_iterations": 6, "output_fps": 30.0},
    "QUALITY": {"dp": 0.0125, "granular_dt": 0.00025, "granular_iterations": 10, "output_fps": 30.0},
}

GRANULAR_MATERIAL_PRESETS = {
    "SAND": (1600.0, 0.55, 0.05, 0.08),
    "GRAVEL": (1800.0, 0.65, 0.10, 0.12),
    "GRAIN": (780.0, 0.40, 0.08, 0.04),
    "RUBBER": (1100.0, 0.80, 0.65, 0.08),
    "STEEL": (7850.0, 0.25, 0.30, 0.01),
}

_RUNNING = None
_RESET_REQUESTED = False
_ENV_JOB = None
_ENVIRONMENT_MODULE = None


def _physics_values(settings):
    """Resolve shared physical parameters only; never replace shader slots."""
    material = settings.physics_material
    if material is not None:
        p = material.grasph_physics
        if not p.enabled:
            raise ValueError(f'{material.name}: not a GraSPH physics material')
        values = dict(density=p.density, friction=p.friction,
                      restitution=p.restitution, rolling=p.rolling_friction)
    elif settings.role == 'GRANULAR_SOURCE':
        values = dict(density=settings.granular_density, friction=settings.granular_friction,
                      restitution=settings.granular_restitution, rolling=settings.granular_rolling_friction)
    else:
        values = dict(density=0., friction=settings.collider_friction,
                      restitution=settings.collider_restitution, rolling=settings.collider_rolling_friction)
    if not all(math.isfinite(v) for v in values.values()):
        raise ValueError('Physics material values must be finite')
    if values['friction'] < 0 or values['rolling'] < 0 or not 0 <= values['restitution'] <= 1:
        raise ValueError('Invalid physics friction/restitution')
    if settings.role == 'GRANULAR_SOURCE' and values['density'] <= 0:
        raise ValueError('Grain density must be positive')
    return values


def _new_physics_material(preset='SAND'):
    material = bpy.data.materials.new('GraSPH Physics - '+preset.title())
    material.use_fake_user = True
    p = material.grasph_physics
    p.enabled = True
    p.density, p.friction, p.restitution, p.rolling_friction = GRANULAR_MATERIAL_PRESETS[preset]
    return material


def _tail_solver_log(log_file):
    path = Path(log_file) if log_file else None
    if path is None or not path.is_file():
        return None, "Waiting for solver output…", 0
    size = path.stat().st_size
    with path.open("rb") as stream:
        stream.seek(max(0, size - 65536))
        text = stream.read().decode("utf-8", errors="replace")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    progress = None
    for line in reversed(lines):
        match = re.search(r"\|\s*([0-9]+(?:\.[0-9]+)?)%\s*\|", line)
        if match:
            progress = float(match.group(1))
            break
    message = lines[-1] if lines else "Solver process started…"
    return progress, message, size


def _tag_redraw(context):
    screen = getattr(context, "screen", None)
    if screen is not None:
        for area in screen.areas:
            area.tag_redraw()


_COLLIDER_ROLES = {"STATIC", "KINEMATIC", "DYNAMIC_HINGE", "DYNAMIC_BODY"}
_COLLIDER_OUTLINE_COLLECTION = "GraSPH Collider Outlines"
_COLLIDER_DISPLAY_ITEMS = (
    ("OFF", "Off", "Hide collider visualization"),
    ("WIRE", "Wire", "Show collider edges only"),
    ("SOLID", "Transparent Solid", "Show a pale-yellow transparent collider mesh"),
    ("BOTH", "Both", "Show transparent surfaces and wire edges"),
)
# Keep datablock names, not RNA Object wrappers: a user can delete/reload a
# preview object while a modal simulation is active, invalidating its wrapper.
_PREVIEW_CLOUDS = {"fluid": None, "granular": None}
_PREVIEW_LAST_PATHS = []


def _refresh_collider_outlines(scene, display_mode):
    if isinstance(display_mode, bool):
        display_mode = "WIRE" if display_mode else "OFF"
    display_mode = str(display_mode).upper()
    collection = bpy.data.collections.get(_COLLIDER_OUTLINE_COLLECTION)
    if collection is None and display_mode != "OFF":
        collection = bpy.data.collections.new(_COLLIDER_OUTLINE_COLLECTION)
        scene.collection.children.link(collection)
        collection.hide_render = True
    if collection is None:
        return
    for outline in list(collection.objects):
        bpy.data.objects.remove(outline, do_unlink=True)
    if display_mode == "OFF":
        bpy.data.collections.remove(collection)
        return

    show_wire = display_mode in {"WIRE", "BOTH"}
    show_solid = display_mode in {"SOLID", "BOTH"}
    colors = {
        "STATIC": (0.12, 0.75, 1.0, 1.0),
        "KINEMATIC": (1.0, 0.65, 0.10, 1.0),
        "DYNAMIC_HINGE": (0.95, 0.30, 0.90, 1.0),
        "DYNAMIC_BODY": (0.25, 1.0, 0.40, 1.0),
    }
    for source in scene.objects:
        role = getattr(getattr(source, "grasph", None), "role", "NONE")
        if role not in _COLLIDER_ROLES or source.type != "MESH":
            continue
        variants = []
        if show_solid:
            variants.append(("Solid", "SOLID"))
        if show_wire:
            variants.append(("Wire", "WIRE"))
        for suffix, display_type in variants:
            overlay = bpy.data.objects.new(
                f"{source.name}_Collider{suffix}", source.data
            )
            collection.objects.link(overlay)
            overlay.matrix_world = source.matrix_world.copy()
            overlay.display_type = display_type
            overlay.show_in_front = display_type == "WIRE"
            overlay.hide_render = True
            overlay.color = (1.0, 0.88, 0.24, 0.24) if display_type == "SOLID" else colors[role]
            overlay.show_transparent = display_type == "SOLID"
            overlay["grasph:collider_outline"] = True
            overlay["grasph:collider_visual_mode"] = display_type
            overlay["grasph:outline_source"] = source.name
            if display_type == "SOLID":
                material = bpy.data.materials.get("GraSPH Collider Preview")
                if material is None:
                    material = bpy.data.materials.new("GraSPH Collider Preview")
                material.diffuse_color = (1.0, 0.88, 0.24, 0.24)
                if hasattr(material, "surface_render_method"):
                    material.surface_render_method = "DITHERED"
                material.use_nodes = False
                overlay.data = source.data.copy()
                overlay.data.materials.clear()
                overlay.data.materials.append(material)
            if hasattr(overlay, "grasph"):
                overlay.grasph.role = "NONE"


def _set_collider_outlines(self, context):
    _refresh_collider_outlines(context.scene, self.collider_display)


def _update_point_cloud(cloud, positions, velocity=None):
    """Resize/update mesh vertices used as an instanced live particle cloud."""
    positions = positions.reshape((-1, 3))
    if len(cloud.vertices) != len(positions):
        old_mesh = cloud
        cloud = bpy.data.meshes.new(old_mesh.name + "_Live")
        cloud.from_pydata(positions.tolist(), [], [])
        if old_mesh.users == 0:
            bpy.data.meshes.remove(old_mesh)
    if len(positions):
        cloud.vertices.foreach_set("co", positions.astype("f", copy=False).ravel())
    cloud.update()
    return cloud


def _add_live_particle_display(obj, kind, radius):
    group = bpy.data.node_groups.new(f"GraSPH Live {kind.title()} Display", "GeometryNodeTree")
    group.is_modifier = True
    interface = group.interface
    interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    radius_socket = interface.new_socket(
        name="Radius", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    radius_socket.default_value = float(radius)
    radius_socket.min_value = 1.0e-6
    nodes, links = group.nodes, group.links
    entry = nodes.new("NodeGroupInput")
    sphere = nodes.new("GeometryNodeMeshIcoSphere")
    sphere.inputs["Subdivisions"].default_value = 1
    instances = nodes.new("GeometryNodeInstanceOnPoints")
    output = nodes.new("NodeGroupOutput")
    links.new(entry.outputs["Geometry"], instances.inputs["Points"])
    if kind == "granular":
        radii = nodes.new("GeometryNodeInputNamedAttribute")
        radii.data_type = "FLOAT"
        radii.inputs["Name"].default_value = "grasph_radius"
        sphere.inputs["Radius"].default_value = 1.0
        links.new(radii.outputs["Attribute"], instances.inputs["Scale"])
    else:
        links.new(entry.outputs["Radius"], sphere.inputs["Radius"])
    links.new(sphere.outputs["Mesh"], instances.inputs["Instance"])
    links.new(instances.outputs["Instances"], output.inputs["Geometry"])
    modifier = obj.modifiers.new(f"GraSPH Live {kind.title()} Display", "NODES")
    modifier.node_group = group
    modifier[radius_socket.identifier] = float(radius)


def _preview_source_object(scene, prim_path, role):
    # USD mesh paths commonly end in the mesh datablock name, while the
    # GraSPH role lives on the Blender object represented by its parent Xform.
    # Prefer the parent component so `/root/Cube_002/Cube_004` resolves to
    # `Cube.002`, even if another role-tagged Blender object is named
    # `Cube.004`. Fall back to other components for direct-mesh USD prims.
    path_names = [
        re.sub(r"[^A-Za-z0-9_]+", "_", part)
        for part in str(prim_path).split("/") if part and part.lower() != "root"
    ]
    candidates = [
        obj for obj in scene.objects
        if getattr(getattr(obj, "grasph", None), "role", "NONE") == role
    ]
    preferred_names = list(reversed(path_names[:-1])) + path_names[-1:]
    for path_name in preferred_names:
        for obj in candidates:
            normalized_name = re.sub(r"[^A-Za-z0-9_]+", "_", obj.name)
            if obj.name == path_name or normalized_name == path_name:
                return obj
    return None
    return cloud


def _update_live_preview(scene, path):
    global _PREVIEW_CLOUDS, _PREVIEW_LAST_PATHS
    path = Path(path)
    if not path.is_file():
        return None
    import numpy as np
    try:
        with np.load(path) as preview:
            fluid = preview["positions"].copy()
            fluid_velocity = preview["velocity"].copy()
            grains = preview["granular_positions"].copy()
            grain_velocity = preview["granular_velocity"].copy()
            grain_radii = preview["granular_radii"].copy()
            if (grain_radii.shape != (len(grains),) or
                    not np.all(np.isfinite(grain_radii)) or np.any(grain_radii <= 0)):
                return None
            time_value = float(preview["time"])
            progress = float(preview["progress"])
            body_paths = preview["body_paths"].astype(str).tolist()
            body_translations = preview["body_translations"].copy()
            body_rotations = preview["body_rotations"].copy()
            hinge_paths = preview["hinge_paths"].astype(str).tolist()
            hinge_angles = preview["hinge_angles"].copy()
    except (OSError, ValueError, KeyError):
        return None

    for kind, coordinates, speeds in (
        ("fluid", fluid, fluid_velocity), ("granular", grains, grain_velocity),
    ):
        cloud_name = _PREVIEW_CLOUDS.get(kind)
        cloud = bpy.data.objects.get(cloud_name) if cloud_name else None
        if cloud is None or cloud.get("grasph:live_preview") != kind:
            cloud_data = bpy.data.meshes.new(f"GraSPH Live {kind.title()} Data")
            cloud = bpy.data.objects.new(f"GraSPH Live {kind.title()} Preview", cloud_data)
            scene.collection.objects.link(cloud)
            cloud["grasph:live_preview"] = kind
            cloud.hide_render = True
        _PREVIEW_CLOUDS[kind] = cloud.name
        old_data = cloud.data
        updated_data = _update_point_cloud(old_data, coordinates, speeds)
        if updated_data is not old_data:
            cloud.data = updated_data
            if old_data.users == 0:
                bpy.data.meshes.remove(old_data)
        if kind == "granular":
            attribute = cloud.data.attributes.get("grasph_radius")
            if attribute is None:
                attribute = cloud.data.attributes.new("grasph_radius", "FLOAT", "POINT")
            attribute.data.foreach_set("value", grain_radii.astype("f", copy=False))
            cloud.data.update()
        if cloud.get("grasph:live_display_ready") != kind and len(coordinates):
            try:
                radius = (1.0
                          if kind == "granular" else
                          max(float(scene.grasph_scene.dp) * 0.18, 1.0e-5))
                _add_live_particle_display(cloud, kind, radius)
                cloud["grasph:live_display_ready"] = kind
            except Exception:
                pass

    for path_string, translation, rotation in zip(
        body_paths, body_translations, body_rotations
    ):
        source = _preview_source_object(scene, path_string, "DYNAMIC_BODY")
        if source is None:
            continue
        origin = source.get("grasph:preview_origin")
        if origin is None:
            origin = tuple(source.location)
            source["grasph:preview_origin"] = origin
        source.location = tuple(origin[i] + float(translation[i]) for i in range(3))
        rotation_origin = source.get("grasph:preview_rotation_origin")
        if rotation_origin is None:
            rotation_origin = tuple(source.rotation_euler)
            source["grasph:preview_rotation_origin"] = rotation_origin
        source.rotation_euler = tuple(
            float(rotation_origin[i]) + float(rotation[i]) for i in range(3)
        )
    for path_string, angle in zip(hinge_paths, hinge_angles):
        source = _preview_source_object(scene, path_string, "DYNAMIC_HINGE")
        if source is None:
            continue
        source.rotation_mode = "XYZ"
        axis = source.grasph.axis
        index = {"X": 0, "Y": 1, "Z": 2}[axis]
        if source.name not in _PREVIEW_LAST_PATHS:
            _PREVIEW_LAST_PATHS.append(source.name)
            source["grasph:preview_rotation_origin"] = tuple(source.rotation_euler)
        baseline = source.get("grasph:preview_rotation_origin", tuple(source.rotation_euler))
        source.rotation_euler[index] = float(baseline[index]) + float(angle)
    outline_collection = bpy.data.collections.get(_COLLIDER_OUTLINE_COLLECTION)
    if outline_collection is not None:
        for outline in outline_collection.objects:
            source = scene.objects.get(outline.get("grasph:outline_source", ""))
            if source is not None:
                outline.matrix_world = source.matrix_world.copy()
    settings = scene.grasph_scene
    settings.live_preview_progress = 100.0 * progress
    settings.live_preview_time = time_value
    return progress, time_value


def _clear_live_preview(scene):
    global _PREVIEW_CLOUDS, _PREVIEW_LAST_PATHS
    for obj in list(scene.objects):
        if obj.get("grasph:live_preview") in {"fluid", "granular"}:
            bpy.data.objects.remove(obj, do_unlink=True)
        else:
            origin = obj.get("grasph:preview_origin")
            rotation = obj.get("grasph:preview_rotation_origin")
            if origin is not None:
                obj.location = origin
                del obj["grasph:preview_origin"]
            if rotation is not None:
                obj.rotation_euler = rotation
                del obj["grasph:preview_rotation_origin"]
    _PREVIEW_CLOUDS = {"fluid": None, "granular": None}
    _PREVIEW_LAST_PATHS = []


def _prepare_live_preview_for_run(scene, preview_file):
    """Remove the previous run's viewport data and snapshot before launching."""
    _clear_live_preview(scene)
    try:
        Path(preview_file).unlink(missing_ok=True)
    except OSError:
        # A stale snapshot must never be presented as the new run. If Windows
        # has it locked, the timestamp guard below still prevents re-importing
        # it as a fresh update.
        pass


def _reset_run_state(scene):
    settings = scene.grasph_scene
    checkpoint_path = str(_path(settings.output_dir) /
                          (settings.output_name + "_checkpoint.usdc"))
    for path in (settings.pause_file,
                 str(_path(settings.output_dir) /
                     (settings.output_name + "_live_preview.npz")),
                 checkpoint_path):
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            pass
    _clear_live_preview(scene)
    settings.run_progress = 0.0
    settings.live_preview_progress = 0.0
    settings.live_preview_time = 0.0
    settings.run_elapsed = "00:00"
    settings.run_message = "Reset; ready to start from the beginning"
    settings.run_status = "Ready"
    settings.continue_from_result = False
    settings.restart_npz = ""
    settings.restart_time = 0.0


def _discover_project_root():
    """Find the repository from the add-on, Blender file, or loaded text file."""
    starts = []
    try:
        starts.append(Path(__file__).resolve())
    except (NameError, OSError):
        pass
    # Blender restricts bpy.data while importing an installed add-on.
    blend_path = getattr(bpy.data, "filepath", "")
    if blend_path:
        starts.append(Path(blend_path).resolve())
    # When this script is run from Blender's Text Editor, __file__ may be
    # undefined (or point to the saved .blend). Text datablocks retain the
    # source file path, which is the reliable project anchor in that workflow.
    for text_block in getattr(bpy.data, "texts", ()):
        text_path = str(getattr(text_block, "filepath", "") or "")
        if text_path and Path(text_path).name.lower() == "grasph_control.py":
            try:
                starts.append(Path(text_path).resolve())
            except (OSError, RuntimeError):
                pass
    starts.append(Path.cwd().resolve())
    visited = set()
    for start in starts:
        folder = start.parent if start.suffix else start
        for candidate in (folder, *folder.parents):
            if candidate in visited:
                continue
            visited.add(candidate)
            if ((candidate / "grasph.py").is_file()
                    and (candidate / "blender" / "grasph_surface.py").is_file()):
                return candidate
    return None


def _default_project_root():
    candidate = _discover_project_root()
    return str(candidate) if candidate is not None else ""


def _ensure_project_root(settings):
    """Populate an empty/stale scene path from the active GraSPH installation."""
    # An installed package must not keep depending on an old development repo
    # merely because a .blend saved its previous project_root setting.
    if __package__ and __package__.endswith(".blender"):
        packaged = _discover_project_root()
        if packaged is not None:
            settings.project_root = str(packaged)
            return packaged
    current = str(getattr(settings, "project_root", "") or "").strip()
    if current:
        try:
            if (_path(current) / "grasph.py").is_file():
                return _path(current)
        except (OSError, RuntimeError, ValueError):
            pass
    discovered = _discover_project_root()
    if discovered is not None:
        settings.project_root = str(discovered)
        return discovered
    return None


def _path(value):
    # Do not let Blender resolve // against its installation/current-working
    # directory. A saved blend owns relative paths; an unsaved blend falls
    # back to the discovered GraSPH project root.
    if str(value).startswith("//"):
        if bpy.data.filepath:
            base = Path(bpy.data.filepath).resolve().parent
        else:
            base = _discover_project_root() or Path.cwd().resolve()
        relative = str(value)[2:].replace("/", os.sep).replace("\\", os.sep)
        return (base / relative).resolve()
    return Path(bpy.path.abspath(value)).resolve()


def _npz_end_time(path, fallback):
    try:
        import numpy as np
        with np.load(path, allow_pickle=False) as result:
            return float(result["times"][-1])
    except (OSError, ImportError, KeyError, ValueError, IndexError):
        return float(fallback)


def _tagged_objects(scene):
    return [obj for obj in scene.objects if obj.grasph.role != "NONE"]


def _companion_module(filename, module_name, scene=None):
    candidates = []
    if scene is not None and hasattr(scene, "grasph_scene"):
        project_root = scene.grasph_scene.project_root
        if project_root:
            candidates.append(_path(project_root) / "blender" / filename)
    discovered_root = _discover_project_root()
    if discovered_root is not None:
        candidates.append(discovered_root / "blender" / filename)
    try:
        candidates.append(Path(__file__).resolve().with_name(filename))
    except (NameError, OSError):
        pass
    candidates.append(Path.cwd() / "blender" / filename)
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        searched = ", ".join(str(candidate) for candidate in candidates)
        raise RuntimeError(
            f"{filename} not found. Set GraSPH Project to the folder containing "
            f"grasph.py. Searched: {searched}"
        )
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _environment_module(scene=None):
    global _ENVIRONMENT_MODULE
    if _ENVIRONMENT_MODULE is None:
        _ENVIRONMENT_MODULE = _companion_module(
            "grasph_environment.py", "grasph_environment_runtime", scene)
    return _ENVIRONMENT_MODULE


def _solver_python(settings, scene):
    environment = _environment_module(scene)
    if not settings.use_managed_environment:
        return environment.split_python_command(settings.python_command)
    root = _ensure_project_root(settings)
    if root is None:
        raise RuntimeError("GraSPH solver files could not be found")
    path = environment.environment_path(root, environment.user_data_path())
    if not environment.is_ready(root, path):
        raise RuntimeError("Install the solver environment under Solver Installation first")
    return [str(environment.python_path(path))]


def _poll_environment_job():
    global _ENV_JOB
    job = _ENV_JOB
    if job is None:
        return None
    result = job["process"].poll()
    if result is None:
        return 1.0
    job["log_handle"].close()
    scene = bpy.data.scenes.get(job["scene"])
    if scene is not None and hasattr(scene, "grasph_scene"):
        settings = scene.grasph_scene
        settings.environment_status = (
            "Installation checked: ready" if result == 0 else
            "Setup/check failed; inspect the installation log")
    _ENV_JOB = None
    if bpy.context.screen:
        _tag_redraw(bpy.context)
    return None


def _surface_module(scene=None):
    return _companion_module(
        "grasph_surface.py", "grasph_surface_runtime", scene
    )


def _granular_display_module(scene=None):
    return _companion_module(
        "grasph_granular_display.py", "grasph_granular_display_runtime", scene
    )


def _surface_modifier(obj):
    """Return the GraSPH display modifier when *obj* is a valid surface."""
    if obj is None or not hasattr(obj, "modifiers"):
        return None
    modifier = obj.modifiers.get("GraSPH Surface")
    if modifier is not None and modifier.node_group is not None:
        return modifier
    # Legacy surfaces and copied objects may have an automatically suffixed
    # modifier name. Identify those by the sockets published by GraSPH.
    for candidate in obj.modifiers:
        group = getattr(candidate, "node_group", None)
        if candidate.type != "NODES" or group is None:
            continue
        sockets = group.get("grasph_sockets", {})
        if "Mesh Radius" in sockets or "Show Surface" in sockets:
            return candidate
    return None


def _find_surface_object(scene, active=None):
    """Prefer the active surface, otherwise find an existing one in the scene."""
    if _surface_modifier(active) is not None:
        return active
    return next(
        (obj for obj in scene.objects if _surface_modifier(obj) is not None),
        None,
    )


def _find_surface_source(scene, active=None):
    """Find the imported point cloud behind an active or existing surface."""
    if active is not None and active.type == "POINTCLOUD":
        return active

    surface = _find_surface_object(scene, active)
    if surface is not None and surface.get("grasph_source"):
        source = bpy.data.objects.get(surface["grasph_source"])
        if source is not None and source.type == "POINTCLOUD":
            return source

    # Prefer the visible/imported result over the hidden cache copy.
    clouds = [obj for obj in scene.objects if obj.type == "POINTCLOUD"]
    return next(
        (obj for obj in clouds if not obj.name.endswith("_RawPoints")),
        clouds[0] if clouds else None,
    )


def _granular_modifier(obj):
    if obj is None or not hasattr(obj, "modifiers"):
        return None
    modifier = obj.modifiers.get("GraSPH Granular Display")
    if modifier is None or modifier.node_group is None:
        return None
    return modifier


def _find_granular_result(scene, active=None):
    if active is not None and active.type == "POINTCLOUD":
        if _granular_modifier(active) is not None or "Granular" in active.name:
            return active
    return next(
        (obj for obj in scene.objects if _granular_modifier(obj) is not None),
        None,
    )


def _object_center(obj, settings):
    if settings.center_source == "ORIGIN":
        return tuple(float(v) for v in obj.matrix_world.translation)
    return tuple(float(v) for v in settings.axle_center)


def scene_summary(scene):
    """Cheap UI estimate; the solver remains the authority after export."""
    tagged = _tagged_objects(scene)
    fluids = [obj for obj in tagged if obj.grasph.role == "FLUID"]
    grains = [obj for obj in tagged if obj.grasph.role == "GRANULAR_SOURCE"]
    colliders = [obj for obj in tagged if obj.grasph.role in {
        "STATIC", "KINEMATIC", "DYNAMIC_HINGE", "DYNAMIC_BODY"
    }]
    dynamic = [obj for obj in tagged if obj.grasph.role in {
        "DYNAMIC_HINGE", "DYNAMIC_BODY"
    }]
    scale = float(scene.unit_settings.scale_length or 1.0)
    dp = max(float(scene.grasph_scene.dp), 1.0e-12)

    def packed_count(obj, spacing):
        return math.prod(
            max(0, int(float(dimension) * scale / spacing))
            for dimension in obj.dimensions
        )

    fluid_particles = sum(packed_count(obj, dp) for obj in fluids)
    grain_particles = 0
    for obj in grains:
        settings = obj.grasph
        if settings.granular_source_mode == "FLOW":
            duration = max(0.0, settings.granular_end_time - settings.granular_start_time)
            grain_particles += int(settings.granular_flow_rate * duration)
        elif settings.granular_radius > 0.0:
            grain_particles += packed_count(obj, 2.0 * settings.granular_radius)
    total = fluid_particles + grain_particles
    memory_mb = (fluid_particles * 360 + grain_particles * 220) / (1024.0 ** 2)
    workload = "Light" if total < 10000 else "Medium" if total < 50000 else "Heavy"
    return {
        "fluid_sources": len(fluids), "granular_sources": len(grains),
        "colliders": len(colliders), "dynamic": len(dynamic),
        "fluid_particles": fluid_particles, "grain_particles": grain_particles,
        "memory_mb": memory_mb, "workload": workload,
        "mixed": bool(fluids and grains),
    }


def validate_scene(scene):
    errors, warnings = [], []
    tagged = _tagged_objects(scene)
    fluid_objects = [obj for obj in tagged if obj.grasph.role == "FLUID"]
    granular_objects = [obj for obj in tagged if obj.grasph.role == "GRANULAR_SOURCE"]
    dynamic_objects = [obj for obj in tagged if obj.grasph.role in {
        "DYNAMIC_HINGE", "DYNAMIC_BODY"
    }]
    if not fluid_objects and not granular_objects and not dynamic_objects:
        errors.append(
            "Assign a Fluid Volume, Granular Source, or moving rigid body to simulate."
        )
    if fluid_objects and granular_objects:
        warnings.append(
            "Mixed fluid/granular mode enabled: grains use buoyancy and drag "
            "with equal-and-opposite reaction on the SPH fluid."
        )
        unit_scale = float(scene.unit_settings.scale_length or 1.0)
        dp = float(scene.grasph_scene.dp)
        fluid_estimate = sum(
            max(0, int(float(obj.dimensions[0]) * unit_scale / dp))
            * max(0, int(float(obj.dimensions[1]) * unit_scale / dp))
            * max(0, int(float(obj.dimensions[2]) * unit_scale / dp))
            for obj in fluid_objects
        )
        if fluid_estimate < 64:
            warnings.append(
                f"Mixed scene has only about {fluid_estimate} fluid particles at "
                f"spacing {dp:g} m; reduce Particle Spacing for stable coupling."
            )
        grain_estimate = 0
        for obj in granular_objects:
            radius = float(obj.grasph.granular_radius)
            if obj.grasph.granular_source_mode != "VOLUME" or radius <= 0.0:
                continue
            diameter = 2.0 * radius
            grain_estimate += (
                max(0, int(float(obj.dimensions[0]) * unit_scale / diameter))
                * max(0, int(float(obj.dimensions[1]) * unit_scale / diameter))
                * max(0, int(float(obj.dimensions[2]) * unit_scale / diameter))
            )
        if grain_estimate > 10000:
            warnings.append(
                f"Granular volumes create about {grain_estimate:,} grains; "
                "increase Granular Radius for a faster first test."
            )
        smallest_radius = min(
            obj.grasph.granular_radius for obj in granular_objects
        )
        if smallest_radius < 0.2 * dp:
            warnings.append(
                f"Granular radius {smallest_radius:g} m is much smaller than fluid "
                f"spacing {dp:g} m; coupling will be coarse."
            )
    dynamic_bodies = [obj for obj in tagged if obj.grasph.role == "DYNAMIC_BODY"]
    if fluid_objects and dynamic_bodies and not scene.grasph_scene.enable_solid_coupling:
        warnings.append(
            "Dynamic Body objects remain fixed during fluid simulation until "
            "Fluid / Solid Coupling is enabled in Advanced Settings."
        )
    if granular_objects:
        smallest_radius = min(obj.grasph.granular_radius for obj in granular_objects)
        timestep_hint = 0.15 * math.sqrt(max(smallest_radius, 1.0e-12) / 9.81)
        if scene.grasph_scene.granular_dt > timestep_hint:
            warnings.append(
                f"Granular Time Step {scene.grasph_scene.granular_dt:g} s is large "
                f"for radius {smallest_radius:g} m; try {timestep_hint:.4g} s or less."
            )

    for obj in tagged:
        settings = obj.grasph
        physical = None
        if settings.role in _COLLIDER_ROLES | {'GRANULAR_SOURCE'}:
            try:
                physical = _physics_values(settings)
            except ValueError as error:
                errors.append(f'{obj.name}: {error}')
                continue
        if settings.role == "JOINT":
            if obj.type != "EMPTY":
                errors.append(f"{obj.name}: Joint Anchor must be an Empty.")
            body = settings.joint_body1
            if body is None or body.grasph.role != "DYNAMIC_BODY":
                errors.append(f"{obj.name}: assign a Dynamic Body as Body B.")
            other = settings.joint_body0
            if other is not None and (other == body or other.grasph.role != 'DYNAMIC_BODY' or other not in tagged):
                errors.append(f'{obj.name}: Body A must be another Dynamic Body in this scene.')
            if body is not None and body not in tagged:
                errors.append(f'{obj.name}: Body B must belong to this scene.')
            if settings.joint_type != 'FIXED' and (other is not None or settings.joint_breakable):
                errors.append(f'{obj.name}: two-body/breakable joints currently require Fixed type.')
            if settings.joint_type == 'FIXED' and fluid_objects and not scene.grasph_scene.enable_solid_coupling:
                errors.append(f'{obj.name}: enable Fluid / Solid Coupling to simulate joints with fluid.')
            continue
        if obj.type != "MESH":
            errors.append(f"{obj.name}: role {settings.role} requires a mesh object.")
        if settings.role == "DYNAMIC_HINGE":
            if settings.moment_of_inertia <= 0.0:
                errors.append(f"{obj.name}: moment of inertia must be positive.")
            if settings.max_angular_speed <= 0.0:
                warnings.append(f"{obj.name}: angular speed is unlimited.")
        if settings.role == "DYNAMIC_BODY":
            if settings.body_mass <= 0.0:
                errors.append(f"{obj.name}: dynamic body mass must be positive.")
            if min(settings.body_inertia) <= 0.0:
                errors.append(f"{obj.name}: dynamic body inertia must be positive.")
            if settings.body_collision_shape == "CONVEX" and settings.body_collision_margin <= 0.0:
                errors.append(f"{obj.name}: convex collision margin must be positive.")
        if settings.role == "GRANULAR_SOURCE":
            if settings.granular_radius <= 0.0:
                errors.append(f"{obj.name}: granular radius must be positive.")
            if physical['density'] <= 0.0:
                errors.append(f"{obj.name}: granular density must be positive.")
            if physical['restitution'] > 0.5:
                warnings.append(
                    f"{obj.name}: restitution {physical['restitution']:g} "
                    "is highly energetic in a dense granular volume."
                )
            if settings.granular_source_mode == "FLOW":
                if settings.granular_flow_rate <= 0.0:
                    errors.append(f"{obj.name}: granular flow rate must be positive.")
                if settings.granular_end_time < settings.granular_start_time:
                    errors.append(f"{obj.name}: flow end time precedes start time.")
        if settings.role == "KINEMATIC" and not obj.animation_data:
            warnings.append(f"{obj.name}: kinematic collider has no object animation.")

    constrained_bodies = {}
    for joint_obj in (obj for obj in tagged if obj.grasph.role == "JOINT"):
        if joint_obj.grasph.joint_type == 'FIXED':
            continue
        body = joint_obj.grasph.joint_body1
        if body is None:
            continue
        if body.name in constrained_bodies:
            errors.append(
                f"{body.name}: only one joint anchor per body is supported "
                f"({constrained_bodies[body.name]} and {joint_obj.name})."
            )
        constrained_bodies[body.name] = joint_obj.name

    if not any(obj.grasph.role in {"STATIC", "KINEMATIC", "DYNAMIC_HINGE", "DYNAMIC_BODY"}
               for obj in tagged):
        warnings.append("No colliders are explicitly assigned.")
    return errors, warnings


def _matching_xform(stage, object_name):
    from pxr import Tf, UsdGeom

    valid_name = Tf.MakeValidIdentifier(object_name)
    direct = stage.GetPrimAtPath("/root/" + valid_name)
    if direct and direct.IsValid():
        return direct
    matches = [prim for prim in stage.Traverse()
               if prim.GetName() == valid_name and prim.IsA(UsdGeom.Xform)]
    return matches[0] if len(matches) == 1 else None


def _author_usd_physics(stage, objects, scene):
    """Add portable rigid-body semantics without replacing GraSPH metadata."""
    from mathutils import Vector
    from pxr import Gf, Sdf, Tf, Usd, UsdGeom, UsdPhysics, UsdShade

    physics_root = stage.DefinePrim("/root/_GraSPHPhysics", "Scope")
    physics_scene = UsdPhysics.Scene.Define(stage, physics_root.GetPath().AppendChild("Scene"))
    gravity = scene.gravity
    gravity_length = float(gravity.length)
    if gravity_length > 1.0e-12:
        direction = Gf.Vec3f(*(float(value) / gravity_length for value in gravity))
        physics_scene.CreateGravityDirectionAttr(direction)
        metres_per_unit = max(float(UsdGeom.GetStageMetersPerUnit(stage)), 1.0e-12)
        physics_scene.CreateGravityMagnitudeAttr(gravity_length / metres_per_unit)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    joints_root = stage.DefinePrim(physics_root.GetPath().AppendChild("Joints"), "Scope")
    materials_root = stage.DefinePrim(
        physics_root.GetPath().AppendChild("Materials"), "Scope"
    )

    for obj in objects:
        settings = obj.grasph
        if settings.role not in {"STATIC", "KINEMATIC", "DYNAMIC_HINGE", "DYNAMIC_BODY"}:
            continue
        physical = _physics_values(settings)
        owner = _matching_xform(stage, obj.name)
        if owner is None:
            continue
        meshes = [prim for prim in Usd.PrimRange(owner) if prim.IsA(UsdGeom.Mesh)]
        material_name = Tf.MakeValidIdentifier(
            (settings.physics_material.name if settings.physics_material else obj.name) + "_PhysicsMaterial")
        if settings.physics_material:
            import hashlib
            material_name += '_' + hashlib.sha256(settings.physics_material.name.encode('utf-8')).hexdigest()[:12]
        material = UsdShade.Material.Define(
            stage, materials_root.GetPath().AppendChild(material_name)
        )
        material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        material_api.CreateStaticFrictionAttr(float(physical['friction']))
        material_api.CreateDynamicFrictionAttr(float(physical['friction']))
        material_api.CreateRestitutionAttr(float(physical['restitution']))
        for mesh in meshes:
            UsdPhysics.CollisionAPI.Apply(mesh).CreateCollisionEnabledAttr(True)
            mesh_api = UsdPhysics.MeshCollisionAPI.Apply(mesh)
            approximation = UsdPhysics.Tokens.none
            if settings.role in {"DYNAMIC_HINGE", "DYNAMIC_BODY"}:
                approximation = (
                    UsdPhysics.Tokens.boundingSphere
                    if settings.role == "DYNAMIC_BODY"
                    and settings.body_collision_shape == "SPHERE"
                    else UsdPhysics.Tokens.convexHull
                )
            mesh_api.CreateApproximationAttr(approximation)
            UsdShade.MaterialBindingAPI.Apply(mesh).Bind(
                material, UsdShade.Tokens.weakerThanDescendants, "physics"
            )

        if settings.role == "STATIC":
            continue
        rigid = UsdPhysics.RigidBodyAPI.Apply(owner)
        rigid.CreateRigidBodyEnabledAttr(True)
        rigid.CreateKinematicEnabledAttr(settings.role == "KINEMATIC")
        if settings.role == "KINEMATIC":
            continue

        mass_api = UsdPhysics.MassAPI.Apply(owner)
        center_world = _object_center(obj, settings)
        center_local = obj.matrix_world.inverted() @ Vector(center_world)
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, center_local)))
        if settings.role == "DYNAMIC_BODY":
            mass_api.CreateMassAttr(float(settings.body_mass))
            mass_api.CreateDiagonalInertiaAttr(
                Gf.Vec3f(*map(float, settings.body_inertia))
            )
            continue

        # GraSPH's hinge currently exposes only the inertia around its free
        # axis. A spherical tensor is the least surprising portable fallback;
        # the exact scalar remains authored as grasph:momentOfInertia.
        inertia = float(settings.moment_of_inertia)
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(inertia, inertia, inertia))
        joint_name = Tf.MakeValidIdentifier(obj.name + "_RevoluteJoint")
        joint = UsdPhysics.RevoluteJoint.Define(
            stage, joints_root.GetPath().AppendChild(joint_name)
        )
        joint.CreateBody1Rel().SetTargets([owner.GetPath()])
        joint.CreateAxisAttr(getattr(UsdPhysics.Tokens, settings.axis.lower()))
        joint.CreateLocalPos0Attr(Gf.Vec3f(*map(float, center_world)))
        joint.CreateLocalPos1Attr(Gf.Vec3f(*map(float, center_local)))
        joint.CreateLocalRot0Attr(Gf.Quatf(1.0))
        rotation = obj.matrix_world.to_quaternion().inverted()
        joint.CreateLocalRot1Attr(Gf.Quatf(
            float(rotation.w),
            Gf.Vec3f(float(rotation.x), float(rotation.y), float(rotation.z)),
        ))
        joint.CreateCollisionEnabledAttr(False)

    bodies_with_world_joint = set()
    for obj in objects:
        settings = obj.grasph
        if settings.role != "JOINT":
            continue
        body = settings.joint_body1
        if body is None or body.grasph.role != "DYNAMIC_BODY":
            continue
        body_prim = _matching_xform(stage, body.name)
        if body_prim is None:
            raise RuntimeError(
                f"Joint {obj.name!r} cannot find its body {body.name!r} in exported USD"
            )
        kind = settings.joint_type
        if kind == 'FIXED':
            other = settings.joint_body0
            other_prim = _matching_xform(stage, other.name) if other else None
            joint = UsdPhysics.FixedJoint.Define(stage, joints_root.GetPath().AppendChild(Tf.MakeValidIdentifier(obj.name)))
            joint.CreateBody1Rel().SetTargets([body_prim.GetPath()])
            if other:
                if other_prim is None:
                    raise RuntimeError(f'Cannot export joint Body A: {other.name}')
                joint.CreateBody0Rel().SetTargets([other_prim.GetPath()])
            anchor = obj.matrix_world.translation.copy()
            frame = obj.matrix_world.to_quaternion()
            for index, endpoint in ((0, other), (1, body)):
                position = endpoint.matrix_world.inverted() @ anchor if endpoint else anchor
                rotation = endpoint.matrix_world.to_quaternion().inverted() @ frame if endpoint else frame
                getattr(joint, f'CreateLocalPos{index}Attr')(Gf.Vec3f(*map(float, position)))
                getattr(joint, f'CreateLocalRot{index}Attr')(Gf.Quatf(float(rotation.w), Gf.Vec3f(float(rotation.x), float(rotation.y), float(rotation.z))))
            joint.CreateBreakForceAttr(settings.joint_break_force if settings.joint_breakable else float('inf'))
            joint.CreateBreakTorqueAttr(settings.joint_break_torque if settings.joint_breakable else float('inf'))
            joint.CreateCollisionEnabledAttr(False)
            joint.GetPrim().CreateAttribute('grasph:managedJoint', Sdf.ValueTypeNames.Bool).Set(True)
            continue
        anchor = obj.matrix_world.translation.copy()
        axis_local = Vector({
            "X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0),
            "Z": (0.0, 0.0, 1.0),
        }[settings.joint_axis])
        axis_world = obj.matrix_world.to_3x3() @ axis_local
        if axis_world.length > 1.0e-12:
            axis_world.normalize()
        for prim in (body_prim, *[p for p in Usd.PrimRange(body_prim)
                                  if p.IsA(UsdGeom.Mesh)]):
            prim.CreateAttribute(
                "grasph:jointType", Sdf.ValueTypeNames.String
            ).Set(kind.lower())
            prim.CreateAttribute(
                "grasph:jointAxis", Sdf.ValueTypeNames.Double3
            ).Set(Gf.Vec3d(*map(float, axis_world)))
            prim.CreateAttribute(
                "grasph:jointAnchor", Sdf.ValueTypeNames.Double3
            ).Set(Gf.Vec3d(*map(float, anchor)))
        if kind == "FREE":
            continue
        if body.name in bodies_with_world_joint:
            raise RuntimeError(
                f"Dynamic Body {body.name!r} has multiple world joints; "
                "this solver supports one constraint per body"
            )
        bodies_with_world_joint.add(body.name)

        joint_path = joints_root.GetPath().AppendChild(
            Tf.MakeValidIdentifier(obj.name + "_" + kind.title() + "Joint")
        )
        if kind == "REVOLUTE":
            joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
            joint.CreateAxisAttr(getattr(UsdPhysics.Tokens, settings.joint_axis.lower()))
        elif kind == "PRISMATIC":
            joint = UsdPhysics.PrismaticJoint.Define(stage, joint_path)
            joint.CreateAxisAttr(getattr(UsdPhysics.Tokens, settings.joint_axis.lower()))
        elif kind == "FIXED":
            joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
        else:
            raise RuntimeError(f"Unsupported GraSPH joint type: {kind}")

        joint.CreateBody1Rel().SetTargets([body_prim.GetPath()])
        joint_rotation = obj.matrix_world.to_quaternion()
        body_rotation = body.matrix_world.to_quaternion()
        body_local_anchor = body.matrix_world.inverted() @ anchor
        body_local_rotation = body_rotation.inverted() @ joint_rotation
        joint.CreateLocalPos0Attr(Gf.Vec3f(*map(float, anchor)))
        joint.CreateLocalPos1Attr(Gf.Vec3f(*map(float, body_local_anchor)))
        # The conversion uses the quaternion's scalar/vector components, not
        # Euler angles; localRot1 preserves the anchor frame on the body.
        joint.CreateLocalRot0Attr(Gf.Quatf(
            float(joint_rotation.w),
            Gf.Vec3f(float(joint_rotation.x), float(joint_rotation.y),
                     float(joint_rotation.z)),
        ))
        joint.CreateLocalRot1Attr(Gf.Quatf(
            float(body_local_rotation.w),
            Gf.Vec3f(float(body_local_rotation.x), float(body_local_rotation.y),
                     float(body_local_rotation.z)),
        ))
        joint.CreateCollisionEnabledAttr(False)


def author_grasph_metadata(usd_path, objects, scene=None):
    """Write simulation semantics into the exported USD layer."""
    from mathutils import Vector
    from pxr import Gf, Sdf, Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Could not reopen exported USD: {usd_path}")

    if scene is not None and scene.grasph_scene.author_usd_physics:
        _author_usd_physics(stage, objects, scene)

    for obj in objects:
        settings = obj.grasph
        owner = _matching_xform(stage, obj.name)
        if owner is None:
            raise RuntimeError(f"Could not match Blender object {obj.name!r} in USD export")
        targets = [owner]
        targets.extend(prim for prim in Usd.PrimRange(owner) if prim.IsA(UsdGeom.Mesh))
        role = ROLE_TO_USD[settings.role]
        for prim in targets:
            prim.CreateAttribute("grasph:role", Sdf.ValueTypeNames.String).Set(role)

        if settings.role == "JOINT":
            body = settings.joint_body1
            if body is None:
                continue
            body_owner = _matching_xform(stage, body.name)
            if body_owner is None:
                raise RuntimeError(
                    f"Joint {obj.name!r} cannot find its Dynamic Body {body.name!r}"
                )
            if settings.joint_type == 'FIXED':
                owner.CreateAttribute('grasph:jointType', Sdf.ValueTypeNames.String).Set('fixed')
                owner.CreateRelationship('grasph:bodyB').SetTargets([body_owner.GetPath()])
                other = settings.joint_body0
                targets_a = []
                if other:
                    other_owner = _matching_xform(stage, other.name)
                    if other_owner is None:
                        raise RuntimeError(f'Cannot export joint Body A: {other.name}')
                    targets_a = [other_owner.GetPath()]
                owner.CreateRelationship('grasph:bodyA').SetTargets(targets_a)
                owner.CreateAttribute('grasph:jointAnchor', Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(*obj.matrix_world.translation))
                owner.CreateAttribute('grasph:breakable', Sdf.ValueTypeNames.Bool).Set(settings.joint_breakable)
                owner.CreateAttribute('grasph:breakForce', Sdf.ValueTypeNames.Double).Set(settings.joint_break_force)
                owner.CreateAttribute('grasph:breakTorque', Sdf.ValueTypeNames.Double).Set(settings.joint_break_torque)
                continue
            axis_local = Vector({
                "X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0),
                "Z": (0.0, 0.0, 1.0),
            }[settings.joint_axis])
            axis_world = obj.matrix_world.to_3x3() @ axis_local
            if axis_world.length > 1.0e-12:
                axis_world.normalize()
            anchor = obj.matrix_world.translation
            body_targets = [body_owner]
            body_targets.extend(
                prim for prim in Usd.PrimRange(body_owner)
                if prim.IsA(UsdGeom.Mesh)
            )
            for prim in body_targets:
                prim.CreateAttribute(
                    "grasph:jointType", Sdf.ValueTypeNames.String
                ).Set(settings.joint_type.lower())
                prim.CreateAttribute(
                    "grasph:jointAxis", Sdf.ValueTypeNames.Double3
                ).Set(Gf.Vec3d(*map(float, axis_world)))
                prim.CreateAttribute(
                    "grasph:jointAnchor", Sdf.ValueTypeNames.Double3
                ).Set(Gf.Vec3d(*map(float, anchor)))
            continue

        if settings.role in {"STATIC", "KINEMATIC", "DYNAMIC_HINGE", "DYNAMIC_BODY"}:
            physical = _physics_values(settings)
            for prim in targets:
                prim.CreateAttribute(
                    "grasph:friction", Sdf.ValueTypeNames.Double
                ).Set(physical['friction'])
                prim.CreateAttribute(
                    "grasph:restitution", Sdf.ValueTypeNames.Double
                ).Set(physical['restitution'])
                prim.CreateAttribute(
                    "grasph:rollingFriction", Sdf.ValueTypeNames.Double
                ).Set(physical['rolling'])
                if settings.physics_material:
                    prim.CreateAttribute('grasph:physicsMaterial', Sdf.ValueTypeNames.String).Set(settings.physics_material.name)

        if settings.role == "DYNAMIC_HINGE":
            center = _object_center(obj, settings)
            for prim in targets:
                prim.CreateAttribute("grasph:hingeEnabled", Sdf.ValueTypeNames.Bool).Set(True)
                prim.CreateAttribute("grasph:axis", Sdf.ValueTypeNames.String).Set(settings.axis)
                prim.CreateAttribute("grasph:axleCenter", Sdf.ValueTypeNames.Double3).Set(
                    Gf.Vec3d(*center)
                )
                prim.CreateAttribute(
                    "grasph:momentOfInertia", Sdf.ValueTypeNames.Double
                ).Set(settings.moment_of_inertia)
                prim.CreateAttribute(
                    "grasph:angularDamping", Sdf.ValueTypeNames.Double
                ).Set(settings.angular_damping)
                prim.CreateAttribute(
                    "grasph:maxAngularSpeed", Sdf.ValueTypeNames.Double
                ).Set(settings.max_angular_speed)
        elif settings.role == "DYNAMIC_BODY":
            center = _object_center(obj, settings)
            for prim in targets:
                prim.CreateAttribute(
                    "grasph:dynamicBodyEnabled", Sdf.ValueTypeNames.Bool
                ).Set(True)
                prim.CreateAttribute(
                    "grasph:bodyCenter", Sdf.ValueTypeNames.Double3
                ).Set(Gf.Vec3d(*center))
                prim.CreateAttribute(
                    "grasph:mass", Sdf.ValueTypeNames.Double
                ).Set(settings.body_mass)
                prim.CreateAttribute(
                    "grasph:inertiaDiagonal", Sdf.ValueTypeNames.Double3
                ).Set(Gf.Vec3d(*settings.body_inertia))
                prim.CreateAttribute(
                    "grasph:linearDamping", Sdf.ValueTypeNames.Double
                ).Set(settings.body_linear_damping)
                prim.CreateAttribute(
                    "grasph:bodyAngularDamping", Sdf.ValueTypeNames.Double
                ).Set(settings.body_angular_damping)
                prim.CreateAttribute(
                    "grasph:collisionRadius", Sdf.ValueTypeNames.Double
                ).Set(settings.body_collision_radius)
                prim.CreateAttribute(
                    "grasph:collisionShape", Sdf.ValueTypeNames.String
                ).Set(settings.body_collision_shape.lower())
                prim.CreateAttribute(
                    "grasph:collisionMargin", Sdf.ValueTypeNames.Double
                ).Set(settings.body_collision_margin)
        elif settings.role == "GRANULAR_SOURCE":
            physical = _physics_values(settings)
            for prim in targets:
                prim.CreateAttribute(
                    "grasph:particleRadius", Sdf.ValueTypeNames.Double
                ).Set(settings.granular_radius)
                prim.CreateAttribute(
                    "grasph:density", Sdf.ValueTypeNames.Double
                ).Set(physical['density'])
                prim.CreateAttribute(
                    "grasph:friction", Sdf.ValueTypeNames.Double
                ).Set(physical['friction'])
                prim.CreateAttribute(
                    "grasph:restitution", Sdf.ValueTypeNames.Double
                ).Set(physical['restitution'])
                prim.CreateAttribute(
                    "grasph:rollingFriction", Sdf.ValueTypeNames.Double
                ).Set(physical['rolling'])
                if settings.physics_material:
                    prim.CreateAttribute('grasph:physicsMaterial', Sdf.ValueTypeNames.String).Set(settings.physics_material.name)
                prim.CreateAttribute(
                    "grasph:sourceMode", Sdf.ValueTypeNames.String
                ).Set(settings.granular_source_mode.lower())
                prim.CreateAttribute(
                    "grasph:flowRate", Sdf.ValueTypeNames.Double
                ).Set(settings.granular_flow_rate)
                prim.CreateAttribute(
                    "grasph:initialVelocity", Sdf.ValueTypeNames.Double3
                ).Set(Gf.Vec3d(*settings.granular_initial_velocity))
                prim.CreateAttribute(
                    "grasph:startTime", Sdf.ValueTypeNames.Double
                ).Set(settings.granular_start_time)
                prim.CreateAttribute(
                    "grasph:endTime", Sdf.ValueTypeNames.Double
                ).Set(settings.granular_end_time)

    stage.GetRootLayer().Save()


def export_scene(context, filepath):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    tagged = _tagged_objects(context.scene)
    export_objects = set(tagged)
    for obj in tagged:
        parent = obj.parent
        while parent is not None:
            export_objects.add(parent)
            parent = parent.parent

    selected_before = set(context.selected_objects)
    active_before = context.view_layer.objects.active
    try:
        bpy.ops.object.select_all(action="DESELECT")
        for obj in export_objects:
            obj.select_set(True)
        if tagged:
            context.view_layer.objects.active = tagged[0]
        bpy.ops.wm.usd_export(
            filepath=str(filepath),
            selected_objects_only=True,
            export_animation=True,
            export_cameras=False,
            export_lights=False,
        )
    finally:
        bpy.ops.object.select_all(action="DESELECT")
        for obj in selected_before:
            if obj.name in context.scene.objects:
                obj.select_set(True)
        if active_before is not None and active_before.name in context.scene.objects:
            context.view_layer.objects.active = active_before
    author_grasph_metadata(filepath, tagged, context.scene)


class GraSPHPhysicsMaterial(PropertyGroup):
    enabled: BoolProperty(name='GraSPH Physics Material', default=False)
    density: FloatProperty(name='Grain Density (kg/m3)', default=1600., min=.001)
    friction: FloatProperty(name='Friction', default=.45, min=0., max=2.)
    restitution: FloatProperty(name='Restitution', default=.1, min=0., max=1.)
    rolling_friction: FloatProperty(name='Rolling Resistance', default=.02, min=0., max=1.)


class GraSPHObjectSettings(PropertyGroup):
    physics_material: PointerProperty(name='Physics Material', type=bpy.types.Material,
        description='Shared physical properties; drop a GraSPH physics Material ID here, not on shader slots',
        poll=lambda self, material: material.grasph_physics.enabled)
    role: EnumProperty(name="Entity", items=ROLE_ITEMS, default="NONE")
    axis: EnumProperty(
        name="World Axis",
        items=(("X", "X", "World X axis"), ("Y", "Y", "World Y axis"),
               ("Z", "Z", "World Z axis")),
        default="X",
    )
    center_source: EnumProperty(
        name="Axle Center",
        items=(("ORIGIN", "Object Origin", "Use the object's world-space origin"),
               ("MANUAL", "Manual", "Use the center entered below")),
        default="ORIGIN",
    )
    axle_center: FloatVectorProperty(name="Center (m)", size=3, subtype="XYZ")
    moment_of_inertia: FloatProperty(
        name="Moment of Inertia", description="kg m²", default=0.002, min=1.0e-9,
    )
    angular_damping: FloatProperty(
        name="Angular Damping", description="N m s", default=0.002, min=0.0,
    )
    max_angular_speed: FloatProperty(
        name="Max Angular Speed", description="rad/s; zero disables the limit",
        default=40.0, min=0.0,
    )
    joint_type: EnumProperty(
        name="Joint Type", items=JOINT_TYPE_ITEMS, default="FIXED",
    )
    joint_axis: EnumProperty(
        name="Joint Axis",
        items=(("X", "X", "Local X axis"), ("Y", "Y", "Local Y axis"),
               ("Z", "Z", "Local Z axis")),
        default="Z",
    )
    joint_body1: PointerProperty(
        name="Body B", type=bpy.types.Object,
        description="Dynamic body constrained by this joint",
        poll=lambda self, obj: obj.grasph.role == "DYNAMIC_BODY",
    )
    joint_body0: PointerProperty(
        name='Body A', type=bpy.types.Object,
        description='Other Dynamic Body; leave empty to attach Body B to the world',
        poll=lambda self, obj: obj.grasph.role == 'DYNAMIC_BODY',
    )
    joint_breakable: BoolProperty(name='Breakable', default=False,
        description='Off means infinite strength; currently supported for Fixed joints')
    joint_break_force: FloatProperty(name='Break Force (N)', default=500., min=.001)
    joint_break_torque: FloatProperty(name='Break Torque (Nm)', default=50., min=.001)
    collider_friction: FloatProperty(
        name="Surface Friction", description="Collider Coulomb friction coefficient",
        default=0.45, min=0.0, max=2.0,
    )
    collider_restitution: FloatProperty(
        name="Surface Restitution", description="Collider bounce coefficient",
        default=0.10, min=0.0, max=1.0,
    )
    collider_rolling_friction: FloatProperty(
        name="Rolling Resistance", description="Resistance to grain rolling",
        default=0.02, min=0.0, max=1.0,
    )
    body_mass: FloatProperty(
        name="Mass", description="Dynamic body mass in kg",
        default=1.0, min=1.0e-9,
    )
    body_inertia: FloatVectorProperty(
        name="Inertia Diagonal", description="Local principal inertia in kg m2",
        size=3, subtype="XYZ", default=(0.1, 0.1, 0.1), min=1.0e-9,
    )
    body_linear_damping: FloatProperty(
        name="Linear Damping", default=0.05, min=0.0,
    )
    body_angular_damping: FloatProperty(
        name="Angular Damping", default=0.05, min=0.0,
    )
    body_collision_radius: FloatProperty(
        name="Collision Radius", description="Spherical collision envelope",
        default=0.1, min=1.0e-6, subtype="DISTANCE",
    )
    body_collision_shape: EnumProperty(
        name="Collision Shape",
        items=(("CONVEX", "Convex Mesh", "Use the authored convex mesh"),
               ("SPHERE", "Sphere", "Use the fast spherical envelope")),
        default="CONVEX",
    )
    body_collision_margin: FloatProperty(
        name="Collision Margin", description="Convex contact skin in metres",
        default=0.002, min=1.0e-6, subtype="DISTANCE",
    )
    granular_radius: FloatProperty(
        name="Particle Radius", description="Radius of each spherical grain in metres",
        default=0.025, min=1.0e-6, subtype="DISTANCE",
    )
    granular_material_preset: EnumProperty(
        name="Material Preset",
        items=(("SAND", "Sand", "Dense, dissipative sand"),
               ("GRAVEL", "Gravel", "Rough grains with rolling resistance"),
               ("GRAIN", "Grain", "Light agricultural grain"),
               ("RUBBER", "Rubber Pellets", "High friction and high bounce"),
               ("STEEL", "Steel Shot", "Dense, smooth metal spheres")),
        default="SAND",
    )
    granular_source_mode: EnumProperty(
        name="Source Type",
        items=(("VOLUME", "Granular Volume", "Create an initial packed volume"),
               ("FLOW", "Granular Flow Source", "Continuously introduce grains")),
        default="VOLUME",
    )
    granular_density: FloatProperty(
        name="Material Density", description="Grain material density in kg/m3",
        default=1600.0, min=1.0e-6,
    )
    granular_friction: FloatProperty(
        name="Friction", description="Coulomb friction coefficient",
        default=0.45, min=0.0, max=2.0,
    )
    granular_restitution: FloatProperty(
        name="Restitution", description="Normal collision restitution",
        default=0.10, min=0.0, max=1.0,
    )
    granular_rolling_friction: FloatProperty(
        name="Rolling Resistance", description="Grain rolling resistance coefficient",
        default=0.02, min=0.0, max=1.0,
    )
    granular_flow_rate: FloatProperty(
        name="Flow Rate", description="Requested grains per second",
        default=100.0, min=0.001,
    )
    granular_initial_velocity: FloatVectorProperty(
        name="Initial Velocity", description="World-space source velocity in m/s",
        size=3, subtype="XYZ", unit="VELOCITY", default=(0.0, 0.0, -1.0),
    )
    granular_start_time: FloatProperty(
        name="Start Time", default=0.0, min=0.0, unit="TIME",
    )
    granular_end_time: FloatProperty(
        name="End Time", default=1.0, min=0.0, unit="TIME",
    )


class GraSPHSceneSettings(PropertyGroup):
    physics_material_index: IntProperty(default=0, min=0)
    physics_material_preset: EnumProperty(name='Example Preset',
        items=tuple((key, key.title(), 'Illustrative simulation values, not calibrated material data')
                    for key in GRANULAR_MATERIAL_PRESETS), default='SAND')
    project_root: StringProperty(name="GraSPH Project", subtype="DIR_PATH", default=_default_project_root())
    usd_file: StringProperty(name="Scene USDC", subtype="FILE_PATH", default="//grasph_scene.usdc")
    python_command: StringProperty(
        name="Python", default="py" if os.name == "nt" else "python3",
        description="External Python command containing the GraSPH dependencies",
    )
    use_managed_environment: BoolProperty(
        name="Managed Solver Environment", default=bool(__package__ and __package__.endswith(".blender")),
        description="Use GraSPH's isolated solver venv, outside Blender and the model directory",
    )
    bootstrap_python: StringProperty(
        name="Setup Python", default="py -3.13" if os.name == "nt" else "python3.13",
        description="Installed 64-bit CPython 3.13 used to create the managed venv",
    )
    environment_status: StringProperty(name="Installation Status", default="Not checked", options={"SKIP_SAVE"})
    environment_log: StringProperty(name="Installation Log", subtype="FILE_PATH", options={"SKIP_SAVE"})
    sim_time: FloatProperty(name="Time (s)", default=1.0, min=1.0e-6)
    restart_npz: StringProperty(
        name="Continue From", subtype="FILE_PATH", default="",
        description="Saved fluid result (.npz); choose a saved frame to continue from",
    )
    continue_from_result: BoolProperty(
        name="Continue From Saved Fluid", default=False,
        description="Resume a fluid-only NPZ result instead of starting a fresh simulation",
    )
    restart_time: FloatProperty(
        name="Resume At (s)", default=0.0, min=0.0, precision=3,
        description="Scene time of the saved frame to continue from",
    )
    dp: FloatProperty(name="Particle Spacing (m)", default=0.05, min=1.0e-6)
    c0: FloatProperty(name="Sound Speed (m/s)", default=20.0, min=1.0e-6)
    enable_diagnostics: BoolProperty(name="Sample Force Diagnostics", default=False,
        description="Log selected particle forces without changing physics; adds diagnostic overhead")
    diagnostics_start: FloatProperty(name="Diagnostics From (s)", default=0.5, min=0.)
    diagnostics_end: FloatProperty(name="Diagnostics To (s)", default=1.5, min=0.)
    diagnostics_interval: FloatProperty(name="Diagnostic Interval (s)", default=0.01, min=0.000001, precision=4)
    diagnostics_ids: StringProperty(name="Track Particle IDs", default="",
        description="Optional comma-separated persistent IDs; fastest particles are also logged")
    output_fps: FloatProperty(name="Output FPS", default=30.0, min=0.1)
    output_dir: StringProperty(name="Output Folder", subtype="DIR_PATH", default="//output")
    output_name: StringProperty(name="Result Name", default="grasph_result")
    new_joint_type: EnumProperty(
        name="Joint Type", items=JOINT_TYPE_ITEMS, default="FIXED",
    )
    new_joint_breakable: BoolProperty(name='Breakable', default=False,
        description='Shared settings for newly created Fixed joints; off means infinite strength')
    new_joint_break_force: FloatProperty(name='Break Force (N)', default=500., min=.001)
    new_joint_break_torque: FloatProperty(name='Break Torque (Nm)', default=50., min=.001)
    live_preview: BoolProperty(
        name="Live Viewport Preview", default=True,
        description="Stream fluid/grain snapshots into Blender while the external solver runs",
    )
    live_preview_interval: bpy.props.IntProperty(
        name="Preview Every (%)", default=5, min=1, max=100,
        description="Publish a viewport update at each percentage of simulated time",
    )
    live_grain_radius: FloatProperty(
        name="Preview Grain Radius", default=0.005, min=1.0e-5,
        description="Visual grain radius for the live viewport preview",
    )
    write_usd_checkpoints: BoolProperty(
        name="Write Partial USD Checkpoints", default=False,
        description=(
            "Write an atomically replaced partial USDC during the run; adds disk I/O"
        ),
    )
    usd_checkpoint_interval: bpy.props.IntProperty(
        name="Checkpoint Every (%)", default=10, min=1, max=100,
        description="Update the partial USDC after this much simulated time",
    )
    live_preview_progress: FloatProperty(
        name="Preview Progress", default=0.0, min=0.0, max=100.0,
        options={"SKIP_SAVE"},
    )
    live_preview_time: FloatProperty(
        name="Preview Time", default=0.0, min=0.0, options={"SKIP_SAVE"},
    )
    collider_display: EnumProperty(
        name="Collider Display", items=_COLLIDER_DISPLAY_ITEMS, default="OFF",
        update=_set_collider_outlines,
        description="Visual-only collider overlays; simulation geometry is unchanged",
    )
    result_usd: StringProperty(name="Last Result", subtype="FILE_PATH")
    log_file: StringProperty(name="Last Log", subtype="FILE_PATH")
    last_input_usd: StringProperty(
        name="Last Input USD", subtype="FILE_PATH", options={"SKIP_SAVE"},
    )
    auto_export: BoolProperty(name="Export Before Run", default=True)
    author_usd_physics: BoolProperty(
        name="Author USD Physics",
        default=True,
        description=(
            "Add portable PhysicsScene, collider, rigid-body, mass and joint schemas "
            "alongside GraSPH metadata"
        ),
    )
    free_surface: EnumProperty(
        name="Free Surface",
        items=(("anchored_band", "Anchored Band", "Atmospheric free-surface model"),
               ("none", "None", "Disable the free-surface correction")),
        default="anchored_band",
    )
    enable_collision: BoolProperty(name="Collision", default=True)
    enable_wall_pressure: BoolProperty(name="Wall Pressure", default=True)
    enable_wall_continuity: BoolProperty(name="Wall Continuity", default=True)
    enable_solid_coupling: BoolProperty(
        name="Fluid / Solid Coupling", default=False,
        description="Enable two-way SPH force and reaction coupling for Dynamic Body objects",
    )
    neighbor_backend: EnumProperty(
        name="Neighbor Backend",
        items=(("auto", "Auto", "Use SciPy cKDTree when available"),
               ("ckdtree", "SciPy cKDTree", "Require the accelerated backend"),
               ("grid", "Python Grid", "Use the fallback spatial grid")),
        default="auto",
    )
    threads: bpy.props.IntProperty(
        name="Neighbor Threads", default=0, min=0,
        description="Zero uses all logical CPU cores",
    )
    droplet_mode: EnumProperty(
        name="Droplet Mode",
        items=(("components", "Components", "Detect disconnected fluid components"),
               ("zero", "Low Neighbor Count", "Use the legacy neighbor cutoff")),
        default="components",
    )
    droplet_neighbors: bpy.props.IntProperty(
        name="Droplet Neighbors", default=0, min=0,
        description="Legacy cutoff used by Low Neighbor Count mode",
    )
    droplet_drag_cd: FloatProperty(
        name="Droplet Air Drag (Cd)", default=0.47, min=0.0, max=2.0,
        precision=2,
        description=(
            "Quadratic air drag for detached fluid only; equivalent-sphere model. "
            "Set to 0 to disable"
        ),
    )
    wall_search: EnumProperty(
        name="Wall Search",
        items=(("cached", "Cached", "Reuse translated-collider search trees"),
               ("rebuild", "Rebuild", "Rebuild wall search from current points")),
        default="cached",
    )
    granular_dt: FloatProperty(
        name="Granular Time Step", default=0.001, min=1.0e-6,
        precision=4, description="Maximum DEM/PBD timestep in seconds",
    )
    granular_iterations: bpy.props.IntProperty(
        name="Granular Iterations", default=4, min=1, max=32,
        description="Sphere contact iterations per timestep",
    )
    show_advanced: BoolProperty(name="Advanced Settings", default=False)
    create_surface: BoolProperty(
        name="Create Splash Surface", default=True,
        description="Create the velocity-driven Geometry Nodes water surface after import",
    )
    run_progress: FloatProperty(
        name="Progress", default=0.0, min=0.0, max=100.0,
        options={"SKIP_SAVE"},
    )
    run_status: StringProperty(
        name="Run Status", default="Idle", options={"SKIP_SAVE"},
    )
    pause_file: StringProperty(
        name="Pause Control", default="", options={"HIDDEN", "SKIP_SAVE"},
    )
    run_message: StringProperty(
        name="Solver Message", default="", options={"SKIP_SAVE"},
    )
    run_elapsed: StringProperty(
        name="Elapsed", default="00:00", options={"SKIP_SAVE"},
    )
    run_pid: bpy.props.IntProperty(
        name="PID", default=0, min=0, options={"SKIP_SAVE"},
    )


class GRASPH_OT_assign_selected(Operator):
    bl_idname = "grasph.assign_selected"
    bl_label = "Apply Role to Selected"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        active = context.active_object
        if active is None:
            self.report({"ERROR"}, "No active object")
            return {"CANCELLED"}
        role = active.grasph.role
        for obj in context.selected_objects:
            obj.grasph.role = role
        self.report({"INFO"}, f"Assigned {role} to {len(context.selected_objects)} object(s)")
        return {"FINISHED"}


class GRASPH_OT_quality_preset(Operator):
    bl_idname = "grasph.quality_preset"
    bl_label = "Apply Quality Preset"
    bl_options = {"REGISTER", "UNDO"}

    preset: EnumProperty(
        items=(("DRAFT", "Draft", "Fast scene check"),
               ("BALANCED", "Balanced", "General-purpose settings"),
               ("QUALITY", "Quality", "Finer, slower simulation")),
    )

    def execute(self, context):
        values = QUALITY_PRESETS[self.preset]
        settings = context.scene.grasph_scene
        for name, value in values.items():
            setattr(settings, name, value)
        self.report({"INFO"}, f"Applied {self.preset.title()} preset")
        return {"FINISHED"}


class GRASPH_OT_resume_current_frame(Operator):
    bl_idname = "grasph.resume_current_frame"
    bl_label = "Use Current Frame"
    bl_description = "Set the resume time from the current Blender timeline frame"

    def execute(self, context):
        settings = context.scene.grasph_scene
        fps = max(float(settings.output_fps), 1.0e-6)
        settings.restart_time = max(
            0.0, (context.scene.frame_current - context.scene.frame_start) / fps
        )
        return {"FINISHED"}


class GRASPH_OT_add_joint_anchor(Operator):
    bl_idname = "grasph.add_joint_anchor"
    bl_label = "Add Joint Anchor"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        body = context.active_object
        if body is not None and body.grasph.role != "DYNAMIC_BODY":
            body = None
        others = [obj for obj in context.selected_objects if obj != body and obj.grasph.role == 'DYNAMIC_BODY']
        bpy.ops.object.empty_add(
            type="ARROWS", align="WORLD", location=context.scene.cursor.location
        )
        anchor = context.active_object
        anchor.name = "GraSPH_Joint"
        anchor.empty_display_size = 0.35
        anchor.show_in_front = True
        anchor.color = (0.20, 0.95, 0.45, 1.0)
        anchor.grasph.role = "JOINT"
        anchor.grasph.joint_type = context.scene.grasph_scene.new_joint_type
        anchor.grasph.joint_body1 = body
        if len(others) == 1 and anchor.grasph.joint_type == 'FIXED':
            anchor.grasph.joint_body0 = others[0]
        if anchor.grasph.joint_type == 'FIXED':
            source = context.scene.grasph_scene
            anchor.grasph.joint_breakable = source.new_joint_breakable
            anchor.grasph.joint_break_force = source.new_joint_break_force
            anchor.grasph.joint_break_torque = source.new_joint_break_torque
        self.report(
            {"INFO"},
            "Joint anchor added at the 3D Cursor; move/rotate it to place the joint",
        )
        return {"FINISHED"}


class GRASPH_OT_connect_selected(Operator):
    bl_idname = 'grasph.connect_selected'
    bl_label = 'Connect Selected to Active'
    bl_description = 'Create one Fixed Joint from each selected Dynamic Body to the active body; existing pairs are skipped'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        active = context.active_object
        if context.mode != 'OBJECT' or active is None or active.type != 'MESH':
            return False
        if active.grasph.role != 'DYNAMIC_BODY' or active not in context.selected_objects:
            return False
        if context.scene.grasph_scene.new_joint_type != 'FIXED':
            return False
        if _RUNNING is not None and _RUNNING.poll() is None:
            return False
        return sum(obj.type == 'MESH' and obj.grasph.role == 'DYNAMIC_BODY'
                   for obj in context.selected_objects) >= 2

    def execute(self, context):
        if not self.poll(context):
            self.report({'ERROR'}, 'Select at least two Dynamic Bodies, make one active and choose Fixed')
            return {'CANCELLED'}
        from mathutils import Vector
        active = context.active_object
        targets = sorted((obj for obj in context.selected_objects if obj != active
                          and obj.type == 'MESH' and obj.grasph.role == 'DYNAMIC_BODY'),
                         key=lambda obj: obj.name)
        existing = {
            frozenset((obj.grasph.joint_body0, obj.grasph.joint_body1))
            for obj in context.scene.objects
            if obj.grasph.role == 'JOINT' and obj.grasph.joint_type == 'FIXED'
            and obj.grasph.joint_body0 and obj.grasph.joint_body1
        }
        missing = [obj for obj in targets if frozenset((obj, active)) not in existing]
        if not missing:
            self.report({'INFO'}, 'All selected body pairs already have Fixed joints; nothing changed')
            return {'FINISHED'}
        collection = next((c for c in context.scene.collection.children
                           if c.get('grasph:jointCollection')), None)
        if collection is None:
            collection = bpy.data.collections.new('GraSPH Joints')
            collection['grasph:jointCollection'] = True
            context.scene.collection.children.link(collection)
        source = context.scene.grasph_scene
        active_center = Vector(_object_center(active, active.grasph))
        for obj in missing:
            anchor = bpy.data.objects.new(f'GraSPH_Joint_{obj.name}_to_{active.name}', None)
            collection.objects.link(anchor)
            anchor.empty_display_type = 'ARROWS'
            anchor.empty_display_size = .25
            anchor.show_in_front = True
            anchor.location = .5*(active_center+Vector(_object_center(obj, obj.grasph)))
            anchor.grasph.role = 'JOINT'
            anchor.grasph.joint_type = 'FIXED'
            anchor.grasph.joint_body0 = obj
            anchor.grasph.joint_body1 = active
            anchor.grasph.joint_breakable = source.new_joint_breakable
            anchor.grasph.joint_break_force = source.new_joint_break_force
            anchor.grasph.joint_break_torque = source.new_joint_break_torque
        self.report({'INFO'}, f'Created {len(missing)} Fixed joints to {active.name}; skipped {len(targets)-len(missing)} existing pairs')
        return {'FINISHED'}


class GRASPH_OT_granular_material_preset(Operator):
    bl_idname = "grasph.granular_material_preset"
    bl_label = "Apply Material Preset"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        active = context.active_object
        if active is None or active.grasph.role != "GRANULAR_SOURCE":
            self.report({"ERROR"}, "Select a Granular Source")
            return {"CANCELLED"}
        targets = [obj for obj in context.selected_objects
                   if obj.grasph.role == "GRANULAR_SOURCE"] or [active]
        preset = active.grasph.granular_material_preset
        density, friction, restitution, rolling = GRANULAR_MATERIAL_PRESETS[preset]
        for obj in targets:
            obj.grasph.granular_material_preset = preset
            obj.grasph.physics_material = None
            obj.grasph.granular_density = density
            obj.grasph.granular_friction = friction
            obj.grasph.granular_restitution = restitution
            obj.grasph.granular_rolling_friction = rolling
        self.report({"INFO"}, f"Applied {preset.title()} to {len(targets)} source(s)")
        return {"FINISHED"}


class GRASPH_OT_validate(Operator):
    bl_idname = "grasph.validate_scene"
    bl_label = "Validate Scene"

    def execute(self, context):
        errors, warnings = validate_scene(context.scene)
        unassigned = [obj.name for obj in context.scene.objects
                      if obj.type == "MESH" and obj.grasph.role == "NONE"
                      and not obj.get("grasph:collider_outline")]
        if unassigned:
            warnings.append(
                f"{len(unassigned)} unassigned mesh object(s) will be excluded from export."
            )
        for message in errors:
            print("GraSPH ERROR:", message)
        for message in warnings:
            print("GraSPH WARNING:", message)
        if errors:
            self.report({"ERROR"}, errors[0])
            return {"CANCELLED"}
        self.report({"WARNING" if warnings else "INFO"},
                    f"Scene valid; {len(warnings)} warning(s)")
        return {"FINISHED"}


class GRASPH_OT_export(Operator):
    bl_idname = "grasph.export_usd"
    bl_label = "Export GraSPH USDC"

    def execute(self, context):
        errors, warnings = validate_scene(context.scene)
        if errors:
            self.report({"ERROR"}, errors[0])
            return {"CANCELLED"}
        filepath = _path(context.scene.grasph_scene.usd_file)
        if filepath.suffix.lower() not in {".usd", ".usda", ".usdc"}:
            filepath = filepath.with_suffix(".usdc")
            context.scene.grasph_scene.usd_file = str(filepath)
        try:
            export_scene(context, filepath)
        except Exception as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        self.report({"INFO"}, f"Exported {filepath.name}; {len(warnings)} warning(s)")
        return {"FINISHED"}


class GRASPH_OT_solver_environment(Operator):
    bl_idname = "grasph.solver_environment"
    bl_label = "Solver Environment"
    action: EnumProperty(items=(("INSTALL", "Install", "Create/install managed solver dependencies"),
                                ("CHECK", "Check", "Check selected solver environment")))

    @classmethod
    def poll(cls, context):
        return _RUNNING is None and _ENV_JOB is None

    def invoke(self, context, event):
        if self.action == "INSTALL":
            return context.window_manager.invoke_props_dialog(self, width=460)
        return self.execute(context)

    def draw(self, context):
        self.layout.label(text="Create/update GraSPH's isolated solver environment?")
        self.layout.label(text="Downloads NumPy, SciPy and USD wheels from PyPI.")
        self.layout.label(text="Needs internet and installed 64-bit CPython 3.13.")
        self.layout.label(text="Blender's Python is not modified.")

    def execute(self, context):
        global _ENV_JOB
        settings = context.scene.grasph_scene
        log_handle = None
        try:
            root = _ensure_project_root(settings)
            if root is None:
                raise RuntimeError("GraSPH solver files could not be found")
            environment = _environment_module(context.scene)
            path = environment.environment_path(root, environment.user_data_path())
            if self.action == "INSTALL":
                if not settings.use_managed_environment:
                    raise RuntimeError("Enable Managed Solver Environment before installing")
                python = environment.split_python_command(settings.bootstrap_python)
            elif settings.use_managed_environment:
                if not environment.python_path(path).is_file():
                    raise RuntimeError("Install the managed solver environment first")
                python = [str(environment.python_path(path))]
            else:
                python = environment.split_python_command(settings.python_command)
            command = python + [str(root / "blender" / "grasph_environment.py"),
                                self.action.lower(), "--root", str(root)]
            if settings.use_managed_environment:
                command += ["--environment", str(path)]
            logs = environment.user_data_path() / "logs"
            logs.mkdir(parents=True, exist_ok=True)
            log = logs / f"setup-{time.time_ns()}-{os.getpid()}.log"
            log_handle = log.open("w", encoding="utf-8")
            process = subprocess.Popen(
                command, cwd=str(root), stdout=log_handle, stderr=subprocess.STDOUT,
                env=environment.child_environment(),
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            _ENV_JOB = {"process": process, "log_handle": log_handle, "scene": context.scene.name}
            settings.environment_log = str(log)
            settings.environment_status = "Installing..." if self.action == "INSTALL" else "Checking..."
            bpy.app.timers.register(_poll_environment_job, first_interval=1.0)
        except (OSError, RuntimeError, ValueError) as error:
            if log_handle is not None:
                log_handle.close()
            settings.environment_status = str(error)
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        return {"FINISHED"}


class GRASPH_OT_open_environment_log(Operator):
    bl_idname = "grasph.open_environment_log"
    bl_label = "Open Installation Log"

    def execute(self, context):
        path = context.scene.grasph_scene.environment_log
        if not path or not Path(path).is_file():
            return {"CANCELLED"}
        bpy.ops.wm.path_open(filepath=path)
        return {"FINISHED"}


class GRASPH_OT_run(Operator):
    bl_idname = "grasph.run_simulation"
    bl_label = "Run Simulation"

    _timer = None
    _log_handle = None
    _started_at = 0.0
    _last_log_size = 0
    _last_preview_mtime = 0.0
    _run_final_time = 0.0

    def execute(self, context):
        global _RUNNING
        if _RUNNING is not None or _ENV_JOB is not None:
            self.report({"ERROR"}, "Wait for the current simulation to finish or reset")
            return {"CANCELLED"}
        settings = context.scene.grasph_scene
        root = _ensure_project_root(settings)
        if root is None:
            self.report(
                {"ERROR"},
                "GraSPH project folder was not detected. Choose the folder containing grasph.py.",
            )
            return {"CANCELLED"}
        try:
            solver_python = _solver_python(settings, context.scene)
        except (OSError, RuntimeError, ValueError) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        solver = root / "grasph.py"
        output_dir = _path(settings.output_dir)
        if not solver.is_file():
            self.report({"ERROR"}, f"grasph.py not found under {root}")
            return {"CANCELLED"}
        output_dir.mkdir(parents=True, exist_ok=True)
        restart_npz = (
            _path(settings.restart_npz)
            if settings.continue_from_result and settings.restart_npz else None
        )
        if settings.continue_from_result and restart_npz is None:
            self.report({"ERROR"}, "Choose a saved fluid result (.npz) to continue from")
            return {"CANCELLED"}
        if restart_npz and not restart_npz.is_file():
            self.report({"ERROR"}, f"Restart result not found: {restart_npz}")
            return {"CANCELLED"}
        result_stem = settings.output_name
        if restart_npz and restart_npz.resolve() == (output_dir / (result_stem + ".npz")).resolve():
            base_stem = result_stem + "_continued"
            result_stem = base_stem
            suffix = 2
            while (output_dir / (result_stem + ".npz")).exists():
                result_stem = f"{base_stem}_{suffix}"
                suffix += 1
            settings.output_name = result_stem
            self.report({"INFO"}, f"Saving continuation as {result_stem} to preserve the source result")
        self._run_final_time = (settings.restart_time if restart_npz else 0.0) + settings.sim_time
        if settings.auto_export:
            errors, warnings = validate_scene(context.scene)
            if errors:
                self.report({"ERROR"}, errors[0])
                return {"CANCELLED"}
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", settings.output_name).strip("._")
            safe_name = safe_name or "grasph_result"
            milliseconds = int(time.time() * 1000.0) % 1000
            stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{milliseconds:03d}"
            source = output_dir / "inputs" / f"{safe_name}_scene_{stamp}.usdc"
            try:
                export_scene(context, source)
            except Exception as error:
                self.report({"ERROR"}, f"USD export failed: {error}")
                return {"CANCELLED"}
            settings.last_input_usd = str(source)
        else:
            source = _path(settings.usd_file)
            if not source.is_file():
                self.report({"ERROR"}, f"Scene USD not found: {source}")
                return {"CANCELLED"}
            settings.last_input_usd = str(source)
        log_file = output_dir / (settings.output_name + ".log")
        result_usd = output_dir / (settings.output_name + ".usdc")
        checkpoint_usd = output_dir / (settings.output_name + "_checkpoint.usdc")
        preview_file = output_dir / (settings.output_name + "_live_preview.npz")
        pause_file = output_dir / (settings.output_name + ".pause")
        pause_file.unlink(missing_ok=True)
        _prepare_live_preview_for_run(context.scene, preview_file)
        if settings.write_usd_checkpoints:
            checkpoint_usd.unlink(missing_ok=True)
        settings.pause_file = str(pause_file)
        self._last_preview_mtime = 0.0
        settings.log_file = str(log_file)
        settings.result_usd = str(
            checkpoint_usd if settings.write_usd_checkpoints else result_usd
        )
        settings.live_preview_progress = 0.0
        settings.live_preview_time = 0.0
        settings.run_progress = 0.0
        settings.run_status = "Starting"
        settings.run_message = "Launching external solver…"
        settings.run_elapsed = "00:00"
        command = solver_python + [
            str(solver), str(source),
            "--time", str(settings.sim_time),
            "--dp", str(settings.dp),
            "--c0", str(settings.c0),
            "--fps", str(settings.output_fps),
            "--out", str(output_dir),
            "--name", settings.output_name,
            "--free-surface", settings.free_surface,
            "--neighbor-backend", settings.neighbor_backend,
            "--threads", str(settings.threads),
            "--droplet-mode", settings.droplet_mode,
            "--droplet-neighbors", str(settings.droplet_neighbors),
            "--droplet-drag-cd", str(settings.droplet_drag_cd),
            "--wall-search", settings.wall_search,
            "--granular-dt", str(settings.granular_dt),
            "--granular-iterations", str(settings.granular_iterations),
            "--control-file", str(pause_file),
        ]
        if restart_npz:
            command.extend([
                "--restart-npz", str(restart_npz),
                "--restart-time", str(settings.restart_time),
            ])
        if settings.write_usd_checkpoints:
            command.extend([
                "--usd-checkpoint", str(checkpoint_usd),
                "--usd-checkpoint-interval",
                str(settings.usd_checkpoint_interval / 100.0),
            ])
        if settings.live_preview:
            command.extend([
                "--preview-file", str(preview_file),
                "--preview-interval", str(settings.live_preview_interval / 100.0),
            ])
        if not settings.enable_collision:
            command.append("--no-collision")
        if not settings.enable_wall_pressure:
            command.append("--no-wall-pressure")
        if not settings.enable_wall_continuity:
            command.append("--no-wall-continuity")
        if settings.enable_solid_coupling:
            command.append("--solid-coupling")
        if settings.enable_diagnostics:
            command.extend(["--diagnostics", "--diagnostics-start", str(settings.diagnostics_start),
                            "--diagnostics-end", str(settings.diagnostics_end),
                            "--diagnostics-interval", str(settings.diagnostics_interval),
                            "--diagnostics-ids", settings.diagnostics_ids])
        try:
            self._log_handle = open(log_file, "w", encoding="utf-8")
            startup = None
            if os.name == "nt":
                startup = subprocess.STARTUPINFO()
                startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            _RUNNING = subprocess.Popen(
                command, cwd=str(root), stdout=self._log_handle,
                stderr=subprocess.STDOUT, startupinfo=startup,
                env=_environment_module(context.scene).child_environment(),
            )
        except Exception as error:
            if self._log_handle:
                self._log_handle.close()
            self.report({"ERROR"}, f"Could not start solver: {error}")
            return {"CANCELLED"}

        self._started_at = time.monotonic()
        self._last_log_size = 0
        settings.run_pid = _RUNNING.pid
        settings.run_status = "Running"
        self._timer = context.window_manager.event_timer_add(0.5, window=context.window)
        context.window_manager.modal_handler_add(self)
        self.report({"INFO"}, f"Simulation started (PID {_RUNNING.pid})")
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        global _RUNNING, _RESET_REQUESTED
        if event.type == "ESC":
            if _RUNNING and _RUNNING.poll() is None:
                context.scene.grasph_scene.run_status = "Stopping"
                _RUNNING.terminate()
            self._finish(context)
            context.scene.grasph_scene.run_status = "Stopped"
            self.report({"WARNING"}, "Simulation stopped")
            return {"CANCELLED"}
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        settings = context.scene.grasph_scene
        pause_requested = bool(settings.pause_file and Path(settings.pause_file).is_file())
        elapsed = max(0, int(time.monotonic() - self._started_at))
        settings.run_elapsed = f"{elapsed // 60:02d}:{elapsed % 60:02d}"
        progress, message, log_size = _tail_solver_log(settings.log_file)
        if progress is not None:
            settings.run_progress = progress
        if message:
            settings.run_message = message
        if settings.live_preview:
            preview_file = _path(settings.output_dir) / (
                settings.output_name + "_live_preview.npz"
            )
            try:
                modified = preview_file.stat().st_mtime
            except OSError:
                modified = 0.0
            if modified > self._last_preview_mtime:
                updated = _update_live_preview(context.scene, preview_file)
                if updated is not None:
                    self._last_preview_mtime = modified
                    settings.run_message = (
                        f"Live preview: {updated[0] * 100:.0f}% · "
                        f"t={updated[1]:.3f}s"
                    )
        if log_size > self._last_log_size and settings.run_status != "Stopping":
            settings.run_status = "Running · log active"
            self._last_log_size = log_size
        if pause_requested:
            settings.run_status = "Paused"
        _tag_redraw(context)
        if _RUNNING is None or _RUNNING.poll() is None:
            return {"PASS_THROUGH"}
        return_code = _RUNNING.returncode
        was_stopping = settings.run_status == "Stopping"
        was_resetting = _RESET_REQUESTED or settings.run_status == "Resetting"
        self._finish(context)
        if was_resetting:
            _RESET_REQUESTED = False
            _reset_run_state(context.scene)
            return {"CANCELLED"}
        if return_code == 0:
            settings.run_progress = 100.0
            settings.run_status = "Complete"
            result_npz = _path(settings.output_dir) / (settings.output_name + ".npz")
            settings.restart_npz = str(result_npz)
            settings.restart_time = _npz_end_time(result_npz, self._run_final_time)
            settings.result_usd = str(
                _path(settings.output_dir) / (settings.output_name + ".usdc")
            )
            self.report({"INFO"}, "Simulation complete")
            return {"FINISHED"}
        if was_stopping:
            settings.run_status = "Stopped"
            self.report({"WARNING"}, "Simulation stopped")
            return {"CANCELLED"}
        settings.run_status = "Failed"
        self.report({"ERROR"}, f"Simulation failed ({return_code}); inspect the log")
        return {"CANCELLED"}

    def _finish(self, context):
        global _RUNNING
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
        _RUNNING = None


class GRASPH_OT_stop(Operator):
    bl_idname = "grasph.stop_simulation"
    bl_label = "Stop Simulation"

    def execute(self, context):
        global _RUNNING
        if _RUNNING is None or _RUNNING.poll() is not None:
            self.report({"INFO"}, "No GraSPH simulation is running")
            return {"CANCELLED"}
        context.scene.grasph_scene.run_status = "Stopping"
        _RUNNING.terminate()
        self.report({"WARNING"}, "Stopping simulation…")
        return {"FINISHED"}


class GRASPH_OT_toggle_pause(Operator):
    bl_idname = "grasph.toggle_pause"
    bl_label = "Pause Simulation"

    def execute(self, context):
        if _RUNNING is None or _RUNNING.poll() is not None:
            self.report({"INFO"}, "No active GraSPH simulation")
            return {"CANCELLED"}
        settings = context.scene.grasph_scene
        if not settings.pause_file:
            self.report({"ERROR"}, "Simulation has no pause control file")
            return {"CANCELLED"}
        pause_file = Path(settings.pause_file)
        if pause_file.exists():
            pause_file.unlink(missing_ok=True)
            settings.run_status = "Running"
        else:
            pause_file.parent.mkdir(parents=True, exist_ok=True)
            pause_file.touch()
            settings.run_status = "Pausing"
        _tag_redraw(context)
        return {"FINISHED"}


class GRASPH_OT_reset_run(Operator):
    bl_idname = "grasph.reset_simulation"
    bl_label = "Reset Simulation"

    def execute(self, context):
        global _RESET_REQUESTED
        settings = context.scene.grasph_scene
        if _RUNNING is not None:
            _RESET_REQUESTED = True
            settings.run_status = "Resetting"
            if settings.pause_file:
                Path(settings.pause_file).unlink(missing_ok=True)
            if _RUNNING.poll() is None:
                _RUNNING.terminate()
            self.report({"WARNING"}, "Stopping and resetting simulation...")
            return {"FINISHED"}
        _reset_run_state(context.scene)
        self.report({"INFO"}, "Simulation reset; ready to start from the beginning")
        return {"FINISHED"}


class GRASPH_OT_open_output_folder(Operator):
    bl_idname = "grasph.open_output_folder"
    bl_label = "Open Output Folder"

    def execute(self, context):
        folder = _path(context.scene.grasph_scene.output_dir)
        if not folder.is_dir():
            self.report({"ERROR"}, f"Output folder does not exist: {folder}")
            return {"CANCELLED"}
        try:
            bpy.ops.wm.path_open(filepath=str(folder))
        except Exception as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        return {"FINISHED"}


class GRASPH_OT_open_result(Operator):
    bl_idname = "grasph.open_result"
    bl_label = "Import Last Result"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        filepath = _path(context.scene.grasph_scene.result_usd)
        if not filepath.is_file():
            self.report({"ERROR"}, f"Result not found: {filepath}")
            return {"CANCELLED"}
        before = set(context.scene.objects)
        bpy.ops.wm.usd_import(filepath=str(filepath), import_points=True)
        clouds = [obj for obj in context.scene.objects
                  if obj not in before and obj.type == "POINTCLOUD"]
        granular_clouds = [obj for obj in clouds if "Granular" in obj.name]
        fluid_clouds = [obj for obj in clouds if obj not in granular_clouds]
        if granular_clouds:
            sources = [obj for obj in context.scene.objects
                       if hasattr(obj, "grasph") and obj.grasph.role == "GRANULAR_SOURCE"]
            radius = sources[0].grasph.granular_radius if sources else 0.025
            try:
                display_module = _granular_display_module(context.scene)
                for cloud in granular_clouds:
                    display_module.add_granular_display(cloud, radius)
            except Exception as error:
                self.report({"WARNING"}, f"Result imported, but grain display failed: {error}")
        if context.scene.grasph_scene.create_surface and fluid_clouds:
            try:
                surface_module = _surface_module(context.scene)
                for cloud in fluid_clouds:
                    surface_module.add_surface(cloud, context.scene.grasph_scene.dp)
            except Exception as error:
                self.report({"WARNING"}, f"Result imported, but surface creation failed: {error}")
        return {"FINISHED"}


class GRASPH_OT_refresh_surface(Operator):
    bl_idname = "grasph.refresh_surface"
    bl_label = "Create / Reset Splash Surface"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        source = _find_surface_source(context.scene, context.active_object)
        if source is None:
            self.report({"ERROR"}, "No imported GraSPH point cloud found in this scene")
            return {"CANCELLED"}
        try:
            _surface_module(context.scene).add_surface(
                source, context.scene.grasph_scene.dp
            )
        except Exception as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        return {"FINISHED"}


class GRASPH_OT_refresh_granular_display(Operator):
    bl_idname = "grasph.refresh_granular_display"
    bl_label = "Create / Reset Grain Display"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        cloud = _find_granular_result(context.scene, context.active_object)
        if cloud is None:
            self.report({"ERROR"}, "No imported GraSPH granular point cloud found")
            return {"CANCELLED"}
        modifier = _granular_modifier(cloud)
        radius = 0.025
        if modifier is not None:
            sockets = modifier.node_group.get("grasph_sockets", {})
            identifier = sockets.get("Grain Radius")
            if identifier and identifier in modifier:
                radius = float(modifier[identifier])
        else:
            sources = [obj for obj in context.scene.objects
                       if hasattr(obj, "grasph") and obj.grasph.role == "GRANULAR_SOURCE"]
            if sources:
                radius = sources[0].grasph.granular_radius
        try:
            _granular_display_module(context.scene).add_granular_display(cloud, radius)
        except Exception as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        return {"FINISHED"}


class GRASPH_UL_physics_materials(UIList):
    def filter_items(self, context, data, propname):
        return ([self.bitflag_filter_item if m.grasph_physics.enabled else 0
                 for m in getattr(data, propname)], [])

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        layout.prop(item, 'name', text='', emboss=False, icon='MATERIAL')


class GRASPH_OT_physics_material(Operator):
    bl_idname = 'grasph.physics_material'
    bl_label = 'Physics Material'
    bl_options = {'REGISTER', 'UNDO'}
    action: EnumProperty(items=(('ADD', 'Add', ''), ('EXAMPLES', 'Examples', ''),
                               ('ASSIGN', 'Assign to Selected', '')))

    def execute(self, context):
        settings = context.scene.grasph_scene
        if self.action in {'ADD', 'EXAMPLES'}:
            keys = GRANULAR_MATERIAL_PRESETS if self.action == 'EXAMPLES' else [settings.physics_material_preset]
            material = None
            for key in keys:
                existing = bpy.data.materials.get('GraSPH Physics - '+key.title())
                if self.action == 'EXAMPLES' and existing and existing.grasph_physics.enabled:
                    material = existing
                else:
                    material = _new_physics_material(key)
            settings.physics_material_index = list(bpy.data.materials).index(material)
            return {'FINISHED'}
        index = settings.physics_material_index
        if index >= len(bpy.data.materials) or not bpy.data.materials[index].grasph_physics.enabled:
            self.report({'ERROR'}, 'Choose a physics material from the list')
            return {'CANCELLED'}
        material = bpy.data.materials[index]
        targets = [obj for obj in context.selected_objects
                   if obj.grasph.role in _COLLIDER_ROLES | {'GRANULAR_SOURCE'}]
        if not targets:
            self.report({'ERROR'}, 'Select colliders, Dynamic Bodies or Granular Sources')
            return {'CANCELLED'}
        for obj in targets:
            obj.grasph.physics_material = material
        self.report({'INFO'}, f'Assigned {material.name} to {len(targets)} entities (visual materials unchanged)')
        return {'FINISHED'}


class GRASPH_PT_physics_materials(Panel):
    bl_label = 'GraSPH Physics Materials'
    bl_idname = 'GRASPH_PT_physics_materials'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'GraSPH'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        settings = context.scene.grasph_scene
        row = layout.row(align=True)
        row.prop(settings, 'physics_material_preset', text='')
        row.operator('grasph.physics_material', text='Add').action = 'ADD'
        layout.operator('grasph.physics_material', text='Load Example Materials').action = 'EXAMPLES'
        layout.template_list('GRASPH_UL_physics_materials', '', bpy.data, 'materials', settings, 'physics_material_index', rows=5)
        layout.operator('grasph.physics_material', text='Assign to Selected').action = 'ASSIGN'
        index = settings.physics_material_index
        if index < len(bpy.data.materials):
            material = bpy.data.materials[index]
            if material.grasph_physics.enabled:
                layout.label(text='Shared properties — affects all linked entities', icon='LINKED')
                for name in ('density', 'friction', 'restitution', 'rolling_friction'):
                    layout.prop(material.grasph_physics, name)
        layout.label(text='Example values, not calibrated material data', icon='INFO')


class GRASPH_PT_entities(Panel):
    bl_label = "GraSPH Entities"
    bl_idname = "GRASPH_PT_entities"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "GraSPH"

    def draw_header(self, context):
        self.layout.label(text="", icon="OBJECT_DATA")

    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        if obj is None:
            layout.label(text="Select an object", icon="INFO")
            return
        settings = obj.grasph
        layout.label(text=obj.name, icon="OBJECT_DATA")
        layout.prop(settings, "role")
        layout.operator("grasph.assign_selected")
        if settings.role in _COLLIDER_ROLES | {'GRANULAR_SOURCE'}:
            material_box = layout.box()
            material_box.label(text='Physics Material (not shader)', icon='PHYSICS')
            material_box.template_ID(settings, 'physics_material')
            if settings.physics_material:
                material_box.label(text='Shared: edits affect all users', icon='LINKED')
                p = settings.physics_material.grasph_physics
                if settings.role == 'GRANULAR_SOURCE':
                    material_box.prop(p, 'density')
                for field in ('friction', 'restitution', 'rolling_friction'):
                    material_box.prop(p, field)
                if settings.role == 'DYNAMIC_BODY':
                    material_box.label(text='Mass / inertia remain explicit')
        if settings.role == "JOINT":
            box = layout.box()
            box.label(text="Joint")
            box.prop(settings, "joint_type")
            if settings.joint_type == 'FIXED':
                box.prop(settings, 'joint_body0')
            box.prop(settings, "joint_body1")
            if settings.joint_type == 'FIXED':
                box.label(text='Empty Body A = World', icon='INFO')
                box.prop(settings, 'joint_breakable')
                if settings.joint_breakable:
                    box.prop(settings, 'joint_break_force')
                    box.prop(settings, 'joint_break_torque')
                else:
                    box.label(text='Unbreakable (infinite strength)')
            if settings.joint_type in {"REVOLUTE", "PRISMATIC"}:
                box.prop(settings, "joint_axis", expand=True)
            box.label(text="Move/rotate this Empty to place its anchor", icon="ORIENTATION_GIMBAL")
            if settings.joint_type == "FREE":
                box.label(text="Free: this body has no joint constraint", icon="INFO")
        elif settings.role == "DYNAMIC_HINGE":
            box = layout.box()
            box.label(text="Dynamic Axle")
            box.prop(settings, "axis", expand=True)
            box.prop(settings, "center_source")
            if settings.center_source == "MANUAL":
                box.prop(settings, "axle_center")
            else:
                center = obj.matrix_world.translation
                box.label(text=f"Center: {center.x:.3f}, {center.y:.3f}, {center.z:.3f} m")
            box.prop(settings, "moment_of_inertia")
            box.prop(settings, "angular_damping")
            box.prop(settings, "max_angular_speed")
        elif settings.role == "DYNAMIC_BODY":
            box = layout.box()
            box.label(text="Free Rigid Body")
            box.prop(settings, "center_source")
            if settings.center_source == "MANUAL":
                box.prop(settings, "axle_center", text="Center")
            box.prop(settings, "body_mass")
            box.prop(settings, "body_inertia")
            box.prop(settings, "body_collision_shape")
            if settings.body_collision_shape == "SPHERE":
                box.prop(settings, "body_collision_radius")
            else:
                box.prop(settings, "body_collision_margin")
            box.prop(settings, "body_linear_damping")
            box.prop(settings, "body_angular_damping")
        elif settings.role == "GRANULAR_SOURCE":
            box = layout.box()
            box.label(text="Granular Physics")
            box.prop(settings, "granular_source_mode")
            box.prop(settings, "granular_radius")
            manual = box.column()
            manual.enabled = settings.physics_material is None
            preset_row = manual.row(align=True)
            preset_row.prop(settings, "granular_material_preset", text="Material")
            preset_row.operator("grasph.granular_material_preset", text="Apply")
            manual.prop(settings, "granular_density")
            manual.prop(settings, "granular_friction")
            manual.prop(settings, "granular_restitution")
            manual.prop(settings, "granular_rolling_friction")
            if settings.granular_source_mode == "FLOW":
                box.prop(settings, "granular_flow_rate")
                box.prop(settings, "granular_initial_velocity")
                row = box.row(align=True)
                row.prop(settings, "granular_start_time")
                row.prop(settings, "granular_end_time")
        if settings.role in {"STATIC", "KINEMATIC", "DYNAMIC_HINGE", "DYNAMIC_BODY"}:
            box = layout.box()
            box.enabled = settings.physics_material is None
            box.label(text="Surface Material")
            box.prop(settings, "collider_friction")
            box.prop(settings, "collider_restitution")
            box.prop(settings, "collider_rolling_friction")


class GRASPH_PT_simulation(Panel):
    bl_label = "GraSPH Simulation"
    bl_idname = "GRASPH_PT_simulation"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "GraSPH"
    bl_order = 0

    def draw_header(self, context):
        self.layout.label(text="", icon="PHYSICS")

    def draw(self, context):
        layout = self.layout
        settings = context.scene.grasph_scene
        version = ".".join(str(part) for part in ADDON_VERSION)
        row = layout.row()
        row.alignment = "CENTER"
        row.label(text=f"GraSPH Controls v{version}", icon="INFO")
        layout.separator()

        summary = scene_summary(context.scene)
        summary_box = layout.box()
        summary_box.label(text="Scene at a Glance")
        row = summary_box.row(align=True)
        row.label(text=f"Fluid: {summary['fluid_particles']:,}", icon="MOD_FLUID")
        row.label(text=f"Grains: {summary['grain_particles']:,}", icon="PARTICLES")
        row = summary_box.row(align=True)
        row.label(text=f"Colliders: {summary['colliders']}")
        row.label(text=f"Moving bodies: {summary['dynamic']}")
        if summary["mixed"]:
            coupling = "Fluid + grains"
        elif not summary["fluid_sources"] and not summary["granular_sources"]:
            coupling = "Rigid bodies only"
        elif summary["fluid_sources"]:
            coupling = "Fluid"
        else:
            coupling = "Granular"
        summary_box.label(
            text=f"Estimated {summary['workload'].lower()} workload, "
                 f"about {summary['memory_mb']:.1f} MB · {coupling}"
        )

        running = _RUNNING is not None
        physics_box = layout.box()
        physics_box.label(text="Simulation", icon="PHYSICS")
        physics_box.enabled = not running
        physics_box.prop(
            settings, "sim_time",
            text="Additional Time (s)" if settings.continue_from_result else "Time (s)",
        )
        physics_box.prop(settings, "continue_from_result")
        if settings.continue_from_result:
            physics_box.prop(settings, "restart_npz")
            physics_box.prop(settings, "restart_time")
            physics_box.operator("grasph.resume_current_frame", icon="TIME")
            physics_box.label(
                text="Fluid resumes here; new bodies need Fluid / Solid Coupling enabled.",
                icon="INFO",
            )
        physics_box.prop(settings, "dp")
        physics_box.label(
            text="Smaller spacing creates more particles and takes longer.",
            icon="INFO",
        )
        preset_box = physics_box.box()
        preset_box.label(text="Quality Presets")
        row = preset_box.row(align=True)
        for preset, label in (("DRAFT", "Draft"), ("BALANCED", "Balanced"),
                              ("QUALITY", "Quality")):
            operator = row.operator("grasph.quality_preset", text=label)
            operator.preset = preset
        if running:
            box = layout.box()
            box.label(text="Simulation Running", icon="TIME")
            status_text = (
                f"{settings.run_status} · PID {settings.run_pid} · "
                f"{settings.run_elapsed}"
            )
            box.label(text=status_text, icon="TIME")
            if hasattr(box, "progress"):
                box.progress(
                    factor=settings.run_progress / 100.0,
                    type="BAR",
                    text=f"{settings.run_progress:.1f}%",
                )
            else:
                box.label(text=f"Progress: {settings.run_progress:.1f}%")
            if settings.run_message:
                box.label(text=settings.run_message[:90], icon="CONSOLE")
            control_row = box.row(align=True)
            paused = bool(settings.pause_file and Path(settings.pause_file).is_file())
            pause = control_row.operator(
                "grasph.toggle_pause", text="Resume" if paused else "Pause",
                icon="PLAY" if paused else "PAUSE",
            )
            control_row.operator("grasph.stop_simulation", text="Stop", icon="CANCEL")
            box.operator("grasph.reset_simulation", text="Reset to Start", icon="FILE_REFRESH")
        else:
            box = layout.box()
            controls = box.row(align=True)
            controls.operator("grasph.run_simulation", text="Run Simulation", icon="PLAY")
            controls.operator("grasph.reset_simulation", text="Reset", icon="FILE_REFRESH")
            if settings.run_status != "Idle":
                icon = "CHECKMARK" if settings.run_status == "Complete" else "INFO"
                box.label(text=settings.run_status, icon=icon)
        row = layout.row(align=True)
        import_label = (
            "Import Checkpoint" if running and settings.write_usd_checkpoints
            else "Import Last Result"
        )
        row.operator("grasph.open_result", text=import_label, icon="IMPORT")
        row.operator("grasph.open_output_folder", icon="FILE_FOLDER")
        layout.prop(settings, "create_surface")


class GRASPH_PT_installation(Panel):
    bl_label = "Solver Installation"
    bl_idname = "GRASPH_PT_installation"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "GraSPH"
    bl_parent_id = "GRASPH_PT_simulation"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        settings = context.scene.grasph_scene
        layout = self.layout
        controls = layout.column()
        controls.enabled = _RUNNING is None and _ENV_JOB is None
        controls.prop(settings, "use_managed_environment")
        if settings.use_managed_environment:
            controls.prop(settings, "bootstrap_python")
            controls.operator("grasph.solver_environment", text="Install Solver Environment",
                              icon="IMPORT").action = "INSTALL"
            controls.label(text="Requires CPython 3.13 (64-bit) and internet")
        else:
            controls.prop(settings, "python_command")
        controls.operator("grasph.solver_environment", text="Check Installation", icon="CHECKMARK").action = "CHECK"
        layout.label(text=settings.environment_status)
        if settings.environment_log:
            layout.operator("grasph.open_environment_log", icon="TEXT")


class GRASPH_PT_project(Panel):
    bl_label = "Project & Scene"
    bl_idname = "GRASPH_PT_project"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "GraSPH"
    bl_parent_id = "GRASPH_PT_simulation"
    bl_order = 1
    bl_options = {"DEFAULT_CLOSED"}

    def draw_header(self, context):
        self.layout.label(text="", icon="FILE_FOLDER")

    def draw(self, context):
        settings = context.scene.grasph_scene
        _ensure_project_root(settings)
        layout = self.layout
        layout.enabled = _RUNNING is None
        layout.prop(settings, "project_root")
        layout.prop(settings, "auto_export")
        layout.prop(settings, "author_usd_physics")
        if settings.author_usd_physics:
            layout.label(text="USD Physics schemas will be authored", icon="CHECKMARK")
        source_row = layout.row()
        source_row.enabled = not settings.auto_export
        source_row.prop(settings, "usd_file")
        row = layout.row(align=True)
        row.operator("grasph.validate_scene")
        export_row = row.row(align=True)
        export_row.enabled = not settings.auto_export
        export_row.operator("grasph.export_usd")

        joints = layout.box()
        joints.label(text="Joint Placement", icon="EMPTY_AXIS")
        joints.label(text="Creates an anchor at the 3D Cursor; move it to position the joint.")
        row = joints.row(align=True)
        row.prop(settings, "new_joint_type", text="")
        row.operator("grasph.add_joint_anchor", text="Add Joint", icon="ADD")
        fixed = joints.column()
        fixed.enabled = settings.new_joint_type == 'FIXED'
        fixed.prop(settings, 'new_joint_breakable')
        if settings.new_joint_breakable:
            fixed.prop(settings, 'new_joint_break_force')
            fixed.prop(settings, 'new_joint_break_torque')
        fixed.operator('grasph.connect_selected', text='Connect Selected to Active', icon='CONSTRAINT')
        fixed.label(text='Fixed joints at body midpoints; existing pairs skipped')


class GRASPH_PT_output_preview(Panel):
    bl_label = "Output & Preview"
    bl_idname = "GRASPH_PT_output_preview"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "GraSPH"
    bl_parent_id = "GRASPH_PT_simulation"
    bl_order = 2
    bl_options = {"DEFAULT_CLOSED"}

    def draw_header(self, context):
        self.layout.label(text="", icon="EXPORT")

    def draw(self, context):
        settings = context.scene.grasph_scene
        layout = self.layout
        running = _RUNNING is not None
        output = layout.column()
        output.enabled = not running
        output.prop(settings, "output_fps")
        output.prop(settings, "output_dir")
        output.prop(settings, "output_name")
        if not settings.use_managed_environment:
            output.prop(settings, "python_command")

        preview = layout.box()
        preview.label(text="Live Viewport Preview", icon="VIEW_CAMERA")
        live_row = preview.row()
        live_row.enabled = not running
        live_row.prop(settings, "live_preview")
        if settings.live_preview:
            row = preview.row(align=True)
            row.enabled = not running
            row.prop(settings, "live_preview_interval")
            row.label(text="%")
            preview.label(text="Grains use their physical particle radii", icon="INFO")
            if settings.live_preview_time > 0.0:
                preview.label(
                    text=f"t={settings.live_preview_time:.3f}s · "
                         f"{settings.live_preview_progress:.0f}%",
                    icon="TIME",
                )
        checkpoint = preview.column()
        checkpoint.enabled = not running
        checkpoint.prop(settings, "write_usd_checkpoints")
        if settings.write_usd_checkpoints:
            checkpoint.prop(settings, "usd_checkpoint_interval")
        if settings.log_file:
            layout.label(text="Log: " + Path(settings.log_file).name, icon="TEXT")
        if settings.last_input_usd:
            layout.label(text="Input: " + Path(settings.last_input_usd).name, icon="FILE_3D")


class GRASPH_PT_advanced(Panel):
    bl_label = "Advanced Solver Settings"
    bl_idname = "GRASPH_PT_advanced"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "GraSPH"
    bl_parent_id = "GRASPH_PT_simulation"
    bl_order = 3
    bl_options = {"DEFAULT_CLOSED"}

    def draw_header(self, context):
        self.layout.label(text="", icon="PREFERENCES")

    def draw(self, context):
        settings = context.scene.grasph_scene
        layout = self.layout
        layout.enabled = _RUNNING is None
        layout.prop(settings, "c0")
        layout.prop(settings, "free_surface")
        layout.prop(settings, "enable_collision")
        layout.prop(settings, "enable_wall_pressure")
        layout.prop(settings, "enable_wall_continuity")
        layout.prop(settings, "enable_solid_coupling")
        layout.prop(settings, "neighbor_backend")
        layout.prop(settings, "threads")
        layout.prop(settings, "droplet_mode")
        if settings.droplet_mode == "zero":
            layout.prop(settings, "droplet_neighbors")
        layout.prop(settings, "droplet_drag_cd")
        layout.prop(settings, "wall_search")
        layout.prop(settings, "granular_dt")
        layout.prop(settings, "granular_iterations")
        diagnostics = layout.box()
        diagnostics.prop(settings, "enable_diagnostics")
        if settings.enable_diagnostics:
            diagnostics.prop(settings, "diagnostics_start")
            diagnostics.prop(settings, "diagnostics_end")
            diagnostics.prop(settings, "diagnostics_interval")
            diagnostics.prop(settings, "diagnostics_ids")
            diagnostics.label(text="Sampled forces only; simulation physics unchanged")


class GRASPH_PT_validation(Panel):
    bl_label = "Scene Check"
    bl_idname = "GRASPH_PT_validation"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "GraSPH"
    bl_parent_id = "GRASPH_PT_simulation"
    bl_order = 5
    bl_options = {"DEFAULT_CLOSED"}

    def draw_header(self, context):
        self.layout.label(text="", icon="CHECKMARK")

    def draw(self, context):
        layout = self.layout
        errors, warnings = validate_scene(context.scene)
        unassigned = [obj.name for obj in context.scene.objects
                      if obj.type == "MESH" and obj.grasph.role == "NONE"
                      and not obj.get("grasph:collider_outline")]
        if unassigned:
            warnings.append(
                f"{len(unassigned)} unassigned mesh object(s) will be excluded from export."
            )
        if not errors and not warnings:
            layout.label(text="Scene looks ready to run.", icon="CHECKMARK")
            return
        for message in errors:
            layout.label(text=message, icon="ERROR")
        for message in warnings:
            layout.label(text=message, icon="INFO")


class GRASPH_PT_display(Panel):
    bl_label = "GraSPH Fluid Display"
    bl_idname = "GRASPH_PT_display"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "GraSPH"

    def draw_header(self, context):
        self.layout.label(text="", icon="MOD_FLUID")

    def draw(self, context):
        layout = self.layout
        layout.label(text="Visual settings only; simulation is unchanged", icon="INFO")
        layout.operator("grasph.refresh_surface")
        surface = _find_surface_object(context.scene, context.active_object)
        modifier = _surface_modifier(surface)
        if modifier is None:
            layout.label(text="No GraSPH surface in scene", icon="INFO")
            return
        layout.label(text=f"Surface: {surface.name}", icon="GEOMETRY_NODES")
        active_version = modifier.node_group.get("grasph_surface_version", "")
        if active_version:
            layout.label(text=f"Surface v{active_version} (active)", icon="CHECKMARK")
        else:
            layout.label(text="Surface: legacy / unversioned", icon="INFO")
        sockets = modifier.node_group.get("grasph_sockets", {})
        for name, identifier in sockets.items():
            layout.prop(modifier, '["' + identifier + '"]', text=name)


class GRASPH_PT_colliders(Panel):
    bl_label = "GraSPH Colliders"
    bl_idname = "GRASPH_PT_colliders"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "GraSPH"
    bl_parent_id = "GRASPH_PT_simulation"
    bl_order = 4
    bl_options = {"DEFAULT_CLOSED"}

    def draw_header(self, context):
        self.layout.label(text="", icon="MOD_PHYSICS")

    def draw(self, context):
        settings = context.scene.grasph_scene
        self.layout.prop(settings, "collider_display")
        self.layout.label(
            text="Visual overlays only; excluded from simulation/export", icon="INFO"
        )


class GRASPH_PT_granular_display(Panel):
    bl_label = "GraSPH Granular Display"
    bl_idname = "GRASPH_PT_granular_display"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "GraSPH"

    def draw_header(self, context):
        self.layout.label(text="", icon="PARTICLES")

    def draw(self, context):
        layout = self.layout
        layout.label(text="Physical particle radii from result cache", icon="INFO")
        layout.operator("grasph.refresh_granular_display")
        obj = _find_granular_result(context.scene, context.active_object)
        modifier = _granular_modifier(obj)
        if modifier is None:
            layout.label(text="No granular result in scene", icon="INFO")
            return
        layout.label(text=f"Result: {obj.name}", icon="PARTICLES")
        version = modifier.node_group.get("grasph_granular_display_version", "")
        if version:
            layout.label(text=f"Grain display v{version}", icon="CHECKMARK")
        for name, identifier in modifier.node_group.get("grasph_sockets", {}).items():
            layout.prop(modifier, '["' + identifier + '"]', text=name)


CLASSES = (
    GraSPHPhysicsMaterial,
    GRASPH_UL_physics_materials,
    GRASPH_OT_physics_material,
    GRASPH_PT_physics_materials,
    GraSPHObjectSettings,
    GraSPHSceneSettings,
    GRASPH_OT_assign_selected,
    GRASPH_OT_quality_preset,
    GRASPH_OT_resume_current_frame,
    GRASPH_OT_add_joint_anchor,
    GRASPH_OT_connect_selected,
    GRASPH_OT_granular_material_preset,
    GRASPH_OT_validate,
    GRASPH_OT_export,
    GRASPH_OT_solver_environment,
    GRASPH_OT_open_environment_log,
    GRASPH_OT_run,
    GRASPH_OT_stop,
    GRASPH_OT_toggle_pause,
    GRASPH_OT_reset_run,
    GRASPH_OT_open_output_folder,
    GRASPH_OT_open_result,
    GRASPH_OT_refresh_surface,
    GRASPH_OT_refresh_granular_display,
    GRASPH_PT_entities,
    GRASPH_PT_simulation,
    GRASPH_PT_installation,
    GRASPH_PT_project,
    GRASPH_PT_output_preview,
    GRASPH_PT_advanced,
    GRASPH_PT_colliders,
    GRASPH_PT_validation,
    GRASPH_PT_display,
    GRASPH_PT_granular_display,
)


def register():
    # Text Editor users commonly press Run Script again while iterating. Remove
    # classes left by the previous execution before registering this version.
    if hasattr(bpy.types.Scene, "grasph_scene"):
        del bpy.types.Scene.grasph_scene
    if hasattr(bpy.types.Object, "grasph"):
        del bpy.types.Object.grasph
    if hasattr(bpy.types.Material, 'grasph_physics'):
        del bpy.types.Material.grasph_physics
    for cls in reversed(CLASSES):
        existing = getattr(bpy.types, cls.__name__, None)
        target = existing if existing is not None else cls
        try:
            bpy.utils.unregister_class(target)
        except RuntimeError:
            pass
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Object.grasph = PointerProperty(type=GraSPHObjectSettings)
    bpy.types.Scene.grasph_scene = PointerProperty(type=GraSPHSceneSettings)
    bpy.types.Material.grasph_physics = PointerProperty(type=GraSPHPhysicsMaterial)


def unregister():
    global _RUNNING
    if _RUNNING is not None and _RUNNING.poll() is None:
        _RUNNING.terminate()
    _RUNNING = None
    if hasattr(bpy.types.Scene, "grasph_scene"):
        del bpy.types.Scene.grasph_scene
    if hasattr(bpy.types.Object, "grasph"):
        del bpy.types.Object.grasph
    if hasattr(bpy.types.Material, 'grasph_physics'):
        del bpy.types.Material.grasph_physics
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
