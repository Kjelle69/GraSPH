"""GraSPH scene-authoring and simulation controls for Blender.

Install this file as a legacy add-on, or run it in Blender's Text Editor.
The numerical solver is launched as a separate process; Blender remains the
scene editor and result viewer.
"""

ADDON_VERSION = (0, 8, 0)


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
import shlex
import subprocess
import sys
import time

import bpy
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import Operator, Panel, PropertyGroup


ROLE_ITEMS = (
    ("NONE", "Unassigned", "No explicit GraSPH role"),
    ("FLUID", "Fluid Volume", "Initial SPH fluid volume"),
    ("GRANULAR_SOURCE", "Granular Source", "Initial volume of rigid spherical grains"),
    ("SINK", "Sink / Drain", "Remove particles entering this volume"),
    ("STATIC", "Static Collider", "Fixed tank or obstacle"),
    ("KINEMATIC", "Kinematic Collider", "Animated collider that drives the fluid"),
    ("DYNAMIC_HINGE", "Dynamic Hinge", "Collider driven by fluid torque around one axle"),
    ("DYNAMIC_BODY", "Dynamic Body", "Freely translating and rotating rigid collider"),
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
    "IGNORE": "ignore",
}

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


def _discover_project_root():
    """Find the repository when run as an add-on or a Blender text block."""
    starts = []
    try:
        starts.append(Path(__file__).resolve())
    except (NameError, OSError):
        pass
    if bpy.data.filepath:
        starts.append(Path(bpy.data.filepath).resolve())
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
    if not fluid_objects and not granular_objects:
        errors.append("No Fluid Volume or Granular Source is assigned.")
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
            if settings.granular_density <= 0.0:
                errors.append(f"{obj.name}: granular density must be positive.")
            if settings.granular_restitution > 0.5:
                warnings.append(
                    f"{obj.name}: restitution {settings.granular_restitution:g} "
                    "is highly energetic in a dense granular volume."
                )
            if settings.granular_source_mode == "FLOW":
                if settings.granular_flow_rate <= 0.0:
                    errors.append(f"{obj.name}: granular flow rate must be positive.")
                if settings.granular_end_time < settings.granular_start_time:
                    errors.append(f"{obj.name}: flow end time precedes start time.")
        if settings.role == "KINEMATIC" and not obj.animation_data:
            warnings.append(f"{obj.name}: kinematic collider has no object animation.")

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


def author_grasph_metadata(usd_path, objects):
    """Write simulation semantics into the exported USD layer."""
    from pxr import Gf, Sdf, Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Could not reopen exported USD: {usd_path}")

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

        if settings.role in {"STATIC", "KINEMATIC", "DYNAMIC_HINGE", "DYNAMIC_BODY"}:
            for prim in targets:
                prim.CreateAttribute(
                    "grasph:friction", Sdf.ValueTypeNames.Double
                ).Set(settings.collider_friction)
                prim.CreateAttribute(
                    "grasph:restitution", Sdf.ValueTypeNames.Double
                ).Set(settings.collider_restitution)
                prim.CreateAttribute(
                    "grasph:rollingFriction", Sdf.ValueTypeNames.Double
                ).Set(settings.collider_rolling_friction)

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
            for prim in targets:
                prim.CreateAttribute(
                    "grasph:particleRadius", Sdf.ValueTypeNames.Double
                ).Set(settings.granular_radius)
                prim.CreateAttribute(
                    "grasph:density", Sdf.ValueTypeNames.Double
                ).Set(settings.granular_density)
                prim.CreateAttribute(
                    "grasph:friction", Sdf.ValueTypeNames.Double
                ).Set(settings.granular_friction)
                prim.CreateAttribute(
                    "grasph:restitution", Sdf.ValueTypeNames.Double
                ).Set(settings.granular_restitution)
                prim.CreateAttribute(
                    "grasph:rollingFriction", Sdf.ValueTypeNames.Double
                ).Set(settings.granular_rolling_friction)
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
    author_grasph_metadata(filepath, tagged)


class GraSPHObjectSettings(PropertyGroup):
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
    project_root: StringProperty(name="GraSPH Project", subtype="DIR_PATH", default=_default_project_root())
    usd_file: StringProperty(name="Scene USDC", subtype="FILE_PATH", default="//grasph_scene.usdc")
    python_command: StringProperty(
        name="Python", default="py" if os.name == "nt" else "python3",
        description="External Python command containing the GraSPH dependencies",
    )
    sim_time: FloatProperty(name="Time (s)", default=1.0, min=1.0e-6)
    dp: FloatProperty(name="Particle Spacing (m)", default=0.05, min=1.0e-6)
    c0: FloatProperty(name="Sound Speed (m/s)", default=20.0, min=1.0e-6)
    output_fps: FloatProperty(name="Output FPS", default=30.0, min=0.1)
    output_dir: StringProperty(name="Output Folder", subtype="DIR_PATH", default="//output")
    output_name: StringProperty(name="Result Name", default="grasph_result")
    result_usd: StringProperty(name="Last Result", subtype="FILE_PATH")
    log_file: StringProperty(name="Last Log", subtype="FILE_PATH")
    last_input_usd: StringProperty(
        name="Last Input USD", subtype="FILE_PATH", options={"SKIP_SAVE"},
    )
    auto_export: BoolProperty(name="Export Before Run", default=True)
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
                      if obj.type == "MESH" and obj.grasph.role == "NONE"]
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


class GRASPH_OT_run(Operator):
    bl_idname = "grasph.run_simulation"
    bl_label = "Run Simulation"

    _timer = None
    _log_handle = None
    _started_at = 0.0
    _last_log_size = 0

    def execute(self, context):
        global _RUNNING
        if _RUNNING is not None and _RUNNING.poll() is None:
            self.report({"ERROR"}, "A GraSPH simulation is already running")
            return {"CANCELLED"}
        settings = context.scene.grasph_scene
        root = _path(settings.project_root)
        solver = root / "grasph.py"
        output_dir = _path(settings.output_dir)
        if not solver.is_file():
            self.report({"ERROR"}, f"grasph.py not found under {root}")
            return {"CANCELLED"}
        output_dir.mkdir(parents=True, exist_ok=True)
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
        settings.log_file = str(log_file)
        settings.result_usd = str(result_usd)
        settings.run_progress = 0.0
        settings.run_status = "Starting"
        settings.run_message = "Launching external solver…"
        settings.run_elapsed = "00:00"
        command = shlex.split(settings.python_command, posix=os.name != "nt") + [
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
            "--wall-search", settings.wall_search,
            "--granular-dt", str(settings.granular_dt),
            "--granular-iterations", str(settings.granular_iterations),
        ]
        if not settings.enable_collision:
            command.append("--no-collision")
        if not settings.enable_wall_pressure:
            command.append("--no-wall-pressure")
        if not settings.enable_wall_continuity:
            command.append("--no-wall-continuity")
        if settings.enable_solid_coupling:
            command.append("--solid-coupling")
        try:
            self._log_handle = open(log_file, "w", encoding="utf-8")
            startup = None
            if os.name == "nt":
                startup = subprocess.STARTUPINFO()
                startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            _RUNNING = subprocess.Popen(
                command, cwd=str(root), stdout=self._log_handle,
                stderr=subprocess.STDOUT, startupinfo=startup,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
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
        global _RUNNING
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
        elapsed = max(0, int(time.monotonic() - self._started_at))
        settings.run_elapsed = f"{elapsed // 60:02d}:{elapsed % 60:02d}"
        progress, message, log_size = _tail_solver_log(settings.log_file)
        if progress is not None:
            settings.run_progress = progress
        if message:
            settings.run_message = message
        if log_size > self._last_log_size and settings.run_status != "Stopping":
            settings.run_status = "Running · log active"
            self._last_log_size = log_size
        _tag_redraw(context)
        if _RUNNING is None or _RUNNING.poll() is None:
            return {"PASS_THROUGH"}
        return_code = _RUNNING.returncode
        was_stopping = settings.run_status == "Stopping"
        self._finish(context)
        if return_code == 0:
            settings.run_progress = 100.0
            settings.run_status = "Complete"
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


class GRASPH_PT_entities(Panel):
    bl_label = "GraSPH Entities"
    bl_idname = "GRASPH_PT_entities"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "GraSPH"

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
        if settings.role == "DYNAMIC_HINGE":
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
            preset_row = box.row(align=True)
            preset_row.prop(settings, "granular_material_preset", text="Material")
            preset_row.operator("grasph.granular_material_preset", text="Apply")
            box.prop(settings, "granular_density")
            box.prop(settings, "granular_friction")
            box.prop(settings, "granular_restitution")
            box.prop(settings, "granular_rolling_friction")
            if settings.granular_source_mode == "FLOW":
                box.prop(settings, "granular_flow_rate")
                box.prop(settings, "granular_initial_velocity")
                row = box.row(align=True)
                row.prop(settings, "granular_start_time")
                row.prop(settings, "granular_end_time")
        if settings.role in {"STATIC", "KINEMATIC", "DYNAMIC_HINGE", "DYNAMIC_BODY"}:
            box = layout.box()
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
        summary_box.label(text="Scene Summary")
        row = summary_box.row(align=True)
        row.label(text=f"Fluid: {summary['fluid_particles']:,}")
        row.label(text=f"Grains: {summary['grain_particles']:,}")
        row = summary_box.row(align=True)
        row.label(text=f"Colliders: {summary['colliders']}")
        row.label(text=f"Dynamic: {summary['dynamic']}")
        coupling = "Mixed coupling ON" if summary["mixed"] else "Single phase"
        summary_box.label(
            text=f"{summary['workload']} workload · ~{summary['memory_mb']:.1f} MB · {coupling}"
        )
        errors, warnings = validate_scene(context.scene)
        for message in errors[:2]:
            summary_box.label(text=message[:100], icon="ERROR")
        for message in warnings[:3]:
            summary_box.label(text=message[:100], icon="INFO")

        layout.prop(settings, "project_root")
        layout.prop(settings, "auto_export")
        source_row = layout.row()
        source_row.enabled = not settings.auto_export
        source_row.prop(settings, "usd_file")
        row = layout.row(align=True)
        row.operator("grasph.validate_scene")
        export_row = row.row(align=True)
        export_row.enabled = not settings.auto_export
        export_row.operator("grasph.export_usd")
        preset_box = layout.box()
        preset_box.label(text="Quality Presets")
        row = preset_box.row(align=True)
        for preset, label in (("DRAFT", "Draft"), ("BALANCED", "Balanced"),
                              ("QUALITY", "Quality")):
            operator = row.operator("grasph.quality_preset", text=label)
            operator.preset = preset

        box = layout.box()
        box.label(text="Physics Resolution")
        box.prop(settings, "sim_time")
        box.prop(settings, "dp")
        box.label(text="Smaller spacing = more particles and longer runtime", icon="INFO")
        output_box = layout.box()
        output_box.label(text="Output")
        output_box.prop(settings, "output_fps")
        output_box.prop(settings, "output_dir")
        output_box.prop(settings, "output_name")
        output_box.prop(settings, "python_command")
        box.prop(settings, "show_advanced", toggle=True)
        if settings.show_advanced:
            advanced = box.column(align=True)
            advanced.prop(settings, "c0")
            advanced.prop(settings, "free_surface")
            advanced.prop(settings, "enable_collision")
            advanced.prop(settings, "enable_wall_pressure")
            advanced.prop(settings, "enable_wall_continuity")
            advanced.prop(settings, "enable_solid_coupling")
            advanced.prop(settings, "neighbor_backend")
            advanced.prop(settings, "threads")
            advanced.prop(settings, "droplet_mode")
            if settings.droplet_mode == "zero":
                advanced.prop(settings, "droplet_neighbors")
            advanced.prop(settings, "wall_search")
            advanced.prop(settings, "granular_dt")
            advanced.prop(settings, "granular_iterations")
        running = _RUNNING is not None and _RUNNING.poll() is None
        if running:
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
            box.operator("grasph.stop_simulation", icon="CANCEL")
        else:
            box.operator("grasph.run_simulation", icon="PLAY")
            if settings.run_status != "Idle":
                icon = "CHECKMARK" if settings.run_status == "Complete" else "INFO"
                box.label(text=settings.run_status, icon=icon)
        row = layout.row(align=True)
        row.operator("grasph.open_result", icon="IMPORT")
        row.operator("grasph.open_output_folder", icon="FILE_FOLDER")
        layout.prop(settings, "create_surface")
        if settings.log_file:
            layout.label(text="Log: " + Path(settings.log_file).name, icon="TEXT")
        if settings.last_input_usd:
            layout.label(
                text="Input: " + Path(settings.last_input_usd).name,
                icon="FILE_3D",
            )


class GRASPH_PT_display(Panel):
    bl_label = "GraSPH Fluid Display"
    bl_idname = "GRASPH_PT_display"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "GraSPH"

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


class GRASPH_PT_granular_display(Panel):
    bl_label = "GraSPH Granular Display"
    bl_idname = "GRASPH_PT_granular_display"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "GraSPH"

    def draw(self, context):
        layout = self.layout
        layout.label(text="Visual grain size only", icon="INFO")
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
    GraSPHObjectSettings,
    GraSPHSceneSettings,
    GRASPH_OT_assign_selected,
    GRASPH_OT_quality_preset,
    GRASPH_OT_granular_material_preset,
    GRASPH_OT_validate,
    GRASPH_OT_export,
    GRASPH_OT_run,
    GRASPH_OT_stop,
    GRASPH_OT_open_output_folder,
    GRASPH_OT_open_result,
    GRASPH_OT_refresh_surface,
    GRASPH_OT_refresh_granular_display,
    GRASPH_PT_entities,
    GRASPH_PT_simulation,
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


def unregister():
    global _RUNNING
    if _RUNNING is not None and _RUNNING.poll() is None:
        _RUNNING.terminate()
    _RUNNING = None
    if hasattr(bpy.types.Scene, "grasph_scene"):
        del bpy.types.Scene.grasph_scene
    if hasattr(bpy.types.Object, "grasph"):
        del bpy.types.Object.grasph
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
