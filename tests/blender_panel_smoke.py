"""Run with Blender --background --factory-startup --python this_file."""
import importlib.util
from pathlib import Path

import bpy
from pxr import Usd


root = Path(__file__).resolve().parents[1]
module_path = root / "blender" / "grasph_control.py"
spec = importlib.util.spec_from_file_location("grasph_control", module_path)
addon = importlib.util.module_from_spec(spec)
spec.loader.exec_module(addon)
addon.register()
# Match Blender Text Editor behavior, where __file__ may point at the .blend,
# and verify that repeated Run Script registration remains safe.
addon.__file__ = str(root / "examples" / "GranularExample.blend")
addon.register()
assert Path(addon._default_project_root()) == root
assert addon._path("//output") == root / "output"
bpy.context.scene.grasph_scene.project_root = ""
assert Path(addon._surface_module(bpy.context.scene).__file__).resolve() == (
    root / "blender" / "grasph_surface.py"
).resolve()
assert addon.ADDON_VERSION == (0, 6, 0)

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

# The display panel must find its surface even when another object is active.
cloud_data = bpy.data.pointclouds.new("Auto Find Cloud")
cloud = bpy.data.objects.new("Auto Find Cloud", cloud_data)
bpy.context.collection.objects.link(cloud)
surface_data = bpy.data.meshes.new("Auto Find Surface Host")
surface = bpy.data.objects.new("Auto Find Surface", surface_data)
bpy.context.collection.objects.link(surface)
surface["grasph_source"] = cloud.name
surface_modifier = surface.modifiers.new("GraSPH Surface", "NODES")
surface_modifier.node_group = bpy.data.node_groups.new(
    "Auto Find Surface Group", "GeometryNodeTree"
)
bpy.context.view_layer.objects.active = None
assert addon._find_surface_object(bpy.context.scene) == surface
assert addon._find_surface_source(bpy.context.scene) == cloud

bpy.data.objects.remove(surface, do_unlink=True)
bpy.data.objects.remove(cloud, do_unlink=True)


def cube(name, role, location):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.grasph.role = role
    return obj


fluid = cube("Fluid Test", "FLUID", (0.0, 0.0, 1.0))
tank = cube("Tank Test", "STATIC", (0.0, 0.0, 0.0))
wheel = cube("Wheel Test", "DYNAMIC_HINGE", (0.0, -1.0, 0.5))
wheel.grasph.axis = "X"
wheel.grasph.moment_of_inertia = 0.125
wheel.grasph.collider_friction = 0.33
wheel.grasph.collider_restitution = 0.2
wheel.grasph.collider_rolling_friction = 0.015

errors, warnings = addon.validate_scene(bpy.context.scene)
assert not errors, errors

output = root / "output" / "blender_panel_smoke.usdc"
bpy.context.scene.grasph_scene.usd_file = str(output)
result = bpy.ops.grasph.export_usd()
assert "FINISHED" in result, result

stage = Usd.Stage.Open(str(output))
roles = {}
hinge = None
for prim in stage.Traverse():
    role = prim.GetAttribute("grasph:role")
    if role and role.HasAuthoredValueOpinion():
        roles[str(prim.GetPath())] = role.Get()
    enabled = prim.GetAttribute("grasph:hingeEnabled")
    if enabled and enabled.Get():
        hinge = prim

assert "fluid" in roles.values(), roles
assert "collider" in roles.values(), roles
assert hinge is not None
assert hinge.GetAttribute("grasph:axis").Get() == "X"
assert abs(hinge.GetAttribute("grasph:momentOfInertia").Get() - 0.125) < 1e-12
assert abs(hinge.GetAttribute("grasph:friction").Get() - 0.33) < 1e-6
assert abs(hinge.GetAttribute("grasph:rollingFriction").Get() - 0.015) < 1e-6

# Granular authoring is a separate, valid scene mode.
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
grains = cube("Granular Source Test", "GRANULAR_SOURCE", (0.0, 0.0, 1.0))
grains.grasph.granular_radius = 0.031
grains.grasph.granular_source_mode = "FLOW"
grains.grasph.granular_density = 1450.0
grains.grasph.granular_friction = 0.38
grains.grasph.granular_restitution = 0.12
grains.grasph.granular_rolling_friction = 0.025
grains.grasph.granular_flow_rate = 42.0
grains.grasph.granular_initial_velocity = (0.0, 1.5, -0.25)
grains.grasph.granular_start_time = 0.2
grains.grasph.granular_end_time = 0.8
cube("Granular Tank Test", "STATIC", (0.0, 0.0, 0.0))
dynamic = cube("Dynamic Body Test", "DYNAMIC_BODY", (0.5, 0.0, 0.5))
dynamic.grasph.body_mass = 2.5
dynamic.grasph.body_inertia = (0.2, 0.3, 0.4)
dynamic.grasph.body_collision_radius = 0.15
dynamic.grasph.body_collision_shape = "CONVEX"
dynamic.grasph.body_collision_margin = 0.004
errors, warnings = addon.validate_scene(bpy.context.scene)
assert not errors, errors
granular_output = root / "output" / "blender_granular_smoke.usdc"
addon.export_scene(bpy.context, granular_output)
granular_stage = Usd.Stage.Open(str(granular_output))
granular_prim = next(
    prim for prim in granular_stage.Traverse()
    if prim.GetAttribute("grasph:role").Get() == "granular_source"
)
assert abs(granular_prim.GetAttribute("grasph:particleRadius").Get() - 0.031) < 1e-6
assert abs(granular_prim.GetAttribute("grasph:density").Get() - 1450.0) < 1e-6
assert granular_prim.GetAttribute("grasph:sourceMode").Get() == "flow"
assert abs(granular_prim.GetAttribute("grasph:flowRate").Get() - 42.0) < 1e-6
assert tuple(granular_prim.GetAttribute("grasph:initialVelocity").Get()) == (0.0, 1.5, -0.25)
dynamic_prim = next(
    prim for prim in granular_stage.Traverse()
    if prim.GetAttribute("grasph:dynamicBodyEnabled").Get()
)
assert abs(dynamic_prim.GetAttribute("grasph:mass").Get() - 2.5) < 1e-6
assert abs(dynamic_prim.GetAttribute("grasph:collisionRadius").Get() - 0.15) < 1e-6
assert dynamic_prim.GetAttribute("grasph:collisionShape").Get() == "convex"
assert abs(dynamic_prim.GetAttribute("grasph:collisionMargin").Get() - 0.004) < 1e-6

# The result display instances renderable spheres on an animated point cloud.
cloud_data = bpy.data.pointclouds.new("Granular Result Points")
cloud = bpy.data.objects.new("GraSPH_GranularResult", cloud_data)
bpy.context.collection.objects.link(cloud)
display_module = addon._granular_display_module(bpy.context.scene)
display_module.add_granular_display(cloud, 0.031)
display_modifier = cloud.modifiers.get("GraSPH Granular Display")
assert display_modifier is not None
assert display_modifier.node_group.get("grasph_granular_display_version") == "0.1.0"
print("GraSPH Blender panel smoke test passed:", output)
