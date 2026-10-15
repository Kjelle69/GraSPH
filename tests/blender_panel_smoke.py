"""Run with Blender --background --factory-startup --python this_file."""
import importlib.util
import os
from pathlib import Path
import tempfile

import bpy
from pxr import Usd, UsdPhysics, UsdShade


root = Path(__file__).resolve().parents[1]
module_path = root / "blender" / "grasph_control.py"
spec = importlib.util.spec_from_file_location("grasph_control", module_path)
addon = importlib.util.module_from_spec(spec)
spec.loader.exec_module(addon)
addon.register()
# Match Blender Text Editor behavior, where __file__ may point at the .blend,
# and verify that repeated Run Script registration remains safe.
addon.__file__ = str(root / "local_scene.blend")
addon.register()
assert Path(addon._default_project_root()) == root
assert addon._path("//output") == root / "output"
# Text Editor scripts retain their source filepath even when __file__ points
# at Blender or the current working directory is unrelated to GraSPH.
source_text = bpy.data.texts.load(str(module_path))
original_cwd = Path.cwd()
try:
    os.chdir(Path(tempfile.gettempdir()))
    addon.__file__ = r"C:\Program Files\Blender Foundation\Blender 5.0\grasph_control.py"
    assert addon._discover_project_root() == root
finally:
    os.chdir(original_cwd)
    bpy.data.texts.remove(source_text)
addon.__file__ = str(root / "local_scene.blend")
bpy.context.scene.grasph_scene.project_root = ""
assert addon._ensure_project_root(bpy.context.scene.grasph_scene) == root
assert Path(addon._surface_module(bpy.context.scene).__file__).resolve() == (
    root / "blender" / "grasph_surface.py"
).resolve()
assert addon.ADDON_VERSION == (0, 14, 0)
scene_settings = bpy.context.scene.grasph_scene
scene_settings.output_fps = 30.0
bpy.context.scene.frame_start = 1
bpy.context.scene.frame_current = 31
bpy.ops.grasph.resume_current_frame()
assert abs(scene_settings.restart_time - 1.0) < 1e-9
import numpy as np
with tempfile.TemporaryDirectory() as temp_dir:
    result_npz = Path(temp_dir) / "result.npz"
    np.savez(result_npz, times=np.asarray([0.0, 2.5]))
    assert addon._npz_end_time(result_npz, -1.0) == 2.5
label_icons = bpy.types.UILayout.bl_rna.functions["label"].parameters["icon"].enum_items
assert "GEOMETRY_NODES" in label_icons
assert "FILE_FOLDER" in label_icons

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

# USD mesh prim names can differ from their role-tagged Blender Xform parent.
body_mesh = bpy.data.meshes.new("Cube.004")
body = bpy.data.objects.new("Cube.002", body_mesh)
bpy.context.collection.objects.link(body)
body.grasph.role = "DYNAMIC_BODY"
collision_mesh = bpy.data.meshes.new("Cube.005")
mesh_name_collision = bpy.data.objects.new("Cube.004", collision_mesh)
bpy.context.collection.objects.link(mesh_name_collision)
mesh_name_collision.grasph.role = "DYNAMIC_BODY"
assert addon._preview_source_object(
    bpy.context.scene, "/root/Cube_002/Cube_004", "DYNAMIC_BODY"
) == body
assert addon._preview_source_object(
    bpy.context.scene, "/root/Cube_004/Cube_005", "DYNAMIC_BODY"
) == mesh_name_collision
bpy.data.objects.remove(body, do_unlink=True)
bpy.data.meshes.remove(body_mesh)
bpy.data.objects.remove(mesh_name_collision, do_unlink=True)
bpy.data.meshes.remove(collision_mesh)

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
surface_modifier.node_group["grasph_sockets"] = {"Show Surface": "Socket_2"}
bpy.context.view_layer.objects.active = None
assert addon._find_surface_object(bpy.context.scene) == surface
assert addon._find_surface_source(bpy.context.scene) == cloud
surface_modifier.name = "GeometryNodes.001"
assert addon._surface_modifier(surface) == surface_modifier

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

summary = addon.scene_summary(bpy.context.scene)
assert summary["fluid_sources"] == 1
assert summary["colliders"] == 2
assert bpy.ops.grasph.quality_preset(preset="DRAFT") == {"FINISHED"}
assert abs(bpy.context.scene.grasph_scene.dp - 0.05) < 1e-6

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
physics_scenes = [prim for prim in stage.Traverse()
                  if prim.IsA(UsdPhysics.Scene)]
revolute_joints = [prim for prim in stage.Traverse()
                   if prim.IsA(UsdPhysics.RevoluteJoint)]
assert len(physics_scenes) == 1
assert len(revolute_joints) == 1
assert hinge.HasAPI(UsdPhysics.CollisionAPI)
assert hinge.HasAPI(UsdPhysics.MeshCollisionAPI)
hinge_material, _ = UsdShade.MaterialBindingAPI(hinge).ComputeBoundMaterial("physics")
assert hinge_material.GetPrim().HasAPI(UsdPhysics.MaterialAPI)
assert abs(UsdPhysics.MaterialAPI(hinge_material).GetDynamicFrictionAttr().Get() - 0.33) < 1e-6
hinge_owner = hinge.GetParent()
assert hinge_owner.HasAPI(UsdPhysics.RigidBodyAPI)
assert hinge_owner.HasAPI(UsdPhysics.MassAPI)
assert UsdPhysics.RevoluteJoint(revolute_joints[0]).GetBody1Rel().GetTargets() == [
    hinge_owner.GetPath()
]

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
bpy.context.view_layer.objects.active = grains
grains.select_set(True)
grains.grasph.granular_material_preset = "GRAVEL"
assert bpy.ops.grasph.granular_material_preset() == {"FINISHED"}
assert abs(grains.grasph.granular_density - 1800.0) < 1e-6
assert abs(grains.grasph.granular_friction - 0.65) < 1e-6
cube("Granular Tank Test", "STATIC", (0.0, 0.0, 0.0))
cube("Mixed Fluid Test", "FLUID", (-0.5, 0.0, 0.5))
dynamic = cube("Dynamic Body Test", "DYNAMIC_BODY", (0.5, 0.0, 0.5))
dynamic.grasph.body_mass = 2.5
dynamic.grasph.body_inertia = (0.2, 0.3, 0.4)
dynamic.grasph.body_collision_radius = 0.15
dynamic.grasph.body_collision_shape = "CONVEX"
dynamic.grasph.body_collision_margin = 0.004
bpy.ops.object.select_all(action="DESELECT")
bpy.context.view_layer.objects.active = dynamic
dynamic.select_set(True)
bpy.context.scene.cursor.location = dynamic.location
bpy.context.scene.grasph_scene.new_joint_type = "PRISMATIC"
assert bpy.ops.grasph.add_joint_anchor() == {"FINISHED"}
joint_anchor = bpy.context.active_object
assert joint_anchor.type == "EMPTY"
assert joint_anchor.grasph.joint_body1 == dynamic
assert joint_anchor.grasph.joint_type == "PRISMATIC"
errors, warnings = addon.validate_scene(bpy.context.scene)
assert not errors, errors
assert any("Mixed fluid/granular" in warning for warning in warnings), warnings

# Collider visualizations are excluded from the solver and USD export.
addon._refresh_collider_outlines(bpy.context.scene, "WIRE")
outlines = [obj for obj in bpy.context.scene.objects
            if obj.get("grasph:collider_outline")]
assert len(outlines) == 2
assert all(obj.display_type == "WIRE" and obj.hide_render for obj in outlines)
assert all(obj.grasph.role == "NONE" for obj in outlines)
addon._refresh_collider_outlines(bpy.context.scene, "SOLID")
solids = [obj for obj in bpy.context.scene.objects
          if obj.get("grasph:collider_visual_mode") == "SOLID"]
assert len(solids) == 2
assert all(obj.display_type == "SOLID" and obj.show_transparent for obj in solids)
assert all(obj.hide_render and obj.grasph.role == "NONE" for obj in solids)
assert all(obj.data.materials[0].diffuse_color[0] > 0.9 for obj in solids)
addon._refresh_collider_outlines(bpy.context.scene, "OFF")

# A solver checkpoint updates one live preview object in place.
import numpy as np
preview_path = root / "output" / "blender_preview_smoke.npz"
np.savez(
    preview_path, time=np.asarray(0.25), progress=np.asarray(0.25),
    positions=np.asarray([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32),
    velocity=np.zeros((2, 3), dtype=np.float32),
    granular_positions=np.asarray([[0.2, 0.0, 1.0]], dtype=np.float32),
    granular_velocity=np.zeros((1, 3), dtype=np.float32),
    granular_radii=np.asarray([0.01], dtype=np.float32),
    body_paths=np.asarray(["/Dynamic_Body_Test"], dtype="U512"),
    body_translations=np.asarray([[0.0, 0.0, -0.02]]),
    body_rotations=np.zeros((1, 3)),
    hinge_paths=np.asarray([], dtype="U512"), hinge_angles=np.asarray([]),
)
assert addon._update_live_preview(bpy.context.scene, preview_path) == (0.25, 0.25)
fluid_preview = bpy.data.objects.get(addon._PREVIEW_CLOUDS["fluid"])
grain_preview = bpy.data.objects.get(addon._PREVIEW_CLOUDS["granular"])
assert fluid_preview is not None
assert len(fluid_preview.data.vertices) == 2
assert grain_preview is not None
assert len(grain_preview.data.vertices) == 1
assert dynamic.location.z < 0.5
# Starting a new run must clear old preview objects and snapshots immediately.
addon._prepare_live_preview_for_run(bpy.context.scene, preview_path)
assert not preview_path.exists()
assert not any(obj.get("grasph:live_preview") for obj in bpy.context.scene.objects)
assert addon._PREVIEW_CLOUDS == {"fluid": None, "granular": None}
# Deleting a preview object while the modal operator is running must cause a
# clean recreation, not a stale StructRNA ReferenceError on the next checkpoint.
np.savez(
    preview_path, time=np.asarray(0.25), progress=np.asarray(0.25),
    positions=np.asarray([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32),
    velocity=np.zeros((2, 3), dtype=np.float32),
    granular_positions=np.asarray([[0.2, 0.0, 1.0]], dtype=np.float32),
    granular_velocity=np.zeros((1, 3), dtype=np.float32),
    granular_radii=np.asarray([0.01], dtype=np.float32),
    body_paths=np.asarray(["/Dynamic_Body_Test"], dtype="U512"),
    body_translations=np.asarray([[0.0, 0.0, -0.02]]),
    body_rotations=np.zeros((1, 3)),
    hinge_paths=np.asarray([], dtype="U512"), hinge_angles=np.asarray([]),
)
assert addon._update_live_preview(bpy.context.scene, preview_path) == (0.25, 0.25)
fluid_preview = bpy.data.objects.get(addon._PREVIEW_CLOUDS["fluid"])
old_fluid_name = fluid_preview.name
bpy.data.objects.remove(fluid_preview, do_unlink=True)
assert addon._update_live_preview(bpy.context.scene, preview_path) == (0.25, 0.25)
fluid_preview = bpy.data.objects.get(addon._PREVIEW_CLOUDS["fluid"])
assert fluid_preview is not None and fluid_preview.name == old_fluid_name
preview_path.unlink()
addon._reset_run_state(bpy.context.scene)
assert abs(dynamic.location.z - 0.5) < 1e-6
assert addon._PREVIEW_CLOUDS == {"fluid": None, "granular": None}
assert not any(obj.get("grasph:live_preview") for obj in bpy.context.scene.objects)

granular_output = root / "output" / "blender_granular_smoke.usdc"
addon.export_scene(bpy.context, granular_output)
granular_stage = Usd.Stage.Open(str(granular_output))
granular_prim = next(
    prim for prim in granular_stage.Traverse()
    if prim.GetAttribute("grasph:role").Get() == "granular_source"
)
assert abs(granular_prim.GetAttribute("grasph:particleRadius").Get() - 0.031) < 1e-6
assert abs(granular_prim.GetAttribute("grasph:density").Get() - 1800.0) < 1e-6
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

dynamic_owner = dynamic_prim
dynamic_mesh = next(
    prim for prim in Usd.PrimRange(dynamic_owner)
    if prim.HasAPI(UsdPhysics.CollisionAPI)
)
assert dynamic_mesh.HasAPI(UsdPhysics.CollisionAPI)
assert dynamic_mesh.HasAPI(UsdPhysics.MeshCollisionAPI)
assert dynamic_owner.HasAPI(UsdPhysics.RigidBodyAPI)
assert dynamic_owner.HasAPI(UsdPhysics.MassAPI)
assert abs(UsdPhysics.MassAPI(dynamic_owner).GetMassAttr().Get() - 2.5) < 1e-6
prismatic_joints = [prim for prim in granular_stage.Traverse()
                    if prim.IsA(UsdPhysics.PrismaticJoint)]
assert len(prismatic_joints) == 1
prismatic = UsdPhysics.PrismaticJoint(prismatic_joints[0])
assert prismatic.GetBody1Rel().GetTargets() == [dynamic_owner.GetPath()]
assert str(prismatic.GetAxisAttr().Get()).lower() == "z"

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

# A rigid-body-only scene is valid, and reset is available without running.
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
cube("Rigid Only Tank", "STATIC", (0.0, 0.0, 0.0))
cube("Rigid Only Body", "DYNAMIC_BODY", (0.0, 0.0, 2.0))
errors, warnings = addon.validate_scene(bpy.context.scene)
assert not errors, errors
settings = bpy.context.scene.grasph_scene
settings.continue_from_result = True
settings.restart_npz = "stale_result.npz"
settings.restart_time = 4.0
addon._reset_run_state(bpy.context.scene)
assert not settings.continue_from_result
assert settings.restart_npz == ""
assert settings.restart_time == 0.0
