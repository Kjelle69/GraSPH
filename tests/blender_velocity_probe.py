"""Probe Blender's name for imported USD Points velocity data."""
import importlib.util
from pathlib import Path

import bpy
from pxr import Gf, Usd, UsdGeom

root = Path(__file__).resolve().parents[1]
path = root / "output" / "blender_velocity_probe.usda"
stage = Usd.Stage.CreateNew(str(path))
points = UsdGeom.Points.Define(stage, "/GraSPH_Result")
points.CreatePointsAttr([Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 0, 0)])
points.CreateWidthsAttr([0.1, 0.1])
points.CreateVelocitiesAttr([Gf.Vec3f(1, 2, 3), Gf.Vec3f(4, 5, 6)])
stage.GetRootLayer().Save()

control_path = root / "blender" / "grasph_control.py"
spec = importlib.util.spec_from_file_location("grasph_control", control_path)
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)
control.register()
bpy.context.scene.grasph_scene.result_usd = str(path)
bpy.context.scene.grasph_scene.dp = 0.1
result = bpy.ops.grasph.open_result()
assert "FINISHED" in result, result
clouds = [obj for obj in bpy.context.scene.objects if obj.type == "POINTCLOUD"]
assert len(clouds) >= 1
cloud = next(obj.data for obj in clouds if obj.data.attributes.get("velocity") is not None)
print("Point cloud attributes:", [(a.name, a.data_type, a.domain) for a in cloud.attributes])
for attribute in cloud.attributes:
    if attribute.data_type == "FLOAT_VECTOR":
        values = [tuple(item.vector) for item in attribute.data]
        print(attribute.name, values)

surfaces = [obj for obj in bpy.context.scene.objects if obj.get("grasph_source")]
assert len(surfaces) == 1
surface = surfaces[0]
modifier = surface.modifiers["GraSPH Surface"]
sockets = modifier.node_group["grasph_sockets"]
for required in ("Splashy Droplets", "Splash Speed", "Velocity Stretch", "Maximum Stretch"):
    assert required in sockets, (required, sockets)
bpy.context.view_layer.update()
evaluated = surface.evaluated_get(bpy.context.evaluated_depsgraph_get())
mesh = evaluated.to_mesh()
assert len(mesh.vertices) > 0
evaluated.to_mesh_clear()
print("Velocity-driven surface test passed")
