"""Build the first static paddle wheel and open channel from the imported TankAnim scene.

Run inside Blender with ``exec(open(path).read())``. Saves new files; the source
scene and its animated gate are left intact on disk.
"""
import math
from pathlib import Path

import bpy
from mathutils import Vector
from pxr import Gf, Sdf, Usd, UsdGeom


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "examples" / "PaddleWheel"
OUTPUT.mkdir(parents=True, exist_ok=True)


def add_box(vertices, faces, lower, upper):
    x0, y0, z0 = lower
    x1, y1, z1 = upper
    start = len(vertices)
    vertices.extend(((x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                     (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)))
    faces.extend(tuple(start + j for j in face) for face in
                 ((3, 2, 1, 0), (4, 5, 6, 7), (0, 1, 5, 4),
                  (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)))


def replace_mesh(obj, vertices, faces):
    mesh = bpy.data.meshes.new(obj.name + "_PaddleTest")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    old = obj.data
    obj.data = mesh
    if old.users == 0:
        bpy.data.meshes.remove(old)


tank = bpy.data.objects["Tank"]
verts, faces = [], []
add_box(verts, faces, (-0.20, -0.55, -0.012), (0.20, 0.80, 0.0))
add_box(verts, faces, (-0.20, -0.55, 0.0), (-0.191, 0.80, 0.20))
add_box(verts, faces, (0.191, -0.55, 0.0), (0.20, 0.80, 0.20))
add_box(verts, faces, (-0.191, 0.78, 0.0), (0.191, 0.80, 0.20))
replace_mesh(tank, verts, faces)

fluid = bpy.data.objects["FluidVolume"]
for vertex in fluid.data.vertices:
    if vertex.co.y > 0.4:
        vertex.co.y = 0.77
fluid.data.update()

center_y, center_z = -0.30, 0.125
verts, faces = [], []
segments = 16
for k in range(segments):
    angle = 2 * math.pi * k / segments
    next_angle = 2 * math.pi * (k + 1) / segments
    # Each ring segment is a closed solid; the narrow hub ties the blades together.
    for radius0, radius1, x0, x1 in ((0.026, 0.034, -0.165, 0.165),):
        poly = []
        for x in (x0, x1):
            for radius, a in ((radius0, angle), (radius1, angle),
                              (radius1, next_angle), (radius0, next_angle)):
                poly.append((x, center_y + radius * math.cos(a),
                             center_z + radius * math.sin(a)))
        base = len(verts)
        verts.extend(poly)
        faces.extend(tuple(base + j for j in face) for face in
                     ((3, 2, 1, 0), (4, 5, 6, 7), (0, 1, 5, 4),
                      (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)))

for k in range(8):
    a = 2 * math.pi * k / 8
    radial = Vector((math.cos(a), math.sin(a)))
    tangent = Vector((-math.sin(a), math.cos(a)))
    cross = []
    for radius, side in ((0.030, -0.006), (0.105, -0.006),
                         (0.105, 0.006), (0.030, 0.006)):
        yz = Vector((center_y, center_z)) + radius * radial + side * tangent
        cross.append((float(yz.x), float(yz.y)))
    start = len(verts)
    verts.extend((x, y, z) for x in (-0.16, 0.16) for y, z in cross)
    faces.extend(tuple(start + j for j in face) for face in
                 ((3, 2, 1, 0), (4, 5, 6, 7), (0, 1, 5, 4),
                  (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)))

wheel_mesh = bpy.data.meshes.new("PaddleWheel_mesh")
wheel_mesh.from_pydata(verts, [], faces)
wheel_mesh.update()
wheel = bpy.data.objects.new("PaddleWheel", wheel_mesh)
bpy.context.scene.collection.objects.link(wheel)
for vertex in wheel_mesh.vertices:
    vertex.co.y -= center_y
    vertex.co.z -= center_z
wheel.location = (0.0, center_y, center_z)
wheel.scale = (0.75, 0.75, 0.75)
wheel["grasph:role"] = "collider"
wheel["grasph:axis"] = "X"
wheel["grasph:axle_center"] = (0.0, center_y, center_z)
wheel["grasph:hingeEnabled"] = True
wheel["grasph:momentOfInertia"] = 0.002
wheel["grasph:angularDamping"] = 0.002
wheel["grasph:maxAngularSpeed"] = 40.0
material = bpy.data.materials.new("PaddleWheel Copper")
material.diffuse_color = (0.55, 0.20, 0.055, 1.0)
material.use_nodes = True
bsdf = material.node_tree.nodes.get("Principled BSDF")
bsdf.inputs["Base Color"].default_value = material.diffuse_color
bsdf.inputs["Metallic"].default_value = 0.55
bsdf.inputs["Roughness"].default_value = 0.38
wheel.data.materials.append(material)

sink_verts, sink_faces = [], []
add_box(sink_verts, sink_faces, (-0.25, -0.82, -0.30), (0.25, -0.555, 0.30))
sink_mesh = bpy.data.meshes.new("SinkVolume_mesh")
sink_mesh.from_pydata(sink_verts, [], sink_faces)
sink_mesh.update()
sink = bpy.data.objects.new("SinkVolume", sink_mesh)
bpy.context.scene.collection.objects.link(sink)
sink["grasph:role"] = "sink"
sink.display_type = "WIRE"
sink.hide_render = False
sink.hide_set(True)

bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT / "PaddleWheel.blend"))
bpy.ops.wm.usd_export(filepath=str(OUTPUT / "PaddleWheel.usdc"),
                      export_animation=True, export_cameras=False,
                      export_lights=False)
stage = Usd.Stage.Open(str(OUTPUT / "PaddleWheel.usdc"))
UsdGeom.Imageable(stage.GetPrimAtPath("/root/SinkVolume")).MakeInvisible()
stage.GetPrimAtPath("/root/SinkVolume/SinkVolume_mesh").CreateAttribute(
    "grasph:role", Sdf.ValueTypeNames.String).Set("sink")
wheel_prim = stage.GetPrimAtPath("/root/PaddleWheel/PaddleWheel_mesh")
wheel_prim.CreateAttribute("grasph:axis", Sdf.ValueTypeNames.String).Set("X")
wheel_prim.CreateAttribute("grasph:axleCenter", Sdf.ValueTypeNames.Double3).Set(
    Gf.Vec3d(0.0, center_y, center_z))
wheel_prim.CreateAttribute("grasph:hingeEnabled", Sdf.ValueTypeNames.Bool).Set(True)
wheel_prim.CreateAttribute("grasph:momentOfInertia", Sdf.ValueTypeNames.Double).Set(0.002)
wheel_prim.CreateAttribute("grasph:angularDamping", Sdf.ValueTypeNames.Double).Set(0.002)
wheel_prim.CreateAttribute("grasph:maxAngularSpeed", Sdf.ValueTypeNames.Double).Set(40.0)
stage.GetRootLayer().Save()
print("Created", OUTPUT / "PaddleWheel.blend", OUTPUT / "PaddleWheel.usdc")
