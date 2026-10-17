"""Run in background Blender: explicit two-body joint authoring, no demo UI."""
import importlib.util
from pathlib import Path

import bpy
from pxr import Usd, UsdPhysics

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('grasph_control', root/'blender/grasph_control.py')
addon = importlib.util.module_from_spec(spec)
spec.loader.exec_module(addon)
addon.register()
assert not hasattr(addon, 'GRASPH_OT_fracture_demo')
assert not hasattr(bpy.types.Object.grasph, 'fracture_group')
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

def block(name, position, dimensions, role):
    bpy.ops.mesh.primitive_cube_add(size=1, location=position)
    obj = bpy.context.object
    obj.name, obj.dimensions = name, dimensions
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.grasph.role = role
    if role == 'DYNAMIC_BODY':
        obj.grasph.body_mass = .5
        obj.grasph.body_inertia = (.013333333,)*3
    return obj

block('Floor', (0, 0, -.1), (4, 4, .2), 'STATIC')
block('Pillar', (-.22, 0, .3), (.3, .5, .6), 'STATIC')
a = block('Body A', (-.204, 0, 2), (.4,)*3, 'DYNAMIC_BODY')
b = block('Body B', (.204, 0, 2), (.4,)*3, 'DYNAMIC_BODY')
bpy.ops.object.select_all(action='DESELECT')
a.select_set(True)
b.select_set(True)
bpy.context.view_layer.objects.active = b
bpy.context.scene.cursor.location = (0, 0, 2)
bpy.context.scene.grasph_scene.new_joint_type = 'FIXED'
assert bpy.ops.grasph.add_joint_anchor() == {'FINISHED'}
joint = bpy.context.object
assert joint.type == 'EMPTY'
assert joint.grasph.joint_body0 == a and joint.grasph.joint_body1 == b
joint.grasph.joint_breakable = True
joint.grasph.joint_break_force = 500
joint.grasph.joint_break_torque = 50
errors, _ = addon.validate_scene(bpy.context.scene)
assert not errors, errors
output = root/'output/shared_joint_smoke.usdc'
addon.export_scene(bpy.context, output)
stage = Usd.Stage.Open(str(output))
fixed = [UsdPhysics.FixedJoint(p) for p in stage.Traverse() if p.IsA(UsdPhysics.FixedJoint)]
assert len(fixed) == 1
assert fixed[0].GetBody0Rel().GetTargets() and fixed[0].GetBody1Rel().GetTargets()
assert fixed[0].GetBreakTorqueAttr().Get() == 50
# Custom metadata works independently of the optional standard Physics export.
bpy.context.scene.grasph_scene.author_usd_physics = False
addon.export_scene(bpy.context, root/'output/shared_joint_custom_smoke.usdc')
# New world-attached fixed joints export through the same shared path.
joint.grasph.joint_body0 = None
joint.grasph.joint_breakable = False
addon.export_scene(bpy.context, root/'output/shared_joint_world_smoke.usdc')
print('Shared joint authoring passed:', output)
