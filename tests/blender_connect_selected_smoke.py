"""Batch fixed-joint creation, duplicate prevention and export in Blender."""
import importlib.util
from pathlib import Path
import sys

import bpy
from pxr import Usd, UsdPhysics

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
spec = importlib.util.spec_from_file_location('grasph_control', root/'blender/grasph_control.py')
addon = importlib.util.module_from_spec(spec)
spec.loader.exec_module(addon)
addon.register()
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
bodies = []
for i in range(3):
    bpy.ops.mesh.primitive_cube_add(size=.2, location=(i, 0, 2))
    body = bpy.context.object
    body.name = 'Batch Body '+str(i)
    body.grasph.role = 'DYNAMIC_BODY'
    bodies.append(body)
bpy.ops.mesh.primitive_cube_add(size=.1, location=(4, 0, 2))
ignored = bpy.context.object
ignored.grasph.role = 'STATIC'
for body in bodies:
    body.select_set(True)
bpy.context.view_layer.objects.active = bodies[2]
selected = set(bpy.context.selected_objects)
settings = bpy.context.scene.grasph_scene
settings.new_joint_type = 'FIXED'
settings.new_joint_breakable = True
settings.new_joint_break_force = 1500
settings.new_joint_break_torque = 75
assert bpy.ops.grasph.connect_selected() == {'FINISHED'}
joints = [o for o in bpy.context.scene.objects if o.grasph.role == 'JOINT']
assert len(joints) == 2  # N-1, not all N*(N-1)/2 pairs
assert bpy.context.active_object == bodies[2]
assert set(bpy.context.selected_objects) == selected
collection = next(c for c in bpy.context.scene.collection.children if c.get('grasph:jointCollection'))
assert set(collection.objects) == set(joints)
for joint in joints:
    p = joint.grasph
    assert p.joint_body1 == bodies[2] and p.joint_body0 in bodies[:2]
    assert p.joint_type == 'FIXED' and p.joint_breakable
    assert p.joint_break_force == 1500 and p.joint_break_torque == 75
    midpoint = .5*(p.joint_body0.matrix_world.translation+p.joint_body1.matrix_world.translation)
    assert (joint.location-midpoint).length < 1e-6
# Repeating the operation must neither duplicate nor overwrite existing joints.
settings.new_joint_breakable = False
assert bpy.ops.grasph.connect_selected() == {'FINISHED'}
assert len([o for o in bpy.context.scene.objects if o.grasph.role == 'JOINT']) == 2
assert all(o.grasph.joint_breakable for o in joints)
# Pair matching is independent of A/B ordering.
bpy.ops.object.select_all(action='DESELECT')
bodies[0].select_set(True)
bodies[2].select_set(True)
bpy.context.view_layer.objects.active = bodies[0]
assert bpy.ops.grasph.connect_selected() == {'FINISHED'}
assert len(collection.objects) == 2
# Unsupported active roles or joint kinds must disable the operator.
bpy.context.view_layer.objects.active = ignored
assert not addon.GRASPH_OT_connect_selected.poll(bpy.context)
bpy.context.view_layer.objects.active = bodies[0]
settings.new_joint_type = 'PRISMATIC'
assert not addon.GRASPH_OT_connect_selected.poll(bpy.context)
settings.new_joint_type = 'FIXED'
output = root/'output/connect_selected_smoke.usdc'
addon.export_scene(bpy.context, output)
stage = Usd.Stage.Open(str(output))
assert sum(p.IsA(UsdPhysics.FixedJoint) for p in stage.Traverse()) == 2
from grasph_scene import load_scene
loaded = load_scene(output)
assert len(loaded.fixed_joints) == 2
assert all(j.break_force == 1500 and j.break_torque == 75 for j in loaded.fixed_joints)
print('Connect Selected smoke passed:', output)
