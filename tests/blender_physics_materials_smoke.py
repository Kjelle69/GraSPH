"""Shared physical material assignment/export/persistence in background Blender."""
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
objects = []
shader = bpy.data.materials.new('Visual Only Shader')
for i, role in enumerate(('STATIC', 'DYNAMIC_BODY', 'GRANULAR_SOURCE')):
    bpy.ops.mesh.primitive_cube_add(size=.2, location=(i, 0, 1))
    obj = bpy.context.object
    obj.grasph.role = role
    obj.data.materials.append(shader)
    objects.append(obj)
objects[1].grasph.body_mass = 3.0
assert bpy.ops.grasph.physics_material(action='EXAMPLES') == {'FINISHED'}
assert len([m for m in bpy.data.materials if m.grasph_physics.enabled]) == 5
material = bpy.data.materials['GraSPH Physics - Steel']
bpy.context.scene.grasph_scene.physics_material_index = list(bpy.data.materials).index(material)
for obj in objects:
    obj.select_set(True)
assert bpy.ops.grasph.physics_material(action='ASSIGN') == {'FINISHED'}
assert all(o.grasph.physics_material == material for o in objects)
assert all(o.active_material == shader for o in objects)
material.grasph_physics.friction = .73
material.grasph_physics.restitution = .17
assert all(abs(addon._physics_values(o.grasph)['friction']-.73) < 1e-6 for o in objects)
output = root/'output/physics_materials_smoke.usdc'
addon.export_scene(bpy.context, output)
stage = Usd.Stage.Open(str(output))
apis = [UsdPhysics.MaterialAPI(p) for p in stage.Traverse() if p.HasAPI(UsdPhysics.MaterialAPI)]
assert len(apis) == 1, 'Linked objects should share a USD physics material'
assert abs(apis[0].GetDynamicFrictionAttr().Get()-.73) < 1e-6
from grasph_scene import load_scene
loaded = load_scene(output)
assert len(loaded.granular_sources) == 1
assert abs(loaded.granular_sources[0].density-7850) < 1e-6
assert abs(loaded.granular_sources[0].friction-.73) < 1e-6
assert all(abs(m['friction']-.73) < 1e-6 for m in loaded.collider_materials.values())
assert next(iter(loaded.dynamic_bodies.values())).mass == 3.0
# Unlink reverts to existing per-object manual values without altering peers.
objects[0].grasph.physics_material = None
assert abs(addon._physics_values(objects[0].grasph)['friction']-.45) < 1e-6
assert objects[1].grasph.physics_material == material
# Reload controls must preserve shared material data and object references.
addon.register()
assert objects[1].grasph.physics_material == material
assert abs(material.grasph_physics.friction-.73) < 1e-6
bpy.ops.wm.save_as_mainfile(filepath=str(root/'output/physics_materials_smoke.blend'))
bpy.ops.wm.open_mainfile(filepath=str(root/'output/physics_materials_smoke.blend'))
material = bpy.data.materials['GraSPH Physics - Steel']
assert abs(material.grasph_physics.friction-.73) < 1e-6
assert sum(o.grasph.physics_material == material for o in bpy.context.scene.objects) == 2
print('Physics materials smoke passed:', output)
