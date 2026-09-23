"""GraSPH point-cloud preview: separate liquid body and isolated spray.

Blender 4.5 / 5.x. Drop-in replacement for blender/grasph_surface.py.
This module only creates Geometry Nodes; the imported USD particle cache and
simulation data remain unchanged. No scene/solver-specific paths are used.
"""

bl_info = {
    'name': 'GraSPH Surface', 'author': 'GraSPH', 'version': (0, 2, 2),
    'blender': (4, 5, 0), 'category': 'Object',
}

SURFACE_VERSION = '0.2.2'

import bpy
from bpy.props import FloatProperty
from bpy_extras.io_utils import ImportHelper


SETTINGS = (
    # name, type, default factory(dp), description
    ('Show Surface', 'NodeSocketBool', lambda dp: True,
     'Show the continuous liquid body'),
    ('Show Spray', 'NodeSocketBool', lambda dp: True,
     'Show separated spray independently of Show Surface (spray-only debug mode supported)'),
    ('Show Particles', 'NodeSocketBool', lambda dp: False,
     'Show every original SPH particle for diagnostics'),
    ('Separate Spray', 'NodeSocketBool', lambda dp: True,
     'Treat isolated particles separately instead of meshing them into the liquid body'),
    ('Mesh Radius', 'NodeSocketFloat', lambda dp: 1.30 * dp,
     'Kernel radius of the continuous liquid surface, in metres'),
    ('Voxel Size', 'NodeSocketFloat', lambda dp: 0.30 * dp,
     'Resolution of the body surface; lower is smoother but more expensive'),
    ('Threshold', 'NodeSocketFloat', lambda dp: 0.65,
     'Body isosurface threshold (visual setting, not the SPH density)'),
    ('Spray Separation', 'NodeSocketFloat', lambda dp: 1.60 * dp,
     'Nearest-neighbour distance above which a particle is rendered as spray, in metres'),
    ('Spray Radius', 'NodeSocketFloat', lambda dp: 0.30 * dp,
     'Physical visual radius of an isolated droplet, in metres'),
    ('Splashy Droplets', 'NodeSocketBool', lambda dp: True,
     'Stretch isolated, fast droplets along their velocity (never body particles)'),
    ('Splash Speed', 'NodeSocketFloat', lambda dp: 0.50,
     'Minimum particle speed for elongation, m/s'),
    ('Velocity Stretch', 'NodeSocketFloat', lambda dp: 0.020,
     'Visual elongation per unit speed above threshold, seconds'),
    ('Maximum Stretch', 'NodeSocketFloat', lambda dp: 1.50 * dp,
     'Maximum extra length of elongated droplets, in metres'),
    ('Fast Droplet Radius', 'NodeSocketFloat', lambda dp: 0.80,
     'Relative cross-section of a fast, elongated droplet'),
    ('Particle Display Radius', 'NodeSocketFloat', lambda dp: 0.20 * dp,
     'Radius of SPH debug particle display spheres, in metres'),
)


def water_material():
    material = bpy.data.materials.get('GraSPH Water Preview v2')
    if material is None:
        material = bpy.data.materials.new('GraSPH Water Preview v2')
        material.use_nodes = True
        bsdf = material.node_tree.nodes.get('Principled BSDF')
        bsdf.inputs['Base Color'].default_value = (0.08, 0.35, 0.52, 1.0)
        bsdf.inputs['Roughness'].default_value = 0.10
        bsdf.inputs['IOR'].default_value = 1.333
        bsdf.inputs['Transmission Weight'].default_value = 0.75
        material.diffuse_color = (0.10, 0.42, 0.60, 1.0)
    return material


def resolution(node, mode):
    # 4.5 uses enum property; Blender 5 exposes a menu socket.
    if hasattr(node, 'resolution_mode'):
        node.resolution_mode = mode.replace(' ', '_').upper()
    else:
        node.inputs['Resolution Mode'].default_value = {
            'Voxel Size': 'Size', 'Voxel Amount': 'Amount',
        }.get(mode, mode)


def surface_group(dp, points):
    """Create one live Geometry Nodes setup from the unmodified USD points."""
    group = bpy.data.node_groups.new(f'GraSPH Surface v{SURFACE_VERSION}', 'GeometryNodeTree')
    group['grasph_surface_version'] = SURFACE_VERSION
    group.is_modifier = True
    iface = group.interface
    iface.new_socket(name='Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
    iface.new_socket(name='Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
    identifiers = {}
    for name, kind, factory, description in SETTINGS:
        socket = iface.new_socket(name=name, in_out='INPUT', socket_type=kind)
        socket.default_value = factory(dp)
        socket.description = description
        if kind == 'NodeSocketFloat':
            socket.min_value = 0.000001
        identifiers[name] = socket.identifier

    nodes, links = group.nodes, group.links

    def new(node_type, label, x, y):
        node = nodes.new(node_type)
        node.label = label
        node.location = (x, y)
        return node

    entry = new('NodeGroupInput', 'Surface settings', -1150, 500)
    exit = new('NodeGroupOutput', 'Visible fluid', 1500, 400)
    info = new('GeometryNodeObjectInfo', 'Original USD particles', -1150, 980)
    info.transform_space = 'RELATIVE'
    info.inputs['Object'].default_value = points
    info.inputs['As Instance'].default_value = False

    # Measure the distance to the closest OTHER point. On an initially regular
    # dp lattice the first neighbours are dp apart; only clearly detached
    # points become spray. This is a visual heuristic, NOT an SPH neighbour count.
    position = new('GeometryNodeInputPosition', 'Point position', -1150, 1250)
    nearest = new('GeometryNodeIndexOfNearest', 'Nearest other particle', -960, 1250)
    at_index = new('GeometryNodeFieldAtIndex', 'Position of neighbour', -760, 1250)
    at_index.data_type = 'FLOAT_VECTOR'
    at_index.domain = 'POINT'
    links.new(nearest.outputs['Index'], at_index.inputs['Index'])
    links.new(position.outputs['Position'], at_index.inputs['Value'])
    distance = new('ShaderNodeVectorMath', 'Nearest-neighbour distance', -560, 1250)
    distance.operation = 'DISTANCE'
    links.new(position.outputs['Position'], distance.inputs[0])
    links.new(at_index.outputs['Value'], distance.inputs[1])
    is_far = new('ShaderNodeMath', 'Separated farther than threshold', -350, 1250)
    is_far.operation = 'GREATER_THAN'
    links.new(distance.outputs['Value'], is_far.inputs[0])
    links.new(entry.outputs['Spray Separation'], is_far.inputs[1])
    enabled = new('FunctionNodeBooleanMath', 'Enable separation', -130, 1250)
    enabled.operation = 'AND'
    links.new(is_far.outputs[0], enabled.inputs[0])
    links.new(entry.outputs['Separate Spray'], enabled.inputs[1])
    separated = new('GeometryNodeSeparateGeometry', 'Body | isolated spray', 100, 1000)
    separated.domain = 'POINT'
    links.new(info.outputs['Geometry'], separated.inputs['Geometry'])
    links.new(enabled.outputs[0], separated.inputs['Selection'])
    body = separated.outputs['Inverted']
    spray = separated.outputs['Selection']

    # BODY: no velocity trails and no large spheres around detached particles.
    body_volume = new('GeometryNodePointsToVolume', 'Continuous fluid only', 360, 1000)
    resolution(body_volume, 'Voxel Size')
    body_volume.inputs['Density'].default_value = 1.0
    links.new(body, body_volume.inputs['Points'])
    links.new(entry.outputs['Mesh Radius'], body_volume.inputs['Radius'])
    links.new(entry.outputs['Voxel Size'], body_volume.inputs['Voxel Size'])
    body_mesh = new('GeometryNodeVolumeToMesh', 'Body isosurface', 590, 1000)
    resolution(body_mesh, 'Grid')
    body_mesh.inputs['Adaptivity'].default_value = 0.0
    links.new(body_volume.outputs['Volume'], body_mesh.inputs['Volume'])
    links.new(entry.outputs['Threshold'], body_mesh.inputs['Threshold'])
    shade = new('GeometryNodeSetShadeSmooth', 'Smooth body normals', 810, 1000)
    links.new(body_mesh.outputs['Mesh'], shade.inputs['Geometry'])
    body_material = new('GeometryNodeSetMaterial', 'Water material', 1020, 1000)
    body_material.inputs['Material'].default_value = water_material()
    links.new(shade.outputs['Geometry'], body_material.inputs['Geometry'])
    body_visible = new('GeometryNodeSwitch', 'Show body', 1250, 1000)
    body_visible.input_type = 'GEOMETRY'
    links.new(entry.outputs['Show Surface'], body_visible.inputs['Switch'])
    links.new(body_material.outputs['Geometry'], body_visible.inputs['True'])

    # SPRAY: render each truly detached particle independently. Unlike the
    # previous version, DO NOT feed extra trailing samples into Points to Volume:
    # those samples created the large merging blue tubes seen in the preview.
    velocity = new('GeometryNodeInputNamedAttribute', 'USD particle velocity', -360, 580)
    velocity.data_type = 'FLOAT_VECTOR'
    velocity.inputs['Name'].default_value = 'velocity'
    # USD import naming differs between exporter/importer builds. Retain the
    # original `velocity` field, but also accept `velocities` when present.
    velocities = new('GeometryNodeInputNamedAttribute', 'Alternative USD velocity', -360, 350)
    velocities.data_type = 'FLOAT_VECTOR'
    velocities.inputs['Name'].default_value = 'velocities'
    selected_velocity = new('GeometryNodeSwitch', 'Choose imported velocity', -160, 350)
    selected_velocity.input_type = 'VECTOR'
    links.new(velocity.outputs['Exists'], selected_velocity.inputs['Switch'])
    links.new(velocity.outputs['Attribute'], selected_velocity.inputs['True'])
    links.new(velocities.outputs['Attribute'], selected_velocity.inputs['False'])
    speed = new('ShaderNodeVectorMath', 'Particle speed', -160, 580)
    speed.operation = 'LENGTH'
    links.new(selected_velocity.outputs[0], speed.inputs[0])
    over_speed = new('ShaderNodeMath', 'Speed above splash threshold', 40, 580)
    over_speed.operation = 'SUBTRACT'
    links.new(speed.outputs['Value'], over_speed.inputs[0])
    links.new(entry.outputs['Splash Speed'], over_speed.inputs[1])
    positive = new('ShaderNodeMath', 'Ignore slow droplets', 240, 580)
    positive.operation = 'MAXIMUM'
    positive.inputs[1].default_value = 0.0
    links.new(over_speed.outputs[0], positive.inputs[0])
    extra = new('ShaderNodeMath', 'Extra visible length', 440, 580)
    extra.operation = 'MULTIPLY'
    links.new(positive.outputs[0], extra.inputs[0])
    links.new(entry.outputs['Velocity Stretch'], extra.inputs[1])
    clamp = new('ShaderNodeMath', 'Limit visual stretch', 640, 580)
    clamp.operation = 'MINIMUM'
    links.new(extra.outputs[0], clamp.inputs[0])
    links.new(entry.outputs['Maximum Stretch'], clamp.inputs[1])
    diameter = new('ShaderNodeMath', 'Droplet diameter', 440, 340)
    diameter.operation = 'MULTIPLY'
    diameter.inputs[1].default_value = 2.0
    links.new(entry.outputs['Spray Radius'], diameter.inputs[0])
    stretch_fraction = new('ShaderNodeMath', 'Relative elongation', 820, 580)
    stretch_fraction.operation = 'DIVIDE'
    links.new(clamp.outputs[0], stretch_fraction.inputs[0])
    links.new(diameter.outputs[0], stretch_fraction.inputs[1])
    axial_scale = new('ShaderNodeMath', 'Long-axis scale', 1000, 580)
    axial_scale.operation = 'ADD'
    axial_scale.inputs[1].default_value = 1.0
    links.new(stretch_fraction.outputs[0], axial_scale.inputs[0])
    stretch_switch = new('GeometryNodeSwitch', 'Stretch only when enabled', 1180, 580)
    stretch_switch.input_type = 'FLOAT'
    stretch_switch.inputs['False'].default_value = 1.0
    links.new(entry.outputs['Splashy Droplets'], stretch_switch.inputs['Switch'])
    links.new(axial_scale.outputs[0], stretch_switch.inputs['True'])

    radius_factor = new('GeometryNodeSwitch', 'Narrow fast drops only', 440, 100)
    radius_factor.input_type = 'FLOAT'
    radius_factor.inputs['False'].default_value = 1.0
    links.new(entry.outputs['Fast Droplet Radius'], radius_factor.inputs['True'])
    is_fast = new('ShaderNodeMath', 'Fast spray', 240, 100)
    is_fast.operation = 'GREATER_THAN'
    links.new(speed.outputs['Value'], is_fast.inputs[0])
    links.new(entry.outputs['Splash Speed'], is_fast.inputs[1])
    both_fast = new('FunctionNodeBooleanMath', 'Splash toggle', 440, -80)
    both_fast.operation = 'AND'
    links.new(is_fast.outputs[0], both_fast.inputs[0])
    links.new(entry.outputs['Splashy Droplets'], both_fast.inputs[1])
    links.new(both_fast.outputs[0], radius_factor.inputs['Switch'])
    cross_radius = new('ShaderNodeMath', 'Droplet cross-section', 650, 100)
    cross_radius.operation = 'MULTIPLY'
    links.new(entry.outputs['Spray Radius'], cross_radius.inputs[0])
    links.new(radius_factor.outputs[0], cross_radius.inputs[1])
    scale = new('ShaderNodeCombineXYZ', 'Droplet instance scale', 1200, 260)
    links.new(cross_radius.outputs[0], scale.inputs['X'])
    links.new(cross_radius.outputs[0], scale.inputs['Y'])
    # The height node below drives Z to keep radius and elongation independent.
    height = new('ShaderNodeMath', 'Droplet half-length', 990, 260)
    height.operation = 'MULTIPLY'
    links.new(entry.outputs['Spray Radius'], height.inputs[0])
    links.new(stretch_switch.outputs[0], height.inputs[1])
    links.new(height.outputs[0], scale.inputs['Z'])

    align = new('FunctionNodeAlignEulerToVector', 'Orient along particle velocity', 1000, -80)
    align.axis = 'Z'
    links.new(selected_velocity.outputs[0], align.inputs['Vector'])
    ball = new('GeometryNodeMeshIcoSphere', 'Individual droplet', 650, -230)
    ball.inputs['Radius'].default_value = 1.0
    ball.inputs['Subdivisions'].default_value = 2
    spray_mat = new('GeometryNodeSetMaterial', 'Same material as body', 840, -230)
    spray_mat.inputs['Material'].default_value = water_material()
    links.new(ball.outputs['Mesh'], spray_mat.inputs['Geometry'])
    inst_spray = new('GeometryNodeInstanceOnPoints', 'Detached droplets', 1420, 100)
    links.new(spray, inst_spray.inputs['Points'])
    links.new(spray_mat.outputs['Geometry'], inst_spray.inputs['Instance'])
    links.new(scale.outputs['Vector'], inst_spray.inputs['Scale'])
    links.new(align.outputs['Rotation'], inst_spray.inputs['Rotation'])
    # Spray has its own visibility control. Previously Show Surface=False also
    # hid spray, making it impossible to inspect the classification on its own.
    spray_visible = new('GeometryNodeSwitch', 'Show detached droplets independently', 1620, 100)
    spray_visible.input_type = 'GEOMETRY'
    links.new(entry.outputs['Show Spray'], spray_visible.inputs['Switch'])
    links.new(inst_spray.outputs['Instances'], spray_visible.inputs['True'])

    # DEBUG: original particles, regardless of how the preview classified them.
    debug_ball = new('GeometryNodeMeshIcoSphere', 'Debug particle', 650, -590)
    debug_ball.inputs['Subdivisions'].default_value = 1
    links.new(entry.outputs['Particle Display Radius'], debug_ball.inputs['Radius'])
    debug_instances = new('GeometryNodeInstanceOnPoints', 'Original particles', 1010, -590)
    links.new(info.outputs['Geometry'], debug_instances.inputs['Points'])
    links.new(debug_ball.outputs['Mesh'], debug_instances.inputs['Instance'])
    debug_visible = new('GeometryNodeSwitch', 'Show simulation particles', 1270, -590)
    debug_visible.input_type = 'GEOMETRY'
    links.new(entry.outputs['Show Particles'], debug_visible.inputs['Switch'])
    links.new(debug_instances.outputs['Instances'], debug_visible.inputs['True'])

    join = new('GeometryNodeJoinGeometry', 'Body + spray + debug', 1830, 450)
    for output in (body_visible.outputs['Output'], spray_visible.outputs['Output'],
                   debug_visible.outputs['Output']):
        links.new(output, join.inputs['Geometry'])
    links.new(join.outputs['Geometry'], exit.inputs['Geometry'])
    group['grasph_sockets'] = identifiers
    return group


def _read_existing_settings(modifier):
    """Retain user-tuned values when Create / Reset updates the node graph."""
    if not modifier or not modifier.node_group:
        return {}
    old_group = modifier.node_group
    old_ids = old_group.get('grasph_sockets', {})
    result = {}
    for name, identifier in old_ids.items():
        if identifier in modifier:
            result[name] = modifier[identifier]
        else:
            for item in old_group.interface.items_tree:
                if item.item_type == 'SOCKET' and item.in_out == 'INPUT' and item.name == name:
                    result[name] = item.default_value
                    break
    return result


def add_surface(source, dp):
    if source is None or source.type != 'POINTCLOUD':
        raise ValueError('Select an imported GraSPH Point Cloud.')
    if dp <= 0:
        raise ValueError('Particle spacing must be positive.')
    scene = bpy.context.scene
    surface = next((o for o in scene.objects
                    if o.get('grasph_source') == source.name), None)
    if surface is None:
        # A fresh raw copy preserves the USD cache; removing imported display
        # Geometry Nodes from this COPY leaves the original source untouched.
        raw = source.copy()
        raw.name = source.name + '_RawPoints'
        bpy.context.collection.objects.link(raw)
        for modifier in list(raw.modifiers):
            if modifier.type == 'NODES':
                raw.modifiers.remove(modifier)
        raw.hide_set(True)
        raw.hide_render = True
        surface = bpy.data.objects.new(source.name + '_Surface',
                                       bpy.data.meshes.new('GraSPH Surface Host'))
        bpy.context.collection.objects.link(surface)
        surface.matrix_world = source.matrix_world.copy()
        surface['grasph_source'] = source.name
        surface['grasph_raw'] = raw.name
        previous = {}
    else:
        raw = bpy.data.objects.get(surface.get('grasph_raw', ''))
        if raw is None or raw.type != 'POINTCLOUD':
            raise RuntimeError('GraSPH raw cache is missing; reimport the USD result.')
        previous = _read_existing_settings(surface.modifiers.get('GraSPH Surface'))

    old = surface.modifiers.get('GraSPH Surface')
    old_group = old.node_group if old else None
    # Build the new graph BEFORE discarding the existing working graph.
    new_group = surface_group(dp, raw)
    if old:
        surface.modifiers.remove(old)
    modifier = surface.modifiers.new('GraSPH Surface', 'NODES')
    modifier.node_group = new_group
    for name, identifier in new_group['grasph_sockets'].items():
        if name in previous:
            modifier[identifier] = previous[name]
    if old_group and old_group.users == 0:
        bpy.data.node_groups.remove(old_group)

    surface['grasph:dp'] = dp
    surface['grasph:surface_version'] = SURFACE_VERSION
    surface.hide_viewport = False
    surface.hide_render = False
    surface.hide_set(False)
    source.hide_set(True)
    source.hide_render = True
    bpy.ops.object.select_all(action='DESELECT')
    surface.select_set(True)
    bpy.context.view_layer.objects.active = surface
    return surface


class GRASPH_OT_surface(bpy.types.Operator):
    bl_idname = 'grasph.add_surface'
    bl_label = 'Create / Update Surface'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        source = context.active_object
        if source is not None and source.get('grasph_source'):
            source = bpy.data.objects.get(source['grasph_source'])
        try:
            add_surface(source, context.scene.grasph_mesh_dp)
        except (ValueError, RuntimeError) as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        return {'FINISHED'}


class GRASPH_OT_import(bpy.types.Operator, ImportHelper):
    bl_idname = 'grasph.import_result'
    bl_label = 'Import GraSPH Result USD'
    bl_options = {'REGISTER', 'UNDO'}
    filter_glob: bpy.props.StringProperty(default='*.usd;*.usda;*.usdc',
                                          options={'HIDDEN'})

    def execute(self, context):
        before = set(context.scene.objects)
        bpy.ops.wm.usd_import(filepath=self.filepath, import_points=True)
        points = [o for o in context.scene.objects
                  if o not in before and o.type == 'POINTCLOUD']
        if not points:
            self.report({'ERROR'}, 'The USD contains no imported Point Clouds.')
            return {'CANCELLED'}
        for source in points:
            add_surface(source, context.scene.grasph_mesh_dp)
        return {'FINISHED'}


class GRASPH_PT_surface(bpy.types.Panel):
    bl_label = 'GraSPH Fluid Display'
    bl_idname = 'GRASPH_PT_surface'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'GraSPH'

    def draw(self, context):
        layout = self.layout
        layout.prop(context.scene, 'grasph_mesh_dp')
        layout.operator('grasph.import_result')
        layout.operator('grasph.add_surface')
        obj = context.active_object
        modifier = obj.modifiers.get('GraSPH Surface') if obj else None
        if modifier and modifier.node_group:
            active_version = modifier.node_group.get('grasph_surface_version')
            layout.label(text=f'Surface v{active_version}' if active_version else 'Surface: legacy / unversioned', icon='INFO')
            for name, identifier in modifier.node_group['grasph_sockets'].items():
                layout.prop(modifier, '["' + identifier + '"]', text=name)


CLASSES = (GRASPH_OT_surface, GRASPH_OT_import, GRASPH_PT_surface)


def register():
    for cls in CLASSES:
        existing = getattr(bpy.types, cls.__name__, None)
        if existing:
            bpy.utils.unregister_class(existing)
        bpy.utils.register_class(cls)
    bpy.types.Scene.grasph_mesh_dp = FloatProperty(
        name='Simulation dp (m)', default=0.02, min=0.000001,
        description='Spacing used for this simulation; only sets initial meshing defaults')


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.grasph_mesh_dp


if __name__ == '__main__':
    register()
