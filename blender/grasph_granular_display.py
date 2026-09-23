"""Geometry Nodes display for animated GraSPH granular USD point caches."""

import bpy


DISPLAY_VERSION = "0.1.0"


def granular_material():
    material = bpy.data.materials.get("GraSPH Granular Material")
    if material is None:
        material = bpy.data.materials.new("GraSPH Granular Material")
        material.use_nodes = True
        bsdf = material.node_tree.nodes.get("Principled BSDF")
        bsdf.inputs["Base Color"].default_value = (0.32, 0.12, 0.025, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.72
        material.diffuse_color = (0.32, 0.12, 0.025, 1.0)
    return material


def _build_group(radius):
    group = bpy.data.node_groups.new(
        f"GraSPH Granular Display v{DISPLAY_VERSION}", "GeometryNodeTree"
    )
    group.is_modifier = True
    group["grasph_granular_display_version"] = DISPLAY_VERSION
    interface = group.interface
    interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    radius_socket = interface.new_socket(
        name="Grain Radius", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    radius_socket.default_value = float(radius)
    radius_socket.min_value = 1.0e-6
    radius_socket.description = "Rendered radius of each granular particle"
    group["grasph_sockets"] = {"Grain Radius": radius_socket.identifier}

    nodes, links = group.nodes, group.links
    entry = nodes.new("NodeGroupInput")
    entry.location = (-500, 100)
    sphere = nodes.new("GeometryNodeMeshIcoSphere")
    sphere.location = (-280, -100)
    sphere.inputs["Subdivisions"].default_value = 2
    instances = nodes.new("GeometryNodeInstanceOnPoints")
    instances.location = (0, 100)
    set_material = nodes.new("GeometryNodeSetMaterial")
    set_material.location = (240, 100)
    set_material.inputs["Material"].default_value = granular_material()
    output = nodes.new("NodeGroupOutput")
    output.location = (470, 100)
    links.new(entry.outputs["Geometry"], instances.inputs["Points"])
    links.new(entry.outputs["Grain Radius"], sphere.inputs["Radius"])
    links.new(sphere.outputs["Mesh"], instances.inputs["Instance"])
    links.new(instances.outputs["Instances"], set_material.inputs["Geometry"])
    links.new(set_material.outputs["Geometry"], output.inputs["Geometry"])
    return group


def add_granular_display(point_cloud, radius):
    if point_cloud is None or point_cloud.type != "POINTCLOUD":
        raise ValueError("A GraSPH granular point cloud is required")
    old = point_cloud.modifiers.get("GraSPH Granular Display")
    old_group = old.node_group if old else None
    if old:
        point_cloud.modifiers.remove(old)
    modifier = point_cloud.modifiers.new("GraSPH Granular Display", "NODES")
    modifier.node_group = _build_group(radius)
    identifier = modifier.node_group["grasph_sockets"]["Grain Radius"]
    modifier[identifier] = float(radius)
    point_cloud["grasph:displayKind"] = "granular"
    if old_group and old_group.users == 0:
        bpy.data.node_groups.remove(old_group)
    return point_cloud
