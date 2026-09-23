"""Shared rigid-body integration and convex sampled-mesh contacts."""

import numpy as np

from grasph_collision import closest_point_triangle_batch


def axis_angle_matrix(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    x, y, z = axis / np.linalg.norm(axis)
    c, s, q = np.cos(angle), np.sin(angle), 1.0 - np.cos(angle)
    return np.array([
        [c + x*x*q, x*y*q - z*s, x*z*q + y*s],
        [y*x*q + z*s, c + y*y*q, y*z*q - x*s],
        [z*x*q - y*s, z*y*q + x*s, c + z*z*q],
    ], dtype=np.float64)


def advance_rotation(rotation, omega, dt):
    speed = float(np.linalg.norm(omega))
    if speed < 1.0e-12:
        return rotation
    return axis_angle_matrix(omega / speed, speed * dt) @ rotation


def rotation_to_euler_xyz(rotation):
    """Return XYZ Euler radians for R = Rz @ Ry @ Rx."""
    sy = float(np.clip(-rotation[2, 0], -1.0, 1.0))
    y = np.arcsin(sy)
    cy = np.cos(y)
    if abs(cy) > 1.0e-8:
        x = np.arctan2(rotation[2, 1], rotation[2, 2])
        z = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        x = np.arctan2(-rotation[1, 2], rotation[1, 1])
        z = 0.0
    return np.asarray((x, y, z), dtype=np.float64)


def world_inertia(body, state):
    rotation = state["rotation"]
    return rotation @ np.diag(body.inertia) @ rotation.T


def _contact_samples(body, state):
    if body.collision_shape == "sphere":
        return (body.center + state["translation"])[None, :], body.collision_radius
    relative = body.collision_samples - body.center[None, :]
    points = (
        body.center[None, :]
        + state["translation"][None, :]
        + relative @ state["rotation"].T
    )
    return points, body.collision_margin


def _oriented_surface_query(points, triangles):
    """Exact closest-face query with the authored outward face normal."""
    best_d2 = np.full(len(points), np.inf, dtype=np.float64)
    best_point = np.zeros_like(points)
    best_normal = np.zeros_like(points)
    for triangle in triangles:
        a, b, c = triangle
        closest = closest_point_triangle_batch(points, a, b, c)
        delta = points - closest
        d2 = np.einsum("ij,ij->i", delta, delta)
        better = d2 < best_d2
        if not np.any(better):
            continue
        normal = np.cross(b - a, c - a)
        length = float(np.linalg.norm(normal))
        if length < 1.0e-15:
            continue
        best_d2[better] = d2[better]
        best_point[better] = closest[better]
        best_normal[better] = normal / length
    signed_distance = np.einsum("ij,ij->i", points - best_point, best_normal)
    return signed_distance, best_normal


def resolve_body_contacts(
    scene, colliders, body_states, collider_translations, collider_velocities,
    collider_rotations, collider_centers, collider_angular_velocities,
    iterations=2,
):
    """Resolve convex dynamic bodies against static, kinematic and dynamic meshes.

    Convex bodies use authored mesh vertices, edge midpoints and face centroids.
    One deepest sample is solved per collider and iteration. Equal-and-opposite
    impulses are applied immediately when the other collider is dynamic; hinge
    reaction impulses are returned to the caller.
    """
    collider_by_name = {collider.name: collider for collider in colliders}
    angular_reactions = {}
    contacts = 0

    for _ in range(max(1, int(iterations))):
        for path, state in body_states.items():
            body = scene.dynamic_bodies[path]
            body_material = scene.collider_materials.get(path, {})
            samples, contact_radius = _contact_samples(body, state)

            for other_path, collider in collider_by_name.items():
                if other_path == path:
                    continue
                # A dynamic pair is solved only once.
                if other_path in body_states and path > other_path:
                    continue

                # A sampled surface point can cross a very thin user margin
                # between steps. Keep at least half a distance-field voxel as
                # a numerical contact skin so the unsigned field is resolved
                # while the point is still on the correct side.
                effective_radius = contact_radius
                if body.collision_shape != "sphere":
                    effective_radius = max(
                        contact_radius, 0.5 * float(collider.voxel_size)
                    )

                other_translation = np.asarray(
                    collider_translations.get(other_path, np.zeros(3)), dtype=np.float64
                )
                other_rotation = np.asarray(
                    collider_rotations.get(other_path, np.eye(3)), dtype=np.float64
                )
                other_center = np.asarray(
                    collider_centers.get(other_path, np.zeros(3)), dtype=np.float64
                )
                relative = samples - other_translation[None, :] - other_center[None, :]
                query = other_center[None, :] + relative @ other_rotation
                distance, normal_local = _oriented_surface_query(
                    query, collider.triangles
                )
                penetration = effective_radius - distance
                candidate = np.where(penetration > 0.0)[0]
                if not len(candidate):
                    continue
                deepest = candidate[int(np.argmax(penetration[candidate]))]
                normal = normal_local[deepest] @ other_rotation.T
                normal_length = float(np.linalg.norm(normal))
                if normal_length < 0.5:
                    continue
                normal /= normal_length
                depth = float(penetration[deepest])
                state["translation"] += normal * depth
                samples += normal[None, :] * depth
                point = samples[deepest] - normal * effective_radius
                center = body.center + state["translation"]
                lever = point - center

                other_state = body_states.get(other_path)
                other_velocity = np.asarray(
                    collider_velocities.get(other_path, np.zeros(3)), dtype=np.float64
                )
                other_omega = np.asarray(
                    collider_angular_velocities.get(other_path, np.zeros(3)), dtype=np.float64
                )
                other_world_center = other_center + other_translation
                if other_state is not None:
                    other_velocity = other_state["velocity"]
                    other_omega = other_state["omega"]
                    other_world_center = (
                        scene.dynamic_bodies[other_path].center
                        + other_state["translation"]
                    )
                wall_lever = point - other_world_center
                relative_velocity = (
                    state["velocity"] + np.cross(state["omega"], lever)
                    - other_velocity - np.cross(other_omega, wall_lever)
                )
                vn = float(np.dot(relative_velocity, normal))
                contacts += 1
                if vn >= 0.0:
                    continue

                inv_inertia = np.linalg.inv(world_inertia(body, state))
                inv_mass = 1.0 / body.mass
                denominator = inv_mass + float(np.dot(
                    normal, np.cross(inv_inertia @ np.cross(lever, normal), lever)
                ))
                other_inv_inertia = None
                if other_state is not None:
                    other_body = scene.dynamic_bodies[other_path]
                    other_inv_inertia = np.linalg.inv(world_inertia(other_body, other_state))
                    denominator += 1.0 / other_body.mass + float(np.dot(
                        normal,
                        np.cross(
                            other_inv_inertia @ np.cross(wall_lever, normal),
                            wall_lever,
                        ),
                    ))
                elif other_path in scene.hinges:
                    hinge = scene.hinges[other_path]
                    arm_cross_n = np.cross(wall_lever, normal)
                    denominator += float(np.dot(arm_cross_n, hinge.axis)) ** 2 / hinge.moment_of_inertia

                wall_material = scene.collider_materials.get(other_path, {})
                restitution = np.sqrt(max(
                    0.0,
                    float(body_material.get("restitution", 0.1))
                    * float(wall_material.get("restitution", 0.1)),
                ))
                normal_impulse_size = -(1.0 + restitution) * vn / max(denominator, 1.0e-12)
                impulse = normal_impulse_size * normal

                tangent = relative_velocity - vn * normal
                tangent_speed = float(np.linalg.norm(tangent))
                if tangent_speed > 1.0e-12:
                    tangent_direction = tangent / tangent_speed
                    tangent_denominator = inv_mass + float(np.dot(
                        tangent_direction,
                        np.cross(
                            inv_inertia @ np.cross(lever, tangent_direction), lever
                        ),
                    ))
                    if other_state is not None:
                        other_body = scene.dynamic_bodies[other_path]
                        tangent_denominator += 1.0 / other_body.mass + float(np.dot(
                            tangent_direction,
                            np.cross(
                                other_inv_inertia @ np.cross(wall_lever, tangent_direction),
                                wall_lever,
                            ),
                        ))
                    friction = np.sqrt(max(
                        0.0,
                        float(body_material.get("friction", 0.45))
                        * float(wall_material.get("friction", 0.45)),
                    ))
                    tangent_size = min(
                        tangent_speed / max(tangent_denominator, 1.0e-12),
                        friction * normal_impulse_size,
                    )
                    impulse -= tangent_size * tangent_direction

                state["velocity"] += impulse / body.mass
                state["omega"] += inv_inertia @ np.cross(lever, impulse)
                reaction = -impulse
                reaction_torque = np.cross(wall_lever, reaction)
                if other_state is not None:
                    other_body = scene.dynamic_bodies[other_path]
                    other_state["velocity"] += reaction / other_body.mass
                    other_state["omega"] += other_inv_inertia @ reaction_torque
                elif other_path in scene.hinges:
                    angular_reactions[other_path] = (
                        angular_reactions.get(other_path, np.zeros(3)) + reaction_torque
                    )

    return contacts, angular_reactions
