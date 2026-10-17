"""Shared rigid-body integration and convex sampled-mesh contacts."""

import numpy as np

from grasph_collision import closest_point_triangle_batch, points_inside_closed_mesh


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


def constrain_body_state(body, state):
    """Project a world-attached prismatic/fixed body onto its allowed DOFs."""
    joint_type = str(getattr(body, "joint_type", "free")).lower()
    if joint_type == "fixed":
        state["translation"][:] = 0.0
        state["velocity"][:] = 0.0
        state["rotation"][:] = np.eye(3)
        state["omega"][:] = 0.0
    elif joint_type == "prismatic":
        axis = np.asarray(body.joint_axis, dtype=np.float64)
        length = float(np.linalg.norm(axis))
        if length < 1.0e-12:
            raise RuntimeError(f"Prismatic body {body.path} has a zero joint axis")
        axis = axis / length
        state["translation"][:] = axis * float(np.dot(state["translation"], axis))
        state["velocity"][:] = axis * float(np.dot(state["velocity"], axis))
        state["rotation"][:] = np.eye(3)
        state["omega"][:] = 0.0


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
    delta = points - best_point
    distance = np.sqrt(np.maximum(best_d2, 0.0))
    # Use the nearest-feature distance, not only its projection on the face
    # normal.  The latter incorrectly extends every triangle's plane
    # infinitely far past its edges (e.g. making a tank rim act like a lid).
    side = np.einsum("ij,ij->i", delta, best_normal)
    inside = side < 0.0
    # At an edge/vertex the closest triangle's normal alone cannot classify
    # inside/outside: roundoff can select an adjacent face and turn a distant
    # exterior point into a large negative penetration. Resolve ambiguous
    # negative signs using the mesh's solid angle (4*pi inside a closed mesh).
    tangent = delta - side[:, None] * best_normal
    feature = np.linalg.norm(tangent, axis=1) > 1e-10 * (1.0 + distance)
    ambiguous = inside & feature
    if np.any(ambiguous):
        inside[ambiguous] = points_inside_closed_mesh(points[ambiguous], triangles)
    signed_distance = np.where(inside, -distance, distance)
    # Corner contacts resolve along the nearest-feature gradient. Face
    # contacts retain the authored face normal, including open thin surfaces.
    feature &= distance > 1e-14
    best_normal[feature] = delta[feature] / distance[feature, None]
    best_normal[feature & inside] *= -1.0
    return signed_distance, best_normal


def _unique_axes(vectors):
    vectors = np.asarray(vectors, dtype=np.float64).reshape(-1, 3)
    lengths = np.linalg.norm(vectors, axis=1)
    axes = vectors[lengths > 1e-10] / lengths[lengths > 1e-10, None]
    if not len(axes):
        return axes
    leading = np.argmax(np.abs(axes) > 1e-10, axis=1)
    axes *= np.where(axes[np.arange(len(axes)), leading] < 0, -1., 1.)[:, None]
    _, indices = np.unique(np.round(axes, 10), axis=0, return_index=True)
    return axes[indices]


def _convex_features(collider):
    """Cache reference-pose vertices, face axes and actual (non-diagonal) edges."""
    cached = getattr(collider, '_rigid_convex_features', None)
    if cached is not None:
        return cached
    triangles = np.asarray(collider.triangles, dtype=np.float64)
    vertices, indices = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    normals = normals / np.maximum(lengths[:, None], 1e-30)
    adjacency = {}
    for face, corners in enumerate(indices.reshape(-1, 3)):
        for a, b in zip(corners, np.roll(corners, -1)):
            adjacency.setdefault(tuple(sorted((int(a), int(b)))), []).append(face)
    edges = []
    for (a, b), faces in adjacency.items():
        if len(faces) > 1 and all(abs(np.dot(normals[faces[0]], normals[f])) > 1-1e-9 for f in faces[1:]):
            continue  # Internal triangulation of a planar convex face.
        edges.append(vertices[b]-vertices[a])
    cached = vertices, _unique_axes(normals), _unique_axes(edges)
    collider._rigid_convex_features = cached
    return cached


def _hull2(points):
    points = sorted(set(map(tuple, np.asarray(points))))
    if len(points) <= 2:
        return np.asarray(points)
    def cross(a, b, c):
        u, v = np.asarray(b)-a, np.asarray(c)-a
        return u[0]*v[1]-u[1]*v[0]
    lower, upper = [], []
    for target, sequence in ((lower, points), (upper, reversed(points))):
        for point in sequence:
            while len(target) >= 2 and cross(target[-2], target[-1], point) <= 1e-14:
                target.pop()
            target.append(point)
    return np.asarray(lower[:-1]+upper[:-1])


def _support_contact_point(a, b, normal):
    """Centroid of projected supporting-feature overlap (face/edge/vertex)."""
    pa, pb = a @ normal, b @ normal
    tolerance = 1e-8 * max(1., np.ptp(pa), np.ptp(pb))
    fa, fb = a[pa <= pa.min()+tolerance], b[pb >= pb.max()-tolerance]
    helper = np.eye(3)[np.argmin(abs(normal))]
    u = np.cross(normal, helper); u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    basis = np.column_stack((u, v))
    aa, bb = _hull2(fa @ basis), _hull2(fb @ basis)
    # Clip a convex polygon/segment/point against the other supporting face.
    subject, clip = (aa, bb) if len(bb) >= 3 else (bb, aa)
    if len(clip) >= 3:
        polygon = list(subject)
        for start, end in zip(clip, np.roll(clip, -1, axis=0)):
            edge = end-start
            def side(p):
                d = p-start
                return edge[0]*d[1]-edge[1]*d[0]
            previous_polygon, polygon = polygon, []
            if not previous_polygon:
                break
            previous = previous_polygon[-1]; sp = side(previous)
            for current in previous_polygon:
                sc = side(current)
                if (sc >= -1e-12) != (sp >= -1e-12):
                    polygon.append(previous + sp/(sp-sc)*(current-previous))
                if sc >= -1e-12:
                    polygon.append(current)
                previous, sp = current, sc
        if polygon:
            tangent = np.mean(np.unique(np.round(polygon, 12), axis=0), axis=0)
            return basis @ tangent + .5*(pa.min()+pb.max())*normal
    # Degenerate edge-edge/vertex contacts, or disjoint inflated features.
    # Closest projected segments also cover point segments (zero length).
    best, tangent = np.inf, .5*(aa.mean(axis=0)+bb.mean(axis=0))
    for x, y in zip(aa, np.roll(aa, -1, axis=0)):
        for z, w in zip(bb, np.roll(bb, -1, axis=0)):
            # Endpoint projections cover parallel edges; the crossing case
            # supplies the interior-interior witness.
            candidates = []
            for p, q, r in ((x,z,w), (y,z,w), (z,x,y), (w,x,y)):
                edge = r-q
                closest = q + np.clip(np.dot(p-q, edge)/max(np.dot(edge,edge),1e-30),0,1)*edge
                candidates.append((p, closest))
            e, f = y-x, w-z
            determinant = e[0]*f[1]-e[1]*f[0]
            if abs(determinant) > 1e-14:
                delta = z-x
                s = (delta[0]*f[1]-delta[1]*f[0])/determinant
                t = (delta[0]*e[1]-delta[1]*e[0])/determinant
                if 0 <= s <= 1 and 0 <= t <= 1:
                    candidates.append((x+s*e, z+t*f))
            for p, q in candidates:
                distance = np.dot(p-q,p-q)
                if distance < best:
                    best, tangent = distance, .5*(p+q)
    return basis @ tangent + .5*(pa.min()+pb.max())*normal


def _convex_pair_contact(body, state, collider, other_body, other_state, other_collider):
    """Complete polyhedral SAT, including edge-edge axes and containment.

    Returns B-to-A separating normal, translation depth including authored
    margins, and a supporting-feature contact point. Authored meshes must be convex.
    """
    va, na, ea = _convex_features(collider)
    vb, nb, eb = _convex_features(other_collider)
    ra, rb = state['rotation'], other_state['rotation']
    a = body.center + state['translation'] + (va-body.center) @ ra.T
    b = other_body.center + other_state['translation'] + (vb-other_body.center) @ rb.T
    margin = body.collision_margin + other_body.collision_margin
    if np.any(a.max(axis=0)+margin < b.min(axis=0)) or np.any(b.max(axis=0)+margin < a.min(axis=0)):
        return None
    cross_axes = np.cross((ea @ ra.T)[:,None,:], (eb @ rb.T)[None,:,:]).reshape(-1,3)
    axes = _unique_axes(np.vstack((na @ ra.T, nb @ rb.T, cross_axes)))
    if not len(axes):
        return None
    # Shift projections near the origin to keep tolerance independent of
    # a scene's world-space translation.
    origin = .5*(a.mean(axis=0)+b.mean(axis=0))
    pa, pb = (a-origin) @ axes.T, (b-origin) @ axes.T
    forward = pb.max(axis=0)-pa.min(axis=0)+margin
    backward = pa.max(axis=0)-pb.min(axis=0)+margin
    depth = np.minimum(forward, backward)
    if np.any(depth <= 1e-10):
        return None
    index = int(np.argmin(depth))
    normal = axes[index] if forward[index] <= backward[index] else -axes[index]
    point = _support_contact_point(a-origin,b-origin,normal)+origin
    return normal, float(depth[index]), point


def _inverse_translation_mass(body):
    joint = getattr(body, 'joint_type', 'free')
    if joint == 'fixed':
        return np.zeros((3,3))
    if joint == 'prismatic':
        axis = np.asarray(body.joint_axis, dtype=np.float64)
        axis = axis/np.linalg.norm(axis)
        return np.outer(axis,axis)/body.mass
    return np.eye(3)/body.mass


def _inverse_rotation_inertia(body, state):
    if getattr(body, 'joint_type', 'free') in {'fixed', 'prismatic'}:
        return np.zeros((3,3))
    return np.linalg.inv(world_inertia(body,state))


def resolve_body_contacts(
    scene, colliders, body_states, collider_translations, collider_velocities,
    collider_rotations, collider_centers, collider_angular_velocities,
    iterations=2,
):
    """Resolve convex dynamic bodies against static, kinematic and dynamic meshes.

    Convex body pairs use complete polyhedral SAT, not sampled surface tests.
    Static/kinematic meshes retain finite-feature sampled contacts. Equal-and-opposite
    impulses are applied immediately when the other collider is dynamic; hinge
    reaction impulses are returned to the caller. Additional bounded sweeps
    recover deep/multiple contacts, with early exit at sub-micrometre corrections.
    """
    collider_by_name = {collider.name: collider for collider in colliders}
    angular_reactions = {}
    contacts = 0

    for sweep in range(max(32, int(iterations))):
        largest_correction = 0.0
        for path, state in body_states.items():
            body = scene.dynamic_bodies[path]
            if getattr(body, "joint_type", "free") == "fixed":
                continue
            body_material = scene.collider_materials.get(path, {})
            samples, contact_radius = _contact_samples(body, state)

            for other_path, collider in collider_by_name.items():
                if other_path == path:
                    continue
                if frozenset((path, other_path)) in getattr(scene, "joint_collision_exclusions", ()):
                    continue
                # A dynamic pair is solved only once.
                if (other_path in body_states and path > other_path
                        and getattr(scene.dynamic_bodies[other_path], "joint_type", "free") != "fixed"):
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
                posed_other = body_states.get(other_path)
                if posed_other is not None:
                    # Contact iterations change poses in place. The dictionaries
                    # passed by the integrator are snapshots from before those
                    # corrections and must not be used for another dynamic body.
                    other_translation = posed_other['translation']
                    other_rotation = posed_other['rotation']
                    other_center = scene.dynamic_bodies[other_path].center
                inv_translation = _inverse_translation_mass(body)
                other_inv_translation = (np.zeros((3,3)) if posed_other is None else
                                         _inverse_translation_mass(scene.dynamic_bodies[other_path]))
                convex_pair = (posed_other is not None and body.collision_shape == 'convex'
                               and scene.dynamic_bodies[other_path].collision_shape == 'convex'
                               and path in collider_by_name)
                if convex_pair:
                    contact = _convex_pair_contact(
                        body, state, collider_by_name[path], scene.dynamic_bodies[other_path],
                        posed_other, collider)
                    if contact is None:
                        continue
                    normal, depth, point = contact
                else:
                    relative = samples - other_translation[None, :] - other_center[None, :]
                    query = other_center[None, :] + relative @ other_rotation
                    distance, normal_local = _oriented_surface_query(query, collider.triangles)
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
                    point = samples[deepest] - normal * effective_radius
                separation_mass = float(normal @ (inv_translation+other_inv_translation) @ normal)
                if separation_mass > 1e-14:
                    shift = (inv_translation @ normal) * (depth/separation_mass)
                    other_shift = -(other_inv_translation @ normal) * (depth/separation_mass)
                    largest_correction = max(largest_correction, np.linalg.norm(shift), np.linalg.norm(other_shift))
                    state['translation'] += shift
                    samples += shift
                    if posed_other is not None:
                        posed_other['translation'] += other_shift
                    if convex_pair:
                        point += .5*(shift+other_shift)
                    else:
                        point += shift
                center = body.center + state["translation"]
                lever = point - center

                other_state = body_states.get(other_path)
                if (other_state is not None
                        and getattr(scene.dynamic_bodies[other_path], "joint_type", "free") == "fixed"):
                    other_state = None
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

                inv_inertia = _inverse_rotation_inertia(body, state)
                denominator = float(normal @ inv_translation @ normal) + float(np.dot(
                    normal, np.cross(inv_inertia @ np.cross(lever, normal), lever)
                ))
                other_inv_inertia = None
                if other_state is not None:
                    other_body = scene.dynamic_bodies[other_path]
                    other_inv_inertia = _inverse_rotation_inertia(other_body, other_state)
                    denominator += float(normal @ other_inv_translation @ normal) + float(np.dot(
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
                if denominator < 1e-14:
                    continue
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
                    tangent_denominator = float(tangent_direction @ inv_translation @ tangent_direction) + float(np.dot(
                        tangent_direction,
                        np.cross(
                            inv_inertia @ np.cross(lever, tangent_direction), lever
                        ),
                    ))
                    if other_state is not None:
                        other_body = scene.dynamic_bodies[other_path]
                        tangent_denominator += float(tangent_direction @ other_inv_translation @ tangent_direction) + float(np.dot(
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

                state["velocity"] += inv_translation @ impulse
                state["omega"] += inv_inertia @ np.cross(lever, impulse)
                reaction = -impulse
                reaction_torque = np.cross(wall_lever, reaction)
                if other_state is not None:
                    other_body = scene.dynamic_bodies[other_path]
                    other_state["velocity"] += other_inv_translation @ reaction
                    other_state["omega"] += other_inv_inertia @ reaction_torque
                elif other_path in scene.hinges:
                    angular_reactions[other_path] = (
                        angular_reactions.get(other_path, np.zeros(3)) + reaction_torque
                    )

        if largest_correction < 1e-7 and sweep+1 >= min(2, max(1, int(iterations))):
            break

    return contacts, angular_reactions
