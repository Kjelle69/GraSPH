import time
import numpy as np


# ============================================================
# GraSPH Collision
#
# Generic mesh collider using an unsigned distance field.
#
# No OpenUSD dependency here.
# No fluid-fluid SPH equations here.
# ============================================================


def closest_point_triangle_batch(P, a, b, c):
    ab = b - a
    ac = c - a

    AP = P - a
    d1 = AP @ ab
    d2 = AP @ ac

    BP = P - b
    d3 = BP @ ab
    d4 = BP @ ac

    CP = P - c
    d5 = CP @ ab
    d6 = CP @ ac

    Q = np.zeros_like(P)
    unresolved = np.ones(len(P), dtype=bool)

    mask = unresolved & (d1 <= 0.0) & (d2 <= 0.0)
    Q[mask] = a
    unresolved[mask] = False

    mask = unresolved & (d3 >= 0.0) & (d4 <= d3)
    Q[mask] = b
    unresolved[mask] = False

    vc = d1 * d4 - d3 * d2
    mask = unresolved & (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    denom = d1 - d3
    safe = mask & (np.abs(denom) > 1e-15)
    v = np.zeros(len(P))
    v[safe] = d1[safe] / denom[safe]
    Q[safe] = a + v[safe, None] * ab
    unresolved[safe] = False

    mask = unresolved & (d6 >= 0.0) & (d5 <= d6)
    Q[mask] = c
    unresolved[mask] = False

    vb = d5 * d2 - d1 * d6
    mask = unresolved & (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    denom = d2 - d6
    safe = mask & (np.abs(denom) > 1e-15)
    w = np.zeros(len(P))
    w[safe] = d2[safe] / denom[safe]
    Q[safe] = a + w[safe, None] * ac
    unresolved[safe] = False

    va = d3 * d6 - d5 * d4
    mask = (
        unresolved
        & (va <= 0.0)
        & ((d4 - d3) >= 0.0)
        & ((d5 - d6) >= 0.0)
    )
    denom = (d4 - d3) + (d5 - d6)
    safe = mask & (np.abs(denom) > 1e-15)
    w = np.zeros(len(P))
    w[safe] = (d4[safe] - d3[safe]) / denom[safe]
    Q[safe] = b + w[safe, None] * (c - b)
    unresolved[safe] = False

    denom = va + vb + vc
    safe = unresolved & (np.abs(denom) > 1e-15)
    v = np.zeros(len(P))
    w = np.zeros(len(P))
    v[safe] = vb[safe] / denom[safe]
    w[safe] = vc[safe] / denom[safe]
    Q[safe] = a + v[safe, None] * ab + w[safe, None] * ac
    unresolved[safe] = False

    if np.any(unresolved):
        Q[unresolved] = a

    return Q


class DistanceFieldCollider:
    def __init__(
        self,
        name,
        triangles,
        voxel_size=0.010,
        margin=0.040,
        build_batch=20000,
    ):
        self.name = name
        self.triangles = np.asarray(triangles, dtype=np.float64)
        self.voxel_size = float(voxel_size)
        self.margin = float(margin)
        self.build_batch = int(build_batch)

        self.origin = None
        self.shape = None
        self.distance = None
        self.gradient = None

        self.build()

    def build(self):
        start = time.perf_counter()

        tri_points = self.triangles.reshape(-1, 3)
        bmin = tri_points.min(axis=0) - self.margin
        bmax = tri_points.max(axis=0) + self.margin

        self.origin = bmin
        size = bmax - bmin

        shape = np.ceil(size / self.voxel_size).astype(int) + 1
        shape = np.maximum(shape, 2)
        self.shape = tuple(int(x) for x in shape)

        nx, ny, nz = self.shape
        voxel_count = nx * ny * nz

        print()
        print("Building distance field:")
        print(f"  collider   : {self.name}")
        print(f"  shape      : {nx} x {ny} x {nz}")
        print(f"  voxels     : {voxel_count:,}")
        print(f"  voxel size : {self.voxel_size:.4f} m")

        ix, iy, iz = np.meshgrid(
            np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"
        )

        grid_points = np.column_stack(
            (ix.ravel(), iy.ravel(), iz.ravel())
        ).astype(np.float64)
        grid_points *= self.voxel_size
        grid_points += self.origin

        total = len(grid_points)
        dist_all = np.empty(total, dtype=np.float32)
        grad_all = np.empty((total, 3), dtype=np.float32)

        for batch_start in range(0, total, self.build_batch):
            batch_end = min(batch_start + self.build_batch, total)
            P = grid_points[batch_start:batch_end]

            best_d2 = np.full(len(P), np.inf, dtype=np.float64)
            best_delta = np.zeros((len(P), 3), dtype=np.float64)
            best_normal = np.zeros((len(P), 3), dtype=np.float64)

            for tri in self.triangles:
                a, b, c = tri
                Q = closest_point_triangle_batch(P, a, b, c)
                delta = P - Q
                d2 = np.einsum("ij,ij->i", delta, delta)
                better = d2 < best_d2

                if not np.any(better):
                    continue

                best_d2[better] = d2[better]
                best_delta[better] = delta[better]

                n = np.cross(b - a, c - a)
                nlen = np.linalg.norm(n)
                if nlen > 1e-15:
                    n = n / nlen
                else:
                    n = np.array([0.0, 0.0, 1.0])
                best_normal[better] = n

            distance = np.sqrt(np.maximum(best_d2, 0.0))
            gradient = np.zeros_like(best_delta)

            nonzero = distance > 1e-10
            gradient[nonzero] = (
                best_delta[nonzero] / distance[nonzero, None]
            )
            gradient[~nonzero] = best_normal[~nonzero]

            dist_all[batch_start:batch_end] = distance.astype(np.float32)
            grad_all[batch_start:batch_end] = gradient.astype(np.float32)

        self.distance = dist_all.reshape(self.shape)
        self.gradient = grad_all.reshape(self.shape + (3,))

        elapsed = time.perf_counter() - start
        memory_mb = (self.distance.nbytes + self.gradient.nbytes) / (1024**2)

        print(f"  build time : {elapsed:.3f} s")
        print(f"  memory     : {memory_mb:.2f} MB")

    def query(self, positions):
        n = len(positions)
        distance = np.full(n, np.inf, dtype=np.float64)
        normal = np.zeros((n, 3), dtype=np.float64)

        g = (positions - self.origin) / self.voxel_size
        base = np.floor(g).astype(np.int64)
        frac = g - base

        nx, ny, nz = self.shape
        valid = (
            (base[:, 0] >= 0)
            & (base[:, 1] >= 0)
            & (base[:, 2] >= 0)
            & (base[:, 0] < nx - 1)
            & (base[:, 1] < ny - 1)
            & (base[:, 2] < nz - 1)
        )

        if not np.any(valid):
            return distance, normal

        ids = np.where(valid)[0]
        b = base[ids]
        f = frac[ids]

        x, y, z = b[:, 0], b[:, 1], b[:, 2]
        fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]

        weights = (
            (1 - fx) * (1 - fy) * (1 - fz),
            fx * (1 - fy) * (1 - fz),
            (1 - fx) * fy * (1 - fz),
            fx * fy * (1 - fz),
            (1 - fx) * (1 - fy) * fz,
            fx * (1 - fy) * fz,
            (1 - fx) * fy * fz,
            fx * fy * fz,
        )
        offsets = (
            (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
            (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1),
        )

        d = np.zeros(len(ids), dtype=np.float64)
        grad = np.zeros((len(ids), 3), dtype=np.float64)

        for weight, offset in zip(weights, offsets):
            ox, oy, oz = offset
            d += weight * self.distance[x + ox, y + oy, z + oz]
            grad += weight[:, None] * self.gradient[x + ox, y + oy, z + oz]

        glen = np.linalg.norm(grad, axis=1)
        good = glen > 1e-12
        grad[good] /= glen[good, None]

        distance[ids] = d
        normal[ids] = grad
        return distance, normal

    def query_exact(self, positions):
        """Return closest-triangle distance and normal for a small point set.

        The cached field is intentionally coarse for speed, but interpolation
        can blend normals from opposite sides of thin meshes.  Contact points
        are few enough that an exact refinement is both cheap and much safer.
        """
        positions = np.asarray(positions, dtype=np.float64)
        distance2 = np.full(len(positions), np.inf, dtype=np.float64)
        delta_best = np.zeros((len(positions), 3), dtype=np.float64)
        face_best = np.zeros((len(positions), 3), dtype=np.float64)

        for triangle in self.triangles:
            a, b, c = triangle
            closest = closest_point_triangle_batch(positions, a, b, c)
            delta = positions - closest
            candidate_distance2 = np.einsum("ij,ij->i", delta, delta)
            better = candidate_distance2 < distance2
            if not np.any(better):
                continue
            distance2[better] = candidate_distance2[better]
            delta_best[better] = delta[better]
            face_normal = np.cross(b - a, c - a)
            length = np.linalg.norm(face_normal)
            if length > 1.0e-15:
                face_normal = face_normal / length
            else:
                face_normal = np.array((0.0, 0.0, 1.0))
            face_best[better] = face_normal

        distance = np.sqrt(np.maximum(distance2, 0.0))
        normal = face_best
        separated = distance > 1.0e-12
        normal[separated] = delta_best[separated] / distance[separated, None]
        return distance, normal


def build_colliders(scene, voxel_size=0.010, margin=0.040):
    colliders = []

    print()
    print("========================================")
    print("GraSPH COLLIDER BUILD")
    print("========================================")

    for mesh in scene.colliders:
        collider = DistanceFieldCollider(
            name=mesh.path,
            triangles=mesh.triangles,
            voxel_size=voxel_size,
            margin=margin,
        )
        material = getattr(scene, "collider_materials", {}).get(mesh.path, {})
        collider.surface_friction = float(material.get("friction", 0.45))
        collider.surface_restitution = float(material.get("restitution", 0.10))
        collider.rolling_friction = float(material.get("rolling_friction", 0.02))
        colliders.append(collider)

    print()
    print(f"Distance-field colliders: {len(colliders)}")
    return colliders


def resolve_collisions(
    pos,
    vel,
    colliders,
    particle_radius,
    restitution=0.0,
    tangential_keep=1.0,
    friction_coefficient=None,
    iterations=2,
    collider_translations=None,
    collider_velocities=None,
    collider_rotations=None,
    collider_centers=None,
    collider_angular_velocities=None,
    particle_mass=None,
    particle_angular_velocities=None,
    particle_inertia=None,
    particle_rolling_friction=None,
    previous_positions=None,
    return_angular_impulses=False,
    return_body_impulses=False,
):
    """Geometric non-penetration and velocity response.

    SPH wall physics is handled separately by the boundary module.

    v0.18 adds rigid *translation* support for kinematic colliders without
    rebuilding their distance fields.  The UDF remains in each collider's
    local/reference coordinates and the query is performed at

        x_local = x_world - translation.

    Rigid rotations are queried by transforming particles back into each
    collider's reference frame.  When requested, the equal-and-opposite
    angular impulse delivered to each collider is returned as well.

    ``friction_coefficient`` selects a Coulomb impulse capped by the normal
    collision impulse. When omitted, ``tangential_keep`` preserves the legacy
    SPH collision response.
    """
    total_contacts = 0
    collider_translations = collider_translations or {}
    collider_velocities = collider_velocities or {}
    collider_rotations = collider_rotations or {}
    collider_centers = collider_centers or {}
    collider_angular_velocities = collider_angular_velocities or {}
    angular_impulses = {}
    linear_impulses = {}
    radii = np.asarray(particle_radius, dtype=np.float64)
    per_particle_radius = radii.ndim > 0
    masses = None if particle_mass is None else np.asarray(particle_mass, dtype=np.float64)
    angular = particle_angular_velocities
    inertias = (
        None if particle_inertia is None
        else np.asarray(particle_inertia, dtype=np.float64)
    )
    restitution_values = np.asarray(restitution, dtype=np.float64)
    friction_values = (
        None if friction_coefficient is None
        else np.asarray(friction_coefficient, dtype=np.float64)
    )
    particle_rolling_values = (
        None if particle_rolling_friction is None
        else np.asarray(particle_rolling_friction, dtype=np.float64)
    )

    previous_positions = (
        None if previous_positions is None
        else np.asarray(previous_positions, dtype=np.float64)
    )

    for iteration in range(iterations):
        for collider in colliders:
            translation = np.asarray(
                collider_translations.get(collider.name, (0.0, 0.0, 0.0)),
                dtype=np.float64,
            )
            wall_velocity = np.asarray(
                collider_velocities.get(collider.name, (0.0, 0.0, 0.0)),
                dtype=np.float64,
            )
            rotation = np.asarray(
                collider_rotations.get(collider.name, np.eye(3)), dtype=np.float64
            )
            center = np.asarray(
                collider_centers.get(collider.name, (0.0, 0.0, 0.0)), dtype=np.float64
            )
            angular_velocity = np.asarray(
                collider_angular_velocities.get(collider.name, (0.0, 0.0, 0.0)),
                dtype=np.float64,
            )

            relative = pos - translation[None, :] - center[None, :]
            query_pos = center[None, :] + relative @ rotation
            distance, normal_local = collider.query(query_pos)
            can_refine = (
                previous_positions is not None
                and hasattr(collider, "query_exact")
            )
            refinement_radius = radii + float(getattr(collider, "voxel_size", 0.0))
            refine = can_refine & (distance < refinement_radius)
            if np.any(refine):
                exact_distance, exact_normal = collider.query_exact(query_pos[refine])
                distance[refine] = exact_distance
                normal_local[refine] = exact_normal
            normal = normal_local @ rotation.T
            if previous_positions is not None and iteration == 0:
                previous_relative = (
                    previous_positions - translation[None, :] - center[None, :]
                )
                previous_query = center[None, :] + previous_relative @ rotation
                previous_distance, previous_normal_local = collider.query(previous_query)
                previous_refine = can_refine & (previous_distance < refinement_radius)
                if np.any(previous_refine):
                    exact_distance, exact_normal = collider.query_exact(
                        previous_query[previous_refine]
                    )
                    previous_distance[previous_refine] = exact_distance
                    previous_normal_local[previous_refine] = exact_normal
                previous_normal = previous_normal_local @ rotation.T
                travel = pos - previous_positions
                normal_travel = np.einsum(
                    "ij,ij->i", travel, previous_normal
                )
                projected_distance = previous_distance - np.abs(normal_travel)
                previous_normal_length = np.linalg.norm(previous_normal, axis=1)
                swept = (
                    (previous_distance >= radii)
                    & (projected_distance < radii)
                    & (previous_normal_length > 0.5)
                )
                if np.any(swept):
                    # Unsigned fields do not guarantee which side their
                    # interpolated gradient points toward. At a swept entry,
                    # the separating normal must oppose the crossing motion.
                    wrong_way = np.einsum(
                        "ij,ij->i", travel, previous_normal
                    ) > 0.0
                    previous_normal = previous_normal.copy()
                    previous_normal[wrong_way] *= -1.0
                    distance = distance.copy()
                    normal = normal.copy()
                    distance[swept] = projected_distance[swept]
                    normal[swept] = previous_normal[swept]
            contact = distance < radii

            if not np.any(contact):
                continue

            ids = np.where(contact)[0]
            n = normal[ids]
            n_length = np.linalg.norm(n, axis=1)
            valid = n_length > 0.5

            ids = ids[valid]
            n = n[valid]

            if len(ids) == 0:
                continue

            contact_radii = radii[ids] if per_particle_radius else radii
            penetration = contact_radii - distance[ids]
            pos[ids] += n * penetration[:, None]

            # Collision response in the moving wall frame. The hinge center
            # follows any authored translation while the angular velocity is
            # evaluated at the corrected contact position.
            lever = pos[ids] - (center + translation)[None, :]
            contact_wall_velocity = (
                wall_velocity[None, :] + np.cross(angular_velocity[None, :], lever)
            )
            if angular is not None:
                contact_radii_for_velocity = (
                    radii[ids] if per_particle_radius
                    else np.full(len(ids), float(radii))
                )
                contact_arms = -n * contact_radii_for_velocity[:, None]
                particle_contact_velocity = (
                    vel[ids] + np.cross(angular[ids], contact_arms)
                )
                rel_v = particle_contact_velocity - contact_wall_velocity
            else:
                contact_arms = None
                rel_v = vel[ids] - contact_wall_velocity
            vn = np.einsum("ij,ij->i", rel_v, n)
            incoming = vn < 0.0

            if np.any(incoming):
                hit_ids = ids[incoming]
                hit_n = n[incoming]
                hit_vn = vn[incoming]
                hit_rel = rel_v[incoming]
                before = vel[hit_ids].copy()

                v_normal = hit_vn[:, None] * hit_n
                v_tangent = hit_rel - v_normal
                hit_restitution = (
                    restitution_values[hit_ids]
                    if restitution_values.ndim > 0 else restitution_values
                )
                if angular is not None and masses is not None and inertias is not None:
                    hit_mass = masses[hit_ids] if masses.ndim > 0 else np.full(
                        len(hit_ids), float(masses)
                    )
                    hit_inertia = inertias[hit_ids] if inertias.ndim > 0 else np.full(
                        len(hit_ids), float(inertias)
                    )
                    hit_radius = (
                        radii[hit_ids] if per_particle_radius
                        else np.full(len(hit_ids), float(radii))
                    )
                    hit_arm = contact_arms[incoming]
                    particle_mu = (
                        friction_values[hit_ids]
                        if friction_values is not None and friction_values.ndim > 0
                        else float(friction_values if friction_values is not None else 0.0)
                    )
                    wall_mu = float(getattr(collider, "surface_friction", 0.45))
                    combined_mu = np.sqrt(np.maximum(0.0, particle_mu * wall_mu))
                    wall_restitution = float(
                        getattr(collider, "surface_restitution", 0.10)
                    )
                    combined_restitution = np.sqrt(
                        np.maximum(0.0, hit_restitution * wall_restitution)
                    )
                    inv_mass = 1.0 / hit_mass
                    normal_impulse = (
                        -(1.0 + combined_restitution) * hit_vn / inv_mass
                    )
                    tangent_speed = np.linalg.norm(v_tangent, axis=1)
                    tangent_direction = v_tangent / np.maximum(
                        tangent_speed[:, None], 1.0e-12
                    )
                    tangent_effective_inverse_mass = (
                        inv_mass + hit_radius**2 / hit_inertia
                    )
                    tangent_impulse = np.minimum(
                        tangent_speed / tangent_effective_inverse_mass,
                        combined_mu * normal_impulse,
                    )
                    particle_impulse = (
                        normal_impulse[:, None] * hit_n
                        - tangent_impulse[:, None] * tangent_direction
                    )
                    vel[hit_ids] += particle_impulse * inv_mass[:, None]
                    angular[hit_ids] += (
                        np.cross(hit_arm, particle_impulse) / hit_inertia[:, None]
                    )

                    particle_rolling = (
                        particle_rolling_values[hit_ids]
                        if particle_rolling_values is not None
                        and particle_rolling_values.ndim > 0
                        else float(
                            particle_rolling_values
                            if particle_rolling_values is not None else 0.02
                        )
                    )
                    rolling_mu = np.sqrt(np.maximum(
                        0.0, particle_rolling
                        * float(getattr(collider, "rolling_friction", 0.02)),
                    ))
                    relative_omega = angular[hit_ids] - angular_velocity[None, :]
                    rolling_omega = (
                        relative_omega
                        - np.einsum("ij,ij->i", relative_omega, hit_n)[:, None] * hit_n
                    )
                    rolling_speed = np.linalg.norm(rolling_omega, axis=1)
                    max_angular_delta = (
                        rolling_mu * normal_impulse * hit_radius / hit_inertia
                    )
                    angular_delta = np.minimum(rolling_speed, max_angular_delta)
                    angular[hit_ids] -= (
                        angular_delta[:, None]
                        * rolling_omega / np.maximum(rolling_speed[:, None], 1.0e-12)
                    )
                elif friction_values is None:
                    vel[hit_ids] = (
                        contact_wall_velocity[incoming]
                        - hit_restitution[..., None] * v_normal
                        + tangential_keep * v_tangent
                    )
                else:
                    # Normal impulse changes vn to -e*vn. Coulomb friction may
                    # remove at most mu times that normal impulse; unlike the
                    # legacy tangential multiplier this is not repeatedly
                    # applied after the normal impact has already resolved.
                    hit_friction = (
                        friction_values[hit_ids]
                        if friction_values.ndim > 0 else friction_values
                    )
                    normal_delta = (
                        -(1.0 + hit_restitution) * hit_vn
                    )
                    tangent_speed = np.linalg.norm(v_tangent, axis=1)
                    tangent_direction = v_tangent / np.maximum(
                        tangent_speed[:, None], 1.0e-12
                    )
                    friction_delta = np.minimum(
                        tangent_speed,
                        np.maximum(0.0, hit_friction) * normal_delta,
                    )
                    new_relative = (
                        hit_rel
                        + normal_delta[:, None] * hit_n
                        - friction_delta[:, None] * tangent_direction
                    )
                    vel[hit_ids] = contact_wall_velocity[incoming] + new_relative
                if (return_angular_impulses or return_body_impulses) and particle_mass is not None:
                    hit_masses = masses[hit_ids] if masses.ndim > 0 else masses
                    fluid_impulse = hit_masses[..., None] * (vel[hit_ids] - before)
                    reaction = -fluid_impulse
                    linear_impulses[collider.name] = (
                        linear_impulses.get(collider.name, np.zeros(3))
                        + np.sum(reaction, axis=0)
                    )
                    contact_position = pos[hit_ids]
                    if contact_arms is not None:
                        contact_position = contact_position + contact_arms[incoming]
                    hit_lever = contact_position - (center + translation)[None, :]
                    impulse = np.sum(np.cross(hit_lever, reaction), axis=0)
                    angular_impulses[collider.name] = (
                        angular_impulses.get(collider.name, np.zeros(3)) + impulse
                    )

            total_contacts += len(ids)

    if return_body_impulses:
        return total_contacts, linear_impulses, angular_impulses
    if return_angular_impulses:
        return total_contacts, angular_impulses
    return total_contacts
