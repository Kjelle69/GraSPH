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


def build_colliders(scene, voxel_size=0.010, margin=0.040):
    colliders = []

    print()
    print("========================================")
    print("GraSPH COLLIDER BUILD")
    print("========================================")

    for mesh in scene.colliders:
        colliders.append(
            DistanceFieldCollider(
                name=mesh.path,
                triangles=mesh.triangles,
                voxel_size=voxel_size,
                margin=margin,
            )
        )

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
    iterations=2,
    collider_translations=None,
    collider_velocities=None,
):
    """Geometric non-penetration only.

    SPH wall physics is handled separately by the boundary module.

    v0.18 adds rigid *translation* support for kinematic colliders without
    rebuilding their distance fields.  The UDF remains in each collider's
    local/reference coordinates and the query is performed at

        x_local = x_world - translation.

    For collision response, normal velocity is evaluated relative to the
    collider velocity.  Rotation is intentionally not handled yet.
    """
    total_contacts = 0
    collider_translations = collider_translations or {}
    collider_velocities = collider_velocities or {}

    for _ in range(iterations):
        for collider in colliders:
            translation = np.asarray(
                collider_translations.get(collider.name, (0.0, 0.0, 0.0)),
                dtype=np.float64,
            )
            wall_velocity = np.asarray(
                collider_velocities.get(collider.name, (0.0, 0.0, 0.0)),
                dtype=np.float64,
            )

            # Translation-only local query. UDF normals are unchanged by a
            # pure translation, so they are already valid in world space.
            distance, normal = collider.query(pos - translation[None, :])
            contact = distance < particle_radius

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

            penetration = particle_radius - distance[ids]
            pos[ids] += n * penetration[:, None]

            # Collision response in the moving wall frame.
            rel_v = vel[ids] - wall_velocity[None, :]
            vn = np.einsum("ij,ij->i", rel_v, n)
            incoming = vn < 0.0

            if np.any(incoming):
                hit_ids = ids[incoming]
                hit_n = n[incoming]
                hit_vn = vn[incoming]
                hit_rel = rel_v[incoming]

                v_normal = hit_vn[:, None] * hit_n
                v_tangent = hit_rel - v_normal

                vel[hit_ids] = (
                    wall_velocity[None, :]
                    - restitution * v_normal
                    + tangential_keep * v_tangent
                )

            total_contacts += len(ids)

    return total_contacts


