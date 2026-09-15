import math
import time
import numpy as np

from pxr import Usd, UsdGeom, Gf


# ============================================================
# GraSPH - DISTANCE FIELD COLLISION TEST
#
# Generic OpenUSD meshes -> unsigned distance fields
# Particle collision entirely vectorized during simulation.
# ============================================================

USD_FILE = "TankTest.usda"
PARTICLE_FILE = "particles.csv"

OUTPUT_USD = "df_collision_test.usda"
OUTPUT_NPZ = "df_collision_test.npz"

DP = 0.020
PARTICLE_RADIUS = 0.009

# Distance-field voxel size.
# dp/2 is a good first test.
VOXEL_SIZE = 0.010

# Field only needs to exist close to each collider.
FIELD_MARGIN = 0.040

GRAVITY = np.array(
    [0.0, 0.0, -9.81],
    dtype=np.float64
)

DT = 0.001
SIM_TIME = 0.30

COLLISION_ITERATIONS = 2

RESTITUTION = 0.0
TANGENTIAL_KEEP = 0.98

REPORT_INTERVAL = 1.0

# Number of grid points processed simultaneously
DF_BUILD_BATCH = 20000


# ============================================================
# USD
# ============================================================

def prim_is_fluid_volume(prim):
    return "FluidVolume" in str(prim.GetPath()).split("/")


def read_mesh_world(stage, prim):

    mesh = UsdGeom.Mesh(prim)

    points = mesh.GetPointsAttr().Get()
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()

    xform_cache = UsdGeom.XformCache(
        Usd.TimeCode.Default()
    )

    matrix = xform_cache.GetLocalToWorldTransform(
        prim
    )

    world_points = []

    for p in points:

        wp = matrix.Transform(
            Gf.Vec3d(
                float(p[0]),
                float(p[1]),
                float(p[2])
            )
        )

        world_points.append(
            [wp[0], wp[1], wp[2]]
        )

    world_points = np.asarray(
        world_points,
        dtype=np.float64
    )

    triangles = []

    offset = 0

    for count_raw in counts:

        count = int(count_raw)

        face = indices[
            offset:offset + count
        ]

        offset += count

        if count < 3:
            continue

        i0 = int(face[0])

        for k in range(1, count - 1):

            i1 = int(face[k])
            i2 = int(face[k + 1])

            triangles.append(
                np.array(
                    [
                        world_points[i0],
                        world_points[i1],
                        world_points[i2],
                    ],
                    dtype=np.float64
                )
            )

    return np.asarray(triangles)


def load_collider_meshes(stage):

    meshes = []

    print("Scanning USD scene...")

    for prim in stage.Traverse():

        if not prim.IsA(UsdGeom.Mesh):
            continue

        if prim_is_fluid_volume(prim):

            print(
                f"  FLUID SOURCE : {prim.GetPath()}"
            )

            continue

        triangles = read_mesh_world(
            stage,
            prim
        )

        if len(triangles) == 0:
            continue

        print(
            f"  COLLIDER     : "
            f"{prim.GetPath()} "
            f"({len(triangles)} triangles)"
        )

        meshes.append(
            (
                str(prim.GetPath()),
                triangles
            )
        )

    return meshes


# ============================================================
# CLOSEST POINT - VECTORIZED
#
# P = N x 3 array of query points.
# Returns closest point on triangle for every query.
# ============================================================

def closest_point_triangle_batch(
    P,
    a,
    b,
    c
):

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

    unresolved = np.ones(
        len(P),
        dtype=bool
    )

    # --------------------------------------------------------
    # Vertex A
    # --------------------------------------------------------

    mask = (
        unresolved
        & (d1 <= 0.0)
        & (d2 <= 0.0)
    )

    Q[mask] = a
    unresolved[mask] = False

    # --------------------------------------------------------
    # Vertex B
    # --------------------------------------------------------

    mask = (
        unresolved
        & (d3 >= 0.0)
        & (d4 <= d3)
    )

    Q[mask] = b
    unresolved[mask] = False

    # --------------------------------------------------------
    # Edge AB
    # --------------------------------------------------------

    vc = d1 * d4 - d3 * d2

    mask = (
        unresolved
        & (vc <= 0.0)
        & (d1 >= 0.0)
        & (d3 <= 0.0)
    )

    denom = d1 - d3

    safe = mask & (
        np.abs(denom) > 1e-15
    )

    v = np.zeros(len(P))

    v[safe] = (
        d1[safe]
        /
        denom[safe]
    )

    Q[safe] = (
        a
        + v[safe, None] * ab
    )

    unresolved[safe] = False

    # --------------------------------------------------------
    # Vertex C
    # --------------------------------------------------------

    mask = (
        unresolved
        & (d6 >= 0.0)
        & (d5 <= d6)
    )

    Q[mask] = c
    unresolved[mask] = False

    # --------------------------------------------------------
    # Edge AC
    # --------------------------------------------------------

    vb = d5 * d2 - d1 * d6

    mask = (
        unresolved
        & (vb <= 0.0)
        & (d2 >= 0.0)
        & (d6 <= 0.0)
    )

    denom = d2 - d6

    safe = mask & (
        np.abs(denom) > 1e-15
    )

    w = np.zeros(len(P))

    w[safe] = (
        d2[safe]
        /
        denom[safe]
    )

    Q[safe] = (
        a
        + w[safe, None] * ac
    )

    unresolved[safe] = False

    # --------------------------------------------------------
    # Edge BC
    # --------------------------------------------------------

    va = d3 * d6 - d5 * d4

    mask = (
        unresolved
        & (va <= 0.0)
        & ((d4 - d3) >= 0.0)
        & ((d5 - d6) >= 0.0)
    )

    denom = (
        (d4 - d3)
        + (d5 - d6)
    )

    safe = mask & (
        np.abs(denom) > 1e-15
    )

    w = np.zeros(len(P))

    w[safe] = (
        (d4[safe] - d3[safe])
        /
        denom[safe]
    )

    Q[safe] = (
        b
        + w[safe, None] * (c - b)
    )

    unresolved[safe] = False

    # --------------------------------------------------------
    # Face interior
    # --------------------------------------------------------

    mask = unresolved

    denom = (
        va + vb + vc
    )

    safe = mask & (
        np.abs(denom) > 1e-15
    )

    v = np.zeros(len(P))
    w = np.zeros(len(P))

    v[safe] = (
        vb[safe]
        /
        denom[safe]
    )

    w[safe] = (
        vc[safe]
        /
        denom[safe]
    )

    Q[safe] = (
        a
        + v[safe, None] * ab
        + w[safe, None] * ac
    )

    unresolved[safe] = False

    # Degenerate fallback
    if np.any(unresolved):
        Q[unresolved] = a

    return Q


# ============================================================
# DISTANCE FIELD
# ============================================================

class DistanceFieldCollider:

    def __init__(
        self,
        name,
        triangles,
        voxel_size,
        margin
    ):

        self.name = name
        self.triangles = triangles

        self.voxel = voxel_size
        self.margin = margin

        self.origin = None
        self.shape = None

        self.distance = None
        self.gradient = None

        self.build()

    # --------------------------------------------------------

    def build(self):

        t0 = time.perf_counter()

        tri_points = self.triangles.reshape(
            -1,
            3
        )

        bmin = (
            tri_points.min(axis=0)
            - self.margin
        )

        bmax = (
            tri_points.max(axis=0)
            + self.margin
        )

        self.origin = bmin

        size = bmax - bmin

        shape = (
            np.ceil(
                size / self.voxel
            ).astype(int)
            + 1
        )

        # Need at least 2 cells per direction
        shape = np.maximum(
            shape,
            2
        )

        self.shape = tuple(
            int(x) for x in shape
        )

        nx, ny, nz = self.shape

        print()
        print(
            f"Building DF: {self.name}"
        )

        print(
            f"  shape      : "
            f"{nx} x {ny} x {nz}"
        )

        print(
            f"  voxels     : "
            f"{nx * ny * nz:,}"
        )

        print(
            f"  voxel size : "
            f"{self.voxel:.4f} m"
        )

        # ----------------------------------------------------
        # Grid coordinates
        # ----------------------------------------------------

        ix, iy, iz = np.meshgrid(
            np.arange(nx),
            np.arange(ny),
            np.arange(nz),
            indexing="ij"
        )

        grid_points = np.column_stack(
            (
                ix.ravel(),
                iy.ravel(),
                iz.ravel()
            )
        ).astype(np.float64)

        grid_points *= self.voxel
        grid_points += self.origin

        total = len(grid_points)

        dist_all = np.empty(
            total,
            dtype=np.float32
        )

        grad_all = np.empty(
            (total, 3),
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Process field in batches
        # ----------------------------------------------------

        for start in range(
            0,
            total,
            DF_BUILD_BATCH
        ):

            end = min(
                start + DF_BUILD_BATCH,
                total
            )

            P = grid_points[
                start:end
            ]

            best_d2 = np.full(
                len(P),
                np.inf,
                dtype=np.float64
            )

            best_delta = np.zeros(
                (len(P), 3),
                dtype=np.float64
            )

            best_normal = np.zeros(
                (len(P), 3),
                dtype=np.float64
            )

            for tri in self.triangles:

                a = tri[0]
                b = tri[1]
                c = tri[2]

                Q = closest_point_triangle_batch(
                    P,
                    a,
                    b,
                    c
                )

                delta = P - Q

                d2 = np.einsum(
                    "ij,ij->i",
                    delta,
                    delta
                )

                better = d2 < best_d2

                if np.any(better):

                    best_d2[better] = d2[better]

                    best_delta[better] = (
                        delta[better]
                    )

                    # Triangle normal fallback
                    n = np.cross(
                        b - a,
                        c - a
                    )

                    nlen = np.linalg.norm(n)

                    if nlen > 1e-15:
                        n = n / nlen
                    else:
                        n = np.array(
                            [0.0, 0.0, 1.0]
                        )

                    best_normal[better] = n

            distance = np.sqrt(
                np.maximum(
                    best_d2,
                    0.0
                )
            )

            gradient = np.zeros_like(
                best_delta
            )

            nonzero = (
                distance > 1e-10
            )

            gradient[nonzero] = (
                best_delta[nonzero]
                /
                distance[nonzero, None]
            )

            # Exactly on surface:
            # use triangle normal.
            gradient[~nonzero] = (
                best_normal[~nonzero]
            )

            dist_all[start:end] = (
                distance.astype(
                    np.float32
                )
            )

            grad_all[start:end] = (
                gradient.astype(
                    np.float32
                )
            )

        self.distance = dist_all.reshape(
            self.shape
        )

        self.gradient = grad_all.reshape(
            self.shape + (3,)
        )

        elapsed = (
            time.perf_counter() - t0
        )

        memory_mb = (
            self.distance.nbytes
            + self.gradient.nbytes
        ) / 1024**2

        print(
            f"  build time  : "
            f"{elapsed:.3f} s"
        )

        print(
            f"  memory      : "
            f"{memory_mb:.2f} MB"
        )

    # ========================================================
    # TRILINEAR SAMPLE
    # ========================================================

    def query(self, positions):

        n = len(positions)

        distance = np.full(
            n,
            np.inf,
            dtype=np.float64
        )

        normal = np.zeros(
            (n, 3),
            dtype=np.float64
        )

        g = (
            (positions - self.origin)
            / self.voxel
        )

        base = np.floor(
            g
        ).astype(np.int64)

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

        idx = np.where(valid)[0]

        b = base[idx]
        f = frac[idx]

        x = b[:, 0]
        y = b[:, 1]
        z = b[:, 2]

        fx = f[:, 0]
        fy = f[:, 1]
        fz = f[:, 2]

        # Eight interpolation weights
        weights = [
            (1-fx)*(1-fy)*(1-fz),
            fx*(1-fy)*(1-fz),
            (1-fx)*fy*(1-fz),
            fx*fy*(1-fz),
            (1-fx)*(1-fy)*fz,
            fx*(1-fy)*fz,
            (1-fx)*fy*fz,
            fx*fy*fz,
        ]

        offsets = [
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (1, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (0, 1, 1),
            (1, 1, 1),
        ]

        d = np.zeros(
            len(idx),
            dtype=np.float64
        )

        grad = np.zeros(
            (len(idx), 3),
            dtype=np.float64
        )

        for w, off in zip(
            weights,
            offsets
        ):

            ox, oy, oz = off

            d += (
                w
                * self.distance[
                    x + ox,
                    y + oy,
                    z + oz
                ]
            )

            grad += (
                w[:, None]
                * self.gradient[
                    x + ox,
                    y + oy,
                    z + oz
                ]
            )

        # Normalize interpolated gradient
        glen = np.linalg.norm(
            grad,
            axis=1
        )

        good = glen > 1e-12

        grad[good] /= (
            glen[good, None]
        )

        distance[idx] = d
        normal[idx] = grad

        return distance, normal


# ============================================================
# COLLISION
# ============================================================

def resolve_collisions(
    pos,
    vel,
    colliders
):

    total_contacts = 0

    for collider in colliders:

        distance, normal = (
            collider.query(pos)
        )

        contact = (
            distance
            < PARTICLE_RADIUS
        )

        if not np.any(contact):
            continue

        ids = np.where(contact)[0]

        n = normal[ids]

        valid_normal = (
            np.linalg.norm(
                n,
                axis=1
            )
            > 0.5
        )

        ids = ids[valid_normal]
        n = n[valid_normal]

        if len(ids) == 0:
            continue

        penetration = (
            PARTICLE_RADIUS
            - distance[ids]
        )

        # Position correction
        pos[ids] += (
            n * penetration[:, None]
        )

        # Velocity collision response
        vn = np.einsum(
            "ij,ij->i",
            vel[ids],
            n
        )

        incoming = vn < 0.0

        if np.any(incoming):

            hit_ids = ids[incoming]
            hit_n = n[incoming]
            hit_vn = vn[incoming]

            v_normal = (
                hit_vn[:, None]
                * hit_n
            )

            v_tangent = (
                vel[hit_ids]
                - v_normal
            )

            vel[hit_ids] = (
                -RESTITUTION * v_normal
                + TANGENTIAL_KEEP
                * v_tangent
            )

        total_contacts += len(ids)

    return total_contacts


# ============================================================
# USD OUTPUT
# ============================================================

def write_result(
    stage,
    positions
):

    prim = UsdGeom.Points.Define(
        stage,
        "/root/DFCollisionParticles"
    )

    usd_points = [
        Gf.Vec3f(
            float(p[0]),
            float(p[1]),
            float(p[2])
        )
        for p in positions
    ]

    prim.CreatePointsAttr(
        usd_points
    )

    prim.CreateWidthsAttr(
        [DP] * len(positions)
    )

    stage.GetRootLayer().Export(
        OUTPUT_USD
    )


# ============================================================
# MAIN
# ============================================================

def main():

    program_start = (
        time.perf_counter()
    )

    print()
    print("========================================")
    print("GraSPH DISTANCE FIELD COLLISION TEST")
    print("========================================")
    print()

    stage = Usd.Stage.Open(
        USD_FILE
    )

    if not stage:
        raise RuntimeError(
            f"Kunde inte öppna {USD_FILE}"
        )

    mesh_data = load_collider_meshes(
        stage
    )

    if len(mesh_data) == 0:
        raise RuntimeError(
            "Inga colliders hittades."
        )

    # --------------------------------------------------------
    # Build distance fields
    # --------------------------------------------------------

    print()
    print("Building distance fields...")

    colliders = []

    for name, triangles in mesh_data:

        colliders.append(
            DistanceFieldCollider(
                name,
                triangles,
                VOXEL_SIZE,
                FIELD_MARGIN
            )
        )

    # --------------------------------------------------------
    # Load particles
    # --------------------------------------------------------

    pos = np.loadtxt(
        PARTICLE_FILE,
        delimiter=",",
        skiprows=1
    )

    vel = np.zeros_like(pos)

    initial_pos = pos.copy()

    total_steps = int(
        math.ceil(
            SIM_TIME / DT
        )
    )

    print()
    print("Simulation:")
    print(f"  particles : {len(pos)}")
    print(f"  dt        : {DT:.6f} s")
    print(f"  steps     : {total_steps}")
    print()

    # --------------------------------------------------------
    # Simulation
    # --------------------------------------------------------

    sim_start = time.perf_counter()
    last_report = sim_start

    total_contacts = 0

    for step in range(total_steps):

        vel += GRAVITY * DT
        pos += vel * DT

        contacts = 0

        for _ in range(
            COLLISION_ITERATIONS
        ):

            contacts += (
                resolve_collisions(
                    pos,
                    vel,
                    colliders
                )
            )

        total_contacts += contacts

        now = time.perf_counter()

        if (
            now - last_report
            >= REPORT_INTERVAL
            or step == total_steps - 1
        ):

            elapsed = (
                now - sim_start
            )

            completed = step + 1

            rate = (
                completed / elapsed
                if elapsed > 0
                else 0.0
            )

            progress = (
                completed / total_steps
            )

            eta = (
                elapsed
                * (1-progress)
                / progress
                if progress > 0
                else 0.0
            )

            speeds = np.linalg.norm(
                vel,
                axis=1
            )

            print(
                f"step "
                f"{completed:4d}/{total_steps} "
                f"| {progress*100:5.1f}% "
                f"| {rate:8.1f} step/s "
                f"| ETA {eta:5.1f}s"
            )

            print(
                f"    contacts={contacts} "
                f"| vmax={speeds.max():.3f} "
                f"| zmin={pos[:,2].min():.4f} "
                f"| zmax={pos[:,2].max():.4f}"
            )

            last_report = now

    sim_elapsed = (
        time.perf_counter()
        - sim_start
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    np.savez_compressed(
        OUTPUT_NPZ,
        initial_pos=initial_pos,
        final_pos=pos,
        velocity=vel,
        dt=DT,
        sim_time=SIM_TIME
    )

    print()
    print("Writing USD...")

    write_result(
        stage,
        pos
    )

    total_elapsed = (
        time.perf_counter()
        - program_start
    )

    print()
    print("========================================")
    print("DISTANCE FIELD TEST COMPLETE")
    print("========================================")
    print(
        f"Simulation time : "
        f"{sim_elapsed:.3f} s"
    )
    print(
        f"Simulation rate : "
        f"{total_steps/sim_elapsed:.1f} step/s"
    )
    print(
        f"Total contacts  : "
        f"{total_contacts}"
    )
    print(
        f"Total wall time : "
        f"{total_elapsed:.3f} s"
    )
    print()
    print(f"Saved {OUTPUT_NPZ}")
    print(f"Saved {OUTPUT_USD}")


if __name__ == "__main__":
    main()