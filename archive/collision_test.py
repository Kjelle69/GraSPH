import math
import time
import numpy as np

from pxr import Usd, UsdGeom, Gf


# ============================================================
# GraSPH - GENERIC COLLISION SMOKE TEST
#
# OpenUSD Mesh -> triangle colliders
# Fluid particles -> gravity -> geometric collision
#
# Ingen SPH ännu.
# ============================================================

USD_FILE = "TankTest.usda"
PARTICLE_FILE = "particles.csv"

OUTPUT_USD = "collision_test.usda"
OUTPUT_NPZ = "collision_test.npz"

DP = 0.020

# Kollisionsradie något mindre än halva spacing
PARTICLE_RADIUS = 0.009

GRAVITY = np.array([0.0, 0.0, -9.81], dtype=np.float64)

DT = 0.001
SIM_TIME = 0.30

# Spatial hash för trianglar
GRID_CELL_SIZE = 0.040

# Några iterationspass behövs vid hörn
COLLISION_ITERATIONS = 3

# 0 = helt oelastisk normal collision
RESTITUTION = 0.0

# 1 = behåll tangentiell hastighet
TANGENTIAL_KEEP = 0.98

REPORT_INTERVAL = 1.0


# ============================================================
# USD MESH READING
# ============================================================

def prim_is_fluid_volume(prim):
    """
    FluidVolume är authoring-geometri, inte collider.
    Vi kontrollerar hela pathen så både Xform och Mesh fångas.
    """
    path = str(prim.GetPath())
    parts = path.split("/")

    return "FluidVolume" in parts


def triangulate_mesh_world(stage, prim):
    """
    Läs en UsdGeom.Mesh och returnera world-space trianglar.
    """

    mesh = UsdGeom.Mesh(prim)

    points = mesh.GetPointsAttr().Get()
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()

    if points is None:
        return []

    xform_cache = UsdGeom.XformCache(
        Usd.TimeCode.Default()
    )

    matrix = xform_cache.GetLocalToWorldTransform(prim)

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
            np.array(
                [wp[0], wp[1], wp[2]],
                dtype=np.float64
            )
        )

    triangles = []

    offset = 0

    for count_raw in counts:
        count = int(count_raw)

        face = indices[offset:offset + count]
        offset += count

        if count < 3:
            continue

        # Fan triangulation:
        # 0,1,2,3 -> (0,1,2) + (0,2,3)
        i0 = int(face[0])

        for k in range(1, count - 1):
            i1 = int(face[k])
            i2 = int(face[k + 1])

            a = world_points[i0]
            b = world_points[i1]
            c = world_points[i2]

            triangles.append(
                np.array([a, b, c], dtype=np.float64)
            )

    return triangles


def load_colliders(stage):

    triangles = []
    sources = []

    print("Scanning USD scene for colliders...")

    for prim in stage.Traverse():

        if not prim.IsA(UsdGeom.Mesh):
            continue

        if prim_is_fluid_volume(prim):
            print(
                f"  SKIP fluid source : {prim.GetPath()}"
            )
            continue

        mesh_triangles = triangulate_mesh_world(
            stage,
            prim
        )

        if len(mesh_triangles) == 0:
            continue

        print(
            f"  COLLIDER          : "
            f"{prim.GetPath()} "
            f"({len(mesh_triangles)} triangles)"
        )

        source_id = len(sources)

        sources.append(str(prim.GetPath()))

        for tri in mesh_triangles:
            triangles.append(
                (tri, source_id)
            )

    return triangles, sources


# ============================================================
# TRIANGLE SPATIAL HASH
# ============================================================

class TriangleGrid:

    def __init__(
        self,
        triangles,
        cell_size,
        padding
    ):
        self.triangles = triangles
        self.cell_size = cell_size
        self.padding = padding

        self.cells = {}

        self.build()

    def cell_index(self, p):

        return (
            math.floor(p[0] / self.cell_size),
            math.floor(p[1] / self.cell_size),
            math.floor(p[2] / self.cell_size)
        )

    def build(self):

        self.cells = {}

        for tri_index, (tri, source_id) in enumerate(
            self.triangles
        ):

            bmin = tri.min(axis=0) - self.padding
            bmax = tri.max(axis=0) + self.padding

            cmin = self.cell_index(bmin)
            cmax = self.cell_index(bmax)

            for ix in range(cmin[0], cmax[0] + 1):
                for iy in range(cmin[1], cmax[1] + 1):
                    for iz in range(cmin[2], cmax[2] + 1):

                        cell = (ix, iy, iz)

                        self.cells.setdefault(
                            cell,
                            []
                        ).append(tri_index)

    def query(self, p):

        c = self.cell_index(p)

        result = set()

        # Ta även grannceller för säkerhets skull.
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):

                    cell = (
                        c[0] + dx,
                        c[1] + dy,
                        c[2] + dz
                    )

                    items = self.cells.get(cell)

                    if items is not None:
                        result.update(items)

        return result


# ============================================================
# CLOSEST POINT ON TRIANGLE
#
# Standard closest-point test.
# ============================================================

def closest_point_triangle(p, a, b, c):

    ab = b - a
    ac = c - a
    ap = p - a

    d1 = np.dot(ab, ap)
    d2 = np.dot(ac, ap)

    if d1 <= 0.0 and d2 <= 0.0:
        return a

    bp = p - b

    d3 = np.dot(ab, bp)
    d4 = np.dot(ac, bp)

    if d3 >= 0.0 and d4 <= d3:
        return b

    vc = d1 * d4 - d3 * d2

    if (
        vc <= 0.0
        and d1 >= 0.0
        and d3 <= 0.0
    ):
        v = d1 / (d1 - d3)

        return a + v * ab

    cp = p - c

    d5 = np.dot(ab, cp)
    d6 = np.dot(ac, cp)

    if d6 >= 0.0 and d5 <= d6:
        return c

    vb = d5 * d2 - d1 * d6

    if (
        vb <= 0.0
        and d2 >= 0.0
        and d6 <= 0.0
    ):
        w = d2 / (d2 - d6)

        return a + w * ac

    va = d3 * d6 - d5 * d4

    if (
        va <= 0.0
        and (d4 - d3) >= 0.0
        and (d5 - d6) >= 0.0
    ):
        w = (
            (d4 - d3)
            /
            (
                (d4 - d3)
                + (d5 - d6)
            )
        )

        return b + w * (c - b)

    denom = 1.0 / (va + vb + vc)

    v = vb * denom
    w = vc * denom

    return a + ab * v + ac * w


# ============================================================
# PARTICLE / MESH COLLISION
# ============================================================

def collide_particle(
    p,
    v,
    grid,
    radius
):

    contacts = 0

    candidate_indices = grid.query(p)

    for tri_index in candidate_indices:

        tri, source_id = grid.triangles[tri_index]

        a = tri[0]
        b = tri[1]
        c = tri[2]

        closest = closest_point_triangle(
            p,
            a,
            b,
            c
        )

        delta = p - closest

        dist2 = np.dot(delta, delta)

        if dist2 >= radius * radius:
            continue

        dist = math.sqrt(
            max(dist2, 0.0)
        )

        if dist > 1e-10:

            normal = delta / dist

        else:

            # Sällsynt fall:
            # partikeln exakt på triangelytan.
            normal = np.cross(
                b - a,
                c - a
            )

            nlen = np.linalg.norm(normal)

            if nlen < 1e-12:
                continue

            normal /= nlen

        penetration = radius - dist

        # Flytta ut partikeln från ytan
        p = p + normal * penetration

        # Normalhastighet
        vn = np.dot(
            v,
            normal
        )

        if vn < 0.0:

            v_normal = vn * normal
            v_tangent = v - v_normal

            # Kollisionsrespons
            v = (
                -RESTITUTION * v_normal
                + TANGENTIAL_KEEP * v_tangent
            )

        contacts += 1

    return p, v, contacts


# ============================================================
# USD OUTPUT
# ============================================================

def write_particles_to_usd(
    input_stage,
    positions
):

    points_prim = UsdGeom.Points.Define(
        input_stage,
        "/root/CollisionTestParticles"
    )

    usd_points = [
        Gf.Vec3f(
            float(p[0]),
            float(p[1]),
            float(p[2])
        )
        for p in positions
    ]

    points_prim.CreatePointsAttr(
        usd_points
    )

    points_prim.CreateWidthsAttr(
        [DP] * len(positions)
    )

    input_stage.GetRootLayer().Export(
        OUTPUT_USD
    )


# ============================================================
# PROGRESS
# ============================================================

def fmt_time(seconds):

    seconds = max(
        0,
        int(seconds)
    )

    m, s = divmod(
        seconds,
        60
    )

    return f"{m:02d}:{s:02d}"


# ============================================================
# MAIN
# ============================================================

def main():

    wall_start = time.perf_counter()

    print()
    print("========================================")
    print("GraSPH COLLISION TEST")
    print("GENERIC OpenUSD COLLIDERS")
    print("========================================")
    print()

    # --------------------------------------------------------
    # USD
    # --------------------------------------------------------

    stage = Usd.Stage.Open(
        USD_FILE
    )

    if not stage:
        raise RuntimeError(
            f"Kunde inte öppna {USD_FILE}"
        )

    triangles, sources = load_colliders(
        stage
    )

    print()
    print(
        f"Collider meshes    : {len(sources)}"
    )

    print(
        f"Collider triangles : {len(triangles)}"
    )

    if len(triangles) == 0:
        raise RuntimeError(
            "Inga collider triangles hittades."
        )

    # --------------------------------------------------------
    # Build collision broadphase
    # --------------------------------------------------------

    print()
    print("Building triangle spatial grid...")

    t0 = time.perf_counter()

    collision_grid = TriangleGrid(
        triangles,
        GRID_CELL_SIZE,
        PARTICLE_RADIUS
    )

    print(
        f"Grid cells: {len(collision_grid.cells)} "
        f"| {time.perf_counter() - t0:.4f} s"
    )

    # --------------------------------------------------------
    # Particles
    # --------------------------------------------------------

    pos = np.loadtxt(
        PARTICLE_FILE,
        delimiter=",",
        skiprows=1
    )

    vel = np.zeros_like(pos)

    initial_pos = pos.copy()

    print()
    print(f"Particles       : {len(pos)}")
    print(f"Particle radius : {PARTICLE_RADIUS:.4f} m")
    print(f"dt              : {DT:.6f} s")
    print(f"Simulation time : {SIM_TIME:.3f} s")

    total_steps = int(
        math.ceil(SIM_TIME / DT)
    )

    print(f"Steps           : {total_steps}")

    # --------------------------------------------------------
    # Simulation
    # --------------------------------------------------------

    print()
    print("Starting gravity + collision test...")
    print()

    sim_start = time.perf_counter()
    last_report = sim_start

    total_contacts = 0

    for step in range(total_steps):

        # Gravity
        vel += GRAVITY * DT

        # Predict
        pos += vel * DT

        contacts_this_step = 0

        # Resolve contacts.
        # Flera pass hjälper i hörn där flera trianglar möts.
        for iteration in range(
            COLLISION_ITERATIONS
        ):

            for i in range(len(pos)):

                p, v, contacts = collide_particle(
                    pos[i],
                    vel[i],
                    collision_grid,
                    PARTICLE_RADIUS
                )

                pos[i] = p
                vel[i] = v

                contacts_this_step += contacts

        total_contacts += contacts_this_step

        # --------------------------------------------
        # Ticker
        # --------------------------------------------

        now = time.perf_counter()

        if (
            now - last_report >= REPORT_INTERVAL
            or step == total_steps - 1
        ):

            elapsed = now - sim_start

            completed = step + 1

            progress = (
                completed / total_steps
            )

            steps_per_sec = (
                completed / elapsed
                if elapsed > 0.0
                else 0.0
            )

            eta = (
                elapsed
                * (1.0 - progress)
                / progress
                if progress > 0.0
                else 0.0
            )

            speeds = np.linalg.norm(
                vel,
                axis=1
            )

            sim_time = completed * DT

            print(
                f"[{fmt_time(elapsed)}] "
                f"step {completed}/{total_steps} "
                f"| {progress*100:5.1f}% "
                f"| sim {sim_time:.3f}/{SIM_TIME:.3f}s "
                f"| {steps_per_sec:6.1f} step/s "
                f"| ETA {fmt_time(eta)}"
            )

            print(
                f"    contacts={contacts_this_step} "
                f"| vmax={speeds.max():.3f} m/s "
                f"| zmin={pos[:,2].min():.4f} "
                f"| zmax={pos[:,2].max():.4f}"
            )

            last_report = now

    # --------------------------------------------------------
    # Save raw result
    # --------------------------------------------------------

    np.savez_compressed(
        OUTPUT_NPZ,
        initial_pos=initial_pos,
        final_pos=pos,
        velocity=vel,
        dt=DT,
        sim_time=SIM_TIME
    )

    # --------------------------------------------------------
    # USD post
    # --------------------------------------------------------

    print()
    print("Writing USD result...")

    write_particles_to_usd(
        stage,
        pos
    )

    wall_elapsed = (
        time.perf_counter()
        - wall_start
    )

    print()
    print("========================================")
    print("COLLISION TEST COMPLETE")
    print("========================================")
    print(f"Total contacts : {total_contacts}")
    print(f"Wall time      : {wall_elapsed:.3f} s")
    print(f"Saved          : {OUTPUT_NPZ}")
    print(f"Saved          : {OUTPUT_USD}")
    print()


if __name__ == "__main__":
    main()