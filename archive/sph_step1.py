import time
import math
import numpy as np

# ============================================================
# GraSPH - STEP 1
# Neighbor search + kernel density
# ============================================================

PARTICLE_FILE = "particles.csv"

DP = 0.020          # Particle spacing [m]
SUPPORT = 0.040     # Kernel support radius [m]
RHO0 = 1000.0       # Rest density [kg/m3]

REPORT_INTERVAL = 1.0


# ------------------------------------------------------------
# Wendland C2 kernel, 3D
#
# SUPPORT = actual cutoff radius.
# q = r / SUPPORT
# W = 0 for q >= 1
# ------------------------------------------------------------

def kernel_wendland(r):
    q = r / SUPPORT

    if q >= 1.0:
        return 0.0

    alpha = 21.0 / (2.0 * math.pi * SUPPORT**3)

    return alpha * (1.0 - q)**4 * (4.0 * q + 1.0)


# ------------------------------------------------------------
# Spatial hashing
# Cell size = support radius
# ------------------------------------------------------------

def cell_index(p):
    return (
        math.floor(p[0] / SUPPORT),
        math.floor(p[1] / SUPPORT),
        math.floor(p[2] / SUPPORT)
    )


def build_grid(pos):

    grid = {}

    for i, p in enumerate(pos):
        c = cell_index(p)

        if c not in grid:
            grid[c] = []

        grid[c].append(i)

    return grid


# ------------------------------------------------------------
# Find neighbours
# ------------------------------------------------------------

def find_neighbors(pos, grid):

    n = len(pos)

    neighbors = [[] for _ in range(n)]

    start = time.perf_counter()
    last_report = start

    for i, p in enumerate(pos):

        cx, cy, cz = cell_index(p)

        # Only 27 surrounding cells required
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):

                    cell = (cx + dx, cy + dy, cz + dz)

                    if cell not in grid:
                        continue

                    for j in grid[cell]:

                        if j == i:
                            continue

                        d = pos[j] - p
                        r2 = np.dot(d, d)

                        if r2 < SUPPORT * SUPPORT:
                            neighbors[i].append(j)

        now = time.perf_counter()

        if now - last_report >= REPORT_INTERVAL:

            done = i + 1
            elapsed = now - start
            rate = done / elapsed
            remaining = (n - done) / rate if rate > 0 else 0

            print(
                f"[neighbors] "
                f"{done:5d}/{n} "
                f"| {100.0 * done/n:5.1f}% "
                f"| {rate:8.1f} particles/s "
                f"| ETA {remaining:5.1f}s"
            )

            last_report = now

    return neighbors


# ------------------------------------------------------------
# Kernel sums
# ------------------------------------------------------------

def calculate_kernel_sums(pos, neighbors):

    n = len(pos)
    sums = np.zeros(n)

    start = time.perf_counter()
    last_report = start

    w0 = kernel_wendland(0.0)

    for i in range(n):

        # Particle contributes to itself
        s = w0

        pi = pos[i]

        for j in neighbors[i]:

            r = np.linalg.norm(pos[j] - pi)
            s += kernel_wendland(r)

        sums[i] = s

        now = time.perf_counter()

        if now - last_report >= REPORT_INTERVAL:

            done = i + 1
            elapsed = now - start
            rate = done / elapsed
            remaining = (n - done) / rate if rate > 0 else 0

            print(
                f"[density ] "
                f"{done:5d}/{n} "
                f"| {100.0 * done/n:5.1f}% "
                f"| ETA {remaining:5.1f}s"
            )

            last_report = now

    return sums


# ============================================================
# MAIN
# ============================================================

def main():

    total_start = time.perf_counter()

    print()
    print("========================================")
    print("GraSPH STEP 1")
    print("NEIGHBOR SEARCH + KERNEL DENSITY")
    print("========================================")
    print()

    # --------------------------------------------------------
    # Load particles
    # --------------------------------------------------------

    print("Loading particles...")

    pos = np.loadtxt(
        PARTICLE_FILE,
        delimiter=",",
        skiprows=1
    )

    n = len(pos)

    print(f"Particles : {n}")
    print(f"Spacing   : {DP:.4f} m")
    print(f"Support   : {SUPPORT:.4f} m")
    print(f"rho0      : {RHO0:.1f} kg/m3")
    print()

    # --------------------------------------------------------
    # Spatial grid
    # --------------------------------------------------------

    print("Building spatial grid...")

    t0 = time.perf_counter()

    grid = build_grid(pos)

    print(
        f"Grid cells: {len(grid)} "
        f"({time.perf_counter()-t0:.3f} s)"
    )

    # --------------------------------------------------------
    # Neighbours
    # --------------------------------------------------------

    print()
    print("Searching neighbours...")

    t0 = time.perf_counter()

    neighbors = find_neighbors(pos, grid)

    neighbor_count = np.array(
        [len(x) for x in neighbors]
    )

    print(
        f"Neighbour search complete "
        f"({time.perf_counter()-t0:.3f} s)"
    )

    print()
    print("Neighbour statistics:")
    print(f"  min    : {neighbor_count.min()}")
    print(f"  mean   : {neighbor_count.mean():.2f}")
    print(f"  median : {np.median(neighbor_count):.1f}")
    print(f"  max    : {neighbor_count.max()}")

    # --------------------------------------------------------
    # Kernel density
    # --------------------------------------------------------

    print()
    print("Calculating kernel sums...")

    kernel_sum = calculate_kernel_sums(
        pos,
        neighbors
    )

    # --------------------------------------------------------
    # Calibrate effective particle mass from bulk particles
    #
    # Use particles with nearly maximum neighbour count.
    # This compensates for discrete kernel sampling.
    # --------------------------------------------------------

    max_neighbors = neighbor_count.max()

    bulk_limit = max_neighbors - 2

    bulk_mask = neighbor_count >= bulk_limit

    if np.sum(bulk_mask) < 10:
        bulk_mask = neighbor_count >= np.percentile(
            neighbor_count,
            80
        )

    bulk_kernel = kernel_sum[bulk_mask]

    particle_mass = RHO0 / bulk_kernel.mean()

    rho = particle_mass * kernel_sum

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print()
    print("========================================")
    print("RESULT")
    print("========================================")

    print(
        f"Bulk threshold       : "
        f">= {bulk_limit} neighbours"
    )

    print(
        f"Bulk particles       : "
        f"{np.sum(bulk_mask)} / {n} "
        f"({100*np.mean(bulk_mask):.1f}%)"
    )

    print(
        f"Effective mass       : "
        f"{particle_mass:.8f} kg"
    )

    print()
    print("Density [kg/m3]:")

    print(f"  min    : {rho.min():.2f}")
    print(f"  mean   : {rho.mean():.2f}")
    print(f"  median : {np.median(rho):.2f}")
    print(f"  max    : {rho.max():.2f}")

    print()
    print("Bulk density [kg/m3]:")

    print(f"  min    : {rho[bulk_mask].min():.2f}")
    print(f"  mean   : {rho[bulk_mask].mean():.2f}")
    print(f"  max    : {rho[bulk_mask].max():.2f}")

    # --------------------------------------------------------
    # Save diagnostics
    # --------------------------------------------------------

    output = np.column_stack(
        (
            pos,
            neighbor_count,
            kernel_sum,
            rho
        )
    )

    np.savetxt(
        "sph_step1.csv",
        output,
        delimiter=",",
        header="x,y,z,neighbors,kernel_sum,rho",
        comments=""
    )

    elapsed = time.perf_counter() - total_start

    print()
    print("Saved sph_step1.csv")

    print()
    print("========================================")
    print(
        f"STEP 1 COMPLETE | wall time {elapsed:.3f} s"
    )
    print("========================================")


if __name__ == "__main__":
    main()