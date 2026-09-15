import time
import numpy as np

from grasph_core import (
    SPHConfig,
    WendlandC2,
    SpatialGrid,
    compute_density,
)


PARTICLE_FILE = "particles.csv"


def main():
    start = time.perf_counter()

    print()
    print("========================================")
    print("GraSPH CORE TEST")
    print("WENDLAND C2 + PHYSICAL PARTICLE MASS")
    print("========================================")

    cfg = SPHConfig()

    print()
    print("Configuration:")
    print(f"  rho0          : {cfg.rho0:.1f} kg/m3")
    print(f"  dp            : {cfg.dp:.4f} m")
    print(f"  h             : {cfg.h:.4f} m")
    print(f"  support       : {cfg.support:.4f} m")
    print(f"  particle mass : {cfg.mass:.6f} kg")

    print()
    print("Loading particles...")

    pos = np.loadtxt(
        PARTICLE_FILE,
        delimiter=",",
        skiprows=1
    )

    n = len(pos)

    print(f"Particles: {n}")

    # --------------------------------------------------------
    # Kernel
    # --------------------------------------------------------

    kernel = WendlandC2(cfg.h)

    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    print()
    print("Building spatial grid...")

    t0 = time.perf_counter()

    grid = SpatialGrid(cfg.support)
    grid.build(pos)

    print(
        f"Cells: {len(grid.cells)} "
        f"| {time.perf_counter() - t0:.4f} s"
    )

    # --------------------------------------------------------
    # Neighbours
    # --------------------------------------------------------

    print()
    print("Finding neighbours...")

    t0 = time.perf_counter()

    neighbors = grid.find_neighbors(
        pos,
        cfg.support
    )

    counts = np.array(
        [len(x) for x in neighbors]
    )

    print(
        f"Done in {time.perf_counter() - t0:.4f} s"
    )

    print()
    print("Neighbour statistics:")
    print(f"  min    : {counts.min()}")
    print(f"  mean   : {counts.mean():.2f}")
    print(f"  median : {np.median(counts):.1f}")
    print(f"  max    : {counts.max()}")

    # --------------------------------------------------------
    # Density
    # --------------------------------------------------------

    print()
    print("Computing density...")

    t0 = time.perf_counter()

    rho = compute_density(
        pos,
        neighbors,
        kernel,
        cfg.mass
    )

    print(
        f"Done in {time.perf_counter() - t0:.4f} s"
    )

    print()
    print("Density statistics:")
    print(f"  min    : {rho.min():.2f}")
    print(f"  mean   : {rho.mean():.2f}")
    print(f"  median : {np.median(rho):.2f}")
    print(f"  max    : {rho.max():.2f}")

    # --------------------------------------------------------
    # Bulk analysis
    # --------------------------------------------------------

    # Perfect cubic bulk at this h/dp ratio should have
    # ~80 neighbours.
    bulk_mask = counts >= 75

    bulk_count = np.sum(bulk_mask)

    print()
    print("Bulk analysis:")
    print(
        f"  particles >=75 neighbours: "
        f"{bulk_count}"
    )

    if bulk_count > 0:
        bulk_rho = rho[bulk_mask]

        print(
            f"  rho min  : {bulk_rho.min():.2f}"
        )
        print(
            f"  rho mean : {bulk_rho.mean():.2f}"
        )
        print(
            f"  rho max  : {bulk_rho.max():.2f}"
        )

    # --------------------------------------------------------
    # Save diagnostics
    # --------------------------------------------------------

    data = np.column_stack(
        (
            pos,
            counts,
            rho
        )
    )

    np.savetxt(
        "density_test.csv",
        data,
        delimiter=",",
        header="x,y,z,neighbors,rho",
        comments=""
    )

    elapsed = time.perf_counter() - start

    print()
    print("Saved density_test.csv")

    print()
    print("========================================")
    print(
        f"TEST COMPLETE | wall time {elapsed:.3f} s"
    )
    print("========================================")


if __name__ == "__main__":
    main()