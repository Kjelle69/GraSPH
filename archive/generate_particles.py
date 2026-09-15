from pxr import Usd, UsdGeom
import numpy as np

USD_FILE = "TankTest.usda"
FLUID_PATH = "/root/FluidVolume"

PARTICLE_SPACING = 0.02   # 20 mm
MARGIN = 0.01             # 10 mm


def get_world_aligned_bounds(stage, path):
    prim = stage.GetPrimAtPath(path)
    if not prim:
        raise RuntimeError(f"Hittade inte prim: {path}")

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_]
    )

    bbox = bbox_cache.ComputeWorldBound(prim)
    aligned = bbox.ComputeAlignedBox()

    minimum = aligned.GetMin()
    maximum = aligned.GetMax()

    return np.array([minimum[0], minimum[1], minimum[2]]), np.array([maximum[0], maximum[1], maximum[2]])


def generate_grid_points(min_corner, max_corner, spacing):
    xs = np.arange(min_corner[0], max_corner[0] + 1e-9, spacing)
    ys = np.arange(min_corner[1], max_corner[1] + 1e-9, spacing)
    zs = np.arange(min_corner[2], max_corner[2] + 1e-9, spacing)

    points = []
    for x in xs:
        for y in ys:
            for z in zs:
                points.append((x, y, z))

    return np.array(points, dtype=np.float64)


def main():
    stage = Usd.Stage.Open(USD_FILE)
    if not stage:
        raise RuntimeError(f"Kunde inte öppna {USD_FILE}")

    bmin, bmax = get_world_aligned_bounds(stage, FLUID_PATH)

    print("========================================")
    print("FluidVolume bounds")
    print("========================================")
    print("Original min:", bmin)
    print("Original max:", bmax)
    print("Original size:", bmax - bmin)

    # Krymp spawnvolymen lite så vi inte lägger partiklar i väggarna
    spawn_min = bmin + MARGIN
    spawn_max = bmax - MARGIN

    print("\nSpawn bounds (med marginal):")
    print("Spawn min:", spawn_min)
    print("Spawn max:", spawn_max)
    print("Spawn size:", spawn_max - spawn_min)

    if np.any(spawn_max <= spawn_min):
        raise RuntimeError("Spawnvolymen blev ogiltig efter marginal. Minska MARGIN.")

    points = generate_grid_points(spawn_min, spawn_max, PARTICLE_SPACING)

    print("\n========================================")
    print("Partiklar")
    print("========================================")
    print("Particle spacing:", PARTICLE_SPACING)
    print("Antal partiklar:", len(points))

    if len(points) > 0:
        print("Första 10 partiklar:")
        for p in points[:10]:
            print(" ", p)

    # Spara till csv för enkel inspektion
    np.savetxt("particles.csv", points, delimiter=",", header="x,y,z", comments="")
    print("\nSparade particles.csv")


if __name__ == "__main__":
    main()