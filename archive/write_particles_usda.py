from pxr import Usd, UsdGeom, Gf
import numpy as np

INPUT_USD = "TankTest.usda"
OUTPUT_USD = "TankTest_particles.usda"
CSV_FILE = "particles.csv"

PARTICLE_RADIUS = 0.01


stage = Usd.Stage.Open(INPUT_USD)

points_np = np.loadtxt(
    CSV_FILE,
    delimiter=",",
    skiprows=1
)

points_prim = UsdGeom.Points.Define(
    stage,
    "/root/SPHParticles"
)

usd_points = [
    Gf.Vec3f(float(x), float(y), float(z))
    for x, y, z in points_np
]

points_prim.CreatePointsAttr(usd_points)

# Width = diameter
points_prim.CreateWidthsAttr(
    [PARTICLE_RADIUS * 2.0] * len(usd_points)
)

stage.GetRootLayer().Export(OUTPUT_USD)

print(f"Skrev {len(usd_points)} partiklar till {OUTPUT_USD}")