"""Small, reproducible float/sink check; does not alter Blender's scene.

py benchmarks/buoyancy_probe.py --time 0.2 --mass 0.5
py benchmarks/buoyancy_probe.py --time 0.2 --mass 8
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom
from grasph_solver import run_simulation, SolverSettings
from grasph_sph import SPHConfig


def box(stage, path, lo, hi, role):
    points = [(lo[0], lo[1], lo[2]), (hi[0], lo[1], lo[2]),
              (hi[0], hi[1], lo[2]), (lo[0], hi[1], lo[2]),
              (lo[0], lo[1], hi[2]), (hi[0], lo[1], hi[2]),
              (hi[0], hi[1], hi[2]), (lo[0], hi[1], hi[2])]
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
             (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
    mesh.CreateFaceVertexCountsAttr([4] * 6)
    mesh.CreateFaceVertexIndicesAttr([i for face in faces for i in face])
    prim = mesh.GetPrim()
    prim.CreateAttribute('grasph:role', Sdf.ValueTypeNames.String).Set(role)
    return prim


def make_scene(path, mass):
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    # Separate boxes leave a real cavity for the initially immersed body.
    sources = [((-0.2, -0.2, 0.04), (0.2, 0.2, 0.22)),
               ((-0.2, -0.2, 0.22), (-0.08, 0.2, 0.32)),
               ((0.08, -0.2, 0.22), (0.2, 0.2, 0.32)),
               ((-0.08, -0.2, 0.22), (0.08, -0.08, 0.32)),
               ((-0.08, 0.08, 0.22), (0.08, 0.2, 0.32))]
    for i, (lo, hi) in enumerate(sources):
        box(stage, f'/Fluid{i}', lo, hi, 'fluid')
    walls = [((-0.24, -0.24, -0.04), (0.24, 0.24, 0.0)),
             ((-0.24, -0.24, 0.0), (-0.2, 0.24, 0.5)),
             ((0.2, -0.24, 0.0), (0.24, 0.24, 0.5)),
             ((-0.2, -0.24, 0.0), (0.2, -0.2, 0.5)),
             ((-0.2, 0.2, 0.0), (0.2, 0.24, 0.5))]
    for i, (lo, hi) in enumerate(walls):
        box(stage, f'/Wall{i}', lo, hi, 'collider')
    driver = UsdGeom.Xform.Define(stage, '/Body')
    driver.AddTranslateOp().Set(Gf.Vec3d(0.0))
    driver.AddRotateXYZOp().Set(Gf.Vec3f(0.0))
    body = box(stage, '/Body/Geom', (-0.08, -0.08, 0.22),
               (0.08, 0.08, 0.38), 'collider')
    for name, kind, value in (
        ('dynamicBodyEnabled', Sdf.ValueTypeNames.Bool, True),
        ('bodyCenter', Sdf.ValueTypeNames.Double3, Gf.Vec3d(0, 0, 0.3)),
        ('mass', Sdf.ValueTypeNames.Double, mass),
        ('inertiaDiagonal', Sdf.ValueTypeNames.Double3, Gf.Vec3d(mass * 0.16**2 / 6)),
        ('collisionShape', Sdf.ValueTypeNames.String, 'convex')):
        body.CreateAttribute('grasph:' + name, kind).Set(value)
    stage.GetRootLayer().Save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--time', type=float, default=0.2)
    parser.add_argument('--mass', type=float, default=0.5)
    parser.add_argument('--c0', type=float, default=40.0)
    parser.add_argument('--wall-search', choices=['cached', 'rebuild'], default='cached')
    args = parser.parse_args()
    directory = Path('output/buoyancy_probe').resolve()
    directory.mkdir(parents=True, exist_ok=True)
    stem = f'body_{args.mass:g}kg_{args.wall_search}'
    source = directory / (stem + '_scene.usda')
    make_scene(source, args.mass)
    result = run_simulation(source, sph_cfg=SPHConfig(dp=0.04, c0=args.c0),
                            settings=SolverSettings(sim_time=args.time, output_fps=100,
                                output_dir=str(directory), output_stem=stem,
                                enable_solid_coupling=True, neighbor_workers=1,
                                wall_search=args.wall_search, report_interval=10))
    with np.load(result.npz_file) as data:
        translation = data['body_translations'][0]
        print(json.dumps(dict(mass_kg=args.mass, time_s=result.time,
                              initial_displaced_mass_kg=float(1000 * 0.16**2 * np.clip(result.surface_level - 0.22, 0, 0.16)),
                              final_body_dz_m=float(translation[-1, 2]),
                              max_body_travel_m=float(np.linalg.norm(translation, axis=1).max()),
                              steps=result.steps)))


if __name__ == '__main__':
    main()
