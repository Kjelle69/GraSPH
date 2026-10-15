"""Fast fixed-body equilibrium check with optional old corner classification.

py benchmarks/still_water_probe.py SCENE.usdc --time 2
py benchmarks/still_water_probe.py SCENE.usdc --time 2 --legacy-corners
Coarser spacing is diagnostic; it is not a fine-resolution validation.
"""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import grasph_free_surface as surface
from grasph_scene import load_scene
from grasph_solver import SolverSettings, run_simulation
from grasph_sph import SPHConfig


def legacy_covariance(pos, triangles, normals, support):
    """Original finite plane-footprint classification, for A/B only."""
    cov = np.zeros((len(pos), 3, 3))
    count = np.zeros(len(pos), dtype=np.int16)
    distance = np.full(len(pos), np.inf)
    for tri, normal in zip(triangles, normals):
        box = np.all((pos >= tri.min(axis=0) - support - 1e-12)
                     & (pos <= tri.max(axis=0) + support + 1e-12), axis=1)
        ids = np.flatnonzero(box)
        signed = (tri[0] - pos[ids]) @ normal
        near = np.abs(signed) <= support + 1e-12
        ids, signed = ids[near], signed[near]
        inside = surface._point_in_triangle_batch(pos[ids] + signed[:, None] * normal, tri, tol=1e-9)
        hits = ids[inside]
        cov[hits] += np.outer(normal, normal)
        count[hits] += 1
        distance[hits] = np.minimum(distance[hits], np.abs(signed[inside]))
    return cov, count, distance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scene')
    parser.add_argument('--time', type=float, default=2)
    parser.add_argument('--dp', type=float, default=.1)
    parser.add_argument('--legacy-corners', action='store_true')
    args = parser.parse_args()
    scene = load_scene(args.scene)
    if any(b.joint_type != 'fixed' for b in scene.dynamic_bodies.values()):
        raise ValueError('Use the fixed-body scene for this probe')
    if args.legacy_corners:
        surface.compute_geometry_normal_covariance = legacy_covariance
    name = 'legacy' if args.legacy_corners else 'corners_fixed'
    result = run_simulation(args.scene, sph_cfg=SPHConfig(dp=args.dp, c0=40),
                            settings=SolverSettings(sim_time=args.time, dt_max=.001,
                                output_dir='output/still_water_probe', output_stem=name,
                                output_fps=10, enable_solid_coupling=True,
                                neighbor_workers=1, report_interval=30))
    with np.load(result.npz_file) as data:
        for target in (0, .5, 1, 1.5, 2):
            f = int(np.abs(data['times'] - target).argmin())
            velocity = data['velocity'][f]
            main = np.isfinite(velocity).all(axis=1) & ~data['detached_mask'][f].astype(bool)
            speed = np.linalg.norm(velocity[main], axis=1)
            print('EQUILIBRIUM', name, 't', float(data['times'][f]),
                  'rms', float(np.sqrt(np.mean(speed**2))), 'max', float(speed.max()))


if __name__ == '__main__':
    main()
