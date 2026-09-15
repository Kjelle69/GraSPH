"""Compare cached wall search to the rebuild reference on saved fluid states."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from grasph_scene import load_scene
from grasph_boundary import (build_boundary_particles, build_layered_wall_particles,
                             build_wall_interactions, CachedWallNeighborSearch,
                             add_static_wall_density_rate, add_static_wall_pressure_acceleration)
from grasph_solver import _apply_boundary_translation, _boundary_velocity_array, _translated_solid_triangles
from grasph_free_surface import FreeSurfaceConfig, collider_triangle_data, compute_atmospheric_state
from grasph_reference import hydrostatic_reference_state
from grasph_sph import SPHConfig, WendlandC2, configure_neighbor_search, build_neighbor_pairs, compute_pair_geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('npz')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--report', default='output/wall_search_benchmark.json')
    args = parser.parse_args()
    configure_neighbor_search('ckdtree')
    with np.load(args.npz) as a:
        cfg = SPHConfig(dp=float(a['dp']))
        kernel = WendlandC2(cfg.h)
        scene = load_scene(str(a['source_file']))
        surface = build_boundary_particles(scene, spacing=cfg.dp)
        boundary = build_layered_wall_particles(surface, cfg.dp, cfg.support, cfg.rho0)
        base, facing = boundary.positions.copy(), boundary.facing_positions.copy()
        cache = CachedWallNeighborSearch(boundary)
        triangles, normals, triangle_paths = collider_triangle_data(scene)
        timings, errors = [], dict(acceleration=0., continuity=0., free_surface=0.)
        for frame, t in enumerate(a['times']):
            motion = scene.collider_motion(float(t))
            _apply_boundary_translation(boundary, base, facing, motion)
            pos, rho, vel = (a[k][frame].astype(np.float64) for k in ('positions', 'rho', 'velocity'))
            old = build_wall_interactions(pos, boundary, kernel, cfg.support)
            new = build_wall_interactions(pos, boundary, kernel, cfg.support, cache, motion)
            oi = np.lexsort((old['pair_j'], old['pair_i']))
            ni = np.lexsort((new['pair_j'], new['pair_i']))
            for key in ('pair_i', 'pair_j', 'rij', 'r', 'W', 'gradW', 'signed'):
                np.testing.assert_array_equal(old[key][oi], new[key][ni], err_msg=f'frame={frame}, key={key}')
            wall_rho, _ = hydrostatic_reference_state(boundary.positions, float(a['surface_level']), cfg)
            motion_next = scene.collider_motion(float(t) + 0.0003)
            wall_vel = _boundary_velocity_array(boundary, {p: (motion_next[p]-motion[p])/0.0003 for p in motion})
            outputs = []
            pair_i, pair_j = build_neighbor_pairs(pos, cfg.support)
            _, _, grad = compute_pair_geometry(pos, pair_i, pair_j, kernel)
            solid = _translated_solid_triangles(triangles, triangle_paths, motion)
            states = []
            for interactions in (old, new):
                acc = np.zeros_like(pos)
                drho = np.zeros(len(pos))
                add_static_wall_pressure_acceleration(acc, rho, a['reduced_pressure'][frame], boundary,
                                                     interactions, wall_rho, np.zeros(len(base)))
                add_static_wall_density_rate(drho, vel, boundary, interactions, wall_rho, wall_vel)
                fs = compute_atmospheric_state(pos, rho, a['rho_reference'][frame], pair_i, pair_j,
                                               grad, boundary, interactions, solid, normals,
                                               float(a['surface_level']), cfg, FreeSurfaceConfig())
                states.append(fs)
                outputs.append((acc, drho, fs['acc']))
            for key in ('shell', 'band', 'anchor'):
                np.testing.assert_array_equal(states[0][key], states[1][key])
            for key, x, y in zip(errors, outputs[0], outputs[1]):
                np.testing.assert_allclose(x, y, rtol=1e-12, atol=1e-10)
                errors[key] = max(errors[key], float(np.abs(x-y).max()))
            if frame in (0, len(a['times'])//3, 2*len(a['times'])//3, len(a['times'])-1):
                durations = {'rebuild': [], 'cached': []}
                for repeat in range(args.repeats):
                    # Alternate order to reduce bias from temperature/frequency drift.
                    for mode in (('rebuild', 'cached') if repeat % 2 == 0 else ('cached', 'rebuild')):
                        start = time.perf_counter()
                        build_wall_interactions(pos, boundary, kernel, cfg.support,
                                                cache if mode == 'cached' else None, motion)
                        durations[mode].append(time.perf_counter()-start)
                row = dict(time=float(t), **{k: float(np.median(v)) for k,v in durations.items()})
                row['speedup'] = row['rebuild']/row['cached']
                timings.append(row)
        report = dict(source=args.npz, verified_frames=len(a['times']), particles=len(pos),
                      wall_particles=len(base), pair_geometry_exact=True, force_max_abs_errors=errors,
                      timings=timings, repeats=args.repeats)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
