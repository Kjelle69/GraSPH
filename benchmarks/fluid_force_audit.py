"""Reconstruct fluid forces from saved fixed-body states (snapshot diagnostics).

py benchmarks/fluid_force_audit.py output/grasph_Fluidtest2.npz
Positive power means the term adds kinetic energy at that saved state.
Snapshots are rounded and post-drift, so these are not exact step integrals.
"""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from grasph_scene import load_scene
from grasph_sph import (SPHConfig, WendlandC2, build_neighbor_pairs,
    compute_pair_geometry, compute_pressure, compute_pressure_acceleration,
    compute_artificial_viscosity_acceleration, detached_component_mask,
    configure_neighbor_search)
from grasph_boundary import (build_boundary_particles, build_layered_wall_particles,
    build_wall_interactions, add_static_wall_pressure_acceleration)
from grasph_reference import hydrostatic_reference_state, compute_well_balanced_base_acceleration
from grasph_free_surface import FreeSurfaceConfig, collider_triangle_data, compute_atmospheric_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('npz')
    parser.add_argument('--times', nargs='+', type=float, default=[0, .5, 1, 1.25, 1.5, 2, 4, 8])
    args = parser.parse_args()
    configure_neighbor_search('auto', 1)
    with np.load(args.npz) as data:
        cfg = SPHConfig(dp=float(data['dp']), c0=float(data['c0']),
                        rho0=float(data['rho0']), gravity=tuple(data['gravity']))
        kernel = WendlandC2(cfg.h)
        level = float(data['surface_level'])
        with contextlib.redirect_stdout(io.StringIO()):
            scene = load_scene(str(data['source_file']))
            boundary = build_layered_wall_particles(build_boundary_particles(scene, spacing=cfg.dp),
                                                     cfg.dp, cfg.support, cfg.rho0)
        if any(body.joint_type != 'fixed' for body in scene.dynamic_bodies.values()):
            raise ValueError('This audit requires fixed body geometry')
        triangles, normals, _ = collider_triangle_data(scene)
        wall_rho, wall_p = hydrostatic_reference_state(boundary.positions, level, cfg)
        for target in args.times:
            frame = int(np.argmin(np.abs(data['times'] - target)))
            count = int(data['active_counts'][frame])
            pos = data['positions'][frame, :count].astype(float)
            vel = data['velocity'][frame, :count].astype(float)
            rho = data['rho'][frame, :count].astype(float)
            i, j = build_neighbor_pairs(pos, cfg.support)
            rij, r, grad = compute_pair_geometry(pos, i, j, kernel)
            detached, _, _ = detached_component_mask(count, i, j)
            rr, pr, depth = hydrostatic_reference_state(pos, level, cfg, return_depth=True)
            rr[detached], pr[detached], depth[detached] = cfg.rho0, 0., 0.
            q = compute_pressure(rho, cfg.rho0, cfg.tait_B, cfg.gamma) - pr
            interactions = build_wall_interactions(pos, boundary, kernel, cfg.support)
            fs = compute_atmospheric_state(pos, rho, rr, i, j, grad, boundary, interactions,
                                          triangles, normals, level, cfg, FreeSurfaceConfig(),
                                          active_mask=~detached)
            pressure = compute_pressure_acceleration(rho, q, i, j, grad, cfg.mass)
            wall = np.zeros_like(pos)
            add_static_wall_pressure_acceleration(wall, rho, q, boundary, interactions,
                                                  wall_rho, np.zeros(len(wall_rho)))
            terms = dict(pressure=pressure, wall_pressure=wall,
                         base=compute_well_balanced_base_acceleration(rho, rr, depth, cfg.gravity),
                         viscosity=compute_artificial_viscosity_acceleration(pos, vel, rho, i, j, rij, r, grad,
                                     cfg.mass, cfg.h, cfg.c0, alpha=.05), free_surface=fs['acc'])
            result = dict(time=float(data['times'][frame]), count=count,
                          kinetic_J=float(.5 * cfg.mass * np.sum(vel**2)), terms={})
            for name, acc in terms.items():
                result['terms'][name] = dict(power_W=float(cfg.mass * np.sum(vel * acc)),
                    acc_rms=float(np.sqrt(np.mean(np.sum(acc**2, axis=1)))),
                    acc_max=float(np.linalg.norm(acc, axis=1).max()))
            shell = fs['shell']
            result['shell_count'] = int(shell.sum())
            result['deep_shell_count'] = int(np.count_nonzero(shell & ((level - pos[:, 2]) > 2 * cfg.dp)))
            result['surface_reference_p_max'] = float(fs['p_ref_surface'][shell].max()) if shell.any() else 0.
            highest = np.argsort(np.linalg.norm(fs['acc'], axis=1))[-5:]
            result['highest_surface_force_samples'] = [dict(position=pos[k].tolist(),
                acceleration=fs['acc'][k].tolist(), geom_rank=int(fs['geom_rank'][k]),
                support_hG=float(fs['support_hG'][k])) for k in highest]
            result['total_power_W'] = sum(term['power_W'] for term in result['terms'].values())
            print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
