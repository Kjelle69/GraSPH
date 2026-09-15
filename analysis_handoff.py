"""Read-only analysis of the saved coarse replay; no solver changes."""
from pathlib import Path
import argparse
import json
import numpy as np
from grasph_sph import build_neighbor_pairs, neighbor_counts, detached_component_mask
from grasph_scene import load_scene
from grasph_free_surface import collider_triangle_data, compute_geometry_normal_covariance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', nargs='?', default='output/TankAnim_grasph.npz')
    parser.add_argument('--report', default='output/handoff_analysis.json')
    args = parser.parse_args()
    source = Path(args.source)
    with np.load(source) as a:
        scene = load_scene(str(a['source_file']))
    triangles, normals, paths = collider_triangle_data(scene)
    paths = np.asarray(paths)
    rows, episodes = [], []
    with np.load(source) as a:
        pos, vel, times = a['positions'], a['velocity'], a['times']
        masks = a['detached_mask'].astype(bool)
        support, dp = float(a['support']), float(a['dp'])
        for f, t in enumerate(times):
            i, j = build_neighbor_pairs(pos[f], support)
            counts = neighbor_counts(len(pos[f]), i, j)
            disconnected, nc, sizes = detached_component_mask(len(pos[f]), i, j)
            moved = triangles.copy()
            for path, shift in scene.collider_motion(float(t)).items():
                moved[paths == path] += shift
            _, _, distance = compute_geometry_normal_covariance(pos[f], moved, normals, support)
            near = distance <= dp
            rows.append(dict(time=float(t), recorded_detached=int(masks[f].sum()),
                             zero_neighbors=int((counts == 0).sum()),
                             disconnected=int(disconnected.sum()), components=int(nc),
                             main_size=int(sizes.max()), detached_near_face=int((masks[f] & near).sum()),
                             detached_descending=int((masks[f] & (vel[f, :, 2] < 0)).sum()),
                             near_face_above_initial_surface=int((near & (pos[f, :, 2] > float(a['surface_level']))).sum()),
                             rho_min=float(a['rho'][f].min()), rho_max=float(a['rho'][f].max()),
                             vmax=float(np.linalg.norm(vel[f], axis=1).max())))
        for p in range(pos.shape[1]):
            starts = np.flatnonzero(masks[:, p] & ~np.r_[False, masks[:-1, p]])
            for start in starts:
                ends = np.flatnonzero(~masks[start + 1:, p])
                end = int(start + 1 + ends[0]) if len(ends) else len(times) - 1
                episodes.append(dict(particle=p, start=float(times[start]), end=float(times[end]),
                                     reconnected_in_recorded_mask=bool(len(ends)),
                                     dz=float(pos[end, p, 2] - pos[start, p, 2]),
                                     initial_vz=float(vel[start, p, 2]), final_vz=float(vel[end, p, 2])))
        report = dict(source=str(source), dp=dp, frames=len(times), particles=pos.shape[1],
                      first_recorded_detached=next((r['time'] for r in rows if r['recorded_detached']), None),
                      max_recorded_detached=int(masks.sum(axis=1).max()),
                      unique_recorded_detached=int(masks.any(axis=0).sum()),
                      final_recorded_detached=int(masks[-1].sum()),
                      finite_state=bool(all(np.isfinite(a[k]).all() for k in ('positions', 'velocity', 'rho'))),
                      rows=rows, episodes=episodes,
                      limitations=['Saved masks describe the pre-drift graph; positions are post-drift.',
                                   'No solver version or droplet mode is stored in this NPZ.',
                                   'Finite-face proximity uses plane projection, not edge distance.',
                                   'NPZ alone does not identify the solver version, mode, or full-step history.'])
    Path(args.report).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('rows', 'episodes')}, indent=2))
    print('Episodes:', len(episodes), 'descending net:', sum(e['dz'] < 0 for e in episodes),
          'reconnected:', sum(e['reconnected_in_recorded_mask'] for e in episodes))
    print('Final frame:', rows[-1])


if __name__ == '__main__':
    main()
