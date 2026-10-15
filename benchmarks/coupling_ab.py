"""Compare identical initial fluid with fixed and free rigid bodies.

py benchmarks/coupling_ab.py SCENE.usdc --time 0.05
Outputs are isolated in output/coupling_ab; original scene/results are preserved.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from pxr import Sdf, Usd
from grasph_scene import load_scene
from grasph_solver import run_simulation, SolverSettings
from grasph_sph import SPHConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scene')
    parser.add_argument('--time', type=float, default=0.05)
    parser.add_argument('--dp', type=float, default=0.05)
    parser.add_argument('--c0', type=float, default=40)
    parser.add_argument('--out', default='output/coupling_ab')
    args = parser.parse_args()
    source = Path(args.scene).resolve()
    directory = Path(args.out).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    scene = load_scene(source)
    fixed_file = directory / 'fixed_scene.usda'
    stage = Usd.Stage.CreateNew(str(fixed_file))
    stage.GetRootLayer().subLayerPaths = [str(source).replace('\\', '/')]
    for path in scene.dynamic_bodies:
        stage.OverridePrim(path).CreateAttribute('grasph:jointType', Sdf.ValueTypeNames.String).Set('fixed')
    stage.GetRootLayer().Save()
    results = {}
    for name, input_file in [('fixed', fixed_file), ('free', source)]:
        result = run_simulation(input_file, sph_cfg=SPHConfig(dp=args.dp, c0=args.c0),
                                settings=SolverSettings(sim_time=args.time, output_fps=100,
                                    output_dir=str(directory), output_stem=name,
                                    enable_solid_coupling=True, neighbor_workers=1,
                                    report_interval=30))
        with np.load(result.npz_file) as data:
            initial = data['source_initial_positions'].copy()
            velocity = data['velocity'][-1]
            main = np.isfinite(velocity).all(axis=1) & ~data['detached_mask'][-1].astype(bool)
            speed = np.linalg.norm(velocity[main], axis=1)
            results[name] = dict(initial=initial, median=float(np.median(speed)),
                                 p95=float(np.quantile(speed, .95)),
                                 rms=float(np.sqrt(np.mean(speed**2))),
                                 body_travel=float(np.linalg.norm(data['body_translations'], axis=-1).max()))
    np.testing.assert_array_equal(results['fixed']['initial'], results['free']['initial'])
    for name, result in results.items():
        print('A/B', name, {key: value for key, value in result.items() if key != 'initial'})
    print('A/B initial fluid centers identical:', len(results['fixed']['initial']))


if __name__ == '__main__':
    main()
