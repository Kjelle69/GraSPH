import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
from benchmarks.buoyancy_probe import make_scene
from grasph_solver import run_simulation, SolverSettings
from grasph_sph import SPHConfig


class BuoyancyTests(unittest.TestCase):
    def test_immersed_light_body_rises_and_heavy_body_sinks(self):
        with tempfile.TemporaryDirectory() as directory:
            dz = []
            for mass in (0.5, 8.0):
                source = Path(directory) / f'body_{mass}.usda'
                make_scene(source, mass)
                with contextlib.redirect_stdout(io.StringIO()):
                    result = run_simulation(
                        source, sph_cfg=SPHConfig(dp=0.04, c0=40),
                        settings=SolverSettings(sim_time=0.006, output_fps=1000,
                            output_dir=directory, output_stem=f'result_{mass}',
                            enable_solid_coupling=True, neighbor_workers=1))
                with np.load(result.npz_file) as data:
                    dz.append(float(data['body_translations'][0, -1, 2]))
                    self.assertTrue(np.isfinite(data['positions']).all())
                    self.assertTrue(np.isfinite(data['rho']).all())
            self.assertGreater(dz[0], 0.0)
            self.assertLess(dz[1], 0.0)


if __name__ == '__main__':
    unittest.main()
