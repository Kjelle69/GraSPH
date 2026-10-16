import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from grasph_diagnostics import ForceDiagnostics, analyze_result, nearest_solid
from grasph_solver import SolverSettings, run_simulation
from grasph_sph import SPHConfig
from tests.scene_helpers import make_scene


class ForceDiagnosticTests(unittest.TestCase):
    def test_watched_particle_uses_persistent_id_after_compaction(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "trace.jsonl"
            diagnostic = ForceDiagnostics(path, 0., 1., .1, top=1, watch_ids=(17,))
            ids = np.array([4, 17, 92])
            position = np.zeros((3, 3))
            velocity = np.array([[3., 0, 0], [0., 0, 0], [1., 0, 0]])
            original = velocity.copy()
            diagnostic.record(
                time=0., dt=.01, step=1, ids=ids,
                before={"position": position, "velocity": velocity},
                after={"position": position, "velocity": velocity},
                acceleration={"total": np.zeros((3, 3))}, delta_velocity={}, density_rate={}, masks={},
                triangles=[], triangle_paths=[], timings={})
            record = json.loads(path.read_text().splitlines()[1])
            self.assertEqual([p["id"] for p in record["particles"]], [4, 17])
            self.assertIsNone(record["particles"][0]["solid_distance_before_m"])
            np.testing.assert_array_equal(velocity, original)

    def test_exact_distance_uses_finite_triangle(self):
        triangles = np.array([[[0., 0, 0], [1., 0, 0], [0., 1, 0]]])
        distance, paths = nearest_solid(np.array([[0.1, 0.1, .2], [2., 0, 0]]), triangles, ["/wall"])
        np.testing.assert_allclose(distance, [.2, 1.])
        self.assertEqual(paths, ["/wall", "/wall"])

    def test_sampling_window_and_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "diagnostics.jsonl"
            diagnostic = ForceDiagnostics(path, .5, 1., .1)
            self.assertFalse(diagnostic.due(.49))
            self.assertTrue(diagnostic.due(.5001))
            self.assertFalse(diagnostic.due(.5004))
            self.assertTrue(diagnostic.due(.7001))
            self.assertFalse(diagnostic.due(.7002))
            self.assertTrue(diagnostic.due(1.))
            self.assertFalse(diagnostic.due(1.1))
            for args in ((1., .5, .1), (0., 1., 0.), (0., float('inf'), .1)):
                with self.assertRaises(ValueError):
                    ForceDiagnostics(path, *args)

    def test_diagnostics_do_not_change_coupled_solution_and_reconcile_updates(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "scene.usda"
            make_scene(source, mass=.5)
            cfg = SPHConfig(dp=.05, c0=40.)
            results = []
            for enabled in (False, True):
                settings = SolverSettings(
                    sim_time=.003, output_dir=folder, output_stem=str(enabled),
                    neighbor_workers=1, enable_solid_coupling=True,
                    enable_diagnostics=enabled, diagnostics_start=0., diagnostics_end=.003,
                    diagnostics_interval=.0006, diagnostics_ids=(0,))
                with contextlib.redirect_stdout(io.StringIO()):
                    results.append(run_simulation(source, cfg, settings))
            for key in ("positions", "velocity", "rho"):
                np.testing.assert_array_equal(getattr(results[0], key), getattr(results[1], key))
            with np.load(results[0].npz_file) as a, np.load(results[1].npz_file) as b:
                for key in ("body_translations", "body_rotations", "atmospheric_mask", "detached_mask"):
                    np.testing.assert_array_equal(a[key], b[key])
            self.assertFalse((root / "False.diagnostics.jsonl").exists())
            records = [json.loads(line) for line in (root / "True.diagnostics.jsonl").read_text().splitlines()]
            self.assertEqual(records[0]["schema"], 1)
            self.assertEqual(len(records) - 1, 5)
            for record in records[1:]:
                self.assertIn(0, [p["id"] for p in record["particles"]])
                for p in record["particles"]:
                    before, after = p['before'], p['after']
                    acceleration = p['acceleration_m_s2']
                    summed = sum(np.array(v) for k, v in acceleration.items() if k != 'total')
                    np.testing.assert_allclose(summed, acceleration['total'], atol=1e-12)
                    expected_v = (np.array(before['velocity']) + np.array(acceleration['total'])*record['dt']
                                  + p['delta_velocity_m_s']['air_drag'] + np.array(p['delta_velocity_m_s']['collision']))
                    np.testing.assert_allclose(expected_v, after['velocity'], atol=1e-12)
                    expected_x = (np.array(before['position'])
                                  + (np.array(after['velocity']) - p['delta_velocity_m_s']['collision'])*record['dt']
                                  + after['collision_position_delta'])
                    np.testing.assert_allclose(expected_x, after['position'], atol=1e-12)
                    expected_rho = before['rho'] + record['dt']*sum(p['density_rate_kg_m3_s'].values())
                    self.assertAlmostEqual(expected_rho, after['rho'], places=10)
            report = analyze_result(results[1].npz_file)
            self.assertEqual(len(report['body_paths']), 1)
            self.assertIsNone(report['first_crossing'])


if __name__ == '__main__':
    unittest.main()
