import unittest

import numpy as np

from grasph_solver import _speed_stats


class SolverDiagnosticTests(unittest.TestCase):
    def test_speed_stats_summarizes_selected_particles(self):
        velocity = np.array([
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 10.0],
        ])

        median, p95, rms = _speed_stats(velocity, np.array([True, True, True, False]))

        self.assertEqual(median, 3.0)
        self.assertAlmostEqual(p95, 3.9)
        self.assertAlmostEqual(rms, np.sqrt(25.0 / 3.0))

    def test_speed_stats_handles_empty_selection(self):
        stats = _speed_stats(np.zeros((2, 3)), np.zeros(2, dtype=bool))
        self.assertEqual(stats, (0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
