import unittest

import numpy as np

from grasph_solver import apply_detached_air_drag


class DetachedAirDragTests(unittest.TestCase):
    def test_drag_reduces_only_detached_particle_speeds(self):
        velocity = np.array([[0.0, 0.0, -80.0], [3.0, 4.0, 0.0]])
        before = velocity.copy()
        apply_detached_air_drag(
            velocity, np.array([True, False]), 0.01, 0.03,
            fluid_density=1000.0, air_density=1.225, drag_coefficient=0.47,
        )
        self.assertLess(np.linalg.norm(velocity[0]), np.linalg.norm(before[0]))
        np.testing.assert_array_equal(velocity[1], before[1])
        self.assertGreater(velocity[0, 2], before[0, 2])

    def test_zero_coefficient_is_exactly_disabled(self):
        velocity = np.array([[0.0, 0.0, -80.0]])
        before = velocity.copy()
        apply_detached_air_drag(velocity, np.array([True]), 0.1, 0.03,
                                drag_coefficient=0.0)
        np.testing.assert_array_equal(velocity, before)


if __name__ == "__main__":
    unittest.main()
