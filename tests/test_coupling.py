import unittest

import numpy as np

from grasph_coupling import couple_fluid_grains


class FluidGranularCouplingTests(unittest.TestCase):
    def test_coupling_conserves_linear_momentum(self):
        fluid_mass = 0.125
        grain_radius = np.asarray([0.02])
        grain_mass = np.asarray([0.05])
        fluid_acc, grain_acc, pairs = couple_fluid_grains(
            np.asarray([[0.0, 0.0, 0.0]]),
            np.asarray([[1.0, 0.0, 0.0]]),
            np.asarray([1000.0]), fluid_mass,
            np.asarray([[0.01, 0.0, 0.0]]),
            np.asarray([[0.0, 0.0, 0.0]]),
            grain_radius, grain_mass,
            np.asarray([0.0, 0.0, -9.81]), 0.1,
        )
        self.assertEqual(pairs, 1)
        total = fluid_mass * fluid_acc[0] + grain_mass[0] * grain_acc[0]
        np.testing.assert_allclose(total, np.zeros(3), atol=1.0e-12)


if __name__ == "__main__":
    unittest.main()
