"""Numerical regression checks for allocation-efficient SPH operators."""
import unittest
from unittest.mock import patch

import numpy as np

import grasph_sph as sph
from grasph_free_surface import (
    FreeSurfaceConfig, build_seed_connected_band_with_anchor, compute_fluid_support_G,
)


class SPHOperatorTests(unittest.TestCase):
    def test_fluid_support_matches_weighted_vector_scatter(self):
        rng = np.random.default_rng(72)
        n = 37
        rho = rng.uniform(950, 1050, n)
        for m in (0, 1500):
            ii = rng.integers(0, n, m, dtype=np.int32)
            jj = rng.integers(0, n, m, dtype=np.int32)
            grad = rng.normal(size=(m, 3))
            expected = sph._scatter_add_vec3(
                n, ii, (.125 / rho[jj])[:, None] * grad,
                jj, -(.125 / rho[ii])[:, None] * grad)
            np.testing.assert_allclose(compute_fluid_support_G(n, rho, .125, ii, jj, grad),
                                       expected, rtol=1e-13, atol=1e-15)

    def test_kernel_gradient_matches_radial_definition(self):
        rng = np.random.default_rng(73)
        for h in (0.013, 0.065, 0.7):
            kernel = sph.WendlandC2(h)
            rij = rng.normal(size=(300, 3)) * h
            rij[:4] = [[0, 0, 0], [1e-13, 0, 0], [2*h, 0, 0], [3*h, 0, 0]]
            r = np.linalg.norm(rij, axis=1)
            expected = np.zeros_like(rij)
            active = (r > 1e-12) & (r < 2*h)
            q = r[active] / h
            derivative = -5 * kernel.alpha * q * (1 - .5*q)**3 / h
            expected[active] = derivative[:, None] * rij[active] / r[active, None]
            np.testing.assert_allclose(kernel.gradW(rij, r), expected, rtol=3e-14)
        self.assertEqual(kernel.gradW(np.empty((0, 3)), np.empty(0)).shape, (0, 3))

    def test_density_rate_matches_scatter_with_repeated_indices(self):
        rng = np.random.default_rng(74)
        for n, m in ((0, 0), (11, 0), (37, 1500)):
            vel = rng.normal(size=(n, 3))
            ii = rng.integers(0, max(n, 1), m, dtype=np.int32)
            jj = rng.integers(0, max(n, 1), m, dtype=np.int32)
            grad = rng.normal(size=(m, 3))
            term = .125 * np.einsum('ij,ij->i', vel[ii] - vel[jj], grad)
            expected = np.zeros(n)
            np.add.at(expected, ii, term)
            np.add.at(expected, jj, term)
            np.testing.assert_allclose(sph.compute_density_rate(vel, ii, jj, grad, .125),
                                       expected, rtol=1e-13, atol=1e-13)

    def test_one_sided_component_graph_matches_union_find(self):
        if sph._scipy_sparse is None:
            self.skipTest('SciPy not installed')
        # Mixed edge directions, repeated edges, self-edges, equal-size clusters,
        # isolated nodes, and empty graphs must retain dominant-cluster ties.
        for ii, jj in (([3, 2, 1, 3, 5, 6], [2, 1, 3, 2, 6, 6]),
                       ([1, 3], [0, 2]), ([], [])):
            ii, jj = np.array(ii, dtype=np.int32), np.array(jj, dtype=np.int32)
            actual = sph.detached_component_mask(9, ii, jj)
            with patch.object(sph, '_scipy_sparse', None):
                expected = sph.detached_component_mask(9, ii, jj)
            np.testing.assert_array_equal(actual[0], expected[0])
            self.assertEqual(actual[1], expected[1])
            np.testing.assert_array_equal(np.sort(actual[2]), np.sort(expected[2]))

    def test_anchor_tie_preserves_forward_then_reverse_order(self):
        pos = np.array([[-1., 0, 0], [1., 0, 0], [0., 0, -.1], [0., 0, 1.]])
        missing = np.tile([0., 0, 1.], (4, 1))
        shell = np.array([True, True, False, False])
        # Reverse edge to seed 0 comes first in input, but forward edge to
        # seed 1 must win the equal-distance tie, just as before filtering.
        ii = np.array([2, 1, 1], dtype=np.int32)
        jj = np.array([0, 2, 3], dtype=np.int32)
        band, anchor, _ = build_seed_connected_band_with_anchor(
            pos, ii, jj, missing, shell, FreeSurfaceConfig())
        np.testing.assert_array_equal(band, [True, True, True, False])
        np.testing.assert_array_equal(anchor, [0, 1, 1, -1])


if __name__ == '__main__':
    unittest.main()
