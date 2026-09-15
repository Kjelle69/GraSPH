import unittest
from unittest.mock import patch
import numpy as np

from grasph_sph import TranslationNeighborSearch, build_cross_neighbor_pairs, configure_neighbor_search, active_neighbor_backend


def pairs(i, j):
    return sorted(zip(i.tolist(), j.tolist()))


class CachedSearchTests(unittest.TestCase):
    def tearDown(self):
        configure_neighbor_search()

    @unittest.skipUnless(active_neighbor_backend() == 'ckdtree', 'SciPy accelerator not installed')
    def test_translation_and_parallel_paths(self):
        rng = np.random.default_rng(42)
        reference = rng.uniform(-1, 1, (80, 3))
        groups = np.arange(80) % 3
        cache = TranslationNeighborSearch(reference, groups)
        trees = [id(g[2]) for g in cache.groups]
        configure_neighbor_search('ckdtree', 2)
        for n in (0, 150, 2601):
            fluid = rng.uniform(-1, 1, (n, 3))
            for step in range(3):
                shifts = {0: np.zeros(3), 1: np.array([0.2 * step, -0.1, 0.3]),
                          2: np.array([-0.1, 0.3 * step, -0.2])}
                world = reference + np.array([shifts[int(g)] for g in groups])
                expected = build_cross_neighbor_pairs(fluid, world, 0.4)
                actual = cache.query(fluid, world, 0.4, shifts)
                self.assertEqual(pairs(*expected), pairs(*actual))
        self.assertEqual(trees, [id(g[2]) for g in cache.groups])
        np.testing.assert_array_equal(cache.reference, reference)

    @unittest.skipUnless(active_neighbor_backend() == 'ckdtree', 'SciPy accelerator not installed')
    def test_strict_radius_and_owned_reference(self):
        configure_neighbor_search('ckdtree', 1)
        reference = np.array([[0., 0., 0.], [2., 0., 0.]])
        cache = TranslationNeighborSearch(reference, np.array([0, 1]))
        shift = np.array([1000., -2000., 3000.])
        world = reference + shift
        reference[:] = 100  # Mutation must not corrupt the persistent trees.
        fluid = world[0] + np.array([[1., 0., 0.], [1. - 1e-10, 0., 0.], [1. + 1e-10, 0., 0.]])
        expected = build_cross_neighbor_pairs(fluid, world, 1.)
        self.assertEqual(pairs(*expected), pairs(*cache.query(fluid, world, 1., {0: shift, 1: shift})))
        self.assertNotIn((0, 0), pairs(*expected))

    def test_empty_walls_and_grid_fallback(self):
        cache = TranslationNeighborSearch(np.empty((0, 3)), np.empty(0, dtype=int))
        self.assertEqual(pairs(*cache.query(np.ones((2, 3)), np.empty((0, 3)), 1., {})), [])
        configure_neighbor_search('grid')
        reference = np.array([[0., 0., 0.], [1., 0., 0.]])
        cache = TranslationNeighborSearch(reference, np.array([0, 1]))
        world = reference + [0.1, 0.2, 0.3]
        fluid = np.array([[0., 0., 0.]])
        self.assertEqual(pairs(*build_cross_neighbor_pairs(fluid, world, 0.8)),
                         pairs(*cache.query(fluid, world, 0.8, {})))

    def test_without_scipy(self):
        configure_neighbor_search('auto')
        with patch('grasph_sph._cKDTree', None):
            reference = np.array([[0., 0., 0.]])
            cache = TranslationNeighborSearch(reference, np.array([0]))
            world = reference + [0.2, 0.1, 0.3]
            fluid = np.array([[0., 0., 0.]])
            self.assertEqual(pairs(*build_cross_neighbor_pairs(fluid, world, 0.5)),
                             pairs(*cache.query(fluid, world, 0.5, {})))


if __name__ == '__main__':
    unittest.main()
