import unittest
from unittest.mock import patch
import numpy as np

from grasph_sph import TranslationNeighborSearch, build_cross_neighbor_pairs, configure_neighbor_search, active_neighbor_backend, CachedFluidNeighborSearch, build_neighbor_pairs
from grasph_boundary import BoundaryParticles, CachedWallNeighborSearch


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

    def test_skin_wall_list_matches_exact_search_between_rebuilds(self):
        configure_neighbor_search('ckdtree' if active_neighbor_backend() == 'ckdtree' else 'grid', 1)
        boundary = BoundaryParticles(
            positions=np.array([[0., 0., 0.], [0.2, 0., 0.], [1., 0., 0.]]),
            normals=np.tile([0., 0., 1.], (3, 1)),
            source_ids=np.zeros(3, dtype=np.int32), source_paths=['/wall'],
        )
        cache = CachedWallNeighborSearch(boundary, skin=0.08)
        for x in (0.14, 0.16, 0.18, 0.20, 0.24, 0.26):
            fluid = np.array([[x, 0., 0.]])
            expected = build_cross_neighbor_pairs(fluid, boundary.positions, 0.15)
            actual = cache.query(fluid, boundary, 0.15, {})
            self.assertEqual(pairs(*expected), pairs(*actual))

    def test_rotating_wall_cache_reuses_candidates_and_matches_rebuild(self):
        rng = np.random.default_rng(123)
        base = rng.uniform(-0.3, 0.3, (60, 3))
        boundary = BoundaryParticles(base.copy(), np.zeros_like(base),
                                     np.zeros(60, dtype=int), ['/body'])
        cache = CachedWallNeighborSearch(boundary, skin=0.08)
        fluid = rng.uniform(-0.4, 0.4, (100, 3))
        references = []
        for angle in (0, 0.01, 0.02, 0.5):
            c, s = np.cos(angle), np.sin(angle)
            rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            boundary.positions[:] = base @ rotation.T + angle * 0.01
            actual = cache.query(fluid, boundary, 0.15, {})
            self.assertEqual(pairs(*actual), pairs(*build_cross_neighbor_pairs(fluid, boundary.positions, 0.15)))
            references.append(cache.reference_boundary)
        self.assertIs(references[0], references[1])
        self.assertIsNot(references[0], references[-1])

    def test_fluid_verlet_pairs_match_exact_after_motion_removal_and_radius_change(self):
        rng = np.random.default_rng(21)
        fluid = rng.uniform(-0.3, 0.3, (100, 3))
        cache = CachedFluidNeighborSearch(0.05)
        for step in range(15):
            fluid += rng.uniform(-0.008, 0.008, fluid.shape)
            if step == 6:
                fluid = fluid[::2].copy()
            support = 0.15 if step < 10 else 0.2
            self.assertEqual(pairs(*cache.query(fluid, support)),
                             pairs(*build_neighbor_pairs(fluid, support)))


if __name__ == '__main__':
    unittest.main()
