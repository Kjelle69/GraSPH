import unittest
from unittest.mock import patch

import numpy as np

import grasph_boundary
import grasph_sph
from grasph_boundary import BoundaryParticles, CachedWallNeighborSearch
from grasph_sph import (
    _scatter_add_vec3,
    compute_pressure_acceleration,
    configure_neighbor_search,
)


class PairChunkingTests(unittest.TestCase):
    def test_pressure_acceleration_matches_unchunked_reference(self):
        rng = np.random.default_rng(2026)
        n = 37
        pair_i = rng.integers(0, n, 113, dtype=np.int32)
        pair_j = (pair_i + rng.integers(1, n, 113)) % n
        rho = rng.uniform(850.0, 1150.0, n)
        pressure = rng.uniform(-2e4, 3e4, n)
        grad_w = rng.normal(size=(len(pair_i), 3))
        mass = 0.017

        rhoi = rho[pair_i]
        rhoj = rho[pair_j]
        factor = mass * (
            pressure[pair_i] / (rhoi * rhoi)
            + pressure[pair_j] / (rhoj * rhoj)
        )
        pair_force = factor[:, None] * grad_w
        expected = _scatter_add_vec3(
            n, pair_i, -pair_force, pair_j, pair_force
        )

        with patch.object(grasph_sph, "_PAIR_WORK_CHUNK", 7):
            actual = compute_pressure_acceleration(
                rho, pressure, pair_i, pair_j, grad_w, mass
            )
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)

    def test_cached_wall_filter_chunks_and_preserves_exact_pairs(self):
        configure_neighbor_search("grid")
        try:
            positions = np.column_stack((np.arange(12, dtype=float) * 0.1,
                                         np.zeros(12), np.zeros(12)))
            boundary = BoundaryParticles(
                positions=positions,
                normals=np.tile([0.0, 0.0, 1.0], (len(positions), 1)),
                source_ids=np.zeros(len(positions), dtype=np.int32),
                source_paths=["/wall"],
            )
            cache = CachedWallNeighborSearch(boundary, skin=0.5)
            fluid = np.array([[0.04, 0.0, 0.0], [0.53, 0.0, 0.0]])
            with patch.object(grasph_boundary, "_WALL_PAIR_CHUNK", 2):
                actual = cache.query(fluid, boundary, 0.21, {})
            expected = []
            for i, point in enumerate(fluid):
                for j, wall_point in enumerate(positions):
                    if np.dot(point - wall_point, point - wall_point) < 0.21**2:
                        expected.append((i, j))
            self.assertEqual(
                sorted(zip(actual[0].tolist(), actual[1].tolist())),
                sorted(expected),
            )
        finally:
            configure_neighbor_search()


if __name__ == "__main__":
    unittest.main()
