import unittest

import numpy as np

from grasph_boundary import BoundaryParticles
from grasph_free_surface import FreeSurfaceConfig, compute_atmospheric_state
from grasph_sph import SPHConfig, WendlandC2


class DetachedFreeSurfaceTests(unittest.TestCase):
    def test_detached_droplets_do_not_receive_pool_anchored_traction(self):
        # Two disconnected particle pairs: the first is the pool, the second
        # a free droplet. With only fluid support, each pair has a free surface.
        pos = np.array([
            [0.00, 0.0, 0.0], [0.02, 0.0, 0.0],
            [2.00, 0.0, 0.0], [2.02, 0.0, 0.0],
        ])
        pair_i = np.array([0, 1, 2, 3], dtype=np.int32)
        pair_j = np.array([1, 0, 3, 2], dtype=np.int32)
        cfg = SPHConfig(dp=0.05, c0=280.0)
        kernel = WendlandC2(cfg.h)
        rij = pos[pair_i] - pos[pair_j]
        r = np.linalg.norm(rij, axis=1)
        gradW = kernel.gradW(rij, r)
        boundary = BoundaryParticles(
            positions=np.empty((0, 3)), normals=np.empty((0, 3)),
            source_ids=np.empty(0, dtype=np.int32), source_paths=[],
        )
        no_walls = {
            "pair_i": np.empty(0, dtype=np.int32),
            "pair_j": np.empty(0, dtype=np.int32),
            "gradW": np.empty((0, 3)),
        }
        args = (
            pos, np.full(4, cfg.rho0), np.full(4, cfg.rho0),
            pair_i, pair_j, gradW, boundary, no_walls,
            np.empty((0, 3, 3)), np.empty((0, 3)), 1.0, cfg,
            FreeSurfaceConfig(),
        )
        before = compute_atmospheric_state(*args)
        after = compute_atmospheric_state(
            *args, active_mask=np.array([True, True, False, False])
        )

        self.assertGreater(np.linalg.norm(before["acc"][2]), 0.0)
        np.testing.assert_array_equal(after["acc"][2:], np.zeros((2, 3)))
        self.assertTrue(np.all(after["band"][2:] == 0))
        self.assertTrue(np.any(after["acc"][:2] != 0.0))


if __name__ == "__main__":
    unittest.main()
