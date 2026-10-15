import unittest
import numpy as np
from grasph_free_surface import compute_geometry_normal_covariance, project_missing_from_solids


class FreeSurfaceCornerTests(unittest.TestCase):
    def test_submerged_convex_corner_blocks_both_face_components(self):
        # Point is beyond both finite faces' projected footprints but its
        # kernel intersects their common corner. The old test saw no solid.
        triangles = np.array([
            [[0., 0., 0.], [0., 1., 1.], [0., 0., 1.]],  # outward +X
            [[0., 0., 0.], [-1., 0., 0.], [-1., 1., 0.]],  # outward -Z
        ])
        normals = np.array([[1., 0., 0.], [0., 0., -1.]])
        pos = np.array([[0.02, 0.01, -0.02]])
        cov, count, distance = compute_geometry_normal_covariance(pos, triangles, normals, .13)
        self.assertEqual(int(count[0]), 2)
        self.assertLess(float(distance[0]), .13)
        air, rank, _ = project_missing_from_solids(np.array([[-2., 0., 3.]]), cov, count > 0, .1)
        self.assertEqual(int(rank[0]), 2)
        np.testing.assert_allclose(air, 0., atol=1e-12)

    def test_finite_edge_does_not_extend_triangle_plane(self):
        triangle = np.array([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]])
        normals = np.array([[0., 0., 1.]])
        _, count, _ = compute_geometry_normal_covariance(np.array([[2., 2., .01]]), triangle, normals, .13)
        self.assertEqual(int(count[0]), 0)

    def test_back_face_of_rim_does_not_block_upward_free_surface(self):
        triangle = np.array([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]])
        normals = np.array([[0., 0., 1.]])
        cov, count, _ = compute_geometry_normal_covariance(np.array([[.2, .2, -.02]]), triangle, normals, .13)
        air, _, _ = project_missing_from_solids(np.array([[0., 0., 2.]]), cov, count > 0, .1)
        np.testing.assert_allclose(air, [[0., 0., 2.]])


if __name__ == '__main__':
    unittest.main()
