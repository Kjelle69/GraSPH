import unittest

import numpy as np

from grasph_rigid import _oriented_surface_query


class RigidContactDistanceTests(unittest.TestCase):
    def test_exterior_cube_corner_cannot_become_deep_penetration_from_roundoff(self):
        vertices = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                             [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float)
        faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
                 (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
        triangles = np.array([vertices[list(t)] for a, b, c, d in faces
                              for t in ((a, b, c), (a, c, d))])
        # The point is clearly outside X, but just beyond a shared Y edge.
        # The old closest-face sign changed to negative under tiny rotations.
        points = np.array([[-4, -1 + dy, -1] for dy in (0, -1e-12, 1e-12)])
        distance, normals = _oriented_surface_query(points, triangles)
        np.testing.assert_allclose(distance, 3.0, atol=1e-10)
        np.testing.assert_allclose(normals[:, 0], -1.0, atol=1e-10)
        inside, _ = _oriented_surface_query(np.array([[0., 0., 0.]]), triangles)
        self.assertAlmostEqual(float(inside[0]), -1.0)

    def test_opening_above_triangle_edge_is_not_treated_as_a_lid(self):
        # One top-rim triangle. The query is above the same plane but outside
        # the triangle, as a body sample is when it is over the tank opening.
        triangle = np.asarray((
            ((0.8, -0.8, 2.0), (1.0, -0.8, 2.0), (1.0, 0.8, 2.0)),
        ))
        point = np.asarray(((0.0, 0.0, 2.01),))
        distance, _ = _oriented_surface_query(point, triangle)
        self.assertGreater(float(distance[0]), 0.01)
        self.assertGreater(float(distance[0]), 0.5)

    def test_point_over_rim_uses_normal_distance_and_keeps_contact(self):
        triangle = np.asarray((
            ((0.8, -0.8, 2.0), (1.0, -0.8, 2.0), (1.0, 0.8, 2.0)),
        ))
        point = np.asarray(((0.95, -0.2, 2.01),))
        distance, normal = _oriented_surface_query(point, triangle)
        self.assertAlmostEqual(float(distance[0]), 0.01)
        self.assertTrue(np.allclose(normal[0], (0.0, 0.0, 1.0)))

    def test_point_below_surface_has_negative_distance(self):
        triangle = np.asarray((
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ))
        point = np.asarray(((0.2, 0.2, -0.03),))
        distance, _ = _oriented_surface_query(point, triangle)
        self.assertAlmostEqual(float(distance[0]), -0.03)


if __name__ == "__main__":
    unittest.main()
