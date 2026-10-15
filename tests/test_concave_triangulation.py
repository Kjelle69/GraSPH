import unittest

import numpy as np
from pxr import Usd, UsdGeom

from grasph_scene import _triangulate, _world_points


class ConcaveMeshTriangulationTests(unittest.TestCase):
    def test_open_tank_rim_ngons_do_not_triangulate_across_opening(self):
        stage = Usd.Stage.CreateInMemory()
        mesh = UsdGeom.Mesh.Define(stage, "/Tank")
        points = np.asarray([
            (1, 1, 2), (1, 1, 0), (.842, .842, .465),
            (1, -1, 2), (1, -1, 0), (.842, -.842, .465),
            (-1, 1, 2), (-1, 1, 0), (-.842, .842, .465),
            (-1, -1, 2), (-1, -1, 0), (-.842, -.842, .465),
            (-.842, -.842, 2), (.842, .842, 2),
            (-.842, .842, 2), (.842, -.842, 2),
        ], dtype=np.float64)
        counts = [6, 6]
        indices = [6, 9, 3, 15, 12, 14, 3, 0, 6, 14, 13, 15]
        mesh.CreatePointsAttr().Set(points)
        mesh.CreateFaceVertexCountsAttr().Set(counts)
        mesh.CreateFaceVertexIndicesAttr().Set(indices)

        triangles = _triangulate(mesh.GetPrim(), points, Usd.TimeCode.Default())
        self.assertEqual(len(triangles), 8)

        def point_in_triangle(point, triangle):
            a, b, c = triangle[:, :2]
            cross = lambda u, v: u[0] * v[1] - u[1] * v[0]
            signs = (
                cross(b - a, point - a),
                cross(c - b, point - b),
                cross(a - c, point - c),
            )
            return min(signs) >= -1e-10 or max(signs) <= 1e-10

        # Points safely inside the square opening must not be covered by any
        # collision triangle; the former vertex-zero fan covered this region.
        for point in ((0.2, -0.2), (-0.2, 0.2), (0.0, 0.0)):
            self.assertFalse(
                any(point_in_triangle(np.asarray(point), tri) for tri in triangles),
                f"collision triangles unexpectedly cover the tank opening at {point}",
            )

    def test_negative_scale_preserves_outward_collision_winding(self):
        stage = Usd.Stage.CreateInMemory()
        parent = UsdGeom.Xform.Define(stage, "/Mirrored")
        parent.AddScaleOp().Set((-1.0, 1.0, 1.0))
        mesh = UsdGeom.Mesh.Define(stage, "/Mirrored/Cube")
        points = np.asarray((
            (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
            (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
        ), dtype=np.float64)
        faces = (
            (0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
            (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
        )
        mesh.CreatePointsAttr().Set(points)
        mesh.CreateFaceVertexCountsAttr().Set([4] * len(faces))
        mesh.CreateFaceVertexIndicesAttr().Set([index for face in faces for index in face])

        world_points = _world_points(mesh.GetPrim(), Usd.TimeCode.Default())
        triangles = _triangulate(mesh.GetPrim(), world_points, Usd.TimeCode.Default())
        normals = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        centers = triangles.mean(axis=1)
        self.assertTrue(np.all(np.einsum("ij,ij->i", normals, centers) > 0.0))


if __name__ == "__main__":
    unittest.main()
