import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
from pxr import Usd
from tests.scene_helpers import box
from grasph_scene import load_scene, exclude_solid_fluid_particles
from grasph_collision import is_closed_triangle_mesh, points_inside_closed_mesh


class InitialFluidExclusionTests(unittest.TestCase):
    def test_closed_solids_removed_but_tank_cavity_and_open_sheet_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'solids.usda'
            stage = Usd.Stage.CreateNew(str(path))
            box(stage, '/Body', (-0.1, -0.1, 0.3), (0.1, 0.1, 0.5), 'collider')
            box(stage, '/Floor', (-1., -1., -0.2), (1., 1., 0.), 'collider')
            stage.GetRootLayer().Save()
            with contextlib.redirect_stdout(io.StringIO()):
                scene = load_scene(path)
            points = np.array([[0., 0., 0.4], [0., 0., -0.1],
                               [0.5, 0., 0.2], [2., 0., 0.4]])
            with contextlib.redirect_stdout(io.StringIO()):
                result = exclude_solid_fluid_particles(points, scene.colliders)
            np.testing.assert_array_equal(result, points[2:])
            floor = next(mesh for mesh in scene.colliders if mesh.path == '/Floor')
            near_floor = np.array([[0., 0., 0.01], [0., 0., 0.04]])
            with contextlib.redirect_stdout(io.StringIO()):
                cleared = exclude_solid_fluid_particles(near_floor, [floor], particle_radius=0.02)
            np.testing.assert_array_equal(cleared, near_floor[1:])
            body = scene.colliders[0]
            self.assertTrue(is_closed_triangle_mesh(body.triangles))
            # Reversed winding is still a volume. Removing one face opens it.
            np.testing.assert_array_equal(points_inside_closed_mesh(points, body.triangles),
                                          points_inside_closed_mesh(points, body.triangles[:, ::-1]))
            body.triangles = body.triangles[:-2]
            self.assertFalse(is_closed_triangle_mesh(body.triangles))
            np.testing.assert_array_equal(exclude_solid_fluid_particles(points, [body]), points)


if __name__ == '__main__':
    unittest.main()
