import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from pxr import Usd, UsdGeom

from grasph_output import write_usd_animation
from grasph_solver import _pad_frames, _sink_keep_mask


class SinkOutputTests(unittest.TestCase):
    def test_sink_removes_only_centers_inside_volume(self):
        sink = SimpleNamespace(bounds_min=np.array([-1., -2., -3.]),
                               bounds_max=np.array([1., 2., 3.]))
        pos = np.array([[0., 0., 0.], [1., 2., 3.], [1.01, 0., 0.]])
        np.testing.assert_array_equal(_sink_keep_mask(pos, [sink]), [False, False, True])

    def test_variable_particle_count_is_padded_in_npz_and_exact_in_usd(self):
        frames = [np.array([[0., 0., 0.], [1., 0., 0.]]),
                  np.array([[1., 0., 0.]])]
        velocities = [np.array([[1., 0., 0.], [0., 1., 0.]]),
                      np.array([[0., 0., 1.]])]
        padded = _pad_frames(frames, 2, (3,))
        self.assertTrue(np.isnan(padded[1, 1]).all())
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.usda'
            result = Path(directory) / 'result.usdc'
            stage = Usd.Stage.CreateNew(str(source))
            stage.GetRootLayer().Save()
            write_usd_animation(
                stage, frames, [0.0, 1.0], 24, .02, result,
                frame_velocities=velocities,
            )
            output = Usd.Stage.Open(str(result))
            points = UsdGeom.Points.Get(output, '/GraSPH_Result')
            self.assertEqual(len(points.GetPointsAttr().Get(0)), 2)
            self.assertEqual(len(points.GetPointsAttr().Get(24)), 1)
            self.assertEqual(len(points.GetWidthsAttr().Get(24)), 1)
            self.assertEqual(len(points.GetVelocitiesAttr().Get(24)), 1)
            self.assertEqual(tuple(points.GetVelocitiesAttr().Get(24)[0]), (0., 0., 1.))


if __name__ == '__main__':
    unittest.main()
