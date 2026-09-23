import tempfile
import unittest
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom

from grasph_collision import resolve_collisions
from grasph_output import write_usd_animation


class _PlaneCollider:
    name = "/Wheel"

    def query(self, points):
        # Reference surface y=0 with normal +Y.
        return points[:, 1].copy(), np.tile([0.0, 1.0, 0.0], (len(points), 1))


class HingedColliderTests(unittest.TestCase):
    def test_coulomb_wall_friction_is_impulse_limited(self):
        pos = np.array([[0.0, 0.05, 0.0]])
        vel = np.array([[2.0, -1.0, 0.0]])
        resolve_collisions(
            pos, vel, [_PlaneCollider()], particle_radius=0.1,
            restitution=0.0, friction_coefficient=0.5, iterations=4,
        )
        # Normal delta-v is 1 m/s, so mu=0.5 may remove only 0.5 m/s
        # tangentially. Extra contact iterations must not damp it again.
        self.assertAlmostEqual(vel[0, 0], 1.5, places=12)
        self.assertAlmostEqual(vel[0, 1], 0.0, places=12)

    def test_sphere_contact_converts_sliding_to_spin(self):
        pos = np.array([[0.0, 0.05, 0.0]])
        vel = np.array([[2.0, -1.0, 0.0]])
        omega = np.zeros((1, 3))
        mass = np.array([1.0])
        radius = np.array([0.1])
        inertia = 0.4 * mass * radius**2
        resolve_collisions(
            pos, vel, [_PlaneCollider()], particle_radius=radius,
            restitution=np.array([0.0]), friction_coefficient=np.array([1.0]),
            iterations=4, particle_mass=mass,
            particle_angular_velocities=omega, particle_inertia=inertia,
        )
        self.assertLess(vel[0, 0], 2.0)
        self.assertGreater(abs(omega[0, 2]), 1.0)
        contact_velocity = vel[0] + np.cross(omega[0], np.array([0.0, -0.1, 0.0]))
        self.assertLess(abs(contact_velocity[0]), 2.0)

    def test_collision_returns_reaction_angular_impulse(self):
        pos = np.array([[0.0, 0.05, 1.0]])
        vel = np.array([[0.0, -2.0, 0.0]])
        contacts, impulses = resolve_collisions(
            pos, vel, [_PlaneCollider()], particle_radius=0.1, iterations=1,
            collider_centers={"/Wheel": np.zeros(3)}, particle_mass=0.5,
            return_angular_impulses=True,
        )
        self.assertEqual(contacts, 1)
        # Fluid gains +Y momentum at z=1, so the wheel receives +X impulse.
        self.assertAlmostEqual(impulses["/Wheel"][0], 1.0)

    def test_output_authors_parent_rotation_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.usda"
            result = Path(tmp) / "result.usdc"
            stage = Usd.Stage.CreateNew(str(source))
            wheel = UsdGeom.Xform.Define(stage, "/Wheel")
            wheel.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            stage.SetStartTimeCode(0)
            stage.SetEndTimeCode(24)
            stage.SetTimeCodesPerSecond(24)
            stage.SetFramesPerSecond(24)
            stage.GetRootLayer().Save()

            write_usd_animation(
                stage, [np.empty((0, 3)), np.empty((0, 3))], [0.0, 1.0],
                24.0, 0.02, result,
                hinge_frames={"/Wheel/Mesh": {
                    "driver_path": "/Wheel",
                    "axis": np.array([1.0, 0.0, 0.0]),
                    "angles": np.array([0.0, np.pi / 2.0]),
                }},
            )
            composed = Usd.Stage.Open(str(result))
            rotate = composed.GetPrimAtPath("/Wheel").GetAttribute("xformOp:rotateXYZ")
            self.assertEqual(rotate.GetTimeSamples(), [0.0, 24.0])
            self.assertAlmostEqual(float(rotate.Get(24.0)[0]), 90.0, places=4)


if __name__ == "__main__":
    unittest.main()
