"""Shared joints, independent of any fracture/demo authoring system."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from grasph_joints import FixedJoint, JointSystem
from tests.test_convex_body_contacts import cube


class JointTests(unittest.TestCase):
    def setup_system(self, joints, count=2):
        triples = [cube(chr(97+i), center=(i, 0, 0), half=(.2,)*3) for i in range(count)]
        scene = SimpleNamespace(dynamic_bodies={t[0].path:t[0] for t in triples}, fixed_joints=joints)
        states = {t[0].path:t[2] for t in triples}
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        return scene, states, JointSystem(scene, states, Path(directory.name)/'joints.jsonl')

    def test_empty_is_noop(self):
        scene, states, system = self.setup_system(())
        states['a']['velocity'][:] = [1, 2, 3]
        system.solve(.01, .01)
        system.project_poses()
        np.testing.assert_array_equal(states['a']['velocity'], [1, 2, 3])
        self.assertFalse(hasattr(scene, 'joint_collision_exclusions'))

    def test_unbreakable_fixed_body_pair_conserves_momentum(self):
        scene, states, system = self.setup_system([FixedJoint('/Joint', 'a', 'b', [.5, 0, 0])])
        states['a']['velocity'][:] = [1, 0, 0]
        system.solve(.01, .01)
        for state in states.values():
            np.testing.assert_allclose(state['velocity'], [.5, 0, 0], atol=1e-12)
        self.assertEqual(system.broken, set())
        self.assertIn(frozenset(('a', 'b')), scene.joint_collision_exclusions)

    def test_pair_angular_momentum(self):
        scene, states, system = self.setup_system([FixedJoint('/Joint', 'a', 'b', [.5, 0, 0])])
        scene.dynamic_bodies['b'].mass = 2
        states['a']['velocity'][:] = [0, 1, 0]
        def momentum():
            linear, angular = np.zeros(3), np.zeros(3)
            for p, s in states.items():
                b = scene.dynamic_bodies[p]
                linear += b.mass*s['velocity']
                angular += np.cross(b.center, b.mass*s['velocity']) + np.diag(b.inertia) @ s['omega']
            return np.r_[linear, angular]
        before = momentum()
        system.solve(.01, .01)
        np.testing.assert_allclose(momentum(), before, atol=1e-12)

    def test_break_rejects_trial_impulse(self):
        scene, states, system = self.setup_system([FixedJoint('/Joint', 'a', 'b', [.5, 0, 0], 1, 1)])
        states['a']['velocity'][:] = [1, 0, 0]
        system.solve(.01, .01)
        np.testing.assert_allclose(states['a']['velocity'], [1, 0, 0])
        np.testing.assert_allclose(states['b']['velocity'], 0)
        self.assertEqual(system.broken, {'/Joint'})
        self.assertFalse(scene.joint_collision_exclusions)
        system.solve(.01, .02)
        self.assertEqual(len(system.events), 1)

    def test_world_joint_holds_or_breaks_under_weight(self):
        for limit, breaks in ((np.inf, False), (5., True)):
            _, states, system = self.setup_system([FixedJoint('/World', None, 'a', [0, 0, 0], limit, np.inf)])
            states['a']['velocity'][:] = [0, 0, -.0981]
            system.solve(.01, .01)
            self.assertEqual(bool(system.broken), breaks)
            if not breaks:
                np.testing.assert_allclose(states['a']['velocity'], 0, atol=1e-12)

    def test_graph_not_star_and_loop_supported(self):
        joints = [FixedJoint('/AB', 'a', 'b', [.5, 0, 0]),
                  FixedJoint('/BC', 'b', 'c', [1.5, 0, 0])]
        for loop in (False, True):
            edges = joints+[FixedJoint('/AC', 'a', 'c', [1, 0, 0])] if loop else joints
            _, states, system = self.setup_system(edges, 3)
            states['c']['velocity'][:] = [3, 0, 0]
            system.solve(.01, .01)
            for state in states.values():
                np.testing.assert_allclose(state['velocity'], [1, 0, 0], atol=1e-10)

    def test_pose_projection_preserves_reference_relative_geometry(self):
        _, states, system = self.setup_system([FixedJoint('/AB', 'a', 'b', [.5, 0, 0])])
        states['a']['translation'][:] = [0, 0, -.1]
        system.project_poses()
        np.testing.assert_allclose(states['a']['translation'], [0, 0, -.05])
        np.testing.assert_allclose(states['b']['translation'], [0, 0, -.05])

    def test_validation(self):
        for limit in (0, -1, np.nan):
            with self.assertRaises(ValueError):
                FixedJoint('/J', 'a', 'b', [0, 0, 0], limit)
        with self.assertRaises(ValueError):
            FixedJoint('/J', 'a', 'a', [0, 0, 0])
        _, _, system = self.setup_system([FixedJoint('/J', 'a', 'b', [0, 0, 0])])
        with self.assertRaises(ValueError):
            system.solve(0, 0)

    def test_endpoint_order_does_not_change_solution(self):
        results = []
        for a, b in (('a', 'b'), ('b', 'a')):
            _, states, system = self.setup_system([FixedJoint('/Joint', a, b, [.2, .1, 0])])
            states['a']['velocity'][:] = [.3, .7, -.2]
            states['b']['omega'][:] = [1, 2, 3]
            system.solve(.01, .01)
            results.append(np.concatenate([np.r_[states[p]['velocity'], states[p]['omega']] for p in sorted(states)]))
        np.testing.assert_allclose(*results, atol=1e-11)

    def test_native_usd_world_fixed_joint(self):
        from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf
        from tests.test_granular import _cube
        from grasph_scene import load_scene
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'native.usda'
            stage = Usd.Stage.CreateNew(str(source))
            owner = UsdGeom.Xform.Define(stage, '/Body').GetPrim()
            _cube(stage, '/Body/Mesh', (-.1, -.1, -.1), (.1, .1, .1), 'collider')
            owner.CreateAttribute('grasph:dynamicBodyEnabled', Sdf.ValueTypeNames.Bool).Set(True)
            owner.CreateAttribute('grasph:mass', Sdf.ValueTypeNames.Double).Set(1)
            owner.CreateAttribute('grasph:inertiaDiagonal', Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(1))
            joint = UsdPhysics.FixedJoint.Define(stage, '/WorldJoint')
            # World at body1, body0 contains the actual dynamic body.
            joint.CreateBody0Rel().SetTargets(['/Body'])
            joint.CreateLocalPos0Attr(Gf.Vec3f(.1, .2, .3))
            stage.GetRootLayer().Save()
            scene = load_scene(source)
            self.assertEqual(len(scene.fixed_joints), 1)
            self.assertIsNone(scene.fixed_joints[0].body_a)
            np.testing.assert_allclose(scene.fixed_joints[0].anchor, [.1, .2, .3])

    def test_usd_scene_and_rigid_run(self):
        from pxr import Usd, UsdGeom, Gf, Sdf
        from tests.test_granular import _cube
        from grasph_scene import load_scene
        from grasph_solver import run_simulation, SolverSettings
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'joint.usda'
            stage = Usd.Stage.CreateNew(str(source))
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdGeom.SetStageMetersPerUnit(stage, 1)
            _cube(stage, '/Pillar', (-.4, -.3, 0), (-.02, .3, .2), 'collider')
            for name, x in (('A', -.11), ('B', .11)):
                owner = UsdGeom.Xform.Define(stage, '/'+name).GetPrim()
                _cube(stage, '/'+name+'/Mesh', (x-.1, -.1, .25), (x+.1, .1, .45), 'collider')
                for attr, kind, value in (
                    ('dynamicBodyEnabled', Sdf.ValueTypeNames.Bool, True),
                    ('bodyCenter', Sdf.ValueTypeNames.Double3, Gf.Vec3d(x, 0, .35)),
                    ('mass', Sdf.ValueTypeNames.Double, .5),
                    ('inertiaDiagonal', Sdf.ValueTypeNames.Double3, Gf.Vec3d(.003333333)),
                ):
                    owner.CreateAttribute('grasph:'+attr, kind).Set(value)
            joint = stage.DefinePrim('/Joint', 'Xform')
            joint.CreateRelationship('grasph:bodyA').SetTargets(['/A'])
            joint.CreateRelationship('grasph:bodyB').SetTargets(['/B'])
            for attr, kind, value in (
                ('jointType', Sdf.ValueTypeNames.String, 'fixed'),
                ('jointAnchor', Sdf.ValueTypeNames.Double3, Gf.Vec3d(0, 0, .35)),
                ('breakable', Sdf.ValueTypeNames.Bool, True),
                ('breakForce', Sdf.ValueTypeNames.Double, 1.),
                ('breakTorque', Sdf.ValueTypeNames.Double, 1.),
            ):
                joint.CreateAttribute('grasph:'+attr, kind).Set(value)
            stage.GetRootLayer().Save()
            loaded = load_scene(source)
            self.assertEqual(len(loaded.fixed_joints), 1)
            self.assertEqual(loaded.fixed_joints[0].body_a, '/A/Mesh')
            run_simulation(source, settings=SolverSettings(sim_time=.2, granular_dt=.002,
                           output_dir=directory, output_stem='result'))
            import json
            events = [json.loads(line) for line in (Path(directory)/'result.joints.jsonl').read_text().splitlines()]
            self.assertEqual(len(events), 1)
            self.assertLess(events[0]['time'], .2)
            self.assertTrue((Path(directory)/'result.usdc').is_file())


if __name__ == '__main__':
    unittest.main()
