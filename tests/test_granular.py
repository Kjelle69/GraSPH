import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from grasph_scene import load_scene, generate_granular_particles
from grasph_solver import SolverSettings, run_simulation
from grasph_sph import SPHConfig
from grasph_rigid import constrain_body_state


def _cube(stage, path, lo, hi, role):
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    points = [
        (lo[0], lo[1], lo[2]), (hi[0], lo[1], lo[2]),
        (hi[0], hi[1], lo[2]), (lo[0], hi[1], lo[2]),
        (lo[0], lo[1], hi[2]), (hi[0], lo[1], hi[2]),
        (hi[0], hi[1], hi[2]), (lo[0], hi[1], hi[2]),
    ]
    faces = [
        (0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
        (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
    ]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in points])
    mesh.CreateFaceVertexCountsAttr([4] * len(faces))
    mesh.CreateFaceVertexIndicesAttr([item for face in faces for item in face])
    mesh.GetPrim().CreateAttribute("grasph:role", Sdf.ValueTypeNames.String).Set(role)
    return mesh.GetPrim()


class GranularTests(unittest.TestCase):
    def test_dynamic_body_reads_joint_type_axis_and_anchor(self):
        with tempfile.TemporaryDirectory() as directory:
            source_file = Path(directory) / "prismatic.usda"
            stage = Usd.Stage.CreateNew(str(source_file))
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdGeom.SetStageMetersPerUnit(stage, 1.0)
            body = UsdGeom.Xform.Define(stage, "/Slider").GetPrim()
            _cube(stage, "/Slider/Geom", (-0.1, -0.1, -0.1),
                  (0.1, 0.1, 0.1), "collider")
            for name, value_type, value in (
                ("grasph:dynamicBodyEnabled", Sdf.ValueTypeNames.Bool, True),
                ("grasph:bodyCenter", Sdf.ValueTypeNames.Double3, Gf.Vec3d(0.0)),
                ("grasph:mass", Sdf.ValueTypeNames.Double, 1.0),
                ("grasph:inertiaDiagonal", Sdf.ValueTypeNames.Double3, Gf.Vec3d(1.0)),
                ("grasph:jointType", Sdf.ValueTypeNames.String, "prismatic"),
                ("grasph:jointAxis", Sdf.ValueTypeNames.Double3, Gf.Vec3d(0.0, 1.0, 0.0)),
                ("grasph:jointAnchor", Sdf.ValueTypeNames.Double3, Gf.Vec3d(0.0, 0.0, 0.5)),
            ):
                body.CreateAttribute(name, value_type).Set(value)
            stage.GetRootLayer().Save()
            scene = load_scene(source_file)
            slider = next(iter(scene.dynamic_bodies.values()))
            self.assertEqual(slider.joint_type, "prismatic")
            np.testing.assert_allclose(slider.joint_axis, (0.0, 1.0, 0.0))
            np.testing.assert_allclose(slider.joint_anchor, (0.0, 0.0, 0.5))

    def test_prismatic_joint_projects_motion_to_its_axis(self):
        body = SimpleNamespace(
            path="/Slider", joint_type="prismatic",
            joint_axis=np.asarray((0.0, 2.0, 0.0)),
        )
        state = {
            "translation": np.asarray((1.0, 2.0, 3.0)),
            "velocity": np.asarray((4.0, 5.0, 6.0)),
            "rotation": np.eye(3),
            "omega": np.asarray((0.1, 0.2, 0.3)),
        }
        constrain_body_state(body, state)
        np.testing.assert_allclose(state["translation"], (0.0, 2.0, 0.0))
        np.testing.assert_allclose(state["velocity"], (0.0, 5.0, 0.0))
        np.testing.assert_allclose(state["omega"], (0.0, 0.0, 0.0))
        np.testing.assert_allclose(state["rotation"], np.eye(3))

    def test_fixed_joint_locks_all_body_dofs(self):
        body = SimpleNamespace(path="/Fixed", joint_type="fixed")
        state = {
            "translation": np.asarray((1.0, 2.0, 3.0)),
            "velocity": np.asarray((4.0, 5.0, 6.0)),
            "rotation": np.diag((-1.0, 1.0, -1.0)),
            "omega": np.asarray((0.1, 0.2, 0.3)),
        }
        constrain_body_state(body, state)
        np.testing.assert_allclose(state["translation"], (0.0, 0.0, 0.0))
        np.testing.assert_allclose(state["velocity"], (0.0, 0.0, 0.0))
        np.testing.assert_allclose(state["omega"], (0.0, 0.0, 0.0))
        np.testing.assert_allclose(state["rotation"], np.eye(3))

    def test_standard_usd_physics_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            source_file = Path(directory) / "standard_physics.usda"
            stage = Usd.Stage.CreateNew(str(source_file))
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")

            hinged_owner = UsdGeom.Xform.Define(stage, "/Hinged").GetPrim()
            hinged_mesh = _cube(
                stage, "/Hinged/Geom", (-0.1, -0.1, 0.4),
                (0.1, 0.1, 0.6), "collider",
            )
            hinged_mesh.RemoveProperty("grasph:role")
            UsdPhysics.RigidBodyAPI.Apply(hinged_owner)
            UsdPhysics.CollisionAPI.Apply(hinged_mesh)
            hinge_mass = UsdPhysics.MassAPI.Apply(hinged_owner)
            hinge_mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.2, 0.3, 0.4))
            joint = UsdPhysics.RevoluteJoint.Define(stage, "/HingeJoint")
            joint.CreateBody1Rel().SetTargets([hinged_owner.GetPath()])
            joint.CreateAxisAttr(UsdPhysics.Tokens.y)
            joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.5))

            free_owner = UsdGeom.Xform.Define(stage, "/Free").GetPrim()
            free_mesh = _cube(
                stage, "/Free/Geom", (0.4, -0.1, 0.4),
                (0.6, 0.1, 0.6), "collider",
            )
            free_mesh.RemoveProperty("grasph:role")
            UsdPhysics.RigidBodyAPI.Apply(free_owner)
            UsdPhysics.CollisionAPI.Apply(free_mesh)
            free_mass = UsdPhysics.MassAPI.Apply(free_owner)
            free_mass.CreateMassAttr(2.0)
            free_mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.1, 0.2, 0.3))
            material = UsdShade.Material.Define(stage, "/PhysicsMaterial")
            material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
            material_api.CreateDynamicFrictionAttr(0.72)
            material_api.CreateRestitutionAttr(0.18)
            UsdShade.MaterialBindingAPI.Apply(free_mesh).Bind(
                material, UsdShade.Tokens.weakerThanDescendants, "physics"
            )
            stage.GetRootLayer().Save()

            scene = load_scene(source_file)
            self.assertEqual(len(scene.hinges), 1)
            self.assertEqual(len(scene.dynamic_bodies), 1)
            hinge = next(iter(scene.hinges.values()))
            body = next(iter(scene.dynamic_bodies.values()))
            np.testing.assert_allclose(hinge.axis, (0.0, 1.0, 0.0))
            self.assertAlmostEqual(hinge.moment_of_inertia, 0.3)
            self.assertAlmostEqual(body.mass, 2.0)
            np.testing.assert_allclose(body.inertia, (0.1, 0.2, 0.3))
            self.assertAlmostEqual(scene.collider_materials["/Free/Geom"]["friction"], 0.72)
            self.assertAlmostEqual(scene.collider_materials["/Free/Geom"]["restitution"], 0.18)

    def test_small_mixed_grain_does_not_tunnel_through_coarse_collider(self):
        with tempfile.TemporaryDirectory() as directory:
            source_file = Path(directory) / "mixed_floor.usda"
            stage = Usd.Stage.CreateNew(str(source_file))
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdGeom.SetStageMetersPerUnit(stage, 1.0)
            _cube(stage, "/Fluid", (0.34, -0.06, 0.04), (0.46, 0.06, 0.16), "fluid")
            grains = _cube(
                stage, "/Grain", (-0.006, -0.006, 0.194),
                (0.006, 0.006, 0.206), "granular_source",
            )
            grains.CreateAttribute(
                "grasph:particleRadius", Sdf.ValueTypeNames.Double
            ).Set(0.005)
            _cube(stage, "/Floor", (-0.5, -0.3, -0.05), (0.5, 0.3, 0.0), "collider")
            stage.GetRootLayer().Save()

            result = run_simulation(
                source_file, sph_cfg=SPHConfig(dp=0.05),
                settings=SolverSettings(
                    sim_time=0.3, output_fps=20.0, output_dir=directory,
                    output_stem="mixed_floor_result", report_interval=100.0,
                ),
            )
            with np.load(result.npz_file) as data:
                final_count = int(data["granular_active_counts"][-1])
                final_z = float(data["granular_positions"][-1, 0, 2])
            self.assertEqual(final_count, 1)
            self.assertGreater(final_z, 0.0)
            self.assertLess(final_z, 0.05)

    def test_mixed_fluid_granular_writes_both_results(self):
        with tempfile.TemporaryDirectory() as directory:
            source_file = Path(directory) / "mixed.usda"
            stage = Usd.Stage.CreateNew(str(source_file))
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdGeom.SetStageMetersPerUnit(stage, 1.0)
            _cube(
                stage, "/Fluid", (-0.05, -0.05, 0.0),
                (0.05, 0.05, 0.10), "fluid",
            )
            grains = _cube(
                stage, "/Grains", (-0.04, -0.04, 0.12),
                (0.04, 0.04, 0.20), "granular_source",
            )
            grains.CreateAttribute(
                "grasph:particleRadius", Sdf.ValueTypeNames.Double
            ).Set(0.02)
            grains.CreateAttribute(
                "grasph:density", Sdf.ValueTypeNames.Double
            ).Set(1600.0)
            stage.GetRootLayer().Save()

            preview_file = Path(directory) / "mixed_preview.npz"
            checkpoint_file = Path(directory) / "mixed_checkpoint.usdc"
            result = run_simulation(
                source_file, sph_cfg=SPHConfig(dp=0.05),
                settings=SolverSettings(
                    sim_time=0.02, output_fps=100.0, output_dir=directory,
                    output_stem="mixed_result", report_interval=100.0,
                    preview_file=str(preview_file), preview_interval=0.05,
                    usd_checkpoint_file=str(checkpoint_file),
                    usd_checkpoint_interval=0.25,
                ),
            )
            output = Usd.Stage.Open(result.usd_file)
            fluid = output.GetPrimAtPath("/GraSPH_Result")
            granular = output.GetPrimAtPath("/GraSPH_GranularResult")
            self.assertTrue(fluid.IsValid())
            self.assertTrue(granular.IsValid())
            self.assertEqual(
                granular.GetAttribute("grasph:resultKind").Get(), "granular"
            )
            self.assertGreater(
                len(UsdGeom.Points(granular).GetPointsAttr().GetTimeSamples()), 1
            )
            self.assertTrue(preview_file.is_file())
            checkpoint = Usd.Stage.Open(str(checkpoint_file))
            checkpoint_fluid = UsdGeom.Points.Get(checkpoint, "/GraSPH_Result")
            checkpoint_granular = UsdGeom.Points.Get(
                checkpoint, "/GraSPH_GranularResult"
            )
            self.assertTrue(checkpoint_fluid)
            self.assertTrue(checkpoint_granular)
            self.assertLess(
                checkpoint.GetEndTimeCode(), output.GetEndTimeCode()
            )
            self.assertFalse(preview_file.with_name(preview_file.name + ".tmp").exists())
            with np.load(preview_file) as preview:
                self.assertAlmostEqual(float(preview["progress"]), 1.0)
                self.assertGreater(len(preview["positions"]), 0)
                self.assertGreater(len(preview["granular_positions"]), 0)

    def test_fluid_dynamic_body_coupling_is_optional(self):
        with tempfile.TemporaryDirectory() as directory:
            source_file = Path(directory) / "fluid_body.usda"
            stage = Usd.Stage.CreateNew(str(source_file))
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdGeom.SetStageMetersPerUnit(stage, 1.0)
            _cube(
                stage, "/Fluid", (-0.05, -0.05, 0.0),
                (0.05, 0.05, 0.10), "fluid",
            )
            driver = UsdGeom.Xform.Define(stage, "/Body")
            driver.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
            driver.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            body = _cube(
                stage, "/Body/Geom", (0.95, -0.05, 0.25),
                (1.05, 0.05, 0.35), "collider",
            )
            for name, value_type, value in (
                ("grasph:dynamicBodyEnabled", Sdf.ValueTypeNames.Bool, True),
                ("grasph:bodyCenter", Sdf.ValueTypeNames.Double3, Gf.Vec3d(1.0, 0.0, 0.3)),
                ("grasph:mass", Sdf.ValueTypeNames.Double, 1.0),
                ("grasph:inertiaDiagonal", Sdf.ValueTypeNames.Double3, Gf.Vec3d(0.01, 0.01, 0.01)),
                ("grasph:collisionShape", Sdf.ValueTypeNames.String, "convex"),
            ):
                body.CreateAttribute(name, value_type).Set(value)
            stage.GetRootLayer().Save()

            result = run_simulation(
                source_file, sph_cfg=SPHConfig(dp=0.05),
                settings=SolverSettings(
                    sim_time=0.02, output_fps=100.0, output_dir=directory,
                    output_stem="coupled", enable_solid_coupling=True,
                    report_interval=100.0,
                ),
            )
            output = Usd.Stage.Open(result.usd_file)
            translate = output.GetPrimAtPath("/Body").GetAttribute("xformOp:translate")
            samples = translate.GetTimeSamples()
            self.assertGreaterEqual(len(samples), 2)
            self.assertLess(float(translate.Get(samples[-1])[2]), 0.0)

    def test_dynamic_body_only_scene_runs_and_authors_transform(self):
        with tempfile.TemporaryDirectory() as directory:
            source_file = Path(directory) / "dynamic_body.usda"
            stage = Usd.Stage.CreateNew(str(source_file))
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdGeom.SetStageMetersPerUnit(stage, 1.0)

            body_driver = UsdGeom.Xform.Define(stage, "/Body")
            body_driver.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
            body_driver.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            body = _cube(
                stage, "/Body/Geom", (-0.05, -0.05, 0.25),
                (0.05, 0.05, 0.35), "collider",
            )
            for name, value_type, value in (
                ("grasph:dynamicBodyEnabled", Sdf.ValueTypeNames.Bool, True),
                ("grasph:bodyCenter", Sdf.ValueTypeNames.Double3, Gf.Vec3d(0.0, 0.0, 0.3)),
                ("grasph:mass", Sdf.ValueTypeNames.Double, 1.0),
                ("grasph:inertiaDiagonal", Sdf.ValueTypeNames.Double3, Gf.Vec3d(0.01, 0.01, 0.01)),
                ("grasph:collisionRadius", Sdf.ValueTypeNames.Double, 0.05),
            ):
                body.CreateAttribute(name, value_type).Set(value)
            _cube(stage, "/Floor", (-0.4, -0.4, -0.05), (0.4, 0.4, 0.0), "collider")
            stage.GetRootLayer().Save()

            result = run_simulation(
                source_file, sph_cfg=SPHConfig(dp=0.05),
                settings=SolverSettings(
                    sim_time=0.3, output_fps=20.0, output_dir=directory,
                    output_stem="dynamic_result", granular_dt=0.002,
                    report_interval=100.0,
                ),
            )
            output = Usd.Stage.Open(result.usd_file)
            translate = output.GetPrimAtPath("/Body").GetAttribute("xformOp:translate")
            self.assertGreater(len(translate.GetTimeSamples()), 2)
            final = translate.Get(translate.GetTimeSamples()[-1])
            self.assertLess(float(final[2]), 0.0)
            self.assertGreater(float(final[2]), -0.30)
            with np.load(result.npz_file) as data:
                self.assertEqual(data["positions"].shape[1], 0)

    def test_volume_source_falls_and_writes_granular_usd(self):
        with tempfile.TemporaryDirectory() as directory:
            source_file = Path(directory) / "granular.usda"
            stage = Usd.Stage.CreateNew(str(source_file))
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdGeom.SetStageMetersPerUnit(stage, 1.0)
            source = _cube(
                stage, "/GranularSource", (-0.09, -0.09, 0.15),
                (0.09, 0.09, 0.33), "granular_source",
            )
            source.CreateAttribute(
                "grasph:particleRadius", Sdf.ValueTypeNames.Double
            ).Set(0.04)
            source.CreateAttribute("grasph:density", Sdf.ValueTypeNames.Double).Set(1600.0)
            source.CreateAttribute("grasph:friction", Sdf.ValueTypeNames.Double).Set(0.45)
            source.CreateAttribute("grasph:restitution", Sdf.ValueTypeNames.Double).Set(0.05)
            _cube(stage, "/Floor", (-0.4, -0.4, -0.05), (0.4, 0.4, 0.0), "collider")
            stage.GetRootLayer().Save()

            scene = load_scene(source_file)
            self.assertEqual(len(scene.granular_sources), 1)
            initial, source_ids = generate_granular_particles(scene)
            self.assertEqual(len(initial), 8)
            self.assertTrue(np.all(source_ids == 0))

            checkpoint_file = Path(directory) / "granular_checkpoint.usdc"
            result = run_simulation(
                source_file,
                sph_cfg=SPHConfig(dp=0.05),
                settings=SolverSettings(
                    sim_time=0.25, output_fps=20.0, output_dir=directory,
                    output_stem="granular_result", granular_dt=0.002,
                    granular_iterations=4, report_interval=100.0,
                    usd_checkpoint_file=str(checkpoint_file),
                    usd_checkpoint_interval=0.25,
                ),
            )
            self.assertEqual(len(result.positions), 8)
            self.assertTrue(np.all(np.isfinite(result.positions)))
            self.assertGreaterEqual(float(result.positions[:, 2].min()), 0.035)
            output = Usd.Stage.Open(result.usd_file)
            prim = output.GetPrimAtPath("/GraSPH_GranularResult")
            self.assertTrue(prim.IsValid())
            self.assertEqual(prim.GetAttribute("grasph:resultKind").Get(), "granular")
            checkpoint = Usd.Stage.Open(str(checkpoint_file))
            checkpoint_prim = checkpoint.GetPrimAtPath("/GraSPH_GranularResult")
            self.assertTrue(checkpoint_prim.IsValid())
            self.assertLess(checkpoint.GetEndTimeCode(), output.GetEndTimeCode())

    def test_flow_source_adds_particles_over_time(self):
        with tempfile.TemporaryDirectory() as directory:
            source_file = Path(directory) / "flow.usda"
            stage = Usd.Stage.CreateNew(str(source_file))
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdGeom.SetStageMetersPerUnit(stage, 1.0)
            source = _cube(
                stage, "/GranularFlowSource", (-0.09, -0.09, 0.3),
                (0.09, 0.09, 0.48), "granular_source",
            )
            attributes = (
                ("grasph:sourceMode", Sdf.ValueTypeNames.String, "flow"),
                ("grasph:particleRadius", Sdf.ValueTypeNames.Double, 0.04),
                ("grasph:density", Sdf.ValueTypeNames.Double, 1600.0),
                ("grasph:friction", Sdf.ValueTypeNames.Double, 0.4),
                ("grasph:restitution", Sdf.ValueTypeNames.Double, 0.05),
                ("grasph:flowRate", Sdf.ValueTypeNames.Double, 25.0),
                ("grasph:initialVelocity", Sdf.ValueTypeNames.Double3,
                 Gf.Vec3d(0.0, 0.0, -1.0)),
                ("grasph:startTime", Sdf.ValueTypeNames.Double, 0.0),
                ("grasph:endTime", Sdf.ValueTypeNames.Double, 0.1),
            )
            for name, value_type, value in attributes:
                source.CreateAttribute(name, value_type).Set(value)
            stage.GetRootLayer().Save()

            result = run_simulation(
                source_file, sph_cfg=SPHConfig(dp=0.05),
                settings=SolverSettings(
                    sim_time=0.12, output_fps=25.0, output_dir=directory,
                    output_stem="flow_result", granular_dt=0.002,
                    report_interval=100.0,
                ),
            )
            self.assertEqual(len(result.positions), 2)
            with np.load(result.npz_file) as cache:
                self.assertEqual(int(cache["active_counts"][0]), 0)
                self.assertEqual(int(cache["active_counts"][-1]), 2)


if __name__ == "__main__":
    unittest.main()
