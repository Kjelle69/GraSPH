import tempfile
import unittest
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom

from grasph_scene import load_scene, generate_granular_particles
from grasph_solver import SolverSettings, run_simulation
from grasph_sph import SPHConfig


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

    def test_dynamic_body_falls_and_authors_transform(self):
        with tempfile.TemporaryDirectory() as directory:
            source_file = Path(directory) / "dynamic_body.usda"
            stage = Usd.Stage.CreateNew(str(source_file))
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdGeom.SetStageMetersPerUnit(stage, 1.0)

            # A dormant flow source selects granular mode without adding grains.
            source = _cube(
                stage, "/Flow", (-0.05, -0.05, 1.0),
                (0.05, 0.05, 1.1), "granular_source",
            )
            for name, value_type, value in (
                ("grasph:sourceMode", Sdf.ValueTypeNames.String, "flow"),
                ("grasph:particleRadius", Sdf.ValueTypeNames.Double, 0.02),
                ("grasph:startTime", Sdf.ValueTypeNames.Double, 1.0),
                ("grasph:endTime", Sdf.ValueTypeNames.Double, 2.0),
            ):
                source.CreateAttribute(name, value_type).Set(value)

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

            result = run_simulation(
                source_file,
                sph_cfg=SPHConfig(dp=0.05),
                settings=SolverSettings(
                    sim_time=0.25, output_fps=20.0, output_dir=directory,
                    output_stem="granular_result", granular_dt=0.002,
                    granular_iterations=4, report_interval=100.0,
                ),
            )
            self.assertEqual(len(result.positions), 8)
            self.assertTrue(np.all(np.isfinite(result.positions)))
            self.assertGreaterEqual(float(result.positions[:, 2].min()), 0.035)
            output = Usd.Stage.Open(result.usd_file)
            prim = output.GetPrimAtPath("/GraSPH_GranularResult")
            self.assertTrue(prim.IsValid())
            self.assertEqual(prim.GetAttribute("grasph:resultKind").Get(), "granular")

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
