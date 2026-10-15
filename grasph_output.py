from pathlib import Path
import os
import numpy as np
from pxr import Usd, UsdGeom, Gf, Sdf


def write_usd_animation(stage, frames, frame_times, fps, dp, output_file,
                        source_start_time_code=None, source_tps=None,
                        hinge_frames=None, frame_velocities=None,
                        prim_path="/GraSPH_Result", particle_widths=None,
                        result_kind="fluid", body_frames=None,
                        extra_particle_sets=None):
    """Write GraSPH Points as a USD overlay on the original source scene.

    v0.54 FIX2 deliberately does *not* flatten/copy the source scene. Instead
    the result layer sublayers the exact original USD file and authors only
    GraSPH_Result plus timeline metadata in the stronger result layer. This is
    the most robust way to preserve Blender-authored time samples/keyframes:
    the source animation is not rewritten at all.

    Consequence: the result USDC depends on the original source USD remaining
    at its referenced path. A self-contained packaging step can be added later.
    """
    output_file = str(output_file)
    out_path = Path(output_file).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    source_layer = stage.GetRootLayer()
    source_identifier = source_layer.realPath or source_layer.identifier
    if not source_identifier:
        raise RuntimeError("Source USD layer has no usable file path; cannot create result overlay.")
    source_path = Path(source_identifier).resolve()

    # Fresh result stage. The original Blender USD is composed as a sublayer
    # and therefore retains its exact authored transform time samples.
    result_stage = Usd.Stage.CreateNew(str(out_path))

    # FIX4: a newly-created USD stage defaults to Y-up and centimeters unless
    # stage coordinate metadata is authored explicitly.  Blender then imports
    # GraSPH_Result with a 90 degree conversion and 0.01 scale even though the
    # source Blender stage is Z-up and meters.  Copy those stage-level spatial
    # conventions from the source BEFORE authoring the overlay points.
    source_up_axis = UsdGeom.GetStageUpAxis(stage)
    source_mpu = UsdGeom.GetStageMetersPerUnit(stage)
    UsdGeom.SetStageUpAxis(result_stage, source_up_axis)
    UsdGeom.SetStageMetersPerUnit(result_stage, float(source_mpu))

    rel_source = os.path.relpath(str(source_path), start=str(out_path.parent)).replace('\\', '/')
    result_stage.GetRootLayer().subLayerPaths.append(rel_source)

    source_start = stage.GetStartTimeCode() if source_start_time_code is None else float(source_start_time_code)
    source_rate = stage.GetTimeCodesPerSecond() if source_tps is None else float(source_tps)
    source_fps = stage.GetFramesPerSecond()
    source_end = stage.GetEndTimeCode()
    if not np.isfinite(source_rate) or source_rate <= 0.0:
        source_rate = float(fps)
    if not np.isfinite(source_fps) or source_fps <= 0.0:
        source_fps = source_rate

    # Keep simulation points separate from source prim hierarchy. This avoids
    # authoring stronger specs on /root or Blender-created parent prims.
    points_prim = UsdGeom.Points.Define(result_stage, prim_path)
    points_prim.GetPrim().CreateAttribute(
        "grasph:resultKind", Sdf.ValueTypeNames.String
    ).Set(str(result_kind))
    points_attr = points_prim.CreatePointsAttr()
    widths_attr = points_prim.CreateWidthsAttr()
    velocities_attr = points_prim.CreateVelocitiesAttr() if frame_velocities is not None else None

    usd_times = [source_start + float(t) * source_rate for t in frame_times]
    for frame_index, (tc, positions) in enumerate(zip(usd_times, frames)):
        usd_points = [Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in positions]
        points_attr.Set(usd_points, tc)
        if particle_widths is None:
            widths = [float(dp)] * len(positions)
        else:
            width_values = np.asarray(particle_widths, dtype=np.float64)
            if width_values.ndim == 0:
                widths = [float(width_values)] * len(positions)
            else:
                widths = [float(v) for v in width_values[:len(positions)]]
        widths_attr.Set(widths, tc)
        if velocities_attr is not None:
            velocities = frame_velocities[frame_index]
            usd_velocities = [
                Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in velocities
            ]
            velocities_attr.Set(usd_velocities, tc)

    for particle_set in (extra_particle_sets or []):
        extra = UsdGeom.Points.Define(result_stage, particle_set["prim_path"])
        extra.GetPrim().CreateAttribute(
            "grasph:resultKind", Sdf.ValueTypeNames.String
        ).Set(str(particle_set.get("result_kind", "particles")))
        extra_points = extra.CreatePointsAttr()
        extra_widths = extra.CreateWidthsAttr()
        extra_velocities = (
            extra.CreateVelocitiesAttr()
            if particle_set.get("velocities") is not None else None
        )
        widths_data = particle_set.get("widths", 1.0)
        for frame_index, (tc, positions) in enumerate(
            zip(usd_times, particle_set["frames"])
        ):
            extra_points.Set([
                Gf.Vec3f(float(p[0]), float(p[1]), float(p[2]))
                for p in positions
            ], tc)
            widths = (
                widths_data[frame_index]
                if isinstance(widths_data, (list, tuple)) else widths_data
            )
            widths = np.asarray(widths, dtype=np.float64)
            if widths.ndim == 0:
                widths = np.full(len(positions), float(widths))
            extra_widths.Set([float(value) for value in widths[:len(positions)]], tc)
            if extra_velocities is not None:
                extra_velocities.Set([
                    Gf.Vec3f(float(v[0]), float(v[1]), float(v[2]))
                    for v in particle_set["velocities"][frame_index]
                ], tc)

    # Overlay the simulated axle rotation on the Blender-authored rotateXYZ
    # operation. The source layer still owns translation, scale, materials and
    # geometry; this stronger layer only supplies time-varying wheel angles.
    for data in (hinge_frames or {}).values():
        driver_path = data["driver_path"]
        source_driver = stage.GetPrimAtPath(driver_path)
        source_rotate = source_driver.GetAttribute("xformOp:rotateXYZ")
        base = source_rotate.Get(source_start) if source_rotate else None
        base = np.asarray(base if base is not None else (0.0, 0.0, 0.0), dtype=np.float64)
        axis = np.asarray(data["axis"], dtype=np.float64)
        component = int(np.argmax(np.abs(axis)))
        sign = float(np.sign(axis[component]))
        driver = result_stage.OverridePrim(driver_path)
        rotate_attr = driver.CreateAttribute("xformOp:rotateXYZ", Sdf.ValueTypeNames.Float3)
        for tc, angle in zip(usd_times, data["angles"]):
            value = base.copy()
            value[component] += sign * np.degrees(float(angle))
            rotate_attr.Set(Gf.Vec3f(*map(float, value)), tc)

    # Free rigid bodies use the Blender-exported translate/rotateXYZ ops. The
    # result layer supplies only simulated deltas; mesh, scale and materials
    # continue to come from the source sublayer.
    for data in (body_frames or {}).values():
        driver_path = data["driver_path"]
        source_driver = stage.GetPrimAtPath(driver_path)
        source_translate = source_driver.GetAttribute("xformOp:translate")
        source_rotate = source_driver.GetAttribute("xformOp:rotateXYZ")
        base_translate = source_translate.Get(source_start) if source_translate else None
        base_rotate = source_rotate.Get(source_start) if source_rotate else None
        base_translate = np.asarray(
            base_translate if base_translate is not None else (0.0, 0.0, 0.0),
            dtype=np.float64,
        )
        base_rotate = np.asarray(
            base_rotate if base_rotate is not None else (0.0, 0.0, 0.0),
            dtype=np.float64,
        )
        driver = result_stage.OverridePrim(driver_path)
        translate_type = (
            source_translate.GetTypeName() if source_translate
            else Sdf.ValueTypeNames.Double3
        )
        rotate_type = (
            source_rotate.GetTypeName() if source_rotate
            else Sdf.ValueTypeNames.Float3
        )
        translate_attr = driver.CreateAttribute("xformOp:translate", translate_type)
        rotate_attr = driver.CreateAttribute("xformOp:rotateXYZ", rotate_type)
        for tc, translation, rotation in zip(
            usd_times, data["translations"], data["rotations"]
        ):
            translated = base_translate + np.asarray(translation, dtype=np.float64)
            rotated = base_rotate + np.degrees(np.asarray(rotation, dtype=np.float64))
            translate_attr.Set(Gf.Vec3d(*map(float, translated)), tc)
            rotate_attr.Set(Gf.Vec3f(*map(float, rotated)), tc)

    result_stage.SetStartTimeCode(min(source_start, usd_times[0] if usd_times else source_start))
    result_stage.SetEndTimeCode(max(source_end, usd_times[-1] if usd_times else source_end))
    result_stage.SetTimeCodesPerSecond(float(source_rate))
    result_stage.SetFramesPerSecond(float(source_fps))

    result_stage.GetRootLayer().Save()

    print(f"USD source timing preserved: start={source_start:g}, tps={source_rate:g}")
    print(f"USD spatial metadata: upAxis={source_up_axis}, metersPerUnit={float(source_mpu):g}")
    print(f"USD frames: {len(frames)}")
    print(f"USD output: simulation overlay; sublayer source = {rel_source}")
    print("USD source animation is preserved verbatim (not flattened/re-authored)")
    print(f"Saved: {output_file}")


def save_npz(output_file, **arrays):
    output_file = str(output_file)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_file, **arrays)
    print(f"Saved: {output_file}")


def write_usd_checkpoint(stage, frames, frame_times, fps, dp, output_file, **kwargs):
    """Atomically replace a partial USD animation without touching final output."""
    import gc

    output_path = Path(output_file).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.stem + ".tmp" + output_path.suffix)
    try:
        temporary.unlink(missing_ok=True)
        write_usd_animation(
            stage, frames, frame_times, fps, dp, temporary, **kwargs
        )
        gc.collect()
        os.replace(temporary, output_path)
    except PermissionError as error:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        print(f"USD checkpoint postponed; file is in use: {error}")
        return False
    print(f"Updated partial USD checkpoint: {output_path}")
    return True
