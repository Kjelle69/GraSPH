from pathlib import Path
import os
import numpy as np
from pxr import Usd, UsdGeom, Gf


def write_usd_animation(stage, frames, frame_times, fps, dp, output_file, source_start_time_code=None, source_tps=None):
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
    points_prim = UsdGeom.Points.Define(result_stage, "/GraSPH_Result")
    points_attr = points_prim.CreatePointsAttr()
    widths_attr = points_prim.CreateWidthsAttr()
    widths_attr.Set([float(dp)] * len(frames[0]))

    usd_times = [source_start + float(t) * source_rate for t in frame_times]
    for tc, positions in zip(usd_times, frames):
        usd_points = [Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in positions]
        points_attr.Set(usd_points, tc)

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
