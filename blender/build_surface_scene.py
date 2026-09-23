"""Background Blender entry point: import a GraSPH result and save a mesh scene.

blender --background --factory-startup --python blender/build_surface_scene.py --
    result.usdc --dp 0.02 --output result_surface.blend
"""
import argparse
from pathlib import Path
import sys

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grasph_surface import add_surface, register


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('usd')
    parser.add_argument('--dp', type=float, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--frame', type=int)
    parser.add_argument('--end', type=int)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    if args.dp <= 0:
        parser.error('--dp must be positive')
    source = Path(args.usd).resolve()
    output = Path(args.output).resolve()
    if not source.is_file() or output.suffix.lower() != '.blend':
        parser.error('Provide an existing USD and a .blend output path')
    if output.exists():
        parser.error('Output already exists; choose a new .blend filename')
    scene = bpy.data.scenes.new(source.stem + '_Surface')
    bpy.context.window.scene = scene
    bpy.ops.wm.usd_import(filepath=str(source), import_points=True)
    points = [o for o in scene.objects if o.type == 'POINTCLOUD']
    if not points:
        raise RuntimeError('No Point Clouds found in this USD')
    register()
    scene.grasph_mesh_dp = args.dp
    for cloud in points:
        add_surface(cloud, args.dp)
    for obj in scene.objects:
        if obj.name.startswith('FluidVolume') or obj.get('grasph:role') == 'fluid':
            obj.hide_set(True)
            obj.hide_render = True
    if args.end is not None:
        scene.frame_end = args.end
    scene.frame_set(args.frame if args.frame is not None else scene.frame_start)
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output))


if __name__ == '__main__':
    main()
