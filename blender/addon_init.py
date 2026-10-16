"""Package entry point used by the GraSPH release builder."""

# Keep this literal: Blender discovers add-on metadata without importing code.
bl_info = {
    "name": "GraSPH Controls",
    "author": "GraSPH",
    "version": (0, 15, 0),
    "blender": (4, 5, 0),
    "location": "3D View > Sidebar > GraSPH",
    "description": "Tag, validate, export and simulate GraSPH USD scenes",
    "category": "Physics",
}

from .blender.grasph_control import register, unregister
