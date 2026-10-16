"""Build the self-contained legacy Blender add-on ZIP (no venv or local assets)."""
import argparse
import ast
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SOLVER_FILES = (
    "grasph.py", "grasph_boundary.py", "grasph_collision.py", "grasph_coupling.py",
    "grasph_free_surface.py", "grasph_granular.py", "grasph_output.py", "grasph_diagnostics.py",
    "grasph_reference.py", "grasph_rigid.py", "grasph_runtime.py", "grasph_scene.py",
    "grasph_solver.py", "grasph_sph.py",
)
PACKAGE_FILES = SOLVER_FILES + (
    "requirements.txt", "requirements-accelerated.txt", "README.md",
    "blender/README.md", "blender/grasph_control.py", "blender/grasph_surface.py",
    "blender/grasph_granular_display.py", "blender/grasph_environment.py",
)


def build_package(root=ROOT, destination=None):
    root = Path(root)
    tree = ast.parse((root / "blender/grasph_control.py").read_text(encoding="utf-8"))
    version = next(ast.literal_eval(node.value) for node in tree.body
                   if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "ADDON_VERSION" for t in node.targets))
    entry = ast.parse((root / "blender/addon_init.py").read_text(encoding="utf-8"))
    info = next(ast.literal_eval(node.value) for node in entry.body
                if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "bl_info" for t in node.targets))
    if info["version"] != version:
        raise ValueError("Package bl_info version and ADDON_VERSION must match")
    destination = Path(destination or root / "output" / ("GraSPH-" + ".".join(map(str, version)) + ".zip"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(root / "blender/addon_init.py", "grasph/__init__.py")
        archive.writestr("grasph/blender/__init__.py", "")
        for name in PACKAGE_FILES:
            archive.write(root / name, "grasph/" + name)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    print(build_package(destination=parser.parse_args().out))
