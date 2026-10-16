"""Blender --background --factory-startup --python tests/blender_install_smoke.py.

Load the release package from a temporary install directory, with no solver
imports from the repository. No environment installation or network access.
"""
import importlib
from pathlib import Path
import os
import sys
import tempfile
import zipfile

import bpy
import addon_utils

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "scripts"))
from build_addon import build_package
sys.path.remove(str(root / "scripts"))

with tempfile.TemporaryDirectory(prefix="grasph installed ") as folder:
    base = Path(folder)
    archive_path = build_package(destination=base / "GraSPH.zip")
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(base / "addons")
    sys.path.insert(0, str(base / "addons"))
    original = Path.cwd()
    try:
        os.chdir(base)
        # Discovery parses literal bl_info; enabling exercises the actual
        # add-on lifecycle rather than just calling register() ourselves.
        discovered = addon_utils._fake_module("grasph", str(base / "addons/grasph/__init__.py"))
        assert discovered.bl_info["version"] == (0, 15, 0)
        package = addon_utils.enable("grasph", default_set=False)
        assert package is not None
        controls = importlib.import_module("grasph.blender.grasph_control")
        settings = bpy.context.scene.grasph_scene
        installed = base / "addons/grasph"
        assert settings.use_managed_environment
        # An old blend's repository path must not override installed files.
        settings.project_root = str(root)
        assert controls._ensure_project_root(settings) == installed
        assert Path(controls._surface_module(bpy.context.scene).__file__).is_relative_to(installed)
        assert Path(controls._granular_display_module(bpy.context.scene).__file__).is_relative_to(installed)
        environment = controls._environment_module(bpy.context.scene)
        environment.user_data_path = lambda: base / "user data"
        try:
            controls._solver_python(settings, bpy.context.scene)
            raise AssertionError("Uninstalled managed environment should not launch")
        except RuntimeError as error:
            assert "Install" in str(error)
        settings.use_managed_environment = False
        settings.python_command = 'py -3.13'
        assert controls._solver_python(settings, bpy.context.scene) == ['py', '-3.13']
        # Optional real environment check through the panel operator and its
        # background-process completion callback (no dependency download).
        runtime_python = os.environ.get("GRASPH_TEST_PYTHON")
        if runtime_python:
            settings.python_command = runtime_python
            assert bpy.ops.grasph.solver_environment(action="CHECK") == {"FINISHED"}
            controls._ENV_JOB["process"].wait(timeout=60)
            controls._poll_environment_job()
            assert settings.environment_status == "Installation checked: ready", settings.environment_log
            if bpy.app.timers.is_registered(controls._poll_environment_job):
                bpy.app.timers.unregister(controls._poll_environment_job)
            runtime = Path(runtime_python).parent.parent
            environment.environment_path = lambda *args: runtime
            settings.use_managed_environment = True
            assert controls._solver_python(settings, bpy.context.scene) == [runtime_python]
            assert bpy.ops.grasph.solver_environment(action="CHECK") == {"FINISHED"}
            controls._ENV_JOB["process"].wait(timeout=60)
            controls._poll_environment_job()
            assert settings.environment_status == "Installation checked: ready", settings.environment_log
            if bpy.app.timers.is_registered(controls._poll_environment_job):
                bpy.app.timers.unregister(controls._poll_environment_job)
        (base / "models").mkdir()
        bpy.ops.wm.save_as_mainfile(filepath=str(base / "models/model.blend"))
        assert controls._path("//output") == base / "models/output"
        addon_utils.disable("grasph", default_set=False)
        assert not hasattr(bpy.types.Scene, "grasph_scene")
    finally:
        os.chdir(original)
        sys.path.remove(str(base / "addons"))
print("GraSPH installed-package smoke test passed")
