import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


env = load_module("environment_test", ROOT / "blender/grasph_environment.py")
builder = load_module("builder_test", ROOT / "scripts/build_addon.py")


class AddonPackageTests(unittest.TestCase):
    def test_package_contains_only_release_files_and_runs_outside_repo(self):
        with tempfile.TemporaryDirectory(prefix="grasph package ") as folder:
            base = Path(folder)
            package = builder.build_package(destination=base / "addon.zip")
            with zipfile.ZipFile(package) as archive:
                expected = {"grasph/" + p for p in builder.PACKAGE_FILES}
                expected.update(("grasph/__init__.py", "grasph/blender/__init__.py"))
                self.assertEqual(set(archive.namelist()), expected)
                archive.extractall(base / "installed")
            completed = subprocess.run(
                [sys.executable, str(base / "installed/grasph/grasph.py"), "--help"],
                cwd=base, env=env.child_environment(), capture_output=True, text=True, timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("--solid-coupling", completed.stdout)

    def test_requirement_changes_invalidate_readiness(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in env.REQUIREMENTS:
                (root / name).write_text("numpy\n", encoding="utf-8")
            environment = env.environment_path(root, root / "user")
            python = env.python_path(environment)
            python.parent.mkdir(parents=True)
            python.touch()
            (environment / "pyvenv.cfg").touch()
            self.assertFalse(env.is_ready(root, environment))
            (environment / env.MARKER).write_text(json.dumps(
                {"requirements": env.requirements_digest(root)}), encoding="utf-8")
            self.assertTrue(env.is_ready(root, environment))
            (root / "requirements.txt").write_text("numpy>=2\n", encoding="utf-8")
            self.assertFalse(env.is_ready(root, environment))
            self.assertNotEqual(environment, env.environment_path(root, root / "user"))

    def test_path_with_spaces_and_quoted_command(self):
        with tempfile.TemporaryDirectory(prefix="python path ") as folder:
            python = Path(folder) / "python.exe"
            python.touch()
            self.assertEqual(env.split_python_command(str(python)), [str(python)])
            self.assertEqual(env.split_python_command('"' + str(python) + '" -I'),
                             [str(python), "-I"])
        with self.assertRaises(ValueError):
            env.split_python_command("  ")

    def test_child_environment_isolated_from_blender(self):
        with patch.dict("os.environ", {"PYTHONHOME": "blender", "PYTHONPATH": "blender",
                                       "PYTHONUSERBASE": "blender"}):
            clean = env.child_environment()
        for key in ("PYTHONHOME", "PYTHONPATH", "PYTHONUSERBASE"):
            self.assertNotIn(key, clean)
        self.assertEqual(clean["PYTHONNOUSERSITE"], "1")

    def test_installer_does_not_modify_arbitrary_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "keep"
            target.mkdir()
            sentinel = target / "model.txt"
            sentinel.write_text("keep", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                env.install_environment(ROOT, target)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")
            self.assertFalse(target.with_name("keep.lock").exists())

    def test_failed_install_never_marks_ready_and_releases_lock(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "venv"
            with patch.object(env.sys, "version_info", (3, 13)), patch.object(
                    env.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "venv")):
                with self.assertRaises(subprocess.CalledProcessError):
                    env.install_environment(ROOT, target)
            self.assertFalse(env.is_ready(ROOT, target))
            self.assertFalse(target.with_name("venv.lock").exists())

    def test_install_lock_prevents_parallel_setup(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "venv"
            lock = target.with_name("venv.lock")
            lock.write_text("another installer", encoding="utf-8")
            with patch.object(env.sys, "version_info", (3, 13)), patch.object(env.subprocess, "run") as run:
                with self.assertRaisesRegex(RuntimeError, "already locked"):
                    env.install_environment(ROOT, target)
                run.assert_not_called()
            self.assertEqual(lock.read_text(encoding="utf-8"), "another installer")

    def test_requirements_hash_does_not_depend_on_line_endings(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in env.REQUIREMENTS:
                (root / name).write_bytes(b"numpy\nscipy\n")
            expected = env.requirements_digest(root)
            for name in env.REQUIREMENTS:
                (root / name).write_bytes(b"numpy\r\nscipy\r\n")
            self.assertEqual(env.requirements_digest(root), expected)


if __name__ == "__main__":
    unittest.main()
