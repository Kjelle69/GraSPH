"""External solver environment management; standard library only, no bpy.

Invoked in a child Python process. Never installs into Blender's interpreter.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys


REQUIREMENTS = ("requirements.txt", "requirements-accelerated.txt")
MARKER = "grasph-ready.json"


def requirements_digest(root):
    digest = hashlib.sha256()
    for name in REQUIREMENTS:
        digest.update(name.encode())
        digest.update((Path(root) / name).read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def environment_path(root, user_data):
    return Path(user_data) / "environments" / ("py313-" + requirements_digest(root)[:12])


def user_data_path():
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local")) / "GraSPH"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "grasph"


def python_path(environment):
    folder = "Scripts" if os.name == "nt" else "bin"
    name = "python.exe" if os.name == "nt" else "python"
    return Path(environment) / folder / name


def is_ready(root, environment):
    try:
        marker = json.loads((Path(environment) / MARKER).read_text(encoding="utf-8"))
        return ((Path(environment) / "pyvenv.cfg").is_file()
                and python_path(environment).is_file()
                and marker["requirements"] == requirements_digest(root))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def split_python_command(command):
    """Accept a bare executable path (including spaces), or command + flags."""
    import shlex
    command = str(command).strip()
    if not command:
        raise ValueError("Choose a Python executable or command")
    if Path(command).is_file():
        return [command]
    parts = shlex.split(command, posix=os.name != "nt")
    if not parts:
        raise ValueError("Choose a Python executable or command")
    # Windows Popen(list) supplies its own argument quoting.
    if os.name == "nt":
        parts = [part[1:-1] if len(part) > 1 and part[0] == part[-1] == '"'
                 else part for part in parts]
    return parts


def child_environment():
    # Do not inherit Blender's Python paths into an independent interpreter.
    env = dict(os.environ)
    for key in ("PYTHONHOME", "PYTHONPATH", "PYTHONUSERBASE"):
        env.pop(key, None)
    env.update(PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1")
    return env


def check_environment(root):
    import importlib.metadata as metadata
    import numpy as np
    from scipy.spatial import cKDTree
    from pxr import Usd

    # Exercise native libraries, not just package metadata/imports.
    assert len(cKDTree(np.array([[0., 0., 0.], [1., 0., 0.]])).query_pairs(2)) == 1
    assert Usd.Stage.CreateInMemory()
    if metadata.version("usd-core") != "26.8":
        raise RuntimeError("GraSPH requires usd-core==26.8")
    if tuple(int(p) for p in metadata.version("scipy").split(".")[:2]) < (1, 11):
        raise RuntimeError("GraSPH requires scipy>=1.11")
    subprocess.run([sys.executable, str(Path(root) / "grasph.py"), "--help"],
                   check=True, stdout=subprocess.DEVNULL, env=child_environment())
    result = {"python": sys.executable, "version": sys.version.split()[0],
              "numpy": metadata.version("numpy"), "scipy": metadata.version("scipy"),
              "usd-core": metadata.version("usd-core")}
    print(json.dumps(result, indent=2), flush=True)
    return result


def install_environment(root, environment):
    root, environment = Path(root).resolve(), Path(environment).resolve()
    if sys.version_info[:2] != (3, 13) or struct.calcsize("P") != 8:
        raise RuntimeError("Managed setup needs 64-bit CPython 3.13. Install it and select its command.")
    environment.parent.mkdir(parents=True, exist_ok=True)
    lock = environment.with_name(environment.name + ".lock")
    try:
        handle = lock.open("x", encoding="utf-8")
    except FileExistsError as error:
        raise RuntimeError(f"Setup is already locked: {lock}. If an earlier setup was interrupted, "
                           "confirm it is no longer running before removing that lock.") from error
    try:
        with handle:
            handle.write(str(os.getpid()))
        marker = environment / MARKER
        if environment.exists() and not (environment / "pyvenv.cfg").is_file():
            raise RuntimeError(f"Refusing to modify a directory that is not a venv: {environment}")
        marker.unlink(missing_ok=True)
        print(f"Creating solver environment: {environment}", flush=True)
        subprocess.run([sys.executable, "-m", "venv", str(environment)],
                       check=True, env=child_environment())
        python = str(python_path(environment))
        command = [python, "-m", "pip", "--isolated", "install", "--no-input", "--disable-pip-version-check",
                   "--only-binary=:all:", "--index-url", "https://pypi.org/simple"]
        for name in REQUIREMENTS:
            command.extend(["-r", str(root / name)])
        subprocess.run(command, check=True, env=child_environment())
        subprocess.run([python, str(Path(__file__).resolve()), "check", "--root", str(root)],
                       check=True, env=child_environment())
        marker.write_text(json.dumps({"requirements": requirements_digest(root)}), encoding="utf-8")
        print("GraSPH solver environment is ready.", flush=True)
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("install", "check"))
    parser.add_argument("--root", required=True)
    parser.add_argument("--environment")
    args = parser.parse_args()
    if args.action == "install":
        if not args.environment:
            parser.error("install requires --environment")
        install_environment(args.root, args.environment)
    else:
        check_environment(args.root)
        if args.environment:
            environment = Path(args.environment).resolve()
            if not (environment / "pyvenv.cfg").is_file() or (
                    Path(sys.executable).resolve() != python_path(environment).resolve()):
                raise RuntimeError("Check was not run with the managed venv interpreter")
            (environment / MARKER).write_text(
                json.dumps({"requirements": requirements_digest(args.root)}), encoding="utf-8")


if __name__ == "__main__":
    main()
