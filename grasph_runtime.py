"""Small cross-platform controls shared by the Blender-launched solver."""

from pathlib import Path
import time

_next_pause_check = 0.0


def wait_if_paused(control_file, poll_interval=0.1):
    """Block at a solver-step boundary while the pause sentinel exists."""
    global _next_pause_check
    if not control_file:
        return
    now = time.monotonic()
    if now < _next_pause_check:
        return
    _next_pause_check = now + poll_interval
    path = Path(control_file)
    if not path.exists():
        return
    print("Simulation paused; waiting for resume...", flush=True)
    while path.exists():
        time.sleep(poll_interval)
    print("Simulation resumed.", flush=True)
