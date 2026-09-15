from pathlib import Path
import argparse
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grasph_sph import SPHConfig
from grasph_free_surface import FreeSurfaceConfig
from grasph_solver import SolverSettings, SimulationHooks, run_simulation


def pulse(t, start=0.05, duration=0.10, peak=4.0):
    if t < start or t > start + duration:
        return np.zeros(3, dtype=np.float64)
    phase = (t - start) / duration
    return np.array([peak * np.sin(np.pi * phase), 0.0, 0.0], dtype=np.float64)


def main():
    p = argparse.ArgumentParser(description="Slosh regression using the generic GraSPH core")
    p.add_argument("usd", nargs="?", default=str(ROOT / "examples" / "SloshTest.usda"))
    p.add_argument("--dp", type=float, default=0.020)
    p.add_argument("--time", type=float, default=0.60)
    args = p.parse_args()

    cfg = SPHConfig(dp=args.dp)
    settings = SolverSettings(
        sim_time=args.time,
        output_fps=30.0,
        output_dir="output",
        output_stem=f"slosh_v53_dp{int(round(args.dp*1000)):02d}",
        free_surface=FreeSurfaceConfig(mode="anchored_band"),
    )
    hooks = SimulationHooks(body_acceleration=pulse, event_times=(0.05, 0.15))
    run_simulation(args.usd, sph_cfg=cfg, settings=settings, hooks=hooks)


if __name__ == "__main__":
    main()
