from pathlib import Path
import argparse
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grasph_sph import SPHConfig
from grasph_free_surface import FreeSurfaceConfig
from grasph_solver import SolverSettings, run_simulation


def main():
    p = argparse.ArgumentParser(description="Ritter dry-bed benchmark using the generic GraSPH core")
    p.add_argument("usd", nargs="?", default=str(ROOT / "examples" / "DamBreakRitter.usda"))
    p.add_argument("--dp", type=float, default=0.020)
    p.add_argument("--time", type=float, default=0.20)
    args = p.parse_args()

    cfg = SPHConfig(dp=args.dp)
    settings = SolverSettings(
        sim_time=args.time,
        output_fps=50.0,
        output_dir="output",
        output_stem=f"ritter_v53_dp{int(round(args.dp*1000)):02d}",
        free_surface=FreeSurfaceConfig(mode="anchored_band"),
    )
    result = run_simulation(args.usd, sph_cfg=cfg, settings=settings)

    data = np.load(result.npz_file)
    p0 = data["positions"][0]
    pf = data["positions"][-1]
    t = float(data["times"][-1])
    front0 = float(np.max(p0[:, 0]) + 0.5 * cfg.dp)
    floor0 = float(np.min(p0[:, 2]) - 0.5 * cfg.dp)
    h0 = float(result.surface_level - floor0)
    speed = 2.0 * np.sqrt(abs(cfg.gravity[2]) * h0)
    ref = front0 + speed * t
    actual = float(np.max(pf[:, 0]) + 0.5 * cfg.dp)

    print("\n========================================")
    print("RITTER BENCHMARK REPORT")
    print("========================================")
    print(f"h0              : {h0:.6f} m")
    print(f"Ritter speed    : {speed:.6f} m/s")
    print(f"Front actual    : {actual:+.6f} m")
    print(f"Front reference : {ref:+.6f} m")
    print(f"Front error     : {actual-ref:+.6f} m")
    if abs(ref-front0) > 1e-12:
        print(f"Travel ratio    : {(actual-front0)/(ref-front0):.6f}")
    print("NOTE: anchored_band is retained only as the current experimental BC;")
    print("      v0.49-v0.52 showed it is not yet resolution independent.")


if __name__ == "__main__":
    main()
