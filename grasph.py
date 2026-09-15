import argparse
from grasph_sph import SPHConfig
from grasph_free_surface import FreeSurfaceConfig
from grasph_solver import SolverSettings, run_simulation
from grasph_scene import load_scene


def main():
    p = argparse.ArgumentParser(description="GraSPH v0.54 FIX7: USD kinematics + cached translation-only wall search")
    p.add_argument("usd", help="Input .usd/.usda/.usdc scene")
    p.add_argument("--time", type=float, default=0.50, help="Simulation time [s]")
    p.add_argument("--dp", type=float, default=0.020, help="Particle spacing [m]")
    p.add_argument("--c0", type=float, default=20.0, help="Artificial sound speed [m/s]")
    p.add_argument("--fps", type=float, default=30.0, help="Output FPS")
    p.add_argument("--out", default="output", help="Output directory")
    p.add_argument("--name", default=None, help="Output stem")
    p.add_argument("--free-surface", choices=["anchored_band", "none"], default="anchored_band")
    p.add_argument("--no-collision", action="store_true")
    p.add_argument("--no-wall-pressure", action="store_true")
    p.add_argument("--no-wall-continuity", action="store_true")
    p.add_argument("--inspect-motion", action="store_true", help="Inspect USD collider animation and exit before SPH")
    p.add_argument("--neighbor-backend", choices=["auto", "ckdtree", "grid"], default="auto",
                   help="Neighbor search: auto uses SciPy cKDTree when available")
    p.add_argument("--threads", type=int, default=0,
                   help="cKDTree cross-query workers; 0 = all logical CPUs")
    p.add_argument("--droplet-mode", choices=["components", "zero"], default="components",
                   help="Detached-fluid fallback: disconnected components (default) or legacy low-neighbor mode")
    p.add_argument("--droplet-neighbors", type=int, default=0,
                   help="Legacy cutoff used only with --droplet-mode zero")
    p.add_argument("--wall-search", choices=["cached", "rebuild"], default="cached",
                   help="Reuse translation-only collider trees or rebuild the reference search each step")
    args = p.parse_args()

    if args.inspect_motion:
        load_scene(args.usd)
        return

    cfg = SPHConfig(dp=args.dp, c0=args.c0)
    fs = FreeSurfaceConfig(mode=args.free_surface)
    settings = SolverSettings(
        sim_time=args.time,
        output_fps=args.fps,
        output_dir=args.out,
        output_stem=args.name,
        enable_collision=not args.no_collision,
        enable_wall_pressure=not args.no_wall_pressure,
        enable_wall_continuity=not args.no_wall_continuity,
        free_surface=fs,
        neighbor_backend=args.neighbor_backend,
        neighbor_workers=(-1 if args.threads == 0 else max(1, args.threads)),
        detached_neighbor_max=max(0, args.droplet_neighbors),
        droplet_mode=args.droplet_mode,
        wall_search=args.wall_search,
    )
    run_simulation(args.usd, sph_cfg=cfg, settings=settings)


if __name__ == "__main__":
    main()
