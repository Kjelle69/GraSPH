import argparse
from grasph_sph import SPHConfig
from grasph_free_surface import FreeSurfaceConfig
from grasph_solver import SolverSettings, run_simulation
from grasph_scene import load_scene


def main():
    p = argparse.ArgumentParser(description="GraSPH: coupled WCSPH, granular DEM/PBD, and USD rigid bodies")
    p.add_argument("usd", help="Input .usd/.usda/.usdc scene")
    p.add_argument("--time", type=float, default=0.50, help="Simulation time [s]")
    p.add_argument("--dp", type=float, default=0.020, help="Particle spacing [m]")
    p.add_argument("--c0", type=float, default=20.0, help="Artificial sound speed [m/s]")
    p.add_argument("--fps", type=float, default=30.0, help="Output FPS")
    p.add_argument("--out", default="output", help="Output directory")
    p.add_argument("--name", default=None, help="Output stem")
    p.add_argument("--restart-npz", default="", help="Continue fluid state from a saved GraSPH NPZ result")
    p.add_argument("--restart-time", type=float, default=0.0,
                   help="Saved scene time [s] to resume from (nearest exact output frame)")
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
    p.add_argument("--air-density", type=float, default=1.225,
                   help="Air density [kg/m3] for detached-droplet drag")
    p.add_argument("--droplet-drag-cd", type=float, default=0.47,
                   help="Quadratic drag coefficient for detached droplets; 0 disables")
    p.add_argument("--wall-search", choices=["cached", "rebuild"], default="cached",
                   help="Cache exact fluid/wall pairs including rotating bodies, or rebuild both searches each step")
    p.add_argument("--granular-dt", type=float, default=0.001,
                   help="Maximum granular solver timestep [s]")
    p.add_argument("--granular-iterations", type=int, default=4,
                   help="Sphere contact projection iterations per step")
    p.add_argument("--solid-coupling", action="store_true",
                   help="Enable two-way SPH coupling for Dynamic Body colliders")
    p.add_argument("--preview-file", default="",
                   help="Atomically updated NPZ snapshot for Blender live preview")
    p.add_argument("--preview-interval", type=float, default=0.05,
                   help="Live preview frequency as a fraction of simulation duration")
    p.add_argument("--control-file", default="",
                   help="Pause the run while this control sentinel exists")
    p.add_argument("--usd-checkpoint", default="",
                   help="Atomically updated partial USD animation path")
    p.add_argument("--usd-checkpoint-interval", type=float, default=0.10,
                   help="Partial USD write interval as a fraction of total time")
    p.add_argument("--diagnostics", action="store_true", help="Write sampled force diagnostics alongside results")
    p.add_argument("--diagnostics-start", type=float, default=0.5)
    p.add_argument("--diagnostics-end", type=float, default=1.5)
    p.add_argument("--diagnostics-interval", type=float, default=0.01, help="Simulation seconds between samples")
    p.add_argument("--diagnostics-ids", default="", help="Comma-separated persistent fluid particle IDs to follow")
    args = p.parse_args()
    try:
        diagnostics_ids = tuple(int(i.strip()) for i in args.diagnostics_ids.split(",") if i.strip())
        if any(i < 0 for i in diagnostics_ids):
            raise ValueError
    except ValueError:
        p.error("--diagnostics-ids must contain nonnegative integers separated by commas")

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
        restart_npz=args.restart_npz,
        restart_time=max(0.0, args.restart_time),
        enable_collision=not args.no_collision,
        enable_wall_pressure=not args.no_wall_pressure,
        enable_wall_continuity=not args.no_wall_continuity,
        free_surface=fs,
        neighbor_backend=args.neighbor_backend,
        neighbor_workers=(-1 if args.threads == 0 else max(1, args.threads)),
        detached_neighbor_max=max(0, args.droplet_neighbors),
        droplet_mode=args.droplet_mode,
        detached_air_density=max(0.0, args.air_density),
        detached_air_drag_cd=max(0.0, args.droplet_drag_cd),
        wall_search=args.wall_search,
        granular_dt=max(1.0e-6, args.granular_dt),
        granular_iterations=max(1, args.granular_iterations),
        enable_solid_coupling=args.solid_coupling,
        preview_file=args.preview_file,
        preview_interval=max(0.01, min(1.0, args.preview_interval)),
        control_file=args.control_file,
        usd_checkpoint_file=args.usd_checkpoint,
        usd_checkpoint_interval=max(0.01, min(1.0, args.usd_checkpoint_interval)),
        enable_diagnostics=args.diagnostics,
        diagnostics_start=args.diagnostics_start,
        diagnostics_end=args.diagnostics_end,
        diagnostics_interval=args.diagnostics_interval,
        diagnostics_ids=diagnostics_ids,
    )
    run_simulation(args.usd, sph_cfg=cfg, settings=settings)


if __name__ == "__main__":
    main()
