from dataclasses import dataclass, field
from pathlib import Path
import time
import numpy as np

from grasph_scene import load_scene, generate_fluid_particles
from grasph_sph import (
    SPHConfig, WendlandC2, build_neighbor_pairs, compute_pair_geometry,
    neighbor_counts, compute_density_rate, compute_pressure,
    compute_pressure_acceleration, compute_artificial_viscosity_acceleration,
    compute_cfl_dt, configure_neighbor_search, active_neighbor_backend, neighbor_worker_count,
    detached_component_mask,
)
from grasph_collision import build_colliders, resolve_collisions
from grasph_boundary import (
    build_boundary_particles, build_layered_wall_particles,
    build_wall_interactions, CachedWallNeighborSearch, add_static_wall_density_rate,
    add_static_wall_pressure_acceleration,
)
from grasph_reference import hydrostatic_reference_state, compute_well_balanced_base_acceleration
from grasph_free_surface import FreeSurfaceConfig, collider_triangle_data, compute_atmospheric_state
from grasph_output import save_npz, write_usd_animation


@dataclass
class SolverSettings:
    sim_time: float = 0.50
    dt_max: float = 0.00030
    cfl: float = 0.25
    output_fps: float = 30.0
    report_interval: float = 1.0
    particle_radius_factor: float = 0.45
    viscosity_alpha: float = 0.05
    collision_iterations: int = 2
    enable_wall_pressure: bool = True
    enable_wall_continuity: bool = True
    enable_collision: bool = True
    output_dir: str = "output"
    output_stem: str | None = None
    free_surface: FreeSurfaceConfig = field(default_factory=FreeSurfaceConfig)
    neighbor_backend: str = "auto"
    neighbor_workers: int = -1
    detached_neighbor_max: int = 0
    droplet_mode: str = "components"
    wall_search: str = "cached"


@dataclass
class SimulationHooks:
    body_acceleration: object | None = None
    event_times: tuple = ()

    def acceleration(self, t):
        if self.body_acceleration is None:
            return np.zeros(3, dtype=np.float64)
        return np.asarray(self.body_acceleration(float(t)), dtype=np.float64)


@dataclass
class SimulationResult:
    scene_file: str
    npz_file: str
    usd_file: str
    positions: np.ndarray
    velocity: np.ndarray
    rho: np.ndarray
    time: float
    steps: int
    surface_level: float
    cfg: SPHConfig


def _fmt_time(seconds):
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _empty_wall_interactions():
    return {
        "pair_i": np.empty(0, dtype=np.int32),
        "pair_j": np.empty(0, dtype=np.int32),
        "rij": np.empty((0, 3), dtype=np.float64),
        "r": np.empty(0, dtype=np.float64),
        "W": np.empty(0, dtype=np.float64),
        "gradW": np.empty((0, 3), dtype=np.float64),
    }



def _translation_array_for_boundary(boundary, translations):
    out = np.zeros_like(boundary.positions, dtype=np.float64)
    for source_id, path in enumerate(boundary.source_paths):
        mask = boundary.source_ids == source_id
        if np.any(mask):
            out[mask] = np.asarray(translations.get(path, (0.0, 0.0, 0.0)), dtype=np.float64)
    return out


def _apply_boundary_translation(boundary, base_positions, base_facing, translations):
    shift = _translation_array_for_boundary(boundary, translations)
    boundary.positions[:] = base_positions + shift
    if boundary.facing_positions is not None and base_facing is not None:
        boundary.facing_positions[:] = base_facing + shift


def _boundary_velocity_array(boundary, velocities):
    out = np.zeros_like(boundary.positions, dtype=np.float64)
    for source_id, path in enumerate(boundary.source_paths):
        mask = boundary.source_ids == source_id
        if np.any(mask):
            out[mask] = np.asarray(velocities.get(path, (0.0, 0.0, 0.0)), dtype=np.float64)
    return out


def _translated_solid_triangles(base_triangles, triangle_paths, translations):
    if len(base_triangles) == 0:
        return base_triangles
    out = np.asarray(base_triangles, dtype=np.float64).copy()
    paths = np.asarray(triangle_paths, dtype=object)
    for path, shift in translations.items():
        mask = paths == path
        if np.any(mask):
            out[mask] += np.asarray(shift, dtype=np.float64)[None, None, :]
    return out


def run_simulation(usd_file, sph_cfg=None, settings=None, hooks=None):
    sph_cfg = sph_cfg or SPHConfig()
    settings = settings or SolverSettings()
    hooks = hooks or SimulationHooks()
    fs_cfg = settings.free_surface
    configure_neighbor_search(settings.neighbor_backend, settings.neighbor_workers)

    kernel = WendlandC2(sph_cfg.h)
    particle_radius = settings.particle_radius_factor * sph_cfg.dp
    stem = settings.output_stem or (Path(str(usd_file)).stem + "_grasph")
    out_dir = Path(settings.output_dir)
    npz_output = out_dir / f"{stem}.npz"
    usd_output = out_dir / f"{stem}.usdc"

    print("\n========================================")
    print("GraSPH v0.54 FIX7 - CACHED TRANSLATION-ONLY WALL SEARCH")
    print("========================================")
    print(f"Scene        : {usd_file}")
    print(f"dp/h/support : {sph_cfg.dp:.4f} / {sph_cfg.h:.4f} / {sph_cfg.support:.4f} m")
    print(f"rho0/c0     : {sph_cfg.rho0:.1f} / {sph_cfg.c0:.1f}")
    print(f"sim time     : {settings.sim_time:.4f} s")
    print(f"free surface : {fs_cfg.mode}")
    print(f"collision    : {settings.enable_collision}")
    print(f"wall P/cont  : {settings.enable_wall_pressure}/{settings.enable_wall_continuity}")
    worker_text = "all" if neighbor_worker_count() == -1 else str(neighbor_worker_count())
    print(f"neighbors    : {active_neighbor_backend()} (large-query workers={worker_text})")
    print(f"wall cross search: {settings.wall_search}, adaptive sparse<=2500 / parallel-radius>2500")
    if settings.droplet_mode == "components":
        print("droplet fallback: disconnected fluid components -> ordinary WCSPH + full gravity")
    else:
        print(f"droplet fallback: fluid neighbors <= {settings.detached_neighbor_max}")

    scene = load_scene(usd_file)
    pos = generate_fluid_particles(scene, dp=sph_cfg.dp)
    if len(pos) == 0:
        raise RuntimeError("No fluid particles generated. Use a FluidVolume mesh or grasph:role='fluid'.")
    n = len(pos)
    vel = np.zeros_like(pos, dtype=np.float64)

    gvec = np.asarray(sph_cfg.gravity, dtype=np.float64)
    gmag = float(np.linalg.norm(gvec))
    up = (-gvec / gmag) if gmag > 1e-15 else np.array([0.0, 0.0, 1.0])
    surface_level = float(np.max(pos @ up) + 0.5 * sph_cfg.dp)
    rho_ref, p_ref, _ = hydrostatic_reference_state(pos, surface_level, sph_cfg, return_depth=True)
    rho = rho_ref.copy()

    colliders = build_colliders(
        scene,
        voxel_size=0.5 * sph_cfg.dp,
        margin=2.0 * sph_cfg.dp,
    ) if scene.colliders else []

    surface_boundary = build_boundary_particles(scene, spacing=sph_cfg.dp)
    boundary = build_layered_wall_particles(surface_boundary, sph_cfg.dp, sph_cfg.support, sph_cfg.rho0)
    base_boundary_positions = boundary.positions.copy()
    wall_search = CachedWallNeighborSearch(boundary) if settings.wall_search == "cached" else None
    base_facing_positions = None if boundary.facing_positions is None else boundary.facing_positions.copy()
    wall_velocity = np.zeros_like(boundary.positions, dtype=np.float64)
    solid_triangles_base, solid_normals, solid_triangle_paths = collider_triangle_data(scene)

    # Reference pose is the USD stage start time. Any animated collider is
    # sampled in source-scene seconds from there. v0.54 supports pure
    # translation (perfect for a vertically lifted gate).
    motion0 = scene.collider_motion(0.0) if scene.colliders else {}
    _apply_boundary_translation(boundary, base_boundary_positions, base_facing_positions, motion0)
    solid_triangles = _translated_solid_triangles(solid_triangles_base, solid_triangle_paths, motion0)

    frame_positions, frame_velocity, frame_rho, frame_rho_ref = [], [], [], []
    frame_q, frame_times = [], []
    frame_shell, frame_band, frame_hG, frame_detached = [], [], [], []

    def snapshot(t, state=None, detached=None):
        rr, pp, _ = hydrostatic_reference_state(pos, surface_level, sph_cfg, return_depth=True)
        if detached is None:
            detached = np.zeros(n, dtype=bool)
        else:
            detached = np.asarray(detached, dtype=bool)
        if np.any(detached):
            rr = rr.copy(); pp = pp.copy()
            rr[detached] = sph_cfg.rho0
            pp[detached] = 0.0
        qq = compute_pressure(rho, sph_cfg.rho0, sph_cfg.tait_B, sph_cfg.gamma) - pp
        frame_positions.append(pos.copy())
        frame_velocity.append(vel.copy())
        frame_rho.append(rho.copy())
        frame_rho_ref.append(rr.copy())
        frame_q.append(qq.copy())
        frame_times.append(float(t))
        if state is None:
            frame_shell.append(np.zeros(n, dtype=np.uint8))
            frame_band.append(np.zeros(n, dtype=np.uint8))
            frame_hG.append(np.zeros(n, dtype=np.float32))
        else:
            frame_shell.append(state["shell"].astype(np.uint8).copy())
            frame_band.append(state["band"].astype(np.uint8).copy())
            frame_hG.append(state["support_hG"].astype(np.float32).copy())
        frame_detached.append(detached.astype(np.uint8).copy())

    pair_i0, pair_j0 = build_neighbor_pairs(pos, sph_cfg.support)
    counts0 = neighbor_counts(n, pair_i0, pair_j0)
    if settings.droplet_mode == "components":
        detached0, ncomp0, sizes0 = detached_component_mask(n, pair_i0, pair_j0)
    else:
        detached0 = counts0 <= settings.detached_neighbor_max
        ncomp0, sizes0 = 1, np.asarray([n], dtype=np.int32)
    _, _, gradW0 = compute_pair_geometry(pos, pair_i0, pair_j0, kernel)
    wall_int0 = build_wall_interactions(pos, boundary, kernel, sph_cfg.support, wall_search, motion0) if len(boundary.positions) else _empty_wall_interactions()
    fs_state = compute_atmospheric_state(
        pos, rho, rho_ref, pair_i0, pair_j0, gradW0,
        boundary, wall_int0, solid_triangles, solid_normals,
        surface_level, sph_cfg, fs_cfg,
    )
    snapshot(0.0, fs_state, detached0)

    output_interval = 1.0 / settings.output_fps
    next_output = output_interval
    t = 0.0
    step = 0
    total_contacts = 0
    max_collider_travel_seen = 0.0
    start = time.perf_counter()
    last_report = start
    perf = {
        "fluid_neighbors": 0.0,
        "droplet_components": 0.0,
        "wall_interactions": 0.0,
        "free_surface": 0.0,
        "sph_forces": 0.0,
        "collision": 0.0,
    }
    detached_ever = np.zeros(n, dtype=bool)
    current_ncomp = int(ncomp0)
    current_main_size = int(sizes0.max()) if len(sizes0) else 0

    print("\n========================================")
    print("STARTING GENERIC WCSPH")
    print("========================================\n")

    while t < settings.sim_time - 1e-15:
        dt = min(settings.dt_max, compute_cfl_dt(vel, sph_cfg.h, sph_cfg.c0, cfl=settings.cfl))
        if t + dt > settings.sim_time:
            dt = settings.sim_time - t
        for event_time in hooks.event_times:
            event_time = float(event_time)
            if t < event_time < t + dt:
                dt = event_time - t
                break

        # Sample Blender/OpenUSD collider transforms at the current and next
        # solver times. Distance fields remain in their reference pose and are
        # queried in translated local coordinates. SPH wall particles and the
        # finite-face free-surface geometry follow the same translation.
        if scene.colliders:
            motion_now = scene.collider_motion(t)
            motion_next = scene.collider_motion(t + dt)
            collider_velocities = {
                path: (motion_next[path] - motion_now[path]) / max(dt, 1.0e-30)
                for path in motion_now
            }
            if motion_now:
                max_collider_travel_seen = max(
                    max_collider_travel_seen,
                    max(float(np.linalg.norm(v)) for v in motion_now.values()),
                )
        else:
            motion_now = {}
            motion_next = {}
            collider_velocities = {}

        _apply_boundary_translation(boundary, base_boundary_positions, base_facing_positions, motion_now)
        wall_velocity = _boundary_velocity_array(boundary, collider_velocities)
        solid_triangles = _translated_solid_triangles(
            solid_triangles_base, solid_triangle_paths, motion_now
        )

        _tp = time.perf_counter()
        pair_i, pair_j = build_neighbor_pairs(pos, sph_cfg.support)
        rij, r, gradW = compute_pair_geometry(pos, pair_i, pair_j, kernel)
        counts = neighbor_counts(n, pair_i, pair_j)
        perf["fluid_neighbors"] += time.perf_counter() - _tp

        _tp = time.perf_counter()
        if settings.droplet_mode == "components":
            detached, current_ncomp, component_sizes = detached_component_mask(
                n, pair_i, pair_j
            )
            current_main_size = int(component_sizes.max()) if len(component_sizes) else 0
        else:
            detached = counts <= settings.detached_neighbor_max
            current_ncomp = 1 + int(np.any(detached))
            current_main_size = int(n - np.count_nonzero(detached))
        detached_ever |= detached
        perf["droplet_components"] += time.perf_counter() - _tp

        rho_ref, p_ref, depth_ref = hydrostatic_reference_state(pos, surface_level, sph_cfg, return_depth=True)
        # FIX6: the analytic hydrostatic reference belongs to the dominant
        # connected liquid body.  A disconnected spray/blob returns to ordinary
        # WCSPH: p_ref=0 and full gravity until it reconnects.
        if np.any(detached):
            rho_ref = rho_ref.copy(); p_ref = p_ref.copy(); depth_ref = depth_ref.copy()
            rho_ref[detached] = sph_cfg.rho0
            p_ref[detached] = 0.0
            depth_ref[detached] = 0.0

        pressure = compute_pressure(rho, sph_cfg.rho0, sph_cfg.tait_B, sph_cfg.gamma)
        q = pressure - p_ref

        need_wall = settings.enable_wall_pressure or settings.enable_wall_continuity or fs_cfg.mode != "none"
        _tp = time.perf_counter()
        wall_interactions = build_wall_interactions(pos, boundary, kernel, sph_cfg.support, wall_search, motion_now) if (need_wall and len(boundary.positions)) else _empty_wall_interactions()
        perf["wall_interactions"] += time.perf_counter() - _tp
        wall_rho_ref, _, _ = hydrostatic_reference_state(boundary.positions, surface_level, sph_cfg, return_depth=True)

        _tp = time.perf_counter()
        fs_state = compute_atmospheric_state(
            pos, rho, rho_ref, pair_i, pair_j, gradW,
            boundary, wall_interactions, solid_triangles, solid_normals,
            surface_level, sph_cfg, fs_cfg,
        )
        perf["free_surface"] += time.perf_counter() - _tp

        _tp = time.perf_counter()
        acc_q = compute_pressure_acceleration(rho, q, pair_i, pair_j, gradW, sph_cfg.mass)
        if settings.enable_wall_pressure and len(boundary.positions):
            q_wall = np.zeros(len(boundary.positions), dtype=np.float64)
            add_static_wall_pressure_acceleration(
                acc_q, rho, q, boundary, wall_interactions, wall_rho_ref, q_wall
            )

        acc_base = compute_well_balanced_base_acceleration(rho, rho_ref, depth_ref, sph_cfg.gravity)
        acc_visc = compute_artificial_viscosity_acceleration(
            pos, vel, rho, pair_i, pair_j, rij, r, gradW,
            sph_cfg.mass, sph_cfg.h, sph_cfg.c0,
            alpha=settings.viscosity_alpha,
        )
        acc = acc_q + acc_base + acc_visc + fs_state["acc"] + hooks.acceleration(t)[None, :]

        # Symplectic kick -> continuity -> drift. This is the frozen v0.16+
        # bulk update; v0.53 changes architecture, not timestep physics.
        vel += acc * dt
        drho = compute_density_rate(vel, pair_i, pair_j, gradW, sph_cfg.mass)
        if settings.enable_wall_continuity and len(boundary.positions):
            add_static_wall_density_rate(
                drho, vel, boundary, wall_interactions, wall_rho_ref,
                wall_velocity=wall_velocity,
            )
        rho += drho * dt
        pos += vel * dt
        perf["sph_forces"] += time.perf_counter() - _tp

        _tp = time.perf_counter()
        if settings.enable_collision and colliders:
            contacts = resolve_collisions(
                pos, vel, colliders,
                particle_radius=particle_radius,
                restitution=0.0,
                tangential_keep=1.0,
                iterations=settings.collision_iterations,
                collider_translations=motion_next,
                collider_velocities=collider_velocities,
            )
        else:
            contacts = 0
        total_contacts += contacts
        perf["collision"] += time.perf_counter() - _tp

        t += dt
        step += 1

        if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(vel)) and np.all(np.isfinite(rho))):
            raise RuntimeError("Non-finite solver state.")
        if rho.min() < 0.5 * sph_cfg.rho0 or rho.max() > 1.5 * sph_cfg.rho0:
            raise RuntimeError(f"Density instability: {rho.min():.1f} .. {rho.max():.1f} kg/m3")

        if t + 1e-12 >= next_output:
            snapshot(t, fs_state, detached)
            while next_output <= t + 1e-12:
                next_output += output_interval

        now = time.perf_counter()
        if now - last_report >= settings.report_interval or t >= settings.sim_time - 1e-15:
            elapsed = now - start
            progress = t / settings.sim_time if settings.sim_time > 0 else 1.0
            eta = elapsed * (1.0 - progress) / progress if progress > 0 else 0.0
            rr, pp, _ = hydrostatic_reference_state(pos, surface_level, sph_cfg, return_depth=True)
            qq = compute_pressure(rho, sph_cfg.rho0, sph_cfg.tait_B, sph_cfg.gamma) - pp
            speed = np.linalg.norm(vel, axis=1)
            print(
                f"[{_fmt_time(elapsed)}] step {step:5d} | {progress*100:5.1f}% "
                f"| sim {t:.4f}/{settings.sim_time:.4f}s | dt={dt:.7f} "
                f"| {step/elapsed:5.1f} step/s | ETA {_fmt_time(eta)}"
            )
            kin_shift = max((float(np.linalg.norm(v)) for v in motion_now.values()), default=0.0)
            kin_speed = max((float(np.linalg.norm(v)) for v in collider_velocities.values()), default=0.0)
            kin_text = f" | kinShift={kin_shift:.3f}m kinV={kin_speed:.3f}m/s" if (kin_shift > 1e-10 or kin_speed > 1e-10) else ""
            print(
                f"    neigh={counts.min()}/{counts.mean():.1f}/{counts.max()} "
                f"| rho={rho.min():.2f}/{rho.mean():.2f}/{rho.max():.2f} "
                f"| dRho={np.mean(np.abs(rho-rr)):.4f}/{np.max(np.abs(rho-rr)):.4f} "
                f"| qabs={np.mean(np.abs(qq)):.1f}/{np.max(np.abs(qq)):.1f}Pa "
                f"| vmax={speed.max():.3f} | shell={np.count_nonzero(fs_state['shell'])} "
                f"| band={np.count_nonzero(fs_state['band'])} | detached={np.count_nonzero(detached)} "
                f"| comps={current_ncomp} main={current_main_size} | contacts={contacts}" + kin_text
            )
            last_report = now

    if frame_times[-1] < t - 1e-12:
        snapshot(t, fs_state, detached)

    arrays = dict(
        positions=np.asarray(frame_positions, dtype=np.float32),
        velocity=np.asarray(frame_velocity, dtype=np.float32),
        rho=np.asarray(frame_rho, dtype=np.float32),
        rho_reference=np.asarray(frame_rho_ref, dtype=np.float32),
        reduced_pressure=np.asarray(frame_q, dtype=np.float32),
        atmospheric_mask=np.asarray(frame_shell, dtype=np.uint8),
        traction_band_mask=np.asarray(frame_band, dtype=np.uint8),
        current_surface_hG=np.asarray(frame_hG, dtype=np.float32),
        detached_mask=np.asarray(frame_detached, dtype=np.uint8),
        times=np.asarray(frame_times, dtype=np.float64),
        source_file=np.asarray(str(usd_file)),
        surface_level=np.asarray(surface_level),
        dp=np.asarray(sph_cfg.dp), h=np.asarray(sph_cfg.h), support=np.asarray(sph_cfg.support),
        mass=np.asarray(sph_cfg.mass), rho0=np.asarray(sph_cfg.rho0), c0=np.asarray(sph_cfg.c0),
        free_surface_mode=np.asarray(fs_cfg.mode),
        usd_start_time_code=np.asarray(scene.start_time_code),
        usd_time_codes_per_second=np.asarray(scene.time_codes_per_second),
    )
    print("\nWriting outputs...")
    save_npz(npz_output, **arrays)
    write_usd_animation(
        scene.stage, frame_positions, frame_times, settings.output_fps, sph_cfg.dp, usd_output,
        source_start_time_code=scene.start_time_code, source_tps=scene.time_codes_per_second,
    )

    elapsed = time.perf_counter() - start
    rr, pp, _ = hydrostatic_reference_state(pos, surface_level, sph_cfg, return_depth=True)
    qq = compute_pressure(rho, sph_cfg.rho0, sph_cfg.tait_B, sph_cfg.gamma) - pp
    print("\n========================================")
    print("SIMULATION COMPLETE")
    print("========================================")
    print(f"Simulated time : {t:.6f} s")
    print(f"Steps          : {step}")
    print(f"Solver wall    : {elapsed:.3f} s")
    print(f"Average rate   : {step/elapsed:.1f} step/s" if elapsed > 0 else "Average rate   : n/a")
    print(f"rho min/mean/max : {rho.min():.4f} / {rho.mean():.4f} / {rho.max():.4f}")
    print(f"rho-ref abs mean/max: {np.mean(np.abs(rho-rr)):.6f} / {np.max(np.abs(rho-rr)):.6f} kg/m3")
    print(f"q abs mean/max  : {np.mean(np.abs(qq)):.3f} / {np.max(np.abs(qq)):.3f} Pa")
    print(f"v max           : {np.linalg.norm(vel, axis=1).max():.6f} m/s")
    print(f"Contacts        : {total_contacts}")
    print(f"Kinematic max travel seen: {max_collider_travel_seen:.6f} m")
    print(f"Surface shell   : {np.count_nonzero(fs_state['shell'])} / {n}")
    print(f"Traction band   : {np.count_nonzero(fs_state['band'])} / {n}")
    print(f"Detached now/ever: {np.count_nonzero(detached)} / {np.count_nonzero(detached_ever)}")
    print(f"Fluid components final/main: {current_ncomp} / {current_main_size}")
    measured = sum(perf.values())
    if measured > 0.0:
        print("CPU phase timing (measured inner loop):")
        for key, value in sorted(perf.items(), key=lambda kv: -kv[1]):
            print(f"  {key:18s}: {value:9.2f} s  ({100.0*value/measured:5.1f}%)")
    print(f"NPZ             : {npz_output}")
    print(f"USD             : {usd_output}")

    return SimulationResult(
        scene_file=str(usd_file), npz_file=str(npz_output), usd_file=str(usd_output),
        positions=pos.copy(), velocity=vel.copy(), rho=rho.copy(), time=t, steps=step,
        surface_level=surface_level, cfg=sph_cfg,
    )
