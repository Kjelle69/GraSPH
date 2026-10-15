from dataclasses import dataclass, field
from pathlib import Path
import time
import numpy as np

from grasph_scene import load_scene, generate_fluid_particles, generate_granular_particles
from grasph_sph import (
    SPHConfig, WendlandC2, build_neighbor_pairs, compute_pair_geometry,
    neighbor_counts, compute_density_rate, compute_pressure,
    compute_pressure_acceleration, compute_artificial_viscosity_acceleration,
    compute_cfl_dt, configure_neighbor_search, active_neighbor_backend, neighbor_worker_count,
    detached_component_mask, CachedFluidNeighborSearch,
)
from grasph_collision import build_colliders, resolve_collisions
from grasph_boundary import (
    build_boundary_particles, build_layered_wall_particles,
    build_wall_interactions, CachedWallNeighborSearch, add_static_wall_density_rate,
    add_static_wall_pressure_acceleration,
)
from grasph_reference import hydrostatic_reference_state, compute_well_balanced_base_acceleration
from grasph_free_surface import FreeSurfaceConfig, collider_triangle_data, compute_atmospheric_state
from grasph_output import save_npz, write_usd_animation, write_usd_checkpoint
from grasph_coupling import couple_fluid_grains
from grasph_granular import (
    _append_particles as append_grains,
    _flow_candidates as grain_flow_candidates,
    resolve_grain_contacts,
)
from grasph_rigid import (
    advance_rotation, resolve_body_contacts, rotation_to_euler_xyz, world_inertia,
    constrain_body_state,
)
from grasph_runtime import wait_if_paused


@dataclass
class SolverSettings:
    sim_time: float = 0.50
    dt_max: float = 0.00030
    cfl: float = 0.25
    output_fps: float = 30.0
    report_interval: float = 1.0
    particle_radius_factor: float = 0.45
    viscosity_alpha: float = 0.05
    detached_air_density: float = 1.225
    detached_air_drag_cd: float = 0.47
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
    granular_dt: float = 0.001
    granular_iterations: int = 4
    enable_solid_coupling: bool = False
    preview_file: str = ""
    preview_interval: float = 0.05
    control_file: str = ""
    usd_checkpoint_file: str = ""
    usd_checkpoint_interval: float = 0.10
    restart_npz: str = ""
    restart_time: float = 0.0


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
    hinge_angles: dict = field(default_factory=dict)
    hinge_angular_velocities: dict = field(default_factory=dict)


def _fmt_time(seconds):
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _speed_stats(velocity, mask=None):
    """Return median, 95th percentile, and RMS speed for selected particles."""
    speed = np.linalg.norm(np.asarray(velocity, dtype=np.float64), axis=1)
    if mask is not None:
        speed = speed[np.asarray(mask, dtype=bool)]
    if not len(speed):
        return 0.0, 0.0, 0.0
    return (
        float(np.median(speed)),
        float(np.percentile(speed, 95)),
        float(np.sqrt(np.mean(speed * speed))),
    )


def _empty_wall_interactions():
    return {
        "pair_i": np.empty(0, dtype=np.int32),
        "pair_j": np.empty(0, dtype=np.int32),
        "rij": np.empty((0, 3), dtype=np.float64),
        "r": np.empty(0, dtype=np.float64),
        "W": np.empty(0, dtype=np.float64),
        "gradW": np.empty((0, 3), dtype=np.float64),
    }


def apply_detached_air_drag(velocity, detached, dt, particle_radius,
                            fluid_density=1000.0, air_density=1.225,
                            drag_coefficient=0.47):
    """Apply implicit quadratic aerodynamic drag to detached fluid parcels.

    Parcels are treated as equivalent spheres. The exact-in-speed update for
    dv/dt = -k |v| v avoids an explicit-drag stability restriction.
    """
    if drag_coefficient <= 0.0 or air_density <= 0.0 or dt <= 0.0:
        return
    ids = np.flatnonzero(detached)
    if not len(ids):
        return
    speed = np.linalg.norm(velocity[ids], axis=1)
    k = 3.0 * air_density * drag_coefficient / (
        8.0 * fluid_density * max(float(particle_radius), 1.0e-12)
    )
    velocity[ids] /= (1.0 + k * speed * dt)[:, None]


def _sink_keep_mask(pos, sinks):
    """Particles whose centers have not entered any sink volume."""
    keep = np.ones(len(pos), dtype=bool)
    for sink in sinks:
        keep &= ~np.all((pos >= sink.bounds_min) & (pos <= sink.bounds_max), axis=1)
    return keep


def _pad_frames(frames, capacity, trailing_shape=(), dtype=np.float32, fill=np.nan):
    out = np.full((len(frames), capacity) + trailing_shape, fill, dtype=dtype)
    for i, frame in enumerate(frames):
        out[i, :len(frame)] = frame
    return out


def _load_fluid_restart(path, frame_time, initial_positions, sph_cfg, fs_cfg):
    """Load a fluid-only state from a saved GraSPH NPZ frame."""
    with np.load(path, allow_pickle=False) as data:
        required = {
            "positions", "velocity", "rho", "active_particle_ids",
            "active_counts", "times", "dp", "rho0", "c0",
        }
        missing = required.difference(data.files)
        if missing:
            raise RuntimeError(f"Restart file is missing fields: {', '.join(sorted(missing))}")
        for key, expected in (("dp", sph_cfg.dp), ("rho0", sph_cfg.rho0), ("c0", sph_cfg.c0)):
            actual = float(np.asarray(data[key]).item())
            if not np.isclose(actual, expected, rtol=1e-6, atol=1e-9):
                raise RuntimeError(
                    f"Restart {key}={actual:g} does not match current scene setting {expected:g}."
                )
        if "h" in data.files and not np.isclose(
            float(np.asarray(data["h"]).item()), sph_cfg.h, rtol=1e-6, atol=1e-9
        ):
            raise RuntimeError("Restart smoothing length does not match current solver settings.")
        if "gravity" in data.files and not np.allclose(
            np.asarray(data["gravity"], dtype=np.float64), sph_cfg.gravity,
            rtol=1e-6, atol=1e-8,
        ):
            raise RuntimeError("Restart gravity does not match the current solver settings.")
        if "free_surface_mode" in data.files:
            old_mode = str(np.asarray(data["free_surface_mode"]).item())
            if old_mode != fs_cfg.mode:
                raise RuntimeError(
                    f"Restart free-surface mode '{old_mode}' does not match '{fs_cfg.mode}'."
                )
        for key in ("body_paths", "hinge_paths"):
            if key in data.files and len(data[key]):
                raise RuntimeError(
                    "This first restart version supports fluid-only results. "
                    f"The selected result contains simulated {key.replace('_', ' ')}."
                )
        if "granular_active_counts" in data.files and np.any(data["granular_active_counts"] > 0):
            raise RuntimeError("This first restart version supports fluid-only results, not granular state.")

        times = np.asarray(data["times"], dtype=np.float64)
        if not len(times):
            raise RuntimeError("Restart file contains no saved frames.")
        index = int(np.argmin(np.abs(times - float(frame_time))))
        selected_time = float(times[index])
        frame_spacing = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
        time_tolerance = max(1e-5, 0.51 * frame_spacing)
        if abs(selected_time - float(frame_time)) > time_tolerance:
            raise RuntimeError(
                f"No saved frame near {float(frame_time):.6g}s; nearest is {selected_time:.6g}s."
            )
        capacity = int(np.asarray(data["positions"]).shape[1])
        count = int(np.asarray(data["active_counts"])[index])
        if count <= 0 or count > capacity:
            raise RuntimeError("Selected restart frame has no valid active fluid particles.")
        ids = np.asarray(data["active_particle_ids"][index, :count], dtype=np.int32)
        pos = np.asarray(data["positions"][index, :count], dtype=np.float64)
        vel = np.asarray(data["velocity"][index, :count], dtype=np.float64)
        rho = np.asarray(data["rho"][index, :count], dtype=np.float64)
        valid = (ids >= 0) & np.all(np.isfinite(pos), axis=1) & np.all(np.isfinite(vel), axis=1) & np.isfinite(rho)
        ids, pos, vel, rho = ids[valid], pos[valid], vel[valid], rho[valid]
        if not len(pos):
            raise RuntimeError("Selected restart frame contains no finite fluid state.")

        baseline_key = "source_initial_positions"
        if baseline_key in data.files:
            saved_initial = np.asarray(data[baseline_key], dtype=np.float64)
        else:
            first_count = int(np.asarray(data["active_counts"])[0])
            saved_initial = np.asarray(data["positions"][0, :first_count], dtype=np.float64)
        if saved_initial.shape != initial_positions.shape or not np.allclose(
            saved_initial, initial_positions, rtol=1e-5, atol=1e-5
        ):
            raise RuntimeError(
                "Current fluid source does not match the selected result's original particle layout. "
                "Keep the source volume and particle spacing unchanged when continuing."
            )
        surface_level = float(np.asarray(data["surface_level"]).item()) if "surface_level" in data.files else None
    return pos, vel, rho, ids, capacity, surface_level, selected_time



def _translation_array_for_boundary(boundary, translations):
    out = np.zeros_like(boundary.positions, dtype=np.float64)
    for source_id, path in enumerate(boundary.source_paths):
        mask = boundary.source_ids == source_id
        if np.any(mask):
            out[mask] = np.asarray(translations.get(path, (0.0, 0.0, 0.0)), dtype=np.float64)
    return out


def _axis_angle_matrix(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    x, y, z = axis / np.linalg.norm(axis)
    c, s, q = np.cos(angle), np.sin(angle), 1.0 - np.cos(angle)
    return np.array([
        [c + x*x*q, x*y*q - z*s, x*z*q + y*s],
        [y*x*q + z*s, c + y*y*q, y*z*q - x*s],
        [z*x*q - y*s, z*y*q + x*s, c + z*z*q],
    ], dtype=np.float64)


def _apply_boundary_motion(
    boundary, base_positions, base_normals, base_facing, translations,
    rotations=None, centers=None,
):
    shift = _translation_array_for_boundary(boundary, translations)
    boundary.positions[:] = base_positions + shift
    boundary.normals[:] = base_normals
    if boundary.facing_positions is not None and base_facing is not None:
        boundary.facing_positions[:] = base_facing + shift
    rotations = rotations or {}
    centers = centers or {}
    for source_id, path in enumerate(boundary.source_paths):
        if path not in rotations:
            continue
        mask = boundary.source_ids == source_id
        rotation = rotations[path]
        center = np.asarray(centers[path], dtype=np.float64)
        path_shift = np.asarray(translations.get(path, (0.0, 0.0, 0.0)), dtype=np.float64)
        boundary.positions[mask] = (
            center + (base_positions[mask] - center) @ rotation.T + path_shift
        )
        boundary.normals[mask] = base_normals[mask] @ rotation.T
        if boundary.facing_positions is not None and base_facing is not None:
            boundary.facing_positions[mask] = (
                center + (base_facing[mask] - center) @ rotation.T + path_shift
            )


def _boundary_velocity_array(boundary, velocities, angular_velocities=None, centers=None, translations=None):
    out = np.zeros_like(boundary.positions, dtype=np.float64)
    for source_id, path in enumerate(boundary.source_paths):
        mask = boundary.source_ids == source_id
        if np.any(mask):
            out[mask] = np.asarray(velocities.get(path, (0.0, 0.0, 0.0)), dtype=np.float64)
            if angular_velocities and path in angular_velocities:
                center = np.asarray(centers[path], dtype=np.float64)
                shift = np.asarray((translations or {}).get(path, (0.0, 0.0, 0.0)))
                lever = boundary.positions[mask] - (center + shift)[None, :]
                out[mask] += np.cross(angular_velocities[path][None, :], lever)
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


def _rigid_solid_geometry(base_triangles, base_normals, triangle_paths, translations,
                          rotations=None, centers=None):
    triangles = _translated_solid_triangles(base_triangles, triangle_paths, translations)
    normals = np.asarray(base_normals, dtype=np.float64).copy()
    paths = np.asarray(triangle_paths, dtype=object)
    for path, rotation in (rotations or {}).items():
        mask = paths == path
        if not np.any(mask):
            continue
        center = np.asarray(centers[path], dtype=np.float64)
        shift = np.asarray(translations.get(path, (0.0, 0.0, 0.0)), dtype=np.float64)
        triangles[mask] = center + (base_triangles[mask] - center) @ rotation.T + shift
        normals[mask] = base_normals[mask] @ rotation.T
    return triangles, normals


def run_simulation(usd_file, sph_cfg=None, settings=None, hooks=None):
    sph_cfg = sph_cfg or SPHConfig()
    settings = settings or SolverSettings()
    hooks = hooks or SimulationHooks()
    fs_cfg = settings.free_surface
    configure_neighbor_search(settings.neighbor_backend, settings.neighbor_workers)

    scene = load_scene(usd_file)
    if settings.restart_npz and scene.granular_sources:
        raise RuntimeError(
            "This first restart version is fluid-only. Remove granular sources from "
            "the continuation scene; granular state is not restored yet."
        )
    if not scene.fluid_sources and (scene.granular_sources or scene.dynamic_bodies or scene.hinges):
        from grasph_granular import run_granular_simulation
        return run_granular_simulation(
            usd_file, scene, settings, gravity=sph_cfg.gravity
        )

    kernel = WendlandC2(sph_cfg.h)
    particle_radius = settings.particle_radius_factor * sph_cfg.dp
    stem = settings.output_stem or (Path(str(usd_file)).stem + "_grasph")
    out_dir = Path(settings.output_dir)
    npz_output = out_dir / f"{stem}.npz"
    usd_output = out_dir / f"{stem}.usdc"

    print("\n========================================")
    print("GraSPH v0.7 - COUPLED SPH + GRANULAR DEM/PBD")
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
    print(f"fluid pair cache: {'ON' if settings.wall_search == 'cached' else 'OFF'} (exact support filtering)")
    print(
        f"detached air drag: rho_air={settings.detached_air_density:.3g} kg/m3, "
        f"Cd={settings.detached_air_drag_cd:.3g} "
        "(equivalent-sphere approximation; Cd=0 disables)"
    )
    if settings.droplet_mode == "components":
        print("droplet fallback: disconnected fluid components -> ordinary WCSPH + full gravity")
    else:
        print(f"droplet fallback: fluid neighbors <= {settings.detached_neighbor_max}")

    initial_positions = generate_fluid_particles(scene, dp=sph_cfg.dp, particle_radius=particle_radius)
    if len(initial_positions) == 0:
        raise RuntimeError(
            "No fluid particles generated. Use a Fluid Volume, Granular Source, "
            "or moving rigid body."
        )
    if scene.granular_sources and len(initial_positions) < 8:
        raise RuntimeError(
            f"Mixed fluid/granular simulation generated only {len(initial_positions)} fluid "
            f"particle(s) at dp={sph_cfg.dp:g} m. Use a larger Fluid Volume or "
            "reduce Particle Spacing; at least 8 are required and 64+ are recommended."
        )
    restart_time = 0.0
    if settings.restart_npz:
        (pos, vel, rho, particle_ids, capacity, saved_surface_level,
         restart_time) = _load_fluid_restart(
            settings.restart_npz, settings.restart_time, initial_positions,
            sph_cfg, fs_cfg,
        )
        if saved_surface_level is None:
            raise RuntimeError("Restart file is missing its fluid surface level.")
        surface_level = saved_surface_level
        removed_count = capacity - len(pos)
        print(
            f"Restarting fluid from {settings.restart_npz} at "
            f"t={restart_time:.6f}s ({len(pos)} active particles)."
        )
    else:
        pos = initial_positions.copy()
        vel = np.zeros_like(pos, dtype=np.float64)
        particle_ids = np.arange(len(pos), dtype=np.int32)
        capacity = len(pos)
        removed_count = 0
    n = len(pos)

    gvec = np.asarray(sph_cfg.gravity, dtype=np.float64)
    gmag = float(np.linalg.norm(gvec))
    up = (-gvec / gmag) if gmag > 1e-15 else np.array([0.0, 0.0, 1.0])
    if not settings.restart_npz:
        surface_level = float(np.max(pos @ up) + 0.5 * sph_cfg.dp)
    rho_ref, p_ref, _ = hydrostatic_reference_state(
        pos, surface_level, sph_cfg, return_depth=True
    )
    if not settings.restart_npz:
        rho = rho_ref.copy()

    # Mixed mode: spherical DEM grains share the WCSPH timestep. This keeps
    # the fast neighbour pipeline and avoids a second global solver loop.
    grain_pos, grain_source_ids = generate_granular_particles(scene)
    grain_sources = scene.granular_sources
    grain_radii = np.asarray([
        grain_sources[index].particle_radius for index in grain_source_ids
    ], dtype=np.float64)
    grain_density = np.asarray([
        grain_sources[index].density for index in grain_source_ids
    ], dtype=np.float64)
    grain_friction = np.asarray([
        grain_sources[index].friction for index in grain_source_ids
    ], dtype=np.float64)
    grain_restitution = np.asarray([
        grain_sources[index].restitution for index in grain_source_ids
    ], dtype=np.float64)
    grain_rolling = np.asarray([
        grain_sources[index].rolling_friction for index in grain_source_ids
    ], dtype=np.float64)
    grain_masses = grain_density * (4.0 / 3.0) * np.pi * grain_radii**3
    grain_inertia = 0.4 * grain_masses * grain_radii**2
    grain_vel = np.zeros_like(grain_pos)
    grain_omega = np.zeros_like(grain_pos)
    grain_ids = np.arange(len(grain_pos), dtype=np.int32)
    next_grain_id = len(grain_pos)
    grain_flow_states = []
    for source in grain_sources:
        if source.source_mode != "flow":
            continue
        candidates = grain_flow_candidates(source)
        if not len(candidates):
            raise RuntimeError(
                f"Granular Flow Source {source.mesh.path} is too small for its radius"
            )
        grain_flow_states.append({
            "source": source, "candidates": candidates, "cursor": 0,
            "remainder": 0.0, "emitted": 0, "blocked": 0,
        })

    colliders = build_colliders(
        scene,
        voxel_size=0.5 * sph_cfg.dp,
        margin=2.0 * sph_cfg.dp,
    ) if scene.colliders else []
    surface_boundary = build_boundary_particles(scene, spacing=sph_cfg.dp)
    boundary = build_layered_wall_particles(surface_boundary, sph_cfg.dp, sph_cfg.support, sph_cfg.rho0)
    base_boundary_positions = boundary.positions.copy()
    base_boundary_normals = boundary.normals.copy()
    wall_search = CachedWallNeighborSearch(boundary) if settings.wall_search == "cached" else None
    fluid_search = CachedFluidNeighborSearch(0.25 * sph_cfg.dp) if settings.wall_search == "cached" else None
    base_facing_positions = None if boundary.facing_positions is None else boundary.facing_positions.copy()
    wall_velocity = np.zeros_like(boundary.positions, dtype=np.float64)
    solid_triangles_base, solid_normals_base, solid_triangle_paths = collider_triangle_data(scene)

    hinge_states = {
        path: {"angle": 0.0, "omega": 0.0, "torque": 0.0}
        for path in scene.hinges
    }
    hinge_centers = {path: hinge.center for path, hinge in scene.hinges.items()}
    body_states = {
        path: {
            "translation": np.zeros(3), "velocity": np.zeros(3),
            "rotation": np.eye(3), "omega": np.zeros(3),
        }
        for path in scene.dynamic_bodies
    } if settings.enable_solid_coupling else {}
    body_paths = list(body_states)
    collider_centers = dict(hinge_centers)
    collider_centers.update({
        path: body.center for path, body in scene.dynamic_bodies.items()
    })

    # Reference pose is the USD stage start time. Any animated collider is
    # sampled in source-scene seconds from there. v0.54 supports pure
    # translation (perfect for a vertically lifted gate).
    motion0 = scene.collider_motion(restart_time) if scene.colliders else {}
    _apply_boundary_motion(
        boundary, base_boundary_positions, base_boundary_normals,
        base_facing_positions, motion0,
    )
    solid_triangles, solid_normals = _rigid_solid_geometry(
        solid_triangles_base, solid_normals_base, solid_triangle_paths, motion0
    )

    frame_positions, frame_velocity, frame_rho, frame_rho_ref = [], [], [], []
    frame_q, frame_times = [], []
    frame_shell, frame_band, frame_hG, frame_detached = [], [], [], []
    frame_ids = []
    frame_hinge_angles, frame_hinge_omegas = [], []
    hinge_paths = list(scene.hinges)
    frame_body_translations = {path: [] for path in body_paths}
    frame_body_rotations = {path: [] for path in body_paths}
    frame_grain_positions, frame_grain_velocities = [], []
    frame_grain_radii, frame_grain_ids = [], []
    preview_interval = max(0.01, float(settings.preview_interval))
    next_preview = restart_time + preview_interval

    def write_preview(t):
        """Atomically publish the latest solver state for Blender's live viewport."""
        nonlocal next_preview
        if not settings.preview_file:
            return
        from pathlib import Path
        import os
        preview_path = Path(settings.preview_file)
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = preview_path.with_name(preview_path.name + ".tmp")
        with temporary.open("wb") as stream:
            np.savez(
                stream, time=np.asarray(float(t)), progress=np.asarray(min(
                    1.0, max(0.0, (float(t) - restart_time) / max(settings.sim_time, 1e-12))
                )), positions=np.asarray(pos, dtype=np.float32),
                velocity=np.asarray(vel, dtype=np.float32),
                granular_positions=np.asarray(grain_pos, dtype=np.float32),
                granular_velocity=np.asarray(grain_vel, dtype=np.float32),
                granular_radii=np.asarray(grain_radii, dtype=np.float32),
                body_paths=np.asarray(body_paths, dtype="U512"),
                body_translations=np.asarray([
                    body_states[path]["translation"] for path in body_paths
                ], dtype=np.float64),
                body_rotations=np.asarray([
                    rotation_to_euler_xyz(body_states[path]["rotation"])
                    for path in body_paths
                ], dtype=np.float64).reshape((-1, 3)),
                hinge_paths=np.asarray(hinge_paths, dtype="U512"),
                hinge_angles=np.asarray([
                    hinge_states[path]["angle"] for path in hinge_paths
                ], dtype=np.float64),
            )
        os.replace(temporary, preview_path)
        elapsed = float(t) - restart_time
        next_preview = restart_time + (
            np.floor(elapsed / preview_interval + 1e-10) + 1.0
        ) * preview_interval

    checkpoint_interval = max(0.01, min(
        1.0, float(getattr(settings, "usd_checkpoint_interval", 0.10))
    ))
    next_checkpoint = checkpoint_interval

    def write_animation(output_file, checkpoint=False):
        writer = write_usd_checkpoint if checkpoint else write_usd_animation
        writer(
            scene.stage, frame_positions, frame_times, settings.output_fps,
            sph_cfg.dp, output_file,
            source_start_time_code=scene.start_time_code,
            source_tps=scene.time_codes_per_second,
            frame_velocities=frame_velocity,
            hinge_frames={
                path: {
                    "driver_path": scene.hinges[path].driver_path,
                    "axis": scene.hinges[path].axis,
                    "angles": np.asarray(frame_hinge_angles)[:, i],
                }
                for i, path in enumerate(hinge_paths)
            },
            body_frames={
                path: {
                    "driver_path": scene.dynamic_bodies[path].driver_path,
                    "translations": np.asarray(frame_body_translations[path]),
                    "rotations": np.asarray(frame_body_rotations[path]),
                }
                for path in body_paths
            },
            extra_particle_sets=([{
                "prim_path": "/GraSPH_GranularResult",
                "result_kind": "granular",
                "frames": frame_grain_positions,
                "velocities": frame_grain_velocities,
                "widths": [2.0 * values for values in frame_grain_radii],
            }] if grain_sources else None),
        )

    def write_partial_checkpoint(t):
        nonlocal next_checkpoint
        checkpoint_file = getattr(settings, "usd_checkpoint_file", "")
        if not checkpoint_file or t >= restart_time + settings.sim_time - 1e-12:
            return
        progress = (float(t) - restart_time) / max(float(settings.sim_time), 1e-12)
        if progress + 1e-10 < next_checkpoint:
            return
        write_animation(checkpoint_file, checkpoint=True)
        next_checkpoint = (
            np.floor(progress / checkpoint_interval + 1e-10) + 1.0
        ) * checkpoint_interval

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
        frame_ids.append(particle_ids.copy())
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
        frame_hinge_angles.append(
            np.asarray([hinge_states[path]["angle"] for path in hinge_paths], dtype=np.float64)
        )
        frame_hinge_omegas.append(
            np.asarray([hinge_states[path]["omega"] for path in hinge_paths], dtype=np.float64)
        )
        for path in body_paths:
            frame_body_translations[path].append(body_states[path]["translation"].copy())
            frame_body_rotations[path].append(
                rotation_to_euler_xyz(body_states[path]["rotation"])
            )
        frame_grain_positions.append(grain_pos.copy())
        frame_grain_velocities.append(grain_vel.copy())
        frame_grain_radii.append(grain_radii.copy())
        frame_grain_ids.append(grain_ids.copy())
        if float(t) + 1e-12 >= next_preview or abs(float(t) - restart_time) < 1e-12:
            write_preview(t)
        write_partial_checkpoint(float(t))

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
        surface_level, sph_cfg, fs_cfg, active_mask=~detached0,
    )
    t = restart_time
    snapshot(t, fs_state, detached0)

    output_interval = 1.0 / settings.output_fps
    next_output = restart_time + output_interval
    end_time = restart_time + settings.sim_time
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
    detached_ever = np.zeros(capacity, dtype=bool)
    current_ncomp = int(ncomp0)
    current_main_size = int(sizes0.max()) if len(sizes0) else 0

    print("\n========================================")
    print("STARTING GENERIC WCSPH")
    print(f"Fluid/solid coupling: {'ON' if body_states else 'OFF'}")
    if grain_sources:
        print(f"Fluid/granular coupling: ON ({len(grain_pos)} initial grains)")
    print("========================================\n")

    while t < end_time - 1e-15:
        wait_if_paused(settings.control_file)
        dt = min(settings.dt_max, compute_cfl_dt(vel, sph_cfg.h, sph_cfg.c0, cfl=settings.cfl))
        if t + dt > end_time:
            dt = end_time - t
        for event_time in hooks.event_times:
            event_time = float(event_time)
            if t < event_time < t + dt:
                dt = event_time - t
                break

        for path, state in body_states.items():
            body = scene.dynamic_bodies[path]
            if getattr(body, "joint_type", "free") == "fixed":
                constrain_body_state(body, state)
                continue
            state["velocity"] += np.asarray(sph_cfg.gravity) * dt
            state["velocity"] /= 1.0 + body.linear_damping * dt
            state["omega"] /= 1.0 + body.angular_damping * dt

        for flow in grain_flow_states:
            source = flow["source"]
            active_dt = max(
                0.0, min(t + dt, source.end_time) - max(t, source.start_time)
            )
            flow["remainder"] += source.flow_rate * active_dt
            requested = int(np.floor(flow["remainder"] + 1.0e-12))
            flow["remainder"] -= requested
            accepted = []
            for _ in range(requested):
                candidate = flow["candidates"][flow["cursor"]].copy()
                flow["cursor"] = (flow["cursor"] + 1) % len(flow["candidates"])
                if len(grain_pos) and np.any(
                    np.linalg.norm(grain_pos - candidate[None, :], axis=1)
                    < grain_radii + source.particle_radius * 0.98
                ):
                    flow["blocked"] += 1
                    continue
                if accepted and np.any(
                    np.linalg.norm(np.asarray(accepted) - candidate[None, :], axis=1)
                    < 1.96 * source.particle_radius
                ):
                    flow["blocked"] += 1
                    continue
                accepted.append(candidate)
            if accepted:
                (grain_pos, grain_vel, grain_omega, grain_radii, grain_masses,
                 grain_inertia, grain_friction, grain_restitution, grain_rolling,
                 grain_ids, next_grain_id) = append_grains(
                    grain_pos, grain_vel, grain_omega, grain_radii, grain_masses,
                    grain_inertia, grain_friction, grain_restitution, grain_rolling,
                    grain_ids, np.asarray(accepted), source, next_grain_id,
                )
                flow["emitted"] += len(accepted)

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

        for path, state in body_states.items():
            motion_now[path] = state["translation"].copy()
            motion_next[path] = state["translation"].copy()
            collider_velocities[path] = state["velocity"].copy()

        collider_rotations = {
            path: _axis_angle_matrix(scene.hinges[path].axis, state["angle"])
            for path, state in hinge_states.items()
        }
        collider_angular_velocities = {
            path: scene.hinges[path].axis * state["omega"]
            for path, state in hinge_states.items()
        }
        collider_rotations.update({
            path: state["rotation"] for path, state in body_states.items()
        })
        collider_angular_velocities.update({
            path: state["omega"] for path, state in body_states.items()
        })
        _apply_boundary_motion(
            boundary, base_boundary_positions, base_boundary_normals,
            base_facing_positions, motion_now, collider_rotations, collider_centers,
        )
        wall_velocity = _boundary_velocity_array(
            boundary, collider_velocities, collider_angular_velocities,
            collider_centers, motion_now,
        )
        solid_triangles, solid_normals = _rigid_solid_geometry(
            solid_triangles_base, solid_normals_base, solid_triangle_paths,
            motion_now, collider_rotations, collider_centers,
        )

        _tp = time.perf_counter()
        pair_i, pair_j = (fluid_search.query(pos, sph_cfg.support) if fluid_search
                          else build_neighbor_pairs(pos, sph_cfg.support))
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
        detached_ever[particle_ids] |= detached
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
        active_wall_search = wall_search
        wall_interactions = build_wall_interactions(
            pos, boundary, kernel, sph_cfg.support, active_wall_search, motion_now
        ) if (need_wall and len(boundary.positions)) else _empty_wall_interactions()
        perf["wall_interactions"] += time.perf_counter() - _tp
        wall_rho_ref, wall_p_ref, _ = hydrostatic_reference_state(boundary.positions, surface_level, sph_cfg, return_depth=True)

        _tp = time.perf_counter()
        fs_state = compute_atmospheric_state(
            pos, rho, rho_ref, pair_i, pair_j, gradW,
            boundary, wall_interactions, solid_triangles, solid_normals,
            surface_level, sph_cfg, fs_cfg, active_mask=~detached,
        )
        perf["free_surface"] += time.perf_counter() - _tp

        _tp = time.perf_counter()
        acc_q = compute_pressure_acceleration(rho, q, pair_i, pair_j, gradW, sph_cfg.mass)
        wall_pair_j = np.empty(0, dtype=np.int32)
        wall_pair_acceleration = np.empty((0, 3), dtype=np.float64)
        if settings.enable_wall_pressure and len(boundary.positions):
            q_wall = np.zeros(len(boundary.positions), dtype=np.float64)
            wall_pair_j, wall_pair_acceleration = add_static_wall_pressure_acceleration(
                acc_q, rho, q, boundary, wall_interactions, wall_rho_ref, q_wall
            )

        pressure_torques = {path: 0.0 for path in hinge_states}
        pressure_body_forces = {path: np.zeros(3) for path in body_states}
        pressure_body_torques = {path: np.zeros(3) for path in body_states}
        hydrostatic_body_forces = {path: np.zeros(3) for path in body_states}
        if len(wall_pair_j) and (hinge_states or body_states):
            reaction_forces = -sph_cfg.mass * wall_pair_acceleration
            # Fluid uses a well-balanced equation with hydrostatic pressure
            # removed analytically. A solid still feels that reference load.
            # Restore it on moving-solid pairs only, using the SAME discrete
            # wall operator as the perturbation pressure (no buoyancy scale).
            moving_sources = [i for i, path in enumerate(boundary.source_paths)
                              if path in hinge_states or path in body_states]
            moving_pairs = np.isin(boundary.source_ids[wall_pair_j], moving_sources)
            reference_interactions = {
                key: wall_interactions[key][moving_pairs]
                for key in ("pair_i", "pair_j", "gradW")
            }
            _, reference_acceleration = add_static_wall_pressure_acceleration(
                np.zeros_like(pos), rho, p_ref, boundary,
                reference_interactions, wall_rho_ref, wall_p_ref,
            )
            reaction_forces[moving_pairs] -= sph_cfg.mass * reference_acceleration
            reference_source_ids = boundary.source_ids[reference_interactions["pair_j"]]
            for source_id, path in enumerate(boundary.source_paths):
                if path in body_states:
                    hydrostatic_body_forces[path] = -sph_cfg.mass * np.sum(
                        reference_acceleration[reference_source_ids == source_id], axis=0)
            for source_id, path in enumerate(boundary.source_paths):
                if path not in hinge_states and path not in body_states:
                    continue
                pair_mask = boundary.source_ids[wall_pair_j] == source_id
                if not np.any(pair_mask):
                    continue
                boundary_ids = wall_pair_j[pair_mask]
                reference_center = (
                    scene.hinges[path].center if path in hinge_states
                    else scene.dynamic_bodies[path].center
                )
                center = reference_center + np.asarray(
                    motion_now.get(path, (0.0, 0.0, 0.0)), dtype=np.float64
                )
                lever = boundary.positions[boundary_ids] - center[None, :]
                torque_vectors = np.cross(lever, reaction_forces[pair_mask])
                if path in hinge_states:
                    pressure_torques[path] = float(
                        np.sum(torque_vectors @ scene.hinges[path].axis)
                    )
                else:
                    pressure_body_forces[path] += np.sum(reaction_forces[pair_mask], axis=0)
                    pressure_body_torques[path] += np.sum(torque_vectors, axis=0)

        acc_base = compute_well_balanced_base_acceleration(rho, rho_ref, depth_ref, sph_cfg.gravity)
        acc_visc = compute_artificial_viscosity_acceleration(
            pos, vel, rho, pair_i, pair_j, rij, r, gradW,
            sph_cfg.mass, sph_cfg.h, sph_cfg.c0,
            alpha=settings.viscosity_alpha,
        )
        fluid_grain_acc, grain_hydro_acc, fluid_grain_pairs = couple_fluid_grains(
            pos, vel, rho, sph_cfg.mass,
            grain_pos, grain_vel, grain_radii, grain_masses,
            sph_cfg.gravity, sph_cfg.support, dt=dt,
        )
        acc = (
            acc_q + acc_base + acc_visc + fs_state["acc"]
            + hooks.acceleration(t)[None, :] + fluid_grain_acc
        )

        # Symplectic kick -> continuity -> drift. This is the frozen v0.16+
        # bulk update; v0.53 changes architecture, not timestep physics.
        vel += acc * dt
        apply_detached_air_drag(
            vel, detached, dt,
            particle_radius=(3.0 * sph_cfg.mass / (4.0 * np.pi * sph_cfg.rho0)) ** (1.0 / 3.0),
            fluid_density=sph_cfg.rho0,
            air_density=settings.detached_air_density,
            drag_coefficient=settings.detached_air_drag_cd,
        )
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
            collision_result = resolve_collisions(
                pos, vel, colliders,
                particle_radius=particle_radius,
                restitution=0.0,
                tangential_keep=1.0,
                iterations=settings.collision_iterations,
                collider_translations=motion_next,
                collider_velocities=collider_velocities,
                collider_rotations=collider_rotations,
                collider_centers=collider_centers,
                collider_angular_velocities=collider_angular_velocities,
                particle_mass=sph_cfg.mass,
                return_angular_impulses=True,
                return_body_impulses=bool(body_states),
            )
            if body_states:
                contacts, linear_impulses, angular_impulses = collision_result
            else:
                contacts, angular_impulses = collision_result
                linear_impulses = {}
        else:
            contacts = 0
            angular_impulses = {}
            linear_impulses = {}
        total_contacts += contacts
        perf["collision"] += time.perf_counter() - _tp

        grain_pair_contacts = 0
        grain_wall_contacts = 0
        if len(grain_pos):
            grain_previous_pos = grain_pos.copy()
            grain_vel += (
                np.asarray(sph_cfg.gravity)[None, :] + grain_hydro_acc
            ) * dt
            grain_pos += grain_vel * dt
            grain_pair_contacts = resolve_grain_contacts(
                grain_pos, grain_vel, grain_omega, grain_radii, grain_masses,
                grain_inertia, grain_friction, grain_restitution, grain_rolling,
                iterations=settings.granular_iterations,
            )
            if settings.enable_collision and colliders:
                (grain_wall_contacts, grain_linear_impulses,
                 grain_angular_impulses) = resolve_collisions(
                    grain_pos, grain_vel, colliders,
                    particle_radius=grain_radii,
                    restitution=grain_restitution,
                    friction_coefficient=grain_friction,
                    iterations=settings.collision_iterations,
                    collider_translations=motion_next,
                    collider_velocities=collider_velocities,
                    collider_rotations=collider_rotations,
                    collider_centers=collider_centers,
                    collider_angular_velocities=collider_angular_velocities,
                    particle_mass=grain_masses,
                    particle_angular_velocities=grain_omega,
                    particle_inertia=grain_inertia,
                    particle_rolling_friction=grain_rolling,
                    previous_positions=grain_previous_pos,
                    return_body_impulses=True,
                )
                for path, impulse in grain_linear_impulses.items():
                    linear_impulses[path] = linear_impulses.get(path, np.zeros(3)) + impulse
                for path, impulse in grain_angular_impulses.items():
                    angular_impulses[path] = angular_impulses.get(path, np.zeros(3)) + impulse
            total_contacts += grain_pair_contacts + grain_wall_contacts

        # The axle removes translation and all off-axis rotation. Pressure
        # supplies torque over dt; collision response supplies an impulse.
        for path, state in hinge_states.items():
            hinge = scene.hinges[path]
            torque = pressure_torques.get(path, 0.0)
            impulse = float(np.dot(
                angular_impulses.get(path, np.zeros(3)), hinge.axis
            ))
            state["omega"] += (torque * dt + impulse) / hinge.moment_of_inertia
            state["omega"] /= 1.0 + hinge.angular_damping * dt / hinge.moment_of_inertia
            if hinge.max_angular_speed > 0.0:
                state["omega"] = float(np.clip(
                    state["omega"], -hinge.max_angular_speed, hinge.max_angular_speed
                ))
            state["angle"] += state["omega"] * dt
            state["torque"] = torque

        for path, state in body_states.items():
            body = scene.dynamic_bodies[path]
            body_inertia = world_inertia(body, state)
            state["velocity"] += (
                pressure_body_forces[path] * dt
                + linear_impulses.get(path, np.zeros(3))
            ) / body.mass
            state["omega"] += np.linalg.solve(
                body_inertia,
                pressure_body_torques[path] * dt
                + angular_impulses.get(path, np.zeros(3)),
            )
            constrain_body_state(body, state)

        body_contacts, body_hinge_reactions = resolve_body_contacts(
            scene, colliders, body_states, motion_next, collider_velocities,
            collider_rotations, collider_centers, collider_angular_velocities,
            iterations=settings.collision_iterations,
        ) if body_states and colliders else (0, {})
        total_contacts += body_contacts
        for path, impulse in body_hinge_reactions.items():
            if path not in hinge_states:
                continue
            hinge = scene.hinges[path]
            hinge_states[path]["omega"] += float(np.dot(impulse, hinge.axis)) / hinge.moment_of_inertia
        for path, state in body_states.items():
            constrain_body_state(scene.dynamic_bodies[path], state)
            state["translation"] += state["velocity"] * dt
            state["rotation"] = advance_rotation(state["rotation"], state["omega"], dt)

        if scene.sinks:
            keep = _sink_keep_mask(pos, scene.sinks)
            removed_count += int(np.count_nonzero(~keep))
            if not np.all(keep):
                pos, vel, rho = pos[keep], vel[keep], rho[keep]
                particle_ids = particle_ids[keep]
                detached = detached[keep]
                for key in ("shell", "band", "support_hG"):
                    fs_state[key] = fs_state[key][keep]
                n = len(pos)
            grain_keep = _sink_keep_mask(grain_pos, scene.sinks)
            if not np.all(grain_keep):
                grain_pos, grain_vel = grain_pos[grain_keep], grain_vel[grain_keep]
                grain_omega, grain_radii = grain_omega[grain_keep], grain_radii[grain_keep]
                grain_masses, grain_inertia = grain_masses[grain_keep], grain_inertia[grain_keep]
                grain_friction = grain_friction[grain_keep]
                grain_restitution = grain_restitution[grain_keep]
                grain_rolling, grain_ids = grain_rolling[grain_keep], grain_ids[grain_keep]

        t += dt
        step += 1

        if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(vel)) and np.all(np.isfinite(rho))):
            raise RuntimeError("Non-finite solver state.")
        if not (np.all(np.isfinite(grain_pos)) and np.all(np.isfinite(grain_vel))):
            raise RuntimeError("Non-finite granular state.")
        if n and (rho.min() < 0.5 * sph_cfg.rho0 or rho.max() > 1.5 * sph_cfg.rho0):
            raise RuntimeError(f"Density instability: {rho.min():.1f} .. {rho.max():.1f} kg/m3")

        if t + 1e-12 >= next_output:
            snapshot(t, fs_state, detached)
            while next_output <= t + 1e-12:
                next_output += output_interval

        if n == 0:
            snapshot(t, fs_state, detached)
            print(f"All {removed_count} particles entered sink volumes at t={t:.4f}s")
            break

        now = time.perf_counter()
        if now - last_report >= settings.report_interval or t >= end_time - 1e-15:
            elapsed = now - start
            progress = (t - restart_time) / settings.sim_time if settings.sim_time > 0 else 1.0
            eta = elapsed * (1.0 - progress) / progress if progress > 0 else 0.0
            rr, pp, _ = hydrostatic_reference_state(pos, surface_level, sph_cfg, return_depth=True)
            qq = compute_pressure(rho, sph_cfg.rho0, sph_cfg.tait_B, sph_cfg.gamma) - pp
            speed = np.linalg.norm(vel, axis=1)
            main_v50, main_v95, main_vrms = _speed_stats(vel, ~detached)
            spray_v50, spray_v95, spray_vrms = _speed_stats(vel, detached)
            print(
                f"[{_fmt_time(elapsed)}] step {step:5d} | {progress*100:5.1f}% "
                f"| sim {t-restart_time:.4f}/{settings.sim_time:.4f}s "
                f"(scene t={t:.4f}s) | dt={dt:.7f} "
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
                f"| vmain50/95/rms={main_v50:.3f}/{main_v95:.3f}/{main_vrms:.3f} "
                f"| vspray50/95/rms={spray_v50:.3f}/{spray_v95:.3f}/{spray_vrms:.3f} "
                f"| comps={current_ncomp} main={current_main_size} | contacts={contacts}" + kin_text
            )
            if grain_sources:
                grain_vmax = (
                    float(np.linalg.norm(grain_vel, axis=1).max())
                    if len(grain_vel) else 0.0
                )
                print(
                    f"    grains={len(grain_pos)} vmax={grain_vmax:.3f} "
                    f"fluidPairs={fluid_grain_pairs} contacts="
                    f"{grain_pair_contacts + grain_wall_contacts}"
                )
            for path, state in hinge_states.items():
                print(
                    f"    axle {Path(path).name}: angle={np.degrees(state['angle']):+.2f}deg "
                    f"omega={state['omega']:+.3f}rad/s torque={state['torque']:+.5f}Nm"
                )
            for path, state in body_states.items():
                body = scene.dynamic_bodies[path]
                center = body.center + state["translation"]
                print(
                    f"    body {Path(path).name}: mass={body.mass:g}kg "
                    f"z={center[2]:.5f}m vz={state['velocity'][2]:+.4f}m/s "
                    f"FpressureZ={pressure_body_forces[path][2]:+.4f}N "
                    f"FhydroZ={hydrostatic_body_forces[path][2]:+.4f}N"
                )
            last_report = now

    if frame_times[-1] < t - 1e-12:
        snapshot(t, fs_state, detached)

    grain_capacity = max((len(frame) for frame in frame_grain_positions), default=0)
    arrays = dict(
        positions=_pad_frames(frame_positions, capacity, (3,)),
        velocity=_pad_frames(frame_velocity, capacity, (3,)),
        rho=_pad_frames(frame_rho, capacity),
        rho_reference=_pad_frames(frame_rho_ref, capacity),
        reduced_pressure=_pad_frames(frame_q, capacity),
        atmospheric_mask=_pad_frames(frame_shell, capacity, dtype=np.uint8, fill=0),
        traction_band_mask=_pad_frames(frame_band, capacity, dtype=np.uint8, fill=0),
        current_surface_hG=_pad_frames(frame_hG, capacity),
        detached_mask=_pad_frames(frame_detached, capacity, dtype=np.uint8, fill=0),
        active_particle_ids=_pad_frames(frame_ids, capacity, dtype=np.int32, fill=-1),
        active_counts=np.asarray([len(ids) for ids in frame_ids], dtype=np.int32),
        removed_count=np.asarray(removed_count, dtype=np.int32),
        times=np.asarray(frame_times, dtype=np.float64),
        source_file=np.asarray(str(usd_file)),
        source_initial_positions=np.asarray(initial_positions, dtype=np.float64),
        surface_level=np.asarray(surface_level),
        dp=np.asarray(sph_cfg.dp), h=np.asarray(sph_cfg.h), support=np.asarray(sph_cfg.support),
        mass=np.asarray(sph_cfg.mass), rho0=np.asarray(sph_cfg.rho0), c0=np.asarray(sph_cfg.c0),
        gravity=np.asarray(sph_cfg.gravity, dtype=np.float64),
        free_surface_mode=np.asarray(fs_cfg.mode),
        usd_start_time_code=np.asarray(scene.start_time_code),
        usd_time_codes_per_second=np.asarray(scene.time_codes_per_second),
        hinge_paths=np.asarray(hinge_paths),
        hinge_angles=np.asarray(frame_hinge_angles, dtype=np.float64),
        hinge_angular_velocities=np.asarray(frame_hinge_omegas, dtype=np.float64),
        body_paths=np.asarray(body_paths),
        body_translations=np.asarray([
            frame_body_translations[path] for path in body_paths
        ], dtype=np.float64),
        body_rotations=np.asarray([
            frame_body_rotations[path] for path in body_paths
        ], dtype=np.float64),
        granular_positions=_pad_frames(
            frame_grain_positions, grain_capacity, (3,)
        ),
        granular_velocity=_pad_frames(
            frame_grain_velocities, grain_capacity, (3,)
        ),
        granular_active_ids=_pad_frames(
            frame_grain_ids, grain_capacity, dtype=np.int32, fill=-1
        ),
        granular_active_counts=np.asarray([
            len(frame) for frame in frame_grain_positions
        ], dtype=np.int32),
    )
    print("\nWriting outputs...")
    write_preview(t)
    save_npz(npz_output, **arrays)
    write_animation(usd_output)

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
    if n:
        print(f"rho min/mean/max : {rho.min():.4f} / {rho.mean():.4f} / {rho.max():.4f}")
    if n:
        print(f"rho-ref abs mean/max: {np.mean(np.abs(rho-rr)):.6f} / {np.max(np.abs(rho-rr)):.6f} kg/m3")
        print(f"q abs mean/max  : {np.mean(np.abs(qq)):.3f} / {np.max(np.abs(qq)):.3f} Pa")
    print(f"v max           : {np.linalg.norm(vel, axis=1).max():.6f} m/s" if n else "v max           : n/a")
    print(f"Sink removed    : {removed_count} / {capacity}")
    print(f"Contacts        : {total_contacts}")
    print(f"Kinematic max travel seen: {max_collider_travel_seen:.6f} m")
    for path, state in hinge_states.items():
        print(
            f"Axle {Path(path).name}: angle={np.degrees(state['angle']):+.3f} deg, "
            f"omega={state['omega']:+.6f} rad/s"
        )
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
        hinge_angles={path: state["angle"] for path, state in hinge_states.items()},
        hinge_angular_velocities={path: state["omega"] for path, state in hinge_states.items()},
    )
