"""Small CPU granular solver for spherical grains authored in USD.

This is the first GraSPH granular milestone: initial volume sources, gravity,
sphere/sphere contacts, mesh contacts, sinks and collision-driven hinges.  It
is deliberately separate from WCSPH so a later fluid/granular coupling layer
can combine two well-defined solvers instead of conflating their physics.
"""

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np

from grasph_collision import build_colliders, resolve_collisions
from grasph_output import save_npz, write_usd_animation
from grasph_rigid import (
    advance_rotation as _advance_rotation,
    axis_angle_matrix as _axis_angle_matrix,
    resolve_body_contacts,
    rotation_to_euler_xyz as _rotation_to_euler_xyz,
    world_inertia,
)
from grasph_scene import generate_granular_particles
from grasph_sph import build_neighbor_pairs


@dataclass
class GranularResult:
    scene_file: str
    npz_file: str
    usd_file: str
    positions: np.ndarray
    velocity: np.ndarray
    time: float
    steps: int


def _sink_keep_mask(pos, sinks):
    keep = np.ones(len(pos), dtype=bool)
    for sink in sinks:
        keep &= ~np.all((pos >= sink.bounds_min) & (pos <= sink.bounds_max), axis=1)
    return keep


def _pad_frames(frames, capacity, trailing_shape=(), dtype=np.float32, fill=np.nan):
    out = np.full((len(frames), capacity) + trailing_shape, fill, dtype=dtype)
    for index, frame in enumerate(frames):
        out[index, :len(frame)] = frame
    return out


def _axis_centers(lo, hi, spacing):
    start = float(lo) + 0.5 * float(spacing)
    stop = float(hi) - 0.5 * float(spacing)
    if stop < start - 1.0e-12:
        return np.empty(0, dtype=np.float64)
    count = int(np.floor((stop - start) / float(spacing) + 1.0e-7)) + 1
    return start + np.arange(count, dtype=np.float64) * float(spacing)


def _flow_candidates(source):
    """Create outlet points on the downstream face of a box-shaped source."""
    radius = float(source.particle_radius)
    diameter = 2.0 * radius
    velocity = np.asarray(source.initial_velocity, dtype=np.float64)
    axis = int(np.argmax(np.abs(velocity))) if np.any(velocity) else 2
    sign = float(np.sign(velocity[axis])) if velocity[axis] != 0.0 else -1.0
    axes = [
        _axis_centers(source.mesh.bounds_min[k], source.mesh.bounds_max[k], diameter)
        for k in range(3)
    ]
    face = (source.mesh.bounds_max[axis] - radius if sign > 0.0
            else source.mesh.bounds_min[axis] + radius)
    axes[axis] = np.asarray([face], dtype=np.float64)
    if any(len(values) == 0 for values in axes):
        return np.empty((0, 3), dtype=np.float64)
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    return np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))


def _append_particles(pos, velocity, angular_velocity, radii, masses, inertia,
                      friction, restitution, rolling_friction, particle_ids,
                      new_pos, source, next_id):
    count = len(new_pos)
    if count == 0:
        return (pos, velocity, angular_velocity, radii, masses, inertia,
                friction, restitution, rolling_friction, particle_ids, next_id)
    radius = float(source.particle_radius)
    mass = float(source.density) * (4.0 / 3.0) * np.pi * radius**3
    pos = np.vstack((pos, new_pos))
    velocity = np.vstack((velocity, np.repeat(
        np.asarray(source.initial_velocity, dtype=np.float64)[None, :], count, axis=0
    )))
    angular_velocity = np.vstack((angular_velocity, np.zeros((count, 3))))
    radii = np.append(radii, np.full(count, radius))
    masses = np.append(masses, np.full(count, mass))
    inertia = np.append(inertia, np.full(count, 0.4 * mass * radius**2))
    friction = np.append(friction, np.full(count, float(source.friction)))
    restitution = np.append(restitution, np.full(count, float(source.restitution)))
    rolling_friction = np.append(
        rolling_friction, np.full(count, float(source.rolling_friction))
    )
    particle_ids = np.append(
        particle_ids, np.arange(next_id, next_id + count, dtype=np.int32)
    )
    return (pos, velocity, angular_velocity, radii, masses, inertia,
            friction, restitution, rolling_friction, particle_ids, next_id + count)


def resolve_grain_contacts(pos, vel, omega, radii, masses, inertia,
                           friction, restitution, rolling_friction, iterations=4):
    """Projected sphere contacts plus restitution and Coulomb friction."""
    if len(pos) < 2:
        return 0
    total = 0
    inverse_mass = 1.0 / masses
    search_radius = 2.0 * float(np.max(radii))

    for _ in range(max(1, int(iterations))):
        pair_i, pair_j = build_neighbor_pairs(pos, search_radius)
        if len(pair_i) == 0:
            break
        delta = pos[pair_j] - pos[pair_i]
        distance = np.linalg.norm(delta, axis=1)
        target = radii[pair_i] + radii[pair_j]
        active = distance < target
        if not np.any(active):
            break
        i, j = pair_i[active], pair_j[active]
        d = distance[active]
        n = delta[active] / np.maximum(d[:, None], 1.0e-12)
        degenerate = d < 1.0e-12
        if np.any(degenerate):
            n[degenerate] = (1.0, 0.0, 0.0)

        wi, wj = inverse_mass[i], inverse_mass[j]
        wsum = wi + wj
        penetration = target[active] - d
        # Relaxed Jacobi projection behaves well in dense initial packings.
        correction = 0.65 * penetration / np.maximum(wsum, 1.0e-30)
        np.add.at(pos, i, -n * (correction * wi)[:, None])
        np.add.at(pos, j, n * (correction * wj)[:, None])

        arm_i = n * radii[i, None]
        arm_j = -n * radii[j, None]
        contact_i = vel[i] + np.cross(omega[i], arm_i)
        contact_j = vel[j] + np.cross(omega[j], arm_j)
        relative = contact_j - contact_i
        vn = np.einsum("ij,ij->i", relative, n)
        incoming = vn < 0.0
        if np.any(incoming):
            ii, jj, nn = i[incoming], j[incoming], n[incoming]
            denom = wsum[incoming]
            bounce = np.minimum(restitution[ii], restitution[jj])
            normal_impulse = -(1.0 + bounce) * vn[incoming] / denom
            impulse = normal_impulse[:, None] * nn

            tangent = relative[incoming] - vn[incoming, None] * nn
            tangent_speed = np.linalg.norm(tangent, axis=1)
            tangent_dir = tangent / np.maximum(tangent_speed[:, None], 1.0e-12)
            mu = np.sqrt(friction[ii] * friction[jj])
            tangent_denom = (
                denom
                + radii[ii]**2 / inertia[ii]
                + radii[jj]**2 / inertia[jj]
            )
            friction_impulse = np.minimum(
                tangent_speed / tangent_denom, mu * normal_impulse
            )
            impulse -= friction_impulse[:, None] * tangent_dir
            np.add.at(vel, ii, -impulse * inverse_mass[ii, None])
            np.add.at(vel, jj, impulse * inverse_mass[jj, None])
            np.add.at(
                omega, ii,
                np.cross(arm_i[incoming], -impulse) / inertia[ii, None],
            )
            np.add.at(
                omega, jj,
                np.cross(arm_j[incoming], impulse) / inertia[jj, None],
            )
            relative_omega = omega[jj] - omega[ii]
            rolling_omega = (
                relative_omega
                - np.einsum("ij,ij->i", relative_omega, nn)[:, None] * nn
            )
            rolling_speed = np.linalg.norm(rolling_omega, axis=1)
            rolling_direction = rolling_omega / np.maximum(
                rolling_speed[:, None], 1.0e-12
            )
            rolling_mu = np.sqrt(rolling_friction[ii] * rolling_friction[jj])
            rolling_effective_inverse = 1.0 / inertia[ii] + 1.0 / inertia[jj]
            rolling_torque_impulse = np.minimum(
                rolling_speed / rolling_effective_inverse,
                rolling_mu * normal_impulse * np.minimum(radii[ii], radii[jj]),
            )
            torque_impulse = rolling_torque_impulse[:, None] * rolling_direction
            np.add.at(omega, ii, torque_impulse / inertia[ii, None])
            np.add.at(omega, jj, -torque_impulse / inertia[jj, None])
        total += len(i)
    return total


def run_granular_simulation(usd_file, scene, settings, gravity=(0.0, 0.0, -9.81)):
    pos, source_ids = generate_granular_particles(scene)

    sources = scene.granular_sources
    flow_sources = [source for source in sources if source.source_mode == "flow"]
    if len(pos) == 0 and not flow_sources:
        raise RuntimeError("No granular particles or Granular Flow Source found.")
    radii = np.asarray([sources[i].particle_radius for i in source_ids], dtype=np.float64)
    density = np.asarray([sources[i].density for i in source_ids], dtype=np.float64)
    friction = np.asarray([sources[i].friction for i in source_ids], dtype=np.float64)
    restitution = np.asarray([sources[i].restitution for i in source_ids], dtype=np.float64)
    rolling_friction = np.asarray(
        [sources[i].rolling_friction for i in source_ids], dtype=np.float64
    )
    masses = density * (4.0 / 3.0) * np.pi * radii**3
    inertia = 0.4 * masses * radii**2
    velocity = np.zeros_like(pos)
    angular_velocity = np.zeros_like(pos)
    particle_ids = np.arange(len(pos), dtype=np.int32)
    next_particle_id = len(pos)

    flow_states = []
    for source in flow_sources:
        candidates = _flow_candidates(source)
        if len(candidates) == 0:
            raise RuntimeError(
                f"Granular Flow Source {source.mesh.path} is too small for its particle radius"
            )
        flow_states.append({
            "source": source, "candidates": candidates, "cursor": 0,
            "remainder": 0.0, "emitted": 0, "blocked": 0,
        })

    stem = settings.output_stem or (Path(str(usd_file)).stem + "_grasph")
    out_dir = Path(settings.output_dir)
    npz_output = out_dir / f"{stem}.npz"
    usd_output = out_dir / f"{stem}.usdc"
    colliders = build_colliders(
        scene,
        voxel_size=max(min(float(source.particle_radius) for source in sources), 1.0e-4),
        margin=2.5 * max(float(source.particle_radius) for source in sources),
    ) if scene.colliders else []

    hinge_states = {
        path: {"angle": 0.0, "omega": 0.0}
        for path in scene.hinges
    }
    hinge_centers = {path: hinge.center for path, hinge in scene.hinges.items()}
    hinge_paths = list(scene.hinges)
    body_states = {
        path: {
            "translation": np.zeros(3), "velocity": np.zeros(3),
            "rotation": np.eye(3), "omega": np.zeros(3),
        }
        for path in scene.dynamic_bodies
    }
    body_paths = list(scene.dynamic_bodies)
    collider_centers = dict(hinge_centers)
    collider_centers.update({
        path: body.center for path, body in scene.dynamic_bodies.items()
    })
    frame_positions, frame_velocity, frame_omega, frame_ids, frame_times = [], [], [], [], []
    frame_hinge_angles = []
    frame_body_translations = {path: [] for path in body_paths}
    frame_body_rotations = {path: [] for path in body_paths}

    def snapshot(t):
        frame_positions.append(pos.copy())
        frame_velocity.append(velocity.copy())
        frame_omega.append(angular_velocity.copy())
        frame_ids.append(particle_ids.copy())
        frame_times.append(float(t))
        frame_hinge_angles.append(np.asarray([
            hinge_states[path]["angle"] for path in hinge_paths
        ], dtype=np.float64))
        for path in body_paths:
            frame_body_translations[path].append(
                body_states[path]["translation"].copy()
            )
            frame_body_rotations[path].append(
                _rotation_to_euler_xyz(body_states[path]["rotation"])
            )

    snapshot(0.0)
    output_interval = 1.0 / settings.output_fps
    next_output = output_interval
    dt_limit = float(getattr(settings, "granular_dt", 0.001))
    iterations = int(getattr(settings, "granular_iterations", 4))
    gravity = np.asarray(gravity, dtype=np.float64)
    t = 0.0
    step = 0
    removed = 0
    total_pair_contacts = 0
    total_wall_contacts = 0
    started = time.perf_counter()
    last_report = started

    print("\n========================================")
    print("STARTING GRANULAR DEM/PBD")
    print("========================================")
    print(f"Initial grains: {len(pos)}")
    print(f"Flow sources  : {len(flow_states)}")
    source_radii = np.asarray([source.particle_radius for source in sources])
    print(f"Radius range  : {source_radii.min():g} .. {source_radii.max():g} m")
    print(f"Time step max : {dt_limit:g} s")
    print(f"Iterations    : {iterations}\n")

    while t < settings.sim_time - 1.0e-15:
        dt = min(dt_limit, settings.sim_time - t)
        for path, state in body_states.items():
            body = scene.dynamic_bodies[path]
            state["velocity"] += gravity * dt
            state["velocity"] /= 1.0 + body.linear_damping * dt
            state["omega"] /= 1.0 + body.angular_damping * dt
        motion_now = scene.collider_motion(t) if scene.colliders else {}
        motion_next = scene.collider_motion(t + dt) if scene.colliders else {}
        collider_velocities = {
            path: (motion_next[path] - motion_now[path]) / max(dt, 1.0e-30)
            for path in motion_now
        }
        for path, state in body_states.items():
            motion_now[path] = state["translation"].copy()
            motion_next[path] = state["translation"].copy()
            collider_velocities[path] = state["velocity"].copy()
        rotations = {
            path: _axis_angle_matrix(scene.hinges[path].axis, state["angle"])
            for path, state in hinge_states.items()
        }
        angular_velocities = {
            path: scene.hinges[path].axis * state["omega"]
            for path, state in hinge_states.items()
        }
        rotations.update({path: state["rotation"] for path, state in body_states.items()})
        angular_velocities.update({path: state["omega"] for path, state in body_states.items()})

        # Flow sources accumulate a deterministic grains/second request. A
        # blocked outlet throttles the actual flow instead of allowing overlap.
        for flow in flow_states:
            source = flow["source"]
            active_dt = max(
                0.0,
                min(t + dt, source.end_time) - max(t, source.start_time),
            )
            flow["remainder"] += source.flow_rate * active_dt
            requested = int(np.floor(flow["remainder"] + 1.0e-12))
            flow["remainder"] -= requested
            accepted = []
            for _ in range(requested):
                candidate = flow["candidates"][flow["cursor"]].copy()
                flow["cursor"] = (flow["cursor"] + 1) % len(flow["candidates"])
                if len(pos):
                    clearance = np.linalg.norm(pos - candidate[None, :], axis=1)
                    if np.any(clearance < radii + source.particle_radius * 0.98):
                        flow["blocked"] += 1
                        continue
                if accepted:
                    accepted_distance = np.linalg.norm(
                        np.asarray(accepted) - candidate[None, :], axis=1
                    )
                    if np.any(accepted_distance < 1.96 * source.particle_radius):
                        flow["blocked"] += 1
                        continue
                accepted.append(candidate)
            if accepted:
                new_pos = np.asarray(accepted, dtype=np.float64)
                (pos, velocity, angular_velocity, radii, masses, inertia,
                 friction, restitution, rolling_friction, particle_ids,
                 next_particle_id) = _append_particles(
                    pos, velocity, angular_velocity, radii, masses, inertia,
                    friction, restitution, rolling_friction, particle_ids,
                    new_pos, source, next_particle_id,
                )
                flow["emitted"] += len(new_pos)

        previous_pos = pos.copy()
        velocity += gravity[None, :] * dt
        pos += velocity * dt
        pair_contacts = resolve_grain_contacts(
            pos, velocity, angular_velocity, radii, masses, inertia,
            friction, restitution, rolling_friction, iterations
        )
        total_pair_contacts += pair_contacts

        if settings.enable_collision and colliders:
            # The first milestone uses the mean material response at walls;
            # per-source values remain available for the next contact model.
            wall_contacts, linear_impulses, angular_impulses = resolve_collisions(
                pos, velocity, colliders, particle_radius=radii,
                restitution=restitution,
                friction_coefficient=friction,
                iterations=settings.collision_iterations,
                collider_translations=motion_next,
                collider_velocities=collider_velocities,
                collider_rotations=rotations,
                collider_centers=collider_centers,
                collider_angular_velocities=angular_velocities,
                particle_mass=masses,
                particle_angular_velocities=angular_velocity,
                particle_inertia=inertia,
                particle_rolling_friction=rolling_friction,
                previous_positions=previous_pos,
                return_angular_impulses=True,
                return_body_impulses=True,
            )
        else:
            wall_contacts, linear_impulses, angular_impulses = 0, {}, {}
        total_wall_contacts += wall_contacts

        for path, state in body_states.items():
            body = scene.dynamic_bodies[path]
            state["velocity"] += linear_impulses.get(path, np.zeros(3)) / body.mass
            body_world_inertia = world_inertia(body, state)
            state["omega"] += np.linalg.solve(
                body_world_inertia, angular_impulses.get(path, np.zeros(3))
            )

        body_contacts, body_reaction_angular = resolve_body_contacts(
            scene, colliders, body_states, motion_next, collider_velocities,
            rotations, collider_centers, angular_velocities,
            iterations=settings.collision_iterations,
        ) if body_states and colliders else (0, {})
        total_wall_contacts += body_contacts

        for path, state in body_states.items():
            state["translation"] += state["velocity"] * dt
            state["rotation"] = _advance_rotation(
                state["rotation"], state["omega"], dt
            )

        for path, state in hinge_states.items():
            hinge = scene.hinges[path]
            impulse = float(np.dot(
                (angular_impulses.get(path, np.zeros(3))
                 + body_reaction_angular.get(path, np.zeros(3))), hinge.axis
            ))
            state["omega"] += impulse / hinge.moment_of_inertia
            state["omega"] /= 1.0 + hinge.angular_damping * dt / hinge.moment_of_inertia
            if hinge.max_angular_speed > 0.0:
                state["omega"] = float(np.clip(
                    state["omega"], -hinge.max_angular_speed, hinge.max_angular_speed
                ))
            state["angle"] += state["omega"] * dt

        if scene.sinks:
            keep = _sink_keep_mask(pos, scene.sinks)
            removed += int(np.count_nonzero(~keep))
            if not np.all(keep):
                pos, velocity = pos[keep], velocity[keep]
                angular_velocity = angular_velocity[keep]
                radii, masses, inertia = radii[keep], masses[keep], inertia[keep]
                friction, restitution = friction[keep], restitution[keep]
                rolling_friction = rolling_friction[keep]
                particle_ids = particle_ids[keep]

        t += dt
        step += 1
        if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(velocity))):
            raise RuntimeError("Non-finite granular solver state.")
        if t + 1.0e-12 >= next_output:
            snapshot(t)
            while next_output <= t + 1.0e-12:
                next_output += output_interval

        now = time.perf_counter()
        if now - last_report >= settings.report_interval or t >= settings.sim_time - 1.0e-15:
            elapsed = now - started
            progress = t / settings.sim_time if settings.sim_time > 0 else 1.0
            eta = elapsed * (1.0 - progress) / max(progress, 1.0e-12)
            vmax = float(np.linalg.norm(velocity, axis=1).max()) if len(pos) else 0.0
            print(
                f"[{elapsed:6.1f}s] step {step:6d} | {progress*100:5.1f}% | "
                f"sim {t:.4f}/{settings.sim_time:.4f}s | dt={dt:.6f} | "
                f"ETA {eta:6.1f}s"
            )
            print(
                f"    grains={len(pos)} vmax={vmax:.3f} "
                f"pairContacts={pair_contacts} wallContacts={wall_contacts}"
            )
            for path, state in hinge_states.items():
                print(
                    f"    axle {Path(path).name}: angle={np.degrees(state['angle']):+.2f}deg "
                    f"omega={state['omega']:+.3f}rad/s"
                )
            last_report = now

    if frame_times[-1] < t - 1.0e-12:
        snapshot(t)

    capacity = max((len(frame) for frame in frame_positions), default=0)
    arrays = dict(
        positions=_pad_frames(frame_positions, capacity, (3,)),
        velocity=_pad_frames(frame_velocity, capacity, (3,)),
        angular_velocity=_pad_frames(frame_omega, capacity, (3,)),
        active_particle_ids=_pad_frames(frame_ids, capacity, dtype=np.int32, fill=-1),
        active_counts=np.asarray([len(ids) for ids in frame_ids], dtype=np.int32),
        times=np.asarray(frame_times, dtype=np.float64),
        particle_radius=np.asarray([source.particle_radius for source in sources], dtype=np.float64),
        removed_count=np.asarray(removed, dtype=np.int32),
        source_file=np.asarray(str(usd_file)),
        solver_kind=np.asarray("granular"),
    )
    print("\nWriting granular outputs...")
    save_npz(npz_output, **arrays)
    # Current sources normally share a radius. Variable widths can be added
    # once variable-radius emission is enabled; use the authored mean now.
    output_width = 2.0 * float(np.mean(source_radii))
    write_usd_animation(
        scene.stage, frame_positions, frame_times, settings.output_fps,
        output_width, usd_output,
        source_start_time_code=scene.start_time_code,
        source_tps=scene.time_codes_per_second,
        frame_velocities=frame_velocity,
        prim_path="/GraSPH_GranularResult",
        particle_widths=output_width,
        result_kind="granular",
        hinge_frames={
            path: {
                "driver_path": scene.hinges[path].driver_path,
                "axis": scene.hinges[path].axis,
                "angles": np.asarray(frame_hinge_angles)[:, index],
            }
            for index, path in enumerate(hinge_paths)
        },
        body_frames={
            path: {
                "driver_path": scene.dynamic_bodies[path].driver_path,
                "translations": np.asarray(frame_body_translations[path]),
                "rotations": np.asarray(frame_body_rotations[path]),
            }
            for path in body_paths
        },
    )
    elapsed = time.perf_counter() - started
    print("\n========================================")
    print("GRANULAR SIMULATION COMPLETE")
    print("========================================")
    print(f"Simulated time : {t:.6f} s")
    print(f"Steps          : {step}")
    print(f"Solver wall    : {elapsed:.3f} s")
    print(f"Pair contacts  : {total_pair_contacts}")
    print(f"Wall contacts  : {total_wall_contacts}")
    print(f"Sink removed   : {removed} / {capacity}")
    for flow in flow_states:
        print(
            f"Flow {flow['source'].mesh.path}: emitted={flow['emitted']} "
            f"blocked={flow['blocked']}"
        )
    return GranularResult(
        scene_file=str(usd_file), npz_file=str(npz_output), usd_file=str(usd_output),
        positions=pos, velocity=velocity, time=t, steps=step,
    )
