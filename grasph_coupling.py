"""Fast momentum-conserving coupling between SPH particles and DEM spheres."""

import numpy as np

from grasph_sph import build_cross_neighbor_pairs


def couple_fluid_grains(
    fluid_pos, fluid_vel, fluid_rho, fluid_mass,
    grain_pos, grain_vel, grain_radii, grain_masses,
    gravity, support, drag_coefficient=0.47,
):
    """Return fluid and grain accelerations with equal/opposite total force.

    Fluid velocity and density are interpolated with a compact cubic weight.
    Spheres receive Archimedes buoyancy plus quadratic drag. Each force is
    scattered back to the contributing SPH particles, conserving momentum.
    """
    fluid_acc = np.zeros_like(fluid_pos, dtype=np.float64)
    grain_acc = np.zeros_like(grain_pos, dtype=np.float64)
    if not len(fluid_pos) or not len(grain_pos):
        return fluid_acc, grain_acc, 0

    fluid_ids, grain_ids = build_cross_neighbor_pairs(
        fluid_pos, grain_pos, float(support)
    )
    if not len(fluid_ids):
        return fluid_acc, grain_acc, 0

    delta = fluid_pos[fluid_ids] - grain_pos[grain_ids]
    distance = np.linalg.norm(delta, axis=1)
    q = np.clip(1.0 - distance / float(support), 0.0, 1.0)
    weight = q * q * q + 1.0e-12
    weight_sum = np.bincount(
        grain_ids, weights=weight, minlength=len(grain_pos)
    )
    normalized = weight / np.maximum(weight_sum[grain_ids], 1.0e-30)

    sampled_velocity = np.zeros_like(grain_pos, dtype=np.float64)
    sampled_density = np.zeros(len(grain_pos), dtype=np.float64)
    np.add.at(
        sampled_velocity, grain_ids,
        fluid_vel[fluid_ids] * normalized[:, None],
    )
    np.add.at(
        sampled_density, grain_ids,
        fluid_rho[fluid_ids] * normalized,
    )
    wet = weight_sum > 0.0
    volume = (4.0 / 3.0) * np.pi * grain_radii**3
    area = np.pi * grain_radii**2
    relative_velocity = sampled_velocity - grain_vel
    relative_speed = np.linalg.norm(relative_velocity, axis=1)
    buoyancy = -sampled_density[:, None] * volume[:, None] * np.asarray(
        gravity, dtype=np.float64
    )[None, :]
    drag = (
        0.5 * float(drag_coefficient) * sampled_density * area * relative_speed
    )[:, None] * relative_velocity
    force = buoyancy + drag
    force[~wet] = 0.0
    grain_acc[:] = force / grain_masses[:, None]

    pair_reaction = -force[grain_ids] * normalized[:, None]
    np.add.at(fluid_acc, fluid_ids, pair_reaction / float(fluid_mass))
    return fluid_acc, grain_acc, len(fluid_ids)
