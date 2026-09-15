import math
import time
from dataclasses import dataclass

import numpy as np


# ============================================================
# GraSPH - WCSPH CORE
# Pure fluid mathematics. No USD / Blender / collider geometry.
# ============================================================


@dataclass
class SPHConfig:
    dp: float = 0.020
    rho0: float = 1000.0
    h_factor: float = 1.30
    c0: float = 20.0
    gamma: float = 7.0
    gravity: tuple = (0.0, 0.0, -9.81)

    @property
    def h(self):
        return self.h_factor * self.dp

    @property
    def support(self):
        return 2.0 * self.h

    @property
    def mass(self):
        return self.rho0 * self.dp**3

    @property
    def tait_B(self):
        return self.rho0 * self.c0**2 / self.gamma


class WendlandC2:
    def __init__(self, h):
        self.h = float(h)
        self.support = 2.0 * self.h
        self.alpha = 21.0 / (16.0 * math.pi * self.h**3)

    def W(self, r):
        r = np.asarray(r, dtype=np.float64)
        q = r / self.h
        result = np.zeros_like(q)
        active = q < 2.0
        qa = q[active]
        a = 1.0 - 0.5 * qa
        result[active] = self.alpha * a**4 * (2.0 * qa + 1.0)
        return result

    def gradW(self, rij, r):
        rij = np.asarray(rij, dtype=np.float64)
        r = np.asarray(r, dtype=np.float64)
        grad = np.zeros_like(rij)

        active = (r > 1e-12) & (r < self.support)
        if not np.any(active):
            return grad

        ra = r[active]
        q = ra / self.h
        a = 1.0 - 0.5 * q
        dWdr = -5.0 * self.alpha * q * a**3 / self.h
        grad[active] = dWdr[:, None] * rij[active] / ra[:, None]
        return grad


HALF_NEIGHBOR_OFFSETS = []
for dx in (-1, 0, 1):
    for dy in (-1, 0, 1):
        for dz in (-1, 0, 1):
            if dx == 0 and dy == 0 and dz == 0:
                continue
            if (
                dx > 0
                or (dx == 0 and dy > 0)
                or (dx == 0 and dy == 0 and dz > 0)
            ):
                HALF_NEIGHBOR_OFFSETS.append((dx, dy, dz))


ALL_NEIGHBOR_OFFSETS = [
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
]


def build_neighbor_pairs(pos, support):
    n = len(pos)
    if n < 2:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    cell_size = support
    cell_coords = np.floor(pos / cell_size).astype(np.int64)
    cells = {}

    for particle_index, c in enumerate(cell_coords):
        key = (int(c[0]), int(c[1]), int(c[2]))
        cells.setdefault(key, []).append(particle_index)

    pair_i_parts = []
    pair_j_parts = []
    support2 = support * support

    for key, indices_list in cells.items():
        A = np.asarray(indices_list, dtype=np.int32)

        if len(A) >= 2:
            ia, ja = np.triu_indices(len(A), k=1)
            I = A[ia]
            J = A[ja]
            rij = pos[I] - pos[J]
            r2 = np.einsum("ij,ij->i", rij, rij)
            keep = r2 < support2
            if np.any(keep):
                pair_i_parts.append(I[keep])
                pair_j_parts.append(J[keep])

        kx, ky, kz = key
        for dx, dy, dz in HALF_NEIGHBOR_OFFSETS:
            neighbour_list = cells.get((kx + dx, ky + dy, kz + dz))
            if neighbour_list is None:
                continue

            B = np.asarray(neighbour_list, dtype=np.int32)
            I = np.repeat(A, len(B))
            J = np.tile(B, len(A))
            rij = pos[I] - pos[J]
            r2 = np.einsum("ij,ij->i", rij, rij)
            keep = r2 < support2
            if np.any(keep):
                pair_i_parts.append(I[keep])
                pair_j_parts.append(J[keep])

    if not pair_i_parts:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    return np.concatenate(pair_i_parts), np.concatenate(pair_j_parts)


def build_cross_neighbor_pairs(fluid_pos, other_pos, support):
    """Unique cross-set neighbour pairs within support (fluid -> other)."""
    if len(fluid_pos) == 0 or len(other_pos) == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    cell_size = support
    support2 = support * support

    other_cells = {}
    other_coords = np.floor(other_pos / cell_size).astype(np.int64)
    for gi, c in enumerate(other_coords):
        key = (int(c[0]), int(c[1]), int(c[2]))
        other_cells.setdefault(key, []).append(gi)

    fluid_coords = np.floor(fluid_pos / cell_size).astype(np.int64)
    fluid_cells = {}
    for fi, c in enumerate(fluid_coords):
        key = (int(c[0]), int(c[1]), int(c[2]))
        fluid_cells.setdefault(key, []).append(fi)

    fi_parts = []
    gj_parts = []

    for key, fluid_list in fluid_cells.items():
        F = np.asarray(fluid_list, dtype=np.int32)
        kx, ky, kz = key

        for dx, dy, dz in ALL_NEIGHBOR_OFFSETS:
            other_list = other_cells.get((kx + dx, ky + dy, kz + dz))
            if other_list is None:
                continue

            G = np.asarray(other_list, dtype=np.int32)
            I = np.repeat(F, len(G))
            J = np.tile(G, len(F))

            rij = fluid_pos[I] - other_pos[J]
            r2 = np.einsum("ij,ij->i", rij, rij)
            keep = r2 < support2

            if np.any(keep):
                fi_parts.append(I[keep])
                gj_parts.append(J[keep])

    if not fi_parts:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    return np.concatenate(fi_parts), np.concatenate(gj_parts)


def compute_pair_geometry(pos, pair_i, pair_j, kernel):
    rij = pos[pair_i] - pos[pair_j]
    r = np.linalg.norm(rij, axis=1)
    gradW = kernel.gradW(rij, r)
    return rij, r, gradW


def compute_cross_pair_geometry(fluid_pos, other_pos, pair_i, pair_j, kernel):
    rij = fluid_pos[pair_i] - other_pos[pair_j]
    r = np.linalg.norm(rij, axis=1)
    gradW = kernel.gradW(rij, r)
    return rij, r, gradW


def neighbor_counts(n_particles, pair_i, pair_j):
    counts = np.zeros(n_particles, dtype=np.int32)
    np.add.at(counts, pair_i, 1)
    np.add.at(counts, pair_j, 1)
    return counts


def compute_density_rate(vel, pair_i, pair_j, gradW, mass):
    n = len(vel)
    drho = np.zeros(n, dtype=np.float64)

    vij = vel[pair_i] - vel[pair_j]
    pair_term = mass * np.einsum("ij,ij->i", vij, gradW)

    np.add.at(drho, pair_i, pair_term)
    np.add.at(drho, pair_j, pair_term)
    return drho


def compute_pressure(rho, rho0, B, gamma):
    return B * ((rho / rho0) ** gamma - 1.0)


def compute_pressure_acceleration(
    rho,
    pressure,
    pair_i,
    pair_j,
    gradW,
    mass,
):
    n = len(rho)
    acc = np.zeros((n, 3), dtype=np.float64)

    rhoi = rho[pair_i]
    rhoj = rho[pair_j]
    pi = pressure[pair_i]
    pj = pressure[pair_j]

    pressure_factor = mass * (pi / (rhoi * rhoi) + pj / (rhoj * rhoj))
    pair_force = pressure_factor[:, None] * gradW

    np.add.at(acc, pair_i, -pair_force)
    np.add.at(acc, pair_j, pair_force)
    return acc


def compute_artificial_viscosity_acceleration(
    pos,
    vel,
    rho,
    pair_i,
    pair_j,
    rij,
    r,
    gradW,
    mass,
    h,
    c0,
    alpha=0.05,
    beta=0.0,
):
    n = len(pos)
    acc = np.zeros((n, 3), dtype=np.float64)

    if len(pair_i) == 0:
        return acc

    vij = vel[pair_i] - vel[pair_j]
    vr = np.einsum("ij,ij->i", vij, rij)
    approaching = vr < 0.0

    if not np.any(approaching):
        return acc

    ids = np.where(approaching)[0]
    r2 = r[ids] * r[ids]
    eta2 = 0.01 * h * h
    mu = h * vr[ids] / (r2 + eta2)

    rho_bar = 0.5 * (rho[pair_i[ids]] + rho[pair_j[ids]])
    pi_ij = (-alpha * c0 * mu + beta * mu * mu) / rho_bar

    pair_acc = mass * pi_ij[:, None] * gradW[ids]
    np.add.at(acc, pair_i[ids], -pair_acc)
    np.add.at(acc, pair_j[ids], pair_acc)
    return acc


def add_gravity(acc, gravity):
    acc += np.asarray(gravity, dtype=np.float64)


def compute_cfl_dt(vel, h, c0, cfl=0.25):
    if len(vel) == 0:
        return 1e-3
    speed = np.linalg.norm(vel, axis=1)
    vmax = speed.max()
    return cfl * h / (c0 + vmax)


def initialize_hydrostatic_density(
    pos,
    rho0,
    B,
    gamma,
    gravity,
    surface_z=None,
):
    gravity = np.asarray(gravity, dtype=np.float64)
    g = np.linalg.norm(gravity)

    if surface_z is None:
        surface_z = pos[:, 2].max()

    depth = np.maximum(surface_z - pos[:, 2], 0.0)
    pressure = rho0 * g * depth
    rho = rho0 * (1.0 + pressure / B) ** (1.0 / gamma)
    return rho, pressure


if __name__ == "__main__":
    from grasph_scene import load_scene, generate_fluid_particles

    print()
    print("========================================")
    print("GraSPH WCSPH CORE TEST")
    print("========================================")

    cfg = SPHConfig()
    scene = load_scene("TankTest.usda")
    pos = generate_fluid_particles(scene, dp=cfg.dp)
    vel = np.zeros_like(pos)
    rho = np.full(len(pos), cfg.rho0, dtype=np.float64)
    kernel = WendlandC2(cfg.h)

    start = time.perf_counter()
    pair_i, pair_j = build_neighbor_pairs(pos, cfg.support)
    elapsed = time.perf_counter() - start
    counts = neighbor_counts(len(pos), pair_i, pair_j)

    print(f"Unique pairs : {len(pair_i):,}")
    print(f"Build time   : {elapsed:.4f} s")
    print(
        f"Neighbours   : {counts.min()} / "
        f"{counts.mean():.2f} / {counts.max()}"
    )

    _, _, gradW = compute_pair_geometry(pos, pair_i, pair_j, kernel)
    drho = compute_density_rate(vel, pair_i, pair_j, gradW, cfg.mass)
    pressure = compute_pressure(rho, cfg.rho0, cfg.tait_B, cfg.gamma)
    acc = compute_pressure_acceleration(
        rho, pressure, pair_i, pair_j, gradW, cfg.mass
    )

    print(f"drho max     : {np.max(np.abs(drho)):.6e}")
    print(f"pressure max : {pressure.max():.6e} Pa")
    print(f"acc max      : {np.linalg.norm(acc, axis=1).max():.6e} m/s2")
    print(f"CFL dt       : {compute_cfl_dt(vel, cfg.h, cfg.c0):.8f} s")
