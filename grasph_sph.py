import math
import time
from dataclasses import dataclass

import numpy as np

# Optional CPU accelerator.  GraSPH remains fully functional without SciPy.
try:
    from scipy.spatial import cKDTree as _cKDTree
    from scipy import sparse as _scipy_sparse
    from scipy.sparse.csgraph import connected_components as _scipy_connected_components
except Exception:
    _cKDTree = None
    _scipy_sparse = None
    _scipy_connected_components = None

_NEIGHBOR_BACKEND = "auto"
_NEIGHBOR_WORKERS = -1

def configure_neighbor_search(backend="auto", workers=-1):
    """Configure neighbour search globally for the NumPy solver.

    backend:
      auto    -> scipy cKDTree when available, otherwise legacy grid
      ckdtree -> require scipy.spatial.cKDTree
      grid    -> original pure NumPy/Python cell grid

    workers is used by cKDTree cross-set queries; -1 means all logical CPUs.
    """
    global _NEIGHBOR_BACKEND, _NEIGHBOR_WORKERS
    backend = str(backend).lower()
    if backend not in {"auto", "ckdtree", "grid"}:
        raise ValueError(f"Unknown neighbour backend: {backend}")
    if backend == "ckdtree" and _cKDTree is None:
        raise RuntimeError("--neighbor-backend ckdtree requires SciPy (py -m pip install scipy)")
    _NEIGHBOR_BACKEND = backend
    _NEIGHBOR_WORKERS = int(workers)

def active_neighbor_backend():
    if _NEIGHBOR_BACKEND == "auto":
        return "ckdtree" if _cKDTree is not None else "grid"
    return _NEIGHBOR_BACKEND

def neighbor_worker_count():
    return _NEIGHBOR_WORKERS

def _scatter_add_vec3(n, i, values_i, j=None, values_j=None):
    """Fast vector scatter-add using three np.bincount reductions.

    This replaces np.add.at on (N,3) arrays, which is a major CPU hotspot.
    Reduction order can differ by a few floating-point ulps from add.at, but
    the mathematical operator is unchanged.
    """
    out = np.zeros((int(n), 3), dtype=np.float64)
    if len(i):
        for k in range(3):
            out[:, k] += np.bincount(
                np.asarray(i, dtype=np.int64),
                weights=np.asarray(values_i[:, k], dtype=np.float64),
                minlength=int(n),
            )
    if j is not None and len(j):
        for k in range(3):
            out[:, k] += np.bincount(
                np.asarray(j, dtype=np.int64),
                weights=np.asarray(values_j[:, k], dtype=np.float64),
                minlength=int(n),
            )
    return out


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


def _build_neighbor_pairs_grid(pos, support):
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


def _build_cross_neighbor_pairs_grid(fluid_pos, other_pos, support):
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


def _build_neighbor_pairs_ckdtree(pos, support):
    n = len(pos)
    if n < 2:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    tree = _cKDTree(np.asarray(pos, dtype=np.float64))
    pairs = tree.query_pairs(float(support), output_type="ndarray")
    if pairs.size == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    pairs = np.asarray(pairs, dtype=np.int32)
    # Preserve the legacy strict r < support rule (cKDTree uses <=).
    rij = pos[pairs[:, 0]] - pos[pairs[:, 1]]
    keep = np.einsum("ij,ij->i", rij, rij) < float(support) ** 2
    pairs = pairs[keep]
    return pairs[:, 0].copy(), pairs[:, 1].copy()


def _build_cross_neighbor_pairs_ckdtree(fluid_pos, other_pos, support):
    """Adaptive cross-set radius query using scipy cKDTree.

    FIX6 deliberately does *not* chase Task Manager CPU percentage.  For small
    and medium fluid sets, spawning many worker threads for every timestep costs
    more than it saves.  In that regime scipy's sparse_distance_matrix path is
    materially faster and avoids Python lists.  For larger fluid sets, the
    parallel query_ball_point path is retained.  Both paths apply the same final
    strict r < support filter and therefore return the same geometric pairs.
    """
    fluid_pos = np.asarray(fluid_pos, dtype=np.float64)
    other_pos = np.asarray(other_pos, dtype=np.float64)
    nf = len(fluid_pos)
    if nf == 0 or len(other_pos) == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    radius = float(support)

    # Empirically, the sparse path wins clearly for the current preview/normal
    # scenes (hundreds to low-thousands of fluid particles).  Above this point
    # the multi-threaded radius query starts to pay for itself.
    if nf <= 2500:
        fluid_tree = _cKDTree(fluid_pos)
        other_tree = _cKDTree(other_pos)
        coo = fluid_tree.sparse_distance_matrix(
            other_tree, radius, output_type="coo_matrix"
        )
        if coo.nnz == 0:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
        keep = np.asarray(coo.data) < radius
        return (
            np.asarray(coo.row[keep], dtype=np.int32),
            np.asarray(coo.col[keep], dtype=np.int32),
        )

    tree = _cKDTree(other_pos)
    lists = tree.query_ball_point(
        fluid_pos, radius, workers=_NEIGHBOR_WORKERS, return_sorted=False
    )
    lengths = np.fromiter((len(x) for x in lists), dtype=np.int64, count=nf)
    total = int(lengths.sum())
    if total == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    pair_i = np.repeat(np.arange(nf, dtype=np.int32), lengths)
    pair_j = np.concatenate(lists).astype(np.int32, copy=False)
    rij = fluid_pos[pair_i] - other_pos[pair_j]
    keep = np.einsum("ij,ij->i", rij, rij) < radius * radius
    return pair_i[keep], pair_j[keep]


def build_neighbor_pairs(pos, support):
    if active_neighbor_backend() == "ckdtree":
        return _build_neighbor_pairs_ckdtree(pos, support)
    return _build_neighbor_pairs_grid(pos, support)


def build_cross_neighbor_pairs(fluid_pos, other_pos, support):
    if active_neighbor_backend() == "ckdtree":
        return _build_cross_neighbor_pairs_ckdtree(fluid_pos, other_pos, support)
    return _build_cross_neighbor_pairs_grid(fluid_pos, other_pos, support)


class TranslationNeighborSearch:
    """Persistent radius-search trees for independently translated point sets.

    Reference coordinates are owned by this cache, never the mutable wall array.
    Callers must supply the translation of each group and current world points.
    Rotation/deformation require a new cache. The grid backend stays unchanged.
    """

    def __init__(self, reference_positions, group_ids):
        self.reference = np.asarray(reference_positions, dtype=np.float64).copy()
        self.groups = []
        for group in np.unique(group_ids):
            ids = np.flatnonzero(np.asarray(group_ids) == group)
            points = self.reference[ids].copy()
            tree = _cKDTree(points) if _cKDTree is not None else None
            self.groups.append((int(group), ids, tree))

    def query(self, fluid_pos, world_positions, support, translations):
        if active_neighbor_backend() != "ckdtree":
            return build_cross_neighbor_pairs(fluid_pos, world_positions, support)
        fluid_pos = np.asarray(fluid_pos, dtype=np.float64)
        if len(fluid_pos) == 0 or not self.groups:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
        fi_parts, fj_parts = [], []
        radius = float(support)
        scale = max(1.0, float(np.abs(fluid_pos).max()),
                    float(np.abs(self.reference).max()))
        for group, ids, tree in self.groups:
            shift = np.asarray(translations.get(group, (0.0, 0.0, 0.0)), dtype=np.float64)
            local_fluid = fluid_pos - shift
            # Coordinate subtraction and world addition round differently.
            # Expand candidates conservatively; resolve the border in world geometry.
            padding = 32.0 * np.finfo(np.float64).eps * max(scale, float(np.abs(shift).max()))
            candidate_radius = np.nextafter(radius + padding, np.inf)
            if len(fluid_pos) <= 2500:
                coo = _cKDTree(local_fluid).sparse_distance_matrix(
                    tree, candidate_radius, output_type="coo_matrix"
                )
                ii = np.asarray(coo.row, dtype=np.int32)
                jj = ids[coo.col]
                # Distances safely inside support need no second bulk norm.
                # Only the rounding-sized border needs a world-coordinate test.
                keep = np.asarray(coo.data) < radius - padding
                border = ~keep
                if np.any(border):
                    rij = fluid_pos[ii[border]] - world_positions[jj[border]]
                    keep[border] = np.linalg.norm(rij, axis=1) < radius
            else:
                lists = tree.query_ball_point(local_fluid, candidate_radius,
                                             workers=_NEIGHBOR_WORKERS, return_sorted=False)
                lengths = np.fromiter((len(x) for x in lists), dtype=np.int64, count=len(fluid_pos))
                if not lengths.sum():
                    continue
                ii = np.repeat(np.arange(len(fluid_pos), dtype=np.int32), lengths)
                jj = ids[np.concatenate(lists).astype(np.int64)]
            if len(fluid_pos) > 2500:
                rij = fluid_pos[ii] - world_positions[jj]
                keep = np.einsum("ij,ij->i", rij, rij) < radius * radius
            if np.any(keep):
                fi_parts.append(ii[keep])
                fj_parts.append(jj[keep].astype(np.int32))
        if not fi_parts:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
        return np.concatenate(fi_parts), np.concatenate(fj_parts)


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
    if len(pair_i) == 0:
        return np.zeros(n_particles, dtype=np.int32)
    return (
        np.bincount(pair_i, minlength=n_particles)
        + np.bincount(pair_j, minlength=n_particles)
    ).astype(np.int32, copy=False)


def fluid_connected_components(n_particles, pair_i, pair_j):
    """Return (component_count, labels, sizes) for the fluid neighbour graph.

    The SciPy path is compiled and fast.  A small pure-Python union-find is kept
    as a functional fallback so the NumPy-only solver still runs.
    """
    n = int(n_particles)
    if n <= 0:
        return 0, np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    if len(pair_i) == 0:
        labels = np.arange(n, dtype=np.int32)
        return n, labels, np.ones(n, dtype=np.int32)

    ii = np.asarray(pair_i, dtype=np.int32)
    jj = np.asarray(pair_j, dtype=np.int32)

    if _scipy_sparse is not None and _scipy_connected_components is not None:
        rows = np.concatenate((ii, jj))
        cols = np.concatenate((jj, ii))
        data = np.ones(len(rows), dtype=np.uint8)
        graph = _scipy_sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
        ncomp, labels = _scipy_connected_components(
            graph, directed=False, return_labels=True
        )
        labels = np.asarray(labels, dtype=np.int32)
        sizes = np.bincount(labels, minlength=int(ncomp)).astype(np.int32)
        return int(ncomp), labels, sizes

    parent = np.arange(n, dtype=np.int32)
    rank = np.zeros(n, dtype=np.uint8)

    def find(a):
        a = int(a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = int(parent[a])
        return a

    for a, b in zip(ii, jj):
        ra = find(a); rb = find(b)
        if ra == rb:
            continue
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    roots = np.fromiter((find(i) for i in range(n)), dtype=np.int32, count=n)
    unique, labels = np.unique(roots, return_inverse=True)
    labels = labels.astype(np.int32, copy=False)
    sizes = np.bincount(labels, minlength=len(unique)).astype(np.int32)
    return len(unique), labels, sizes


def detached_component_mask(n_particles, pair_i, pair_j):
    """Classify fluid clusters disconnected from the dominant liquid body.

    The well-balanced hydrostatic reference is only valid for the connected
    bulk liquid.  Any disconnected spray/blob uses ordinary WCSPH + full
    gravity until it reconnects.  If there is no dominant component at all
    (largest <= 1 particle), every particle is detached.
    """
    ncomp, labels, sizes = fluid_connected_components(
        n_particles, pair_i, pair_j
    )
    if ncomp <= 1:
        return np.zeros(int(n_particles), dtype=bool), ncomp, sizes
    largest = int(np.argmax(sizes))
    if int(sizes[largest]) <= 1:
        return np.ones(int(n_particles), dtype=bool), ncomp, sizes
    detached = labels != largest
    return detached, ncomp, sizes


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

    return _scatter_add_vec3(
        n, pair_i, -pair_force, pair_j, pair_force
    )


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
    return _scatter_add_vec3(
        n, pair_i[ids], -pair_acc, pair_j[ids], pair_acc
    )


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
