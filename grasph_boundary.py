import math
import os
import sys
import time
from dataclasses import dataclass

import numpy as np

_WALL_PAIR_CHUNK = 16384
from pxr import UsdGeom, Gf

from grasph_scene import load_scene, generate_fluid_particles
from grasph_sph import (
    SPHConfig,
    WendlandC2,
    build_neighbor_pairs,
    build_cross_neighbor_pairs, _scatter_add_vec3,
    neighbor_counts, TranslationNeighborSearch,
)


# ============================================================
# GraSPH Boundary Preprocessor v0.7
#
# Collider mesh -> fixed SPH boundary particles.
#
# IMPORTANT:
#   These particles are NOT used for geometric collision.
#   Geometric non-penetration remains the responsibility of
#   the distance-field collider in grasph_collision.py.
#
#   Boundary particles exist only to provide SPH wall support.
#
# v0.7 adds:
#   - fixed volumetric wall layers at half-cell offsets
#   - surface anchors retained for thin-wall facing tests
#
# v0.5 added:
#   - physically motivated half-cell wall volume Vb = dp^2 * dp/2
#   - static-wall WCSPH coupling helpers (pressure + continuity)
#   - front-facing wall pair interaction API for the runtime solver
#
# v0.4 added:
#   - front-facing fluid/boundary pair filtering using surface normals
#   - same-surface boundary quadrature (opposite/thin-wall faces excluded)
#   - calibrated diagnostic wall scale (diagnostic only)
#
# v0.3 added:
#   - boundary quadrature / pseudo-volume diagnostics
#   - compares equal-volume and Akinci-style surface weights
#   - kernel partition diagnostics before physics coupling
#
# v0.2 changes:
#   - area-aware / surface-grid triangle sampling
#   - avoids severe oversampling of long, thin triangles
#   - seam deduplication
#   - deterministic Poisson-like thinning on the same surface
#   - preserves opposite faces of thin solids
# ============================================================


@dataclass
class BoundaryParticles:
    positions: np.ndarray
    normals: np.ndarray
    source_ids: np.ndarray
    source_paths: list
    volumes: np.ndarray | None = None
    pseudo_masses: np.ndarray | None = None
    # Optional anchor positions used only for front/back-facing tests.
    # For ordinary surface particles this is None and positions are used.
    # For volumetric wall layers, each virtual particle keeps the surface
    # point from which it was extruded so a deep layer can never make the
    # opposite side of a thin solid appear front-facing.
    facing_positions: np.ndarray | None = None
    layer_offsets: np.ndarray | None = None


# ============================================================
# GEOMETRY HELPERS
# ============================================================

def _triangle_normal(a, b, c):
    n = np.cross(b - a, c - a)
    length = np.linalg.norm(n)
    if length <= 1e-14:
        return None
    return n / length


def _edge_samples(a, b, spacing):
    length = np.linalg.norm(b - a)
    if length <= 1e-14:
        return np.asarray([a], dtype=np.float64)

    divisions = max(1, int(math.ceil(length / spacing)))
    t = np.linspace(0.0, 1.0, divisions + 1)
    return a[None, :] + t[:, None] * (b - a)[None, :]


# ============================================================
# TRIANGLE SURFACE SAMPLING
#
# The old implementation used the longest triangle edge to
# subdivide BOTH barycentric directions. On a long, thin
# triangle that can oversample the short direction by 5-50x.
#
# Here we project the triangle to a local 2D plane and place a
# staggered near-equilateral grid directly in surface space.
# ============================================================

def _sample_triangle(a, b, c, spacing):
    normal = _triangle_normal(a, b, c)
    if normal is None:
        return np.empty((0, 3)), np.empty((0, 3))

    # Use the longest edge as local x axis. This makes the
    # surface grid stable for skinny triangles.
    vertices = [a, b, c]
    edges = [
        (0, 1, np.linalg.norm(b - a)),
        (1, 2, np.linalg.norm(c - b)),
        (2, 0, np.linalg.norm(a - c)),
    ]
    i0, i1, longest = max(edges, key=lambda e: e[2])
    i2 = 3 - i0 - i1

    p0 = vertices[i0]
    p1 = vertices[i1]
    p2 = vertices[i2]

    if longest <= 1e-14:
        return np.empty((0, 3)), np.empty((0, 3))

    u = (p1 - p0) / longest
    v = np.cross(normal, u)
    vlen = np.linalg.norm(v)
    if vlen <= 1e-14:
        return np.empty((0, 3)), np.empty((0, 3))
    v /= vlen

    # Project p2 to local plane. Flip v so height is positive.
    x2 = float(np.dot(p2 - p0, u))
    y2 = float(np.dot(p2 - p0, v))
    if y2 < 0.0:
        v = -v
        y2 = -y2

    # Degenerate / ultra-thin triangle: one edge row is enough.
    row_spacing = spacing * math.sqrt(3.0) / 2.0
    if y2 < 1e-10:
        pts = _edge_samples(p0, p1, spacing)
        normals = np.repeat(normal[None, :], len(pts), axis=0)
        return pts, normals

    pts_2d = []

    # Horizontal intersections with the 2D triangle.
    tri2 = np.array(
        [[0.0, 0.0], [longest, 0.0], [x2, y2]],
        dtype=np.float64,
    )

    def row_intersections(y):
        xs = []
        for k in range(3):
            q0 = tri2[k]
            q1 = tri2[(k + 1) % 3]
            y0, y1 = q0[1], q1[1]

            if abs(y1 - y0) <= 1e-14:
                if abs(y - y0) <= 1e-10:
                    xs.extend([q0[0], q1[0]])
                continue

            lo = min(y0, y1) - 1e-10
            hi = max(y0, y1) + 1e-10
            if y < lo or y > hi:
                continue

            t = (y - y0) / (y1 - y0)
            if -1e-10 <= t <= 1.0 + 1e-10:
                xs.append(q0[0] + t * (q1[0] - q0[0]))

        if not xs:
            return None

        return min(xs), max(xs)

    # Always include y=0, then staggered rows in the interior,
    # and finally the apex height. This also works for narrow
    # surfaces thinner than dp.
    ys = [0.0]
    if y2 > row_spacing:
        y = row_spacing
        while y < y2 - 1e-10:
            ys.append(y)
            y += row_spacing
    if y2 > 1e-10:
        ys.append(y2)

    for row, y in enumerate(ys):
        limits = row_intersections(y)
        if limits is None:
            continue

        xmin, xmax = limits
        if xmax < xmin:
            xmin, xmax = xmax, xmin

        width = xmax - xmin
        if width <= 1e-10:
            pts_2d.append((0.5 * (xmin + xmax), y))
            continue

        # Stagger alternate rows for a hexagonal-ish packing.
        offset = 0.5 * spacing if (row % 2) else 0.0
        x = xmin + offset

        # If stagger offset would skip a very short row, use center.
        if x > xmax + 1e-10:
            pts_2d.append((0.5 * (xmin + xmax), y))
        else:
            while x <= xmax + 1e-10:
                pts_2d.append((x, y))
                x += spacing

        # Preserve endpoints; seam pass will remove duplicates.
        pts_2d.append((xmin, y))
        pts_2d.append((xmax, y))

    if not pts_2d:
        return np.empty((0, 3)), np.empty((0, 3))

    pts_2d = np.asarray(pts_2d, dtype=np.float64)
    pts = (
        p0[None, :]
        + pts_2d[:, 0, None] * u[None, :]
        + pts_2d[:, 1, None] * v[None, :]
    )

    normals = np.repeat(normal[None, :], len(pts), axis=0)
    return pts, normals


# ============================================================
# SEAM DEDUPLICATION
#
# Merge only points that are nearly identical. This handles
# shared triangle edges and vertices without collapsing
# opposite surfaces of thin solids.
# ============================================================

def _deduplicate_points(points, normals, min_distance):
    if len(points) == 0:
        return points, normals

    cell_size = min_distance
    min_d2 = min_distance * min_distance

    cells = {}
    accepted_points = []
    accepted_normal_sums = []

    def cell_index(p):
        return (
            math.floor(p[0] / cell_size),
            math.floor(p[1] / cell_size),
            math.floor(p[2] / cell_size),
        )

    for p, n in zip(points, normals):
        c = cell_index(p)
        merged_index = None

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    candidates = cells.get((c[0] + dx, c[1] + dy, c[2] + dz))
                    if candidates is None:
                        continue

                    for idx in candidates:
                        d = accepted_points[idx] - p
                        if np.dot(d, d) < min_d2:
                            merged_index = idx
                            break

                    if merged_index is not None:
                        break
                if merged_index is not None:
                    break
            if merged_index is not None:
                break

        if merged_index is None:
            idx = len(accepted_points)
            accepted_points.append(p.copy())
            accepted_normal_sums.append(n.copy())
            cells.setdefault(c, []).append(idx)
        else:
            accepted_normal_sums[merged_index] += n

    out_points = np.asarray(accepted_points, dtype=np.float64)
    out_normals = np.asarray(accepted_normal_sums, dtype=np.float64)

    lengths = np.linalg.norm(out_normals, axis=1)
    good = lengths > 1e-12
    out_normals[good] /= lengths[good, None]
    out_normals[~good] = np.array([0.0, 0.0, 1.0])

    return out_points, out_normals


# ============================================================
# SAME-SURFACE THINNING
#
# Ensures roughly dp-scale particle spacing across adjacent
# triangles. Opposite faces of a thin wall are retained by
# requiring reasonably aligned normals before one point may
# suppress another.
# ============================================================

def _thin_surface_points(
    points,
    normals,
    min_distance,
    normal_dot_min=0.35,
    seed=12345,
):
    if len(points) == 0:
        return points, normals

    cell_size = min_distance
    min_d2 = min_distance * min_distance

    # Deterministic shuffle prevents triangle-order striping.
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(points))

    cells = {}
    accepted_points = []
    accepted_normals = []

    def cell_index(p):
        return (
            math.floor(p[0] / cell_size),
            math.floor(p[1] / cell_size),
            math.floor(p[2] / cell_size),
        )

    for source_index in order:
        p = points[source_index]
        n = normals[source_index]
        c = cell_index(p)
        reject = False

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    candidates = cells.get((c[0] + dx, c[1] + dy, c[2] + dz))
                    if candidates is None:
                        continue

                    for idx in candidates:
                        d = accepted_points[idx] - p
                        if np.dot(d, d) >= min_d2:
                            continue

                        # Only thin points belonging to approximately
                        # the same surface orientation. This prevents
                        # inner/outer faces of a thin solid collapsing.
                        if np.dot(accepted_normals[idx], n) >= normal_dot_min:
                            reject = True
                            break

                    if reject:
                        break
                if reject:
                    break
            if reject:
                break

        if not reject:
            idx = len(accepted_points)
            accepted_points.append(p.copy())
            accepted_normals.append(n.copy())
            cells.setdefault(c, []).append(idx)

    return (
        np.asarray(accepted_points, dtype=np.float64),
        np.asarray(accepted_normals, dtype=np.float64),
    )


# ============================================================
# BUILD BOUNDARY PARTICLES FROM ALL COLLIDER MESHES
# ============================================================

def build_boundary_particles(
    scene,
    spacing,
    seam_distance=None,
    thin_distance=None,
):
    if seam_distance is None:
        seam_distance = 0.20 * spacing

    if thin_distance is None:
        # 0.80*dp gives useful wall support while removing the
        # severe dense bands from skinny-triangle sampling.
        thin_distance = 0.80 * spacing

    all_positions = []
    all_normals = []
    all_source_ids = []
    source_paths = []

    print()
    print("========================================")
    print("GraSPH BOUNDARY BUILD v0.4")
    print("========================================")
    print(f"Boundary spacing : {spacing:.4f} m")
    print(f"Seam distance    : {seam_distance:.4f} m")
    print(f"Thin distance    : {thin_distance:.4f} m")

    for source_id, mesh in enumerate(scene.colliders):
        source_paths.append(mesh.path)

        raw_pos_parts = []
        raw_normal_parts = []

        t0 = time.perf_counter()

        for tri in mesh.triangles:
            p, n = _sample_triangle(tri[0], tri[1], tri[2], spacing)
            if len(p):
                raw_pos_parts.append(p)
                raw_normal_parts.append(n)

        if raw_pos_parts:
            raw_pos = np.vstack(raw_pos_parts)
            raw_normals = np.vstack(raw_normal_parts)

            seam_pos, seam_normals = _deduplicate_points(
                raw_pos,
                raw_normals,
                seam_distance,
            )

            positions, normals = _thin_surface_points(
                seam_pos,
                seam_normals,
                thin_distance,
                seed=12345 + source_id,
            )
        else:
            raw_pos = np.empty((0, 3), dtype=np.float64)
            seam_pos = np.empty((0, 3), dtype=np.float64)
            positions = np.empty((0, 3), dtype=np.float64)
            normals = np.empty((0, 3), dtype=np.float64)

        elapsed = time.perf_counter() - t0

        print()
        print(f"  {mesh.path}")
        print(f"      triangles : {len(mesh.triangles)}")
        print(f"      raw       : {len(raw_pos)}")
        print(f"      seam      : {len(seam_pos)}")
        print(f"      boundary  : {len(positions)}")
        print(f"      time      : {elapsed:.3f} s")

        if len(positions):
            all_positions.append(positions)
            all_normals.append(normals)
            all_source_ids.append(
                np.full(len(positions), source_id, dtype=np.int32)
            )

    if not all_positions:
        return BoundaryParticles(
            positions=np.empty((0, 3), dtype=np.float64),
            normals=np.empty((0, 3), dtype=np.float64),
            source_ids=np.empty(0, dtype=np.int32),
            source_paths=source_paths,
        )

    result = BoundaryParticles(
        positions=np.vstack(all_positions),
        normals=np.vstack(all_normals),
        source_ids=np.concatenate(all_source_ids),
        source_paths=source_paths,
    )

    print()
    print(f"Total boundary particles: {len(result.positions)}")

    return result


# ============================================================
# ORIENTED FLUID/BOUNDARY PAIR FILTER
#
# Boundary normals point away from the solid surface. A fluid
# particle should only receive SPH wall support from the side of
# a face that is geometrically visible to it. This rejects the
# opposite face of thin walls while preserving the face that is
# exposed to the fluid.
# ============================================================

def filter_front_facing_pairs(fluid_pos, boundary, pair_i, pair_j, eps=1e-12):
    if len(pair_i) == 0:
        return pair_i, pair_j, np.empty(0, dtype=np.float64)

    facing_positions = (
        boundary.positions
        if boundary.facing_positions is None
        else boundary.facing_positions
    )

    to_fluid = fluid_pos[pair_i] - facing_positions[pair_j]
    signed = np.einsum(
        "ij,ij->i",
        to_fluid,
        boundary.normals[pair_j],
    )

    keep = signed > eps

    return pair_i[keep], pair_j[keep], signed[keep]


# ============================================================
# SUPPORT DIAGNOSTICS
# ============================================================

def analyze_boundary_support(fluid_pos, boundary, support):
    n = len(fluid_pos)

    fi, fj = build_neighbor_pairs(fluid_pos, support)
    fluid_counts = neighbor_counts(n, fi, fj)

    raw_bi, raw_bj = build_cross_neighbor_pairs(
        fluid_pos,
        boundary.positions,
        support,
    )

    bi, bj, facing_signed = filter_front_facing_pairs(
        fluid_pos,
        boundary,
        raw_bi,
        raw_bj,
    )

    raw_boundary_counts = np.zeros(n, dtype=np.int32)
    if len(raw_bi):
        np.add.at(raw_boundary_counts, raw_bi, 1)

    boundary_counts = np.zeros(n, dtype=np.int32)
    if len(bi):
        np.add.at(boundary_counts, bi, 1)

    combined_counts = fluid_counts + boundary_counts
    near_wall = boundary_counts > 0

    print()
    print("========================================")
    print("BOUNDARY SUPPORT DIAGNOSTICS")
    print("========================================")
    print(f"Fluid-fluid pairs          : {len(fi):,}")
    print(f"Fluid-boundary raw pairs   : {len(raw_bi):,}")
    print(f"Fluid-boundary facing pairs: {len(bi):,}")
    if len(raw_bi):
        print(f"Facing fraction            : {100.0*len(bi)/len(raw_bi):.1f}%")

    print()
    print("Fluid neighbours min/mean/max:")
    print(
        f"  {fluid_counts.min()} / "
        f"{fluid_counts.mean():.2f} / "
        f"{fluid_counts.max()}"
    )

    print("Boundary neighbours RAW min/mean/max:")
    print(
        f"  {raw_boundary_counts.min()} / "
        f"{raw_boundary_counts.mean():.2f} / "
        f"{raw_boundary_counts.max()}"
    )

    print("Boundary neighbours FACING min/mean/max:")
    print(
        f"  {boundary_counts.min()} / "
        f"{boundary_counts.mean():.2f} / "
        f"{boundary_counts.max()}"
    )

    print("Combined FACING raw support min/mean/max:")
    print(
        f"  {combined_counts.min()} / "
        f"{combined_counts.mean():.2f} / "
        f"{combined_counts.max()}"
    )

    print()
    print(f"Fluid particles near a facing wall: {np.sum(near_wall)} / {n}")

    if np.any(near_wall):
        print("Near-wall combined FACING support min/mean/max:")
        c = combined_counts[near_wall]
        print(f"  {c.min()} / {c.mean():.2f} / {c.max()}")

    print()
    print("NOTE: pair orientation uses collider surface normals.")
    print("Pairs through the back side of a thin solid are rejected.")
    print("Raw neighbour count remains geometry diagnostic only.")

    return {
        "fluid_counts": fluid_counts,
        "raw_boundary_counts": raw_boundary_counts,
        "boundary_counts": boundary_counts,
        "combined_counts": combined_counts,
        "fluid_pair_i": fi,
        "fluid_pair_j": fj,
        "raw_boundary_pair_i": raw_bi,
        "raw_boundary_pair_j": raw_bj,
        "boundary_pair_i": bi,
        "boundary_pair_j": bj,
        "boundary_pair_signed": facing_signed,
    }


# ============================================================
# BOUNDARY QUADRATURE / PSEUDO VOLUMES
#
# For an irregular surface sample, assigning identical mass to
# every boundary point can over-weight dense regions and
# under-weight sparse regions.  Akinci-style pseudo volumes use
# the local boundary kernel sum as a quadrature weight:
#
#     V_b = 1 / sum_k W_bk
#
# The sum includes the boundary particle itself and co-oriented
# points on the same surface sheet of the same collider / rigid body.
# Opposite sides of thin solids and perpendicular corner faces are
# excluded from the quadrature sum. The corresponding
# pseudo mass is psi_b = rho0 * V_b.
#
# We do NOT couple these weights to WCSPH yet.  v0.3 only measures
# how they affect kernel support so we can choose the wall model
# from data rather than by trial-and-error.
# ============================================================

def compute_boundary_pseudo_volumes(boundary, kernel, support, rho0):
    n = len(boundary.positions)
    volumes = np.zeros(n, dtype=np.float64)

    w0 = float(kernel.W(np.asarray([0.0], dtype=np.float64))[0])

    print()
    print("========================================")
    print("BOUNDARY QUADRATURE WEIGHTS")
    print("========================================")

    for source_id, path in enumerate(boundary.source_paths):
        ids = np.where(boundary.source_ids == source_id)[0]
        if len(ids) == 0:
            continue

        local_pos = boundary.positions[ids]
        local_normals = boundary.normals[ids]
        raw_pi, raw_pj = build_neighbor_pairs(local_pos, support)

        # Pseudo-volume is a surface quadrature. Neighbours from
        # the opposite side of a thin wall or a perpendicular face
        # must not reduce the local boundary volume. Keep only
        # reasonably co-oriented samples of the same surface sheet.
        if len(raw_pi):
            ndot = np.einsum(
                "ij,ij->i",
                local_normals[raw_pi],
                local_normals[raw_pj],
            )
            aligned = ndot > 0.50
            pi = raw_pi[aligned]
            pj = raw_pj[aligned]
        else:
            pi = raw_pi
            pj = raw_pj

        kernel_sum = np.full(len(ids), w0, dtype=np.float64)

        if len(pi):
            rij = local_pos[pi] - local_pos[pj]
            r = np.linalg.norm(rij, axis=1)
            wij = kernel.W(r)
            np.add.at(kernel_sum, pi, wij)
            np.add.at(kernel_sum, pj, wij)

        local_volumes = 1.0 / np.maximum(kernel_sum, 1e-30)
        volumes[ids] = local_volumes

        print()
        print(f"  {path}")
        print(f"      particles : {len(ids)}")
        print(f"      bb raw/aligned : {len(raw_pi):,} / {len(pi):,}")
        print(
            "      V min/mean/max : "
            f"{local_volumes.min():.6e} / "
            f"{local_volumes.mean():.6e} / "
            f"{local_volumes.max():.6e} m3"
        )

    pseudo_masses = rho0 * volumes
    boundary.volumes = volumes
    boundary.pseudo_masses = pseudo_masses

    print()
    print("All boundary points:")
    print(
        "  volume min/mean/max : "
        f"{volumes.min():.6e} / "
        f"{volumes.mean():.6e} / "
        f"{volumes.max():.6e} m3"
    )
    print(
        "  pseudo mass min/mean/max : "
        f"{pseudo_masses.min():.6e} / "
        f"{pseudo_masses.mean():.6e} / "
        f"{pseudo_masses.max():.6e} kg"
    )

    return volumes, pseudo_masses


# ============================================================
# KERNEL PARTITION DIAGNOSTICS
#
# For a well-sampled 3-D fluid, the discrete kernel partition
#
#     sum_j V_j W_ij
#
# should be close to one.  A free surface is intentionally below
# one; a solid wall should be repaired by the boundary quadrature.
#
# We compare two wall weights:
#   A) equal boundary volume = dp^3
#   B) Akinci-style pseudo volume from local boundary sampling
#
# This is diagnostic only.  No force law is changed here.
# ============================================================

def analyze_weighted_kernel_support(
    fluid_pos,
    boundary,
    cfg,
    support_data=None,
):
    kernel = WendlandC2(cfg.h)
    n = len(fluid_pos)

    if boundary.volumes is None:
        compute_boundary_pseudo_volumes(
            boundary,
            kernel,
            cfg.support,
            cfg.rho0,
        )

    if support_data is None:
        fi, fj = build_neighbor_pairs(fluid_pos, cfg.support)
        bi, bj = build_cross_neighbor_pairs(
            fluid_pos, boundary.positions, cfg.support
        )
    else:
        fi = support_data["fluid_pair_i"]
        fj = support_data["fluid_pair_j"]
        bi = support_data["boundary_pair_i"]
        bj = support_data["boundary_pair_j"]

    # Fluid-fluid discrete volume partition.
    w0 = float(kernel.W(np.asarray([0.0], dtype=np.float64))[0])
    fluid_sum_w = np.full(n, w0, dtype=np.float64)

    if len(fi):
        rij = fluid_pos[fi] - fluid_pos[fj]
        r = np.linalg.norm(rij, axis=1)
        wij = kernel.W(r)
        np.add.at(fluid_sum_w, fi, wij)
        np.add.at(fluid_sum_w, fj, wij)

    Vf = cfg.mass / cfg.rho0
    fluid_partition = Vf * fluid_sum_w

    boundary_equal = np.zeros(n, dtype=np.float64)
    boundary_pseudo = np.zeros(n, dtype=np.float64)

    if len(bi):
        rijb = fluid_pos[bi] - boundary.positions[bj]
        rb = np.linalg.norm(rijb, axis=1)
        wb = kernel.W(rb)

        np.add.at(
            boundary_equal,
            bi,
            Vf * wb,
        )

        np.add.at(
            boundary_pseudo,
            bi,
            boundary.volumes[bj] * wb,
        )

    total_equal = fluid_partition + boundary_equal
    total_pseudo = fluid_partition + boundary_pseudo

    near_wall = boundary_equal > 0.0

    # Exclude the free-surface band for the most meaningful wall
    # completeness metric.  Works for arbitrary gravity direction.
    g = np.asarray(cfg.gravity, dtype=np.float64)
    gmag = np.linalg.norm(g)
    if gmag > 1e-12:
        up = -g / gmag
        height = fluid_pos @ up
        surface_height = height.max() + 0.5 * cfg.dp
        submerged = (surface_height - height) > cfg.support
    else:
        submerged = np.ones(n, dtype=bool)

    wall_submerged = near_wall & submerged

    # Bulk target from fluid particles with essentially full fluid
    # neighbourhood.  This preserves the known discrete Wendland
    # value (~1.01 for our h/dp) instead of forcing an exact 1.0.
    fluid_counts = np.zeros(n, dtype=np.int32)
    if len(fi):
        np.add.at(fluid_counts, fi, 1)
        np.add.at(fluid_counts, fj, 1)

    bulk = fluid_counts >= max(1, int(0.94 * fluid_counts.max()))
    bulk_target = float(fluid_partition[bulk].mean()) if np.any(bulk) else 1.0

    def stats(name, values, mask=None):
        arr = values if mask is None else values[mask]
        if len(arr) == 0:
            print(f"  {name:<24}: n/a")
            return
        print(
            f"  {name:<24}: "
            f"{arr.min():.4f} / {arr.mean():.4f} / {arr.max():.4f}"
        )

    print()
    print("========================================")
    print("WEIGHTED KERNEL SUPPORT DIAGNOSTICS")
    print("========================================")
    print(f"Bulk discrete target       : {bulk_target:.4f}")
    print(f"Wall + submerged particles : {np.sum(wall_submerged)} / {n}")
    print()
    print("Partition min / mean / max:")
    stats("fluid only, all", fluid_partition)
    stats("equal wall, all", total_equal)
    stats("pseudo wall, all", total_pseudo)
    print()
    stats("fluid only, wall wet", fluid_partition, wall_submerged)
    stats("equal wall, wall wet", total_equal, wall_submerged)
    stats("pseudo wall, wall wet", total_pseudo, wall_submerged)

    pseudo_scale = 1.0
    total_pseudo_scaled = total_pseudo.copy()

    if np.any(wall_submerged):
        err_equal = np.mean(np.abs(total_equal[wall_submerged] - bulk_target))
        err_pseudo = np.mean(np.abs(total_pseudo[wall_submerged] - bulk_target))
        print()
        print("Mean absolute error vs bulk target:")
        print(f"  equal volume : {err_equal:.5f}")
        print(f"  pseudo volume: {err_pseudo:.5f}")
        if err_equal < err_pseudo:
            print("  -> equal dp^3 wall volume is closer in this scene")
        else:
            print("  -> pseudo-volume wall quadrature is closer in this scene")

        # Diagnostic-only global scale. This is NOT applied to the
        # saved boundary volumes. It tells us whether the oriented
        # pseudo-volume quadrature has the correct overall magnitude.
        m = wall_submerged & (boundary_pseudo > 1e-14)
        if np.any(m):
            deficit = bulk_target - fluid_partition[m]
            b = boundary_pseudo[m]
            denom = float(np.dot(b, b))
            if denom > 1e-30:
                pseudo_scale = max(0.0, float(np.dot(b, deficit) / denom))
                total_pseudo_scaled = fluid_partition + pseudo_scale * boundary_pseudo
                err_scaled = np.mean(
                    np.abs(total_pseudo_scaled[wall_submerged] - bulk_target)
                )
                print()
                print("Diagnostic global pseudo-volume scale:")
                print(f"  least-squares scale : {pseudo_scale:.4f}")
                print(f"  scaled mean abs err : {err_scaled:.5f}")
                stats(
                    "scaled pseudo, wall wet",
                    total_pseudo_scaled,
                    wall_submerged,
                )

    return {
        "fluid_partition": fluid_partition,
        "boundary_equal": boundary_equal,
        "boundary_pseudo": boundary_pseudo,
        "total_equal": total_equal,
        "total_pseudo": total_pseudo,
        "total_pseudo_scaled": total_pseudo_scaled,
        "pseudo_scale": pseudo_scale,
        "wall_submerged": wall_submerged,
        "bulk_target": bulk_target,
    }


# ============================================================
# RUNTIME WALL QUADRATURE v0.5
#
# A single surface sheet is not a 3-D particle lattice. Giving
# each wall sample a full dp^3 volume over-weights the wall.
# The v0.4 diagnostics showed an optimum mean volume close to
# 0.5*dp^3. That has a direct geometric interpretation:
#
#     V_b = (surface patch dp^2) * (half-cell thickness dp/2)
#         = 0.5 * dp^3
#
# This is now our default wall quadrature for fixed boundaries.
# ============================================================
# VOLUMETRIC FIXED WALL LAYERS
#
# Surface samples alone are a zero-thickness quadrature.  They can restore
# the zeroth kernel moment reasonably well but are poor at reproducing the
# first moment needed for a hydrostatic pressure gradient.  For validation
# we therefore extend each fluid-facing surface sample into the solid with
# fixed virtual particles at half-cell offsets:
#
#       0.5 dp, 1.5 dp, 2.5 dp, ... < support
#
# Every layer particle represents one full dp^3 cell.  These particles are
# STATIC SPH support only; geometric non-penetration remains the DF collider.
# The original surface anchor is retained for front-facing filtering so
# opposite faces of thin solids cannot leak through merely because an
# extruded layer crosses the physical wall thickness.
# ============================================================

def build_layered_wall_particles(
    surface_boundary,
    dp,
    support,
    rho0,
):
    dp = float(dp)
    support = float(support)

    offsets = []
    k = 0
    while True:
        off = (k + 0.5) * dp
        if off >= support - 1e-12:
            break
        offsets.append(off)
        k += 1

    offsets = np.asarray(offsets, dtype=np.float64)

    if len(surface_boundary.positions) == 0 or len(offsets) == 0:
        return BoundaryParticles(
            positions=np.empty((0, 3), dtype=np.float64),
            normals=np.empty((0, 3), dtype=np.float64),
            source_ids=np.empty(0, dtype=np.int32),
            source_paths=surface_boundary.source_paths,
            volumes=np.empty(0, dtype=np.float64),
            pseudo_masses=np.empty(0, dtype=np.float64),
            facing_positions=np.empty((0, 3), dtype=np.float64),
            layer_offsets=np.empty(0, dtype=np.float64),
        )

    nl = len(offsets)

    anchors = np.repeat(surface_boundary.positions, nl, axis=0)
    normals = np.repeat(surface_boundary.normals, nl, axis=0)
    source_ids = np.repeat(surface_boundary.source_ids, nl)
    layer_offsets = np.tile(offsets, len(surface_boundary.positions))

    positions = anchors - layer_offsets[:, None] * normals

    volume = dp ** 3
    volumes = np.full(len(positions), volume, dtype=np.float64)
    pseudo_masses = rho0 * volumes

    result = BoundaryParticles(
        positions=positions,
        normals=normals,
        source_ids=source_ids,
        source_paths=surface_boundary.source_paths,
        volumes=volumes,
        pseudo_masses=pseudo_masses,
        facing_positions=anchors,
        layer_offsets=layer_offsets,
    )

    print()
    print("========================================")
    print("GraSPH VOLUMETRIC WALL LAYERS v0.7")
    print("========================================")
    print(f"Surface samples : {len(surface_boundary.positions)}")
    print(f"Layers          : {nl}")
    print(
        "Offsets         : "
        + ", ".join(f"{x:.4f}" for x in offsets)
        + " m"
    )
    print(f"Wall particles  : {len(result.positions)}")
    print(f"Cell volume     : {volume:.6e} m3")
    print(f"Mass @ rho0     : {rho0 * volume:.6e} kg")

    return result


# ============================================================

def assign_half_cell_wall_volumes(
    boundary,
    dp,
    rho0,
    thickness_factor=0.5,
):
    volume = float(thickness_factor) * float(dp) ** 3

    boundary.volumes = np.full(
        len(boundary.positions),
        volume,
        dtype=np.float64,
    )

    boundary.pseudo_masses = (
        rho0 * boundary.volumes
    )

    return boundary.volumes, boundary.pseudo_masses


# ============================================================
# BUILD ACTIVE FLUID/WALL PAIRS
# ============================================================

class CachedWallNeighborSearch:
    """Skin-buffered wall neighbor list with exact per-step radius filtering.

    Candidates are rebuilt when the sum of maximum fluid and wall displacement
    reaches the skin width. Translation and rotation are both supported; actual
    distances, normals and forces are evaluated at the current pose every step.
    """

    def __init__(self, boundary, skin=0.025):
        self.skin = max(0.0, float(skin))
        self.reference_fluid = None
        self.candidate_i = np.empty(0, dtype=np.int32)
        self.candidate_j = np.empty(0, dtype=np.int32)
        self.reference_boundary = None
        self.reference_support = None

    def query(self, fluid_pos, boundary, support, translations):
        fluid_pos = np.asarray(fluid_pos, dtype=np.float64)
        rebuild = (
            self.reference_fluid is None
            or self.reference_fluid.shape != fluid_pos.shape
            or self.reference_boundary.shape != boundary.positions.shape
            or self.reference_support != float(support)
        )
        if not rebuild and len(fluid_pos):
            fluid_move = float(np.linalg.norm(fluid_pos - self.reference_fluid, axis=1).max())
            wall_move = float(np.linalg.norm(boundary.positions - self.reference_boundary, axis=1).max()) if len(boundary.positions) else 0.0
            # Triangle inequality: even rotating walls cannot enter support
            # from outside the candidate radius before this bound is reached.
            rebuild = fluid_move + wall_move >= self.skin
        if rebuild:
            radius = float(support) + self.skin
            self.candidate_i, self.candidate_j = build_cross_neighbor_pairs(
                fluid_pos, boundary.positions, radius
            )
            self.reference_fluid = fluid_pos.copy()
            self.reference_boundary = boundary.positions.copy()
            self.reference_support = float(support)
        if not len(self.candidate_i):
            return self.candidate_i, self.candidate_j
        # Candidate lists include the Verlet skin and may be much larger than
        # the exact interaction list. Filter in bounded blocks to avoid several
        # simultaneous Mx3 temporaries (fluid positions, wall positions, rij).
        support2 = float(support) ** 2
        kept_i = []
        kept_j = []
        for start in range(0, len(self.candidate_i), _WALL_PAIR_CHUNK):
            end = min(start + _WALL_PAIR_CHUNK, len(self.candidate_i))
            candidate_i = self.candidate_i[start:end]
            candidate_j = self.candidate_j[start:end]
            rij = fluid_pos[candidate_i] - boundary.positions[candidate_j]
            keep = np.einsum("ij,ij->i", rij, rij) < support2
            if np.any(keep):
                kept_i.append(candidate_i[keep])
                kept_j.append(candidate_j[keep])
        if not kept_i:
            return np.empty(0, dtype=self.candidate_i.dtype), np.empty(
                0, dtype=self.candidate_j.dtype
            )
        return np.concatenate(kept_i), np.concatenate(kept_j)


def build_wall_interactions(
    fluid_pos,
    boundary,
    kernel,
    support,
    neighbor_search=None,
    translations=None,
):
    if neighbor_search is None:
        raw_i, raw_j = build_cross_neighbor_pairs(fluid_pos, boundary.positions, support)
    else:
        raw_i, raw_j = neighbor_search.query(fluid_pos, boundary, support, translations or {})

    pair_i, pair_j, signed = filter_front_facing_pairs(
        fluid_pos,
        boundary,
        raw_i,
        raw_j,
    )

    if len(pair_i) == 0:
        return {
            "pair_i": pair_i,
            "pair_j": pair_j,
            "rij": np.empty((0, 3), dtype=np.float64),
            "r": np.empty(0, dtype=np.float64),
            "W": np.empty(0, dtype=np.float64),
            "gradW": np.empty((0, 3), dtype=np.float64),
            "signed": signed,
            "raw_count": len(raw_i),
        }

    rij = (
        fluid_pos[pair_i]
        - boundary.positions[pair_j]
    )
    r = np.linalg.norm(rij, axis=1)
    W = kernel.W(r)
    gradW = kernel.gradW(rij, r)

    return {
        "pair_i": pair_i,
        "pair_j": pair_j,
        "rij": rij,
        "r": r,
        "W": W,
        "gradW": gradW,
        "signed": signed,
        "raw_count": len(raw_i),
    }


# ============================================================
# STATIC WALL STATE
#
# Adami-like pressure extrapolation:
#
#   p_b ~= weighted average of
#          p_i + rho_i * g dot (x_b - x_i)
#
# This exactly has the right sign for hydrostatics: a boundary
# point below a fluid particle receives a larger pressure.
# Boundary density follows from the same Tait EOS as the fluid.
# ============================================================

def compute_static_wall_state(
    fluid_pos,
    fluid_rho,
    fluid_pressure,
    boundary,
    interactions,
    gravity,
    rho0,
    B,
    gamma,
):
    nb = len(boundary.positions)

    wall_pressure = np.zeros(nb, dtype=np.float64)
    wall_rho = np.full(nb, rho0, dtype=np.float64)

    pair_i = interactions["pair_i"]
    pair_j = interactions["pair_j"]

    if len(pair_i) == 0:
        return wall_pressure, wall_rho

    W = interactions["W"]

    xb_minus_xi = (
        boundary.positions[pair_j]
        - fluid_pos[pair_i]
    )

    g = np.asarray(gravity, dtype=np.float64)
    hydro = (
        fluid_rho[pair_i]
        * np.einsum("j,ij->i", g, xb_minus_xi)
    )

    candidate_pressure = (
        fluid_pressure[pair_i]
        + hydro
    )

    numerator = np.zeros(nb, dtype=np.float64)
    denominator = np.zeros(nb, dtype=np.float64)

    np.add.at(
        numerator,
        pair_j,
        candidate_pressure * W,
    )
    np.add.at(
        denominator,
        pair_j,
        W,
    )

    active = denominator > 1e-30
    wall_pressure[active] = (
        numerator[active]
        / denominator[active]
    )

    wall_pressure = np.maximum(
        wall_pressure,
        0.0,
    )

    wall_rho = (
        rho0
        * (1.0 + wall_pressure / B) ** (1.0 / gamma)
    )

    return wall_pressure, wall_rho




# ============================================================
# REFERENCE HYDROSTATIC WALL STATE
#
# Validation helper only.  This does not try to infer the wall
# pressure from the moving fluid.  Instead it applies the exact
# hydrostatic pressure associated with a prescribed free-surface
# plane.  It lets us validate wall quadrature + pressure-gradient
# coupling independently of the dynamic wall-state model.
# ============================================================

def compute_reference_hydrostatic_wall_state(
    boundary,
    surface_level,
    gravity,
    rho0,
    B,
    gamma,
):
    g = np.asarray(gravity, dtype=np.float64)
    gmag = float(np.linalg.norm(g))

    if gmag <= 1e-15:
        pressure = np.zeros(len(boundary.positions), dtype=np.float64)
        rho = np.full(len(boundary.positions), rho0, dtype=np.float64)
        return pressure, rho

    up = -g / gmag

    level = boundary.positions @ up
    depth = np.maximum(float(surface_level) - level, 0.0)

    pressure = rho0 * gmag * depth

    rho = (
        rho0
        * (1.0 + pressure / B) ** (1.0 / gamma)
    )

    return pressure, rho


# ============================================================
# WALL CONTRIBUTION TO CONTINUITY
#
# Static wall velocity is zero in v0.5.  Moving/kinematic wall
# velocity will later enter here without changing the WCSPH core.
# ============================================================

def add_static_wall_density_rate(
    drho,
    fluid_vel,
    boundary,
    interactions,
    wall_rho,
    wall_velocity=None,
):
    """Add the wall contribution to the continuity equation.

    Historically this helper handled a static wall (v_b=0), hence the
    retained function name.  v0.18 extends it to a translating/kinematic
    wall by accepting either one 3-vector or one velocity per boundary
    particle.  The matched relative velocity is

        v_ij = v_i - v_b

    which is the continuity counterpart of the pressure force on the same
    wall quadrature points.
    """
    pair_i = interactions["pair_i"]
    pair_j = interactions["pair_j"]

    if len(pair_i) == 0:
        return

    gradW = interactions["gradW"]

    wall_mass = (
        wall_rho[pair_j]
        * boundary.volumes[pair_j]
    )

    if wall_velocity is None:
        vij = fluid_vel[pair_i]
    else:
        wall_velocity = np.asarray(wall_velocity, dtype=np.float64)
        if wall_velocity.ndim == 1:
            if wall_velocity.shape != (3,):
                raise ValueError("wall_velocity must be shape (3,) or (Nb,3)")
            vwall = np.repeat(wall_velocity[None, :], len(pair_i), axis=0)
        else:
            if wall_velocity.shape != boundary.positions.shape:
                raise ValueError(
                    "Per-particle wall_velocity must match boundary.positions shape"
                )
            vwall = wall_velocity[pair_j]
        vij = fluid_vel[pair_i] - vwall

    contribution = (
        wall_mass
        * np.einsum(
            "ij,ij->i",
            vij,
            gradW,
        )
    )

    np.add.at(
        drho,
        pair_i,
        contribution,
    )


# ============================================================
# WALL CONTRIBUTION TO PRESSURE ACCELERATION
# ============================================================

def add_static_wall_pressure_acceleration(
    acc,
    fluid_rho,
    fluid_pressure,
    boundary,
    interactions,
    wall_rho,
    wall_pressure,
):
    pair_i = interactions["pair_i"]
    pair_j = interactions["pair_j"]

    if len(pair_i) == 0:
        return pair_j, np.empty((0, 3), dtype=np.float64)

    gradW = interactions["gradW"]

    rhoi = fluid_rho[pair_i]
    pi = fluid_pressure[pair_i]
    rhob = wall_rho[pair_j]
    pb = wall_pressure[pair_j]

    wall_mass = (
        rhob
        * boundary.volumes[pair_j]
    )

    factor = (
        wall_mass
        * (
            pi / (rhoi * rhoi)
            + pb / (rhob * rhob)
        )
    )

    pair_acceleration = -factor[:, None] * gradW
    acc += _scatter_add_vec3(len(acc), pair_i, pair_acceleration)
    return pair_j, pair_acceleration

# ============================================================
# USD PREVIEW
# ============================================================

def _to_usd_points(points):
    return [
        Gf.Vec3f(float(p[0]), float(p[1]), float(p[2]))
        for p in points
    ]


def write_boundary_preview(
    stage,
    fluid_pos,
    boundary,
    dp,
    output_file,
):
    print()
    print("Writing OpenUSD boundary preview...")

    UsdGeom.Xform.Define(stage, "/GraSPH_BoundaryPreview")

    fluid_prim = UsdGeom.Points.Define(
        stage,
        "/GraSPH_BoundaryPreview/FluidParticles",
    )
    fluid_prim.CreatePointsAttr(_to_usd_points(fluid_pos))
    fluid_prim.CreateWidthsAttr([0.65 * dp] * len(fluid_pos))

    for source_id, path in enumerate(boundary.source_paths):
        ids = np.where(boundary.source_ids == source_id)[0]
        if len(ids) == 0:
            continue

        prim = UsdGeom.Points.Define(
            stage,
            f"/GraSPH_BoundaryPreview/Boundary_{source_id:03d}",
        )
        prim.CreatePointsAttr(_to_usd_points(boundary.positions[ids]))
        prim.CreateWidthsAttr([0.40 * dp] * len(ids))
        prim.GetPrim().SetCustomDataByKey("sourcePath", path)

    stage.GetRootLayer().Export(output_file)
    print(f"Saved: {output_file}")


# ============================================================
# SAVE NUMERICAL BOUNDARY DATA
# ============================================================

def save_boundary_npz(boundary, output_file):
    payload = {
        "positions": boundary.positions.astype(np.float32),
        "normals": boundary.normals.astype(np.float32),
        "source_ids": boundary.source_ids,
    }
    if boundary.volumes is not None:
        payload["volumes"] = boundary.volumes.astype(np.float32)
    if boundary.pseudo_masses is not None:
        payload["pseudo_masses"] = boundary.pseudo_masses.astype(np.float32)
    np.savez_compressed(output_file, **payload)
    print(f"Saved: {output_file}")


# ============================================================
# STANDALONE PREVIEW / SMOKE TEST
# ============================================================

def main():
    usd_file = sys.argv[1] if len(sys.argv) > 1 else "TankTest.usda"

    cfg = SPHConfig()

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    preview_file = os.path.join(output_dir, "boundary_preview.usdc")
    npz_file = os.path.join(output_dir, "boundary_particles.npz")

    print()
    print("========================================")
    print("GraSPH BOUNDARY PREVIEW v0.5")
    print("========================================")
    print(f"Scene   : {usd_file}")
    print(f"dp      : {cfg.dp:.4f} m")
    print(f"support : {cfg.support:.4f} m")

    scene = load_scene(usd_file)
    fluid_pos = generate_fluid_particles(scene, dp=cfg.dp)

    boundary = build_boundary_particles(
        scene,
        spacing=cfg.dp,
    )

    support_data = analyze_boundary_support(
        fluid_pos,
        boundary,
        cfg.support,
    )

    kernel = WendlandC2(cfg.h)
    compute_boundary_pseudo_volumes(
        boundary,
        kernel,
        cfg.support,
        cfg.rho0,
    )

    analyze_weighted_kernel_support(
        fluid_pos,
        boundary,
        cfg,
        support_data=support_data,
    )

    # Runtime choice from v0.4 result: one surface layer gets
    # half a fluid-cell thickness, Vb = 0.5*dp^3.
    assign_half_cell_wall_volumes(
        boundary,
        cfg.dp,
        cfg.rho0,
        thickness_factor=0.5,
    )

    print()
    print("========================================")
    print("RUNTIME WALL VOLUME v0.5")
    print("========================================")
    print(f"Vb = 0.5 * dp^3 = {boundary.volumes[0]:.6e} m3")
    print(f"mb(rho0)       = {boundary.pseudo_masses[0]:.6e} kg")

    save_boundary_npz(boundary, npz_file)

    write_boundary_preview(
        scene.stage,
        fluid_pos,
        boundary,
        cfg.dp,
        preview_file,
    )

    print()
    print("========================================")
    print("BOUNDARY PREVIEW READY")
    print("========================================")
    print("Open this file in Blender:")
    print(f"  {preview_file}")
    print()
    print("Inspect especially:")
    print("  - uniform spacing on long/thin faces")
    print("  - bottom and side-wall coverage")
    print("  - Gate coverage")
    print("  - remaining dense bands around edges")


if __name__ == "__main__":
    main()
