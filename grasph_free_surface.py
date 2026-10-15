from dataclasses import dataclass
import numpy as np
from grasph_collision import closest_point_triangle_batch

from grasph_reference import hydrostatic_reference_state
from grasph_sph import _scatter_add_vec3


@dataclass
class FreeSurfaceConfig:
    mode: str = "anchored_band"
    hG_threshold: float = 0.35
    collider_eigen_rel: float = 0.10
    probe_offset_factor: float = 0.50
    normal_compat_dot: float = 0.50
    eps: float = 1.0e-12


def compute_fluid_support_G(n, rho_ref, mass, pair_i, pair_j, gradW):
    G = np.zeros((n, 3), dtype=np.float64)
    if len(pair_i):
        Vi = mass / np.maximum(rho_ref[pair_i], 1.0e-30)
        Vj = mass / np.maximum(rho_ref[pair_j], 1.0e-30)
        G = _scatter_add_vec3(
            n, pair_i, Vj[:, None] * gradW,
            pair_j, -Vi[:, None] * gradW,
        )
    return G


def collider_triangle_data(scene):
    normals, triangles, paths = [], [], []
    for mesh in scene.colliders:
        for tri in np.asarray(mesh.triangles, dtype=np.float64):
            nn = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            ln = float(np.linalg.norm(nn))
            if ln <= 1.0e-15:
                continue
            normals.append(nn / ln)
            triangles.append(tri)
            paths.append(mesh.path)
    if not triangles:
        return np.empty((0, 3, 3)), np.empty((0, 3)), []
    return np.asarray(triangles), np.asarray(normals), paths


def _point_in_triangle_batch(points, tri, tol=1.0e-10):
    p = np.asarray(points, dtype=np.float64)
    a, b, c = np.asarray(tri, dtype=np.float64)
    v0, v1, v2 = b - a, c - a, p - a[None, :]
    d00, d01, d11 = float(np.dot(v0, v0)), float(np.dot(v0, v1)), float(np.dot(v1, v1))
    denom = d00 * d11 - d01 * d01
    if abs(denom) <= 1.0e-30:
        return np.zeros(len(p), dtype=bool)
    d20, d21 = v2 @ v0, v2 @ v1
    inv = 1.0 / denom
    v = (d11 * d20 - d01 * d21) * inv
    w = (d00 * d21 - d01 * d20) * inv
    u = 1.0 - v - w
    return (u >= -tol) & (v >= -tol) & (w >= -tol)


def compute_geometry_normal_covariance(pos, triangles, normals, support):
    """Collider normal span near each fluid particle.

    Expanded triangle AABBs cull distant particles. Only outward-facing finite
    faces within support contribute, including nearest edge/corner features.
    """
    n = len(pos)
    cov = np.zeros((n, 3, 3), dtype=np.float64)
    tri_count = np.zeros(n, dtype=np.int16)
    min_distance = np.full(n, np.inf, dtype=np.float64)
    pad = float(support) + 1.0e-12
    for tri, normal in zip(triangles, normals):
        bmin = np.min(tri, axis=0) - pad
        bmax = np.max(tri, axis=0) + pad
        box = np.all((pos >= bmin[None, :]) & (pos <= bmax[None, :]), axis=1)
        if not np.any(box):
            continue
        cand = np.flatnonzero(box)
        signed = (tri[0][None, :] - pos[cand]) @ normal
        d = np.abs(signed)
        # Closed solid faces block their outward (fluid) side. A nearby back
        # face must not block a real free surface across a rim or thin wall.
        near_local = (d <= float(support) + 1.0e-12) & (signed <= 1.0e-12)
        if not np.any(near_local):
            continue
        idx = cand[near_local]
        dnear = d[near_local]
        signed_near = signed[near_local]
        projected = pos[idx] + signed_near[:, None] * normal[None, :]
        inside = _point_in_triangle_batch(projected, tri, tol=1.0e-9)
        # A fluid kernel also intersects a solid across a finite edge/corner.
        # Plane-projection-only tests miss BOTH faces just outside a convex
        # corner and incorrectly seed a submerged "air" surface there.
        # Admit those points only within the actual finite-triangle radius.
        outside = ~inside
        if np.any(outside):
            closest = closest_point_triangle_batch(pos[idx[outside]], *tri)
            edge_distance = np.linalg.norm(pos[idx[outside]] - closest, axis=1)
            inside[outside] = edge_distance <= float(support) + 1.0e-12
            dnear[outside] = edge_distance
        if not np.any(inside):
            continue
        hit = idx[inside]
        dhit = dnear[inside]
        cov[hit] += np.outer(normal, normal)
        tri_count[hit] += 1
        min_distance[hit] = np.minimum(min_distance[hit], dhit)
    return cov, tri_count, min_distance


def project_missing_from_solids(missing, cov, geom_active, eigen_rel):
    """Remove missing-support components blocked by finite solid faces.

    Same 3x3 eigenspace projection as before, now batched instead of one
    Python loop per active fluid particle.
    """
    missing = np.asarray(missing, dtype=np.float64)
    air = missing.copy()
    rank = np.zeros(len(missing), dtype=np.int8)
    ids = np.flatnonzero(geom_active)
    if len(ids):
        vals, vecs = np.linalg.eigh(cov[ids])
        vmax = vals[:, -1]
        usable = vmax > 1.0e-30
        if np.any(usable):
            uid = ids[usable]
            vv = vals[usable]
            V = vecs[usable]
            keep = vv > float(eigen_rel) * vmax[usable, None]
            coords = np.einsum("nij,ni->nj", V, missing[uid])
            coords *= keep
            projected = np.einsum("nij,nj->ni", V, coords)
            air[uid] = missing[uid] - projected
            rank[uid] = np.sum(keep, axis=1).astype(np.int8)
    mag = np.linalg.norm(missing, axis=1)
    amag = np.linalg.norm(air, axis=1)
    blocked = np.zeros(len(missing), dtype=np.float64)
    good = mag > 1.0e-30
    blocked[good] = np.clip(1.0 - amag[good] / mag[good], 0.0, 1.0)
    return air, rank, blocked


def build_wetted_wall_support_G(pos, boundary, interactions, air_seed, sph_cfg, fs_cfg):
    G = np.zeros((len(pos), 3), dtype=np.float64)
    pair_i = np.asarray(interactions["pair_i"], dtype=np.int64)
    pair_j = np.asarray(interactions["pair_j"], dtype=np.int64)
    if len(pair_i) == 0:
        return G, np.zeros(0, dtype=bool)
    amag = np.linalg.norm(air_seed, axis=1)
    normal = np.zeros_like(pos)
    good = amag > fs_cfg.eps
    normal[good] = air_seed[good] / amag[good, None]
    plane_point = pos + fs_cfg.probe_offset_factor * sph_cfg.dp * normal
    rel = np.asarray(boundary.positions[pair_j], dtype=np.float64) - plane_point[pair_i]
    signed_air = np.einsum("ij,ij->i", rel, normal[pair_i])
    wet = good[pair_i] & (signed_air <= 1.0e-12)
    if np.any(wet):
        ii, jj = pair_i[wet], pair_j[wet]
        Vb = np.asarray(boundary.volumes[jj], dtype=np.float64)
        grad = np.asarray(interactions["gradW"], dtype=np.float64)[wet]
        G = _scatter_add_vec3(len(pos), ii, Vb[:, None] * grad)
    return G, wet


def build_seed_connected_band_with_anchor(pos, pair_i, pair_j, air_missing, shell_mask, fs_cfg):
    n = len(pos)
    amag = np.linalg.norm(air_missing, axis=1)
    normals = np.zeros_like(pos)
    nz = amag > fs_cfg.eps
    normals[nz] = air_missing[nz] / amag[nz, None]
    band = shell_mask.copy()
    anchor = np.full(n, -1, dtype=np.int64)
    shell_idx = np.flatnonzero(shell_mask)
    anchor[shell_idx] = shell_idx

    if len(pair_i) == 0 or len(shell_idx) == 0:
        return band, anchor, normals

    seed = np.concatenate((pair_i, pair_j)).astype(np.int64, copy=False)
    cand = np.concatenate((pair_j, pair_i)).astype(np.int64, copy=False)
    valid = shell_mask[seed] & nz[cand] & (~shell_mask[cand])
    if not np.any(valid):
        return band, anchor, normals
    seed = seed[valid]
    cand = cand[valid]
    rel = pos[cand] - pos[seed]
    inward = np.einsum("ij,ij->i", rel, normals[seed]) <= 1.0e-10
    compatible = np.einsum("ij,ij->i", normals[cand], normals[seed]) >= fs_cfg.normal_compat_dot
    ok = inward & compatible
    if not np.any(ok):
        return band, anchor, normals
    seed = seed[ok]
    cand = cand[ok]
    d2 = np.einsum("ij,ij->i", rel[ok], rel[ok])

    # Sort by candidate then distance and take the nearest compatible shell seed.
    order = np.lexsort((d2, cand))
    cand_s = cand[order]
    seed_s = seed[order]
    first = np.empty(len(cand_s), dtype=bool)
    first[0] = True
    first[1:] = cand_s[1:] != cand_s[:-1]
    cc = cand_s[first]
    ss = seed_s[first]
    anchor[cc] = ss
    band[cc] = True
    return band, anchor, normals


def anchored_band_traction(pos, rho, air_missing, band, anchor, normals, surface_level, sph_cfg, fs_cfg):
    shell_interface = pos + fs_cfg.probe_offset_factor * sph_cfg.dp * normals
    probe = pos.copy()
    valid = band & (anchor >= 0)
    probe[valid] = shell_interface[anchor[valid]]
    rho_s, p_s, _ = hydrostatic_reference_state(probe, surface_level, sph_cfg, return_depth=True)
    rho_safe = np.maximum(rho, 1.0e-30)
    rho_s_safe = np.maximum(rho_s, 1.0e-30)
    coeff = p_s * (rho_s / (rho_safe * rho_safe) + 1.0 / rho_s_safe)
    acc = np.zeros_like(pos, dtype=np.float64)
    acc[valid] = coeff[valid, None] * air_missing[valid]
    return acc, p_s, probe, valid


def compute_atmospheric_state(
    pos, rho, rho_ref, pair_i, pair_j, gradW,
    boundary, wall_interactions, solid_triangles, solid_normals,
    surface_level, sph_cfg, fs_cfg, active_mask=None,
):
    n = len(pos)
    if fs_cfg.mode == "none":
        z3 = np.zeros((n, 3), dtype=np.float64)
        z1 = np.zeros(n, dtype=np.float64)
        zb = np.zeros(n, dtype=bool)
        zi = np.full(n, -1, dtype=np.int64)
        return {
            "acc": z3, "shell": zb, "band": zb.copy(), "anchor": zi,
            "normal": z3.copy(), "support_hG": z1, "raw_fluid_hG": z1.copy(),
            "geom_rank": np.zeros(n, dtype=np.int8), "blocked_fraction": z1.copy(),
            "p_ref_surface": z1.copy(), "valid_band": zb.copy(), "wet_pairs": np.zeros(0, dtype=bool),
            "air_missing": z3.copy(),
        }
    if fs_cfg.mode != "anchored_band":
        raise ValueError(f"Unknown free-surface mode: {fs_cfg.mode}")

    G_fluid = compute_fluid_support_G(n, rho_ref, sph_cfg.mass, pair_i, pair_j, gradW)
    missing_fluid = -G_fluid
    raw_fluid_hG = sph_cfg.h * np.linalg.norm(missing_fluid, axis=1)

    if len(solid_triangles):
        geom_cov, geom_tri_count, _ = compute_geometry_normal_covariance(
            pos, solid_triangles, solid_normals, sph_cfg.support
        )
        geom_active = geom_tri_count > 0
        air_seed, _, _ = project_missing_from_solids(
            missing_fluid, geom_cov, geom_active, fs_cfg.collider_eigen_rel
        )
    else:
        geom_cov = np.zeros((n, 3, 3), dtype=np.float64)
        geom_active = np.zeros(n, dtype=bool)
        air_seed = missing_fluid.copy()

    G_wall_wet, wet_pairs = build_wetted_wall_support_G(
        pos, boundary, wall_interactions, air_seed, sph_cfg, fs_cfg
    )
    missing_completed = -(G_fluid + G_wall_wet)
    if len(solid_triangles):
        air_missing, geom_rank, blocked_fraction = project_missing_from_solids(
            missing_completed, geom_cov, geom_active, fs_cfg.collider_eigen_rel
        )
    else:
        air_missing = missing_completed
        geom_rank = np.zeros(n, dtype=np.int8)
        blocked_fraction = np.zeros(n, dtype=np.float64)

    support_hG = sph_cfg.h * np.linalg.norm(air_missing, axis=1)
    shell = support_hG > fs_cfg.hG_threshold
    # Anchored-band traction references the dominant pool's hydrostatic
    # surface. Applying it to a disconnected droplet far above/below that pool
    # creates fictitious pressure (and can launch runaway particles). Detached
    # components still receive gravity and optional aerodynamic drag, but no
    # pool-anchored free-surface traction.
    if active_mask is not None:
        active_mask = np.asarray(active_mask, dtype=bool)
        if active_mask.shape != (n,):
            raise ValueError("active_mask must have one boolean per fluid particle")
        shell &= active_mask
    band, anchor, normals = build_seed_connected_band_with_anchor(
        pos, pair_i, pair_j, air_missing, shell, fs_cfg
    )
    acc, p_ref_surface, _probe, valid_band = anchored_band_traction(
        pos, rho, air_missing, band, anchor, normals, surface_level, sph_cfg, fs_cfg
    )
    return {
        "acc": acc,
        "shell": shell,
        "band": band,
        "anchor": anchor,
        "normal": normals,
        "support_hG": support_hG,
        "raw_fluid_hG": raw_fluid_hG,
        "geom_rank": geom_rank,
        "blocked_fraction": blocked_fraction,
        "p_ref_surface": p_ref_surface,
        "valid_band": valid_band,
        "wet_pairs": wet_pairs,
        "air_missing": air_missing,
    }
