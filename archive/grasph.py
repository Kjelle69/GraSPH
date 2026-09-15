import sys
import time
import os

import numpy as np

from pxr import UsdGeom, Gf

from grasph_scene import load_scene, generate_fluid_particles
from grasph_collision import build_colliders, resolve_collisions, closest_point_triangle_batch
from grasph_boundary import (
    build_boundary_particles,
    build_layered_wall_particles,
    build_wall_interactions,
    add_static_wall_density_rate,
    add_static_wall_pressure_acceleration,
)
from grasph_sph import (
    SPHConfig,
    WendlandC2,
    build_neighbor_pairs,
    compute_pair_geometry,
    neighbor_counts,
    compute_density_rate,
    compute_pressure,
    compute_pressure_acceleration,
    compute_artificial_viscosity_acceleration,
    compute_cfl_dt,
)


# ============================================================
# GraSPH v0.51 - ANCHOR-NORMAL CONSISTENCY REPLAY
#
# Independent physical benchmark after freezing the face-projected
# AIR-vs-SOLID model in v0.37. No gate and no external forcing.
# A hydrostatic rectangular column has an exposed vertical free face
# and collapses onto a dry bed.
#
# Ritter ideal shallow-water reference:
#   x_f(t) = x_f(0) + 2*sqrt(g*h0)*t
#
# Bulk SPH physics and the v0.45 atmospheric model remain unchanged.
# NO physics term is changed. The diagnostic can run the exact same model at
# dp=0.020 or 0.010 m and records a global X-momentum / force decomposition.
# v0.47 showed that essentially all resolution-dependent global X-force comes
# from atmospheric traction. v0.48 then showed that deep graph-hop particles
# (L3/OUT) are only a minority of the fine-resolution over-force: L1+L2 alone
# can already exceed the continuum force. v0.49 changes NO physics and asks
# whether the diffuse SPH free-surface measure itself loses the continuum
# projected-area / first-depth-moment identities as the surface deforms.
# ============================================================

SIM_TIME = 0.12
DT_MAX = 0.00030
DEFAULT_DP = 0.020
H_FACTOR_EXPECTED = 1.30
COLLISION_VOXEL_FACTOR = 0.50
COLLISION_MARGIN_FACTOR = 2.00
CFL = 0.25
OUTPUT_FPS = 50.0
REPORT_INTERVAL = 1.0
PARTICLE_RADIUS_FACTOR = 0.45
ARTIFICIAL_VISCOSITY_ALPHA = 0.05
COLLISION_ITERATIONS = 2

ENABLE_WALL_CONTINUITY = True
ENABLE_WALL_PRESSURE = True
FLUID_STATE_MODE = "well_balanced_continuity"
ENABLE_COLLISION = True
PRESSURE_OPERATOR_MODE = "pairwise_symmetric_reduced_pressure"
WALL_MODE = "energy_matched_wetted_band_anchored_surface_traction"
INTEGRATOR_MODE = "symplectic_kick_continuity_drift"

DRIVE_START_TIME = 99.0
DRIVE_DURATION = 1.0
DRIVE_END_TIME = DRIVE_START_TIME + DRIVE_DURATION
DRIVE_PEAK_ACCEL = 0.0
DRIVE_AXIS = np.array([1.0, 0.0, 0.0], dtype=np.float64)

# v0.32 known-geometry calibration gave a wide clean separation at 0.35:
# 360/360 known top surface, 0 solid FP, 0 non-wall FP.
CURRENT_SURFACE_HG_THRESHOLD = 0.35
COLLIDER_NORMAL_EIGEN_REL = 0.10
SURFACE_PROBE_OFFSET_FACTOR = 0.50
AIR_EPS = 1.0e-12
NORMAL_COMPAT_DOT = 0.50

OUTPUT_DIR = "output"
NPZ_OUTPUT = None
USD_OUTPUT = None


def format_time(seconds):
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def drive_acceleration(time_s):
    """Smooth +X half-sine body acceleration used only to excite slosh."""
    t = float(time_s)
    if t < DRIVE_START_TIME or t > DRIVE_END_TIME:
        return np.zeros(3, dtype=np.float64)
    phase = (t - DRIVE_START_TIME) / DRIVE_DURATION
    return DRIVE_AXIS * (DRIVE_PEAK_ACCEL * np.sin(np.pi * phase))


def write_usd_animation(stage, frames, fps, dp, output_file):
    print()
    print("Writing OpenUSD animation...")

    points_prim = UsdGeom.Points.Define(stage, "/root/GraSPH_Result")
    points_attr = points_prim.CreatePointsAttr()
    widths_attr = points_prim.CreateWidthsAttr()

    n_particles = len(frames[0])
    widths_attr.Set([dp] * n_particles)

    for frame_index, positions in enumerate(frames):
        usd_points = [
            Gf.Vec3f(float(p[0]), float(p[1]), float(p[2]))
            for p in positions
        ]
        points_attr.Set(usd_points, frame_index)

    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(max(0, len(frames) - 1))
    stage.SetTimeCodesPerSecond(float(fps))
    stage.SetFramesPerSecond(float(fps))
    stage.GetRootLayer().Export(output_file)

    print(f"USD frames: {len(frames)}")
    print(f"Saved: {output_file}")


# ============================================================
# EXACT TAIT-EOS HYDROSTATIC REFERENCE
#
# Tait EOS:
#   p = B[(rho/rho0)^gamma - 1]
#
# Hydrostatic equilibrium:
#   grad(p) = rho * g
#
# Integrating the two gives, for depth d below the reference free surface,
#
#   rho/rho0 = [1 + (gamma-1) g d / c0^2]^(1/(gamma-1))
#
# This is slightly more exact than the old linear p=rho0*g*d start state.
# ============================================================

def hydrostatic_reference_state(pos, surface_level, cfg, return_depth=False):
    pos = np.asarray(pos, dtype=np.float64)
    gvec = np.asarray(cfg.gravity, dtype=np.float64)
    gmag = float(np.linalg.norm(gvec))

    if gmag <= 1.0e-15:
        rho = np.full(len(pos), cfg.rho0, dtype=np.float64)
        pressure = np.zeros(len(pos), dtype=np.float64)
        depth = np.zeros(len(pos), dtype=np.float64)
        if return_depth:
            return rho, pressure, depth
        return rho, pressure

    up = -gvec / gmag
    level = pos @ up
    depth = np.maximum(float(surface_level) - level, 0.0)

    if abs(cfg.gamma - 1.0) < 1.0e-12:
        ratio = np.exp(gmag * depth / (cfg.c0 * cfg.c0))
    else:
        base = 1.0 + (cfg.gamma - 1.0) * gmag * depth / (cfg.c0 * cfg.c0)
        ratio = base ** (1.0 / (cfg.gamma - 1.0))

    rho = cfg.rho0 * ratio
    pressure = cfg.tait_B * (ratio ** cfg.gamma - 1.0)

    if return_depth:
        return rho, pressure, depth
    return rho, pressure


def extrapolate_wall_reduced_pressure(fluid_q, boundary, interactions):
    """Zero-normal-gradient wall state for the dynamic pressure q.

    The hydrostatic background is handled analytically.  The fixed wall only
    needs a state for the *perturbation* pressure.  A kernel-weighted average
    of neighbouring fluid q values is a simple static-wall Neumann closure.
    It is exactly q_wall=0 when the fluid is at the reference hydrostatic
    state, which is essential for the well-balanced test.
    """
    nb = len(boundary.positions)
    q_wall = np.zeros(nb, dtype=np.float64)

    pair_i = interactions["pair_i"]
    pair_j = interactions["pair_j"]
    if len(pair_i) == 0:
        return q_wall

    W = interactions["W"]
    num = np.zeros(nb, dtype=np.float64)
    den = np.zeros(nb, dtype=np.float64)

    np.add.at(num, pair_j, fluid_q[pair_i] * W)
    np.add.at(den, pair_j, W)

    active = den > 1.0e-30
    q_wall[active] = num[active] / den[active]
    return q_wall


def compute_well_balanced_base_acceleration(rho, rho_ref, depth_ref, gravity):
    """Exact analytical contribution of gravity + reference pressure.

    Below the fixed reference free surface:

        a_base = g - (rho_ref/rho) g

    because grad(p_ref)=rho_ref*g.  Above the reference free surface p_ref=0,
    so gravity remains fully active.
    """
    rho = np.asarray(rho, dtype=np.float64)
    rho_ref = np.asarray(rho_ref, dtype=np.float64)
    gvec = np.asarray(gravity, dtype=np.float64)

    acc = np.repeat(gvec[None, :], len(rho), axis=0)
    submerged = depth_ref > 0.0
    if np.any(submerged):
        ratio = rho_ref[submerged] / np.maximum(rho[submerged], 1.0e-30)
        acc[submerged] *= (1.0 - ratio)[:, None]
    return acc



def apply_density_pulse(pos, rho, amplitude, dp, sigma_factor=2.5):
    """Add a tiny smooth deterministic density perturbation.

    The pulse is intentionally tiny relative to rho0.  It exists only to test
    whether the continuity + pressure pair propagates/damps an acoustic mode or
    amplifies it numerically.
    """
    if amplitude == 0.0 or len(pos) == 0:
        return np.zeros(len(pos), dtype=np.float64)

    center = np.mean(pos, axis=0)
    sigma = max(float(sigma_factor) * float(dp), 1.0e-12)
    d = pos - center[None, :]
    r2 = np.einsum("ij,ij->i", d, d)
    shape = np.exp(-0.5 * r2 / (sigma * sigma))

    # Remove the spatial mean so the perturbation has no bulk DC component.
    shape -= np.mean(shape)
    peak = np.max(np.abs(shape))
    if peak > 0.0:
        shape /= peak

    delta = float(amplitude) * shape
    rho += delta
    return delta

def acoustic_energy_proxy(vel, rho, rho_ref, mass, c0):
    """Small-perturbation acoustic energy proxy.

    E = sum_i 0.5*m*(|v|^2 + c0^2*(delta_rho/rho_ref)^2)

    This is diagnostic only; it is not used by the solver.
    """
    vel = np.asarray(vel, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    rho_ref = np.asarray(rho_ref, dtype=np.float64)
    dr = rho - rho_ref
    kinetic = np.einsum("ij,ij->i", vel, vel)
    compressive = (float(c0) * dr / np.maximum(rho_ref, 1.0e-30)) ** 2
    return 0.5 * float(mass) * float(np.sum(kinetic + compressive))




def compute_fluid_support_G(n, rho_ref, mass, pair_i, pair_j, gradW):
    """Kernel-gradient support imbalance from FLUID neighbors only."""
    G = np.zeros((n, 3), dtype=np.float64)
    if len(pair_i):
        Vi = mass / np.maximum(rho_ref[pair_i], 1.0e-30)
        Vj = mass / np.maximum(rho_ref[pair_j], 1.0e-30)
        np.add.at(G, pair_i, Vj[:, None] * gradW)
        np.add.at(G, pair_j, -Vi[:, None] * gradW)
    return G


def collider_triangle_data(scene):
    """Return actual collider triangles and unit normals.

    Normal sign is irrelevant because the solid classifier uses n n^T.
    """
    normals = []
    triangles = []
    paths = []
    for mesh in scene.colliders:
        tri = np.asarray(mesh.triangles, dtype=np.float64)
        for t in tri:
            e1 = t[1] - t[0]
            e2 = t[2] - t[0]
            nn = np.cross(e1, e2)
            ln = float(np.linalg.norm(nn))
            if ln <= 1.0e-15:
                continue
            normals.append(nn / ln)
            triangles.append(t)
            paths.append(mesh.path)
    if not triangles:
        return np.empty((0, 3, 3)), np.empty((0, 3)), []
    return np.asarray(triangles), np.asarray(normals), paths


def point_in_triangle_batch(points, tri, tol=1.0e-10):
    """Barycentric inside test for many coplanar projected points."""
    p = np.asarray(points, dtype=np.float64)
    a, b, c = np.asarray(tri, dtype=np.float64)
    v0 = b - a
    v1 = c - a
    v2 = p - a[None, :]

    d00 = float(np.dot(v0, v0))
    d01 = float(np.dot(v0, v1))
    d11 = float(np.dot(v1, v1))
    denom = d00 * d11 - d01 * d01
    if abs(denom) <= 1.0e-30:
        return np.zeros(len(p), dtype=bool)

    d20 = v2 @ v0
    d21 = v2 @ v1
    inv = 1.0 / denom
    v = (d11 * d20 - d01 * d21) * inv
    w = (d00 * d21 - d01 * d20) * inv
    u = 1.0 - v - w
    return (u >= -tol) & (v >= -tol) & (w >= -tol)


def compute_geometry_normal_covariance(pos, triangles, normals, support):
    """Local SOLID-normal span from finite collider face projection.

    A face contributes only if the particle is within `support` in the face
    normal direction AND the particle's orthogonal projection lies inside the
    actual finite triangle footprint. This is the v0.35-tested cure for
    tangential over-occlusion near finite edges.
    """
    n = len(pos)
    cov = np.zeros((n, 3, 3), dtype=np.float64)
    tri_count = np.zeros(n, dtype=np.int16)
    min_distance = np.full(n, np.inf, dtype=np.float64)

    for tri, normal in zip(triangles, normals):
        a = tri[0]
        signed = (a[None, :] - pos) @ normal
        d = np.abs(signed)
        near = d <= float(support) + 1.0e-12
        if not np.any(near):
            continue

        idx = np.flatnonzero(near)
        projected = pos[idx] + signed[idx, None] * normal[None, :]
        inside = point_in_triangle_batch(projected, tri, tol=1.0e-9)
        if not np.any(inside):
            continue

        hit = idx[inside]
        cov[hit] += np.outer(normal, normal)
        tri_count[hit] += 1
        min_distance[hit] = np.minimum(min_distance[hit], d[hit])

    return cov, tri_count, min_distance


def project_missing_from_solids(missing, cov, geom_active, eigen_rel):
    """Remove from missing support every locally represented solid-normal mode."""
    missing = np.asarray(missing, dtype=np.float64)
    n = len(missing)
    air = missing.copy()
    rank = np.zeros(n, dtype=np.int8)
    active = np.flatnonzero(geom_active)
    for i in active:
        vals, vecs = np.linalg.eigh(cov[i])
        vmax = float(vals[-1])
        if vmax <= 1.0e-30:
            continue
        keep = vals > (float(eigen_rel) * vmax)
        if not np.any(keep):
            continue
        B = vecs[:, keep]
        rank[i] = int(B.shape[1])
        air[i] = missing[i] - B @ (B.T @ missing[i])

    mag = np.linalg.norm(missing, axis=1)
    amag = np.linalg.norm(air, axis=1)
    blocked = np.zeros(n, dtype=np.float64)
    good = mag > 1.0e-30
    blocked[good] = np.clip(1.0 - amag[good] / mag[good], 0.0, 1.0)
    return air, rank, blocked


def build_wetted_wall_support_G(pos, boundary, interactions, air_seed, cfg):
    """Complete kernel support with wall samples only on the wet side.

    The local AIR plane is seeded from the fluid-only, collider-occluded
    missing-support vector. Wall samples ahead of that interface are rejected.
    This is the v0.41 contact-line completion rule.
    """
    n = len(pos)
    G = np.zeros((n, 3), dtype=np.float64)
    pair_i = np.asarray(interactions["pair_i"], dtype=np.int64)
    pair_j = np.asarray(interactions["pair_j"], dtype=np.int64)
    if len(pair_i) == 0:
        return G, np.zeros(0, dtype=bool)

    amag = np.linalg.norm(air_seed, axis=1)
    normal = np.zeros_like(pos)
    good = amag > AIR_EPS
    normal[good] = air_seed[good] / amag[good, None]
    plane_point = pos + SURFACE_PROBE_OFFSET_FACTOR * cfg.dp * normal

    rel = np.asarray(boundary.positions[pair_j], dtype=np.float64) - plane_point[pair_i]
    signed_air = np.einsum("ij,ij->i", rel, normal[pair_i])
    wet = good[pair_i] & (signed_air <= 1.0e-12)

    if np.any(wet):
        ii = pair_i[wet]
        jj = pair_j[wet]
        Vb = np.asarray(boundary.volumes[jj], dtype=np.float64)
        grad = np.asarray(interactions["gradW"], dtype=np.float64)[wet]
        np.add.at(G, ii, Vb[:, None] * grad)
    return G, wet


def build_seed_connected_band_with_anchor(pos, pair_i, pair_j, air_missing, shell_mask):
    """Surface shell + one inward SPH neighbor layer with shell anchors.

    Each inner-band particle inherits the nearest compatible shell seed.
    The seed defines the physical interface point used for p_ref, avoiding
    the hydrostatic bias demonstrated by v0.44.
    """
    n = len(pos)
    amag = np.linalg.norm(air_missing, axis=1)
    normals = np.zeros_like(pos)
    nz = amag > AIR_EPS
    normals[nz] = air_missing[nz] / amag[nz, None]

    band = shell_mask.copy()
    anchor = np.full(n, -1, dtype=np.int64)
    shell_idx = np.flatnonzero(shell_mask)
    anchor[shell_idx] = shell_idx
    best_d2 = np.full(n, np.inf, dtype=np.float64)
    best_d2[shell_idx] = 0.0

    def consider(seed_idx, cand_idx):
        valid = shell_mask[seed_idx] & nz[cand_idx] & (~shell_mask[cand_idx])
        if not np.any(valid):
            return
        ss = seed_idx[valid]
        cc = cand_idx[valid]
        rel = pos[cc] - pos[ss]
        inward = np.einsum("ij,ij->i", rel, normals[ss]) <= 1.0e-10
        compatible = np.einsum("ij,ij->i", normals[cc], normals[ss]) >= NORMAL_COMPAT_DOT
        ok = inward & compatible
        if not np.any(ok):
            return
        ss = ss[ok]
        cc = cc[ok]
        d2 = np.einsum("ij,ij->i", pos[cc] - pos[ss], pos[cc] - pos[ss])
        for sidx, cidx, dd in zip(ss, cc, d2):
            if dd < best_d2[cidx]:
                best_d2[cidx] = dd
                anchor[cidx] = sidx
                band[cidx] = True

    consider(pair_i, pair_j)
    consider(pair_j, pair_i)
    return band, anchor, normals


def anchored_band_traction(pos, rho, air_missing, band, anchor, normals, surface_level, cfg):
    """Atmospheric reference traction over the band using shell interface anchors."""
    shell_interface = pos + SURFACE_PROBE_OFFSET_FACTOR * cfg.dp * normals
    probe = pos.copy()
    valid = band & (anchor >= 0)
    probe[valid] = shell_interface[anchor[valid]]

    rho_s, p_s, _ = hydrostatic_reference_state(
        probe, surface_level, cfg, return_depth=True
    )
    rho_safe = np.maximum(rho, 1.0e-30)
    rho_s_safe = np.maximum(rho_s, 1.0e-30)
    coeff = p_s * (rho_s / (rho_safe * rho_safe) + 1.0 / rho_s_safe)

    acc = np.zeros_like(pos, dtype=np.float64)
    acc[valid] = coeff[valid, None] * air_missing[valid]
    return acc, p_s, probe, valid


def traction_band_depth_diagnostic(pos, acc_atm, band, anchor, normals, probe, cfg):
    """Diagnostic only: split atmospheric Fx by inward distance from anchor plane.

    The current v0.45/v0.47 band is a one-hop neighbor graph expansion. A
    single SPH hop can span the full 2h support, so it is not automatically
    equivalent to one physical lattice layer. This diagnostic bins every
    active traction particle by signed inward depth from its inherited shell
    interface. It DOES NOT alter acc_atm or the simulated physics.
    """
    n = len(pos)
    depth = np.full(n, np.nan, dtype=np.float64)
    valid = band & (anchor >= 0)
    if np.any(valid):
        idx = np.flatnonzero(valid)
        aidx = anchor[idx]
        n_anchor = normals[aidx]
        rel = pos[idx] - probe[idx]
        depth[idx] = -np.einsum("ij,ij->i", rel, n_anchor)

    tol = 1.0e-10 * max(1.0, cfg.dp)
    ahead = valid & (depth < -tol)
    nonneg = valid & (~ahead)
    l1 = nonneg & (depth <= 1.0 * cfg.dp + tol)
    l2 = nonneg & (depth > 1.0 * cfg.dp + tol) & (depth <= 2.0 * cfg.dp + tol)
    l3 = nonneg & (depth > 2.0 * cfg.dp + tol) & (depth <= cfg.support + tol)
    outside = nonneg & (depth > cfg.support + tol)

    def fx(mask):
        return cfg.mass * float(np.sum(acc_atm[mask, 0])) if np.any(mask) else 0.0

    return {
        "depth": depth,
        "valid": valid,
        "ahead": ahead,
        "l1": l1,
        "l2": l2,
        "l3": l3,
        "outside": outside,
        "Fx_ahead": fx(ahead),
        "Fx_l1": fx(l1),
        "Fx_l2": fx(l2),
        "Fx_l3": fx(l3),
        "Fx_outside": fx(outside),
    }



def traction_surface_moment_diagnostic(
    rho,
    air_missing,
    band,
    shell_mask,
    anchor,
    probe,
    surface_level,
    cfg,
):
    """Diagnostic-only diffuse interface area/moment carried by traction.

    The atmospheric operator can be written particlewise as

        F_i = p_ref(interface_i) * dA_i

    with the signed diffuse area vector

        dA_i = m_i * [rho_s/rho_i^2 + 1/rho_s] * G_air_i.

    For the Ritter geometry before the rear rarefaction reaches the upstream
    wall, continuum geometry requires the free surface to satisfy the signed
    X-projection identities

        integral n_x dA                 = width * h0
        integral depth_ref * n_x dA    = width * h0^2 / 2

    independent of how wrinkled/sloped the free surface becomes.  This
    diagnostic measures those two quantities without changing acc_atm.
    """
    valid = band & (anchor >= 0)

    rho_s, p_s, depth_s = hydrostatic_reference_state(
        probe, surface_level, cfg, return_depth=True
    )
    rho_safe = np.maximum(np.asarray(rho, dtype=np.float64), 1.0e-30)
    rho_s_safe = np.maximum(np.asarray(rho_s, dtype=np.float64), 1.0e-30)

    beta = rho_s / (rho_safe * rho_safe) + 1.0 / rho_s_safe
    dA = cfg.mass * beta[:, None] * np.asarray(air_missing, dtype=np.float64)

    shell = valid & shell_mask
    inner = valid & (~shell_mask)

    def stats(mask):
        if not np.any(mask):
            return {
                "count": 0,
                "Ax": 0.0,
                "Ax_abs": 0.0,
                "Ax_pos": 0.0,
                "Ax_neg": 0.0,
                "Hx": 0.0,
                "Fx": 0.0,
            }
        dax = dA[mask, 0]
        return {
            "count": int(np.count_nonzero(mask)),
            "Ax": float(np.sum(dax)),
            "Ax_abs": float(np.sum(np.abs(dax))),
            "Ax_pos": float(np.sum(np.maximum(dax, 0.0))),
            "Ax_neg": float(np.sum(np.minimum(dax, 0.0))),
            "Hx": float(np.sum(depth_s[mask] * dax)),
            "Fx": float(np.sum(p_s[mask] * dax)),
        }

    return {
        "all": stats(valid),
        "shell": stats(shell),
        "inner": stats(inner),
    }




def wendland_plane_J(depth, h):
    """Exact plane integral J(s)=integral_plane W(sqrt(s^2+r^2)) dA for 3D Wendland C2.

    For a locally planar free surface and a fluid particle at inward normal
    distance s>=0, the continuum missing-support vector is J(s)*n.  This is
    independent of particle disorder and uses only the already reconstructed
    shell interface plane.
    """
    depth = np.asarray(depth, dtype=np.float64)
    out = np.zeros_like(depth)
    active = (depth >= 0.0) & (depth < 2.0 * h)
    if not np.any(active):
        return out
    q = depth[active] / h

    def F(x):
        return (
            x**7 / 56.0
            - 5.0 * x**6 / 32.0
            + x**5 / 2.0
            - 5.0 * x**4 / 8.0
            + x**2 / 2.0
        )

    out[active] = (21.0 / (8.0 * h)) * (F(2.0) - F(q))
    return out


def analytic_halfspace_surface_moment_diagnostic(
    pos,
    rho,
    band,
    shell_mask,
    anchor,
    normals,
    probe,
    surface_level,
    cfg,
):
    """Candidate diagnostic only: replace raw |G_air| by exact local half-space J(s).

    The interface location and normal are NOT changed.  Each active band
    particle inherits the shell anchor's local tangent plane, and the missing
    kernel integral is evaluated analytically from its inward normal distance.
    No force is fed back into the simulation in v0.50.
    """
    n = len(pos)
    valid = band & (anchor >= 0)
    depth_geom = np.full(n, np.nan, dtype=np.float64)
    n_anchor = np.zeros_like(pos, dtype=np.float64)
    if np.any(valid):
        idx = np.flatnonzero(valid)
        aidx = anchor[idx]
        n_anchor[idx] = normals[aidx]
        rel = pos[idx] - probe[idx]
        depth_geom[idx] = -np.einsum('ij,ij->i', rel, n_anchor[idx])

    J = np.zeros(n, dtype=np.float64)
    active = valid & np.isfinite(depth_geom) & (depth_geom >= 0.0) & (depth_geom < cfg.support)
    if np.any(active):
        J[active] = wendland_plane_J(depth_geom[active], cfg.h)

    rho_s, p_s, depth_s = hydrostatic_reference_state(
        probe, surface_level, cfg, return_depth=True
    )
    rho_safe = np.maximum(np.asarray(rho, dtype=np.float64), 1.0e-30)
    rho_s_safe = np.maximum(np.asarray(rho_s, dtype=np.float64), 1.0e-30)
    beta = rho_s / (rho_safe * rho_safe) + 1.0 / rho_s_safe

    dA = np.zeros_like(pos, dtype=np.float64)
    dA[active] = (
        cfg.mass
        * beta[active, None]
        * J[active, None]
        * n_anchor[active]
    )

    shell = active & shell_mask
    inner = active & (~shell_mask)

    def stats(mask):
        if not np.any(mask):
            return {
                'count': 0, 'Ax': 0.0, 'Ax_abs': 0.0, 'Ax_pos': 0.0,
                'Ax_neg': 0.0, 'Hx': 0.0, 'Fx': 0.0,
            }
        dax = dA[mask, 0]
        return {
            'count': int(np.count_nonzero(mask)),
            'Ax': float(np.sum(dax)),
            'Ax_abs': float(np.sum(np.abs(dax))),
            'Ax_pos': float(np.sum(np.maximum(dax, 0.0))),
            'Ax_neg': float(np.sum(np.minimum(dax, 0.0))),
            'Hx': float(np.sum(depth_s[mask] * dax)),
            'Fx': float(np.sum(p_s[mask] * dax)),
        }

    return {
        'all': stats(active),
        'shell': stats(shell),
        'inner': stats(inner),
        'active': active,
        'depth_geom': depth_geom,
        'J': J,
    }


def run_halfspace_replay(usd_file, npz_file):
    """Replay saved v0.49 frames with an analytic local half-space candidate.

    This advances NO physics.  It reconstructs the exact v0.49 shell/band/anchor
    geometry on each saved frame and compares the current diffuse G-based
    surface measure with J(s)*n from the inherited shell tangent plane.
    """
    data = np.load(npz_file, allow_pickle=False)
    positions = np.asarray(data['positions'], dtype=np.float64)
    rho_frames = np.asarray(data['rho'], dtype=np.float64)
    times = np.asarray(data['times'], dtype=np.float64)
    dp = float(np.asarray(data['dp']))
    surface_level = float(np.asarray(data['surface_level']))

    cfg = SPHConfig()
    cfg.dp = dp
    kernel = WendlandC2(cfg.h)

    scene = load_scene(usd_file)
    solid_triangles, solid_normals, _solid_paths = collider_triangle_data(scene)
    if len(solid_triangles) == 0:
        raise RuntimeError('No valid collider triangles available.')

    surface_boundary = build_boundary_particles(scene, spacing=cfg.dp)
    boundary = build_layered_wall_particles(
        surface_boundary, cfg.dp, cfg.support, cfg.rho0
    )

    pos0 = positions[0]
    initial_floor_z = float(np.min(pos0[:, 2]) - 0.5 * cfg.dp)
    initial_height = float(surface_level - initial_floor_z)
    initial_width = float(np.max(pos0[:, 1]) - np.min(pos0[:, 1]) + cfg.dp)
    gmag = float(np.linalg.norm(np.asarray(cfg.gravity, dtype=np.float64)))
    Ax_ref = initial_width * initial_height
    Hx_ref = 0.5 * initial_width * initial_height**2
    depth_grid = np.linspace(0.0, initial_height, 20001)
    ratio_grid = (
        1.0 + (cfg.gamma - 1.0) * gmag * depth_grid / (cfg.c0 * cfg.c0)
    ) ** (1.0 / (cfg.gamma - 1.0))
    p_grid = cfg.tait_B * (ratio_grid**cfg.gamma - 1.0)
    Fx_ref = initial_width * float(np.trapezoid(p_grid, depth_grid))

    print()
    print('========================================')
    print('GraSPH')
    print('ANALYTIC HALF-SPACE TRACTION QUADRATURE REPLAY v0.50')
    print('========================================')
    print()
    print('Physics       : OFF (saved-frame replay only)')
    print(f'Input NPZ     : {npz_file}')
    print(f'dp/h/support  : {cfg.dp:.4f} / {cfg.h:.4f} / {cfg.support:.4f} m')
    print('Interface     : v0.49 shell + inherited anchor plane (UNCHANGED)')
    print('Current       : diffuse raw G_air measure (v0.49)')
    print('Candidate     : exact local planar half-space J(s) * anchor normal')
    print(f'Reference Ax  : {Ax_ref:.8f} m2')
    print(f'Reference Hx  : {Hx_ref:.8e} m3')
    print(f'Reference Fx  : {Fx_ref:.6f} N')
    print()
    print(' time  shell band | current Ax/Hx/Fx ratios | halfspace Ax/Hx/Fx ratios | fold current/half')

    rows = []
    for frame_idx, (t, pos, rho) in enumerate(zip(times, positions, rho_frames)):
        n = len(pos)
        rho_ref, _p_ref, _depth_ref = hydrostatic_reference_state(
            pos, surface_level, cfg, return_depth=True
        )
        pair_i, pair_j = build_neighbor_pairs(pos, cfg.support)
        _rij, _r, gradW = compute_pair_geometry(pos, pair_i, pair_j, kernel)
        G_fluid = compute_fluid_support_G(
            n, rho_ref, cfg.mass, pair_i, pair_j, gradW
        )
        missing_fluid = -G_fluid
        geom_cov, geom_tri_count, _geom_min_dist = compute_geometry_normal_covariance(
            pos, solid_triangles, solid_normals, cfg.support
        )
        geom_active = geom_tri_count > 0
        air_seed, _seed_rank, _seed_blocked = project_missing_from_solids(
            missing_fluid, geom_cov, geom_active, COLLIDER_NORMAL_EIGEN_REL
        )
        wall_interactions = build_wall_interactions(
            pos, boundary, kernel, cfg.support
        )
        G_wall_wet, _wet_pairs = build_wetted_wall_support_G(
            pos, boundary, wall_interactions, air_seed, cfg
        )
        missing_completed = -(G_fluid + G_wall_wet)
        air_missing, _geom_rank, _blocked_frac = project_missing_from_solids(
            missing_completed, geom_cov, geom_active, COLLIDER_NORMAL_EIGEN_REL
        )
        support_hG = cfg.h * np.linalg.norm(air_missing, axis=1)
        shell = support_hG > CURRENT_SURFACE_HG_THRESHOLD
        band, anchor, normals = build_seed_connected_band_with_anchor(
            pos, pair_i, pair_j, air_missing, shell
        )
        _acc, _p_s, probe, _valid = anchored_band_traction(
            pos, rho, air_missing, band, anchor, normals, surface_level, cfg
        )

        current = traction_surface_moment_diagnostic(
            rho, air_missing, band, shell, anchor, probe, surface_level, cfg
        )
        half = analytic_halfspace_surface_moment_diagnostic(
            pos, rho, band, shell, anchor, normals, probe, surface_level, cfg
        )
        c = current['all']
        h = half['all']
        cr = (c['Ax']/Ax_ref, c['Hx']/Hx_ref, c['Fx']/Fx_ref)
        hr = (h['Ax']/Ax_ref, h['Hx']/Hx_ref, h['Fx']/Fx_ref)
        cfold = c['Ax_abs']/max(abs(c['Ax']), 1.0e-30)
        hfold = h['Ax_abs']/max(abs(h['Ax']), 1.0e-30)
        rows.append((t, *cr, *hr, cfold, hfold))
        print(
            f'{t:5.3f} {np.count_nonzero(shell):5d} {np.count_nonzero(band):5d} | '
            f'{cr[0]:6.3f}/{cr[1]:6.3f}/{cr[2]:6.3f} | '
            f'{hr[0]:6.3f}/{hr[1]:6.3f}/{hr[2]:6.3f} | '
            f'{cfold:5.2f}/{hfold:5.2f}'
        )

    arr = np.asarray(rows, dtype=np.float64)
    cur_hx_err = np.sqrt(np.mean((arr[:,2] - 1.0)**2))
    half_hx_err = np.sqrt(np.mean((arr[:,5] - 1.0)**2))
    cur_fx_err = np.sqrt(np.mean((arr[:,3] - 1.0)**2))
    half_fx_err = np.sqrt(np.mean((arr[:,6] - 1.0)**2))
    last = rows[-1]

    print()
    print('========================================')
    print('REPLAY SUMMARY')
    print('========================================')
    print(f'Frames                    : {len(rows)}')
    print(f'Current Hx ratio final     : {last[2]:.6f}')
    print(f'Halfspace Hx ratio final   : {last[5]:.6f}')
    print(f'Current Fx ratio final     : {last[3]:.6f}')
    print(f'Halfspace Fx ratio final   : {last[6]:.6f}')
    print(f'Current Hx RMS ratio error : {cur_hx_err:.6f}')
    print(f'Halfspace Hx RMS error     : {half_hx_err:.6f}')
    print(f'Current Fx RMS ratio error : {cur_fx_err:.6f}')
    print(f'Halfspace Fx RMS error     : {half_fx_err:.6f}')
    print(f'Current fold final         : {last[7]:.6f}')
    print(f'Halfspace fold final       : {last[8]:.6f}')
    print()
    print('Decision target: does replacing only the RAW G_air magnitude/direction')
    print('by the exact local planar half-space kernel integral collapse the')
    print('dp20/dp10 projected-moment divergence WITHOUT changing interface geometry?')



def smooth_shell_normals_pca(pos, pair_i, pair_j, shell_mask, raw_normals, support):
    """Diagnostic-only local tangent-plane normals for existing shell seeds.

    This changes ONLY the shell normal direction.  Shell membership, band
    membership, anchor identity and the already reconstructed v0.49 interface
    point are all kept fixed.  For each shell seed, fit a local plane to nearby
    shell seeds inside the existing SPH support.  Raw AIR normals are used only
    to reject incompatible faces at sharp corners and to orient the fitted
    eigenvector consistently outward.
    """
    pos = np.asarray(pos, dtype=np.float64)
    raw = np.asarray(raw_normals, dtype=np.float64)
    shell = np.asarray(shell_mask, dtype=bool)
    n = len(pos)
    out = raw.copy()

    # Shell-shell adjacency from the already built SPH neighbor graph.
    adj = [[] for _ in range(n)]
    for a, b in zip(np.asarray(pair_i, dtype=np.int64), np.asarray(pair_j, dtype=np.int64)):
        if shell[a] and shell[b]:
            adj[a].append(int(b))
            adj[b].append(int(a))

    fitted = 0
    for sidx in np.flatnonzero(shell):
        nr = raw[sidx]
        if np.linalg.norm(nr) <= AIR_EPS:
            continue
        cand = [int(sidx)] + adj[int(sidx)]
        # Keep only the same local face family. This prevents top/front corner
        # averaging in the initial Ritter block.
        keep = []
        for j in cand:
            nj = raw[j]
            if np.linalg.norm(nj) <= AIR_EPS:
                continue
            if float(np.dot(nj, nr)) >= NORMAL_COMPAT_DOT:
                keep.append(j)
        if len(keep) < 3:
            continue

        idx = np.asarray(keep, dtype=np.int64)
        rel0 = pos[idx] - pos[sidx]
        r = np.linalg.norm(rel0, axis=1)
        q = np.clip(r / max(float(support), 1.0e-30), 0.0, 1.0)
        # Compact, parameter-free within the already fixed support radius.
        w = (1.0 - q) ** 2
        w[0] = max(w[0], 1.0)
        sw = float(np.sum(w))
        if sw <= 1.0e-30:
            continue
        centroid = np.sum(w[:, None] * pos[idx], axis=0) / sw
        dx = pos[idx] - centroid
        cov = np.einsum('i,ij,ik->jk', w, dx, dx) / sw
        try:
            vals, vecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            continue
        nf = vecs[:, int(np.argmin(vals))]
        ln = float(np.linalg.norm(nf))
        if ln <= 1.0e-30:
            continue
        nf = nf / ln
        if float(np.dot(nf, nr)) < 0.0:
            nf = -nf
        out[sidx] = nf
        fitted += 1

    # Inner-particle entries are not used directly by the half-space replay;
    # nevertheless retain raw values there to make the array well-defined.
    sh = np.flatnonzero(shell)
    if len(sh):
        dots = np.einsum('ij,ij->i', raw[sh], out[sh])
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.degrees(np.arccos(dots))
        mean_angle = float(np.mean(angles))
        max_angle = float(np.max(angles))
    else:
        mean_angle = max_angle = 0.0
    return out, fitted, mean_angle, max_angle


def run_anchor_normal_replay(usd_file, npz_file):
    """Replay v0.49 frames and change only inherited shell normal direction.

    Baseline is the v0.50 analytic half-space J(s) quadrature with the original
    v0.49 AIR normal. Candidate uses a local PCA tangent-plane normal while
    preserving shell/band masks, anchor indices and the original interface
    point. No physics is advanced.
    """
    data = np.load(npz_file, allow_pickle=False)
    positions = np.asarray(data['positions'], dtype=np.float64)
    rho_frames = np.asarray(data['rho'], dtype=np.float64)
    times = np.asarray(data['times'], dtype=np.float64)
    dp = float(np.asarray(data['dp']))
    surface_level = float(np.asarray(data['surface_level']))

    cfg = SPHConfig()
    cfg.dp = dp
    kernel = WendlandC2(cfg.h)

    scene = load_scene(usd_file)
    solid_triangles, solid_normals, _solid_paths = collider_triangle_data(scene)
    if len(solid_triangles) == 0:
        raise RuntimeError('No valid collider triangles available.')

    surface_boundary = build_boundary_particles(scene, spacing=cfg.dp)
    boundary = build_layered_wall_particles(
        surface_boundary, cfg.dp, cfg.support, cfg.rho0
    )

    pos0 = positions[0]
    initial_floor_z = float(np.min(pos0[:, 2]) - 0.5 * cfg.dp)
    initial_height = float(surface_level - initial_floor_z)
    initial_width = float(np.max(pos0[:, 1]) - np.min(pos0[:, 1]) + cfg.dp)
    gmag = float(np.linalg.norm(np.asarray(cfg.gravity, dtype=np.float64)))
    Ax_ref = initial_width * initial_height
    Hx_ref = 0.5 * initial_width * initial_height**2
    depth_grid = np.linspace(0.0, initial_height, 20001)
    ratio_grid = (
        1.0 + (cfg.gamma - 1.0) * gmag * depth_grid / (cfg.c0 * cfg.c0)
    ) ** (1.0 / (cfg.gamma - 1.0))
    p_grid = cfg.tait_B * (ratio_grid**cfg.gamma - 1.0)
    Fx_ref = initial_width * float(np.trapezoid(p_grid, depth_grid))

    print()
    print('========================================')
    print('GraSPH')
    print('ANCHOR-NORMAL CONSISTENCY REPLAY v0.51')
    print('========================================')
    print()
    print('Physics       : OFF (saved-frame replay only)')
    print(f'Input NPZ     : {npz_file}')
    print(f'dp/h/support  : {cfg.dp:.4f} / {cfg.h:.4f} / {cfg.support:.4f} m')
    print('Shell/band    : v0.49 reconstruction (UNCHANGED)')
    print('Anchors       : v0.49 nearest compatible shell seed (UNCHANGED)')
    print('Interface pt  : original v0.49 +0.5dp shell point (UNCHANGED)')
    print('Baseline      : v0.50 half-space J(s) + raw AIR anchor normal')
    print('Candidate     : same J(s), same interface point, local PCA anchor normal')
    print(f'Reference Ax  : {Ax_ref:.8f} m2')
    print(f'Reference Hx  : {Hx_ref:.8e} m3')
    print(f'Reference Fx  : {Fx_ref:.6f} N')
    print()
    print(' time shell band | raw-normal Ax/Hx/Fx | PCA-normal Ax/Hx/Fx | fold raw/PCA | dN mean/max deg')

    rows = []
    for t, pos, rho in zip(times, positions, rho_frames):
        n = len(pos)
        rho_ref, _p_ref, _depth_ref = hydrostatic_reference_state(
            pos, surface_level, cfg, return_depth=True
        )
        pair_i, pair_j = build_neighbor_pairs(pos, cfg.support)
        _rij, _r, gradW = compute_pair_geometry(pos, pair_i, pair_j, kernel)
        G_fluid = compute_fluid_support_G(
            n, rho_ref, cfg.mass, pair_i, pair_j, gradW
        )
        missing_fluid = -G_fluid
        geom_cov, geom_tri_count, _geom_min_dist = compute_geometry_normal_covariance(
            pos, solid_triangles, solid_normals, cfg.support
        )
        geom_active = geom_tri_count > 0
        air_seed, _seed_rank, _seed_blocked = project_missing_from_solids(
            missing_fluid, geom_cov, geom_active, COLLIDER_NORMAL_EIGEN_REL
        )
        wall_interactions = build_wall_interactions(
            pos, boundary, kernel, cfg.support
        )
        G_wall_wet, _wet_pairs = build_wetted_wall_support_G(
            pos, boundary, wall_interactions, air_seed, cfg
        )
        missing_completed = -(G_fluid + G_wall_wet)
        air_missing, _geom_rank, _blocked_frac = project_missing_from_solids(
            missing_completed, geom_cov, geom_active, COLLIDER_NORMAL_EIGEN_REL
        )
        support_hG = cfg.h * np.linalg.norm(air_missing, axis=1)
        shell = support_hG > CURRENT_SURFACE_HG_THRESHOLD
        band, anchor, raw_normals = build_seed_connected_band_with_anchor(
            pos, pair_i, pair_j, air_missing, shell
        )
        _acc, _p_s, probe, _valid = anchored_band_traction(
            pos, rho, air_missing, band, anchor, raw_normals, surface_level, cfg
        )

        raw_half = analytic_halfspace_surface_moment_diagnostic(
            pos, rho, band, shell, anchor, raw_normals, probe, surface_level, cfg
        )
        pca_normals, fitted, mean_ang, max_ang = smooth_shell_normals_pca(
            pos, pair_i, pair_j, shell, raw_normals, cfg.support
        )
        pca_half = analytic_halfspace_surface_moment_diagnostic(
            pos, rho, band, shell, anchor, pca_normals, probe, surface_level, cfg
        )

        r = raw_half['all']
        q = pca_half['all']
        rr = (r['Ax']/Ax_ref, r['Hx']/Hx_ref, r['Fx']/Fx_ref)
        qr = (q['Ax']/Ax_ref, q['Hx']/Hx_ref, q['Fx']/Fx_ref)
        rfold = r['Ax_abs']/max(abs(r['Ax']), 1.0e-30)
        qfold = q['Ax_abs']/max(abs(q['Ax']), 1.0e-30)
        rows.append((t, *rr, *qr, rfold, qfold, mean_ang, max_ang, fitted))
        print(
            f'{t:5.3f} {np.count_nonzero(shell):5d} {np.count_nonzero(band):4d} | '
            f'{rr[0]:6.3f}/{rr[1]:6.3f}/{rr[2]:6.3f} | '
            f'{qr[0]:6.3f}/{qr[1]:6.3f}/{qr[2]:6.3f} | '
            f'{rfold:5.2f}/{qfold:5.2f} | {mean_ang:5.1f}/{max_ang:5.1f}'
        )

    arr = np.asarray([r[:-1] for r in rows], dtype=np.float64)
    raw_hx_err = np.sqrt(np.mean((arr[:,2] - 1.0)**2))
    pca_hx_err = np.sqrt(np.mean((arr[:,5] - 1.0)**2))
    raw_fx_err = np.sqrt(np.mean((arr[:,3] - 1.0)**2))
    pca_fx_err = np.sqrt(np.mean((arr[:,6] - 1.0)**2))
    last = rows[-1]

    print()
    print('========================================')
    print('REPLAY SUMMARY')
    print('========================================')
    print(f'Frames                     : {len(rows)}')
    print(f'Raw-normal Hx ratio final   : {last[2]:.6f}')
    print(f'PCA-normal Hx ratio final   : {last[5]:.6f}')
    print(f'Raw-normal Fx ratio final   : {last[3]:.6f}')
    print(f'PCA-normal Fx ratio final   : {last[6]:.6f}')
    print(f'Raw-normal Hx RMS error     : {raw_hx_err:.6f}')
    print(f'PCA-normal Hx RMS error     : {pca_hx_err:.6f}')
    print(f'Raw-normal Fx RMS error     : {raw_fx_err:.6f}')
    print(f'PCA-normal Fx RMS error     : {pca_fx_err:.6f}')
    print(f'Raw/PCA fold final          : {last[7]:.6f} / {last[8]:.6f}')
    print(f'PCA normal change final deg : mean {last[9]:.3f} / max {last[10]:.3f}')
    print(f'PCA fitted shell seeds final: {int(last[11])}')
    print()
    print('Decision target: is the dp20/dp10 divergence primarily caused by noisy')
    print('anchor-normal directions, or does it remain when ONLY the normal is')
    print('replaced by a local geometric tangent-plane fit?')


def gate_pca_normals(raw_normals, pca_normals, shell_mask, gate_deg):
    """Diagnostic-only angular gate around the already validated raw AIR normal."""
    raw = np.asarray(raw_normals, dtype=np.float64)
    pca = np.asarray(pca_normals, dtype=np.float64)
    shell = np.asarray(shell_mask, dtype=bool)
    out = raw.copy()
    idx = np.flatnonzero(shell)
    if len(idx) == 0:
        return out, 0, 0, 0.0, 0.0
    dots = np.einsum("ij,ij->i", raw[idx], pca[idx])
    dots = np.clip(dots, -1.0, 1.0)
    ang = np.degrees(np.arccos(dots))
    changed = ang > 1.0e-6
    accepted = changed & (ang <= float(gate_deg) + 1.0e-12)
    if np.any(accepted):
        out[idx[accepted]] = pca[idx[accepted]]
    n_changed = int(np.count_nonzero(changed))
    n_accepted = int(np.count_nonzero(accepted))
    mean_acc = float(np.mean(ang[accepted])) if n_accepted else 0.0
    max_acc = float(np.max(ang[accepted])) if n_accepted else 0.0
    return out, n_changed, n_accepted, mean_acc, max_acc


def run_anchor_normal_gate_replay(usd_file, npz_file):
    """Replay v0.49 frames and sweep ONLY an angular acceptance gate on v0.51 PCA normals."""
    data = np.load(npz_file, allow_pickle=False)
    positions = np.asarray(data["positions"], dtype=np.float64)
    rho_frames = np.asarray(data["rho"], dtype=np.float64)
    times = np.asarray(data["times"], dtype=np.float64)
    dp = float(np.asarray(data["dp"]))
    surface_level = float(np.asarray(data["surface_level"]))

    cfg = SPHConfig(); cfg.dp = dp
    kernel = WendlandC2(cfg.h)
    scene = load_scene(usd_file)
    solid_triangles, solid_normals, _solid_paths = collider_triangle_data(scene)
    if len(solid_triangles) == 0:
        raise RuntimeError("No valid collider triangles available.")
    surface_boundary = build_boundary_particles(scene, spacing=cfg.dp)
    boundary = build_layered_wall_particles(surface_boundary, cfg.dp, cfg.support, cfg.rho0)

    pos0 = positions[0]
    initial_floor_z = float(np.min(pos0[:, 2]) - 0.5 * cfg.dp)
    initial_height = float(surface_level - initial_floor_z)
    initial_width = float(np.max(pos0[:, 1]) - np.min(pos0[:, 1]) + cfg.dp)
    gmag = float(np.linalg.norm(np.asarray(cfg.gravity, dtype=np.float64)))
    Ax_ref = initial_width * initial_height
    Hx_ref = 0.5 * initial_width * initial_height**2
    depth_grid = np.linspace(0.0, initial_height, 20001)
    ratio_grid = (1.0 + (cfg.gamma - 1.0) * gmag * depth_grid / (cfg.c0 * cfg.c0)) ** (1.0 / (cfg.gamma - 1.0))
    p_grid = cfg.tait_B * (ratio_grid**cfg.gamma - 1.0)
    Fx_ref = initial_width * float(np.trapezoid(p_grid, depth_grid))

    gates = [10.0, 20.0, 30.0, 45.0, 60.0, 75.0, 90.0]
    stats = {g: [] for g in gates}
    raw_rows, full_rows = [], []
    final_accept = {}

    print()
    print("========================================")
    print("GraSPH")
    print("ANGLE-GATED ANCHOR-NORMAL REPLAY v0.52")
    print("========================================")
    print()
    print("Physics       : OFF (saved-frame replay only)")
    print(f"Input NPZ     : {npz_file}")
    print(f"dp/h/support  : {cfg.dp:.4f} / {cfg.h:.4f} / {cfg.support:.4f} m")
    print("Shell/band    : v0.49 reconstruction (UNCHANGED)")
    print("PCA fit       : v0.51 local tangent-plane fit (UNCHANGED)")
    print("Candidate     : accept PCA only if angle(raw,PCA) <= gate; else RAW")
    print("Only variable : angular acceptance gate (diagnostic sweep)")
    print(f"Reference Ax  : {Ax_ref:.8f} m2")
    print(f"Reference Hx  : {Hx_ref:.8e} m3")
    print(f"Reference Fx  : {Fx_ref:.6f} N")
    print()
    print("Per-frame: raw Hx | full-PCA Hx | gate20/30/45 Hx")

    for fi, (t, pos, rho) in enumerate(zip(times, positions, rho_frames)):
        n = len(pos)
        rho_ref, _p_ref, _depth_ref = hydrostatic_reference_state(pos, surface_level, cfg, return_depth=True)
        pair_i, pair_j = build_neighbor_pairs(pos, cfg.support)
        _rij, _r, gradW = compute_pair_geometry(pos, pair_i, pair_j, kernel)
        G_fluid = compute_fluid_support_G(n, rho_ref, cfg.mass, pair_i, pair_j, gradW)
        missing_fluid = -G_fluid
        geom_cov, geom_tri_count, _geom_min_dist = compute_geometry_normal_covariance(pos, solid_triangles, solid_normals, cfg.support)
        geom_active = geom_tri_count > 0
        air_seed, _seed_rank, _seed_blocked = project_missing_from_solids(missing_fluid, geom_cov, geom_active, COLLIDER_NORMAL_EIGEN_REL)
        wall_interactions = build_wall_interactions(pos, boundary, kernel, cfg.support)
        G_wall_wet, _wet_pairs = build_wetted_wall_support_G(pos, boundary, wall_interactions, air_seed, cfg)
        missing_completed = -(G_fluid + G_wall_wet)
        air_missing, _geom_rank, _blocked_frac = project_missing_from_solids(missing_completed, geom_cov, geom_active, COLLIDER_NORMAL_EIGEN_REL)
        support_hG = cfg.h * np.linalg.norm(air_missing, axis=1)
        shell = support_hG > CURRENT_SURFACE_HG_THRESHOLD
        band, anchor, raw_normals = build_seed_connected_band_with_anchor(pos, pair_i, pair_j, air_missing, shell)
        _acc, _p_s, probe, _valid = anchored_band_traction(pos, rho, air_missing, band, anchor, raw_normals, surface_level, cfg)

        raw_half = analytic_halfspace_surface_moment_diagnostic(pos, rho, band, shell, anchor, raw_normals, probe, surface_level, cfg)
        pca_normals, fitted, mean_ang, max_ang = smooth_shell_normals_pca(pos, pair_i, pair_j, shell, raw_normals, cfg.support)
        full_half = analytic_halfspace_surface_moment_diagnostic(pos, rho, band, shell, anchor, pca_normals, probe, surface_level, cfg)
        raw_hx = raw_half["all"]["Hx"] / Hx_ref
        full_hx = full_half["all"]["Hx"] / Hx_ref
        raw_rows.append(raw_hx); full_rows.append(full_hx)

        frame_hx = {}
        for g in gates:
            gn, n_changed, n_acc, mean_acc, max_acc = gate_pca_normals(raw_normals, pca_normals, shell, g)
            gh = analytic_halfspace_surface_moment_diagnostic(pos, rho, band, shell, anchor, gn, probe, surface_level, cfg)
            a = gh["all"]
            hx = a["Hx"] / Hx_ref
            fx = a["Fx"] / Fx_ref
            ax = a["Ax"] / Ax_ref
            fold = a["Ax_abs"] / max(abs(a["Ax"]), 1e-30)
            stats[g].append((hx, fx, ax, fold, n_changed, n_acc, mean_acc, max_acc))
            frame_hx[g] = hx
            if fi == len(times)-1:
                final_accept[g] = (n_changed, n_acc, mean_acc, max_acc)

        print(f"{t:5.3f} shell={np.count_nonzero(shell):4d} band={np.count_nonzero(band):4d} | raw {raw_hx:6.3f} | PCA {full_hx:6.3f} | g20 {frame_hx[20.0]:6.3f} g30 {frame_hx[30.0]:6.3f} g45 {frame_hx[45.0]:6.3f} | dN {mean_ang:4.1f}/{max_ang:4.1f}")

    print()
    print("========================================")
    print("GATE SWEEP SUMMARY")
    print("========================================")
    raw_rms = float(np.sqrt(np.mean((np.asarray(raw_rows)-1.0)**2)))
    full_rms = float(np.sqrt(np.mean((np.asarray(full_rows)-1.0)**2)))
    print(f"RAW baseline : final Hx={raw_rows[-1]:.6f} | RMS={raw_rms:.6f}")
    print(f"FULL PCA     : final Hx={full_rows[-1]:.6f} | RMS={full_rms:.6f}")
    print(" gate | final Hx | Hx RMS | final Fx | final Ax | fold | accepted changed seeds")
    for g in gates:
        arr = np.asarray(stats[g], dtype=np.float64)
        hx_rms = float(np.sqrt(np.mean((arr[:,0]-1.0)**2)))
        final = arr[-1]
        nc, na, ma, xa = final_accept[g]
        print(f"{g:5.0f} | {final[0]:8.4f} | {hx_rms:6.4f} | {final[1]:8.4f} | {final[2]:8.4f} | {final[3]:5.3f} | {na:4d}/{nc:4d} (mean/max {ma:4.1f}/{xa:4.1f} deg)")
    print()
    print("Decision target: are the v0.51 failures dominated by large-angle PCA")
    print("outliers, such that a bounded correction improves BOTH dp20 and dp10?")
    print("If no single gate improves both resolutions, abandon angle gating as a")
    print("production fix and move to a conservative surface quadrature/reconstruction.")

def compute_support_metrics(
    n,
    rho_ref,
    mass,
    pair_i,
    pair_j,
    rij,
    gradW,
    boundary,
    wall_interactions,
):
    """Return first-moment completeness C and kernel-gradient imbalance G."""
    C = np.zeros(n, dtype=np.float64)
    G = np.zeros((n, 3), dtype=np.float64)

    if len(pair_i):
        trace_pair = -np.einsum("ij,ij->i", rij, gradW) / 3.0
        Vi = mass / np.maximum(rho_ref[pair_i], 1.0e-30)
        Vj = mass / np.maximum(rho_ref[pair_j], 1.0e-30)
        np.add.at(C, pair_i, Vj * trace_pair)
        np.add.at(C, pair_j, Vi * trace_pair)
        np.add.at(G, pair_i, Vj[:, None] * gradW)
        np.add.at(G, pair_j, -Vi[:, None] * gradW)

    wi = wall_interactions["pair_i"]
    wj = wall_interactions["pair_j"]
    if len(wi):
        wr = wall_interactions["rij"]
        wg = wall_interactions["gradW"]
        Vb = boundary.volumes[wj]
        trace_wall = -np.einsum("ij,ij->i", wr, wg) / 3.0
        np.add.at(C, wi, Vb * trace_wall)
        np.add.at(G, wi, Vb[:, None] * wg)

    return C, G

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  py grasph.py DamBreakRitter.usda output\\ritter_surface_moment_v49_dp20.npz")
        print("  py grasph.py DamBreakRitter.usda output\\ritter_surface_moment_v49_dp10.npz")
        print("Legacy full-physics v0.49 mode is also retained with numeric dp:")
        print("  py grasph.py DamBreakRitter.usda 0.020")
        return

    usd_file = sys.argv[1]
    if len(sys.argv) >= 3 and sys.argv[2].lower().endswith(".npz"):
        run_anchor_normal_gate_replay(usd_file, sys.argv[2])
        return
    test_dp = float(sys.argv[2]) if len(sys.argv) >= 3 else DEFAULT_DP
    if test_dp <= 0.0:
        raise ValueError("dp must be > 0")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cfg = SPHConfig()
    # v0.48 changes NO physics. dp is only a selectable diagnostic input so
    # v0.45 and v0.46 can be compared with identical force bookkeeping.
    cfg.dp = test_dp

    expected_h = H_FACTOR_EXPECTED * cfg.dp
    if not np.isclose(cfg.h, expected_h, rtol=0.0, atol=1.0e-12):
        raise RuntimeError(
            f"SPHConfig h/dp mismatch: h={cfg.h:.9g}, "
            f"expected {expected_h:.9g} for h/dp={H_FACTOR_EXPECTED:.3f}"
        )

    tag_mm = int(round(1000.0 * cfg.dp))
    NPZ_OUTPUT = os.path.join(OUTPUT_DIR, f"ritter_surface_moment_v49_dp{tag_mm:02d}.npz")
    USD_OUTPUT = os.path.join(OUTPUT_DIR, f"ritter_surface_moment_v49_dp{tag_mm:02d}.usdc")

    kernel = WendlandC2(cfg.h)
    particle_radius = PARTICLE_RADIUS_FACTOR * cfg.dp

    print()
    print("========================================")
    print("GraSPH")
    print("WCSPH RITTER PROJECTED-SURFACE MOMENT DIAGNOSTIC v0.49")
    print("========================================")
    print()
    print("SPH configuration:")
    print(f"  dp         : {cfg.dp:.4f} m")
    print(f"  h          : {cfg.h:.4f} m")
    print(f"  support    : {cfg.support:.4f} m")
    print(f"  mass       : {cfg.mass:.6f} kg")
    print(f"  diagnostic dp : {cfg.dp:.4f} m (physics otherwise unchanged)")
    print(f"  rho0       : {cfg.rho0:.1f} kg/m3")
    print(f"  c0         : {cfg.c0:.1f} m/s")
    print(f"  gamma      : {cfg.gamma:.1f}")
    print(f"  Tait B     : {cfg.tait_B:.1f} Pa")
    print(f"  radius     : {particle_radius:.4f} m")
    print(f"  viscosity α: {ARTIFICIAL_VISCOSITY_ALPHA:.3f}")
    print(f"  pressure   : {PRESSURE_OPERATOR_MODE}")
    print(f"  wall press.: {ENABLE_WALL_PRESSURE}")
    print(f"  wall cont. : {ENABLE_WALL_CONTINUITY}")
    print(f"  fluid state: {FLUID_STATE_MODE}")
    print(f"  collision   : {ENABLE_COLLISION}")
    print(f"  wall mode   : {WALL_MODE}")
    print(f"  integrator  : {INTEGRATOR_MODE}")
    print(f"  surface air h|G| : > {CURRENT_SURFACE_HG_THRESHOLD:.3f}")
    print(f"  collider eig rel  : {COLLIDER_NORMAL_EIGEN_REL:.3f}")

    # --------------------------------------------------------
    # Scene + fluid spawn
    # --------------------------------------------------------
    scene = load_scene(usd_file)
    solid_triangles, solid_normals, solid_paths = collider_triangle_data(scene)
    if len(solid_triangles) == 0:
        raise RuntimeError("No valid collider triangles available for AIR-vs-SOLID classification.")
    pos = generate_fluid_particles(scene, dp=cfg.dp)
    n = len(pos)
    if n == 0:
        raise RuntimeError("No fluid particles generated.")

    vel = np.zeros_like(pos, dtype=np.float64)

    # Reference free-surface plane measured along -gravity.
    gvec = np.asarray(cfg.gravity, dtype=np.float64)
    gmag = float(np.linalg.norm(gvec))
    if gmag > 1.0e-15:
        up = -gvec / gmag
        surface_level = float(np.max(pos @ up) + 0.5 * cfg.dp)
    else:
        surface_level = 0.0

    rho_ref, p_ref, depth_ref = hydrostatic_reference_state(
        pos, surface_level, cfg, return_depth=True
    )

    # Actual WCSPH density starts in the exact hydrostatic reference state.
    # No acoustic seed and no external forcing. The exposed downstream
    # vertical free face is the only source of motion.
    rho = rho_ref.copy()
    pressure = compute_pressure(rho, cfg.rho0, cfg.tait_B, cfg.gamma)
    q = pressure - p_ref

    print()
    print("Exact Tait hydrostatic initialization:")
    print(f"  surface level : {surface_level:.4f} m")
    print(f"  rho ref       : {rho_ref.min():.3f} / {rho_ref.mean():.3f} / {rho_ref.max():.3f} kg/m3")
    print(f"  q min/max     : {q.min():+.6e} / {q.max():+.6e} Pa")

    # --------------------------------------------------------
    # Geometry collision + fixed layered SPH wall support
    # --------------------------------------------------------
    colliders = build_colliders(
        scene,
        voxel_size=COLLISION_VOXEL_FACTOR * cfg.dp,
        margin=COLLISION_MARGIN_FACTOR * cfg.dp,
    )

    surface_boundary = build_boundary_particles(scene, spacing=cfg.dp)
    boundary = build_layered_wall_particles(
        surface_boundary, cfg.dp, cfg.support, cfg.rho0
    )

    # All solids are static in v0.37.  Wall continuity therefore sees
    # v_b=0 everywhere and collision uses the untransformed distance fields.
    wall_velocity = np.zeros_like(boundary.positions, dtype=np.float64)

    wall_rho_ref, wall_p_ref, _wall_depth_ref = hydrostatic_reference_state(
        boundary.positions, surface_level, cfg, return_depth=True
    )

    # Initial AIR-facing signal. First build the proven fluid-only AIR seed,
    # then add only WET-side wall support and re-project the completed support
    # against the exact finite collider-normal span. The high-confidence shell
    # remains h|G_air|>0.35; traction is distributed to one inward SPH layer.
    pair_i0, pair_j0 = build_neighbor_pairs(pos, cfg.support)
    _rij0, _r0, gradW0 = compute_pair_geometry(pos, pair_i0, pair_j0, kernel)
    G_fluid_init = compute_fluid_support_G(
        n, rho_ref, cfg.mass, pair_i0, pair_j0, gradW0
    )
    missing_fluid_init = -G_fluid_init
    raw_fluid_hG_init = cfg.h * np.linalg.norm(missing_fluid_init, axis=1)
    geom_cov_init, geom_tri_count_init, _geom_min_dist_init = compute_geometry_normal_covariance(
        pos, solid_triangles, solid_normals, cfg.support
    )
    geom_active_init = geom_tri_count_init > 0
    air_seed_init, _seed_rank_init, _seed_blocked_init = project_missing_from_solids(
        missing_fluid_init, geom_cov_init, geom_active_init, COLLIDER_NORMAL_EIGEN_REL
    )
    wall_interactions_init = build_wall_interactions(pos, boundary, kernel, cfg.support)
    G_wall_wet_init, wet_pairs_init = build_wetted_wall_support_G(
        pos, boundary, wall_interactions_init, air_seed_init, cfg
    )
    missing_completed_init = -(G_fluid_init + G_wall_wet_init)
    air_missing_init, geom_rank_init, blocked_frac_init = project_missing_from_solids(
        missing_completed_init, geom_cov_init, geom_active_init, COLLIDER_NORMAL_EIGEN_REL
    )
    support_hG_init = cfg.h * np.linalg.norm(air_missing_init, axis=1)
    surface_mask_init = support_hG_init > CURRENT_SURFACE_HG_THRESHOLD
    traction_band_init, anchor_init, normals_init = build_seed_connected_band_with_anchor(
        pos, pair_i0, pair_j0, air_missing_init, surface_mask_init
    )

    print()
    print("SPH wall boundary:")
    print(f"  surface samples : {len(surface_boundary.positions)}")
    print(f"  volume particles: {len(boundary.positions)}")
    print(f"  Vb              : {boundary.volumes[0]:.6e} m3")
    print(f"  mb @ rho0       : {cfg.rho0 * boundary.volumes[0]:.6e} kg")
    initial_front_x = float(np.max(pos[:, 0]) + 0.5 * cfg.dp)
    initial_rear_x = float(np.min(pos[:, 0]) - 0.5 * cfg.dp)
    initial_floor_z = float(np.min(pos[:, 2]) - 0.5 * cfg.dp)
    initial_height = float(surface_level - initial_floor_z)
    gmag_bench = float(np.linalg.norm(np.asarray(cfg.gravity, dtype=np.float64)))
    shallow_c = float(np.sqrt(max(gmag_bench * initial_height, 0.0)))
    ritter_front_speed = 2.0 * shallow_c
    reservoir_length = initial_front_x - initial_rear_x
    rear_wall_time = reservoir_length / shallow_c if shallow_c > 1.0e-30 else np.inf
    initial_width = float(np.max(pos[:, 1]) - np.min(pos[:, 1]) + cfg.dp)
    total_mass = float(n * cfg.mass)
    force_ref_incompressible = 0.5 * cfg.rho0 * gmag_bench * initial_width * initial_height**2
    depth_grid = np.linspace(0.0, initial_height, 20001)
    ratio_grid = (
        1.0 + (cfg.gamma - 1.0) * gmag_bench * depth_grid / (cfg.c0 * cfg.c0)
    ) ** (1.0 / (cfg.gamma - 1.0))
    p_grid = cfg.tait_B * (ratio_grid**cfg.gamma - 1.0)
    force_ref_tait = initial_width * float(np.trapezoid(p_grid, depth_grid))
    projected_area_x_ref = initial_width * initial_height
    projected_depth_moment_x_ref = 0.5 * initial_width * initial_height**2

    print()
    print("Ritter dry-bed dam-break benchmark:")
    print(f"  initial rear X  : {initial_rear_x:+.4f} m")
    print(f"  initial front X : {initial_front_x:+.4f} m")
    print(f"  initial h0      : {initial_height:.4f} m")
    print(f"  shallow c       : {shallow_c:.4f} m/s")
    print(f"  Ritter speed    : {ritter_front_speed:.4f} m/s")
    print(f"  rear-wall time  : {rear_wall_time:.4f} s")
    print(f"  simulated time  : {SIM_TIME:.4f} s")
    print(f"  fluid mass      : {total_mass:.6f} kg")
    print(f"  force ref incompr: {force_ref_incompressible:.6f} N")
    print(f"  force ref Tait   : {force_ref_tait:.6f} N")
    print(f"  projected Ax ref : {projected_area_x_ref:.6f} m2")
    print(f"  depth moment Hx ref: {projected_depth_moment_x_ref:.6e} m3")
    print("  v0.49 question   : does diffuse traction preserve projected area/moment?")
    print("  benchmark focus : global X-momentum / force decomposition")
    print("  external drive  : NONE")
    print("  solid walls     : static")
    print(f"  collider tris   : {len(solid_triangles)}")
    print("  detector seed   : FLUID-only AIR support")
    print("  support complete: wet-side SPH wall samples")
    print("  traction band   : shell + one inward SPH layer")
    print("  interface probe : inherited from nearest compatible shell seed")
    print("  solid model     : finite-face projection collider-normal span")
    print(f"  collider eig rel: {COLLIDER_NORMAL_EIGEN_REL:.3f}")
    print(f"  surface cutoff  : air h*|G| > {CURRENT_SURFACE_HG_THRESHOLD:.3f}")
    print(f"  surface probe   : {SURFACE_PROBE_OFFSET_FACTOR:.2f} dp outward")
    print("  atmosphere BC   : wetted-wall completed, anchored traction-band support")

    # --------------------------------------------------------
    # Output snapshots
    # --------------------------------------------------------
    frame_positions = [pos.copy()]
    frame_rho = [rho.copy()]
    frame_rho_ref = [rho_ref.copy()]
    frame_q = [q.copy()]
    frame_q_force = [q.copy()]
    frame_surface_hG = [support_hG_init.copy()]
    frame_raw_fluid_hG = [raw_fluid_hG_init.copy()]
    frame_geom_rank = [geom_rank_init.copy()]
    frame_blocked_frac = [blocked_frac_init.copy()]
    frame_atmospheric = [surface_mask_init.astype(np.uint8).copy()]
    frame_traction_band = [traction_band_init.astype(np.uint8).copy()]
    frame_velocity = [vel.copy()]
    frame_times = [0.0]
    frame_drive_accel = [drive_acceleration(0.0).copy()]

    output_interval = 1.0 / OUTPUT_FPS
    next_output_time = output_interval

    print()
    print("========================================")
    print("STARTING ANCHORED-BAND RITTER WCSPH")
    print("========================================")
    print()

    sim_start = time.perf_counter()
    last_report = sim_start
    t = 0.0
    step = 0
    total_contacts = 0
    initial_diag_printed = False

    # Diagnostic tracking of the CURRENT surface mask. The mask itself drives
    # only the one-sided atmospheric traction; history never feeds back.
    prev_atmospheric = surface_mask_init.copy()
    ever_atmospheric = surface_mask_init.copy()
    atm_new_since_report = 0
    atm_lost_since_report = 0

    # Diagnostic-only momentum bookkeeping; never feeds back into physics.
    J_atm_x = J_qff_x = J_qwall_x = 0.0
    J_visc_x = J_base_x = J_collision_x = 0.0
    report_J_atm_x = report_J_qff_x = report_J_qwall_x = 0.0
    report_J_visc_x = report_J_base_x = report_J_collision_x = 0.0
    # v0.48 diagnostic-only atmospheric traction split by anchor-plane depth.
    report_J_depth_l1_x = report_J_depth_l2_x = report_J_depth_l3_x = 0.0
    report_J_depth_out_x = report_J_depth_ahead_x = 0.0
    report_dt_sim = 0.0

    while t < SIM_TIME:
        dt = min(DT_MAX, compute_cfl_dt(vel, cfg.h, cfg.c0, cfl=CFL))
        if t + dt > SIM_TIME:
            dt = SIM_TIME - t
        # Land exactly on the start/end of the smooth forcing window.
        for drive_event_time in (DRIVE_START_TIME, DRIVE_END_TIME):
            if t < drive_event_time < t + dt:
                dt = drive_event_time - t
                break

        drive_acc = drive_acceleration(t)

        # Static solid walls: no translations and v_b=0.
        wall_velocity.fill(0.0)

        # Same wall quadrature mass is used by pressure and continuity.
        wall_rho_ref, wall_p_ref, _wall_depth_ref = hydrostatic_reference_state(
            boundary.positions, surface_level, cfg, return_depth=True
        )

        pair_i, pair_j = build_neighbor_pairs(pos, cfg.support)
        rij, r, gradW = compute_pair_geometry(pos, pair_i, pair_j, kernel)

        # Current analytical background at CURRENT particle positions.
        rho_ref, p_ref, depth_ref = hydrostatic_reference_state(
            pos, surface_level, cfg, return_depth=True
        )

        # Signed reduced pressure from continuity density. Do NOT clip the EOS.
        pressure = compute_pressure(rho, cfg.rho0, cfg.tait_B, cfg.gamma)
        q = pressure - p_ref

        if ENABLE_WALL_PRESSURE or ENABLE_WALL_CONTINUITY:
            wall_interactions = build_wall_interactions(
                pos, boundary, kernel, cfg.support
            )
        else:
            wall_interactions = {
                "pair_i": np.empty(0, dtype=np.int32),
                "pair_j": np.empty(0, dtype=np.int32),
                "rij": np.empty((0, 3), dtype=np.float64),
                "gradW": np.empty((0, 3), dtype=np.float64),
            }

        # AIR seed from fluid-only support, as in v0.37. Exact finite collider
        # faces remove SOLID-normal modes; this seed is used only to define the
        # local wet/dry wall plane for support completion.
        G_fluid = compute_fluid_support_G(
            n, rho_ref, cfg.mass, pair_i, pair_j, gradW
        )
        missing_fluid = -G_fluid
        raw_fluid_hG = cfg.h * np.linalg.norm(missing_fluid, axis=1)
        geom_cov, geom_tri_count, _geom_min_dist = compute_geometry_normal_covariance(
            pos, solid_triangles, solid_normals, cfg.support
        )
        geom_active = geom_tri_count > 0
        air_seed, _seed_rank, _seed_blocked = project_missing_from_solids(
            missing_fluid, geom_cov, geom_active, COLLIDER_NORMAL_EIGEN_REL
        )

        # v0.41: complete only the WET side of solid-truncated kernel support.
        G_wall_wet, wet_pairs = build_wetted_wall_support_G(
            pos, boundary, wall_interactions, air_seed, cfg
        )
        missing_completed = -(G_fluid + G_wall_wet)
        air_missing, geom_rank, blocked_frac = project_missing_from_solids(
            missing_completed, geom_cov, geom_active, COLLIDER_NORMAL_EIGEN_REL
        )
        air_missing_mag = np.linalg.norm(air_missing, axis=1)
        support_hG = cfg.h * air_missing_mag

        # High-confidence visible surface shell remains unchanged. Traction,
        # however, occupies shell + one compatible inward SPH layer (v0.42).
        atmospheric = support_hG > CURRENT_SURFACE_HG_THRESHOLD
        traction_band, traction_anchor, surface_normal = build_seed_connected_band_with_anchor(
            pos, pair_i, pair_j, air_missing, atmospheric
        )

        newly_atmospheric = atmospheric & (~prev_atmospheric)
        lost_atmospheric = prev_atmospheric & (~atmospheric)
        atm_new_since_report += int(np.count_nonzero(newly_atmospheric))
        atm_lost_since_report += int(np.count_nonzero(lost_atmospheric))
        ever_atmospheric |= atmospheric
        prev_atmospheric = atmospheric.copy()

        # Keep the stable bulk reduced-pressure state untouched.
        q_force = q.copy()

        # v0.44: every inner-band particle inherits the physical interface
        # probe from its nearest compatible shell seed. This preserves
        # hydrostatic neutrality at a horizontal free surface.
        acc_atm, p_ref_surface, surface_probe, valid_band = anchored_band_traction(
            pos, rho, air_missing, traction_band, traction_anchor,
            surface_normal, surface_level, cfg
        )

        # Energy-matched solid wall carries q_b=0.
        q_wall = np.zeros(len(boundary.positions), dtype=np.float64)

        # ----------------------------------------------------
        # Reduced-pressure pairwise force
        #
        # This is the standard symmetric SPH pressure operator, but applied
        # only to q = p - p_ref.  It is the conservative counterpart of the
        # continuity equation above and vanishes identically at hydrostatic
        # equilibrium where q=0.
        # ----------------------------------------------------
        acc_q_ff = compute_pressure_acceleration(
            rho,
            q_force,
            pair_i,
            pair_j,
            gradW,
            cfg.mass,
        )
        acc_q = acc_q_ff.copy()

        # Wall layers use q_b=0. Keep the wall contribution separate for the
        # v0.48 global momentum budget; actual physics is unchanged.
        if ENABLE_WALL_PRESSURE:
            add_static_wall_pressure_acceleration(
                acc_q,
                rho,
                q_force,
                boundary,
                wall_interactions,
                wall_rho_ref,
                q_wall,
            )
        acc_q_wall = acc_q - acc_q_ff

        # Exact gravity + hydrostatic-background contribution.
        acc_base = compute_well_balanced_base_acceleration(
            rho, rho_ref, depth_ref, cfg.gravity
        )

        acc_visc = compute_artificial_viscosity_acceleration(
            pos,
            vel,
            rho,
            pair_i,
            pair_j,
            rij,
            r,
            gradW,
            cfg.mass,
            cfg.h,
            cfg.c0,
            alpha=ARTIFICIAL_VISCOSITY_ALPHA,
        )

        acc = acc_q + acc_base + acc_visc + acc_atm + drive_acc[None, :]

        # v0.49 diagnostic-only X-force budget.
        Fx_atm = cfg.mass * float(np.sum(acc_atm[:, 0]))
        Fx_qff = cfg.mass * float(np.sum(acc_q_ff[:, 0]))
        Fx_qwall = cfg.mass * float(np.sum(acc_q_wall[:, 0]))
        Fx_visc = cfg.mass * float(np.sum(acc_visc[:, 0]))
        Fx_base = cfg.mass * float(np.sum(acc_base[:, 0]))

        # v0.48 carry-over diagnostic: where, in physical normal distance from the
        # inherited interface plane, is atmospheric Fx being carried?
        depth_diag = traction_band_depth_diagnostic(
            pos, acc_atm, traction_band, traction_anchor,
            surface_normal, surface_probe, cfg
        )
        surf_moment_diag = traction_surface_moment_diagnostic(
            rho, air_missing, traction_band, atmospheric, traction_anchor,
            surface_probe, surface_level, cfg
        )
        report_J_depth_l1_x += depth_diag["Fx_l1"] * dt
        report_J_depth_l2_x += depth_diag["Fx_l2"] * dt
        report_J_depth_l3_x += depth_diag["Fx_l3"] * dt
        report_J_depth_out_x += depth_diag["Fx_outside"] * dt
        report_J_depth_ahead_x += depth_diag["Fx_ahead"] * dt

        J_atm_x += Fx_atm * dt
        J_qff_x += Fx_qff * dt
        J_qwall_x += Fx_qwall * dt
        J_visc_x += Fx_visc * dt
        J_base_x += Fx_base * dt
        report_J_atm_x += Fx_atm * dt
        report_J_qff_x += Fx_qff * dt
        report_J_qwall_x += Fx_qwall * dt
        report_J_visc_x += Fx_visc * dt
        report_J_base_x += Fx_base * dt
        report_dt_sim += dt

        if not initial_diag_printed:
            amag = np.linalg.norm(acc, axis=1)
            qamag = np.linalg.norm(acc_q, axis=1)
            bamag = np.linalg.norm(acc_base, axis=1)
            atmag = np.linalg.norm(acc_atm, axis=1)
            horizontal = np.linalg.norm(acc[:, :2], axis=1)
            delta_rho = rho - rho_ref

            print("========================================")
            print("INITIAL RITTER PROJECTED-SURFACE MOMENT DIAGNOSTIC v0.49")
            print("========================================")
            print(f"Pressure operator : {PRESSURE_OPERATOR_MODE}")
            print(f"Fluid state       : {FLUID_STATE_MODE}")
            print(f"Wall pressure     : {ENABLE_WALL_PRESSURE}")
            print(f"Wall continuity   : {ENABLE_WALL_CONTINUITY}")
            print(f"DF collision      : {ENABLE_COLLISION}")
            print(f"Wall mode         : {WALL_MODE}")
            print(f"Wall q_b          : 0 (energy-matched boundary)")
            print(f"Integrator        : {INTEGRATOR_MODE}")
            print("External drive    : NONE")
            print("Benchmark         : Ritter dry-bed shallow-water front")
            print("Front metric      : max particle X + 0.5*dp")
            print(f"Surface threshold : air h*|G| > {CURRENT_SURFACE_HG_THRESHOLD:.3f}")
            print(f"Collider eig rel  : {COLLIDER_NORMAL_EIGEN_REL:.3f}")
            print(f"Surface probe     : {SURFACE_PROBE_OFFSET_FACTOR:.2f} dp outward")
            print("Atmosphere BC     : wetted-wall completed, anchored traction-band support")
            print()
            print(
                f"delta rho min/mean/max : "
                f"{delta_rho.min():+.6e} / {delta_rho.mean():+.6e} / {delta_rho.max():+.6e} kg/m3"
            )
            print(
                f"q EOS min/mean/max     : "
                f"{q.min():+.6e} / {q.mean():+.6e} / {q.max():+.6e} Pa"
            )
            print(
                f"q force min/mean/max   : "
                f"{q_force.min():+.6e} / {q_force.mean():+.6e} / {q_force.max():+.6e} Pa"
            )
            print(f"surface shell particles: {np.count_nonzero(atmospheric)} / {n}")
            print(f"traction band particles: {np.count_nonzero(traction_band)} / {n}")
            print(f"valid band anchors     : {np.count_nonzero(valid_band)} / {np.count_nonzero(traction_band)}")
            print(f"wet wall pairs kept    : {np.count_nonzero(wet_pairs)} / {len(wet_pairs)}")
            print(
                f"raw fluid h|G| min/max  : "
                f"{raw_fluid_hG.min():+.6e} / {raw_fluid_hG.max():+.6e}"
            )
            print(
                f"air h|G| min/max        : "
                f"{support_hG.min():+.6e} / {support_hG.max():+.6e}"
            )
            raw_candidates = raw_fluid_hG > CURRENT_SURFACE_HG_THRESHOLD
            blocked_candidates = raw_candidates & (~atmospheric) & geom_active
            print(f"solid-blocked raw cand. : {np.count_nonzero(blocked_candidates)}")
            if np.any(geom_active):
                rr = geom_rank[geom_active]
                print(
                    f"geom ranks 1/2/3       : "
                    f"{np.count_nonzero(rr==1)}/{np.count_nonzero(rr==2)}/{np.count_nonzero(rr==3)}"
                )
            if np.any(traction_band):
                print(
                    f"band p_ref mean/max     : "
                    f"{np.mean(p_ref_surface[traction_band]):.6e} / {np.max(p_ref_surface[traction_band]):.6e} Pa"
                )
            print(
                f"|a_q| mean/max         : "
                f"{qamag.mean():.6e} / {qamag.max():.6e} m/s2"
            )
            print(
                f"|a_base| mean/max      : "
                f"{bamag.mean():.6e} / {bamag.max():.6e} m/s2"
            )
            print(
                f"|a_atm| mean/max       : "
                f"{atmag.mean():.6e} / {atmag.max():.6e} m/s2"
            )
            print(
                f"|a| min/mean/max       : "
                f"{amag.min():.6e} / {amag.mean():.6e} / {amag.max():.6e} m/s2"
            )
            print(
                f"|axy| mean/max         : "
                f"{horizontal.mean():.6e} / {horizontal.max():.6e} m/s2"
            )
            print(f"Fx atm                 : {Fx_atm:+.6f} N")
            print(f"Fx q fluid-fluid       : {Fx_qff:+.6e} N")
            print(f"Fx q wall              : {Fx_qwall:+.6f} N")
            print(f"Fx viscosity           : {Fx_visc:+.6e} N")
            print(f"Fx base                : {Fx_base:+.6e} N")
            print(f"Fx kick total          : {Fx_atm+Fx_qff+Fx_qwall+Fx_visc+Fx_base:+.6f} N")
            print(f"Fx Tait reference      : {force_ref_tait:+.6f} N")
            print(
                "band depth count L1/L2/L3/OUT/AHEAD: "
                f"{np.count_nonzero(depth_diag['l1'])}/"
                f"{np.count_nonzero(depth_diag['l2'])}/"
                f"{np.count_nonzero(depth_diag['l3'])}/"
                f"{np.count_nonzero(depth_diag['outside'])}/"
                f"{np.count_nonzero(depth_diag['ahead'])}"
            )
            print(
                "band depth Fx [N] L1/L2/L3/OUT/AHEAD: "
                f"{depth_diag['Fx_l1']:+.6f}/"
                f"{depth_diag['Fx_l2']:+.6f}/"
                f"{depth_diag['Fx_l3']:+.6f}/"
                f"{depth_diag['Fx_outside']:+.6f}/"
                f"{depth_diag['Fx_ahead']:+.6f}"
            )
            sm = surf_moment_diag["all"]
            sms = surf_moment_diag["shell"]
            smi = surf_moment_diag["inner"]
            print(
                f"projected Ax signed/abs : {sm['Ax']:+.6e} / {sm['Ax_abs']:.6e} m2 "
                f"(ref {projected_area_x_ref:.6e}, ratio {sm['Ax']/projected_area_x_ref:.4f})"
            )
            print(
                f"depth moment Hx         : {sm['Hx']:+.6e} m3 "
                f"(ref {projected_depth_moment_x_ref:.6e}, ratio {sm['Hx']/projected_depth_moment_x_ref:.4f})"
            )
            print(
                f"surface-moment Fx       : {sm['Fx']:+.6f} N "
                f"(Tait ratio {sm['Fx']/force_ref_tait:.4f})"
            )
            print(f"shell/inner Ax          : {sms['Ax']:+.6e} / {smi['Ax']:+.6e} m2")
            print(f"shell/inner Fx          : {sms['Fx']:+.6f} / {smi['Fx']:+.6f} N")
            print(f"Ax pos/neg              : {sm['Ax_pos']:+.6e} / {sm['Ax_neg']:+.6e} m2")
            print(f"wall pairs             : {len(wall_interactions['pair_i'])}")
            print()
            initial_diag_printed = True

        # ----------------------------------------------------
        # SYMPLECTIC ACOUSTIC UPDATE
        #
        # Kick velocity first, then evaluate BOTH fluid-fluid and wall
        # continuity using v_{n+1}. All walls use v_b=0. The same wall
        # quadrature mass appears in pressure and continuity, preserving the
        # matched pair from v0.16.
        # ----------------------------------------------------
        vel += acc * dt

        drho = compute_density_rate(
            vel, pair_i, pair_j, gradW, cfg.mass
        )
        if ENABLE_WALL_CONTINUITY:
            add_static_wall_density_rate(
                drho,
                vel,
                boundary,
                wall_interactions,
                wall_rho_ref,
                wall_velocity=wall_velocity,
            )

        rho += drho * dt
        pos += vel * dt

        px_before_collision = cfg.mass * float(np.sum(vel[:, 0]))
        if ENABLE_COLLISION:
            contacts = resolve_collisions(
                pos,
                vel,
                colliders,
                particle_radius=particle_radius,
                restitution=0.0,
                tangential_keep=1.0,
                iterations=COLLISION_ITERATIONS,
            )
        else:
            contacts = 0
        px_after_collision = cfg.mass * float(np.sum(vel[:, 0]))
        dJ_collision_x = px_after_collision - px_before_collision
        J_collision_x += dJ_collision_x
        report_J_collision_x += dJ_collision_x
        total_contacts += contacts

        t += dt
        step += 1

        if not np.all(np.isfinite(pos)):
            raise RuntimeError("Non-finite particle position.")
        if not np.all(np.isfinite(vel)):
            raise RuntimeError("Non-finite particle velocity.")
        if not np.all(np.isfinite(rho)):
            raise RuntimeError("Non-finite density.")

        if rho.min() < 0.5 * cfg.rho0 or rho.max() > 1.5 * cfg.rho0:
            raise RuntimeError(
                f"Density instability detected: {rho.min():.1f} .. {rho.max():.1f} kg/m3"
            )

        if t >= next_output_time:
            rho_ref_snap, p_ref_snap, _ = hydrostatic_reference_state(
                pos, surface_level, cfg, return_depth=True
            )
            p_snap = compute_pressure(rho, cfg.rho0, cfg.tait_B, cfg.gamma)
            q_snap = p_snap - p_ref_snap
            q_force_snap = q_snap.copy()

            frame_positions.append(pos.copy())
            frame_rho.append(rho.copy())
            frame_rho_ref.append(rho_ref_snap.copy())
            frame_q.append(q_snap.copy())
            frame_q_force.append(q_force_snap.copy())
            frame_surface_hG.append(support_hG.copy())
            frame_raw_fluid_hG.append(raw_fluid_hG.copy())
            frame_geom_rank.append(geom_rank.copy())
            frame_blocked_frac.append(blocked_frac.copy())
            frame_atmospheric.append(atmospheric.astype(np.uint8).copy())
            frame_traction_band.append(traction_band.astype(np.uint8).copy())
            frame_velocity.append(vel.copy())
            frame_times.append(t)
            frame_drive_accel.append(drive_acceleration(t).copy())

            while next_output_time <= t:
                next_output_time += output_interval

        now = time.perf_counter()
        report_due = (now - last_report >= REPORT_INTERVAL) or t >= SIM_TIME
        if report_due:
            elapsed = now - sim_start
            progress = t / SIM_TIME
            eta = elapsed * (1.0 - progress) / progress if progress > 0 else 0.0
            rate = step / elapsed if elapsed > 0 else 0.0

            speed = np.linalg.norm(vel, axis=1)
            counts = neighbor_counts(n, pair_i, pair_j)
            rho_ref_report, p_ref_report, _ = hydrostatic_reference_state(
                pos, surface_level, cfg, return_depth=True
            )
            p_report = compute_pressure(rho, cfg.rho0, cfg.tait_B, cfg.gamma)
            q_report = p_report - p_ref_report
            rho_err = rho - rho_ref_report
            drive_report = drive_acceleration(t)
            com = np.mean(pos, axis=0)
            mean_vel = np.mean(vel, axis=0)
            front_x = float(np.max(pos[:, 0]) + 0.5 * cfg.dp)
            ritter_x = float(initial_front_x + ritter_front_speed * t)
            kinetic_energy = 0.5 * cfg.mass * float(
                np.sum(np.einsum("ij,ij->i", vel, vel))
            )
            atmospheric_count = int(np.count_nonzero(atmospheric))
            traction_band_count = int(np.count_nonzero(traction_band))
            ever_atmospheric_count = int(np.count_nonzero(ever_atmospheric))
            if atmospheric_count:
                hG_active_mean = float(np.mean(support_hG[atmospheric]))
                hG_active_max = float(np.max(support_hG[atmospheric]))
                good = air_missing_mag[atmospheric] > 1.0e-30
                if np.any(good):
                    surf_dir = surface_normal[atmospheric][good]
                    surface_nx = float(np.mean(surf_dir[:, 0]))
                else:
                    surface_nx = 0.0
                if traction_band_count:
                    surf_p_mean = float(np.mean(p_ref_surface[traction_band]))
                    surf_p_max = float(np.max(p_ref_surface[traction_band]))
                    atm_mag_active = np.linalg.norm(acc_atm[traction_band], axis=1)
                    atm_acc_mean = float(np.mean(atm_mag_active))
                    atm_acc_max = float(np.max(atm_mag_active))
                    atm_x = float(np.mean(acc_atm[traction_band, 0]))
                else:
                    surf_p_mean = surf_p_max = atm_acc_mean = atm_acc_max = atm_x = 0.0
            else:
                hG_active_mean = 0.0
                hG_active_max = 0.0
                surface_nx = 0.0
                surf_p_mean = 0.0
                surf_p_max = 0.0
                atm_acc_mean = 0.0
                atm_acc_max = 0.0
                atm_x = 0.0

            print(
                f"[{format_time(elapsed)}] "
                f"step {step:5d} "
                f"| {progress*100:5.1f}% "
                f"| sim {t:.4f}/{SIM_TIME:.4f}s "
                f"| dt={dt:.7f} "
                f"| {rate:5.1f} step/s "
                f"| ETA {format_time(eta)}"
            )
            raw_candidates_report = raw_fluid_hG > CURRENT_SURFACE_HG_THRESHOLD
            blocked_candidates_report = raw_candidates_report & (~atmospheric) & geom_active

            print(
                f"    neigh={counts.min()}/{counts.mean():.1f}/{counts.max()} "
                f"| rho={rho.min():.2f}/{rho.mean():.2f}/{rho.max():.2f} "
                f"| dRho={np.mean(np.abs(rho_err)):.4f}/{np.max(np.abs(rho_err)):.4f} "
                f"| qabs={np.mean(np.abs(q_report)):.1f}/{np.max(np.abs(q_report)):.1f} Pa "
                f"| vmax={speed.max():.3f} m/s "
                f"| frontX={front_x:+.4f} "
                f"| Ritter={ritter_x:+.4f} "
                f"| err={front_x-ritter_x:+.4f} "
                f"| KE={kinetic_energy:.4e}J "
                f"| shell={atmospheric_count} "
                f"| band={traction_band_count} "
                f"| atmNew/Lost={atm_new_since_report}/{atm_lost_since_report} "
                f"| atmEver={ever_atmospheric_count} "
                f"| airHG={hG_active_mean:.3f}/{hG_active_max:.3f} "
                f"| solidBlock={np.count_nonzero(blocked_candidates_report)} "
                f"| nX={surface_nx:+.3f} "
                f"| surfP={surf_p_mean:.0f}/{surf_p_max:.0f}Pa "
                f"| atmA={atm_acc_mean:.2f}/{atm_acc_max:.2f} "
                f"| atmX={atm_x:+.2f} "
                f"| wetWall={np.count_nonzero(wet_pairs)} "
                f"| wallpairs={len(wall_interactions['pair_i'])} "
                f"| contacts={contacts}"
            )
            if report_dt_sim > 0.0:
                Fav_atm = report_J_atm_x / report_dt_sim
                Fav_qff = report_J_qff_x / report_dt_sim
                Fav_qwall = report_J_qwall_x / report_dt_sim
                Fav_visc = report_J_visc_x / report_dt_sim
                Fav_base = report_J_base_x / report_dt_sim
                Fav_coll = report_J_collision_x / report_dt_sim
            else:
                Fav_atm = Fav_qff = Fav_qwall = Fav_visc = Fav_base = Fav_coll = 0.0
            px_actual = cfg.mass * float(np.sum(vel[:, 0]))
            px_ref = force_ref_tait * t
            J_sum = J_atm_x + J_qff_x + J_qwall_x + J_visc_x + J_base_x + J_collision_x
            print(
                f"    FORCE-X avg[N]: atm={Fav_atm:+.4f} qFF={Fav_qff:+.3e} "
                f"qWall={Fav_qwall:+.4f} visc={Fav_visc:+.3e} "
                f"base={Fav_base:+.3e} coll={Fav_coll:+.4f} "
                f"sum={Fav_atm+Fav_qff+Fav_qwall+Fav_visc+Fav_base+Fav_coll:+.4f} "
                f"ref={force_ref_tait:+.4f}"
            )
            if report_dt_sim > 0.0:
                Fd_l1 = report_J_depth_l1_x / report_dt_sim
                Fd_l2 = report_J_depth_l2_x / report_dt_sim
                Fd_l3 = report_J_depth_l3_x / report_dt_sim
                Fd_out = report_J_depth_out_x / report_dt_sim
                Fd_ahead = report_J_depth_ahead_x / report_dt_sim
            else:
                Fd_l1 = Fd_l2 = Fd_l3 = Fd_out = Fd_ahead = 0.0
            print(
                f"    BAND-DEPTH avg[N]: L1={Fd_l1:+.4f} L2={Fd_l2:+.4f} "
                f"L3={Fd_l3:+.4f} OUT={Fd_out:+.4f} AHEAD={Fd_ahead:+.4f} "
                f"sum={Fd_l1+Fd_l2+Fd_l3+Fd_out+Fd_ahead:+.4f}"
            )
            print(
                "    BAND-DEPTH count: "
                f"L1={np.count_nonzero(depth_diag['l1'])} "
                f"L2={np.count_nonzero(depth_diag['l2'])} "
                f"L3={np.count_nonzero(depth_diag['l3'])} "
                f"OUT={np.count_nonzero(depth_diag['outside'])} "
                f"AHEAD={np.count_nonzero(depth_diag['ahead'])} "
                f"| Fx<=2dp={Fd_l1+Fd_l2:+.4f} "
                f"Fx<=support={Fd_l1+Fd_l2+Fd_l3:+.4f}"
            )
            sm = surf_moment_diag["all"]
            sms = surf_moment_diag["shell"]
            smi = surf_moment_diag["inner"]
            ax_ratio = sm["Ax"] / projected_area_x_ref if abs(projected_area_x_ref) > 1.0e-30 else 0.0
            hx_ratio = sm["Hx"] / projected_depth_moment_x_ref if abs(projected_depth_moment_x_ref) > 1.0e-30 else 0.0
            fold_ratio = sm["Ax_abs"] / abs(sm["Ax"]) if abs(sm["Ax"]) > 1.0e-30 else 0.0
            print(
                f"    SURF-MOMENT now: Ax={sm['Ax']:+.6e}m2 ({ax_ratio:.3f}x) "
                f"| |Ax|={sm['Ax_abs']:.6e} fold={fold_ratio:.3f} "
                f"| Hx={sm['Hx']:+.6e}m3 ({hx_ratio:.3f}x) "
                f"| Fx={sm['Fx']:+.4f}N"
            )
            print(
                f"    SURF-MOMENT split: shell Ax/Fx={sms['Ax']:+.6e}/{sms['Fx']:+.4f} "
                f"| inner Ax/Fx={smi['Ax']:+.6e}/{smi['Fx']:+.4f} "
                f"| Ax+/-={sm['Ax_pos']:+.6e}/{sm['Ax_neg']:+.6e}"
            )
            print(
                f"    MOM-X cumulative: P={px_actual:+.6f} Ns | Jsum={J_sum:+.6f} Ns "
                f"| Ritter={px_ref:+.6f} Ns | P/Ritter={(px_actual/px_ref if abs(px_ref)>1e-30 else 0.0):.4f}"
            )
            atm_new_since_report = 0
            atm_lost_since_report = 0
            report_J_atm_x = report_J_qff_x = report_J_qwall_x = 0.0
            report_J_visc_x = report_J_base_x = report_J_collision_x = 0.0
            report_J_depth_l1_x = report_J_depth_l2_x = report_J_depth_l3_x = 0.0
            report_J_depth_out_x = report_J_depth_ahead_x = 0.0
            report_dt_sim = 0.0
            last_report = now

    # Ensure exact final snapshot exists.
    if frame_times[-1] < t - 1.0e-12:
        rho_ref_snap, p_ref_snap, _ = hydrostatic_reference_state(
            pos, surface_level, cfg, return_depth=True
        )
        p_snap = compute_pressure(rho, cfg.rho0, cfg.tait_B, cfg.gamma)
        q_snap = p_snap - p_ref_snap
        q_force_snap = q_snap.copy()
        frame_positions.append(pos.copy())
        frame_rho.append(rho.copy())
        frame_rho_ref.append(rho_ref_snap.copy())
        frame_q.append(q_snap.copy())
        frame_q_force.append(q_force_snap.copy())
        frame_surface_hG.append(support_hG.copy())
        frame_raw_fluid_hG.append(raw_fluid_hG.copy())
        frame_geom_rank.append(geom_rank.copy())
        frame_blocked_frac.append(blocked_frac.copy())
        frame_atmospheric.append(atmospheric.astype(np.uint8).copy())
        frame_traction_band.append(traction_band.astype(np.uint8).copy())
        frame_velocity.append(vel.copy())
        frame_times.append(t)
        frame_drive_accel.append(drive_acceleration(t).copy())

    frame_positions_np = np.asarray(frame_positions, dtype=np.float32)
    frame_rho_np = np.asarray(frame_rho, dtype=np.float32)
    frame_rho_ref_np = np.asarray(frame_rho_ref, dtype=np.float32)
    frame_q_np = np.asarray(frame_q, dtype=np.float32)
    frame_q_force_np = np.asarray(frame_q_force, dtype=np.float32)
    frame_surface_hG_np = np.asarray(frame_surface_hG, dtype=np.float32)
    frame_raw_fluid_hG_np = np.asarray(frame_raw_fluid_hG, dtype=np.float32)
    frame_geom_rank_np = np.asarray(frame_geom_rank, dtype=np.int8)
    frame_blocked_frac_np = np.asarray(frame_blocked_frac, dtype=np.float32)
    frame_atmospheric_np = np.asarray(frame_atmospheric, dtype=np.uint8)
    frame_traction_band_np = np.asarray(frame_traction_band, dtype=np.uint8)
    frame_velocity_np = np.asarray(frame_velocity, dtype=np.float32)
    frame_times_np = np.asarray(frame_times, dtype=np.float64)
    frame_drive_accel_np = np.asarray(frame_drive_accel, dtype=np.float32)
    frame_front_x_np = np.max(frame_positions_np[:, :, 0], axis=1) + 0.5 * cfg.dp
    frame_ritter_x_np = initial_front_x + ritter_front_speed * frame_times_np

    np.savez_compressed(
        NPZ_OUTPUT,
        positions=frame_positions_np,
        rho=frame_rho_np,
        rho_reference=frame_rho_ref_np,
        reduced_pressure_eos=frame_q_np,
        reduced_pressure_force=frame_q_force_np,
        current_surface_hG=frame_surface_hG_np,
        raw_fluid_support_hG=frame_raw_fluid_hG_np,
        collider_geom_rank=frame_geom_rank_np,
        solid_blocked_fraction=frame_blocked_frac_np,
        atmospheric_mask=frame_atmospheric_np,
        traction_band_mask=frame_traction_band_np,
        atmospheric_ever=ever_atmospheric.astype(np.uint8),
        current_surface_hG_threshold=np.asarray(CURRENT_SURFACE_HG_THRESHOLD),
        collider_normal_eigen_rel=np.asarray(COLLIDER_NORMAL_EIGEN_REL),
        collider_triangle_count=np.asarray(len(solid_triangles)),
        surface_probe_offset_factor=np.asarray(SURFACE_PROBE_OFFSET_FACTOR),
        velocity=frame_velocity_np,
        times=frame_times_np,
        drive_acceleration=frame_drive_accel_np,
        drive_start_time=np.asarray(DRIVE_START_TIME),
        drive_duration=np.asarray(DRIVE_DURATION),
        drive_peak_accel=np.asarray(DRIVE_PEAK_ACCEL),
        front_x=frame_front_x_np.astype(np.float32),
        ritter_front_x=frame_ritter_x_np.astype(np.float32),
        ritter_initial_front_x=np.asarray(initial_front_x),
        ritter_initial_height=np.asarray(initial_height),
        ritter_front_speed=np.asarray(ritter_front_speed),
        ritter_rear_wall_time=np.asarray(rear_wall_time),
        surface_level=np.asarray(surface_level),
        dp=np.asarray(cfg.dp),
        h=np.asarray(cfg.h),
    )

    write_usd_animation(
        scene.stage,
        frame_positions,
        OUTPUT_FPS,
        cfg.dp,
        USD_OUTPUT,
    )

    rho_ref_final, p_ref_final, _ = hydrostatic_reference_state(
        pos, surface_level, cfg, return_depth=True
    )
    p_final = compute_pressure(rho, cfg.rho0, cfg.tait_B, cfg.gamma)
    q_final = p_final - p_ref_final
    speed = np.linalg.norm(vel, axis=1)
    rho_err = rho - rho_ref_final

    total_wall = time.perf_counter() - sim_start
    solver_wall = total_wall
    avg_rate = step / solver_wall if solver_wall > 0 else 0.0

    print()
    print("========================================")
    print("SIMULATION COMPLETE")
    print("========================================")
    print(f"Simulated time : {t:.6f} s")
    print(f"Steps          : {step}")
    print(f"Solver wall    : {solver_wall:.3f} s")
    print(f"Average rate   : {avg_rate:.1f} step/s")
    print(
        f"rho min/mean/max : "
        f"{rho.min():.4f} / {rho.mean():.4f} / {rho.max():.4f}"
    )
    print(
        f"rho-ref abs mean/max: "
        f"{np.mean(np.abs(rho_err)):.6f} / {np.max(np.abs(rho_err)):.6f} kg/m3"
    )
    kinetic_energy_final = 0.5 * cfg.mass * float(
        np.sum(np.einsum("ij,ij->i", vel, vel))
    )
    print(
        f"q abs mean/max  : "
        f"{np.mean(np.abs(q_final)):.6f} / {np.max(np.abs(q_final)):.6f} Pa"
    )
    print(f"v max          : {speed.max():.6f} m/s")
    print(f"Kinetic energy : {kinetic_energy_final:.6e} J")
    com_final = np.mean(pos, axis=0)
    mean_vel_final = np.mean(vel, axis=0)
    front_final = float(np.max(pos[:, 0]) + 0.5 * cfg.dp)
    ritter_final = float(initial_front_x + ritter_front_speed * t)
    print(f"External drive : NONE")
    print(f"Front X actual : {front_final:+.6f} m")
    print(f"Front X Ritter : {ritter_final:+.6f} m")
    print(f"Front error    : {front_final-ritter_final:+.6f} m")
    if abs(ritter_final - initial_front_x) > 1.0e-12:
        print(f"Front travel ratio actual/Ritter: {(front_final-initial_front_x)/(ritter_final-initial_front_x):.6f}")
    print(f"COM xyz        : {com_final[0]:+.4f} / {com_final[1]:+.4f} / {com_final[2]:+.4f} m")
    print(f"Mean velocity  : {mean_vel_final[0]:+.4f} / {mean_vel_final[1]:+.4f} / {mean_vel_final[2]:+.4f} m/s")
    print(f"X min/mean/max : {pos[:,0].min():.4f} / {pos[:,0].mean():.4f} / {pos[:,0].max():.4f} m")
    print(f"Z min/mean/max : {pos[:,2].min():.4f} / {pos[:,2].mean():.4f} / {pos[:,2].max():.4f} m")
    print(f"Integrator     : {INTEGRATOR_MODE}")
    print(f"Wall pressure  : {ENABLE_WALL_PRESSURE}")
    print(f"Wall continuity: {ENABLE_WALL_CONTINUITY}")
    final_atmospheric_count = int(np.count_nonzero(atmospheric))
    final_band_count = int(np.count_nonzero(traction_band))
    print(f"Wall q_b       : 0.0 Pa (energy matched)")
    print("Atmospheric BC : wetted-wall completed, anchored traction-band support")
    print(f"Surface shell  : {final_atmospheric_count} / {n} particles")
    print(f"Traction band  : {final_band_count} / {n} particles")
    print(f"Atmospheric ever: {np.count_nonzero(ever_atmospheric)} / {n} particles")
    if final_atmospheric_count:
        print(
            f"air h|G| shell mean/max: "
            f"{np.mean(support_hG[atmospheric]):.6f} / "
            f"{np.max(support_hG[atmospheric]):.6f}"
        )
    if final_band_count:
        print(
            f"band p_ref mean/max: "
            f"{np.mean(p_ref_surface[traction_band]):.6f} / "
            f"{np.max(p_ref_surface[traction_band]):.6f} Pa"
        )
    print(f"Contacts       : {total_contacts}")
    print(f"Collision      : {ENABLE_COLLISION}")
    print(f"Pressure op    : {PRESSURE_OPERATOR_MODE}")
    print(f"Wall Vb        : {boundary.volumes[0]:.6e} m3")
    print(f"Fluid state    : {FLUID_STATE_MODE}")
    print(f"Ritter h0      : {initial_height:.4f} m")
    print(f"Ritter speed   : {ritter_front_speed:.4f} m/s")
    px_actual_final = cfg.mass * float(np.sum(vel[:, 0]))
    px_ref_final = force_ref_tait * t
    J_sum_final = J_atm_x + J_qff_x + J_qwall_x + J_visc_x + J_base_x + J_collision_x
    print(f"Rear-wall time : {rear_wall_time:.4f} s")
    print(f"Momentum X actual/ref: {px_actual_final:.6f} / {px_ref_final:.6f} Ns")
    print(f"Momentum X ratio : {(px_actual_final/px_ref_final if abs(px_ref_final)>1e-30 else 0.0):.6f}")
    print(f"Impulse atm/qWall/coll: {J_atm_x:+.6f} / {J_qwall_x:+.6f} / {J_collision_x:+.6f} Ns")
    print(f"Impulse budget sum: {J_sum_final:+.6f} Ns")
    sm_final = surf_moment_diag["all"]
    sms_final = surf_moment_diag["shell"]
    smi_final = surf_moment_diag["inner"]
    print(
        f"Projected Ax final/ref: {sm_final['Ax']:+.6e} / {projected_area_x_ref:.6e} m2 "
        f"(ratio {sm_final['Ax']/projected_area_x_ref:.6f})"
    )
    print(
        f"Depth Hx final/ref: {sm_final['Hx']:+.6e} / {projected_depth_moment_x_ref:.6e} m3 "
        f"(ratio {sm_final['Hx']/projected_depth_moment_x_ref:.6f})"
    )
    print(
        f"Surface-moment Fx final: {sm_final['Fx']:+.6f} N "
        f"(Tait ratio {sm_final['Fx']/force_ref_tait:.6f})"
    )
    print(f"Shell/inner Fx final: {sms_final['Fx']:+.6f} / {smi_final['Fx']:+.6f} N")
    print(f"Ax signed/abs final: {sm_final['Ax']:+.6e} / {sm_final['Ax_abs']:.6e} m2")
    print("Support model  : fluid AIR seed + wet-side wall completion + finite-face projection")
    print("Traction model : shell + one inward band layer with shell-seed interface anchors")
    print(f"Collider tris  : {len(solid_triangles)}")
    print(f"Collider eigrel: {COLLIDER_NORMAL_EIGEN_REL:.3f}")
    print(f"Output frames  : {len(frame_positions)}")
    print(f"Total wall     : {total_wall:.3f} s")
    print()
    print(f"Saved: {NPZ_OUTPUT}")
    print(f"Saved: {USD_OUTPUT}")


if __name__ == "__main__":
    main()
