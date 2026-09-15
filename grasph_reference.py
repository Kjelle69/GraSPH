import numpy as np


def hydrostatic_reference_state(pos, surface_level, cfg, return_depth=False):
    pos = np.asarray(pos, dtype=np.float64)
    gvec = np.asarray(cfg.gravity, dtype=np.float64)
    gmag = float(np.linalg.norm(gvec))

    if gmag <= 1.0e-15:
        rho = np.full(len(pos), cfg.rho0, dtype=np.float64)
        pressure = np.zeros(len(pos), dtype=np.float64)
        depth = np.zeros(len(pos), dtype=np.float64)
        return (rho, pressure, depth) if return_depth else (rho, pressure)

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
    return (rho, pressure, depth) if return_depth else (rho, pressure)


def compute_well_balanced_base_acceleration(rho, rho_ref, depth_ref, gravity):
    rho = np.asarray(rho, dtype=np.float64)
    rho_ref = np.asarray(rho_ref, dtype=np.float64)
    gvec = np.asarray(gravity, dtype=np.float64)
    acc = np.repeat(gvec[None, :], len(rho), axis=0)
    submerged = depth_ref > 0.0
    if np.any(submerged):
        ratio = rho_ref[submerged] / np.maximum(rho[submerged], 1.0e-30)
        acc[submerged] *= (1.0 - ratio)[:, None]
    return acc
