import numpy as np


# ============================================================
# GraSPH - FIRST-ORDER CONSISTENCY / VALIDATION OPERATORS
#
# This module deliberately contains a non-conservative local
# weighted-least-squares pressure-gradient operator.  It is used
# as a diagnostic/reference operator while validating the wall
# discretisation.  For any exactly linear pressure field and a
# full-rank local neighbour cloud, it reproduces grad(p) exactly
# (up to floating-point error).
#
# It is NOT yet intended as the final production pressure force;
# once the hydrostatic test is clean, the same consistency idea
# can be folded into a conservative corrected-SPH formulation.
# ============================================================


def compute_wls_pressure_acceleration(
    pos,
    rho,
    pressure,
    pair_i,
    pair_j,
    r,
    kernel,
    fluid_mass,
    boundary,
    wall_interactions,
    wall_pressure,
    rcond=1.0e-10,
):
    """Estimate grad(p) by local weighted least squares.

    We fit

        p_j - p_i ~= grad(p)_i dot (x_j - x_i)

    using both fluid neighbours and fixed wall samples.  Kernel
    weight times neighbour volume is used as quadrature weight.

    Returns
    -------
    acc_pressure : (N,3)
        -grad(p)/rho for every fluid particle.
    diagnostics : dict
        Local matrix conditioning / rank information.
    """

    pos = np.asarray(pos, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    pressure = np.asarray(pressure, dtype=np.float64)

    n = len(pos)
    moment = np.zeros((n, 3, 3), dtype=np.float64)
    rhs = np.zeros((n, 3), dtype=np.float64)

    # --------------------------------------------------------
    # Fluid-fluid samples. Pair list is unique/unordered, so we
    # add the regression sample to both endpoints.
    # --------------------------------------------------------

    if len(pair_i):
        W = kernel.W(r)
        fluid_volume = float(fluid_mass) / np.maximum(rho, 1.0e-30)

        d = pos[pair_j] - pos[pair_i]
        dpair = pressure[pair_j] - pressure[pair_i]
        outer = np.einsum("ni,nj->nij", d, d)
        qd = dpair[:, None] * d

        wi = W * fluid_volume[pair_j]
        wj = W * fluid_volume[pair_i]

        np.add.at(moment, pair_i, wi[:, None, None] * outer)
        np.add.at(moment, pair_j, wj[:, None, None] * outer)

        # For endpoint j both dp and d change sign, therefore
        # (p_i-p_j)*(x_i-x_j) == (p_j-p_i)*(x_j-x_i).
        np.add.at(rhs, pair_i, wi[:, None] * qd)
        np.add.at(rhs, pair_j, wj[:, None] * qd)

    # --------------------------------------------------------
    # Fluid-wall samples.  Only front-facing wall pairs are in
    # wall_interactions, so samples through thin solids are not
    # included.
    # --------------------------------------------------------

    wall_i = wall_interactions["pair_i"]
    wall_j = wall_interactions["pair_j"]

    if len(wall_i):
        Wb = wall_interactions["W"]
        d = boundary.positions[wall_j] - pos[wall_i]
        dpair = wall_pressure[wall_j] - pressure[wall_i]
        outer = np.einsum("ni,nj->nij", d, d)
        qd = dpair[:, None] * d
        wb = Wb * boundary.volumes[wall_j]

        np.add.at(moment, wall_i, wb[:, None, None] * outer)
        np.add.at(rhs, wall_i, wb[:, None] * qd)

    # --------------------------------------------------------
    # Pseudo-inverse is intentional here.  It gives a graceful
    # result for a badly one-sided/free-surface cloud and also
    # lets us report exactly where local consistency is weak.
    # --------------------------------------------------------

    eig = np.linalg.eigvalsh(moment)
    max_eig = eig[:, 2]
    min_eig = eig[:, 0]

    positive = max_eig > 1.0e-30
    condition = np.full(n, np.inf, dtype=np.float64)
    good_min = positive & (min_eig > 1.0e-30)
    condition[good_min] = max_eig[good_min] / min_eig[good_min]

    threshold = np.maximum(max_eig * float(rcond), 1.0e-30)
    rank = np.sum(eig > threshold[:, None], axis=1)

    inv_moment = np.linalg.pinv(moment, rcond=float(rcond))
    grad_p = np.einsum("nij,nj->ni", inv_moment, rhs)

    acc_pressure = -grad_p / np.maximum(rho[:, None], 1.0e-30)

    finite_condition = condition[np.isfinite(condition)]
    if len(finite_condition):
        cond_median = float(np.median(finite_condition))
        cond_p95 = float(np.percentile(finite_condition, 95.0))
        cond_max = float(np.max(finite_condition))
    else:
        cond_median = np.inf
        cond_p95 = np.inf
        cond_max = np.inf

    diagnostics = {
        "grad_p": grad_p,
        "moment": moment,
        "rank": rank,
        "rank_deficient": int(np.count_nonzero(rank < 3)),
        "condition": condition,
        "condition_median": cond_median,
        "condition_p95": cond_p95,
        "condition_max": cond_max,
    }

    return acc_pressure, diagnostics
