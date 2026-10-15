# Fixed-body fluid force audit and corner correction, 2026-10-14

Source is `output/inputs/grasph_Fluidtest2_scene_20261014_082758_116.usdc`.
All three rigid bodies have fixed joints. Original saved fluid RMS speed grew
from 0.074 m/s at 1 s to 0.452 m/s at 2 s and remained about 0.435 m/s at 8 s.

## Reconstructed force work

`fluid_force_audit.py` rebuilds operators at saved states. Snapshot positions
are post-drift and rounded, so these powers are diagnostic estimates, not exact
integrated step energy balances. It assumes the run's default viscosity alpha
0.05 and anchored-band free surface.

At 1.5 s the free-surface term supplied about +5,693 W; viscosity removed about
2,150 W. The surface detector had 235 shell particles more than two particle
spacings below the initial water level. Some of that population had grown out
of incorrect initial corner classification; replaying an already disturbed
state with the correction does not undo its accumulated motion.

## Geometry defect

The detector accepted a triangle only when the perpendicular projection fell
inside its footprint. Just outside a submerged convex corner, a fluid kernel
can intersect both faces even though the point projects outside both. Neither
face contributed and the missing support was mistaken for air. That false
surface received pool-referenced atmospheric traction below the water level.

The correction accepts finite nearest edge/corner features within kernel
support, on the face's outward side. Back-facing rim faces are excluded so a
nearby rim does not block an upward free surface across its back side. The
pressure operator, viscosity and atmospheric traction formula are unchanged.

At the original initial state:

| Metric | Original geometry | Corrected geometry |
| --- | ---: | ---: |
| Deep surface seeds (>0.1 m below reference) | 29 | 0 |
| Free-surface acceleration RMS | 0.828 m/s² | 0.0119 m/s² |
| Maximum free-surface acceleration | 29.52 m/s² | 0.573 m/s² |

## Fresh-run verification

At original dp=0.05 m, c0=40 m/s and 31,265 particles, a 100-step fresh run
reached 0.03 s with RMS speed 0.000238 m/s and maximum 0.01082 m/s. Mean/max
pressure deviations were 0.520/39.597 Pa. No particle collisions or detached
components occurred. Output: `output/surface_corner_probe.npz` and `.usdc`.

A longer diagnostic uses dp=0.1 m, c0=40 m/s, dt_max=0.001 s, CFL=0.25,
the same fixed geometry and 3,771 particles. This respects the CFL limit and
uses 2,463 steps. The coarser spacing is not a fine-resolution validation.

| Time | Original classification RMS | Corrected fluid RMS |
| --- | ---: | ---: |
| 0.5 s | 0.0703 m/s | 0.000698 m/s |
| 1.0 s | 0.0718 m/s | 0.000477 m/s |
| 1.5 s | 0.1763 m/s | 0.000378 m/s |
| 2.0 s | 0.1977 m/s | 0.000313 m/s |

At 2 s, maximum speed is 0.00456 m/s, mean/max pressure deviations are
0.313/3.687 Pa, and no particles detached or entered sinks.
The original detector produced maximum speed 1.995 m/s and mean/max pressure
deviations 1,731/33,640 Pa at 2 s. Identical initial particle arrays and fixed
body translations were checked. The revised detector reduces RMS speed by
about 99.84% in this coarse two-second comparison; it is not a claim about
every simulation or the original fine-resolution eight-second run.

Reproduce the force audit and longer diagnostic:

```powershell
py benchmarks/fluid_force_audit.py output/grasph_Fluidtest2.npz
py benchmarks/still_water_probe.py output/inputs/grasph_Fluidtest2_scene_20261014_082758_116.usdc --time 2
py benchmarks/still_water_probe.py output/inputs/grasph_Fluidtest2_scene_20261014_082758_116.usdc --time 2 --legacy-corners
```

The legacy switch restores only the original geometry detector for the
comparison process. It does not change the production solver. Separate result
names preserve both runs and the user's original output.

The full suite passed 43 tests, including new coverage for a submerged convex
corner, a triangle's finite extent and a back-facing rim. A multi-second run at
the original fine spacing and a new free-body buoyancy run remain to verify.
