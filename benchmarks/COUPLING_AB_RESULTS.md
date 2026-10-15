# Startup comparison, 2026-10-13

Source: `output/inputs/grasph_Fluidtest2_scene_20261013_140912_806.usdc`.
Each comparison uses 0.05 s, dp=0.05 m, c0=40 m/s, identical particle positions,
wall pressure, wall continuity and solid coupling. Only the body joint type
changes between fixed and free. Fixed body translation was verified as zero.

| Startup rule | Particle count | Fixed-body fluid RMS | Free-body fluid RMS |
| --- | ---: | ---: | ---: |
| Exclude centers inside solids | 17,408 | 0.738 m/s | 0.739 m/s |
| Also clear the particle collision radius | 16,686 | 0.924 m/s | 0.924 m/s |

Original generation produced 17,472 particles. The full clearance excludes 786
initially overlapping particles, including the 64 centers inside the large
body. This prevents overlap/contact corrections but did not reduce the early
fluid speed. Do not interpret it as a demonstrated settling fix.

The scene's initial fluid bounds are approximately X=[-0.656,0.678] and
Y=[-0.611,0.611]. Tank inner walls are X/Y=+/-0.842474 m, with floor Z=0.465058 m.
Consequently the initial water block has free vertical sides, with about
0.16–0.23 m of air to the walls. A hydrostatic pressure profile in that block
is not a stationary filled-basin initial condition.

At startup the anchored free-surface term has RMS acceleration 54.55 m/s² and
maximum 251.84 m/s². The largest acceleration is lateral near the lower edge
of the water block. Lateral spreading is consistent with exposed vertical
fluid surfaces. These observations alone do not establish a free-surface
implementation bug or explain the entire later three-second motion.

Next useful baseline: fluid spanning the basin's interior, bottom at the
inner floor, fixed bodies, then the same setup with free bodies. This tests
stationary water instead of releasing a narrower water column.

Reproduce the current comparison:

```powershell
py benchmarks/coupling_ab.py output/inputs/grasph_Fluidtest2_scene_20261013_140912_806.usdc --time 0.05 --out output/coupling_ab_clearance
```

Results are `output/coupling_ab_clearance/fixed.usdc` and `free.usdc`, alongside
their NPZ files. The fixed scene is an overlay and keeps the original scene.
The older center-only comparison remains in `output/coupling_ab`.

New solver runs apply solid-volume exclusion and the configured particle-radius
clearance automatically. Start fresh: older NPZ files whose initial layout
differs are rejected by the existing restart compatibility check. The full
unit suite passed 40 tests after these changes.
