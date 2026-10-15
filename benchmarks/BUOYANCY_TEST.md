# Buoyancy and CPU changes, 2026-10-13

The fluid solver subtracts its hydrostatic reference pressure to preserve
equilibrium. Moving solids previously received only the pressure deviation.
Their reaction now also includes the reference pressure through the existing
wall pair operator. Gravity and collision impulses remain active. This is a
discrete reference load, not an imposed rise speed or a scaled buoyancy force.

The wall candidate cache now uses actual wall displacement, including rotation.
It rebuilds when maximum fluid travel plus maximum wall travel reaches its skin
width. A fluid pair cache similarly rebuilds at twice the maximum fluid travel.
Both filter current distances with the original strict support radius each
step. `--wall-search rebuild` disables both caches for comparison.

## Reproducible small test

From the project directory:

```powershell
py benchmarks/buoyancy_probe.py --time 0.2 --mass 0.5
py benchmarks/buoyancy_probe.py --time 0.2 --mass 8
```

Outputs and source scenes are in `output/buoyancy_probe`. Each result USDC
references its corresponding source USDA; keep both files together. The test
uses 556 particles, 0.04 m spacing, 40 m/s sound speed, stationary walls and a
0.16 m cube initially immersed at rest. The initial discrete reference level
is about 0.30 m, giving about 2.05 kg of displaced water. Sources leave a cavity
for the body to avoid starting with particles inside the solid.

Observed 0.06 s results: 0.5 kg rises 0.02436 m; 8 kg sinks 0.01025 m.
By 0.2 s the 0.5 kg cube rises 0.09068 m and the 8 kg cube sinks 0.10112 m.
These are short qualitative checks,
not a demonstration of settled flotation or quantitatively exact buoyancy.
At this coarse resolution the initial discrete hydrostatic vertical load is
about 15.28 N versus the continuum estimate of 20.09 N. Refinement, longer
equilibrium runs and motion near an evolving free surface remain necessary.

Cached/rebuild runs at 0.06 s agree within floating-point rounding. On this
small scene the measured solver times were 8.46 s cached and 10.19 s rebuild.
On the user's 9,360-particle scene, a repeated 52-step probe took 14.47 s cached
and 15.71 s rebuilding (about 8% less elapsed time). Wall interactions alone
fell from 2.52 s to 0.35 s. The shorter wall search does not remove the costs
of fluid geometry, atmospheric state, SPH forces and connectivity. Earlier
timings varied with CPU load; this is not a promise of a fixed speedup.
The full unit suite includes light-rise/heavy-sink, rotating walls, changing
support radius and particle removal. Performance timings are short CPU probes
and do not predict a complete turbulent run's elapsed time.

## Next Blender run

The solver code is loaded automatically by the next external run. Start with
a short run (0.5 s), one body initially partially immersed, and static tank
walls. Keep wall pressure, wall continuity and Fluid / Solid Coupling enabled.
Use a new result name to preserve the earlier simulation. Body log lines now
report mass, world Z, vertical velocity and pressure/reference forces.

For a controlled low-speed trial, 40 m/s sound speed substantially reduces
the CFL step count relative to 280 m/s. Check density deviations and particle
speeds before applying it to a falling/impacting body or extending the run.
Draft currently changes spacing but preserves the scene's sound-speed setting.
