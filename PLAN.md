# Development plan

## Shared physics materials (2026-10-17)

Implemented in development controls: a filtered physics material library using
Blender Material IDs as storage, separate from shader slot assignment. Entity
references resolve shared parameters at export; manual settings remain the
fallback when unlinked. Supports colliders, rigid bodies/hinges and granular
sources. Grain density is included; rigid mass/inertia are not auto-derived.
Native ID selectors accept material reference drops from Outliner; direct
viewport shader dropping is deliberately not repurposed as physics assignment.
The panel also supports multi-selected entity assignment and editable example
presets. Shared data/reference persistence, reload and USD export are covered by
`tests/blender_physics_materials_smoke.py`.

Future: dedicated asset/viewport drag interaction if desired, calibrated material
libraries, optional geometry-derived rigid mass/inertia and separate joint
strength profiles. No claim of per-material fluid viscosity/density support.

## Shared joints and future fracture authoring

Batch authoring: Connect Selected to Active now creates one ordinary Fixed
Joint per selected Dynamic Body to the active body, using common new-joint
break settings. Midpoint anchors, scene-local Joint collection, duplicate-pair
prevention and unchanged selection are covered by Blender smoke tests. No
special group state or new physics path is introduced.

Decision 2026-10-17: no separate fracture physics system. The experimental
Fracture Group/star coupling and demo-scene operator have been removed.
`grasph_joints.py` solves explicit Fixed joints between two Dynamic Bodies or
a body and world. Blender Joint empties expose Body A (optional world), Body B,
Breakable and force/torque limits. Breakable off is infinite strength. Old
world-only hinges/sliders retain their original, non-breakable behavior.

Constraints use a simultaneous bilateral impulse solve with world-space inverse
inertia, per-joint reaction loads, irreversible breaks and connected-component
collision filtering/pose stabilization. Loops use a minimum-norm solution.
Events use .joints.jsonl; existing USD motion output is reused. Legacy fracture
USD is rejected with migration guidance. Plain scenes retain their original
path. No distribution has been built.

Limitations: per-body external contacts rather than compound manifolds;
stabilization can change energy; impact force limits depend on timestep;
loop reactions are not unique material stresses; no full-state joint restart.
Two-body/breakable hinges and sliders need constraint-specific solvers.

Next: timestep/energy tests and larger graphs, then Voronoi authoring that
generates convex bodies and explicit neighbor joints. Never introduce a
parallel fracture integrator. Preserve original meshes/scenes during generation.
Coupled-fluid joint validation remains pending.

## Fluid / rigid-body baseline freeze (2026-10-16)

User decision: freeze the current fluid and rigid-body implementation as the
development baseline and move on to granular / fluid testing. This is a source
checkpoint, not a new distribution release or a claim of complete physical
validation. Keep the known late-time free-surface investigation open below.

The user reports no penetration in the new eight-second rigid-only test.
The subsequent coupled test completed four seconds with dp=0.055 m, c0=40 m/s
and three 5 kg bodies (9,570 initial fluid particles; 3,307.270 s solver wall
time). Saved fluid RMS speeds remain approximately 1.1-1.3 m/s from one to four
seconds: no settling has been demonstrated, despite promising visual results.
Local result files are excluded from Git and may be overwritten by later runs.

Next: small, isolated granular / fluid tests, retaining this fluid / rigid
baseline for regression comparisons. Avoid unrelated fluid / rigid changes
unless a reproduced defect requires them.

## Release workflow

Decision (2026-10-15): continue day-to-day development and simulation testing
from the repository, using the Blender Text Editor controls and an existing
external Python/venv. The packaged add-on is a distribution snapshot, not the
default development installation.

Build a new distribution at meaningful, verified milestones, not after every
edit. Packaging support does not mean a release has been published. Version
0.15.0 is the first locally verified Windows distribution candidate; its ZIP
has been built and tested, but no GitHub release/publication is implied.

Milestone distribution checklist:

1. Confirm the intended feature/fix scope and record changes, limitations and
   supported Blender/Python versions in release notes.
2. Run regression tests, Blender panel and installed-package smoke tests, and
   a representative simulation appropriate to the change. Performance claims
   require controlled before/after measurements.
3. Set a new matching version in `grasph_control.py` and `addon_init.py`; update
   version-specific documentation and test expectations.
4. Commit the intended source snapshot before making the final package. Build
   the ZIP with `py scripts/build_addon.py` and verify that exact artifact from
   a separate installation directory, without relying on the development repo.
5. Record the source commit, package version, dependency versions and test
   results alongside the release notes. Only tag, publish or upload the
   distribution when explicitly agreed.

Keep generated ZIPs, environments and local simulation output out of Git.
Do not activate the installed distribution and Text Editor controls together
in the same Blender session: they register the same panels and properties.
Use a separate Blender session for distribution testing. Never reload the
controls or replace the active solver installation during a simulation.

## Packaged Blender installation

Status: implemented in version 0.15.0. A self-distributed Windows legacy add-on
ZIP includes solver and display modules. Explicit background setup/check actions
manage a CPython 3.13 venv in user storage, with an existing-Python alternative.
Blender's Python is not modified and model projects need no source repository.

Verified with Blender 5.0.1: add-on metadata discovery, enable/disable, companion
modules and path resolution outside the repository; installation checks in both
managed and custom-Python modes. A fresh venv installed NumPy, SciPy and USD from
wheels and passed all 56 regression tests. Setup failure, locking and requirement
changes have automated coverage. The ZIP is built by `scripts/build_addon.py`.
This does not change the pending solver-performance work below.

## Next update: SPH force performance

Status: planned; investigate the Fluidtest2 late-time velocity growth first.
2026-10-16 addendum: saved poses confirm substantial small/large cube overlap.
The rigid contact sampler only checks one lexicographically chosen direction
and can miss this contact. The convex-pair fix is now implemented: full SAT,
supporting-feature contact points, mass/constraint-aware separation, fresh
iterated poses and bounded adaptive recovery. All 70 regressions pass. A frozen
recovery of all 193 saved poses removes the previously measured 0.115885 m
maximum geometric overlap; a separate one-second rigid-only trajectory also
has no geometric body-pair overlap in its 121 output frames. This is not yet a
new long coupled-fluid validation; see the analysis P0 and its implementation
addendum. No distribution was rebuilt.
The [eight-second investigation](docs/analysis/2026-10-15-fluidtest2-longrun.md)
now identifies deep wall-adjacent free-surface traction as the leading
amplification mechanism, supported by frozen force replays and a controlled
particle-disorder test. Prioritize seed/air verification and a late-time work
budget before optimizing or adding damping. The initial trigger and late
collision work are not yet established. No physics fix has been applied.
Do not change solver code or run competing benchmarks while
the current user simulation is running. Keep its results as a reference.

1. Split force-phase timing into pressure, artificial viscosity, fluid-solid
   coupling and density continuity. Measure overhead separately where practical.
2. Identify the largest cost using the same representative scene and settings.
3. Investigate shared block processing of pressure and viscosity to reduce
   repeated particle-pair traversal, gathers and temporary arrays. Implement
   only changes supported by measurements; keep memory use bounded.
4. Preserve the integration order: force evaluation, velocity kick, detached
   air drag, density continuity using the updated velocity, then position drift.
   Do not reuse pre-kick relative velocities for continuity.
5. Run regression tests and a controlled before/after benchmark with identical
   scene, initial state, timestep, particle spacing, sound speed, coupling,
   thread settings and output options. Compare particle states, free-surface
   and detached masks, body motion and contact behavior within justified
   floating-point tolerances.
6. Confirm stability in a longer coupled simulation before accepting the update.

Report measured solver and phase timings, numerical differences and remaining
bottlenecks. Do not claim a physics-settings change as a solver optimization.
Keep one-off benchmark scripts and outputs outside version control under
`output/`; retain useful automated regression tests in `tests/`.

## Follow-up validation

- Fluidtest2 source `grasph_Fluidtest2_scene_20261015_132042_412.usdc` has two
  dynamic small bodies and a static large cube. Its saved states first cross
  1 m/s between t=0.9585 and 1.0002 s (IDs 30774 and 30806 near the tank wall).
  Approximate force evaluations highlight free-surface traction as a candidate,
  not a confirmed cause. The findings are in local output, not tracked assets.
- Opt-in sampled force diagnostics and read-only NPZ analysis are implemented.
  Next measure a fresh, unchanged-scene run through t=1.1 s, sampling
  t=0.5..1.1 s and following IDs 25837, 30774, 30806, 30837. Separate wall,
  pressure, free-surface and collision effects before making a physics fix.
  These development changes do not create a new distribution automatically.
- Interpret velocity peaks in relation to the cube entering the fluid via the
  chute. Increased velocity alone does not establish numerical instability.
- If needed, compare the dynamic chute cube with that same cube fixed, keeping
  everything else unchanged; check whether motion decays after impact.
- Defer the terminal-versus-Blender performance comparison until later.

## Previous performance pass

Implemented allocation and graph-processing improvements. All 48 regression
tests passed. In a 100-step test with 31,265 particles and three moving bodies,
solver wall time decreased from 86.515 s to 65.911 s (23.8% reduction).
Particle positions, densities and classification masks matched exactly in the
saved output; other compared states differed only at rounding scale.
This short benchmark does not establish the speedup or stability of longer runs.
