# GraSPH

GraSPH simulates SPH fluids, granular particles and rigid bodies from OpenUSD
scenes. Blender controls handle scene roles, export, simulation and result
display. The numerical solver runs in an external Python process.

## Setup

Day-to-day work uses the repository's development version and an existing
external Python environment. ZIP packages are occasional verified milestone
snapshots, not a required step after each change. Version 0.15.0 is the first
locally tested distribution candidate. See the
[release workflow](PLAN.md#release-workflow) for future distributions.

For the packaged Windows Blender add-on, build `output/GraSPH-0.15.0.zip` with:

```powershell
py scripts/build_addon.py
```

Install the ZIP using Blender's **Install from Disk**, then enable **GraSPH
Controls**. The ZIP contains the panel, solver and display modules; the original
repository is not needed on the target machine. See
[packaged installation](blender/README.md#packaged-installation-windows) for
creating the isolated solver environment. Do not install just `grasph_control.py`.

For development or an existing external Python environment:

Install the solver dependencies in the Python environment that Blender will
launch:

```powershell
py -m pip install -r requirements.txt -r requirements-accelerated.txt
```

SciPy provides accelerated neighbor searches. A slower NumPy/Python search is
available without it. The Blender workflow is tested with Blender 5.0.1.

## Blender workflow

1. Enable the packaged add-on, or run `blender/grasph_control.py` from Blender's
   Text Editor for development.
2. In the GraSPH sidebar, assign roles to fluid volumes, granular sources,
   colliders, dynamic bodies and sinks.
3. Configure simulation time, particle spacing and output settings. Keep
   **Export Before Run** enabled to export the current scene.
4. Run the simulation. Pause, stop, reset and live particle preview are
   available while it runs.
5. Use **Import Last Result** to load the animated output and surface display.

See [Blender controls](blender/README.md) for joints, preview and display details.
Scenes and assets are supplied locally; no experimental scene files are
required by the solver or test suite.

## Command line

Export a scene using the Blender controls, then run:

```powershell
py grasph.py path/to/scene.usdc --time 0.5 --dp 0.05 --c0 40 --solid-coupling --name result
```

Use `py grasph.py --help` for all options. Results default to `output/` and
include an NPZ particle cache and a USD animation overlay. The overlay references
the source USD, so keep both files and their relative paths intact. Simulation
seconds map to the source stage's start time and time-codes-per-second.

Fluid-solid coupling is optional. Without it, dynamic bodies act as fixed fluid
colliders in SPH runs. Granular-only and rigid-body-only scenes are also
supported. Fluid and granular sources can share a scene.

## Simulation behavior

- Fluid volumes are initially sampled on a regular grid within their world-space
  bounding boxes. Initial solid intersections and particle-radius overlaps are
  excluded.
- Static and translated kinematic colliders use triangle distance fields.
  Authored rotation, scale animation and deforming collision meshes are not
  supported. The solver can integrate rigid-body and hinge rotation.
- Dynamic bodies support spherical or sampled convex collision geometry.
  Joints constrain one body to the world: revolute, prismatic or fixed.
  Joints between two moving bodies are not supported.
- Convex dynamic-body pairs use full polyhedral SAT (face and edge-edge axes),
  authored margins and a supporting-feature contact point. Position corrections
  respect mass and fixed/prismatic constraints and use fresh poses in bounded
  adaptive contact sweeps. Static/kinematic meshes and mixed sphere pairs retain
  the sampled contact path. Contacts are discrete, not continuous collision
  detection; a full multi-point contact manifold is not implemented.
- The fluid uses a hydrostatic reference and an anchored-band free-surface
  model. The main connected fluid component uses that reference; detached
  components use full gravity with optional air drag.
- Smaller spacing increases particle count and cost. Sound speed affects the
  stable timestep and compressibility. Start with a short run and check the
  reported density deviations and velocities before increasing duration.
- Surface meshing changes the visualization only; it does not change physics.

## Fluid force diagnostics

Diagnostics are opt-in and do not change integration, forces or timestep settings.
In Blender, enable **Advanced Solver Settings > Sample Force Diagnostics**,
choose a time window and optionally enter comma-separated persistent particle
IDs. Sampled traces are written to `<result-name>.diagnostics.jsonl` alongside
the normal output. Use a new result name to preserve reference results.

CLI example (fresh simulation; not a coupled restart):

```powershell
py grasph.py path/to/scene.usdc --time 1.1 --dp 0.05 --c0 40 --solid-coupling `
  --name diagnostic_test --diagnostics --diagnostics-start 0.5 `
  --diagnostics-end 1.1 --diagnostics-interval 0.01 --diagnostics-ids 30774,30806
```

Samples follow the fastest particles before/after each sampled step plus the
requested IDs. They separate fluid pressure, wall perturbation pressure,
hydrostatic-reference/base acceleration, viscosity, free-surface and other
accelerations. Drag and collision velocity changes, collision position changes,
fluid/wall density rates, pre-step masks, neighbor counts, geometry rank,
free-surface support/reference pressure and persistent anchor IDs, and distance to the
pre-step finite solid triangles are recorded. Sample timings split several
force phases; diagnostic work adds overhead, so do not use these runs as clean
performance baselines. Collision corrections are net changes over that step,
not per-contact impulses. Forces are accelerations in m/s^2.

Saved-frame analysis without a simulation:

```powershell
py -m grasph_diagnostics output/result.npz --out output/result-analysis.json
```

This reports speed distributions, fluid kinetic energy, fastest particle IDs
and the first saved frame exceeding a speed threshold. It cannot recover the
original per-step forces or prove the source of an instability. Sampled JSONL
traces likewise do not constitute a complete energy history. Coupled restarts
are not supported by the current fluid-only restart format.

## Development

See [Development plan](PLAN.md) for the next performance update and validation.

The repository contains the solver (`grasph*.py`), Blender integration
(`blender/`), dependency manifests and automated regression tests (`tests/`).
Tests generate their own USD fixtures in temporary directories.

```powershell
py -m unittest discover -s tests -v
```

For performance work, profile a short representative scene with unchanged
spacing, sound speed, coupling and timestep settings:

```powershell
py -m cProfile -o output/profile.pstats grasph.py path/to/scene.usdc `
  --time 0.03 --dp 0.05 --c0 40 --solid-coupling --name profile
py -m pstats output/profile.pstats
```

Compare the log's solver wall time and phase timings separately from scene
loading and output. Check particle state, free-surface masks and body transforms
against the baseline; faster settings alone do not establish a solver speedup.
Keep profiles and benchmark outputs under the ignored `output/` directory.

Blender integration check:

```powershell
& 'C:\Program Files\Blender Foundation\Blender 5.0\blender.exe' `
  --background --factory-startup --python-exit-code 1 `
  --python tests/blender_panel_smoke.py
```

Simulation output, local scenes, textures, videos, archived prototypes and
one-off diagnostic scripts are excluded from version control. Git history
retains the earlier experiments; a fresh checkout contains only the maintained
source, documentation and tests.
