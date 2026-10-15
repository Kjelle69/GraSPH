# GraSPH

GraSPH simulates SPH fluids, granular particles and rigid bodies from OpenUSD
scenes. Blender controls handle scene roles, export, simulation and result
display. The numerical solver runs in an external Python process.

## Setup

Install the solver dependencies in the Python environment that Blender will
launch:

```powershell
py -m pip install -r requirements.txt -r requirements-accelerated.txt
```

SciPy provides accelerated neighbor searches. A slower NumPy/Python search is
available without it. The Blender workflow is tested with Blender 5.0.1.

## Blender workflow

1. Open `blender/grasph_control.py` in Blender's Text Editor and run it.
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
- The fluid uses a hydrostatic reference and an anchored-band free-surface
  model. The main connected fluid component uses that reference; detached
  components use full gravity with optional air drag.
- Smaller spacing increases particle count and cost. Sound speed affects the
  stable timestep and compressibility. Start with a short run and check the
  reported density deviations and velocities before increasing duration.
- Surface meshing changes the visualization only; it does not change physics.

## Development

The repository contains the solver (`grasph*.py`), Blender integration
(`blender/`), dependency manifests and automated regression tests (`tests/`).
Tests generate their own USD fixtures in temporary directories.

```powershell
py -m unittest discover -s tests -v
```

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
