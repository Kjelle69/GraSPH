# Blender controls

Install the complete ZIP, or open `grasph_control.py` in Blender's Text Editor
and run it for development. The 3D View's
**GraSPH** sidebar provides scene roles, simulation controls and result display.
The solver uses the external Python configured under **Output & Preview**.
Project-root discovery is automatic when the script's location is available;
**Project & Scene** allows an explicit project directory when needed.

Our normal development workflow remains the Text Editor/repository version
with an existing external Python environment. Build and test a distribution
only at meaningful milestones; see the [release workflow](../PLAN.md#release-workflow).
Do not enable the packaged add-on and run the development controls together
in the same Blender session. Test distributions in a separate session, and
wait for any active simulation to finish before changing its installation.

## Packaged installation (Windows)

1. Install 64-bit CPython 3.13 (including the Windows `py` launcher).
2. Install `GraSPH-0.15.0.zip` using Blender's **Install from Disk** and enable
   **GraSPH Controls**. This is a self-distributed legacy add-on ZIP, not a
   Blender Extensions repository submission.
3. Open **GraSPH > Solver Installation**. Enable **Managed Solver Environment**
   and leave **Setup Python** as `py -3.13`, or enter the full path to that
   interpreter. Paths containing spaces are supported.
4. Click **Install Solver Environment** and confirm the dependency download.
   Setup runs in a separate process; Blender remains usable. Use
   **Open Installation Log** for progress and troubleshooting.
5. Once complete, click **Check Installation**. This loads NumPy, SciPy and USD,
   exercises native libraries, and checks that the packaged solver starts.

The managed venv lives under `%LOCALAPPDATA%\GraSPH\environments`, with a path
derived from the packaged requirements. Logs live under `%LOCALAPPDATA%\GraSPH\logs`.
No packages are installed into Blender's Python. No administrator access is
needed for these user directories. Setup installs binary wheels from PyPI and
fails with a log if a compatible wheel or network access is unavailable.
Enabling the add-on never downloads dependencies automatically.

The first setup needs internet and installed CPython 3.13; normal simulation
does not. Keep that base Python installed: a venv is not a portable Python
distribution. Recreate the environment on each machine rather than copying it.
If requirements change, a new environment is selected; previous environments
are left intact. No automatic environment deletion is performed. Interrupted
setup may leave a `.lock` beside the venv: confirm the installer has exited
before manually removing that lock and retrying.

For an existing Python/venv, disable **Managed Solver Environment**, set
**Python** to its executable (or command), and click **Check Installation**.
This checks it without installing or modifying its packages. Development via
Text Editor defaults to this external-Python mode; the complete installed
package defaults to managed mode. The choice is saved in the `.blend` file;
an older file can retain its external-Python choice.

Save each `.blend` wherever you want. `//output` resolves beside that file,
and the installed package supplies the solver independently of the source repo.
For an existing GraSPH-authored USD, disable **Export Before Run** and select
the USD input. Unsaved Blender files still use the solver directory for `//`
paths: save the model or choose an absolute output folder first.

Developers build the ZIP with `py scripts/build_addon.py`. The explicit file
list excludes local models, output, tests, venvs and experiments. Test a package
without network access using Blender's background mode and
`tests/blender_install_smoke.py`.

## Scene roles

For targeted fluid debugging, **Advanced Solver Settings > Sample Force
Diagnostics** enables sampled force traces. Set **Diagnostics From/To (s)**,
the sampling interval and optional **Track Particle IDs**. The default is off.
See [fluid force diagnostics](../README.md#fluid-force-diagnostics) for fields,
units and limitations. Use a separate result name; keep a reference scene and
its physics settings unchanged. A coupled diagnostic run starts from the
original scene, not the fluid-only continuation option.

Assign **Fluid Volume**, **Granular Source**, **Static Collider**,
**Kinematic Collider**, **Dynamic Hinge**, **Dynamic Body**, **Sink / Drain** or
**Visual Only** to the relevant mesh objects. Use **Apply Role to Selected** to
apply a role to multiple objects. Validate the scene before running it.

Fluid volumes spawn particles inside their world-space bounding boxes, excluding
solid intersections. Granular sources support initial volumes and timed flow
outlets with radius, density, velocity and material/contact settings. Mixed
fluid/granular scenes are supported.

Dynamic bodies expose mass, inertia, damping and sphere/convex collision shape.
For SPH runs, enable **Advanced Solver Settings > Fluid / Solid Coupling** to
integrate them. A body left unconstrained is free by default.

## World-attached joints

Select the dynamic body and use **Project & Scene > Add Joint Anchor** to create
a separate Empty at the 3D Cursor. Position and rotate the Empty, select its
joint type and verify its **Dynamic Body** target:

- **Revolute (Hinge)** permits rotation around the anchor axis.
- **Prismatic (Linear)** permits translation along the axis.
- **Fixed** locks the body at its initial pose.
- **Free** leaves its motion unconstrained.

Use at most one anchor per body. Keep the tank and other environmental meshes
as colliders; an anchor is a separate object. Export authors USD Physics schemas
and the solver's `grasph:*` metadata. The current joints connect bodies to the
world, not to another moving body.

## Running and preview

Set time, spacing, output name and output folder. **Export Before Run** keeps
the source USD synchronized with the current scene. Quality presets change
resolution and granular solver settings; they preserve sound speed.

**Live Viewport Preview** updates particles and body transforms from compact
solver snapshots. **Pause**, **Stop** and **Reset** control the active run.
Reset restores the starting preview/body state while preserving saved results.
Rigid-body-only scenes do not require a fluid or granular source.

Optional **Write Partial USD Checkpoints** publishes an importable partial
animation during the run. It is disabled by default to avoid repeated USD I/O.
Use **Import Checkpoint** during a run or **Import Last Result** afterward.
Keep the source USD alongside its result overlay at the expected relative path.

**Continue From Saved Fluid** resumes a compatible NPZ fluid state. Start a new
run after changing the fluid layout or initialization; restart compatibility
checks reject a mismatched initial particle arrangement.

## Result display

`grasph_surface.py` provides surface reconstruction and spray/particle controls;
`grasph_granular_display.py` displays granular points as spheres. These modules
are part of the integration and must remain next to `grasph_control.py`.

Use **Create / Reset Splash Surface** for an imported fluid cache. Mesh Radius,
Voxel Size, Threshold, spray separation and velocity stretch affect display
only. They do not change particle positions or forces. Meshing does not clip
against colliders, so the reconstructed surface may visually extend into walls.
Results containing particle velocities can use velocity-aligned splash trails;
older results without them use the round fallback.

Collider wire overlays are viewport-only, hidden from rendering and excluded
from scene export. The integration smoke test is documented in the
[project README](../README.md#development).
