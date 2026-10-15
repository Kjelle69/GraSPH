# Blender controls

Open `grasph_control.py` in Blender's Text Editor and run it. The 3D View's
**GraSPH** sidebar provides scene roles, simulation controls and result display.
The solver uses the external Python configured under **Output & Preview**.
Project-root discovery is automatic when the script's location is available;
**Project & Scene** allows an explicit project directory when needed.

## Scene roles

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
