# GraSPH fluid/granular coupling (Blender controls v0.7.3)

Fluid Volume and Granular Source may now be used in the same scene. Spherical
grains sample the local SPH velocity and density, receive buoyancy and
quadratic drag, and scatter the equal-and-opposite force back to nearby fluid
particles. Grain contact remains the fast impulse DEM/PBD model. Mixed results
contain both `/GraSPH_Result` and `/GraSPH_GranularResult` point animations.

v0.7.1 adds a coarse-field anti-tunnelling collision skin for grains smaller
than the SPH collider voxel size. Partial-neighbour submergence and an
explicit drag-impulse limiter make entry into sparse fluid less explosive.
v0.7.2 replaces oversized collision skins with swept grain-to-mesh contact and
rejects mixed runs with fewer than eight fluid particles.
v0.7.3 restores Fluid Display controls for legacy or renamed surface modifiers.

Dynamic Body now supports a sampled convex mesh collider instead of requiring
a spherical envelope. In Blender choose `Dynamic Body` and `Convex Mesh`; the
authored mesh must be convex. Static and kinematic collision meshes may still
be arbitrary triangle meshes. `Collision Margin` controls the thin contact
skin.

For SPH scenes, enable `Advanced Settings > Fluid / Solid Coupling` to apply
fluid pressure and collision reactions to Dynamic Body objects and write their
translation/rotation animation to the result USD. With the option disabled the
bodies remain fixed fluid colliders. Granular and fluid source particles are
still separate simulation modes.

Blender can now author initial `Granular Volume` sources and timed `Granular
Flow Source` outlets with particle radius, material density, friction,
restitution, flow rate and initial velocity. The external solver automatically
runs a spherical DEM/PBD path with gravity, grain contacts, mesh collision,
sinks and collision-driven hinges. Results are written as
`/GraSPH_GranularResult` USD Points and imported with a Geometry Nodes sphere display. See
[`blender/README.md`](blender/README.md) for the current scope and workflow.

Granular spheres now carry angular velocity and solid-sphere inertia. Coulomb
contact impulses therefore transfer sliding motion into spin, while separate
rolling resistance controls how quickly rolling decays. Static, kinematic,
hinged and free dynamic colliders expose friction, restitution and rolling
resistance as authored `grasph:*` surface properties.

`Dynamic Body` is the first free 6-DOF rigid-body path: mass, inertia, damping
and simulated translation/rotation are written back to USD. Its environment
collision envelope is currently an authored sphere radius; mesh/convex contact
is a planned refinement.

---

# GraSPH fluid surface preview in Blender

The first Geometry Nodes meshing workflow is available in
[`blender/README.md`](blender/README.md). It retains the animated USD particle
cache and adds independent surface/particle display controls. SPH physics is
unchanged.

---

# Dynamic paddle-wheel axle

`examples/PaddleWheel/PaddleWheel.usdc` now marks the wheel as a free X-axis
hinge. The runtime keeps its axle center fixed and integrates angular motion
from SPH wall-pressure reaction torque and geometric collision impulses. Wall
particles, normals, distance-field queries and finite-face geometry follow the
rotation. The result USD animates the wheel transform, while the NPZ records
`hinge_angles` and `hinge_angular_velocities`.

The hinge parameters are authored as `grasph:hingeEnabled`,
`grasph:axleCenter`, `grasph:momentOfInertia`, `grasph:angularDamping` and
`grasph:maxAngularSpeed`. Authored collider animation remains translation-only;
the solver-generated hinge rotation is the supported rotational path.

---

# GraSPH v0.54 FIX7 — cached collider neighbour trees

FIX7 is a CPU-only change. Each collider's SPH wall particles keep a SciPy
cKDTree in their reference pose. Fluid query positions are translated into that
pose every timestep; world-space wall particles, facing anchors and wall
velocities continue to follow the source USD animation. Candidate searches have
a conservative floating-point border, with ambiguous support-distance pairs
checked in world coordinates. The strict support boundary is preserved.

Trees belong to a single simulation and are not shared between scenes.
Without SciPy, the original grid search remains available.

Default: `--wall-search cached`. Use `--wall-search rebuild` to run the
previous reference search for A/B comparisons. Existing callers of
`build_wall_interactions` without a cache retain the reference implementation.

Validation on all 91 saved states of the 20 mm, 3 s TankAnim test found exact
accepted pairs and pair geometry, matching surface/band/anchor classifications,
and wall/free-surface force differences at floating-point rounding level.
An alternating-order benchmark (25 repeats at four saved states) measured
roughly 7–14% faster full wall-interaction construction. This is a phase
benchmark, not a measured end-to-end speedup or a guarantee for other scenes.

A short 30 mm A/B simulation to 0.8 s (including animated Gate motion) produced
identical saved positions, densities and surface/band/detached masks; other
differences were at rounding level. The cached wall phase took 26.24 s versus
26.84 s for rebuild, but total time was 60.75 s versus 56.10 s because other
phases took longer. These single runs do not demonstrate an end-to-end gain.
Reports are saved in `output/wall_search_benchmark.json` and
`output/wall_search_simulation_check.json`.

Reproduce checks:

```powershell
py -m unittest discover -s tests -v
py benchmarks/wall_search.py output/sharp_TankAnim_dp20_3s.npz --repeats 25
```

---

# GraSPH v0.54 FIX6 — detached fluid clusters + adaptive wall-neighbour search

FIX6 is based directly on FIX5.  USD kinematics, the anchored-band free-surface
model, collision and output layering are unchanged.

## Why FIX5 did not trigger the droplet fallback

The coarse 30 mm replay finished with `Detached now/ever: 0 / 0`, even though
the particle cloud had split into small disconnected clusters.  The reason is
simple: a spray blob can contain several particles, so every particle can still
have neighbours even though the entire blob is disconnected from the main
liquid.  `neighbor_count == 0` was therefore too narrow.

## FIX6 physics change

At every step GraSPH builds connected components of the existing fluid-neighbour
graph.  The largest connected component keeps the validated well-balanced
hydrostatic reference.  Any disconnected component temporarily uses ordinary
WCSPH with `p_ref=0`, `rho_ref=rho0`, `depth_ref=0`, i.e. full gravity.  When the
blob reconnects to the main liquid, the well-balanced reference is used again.

This is intended to produce the desired behaviour:

    spray -> wall impact -> slide/fall -> reconnect to liquid

Use the old strict single-particle rule for A/B comparison with:

    py grasph.py TankAnim.usdc --time 3.0 --dp 0.03 --droplet-mode zero

The default is now:

    --droplet-mode components

The ticker prints both `detached=N` and `comps=K main=M`.

## FIX6 CPU change

FIX5 profiling showed that wall interactions dominated runtime.  FIX6 changes
only the *search implementation*, not the accepted geometric pairs:

- <=2500 fluid particles: scipy `sparse_distance_matrix` cross search.  This
  avoids creating hundreds/thousands of Python neighbour lists and avoids
  paying thread-launch overhead every timestep.
- >2500 fluid particles: the existing parallel `query_ball_point` path remains,
  using `--threads` (`0` = all logical CPUs).
- the final strict `r < support` test is identical in both paths.

For the next useful test:

    py grasph.py TankAnim.usdc --time 3.0 --dp 0.02

For a very fast pipeline check:

    py grasph.py TankAnim.usdc --time 3.0 --dp 0.03

Do not judge speed by Windows total CPU percentage alone: the 30 mm case has
only 432 fluid particles, so forcing 32 threads can be slower than a leaner
compiled search.  Fine runs automatically return to the parallel path.

---

# GraSPH v0.54 FIX4 — Blender spatial metadata preservation

FIX4 keeps the FIX3 kinematic-physics motion table and adds one output-only correction:

- the result overlay now copies `upAxis` and `metersPerUnit` from the source USD stage;
- this prevents Blender from importing `GraSPH_Result` with the default USD Y-up / centimeter conversion (`Rotation X = 90°`, `Scale = 0.01`);
- SPH physics is unchanged.

Expected output for a Blender meter/Z-up scene includes:

```text
USD spatial metadata: upAxis=Z, metersPerUnit=1
```

---

# GraSPH v0.54 — generic USD core + Blender/USD kinematic colliders

v0.54 keeps the v0.53 generic solver architecture and adds one deliberately
small runtime feature: **translation-only animated colliders from OpenUSD**.

This is intended for the existing Tank/Gate model: animate `Gate` in Blender
with ordinary object-location keyframes, export USD with animation, then run
GraSPH on that USD. No hard-coded gate timing is needed in the solver.

## Blender workflow

1. Open the existing Tank/Gate/FluidVolume scene.
2. Keep `Tank` and `FluidVolume` static.
3. Select `Gate`.
4. At the first animation frame, leave the gate closed and insert a **Location** keyframe (`I` -> Location).
5. Move to the desired opening frame, move the gate upward in **Z**, and insert another Location keyframe.
6. Optionally add a hold keyframe before opening and/or after it is fully open.
7. Preview the motion normally in Blender.
8. Export as USD (`.usda` or `.usdc`) with **Animation enabled**.
9. Run the exported file with GraSPH.

Example:

```powershell
py grasph.py TankGateAnimated.usda --time 1.0
```

The generated `output/TankGateAnimated_grasph.usdc` contains the original USD
scene animation plus the GraSPH particle animation on the same source timeline.

## Timing

GraSPH interprets simulation time in seconds starting at the USD stage start
timecode:

`source_timecode = stage_start + simulation_seconds * timeCodesPerSecond`

So if Blender exports at 24 fps, a gate motion from frame 1 to frame 13 lasts
about 0.5 s. If you want easy millisecond-style timing, setting Blender to
100 fps is convenient: frame 1 is t=0, frame 6 is t=0.05 s, frame 28 is
t=0.27 s.

## Scope of v0.54

Supported now:
- object/parent translation animation from USD
- solver-driven single-axis hinge rotation from fluid forces
- static and translated distance-field collision without DF rebuild
- collision response in the moving-wall frame
- SPH wall particles translated with the collider
- wall continuity using collider velocity
- free-surface solid geometry translated with the collider
- source USD timing preserved in the result USDC

Not supported yet:
- authored collider rotation animation
- animated scale
- deforming collider mesh points
- unconstrained six-degree-of-freedom rigid bodies

If an animated collider rotates or scales, v0.54 stops with a clear error
instead of silently using incorrect collision geometry.

## Existing generic commands

```powershell
py grasph.py TankTest.usda --time 0.5
py benchmarks\slosh.py --time 1.0
```

Fluid source convention remains unchanged: meshes named `FluidVolume` (or with
`grasph:role = "fluid"`) spawn fluid; other meshes are colliders unless marked
`grasph:role = "ignore"`.

The anchored-band atmospheric boundary is still the current experimental
free-surface model. v0.54 changes kinematic scene handling, not that physics.

## FIX1 — preserve Blender animation in result USDC

The result writer now exports `stage.Flatten()` instead of only the USD root
layer. This makes the result self-contained and preserves animation authored
through USD sublayers/references together with the GraSPH particle animation.
Physics is unchanged.


## FIX2 — result USD is now a true USD simulation overlay

FIX1 flattening still allowed Blender-authored collider animation to collapse to
the object's default/current pose in some imports. FIX2 no longer rewrites the
source scene at all. The generated `*_grasph.usdc` is a stronger USD layer that
**sublayers the original animated source USD verbatim** and adds only
`/GraSPH_Result` point samples.

This is both more robust and more USD-native: source scene/animation and
simulation result remain separate layers.

Important: keep the original source USD in place relative to the `output/`
folder. The result USDC references it. A self-contained packaging/export option
can be added later.

Quick verification:
1. Open the original animated USD in Blender and verify Gate is closed at its
   first frame.
2. Run GraSPH.
3. Open the new `output/*_grasph.usdc`; Gate should now have exactly the same
   source animation while `/GraSPH_Result` carries the fluid animation.

## FIX3 — kinematic motion is now a solver input, not just an output animation

The previous FIX2 only made the result USD preserve the Blender animation.  A
5 s run that stayed at `vmax≈0`, `shell=0` proved that this was not sufficient:
visual Gate motion and the collision/boundary motion used by physics had become
two separate things.

FIX3 samples the *composed world transform* of every collider at the source
USD frame cadence before the SPH run.  Parent-Xform animation from Blender is
included.  The solver then interpolates that sampled translation table for
collision fields, SPH wall particles, wall velocity, and finite-face solid
geometry.

Before any long simulation, run:

```powershell
py grasph.py TankAnim.usdc --inspect-motion
```

You should see `Animated/kinematic colliders used by PHYSICS:` and a non-zero
Gate travel plus a five-point translation table. If Gate is absent or all
translations are zero, do not start SPH — the source USD animation has not
been detected.

During a normal run the ticker includes `kinMax=...m`, and the final summary
prints `Kinematic max travel seen`.  This makes it impossible for a visually
animated result USD to masquerade as a kinematic physics run.

---

## FIX5 - detached droplets + CPU acceleration

FIX5 makes one isolated physics change and several implementation-only speedups.

### Detached-droplet gravity fallback

A fluid particle with **zero fluid neighbours** is no longer treated as part of
the original hydrostatic reference column. For that particle only, GraSPH uses
`rho_ref=rho0`, `p_ref=0`, `depth_ref=0`, so ordinary gravity acts again. This is
intended for detached spray/droplets that hit a wall or ceiling: they should
fall/slide back instead of being artificially suspended by the well-balanced
reference split.

Default is deliberately strict: only `neighbor_count == 0`. The diagnostic
cutoff is exposed as `--droplet-neighbors N`, but do not raise it unless a
separate test justifies doing so.

### CPU acceleration

If SciPy is installed, `--neighbor-backend auto` uses `scipy.spatial.cKDTree`.
Cross-set fluid/wall radius queries use all logical CPU cores by default.
The pure legacy grid remains available with `--neighbor-backend grid`.

Install the optional accelerator if needed:

    py -m pip install scipy

Use all logical CPU cores for cKDTree cross queries (default):

    py grasph.py TankAnim.usdc --time 3.0 --dp 0.03

Force legacy neighbor search for A/B comparison:

    py grasph.py TankAnim.usdc --time 3.0 --dp 0.03 --neighbor-backend grid

Other speedups keep the mathematical operators unchanged: vector scatter-adds
replace slow `(N,3) np.add.at` calls, solid-face testing gets a conservative
expanded-AABB pre-cull, and the 3x3 solid-projection eigensolve + surface-band
anchor selection are batched/vectorized.

The end-of-run report now prints CPU phase timing so the next real bottleneck
can be attacked with evidence rather than guessing.
