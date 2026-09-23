# GraSPH Blender tools

## Scene controls and simulation

Install `grasph_control.py` as a legacy Blender add-on, or open it in the Text
Editor and choose **Run Script**. The 3D View sidebar then contains a **GraSPH**
tab with entity, simulation and result-display panels:

- **GraSPH Entities** assigns Fluid Volume, Granular Source, Sink, Static
  Collider, Kinematic Collider, Dynamic Hinge or Visual Only to mesh objects.
- **GraSPH Simulation** validates and exports a generic USDC, launches
  `grasph.py` as an external process, records a log beside the result and can
  import the finished result back into Blender.

The simulation panel exposes time, particle spacing, output FPS/folder/name and
an Advanced Settings section with sound speed, free-surface mode, collision,
wall pressure/continuity, neighbor backend and threads, droplet handling and
wall-search mode. **Export Before Run** keeps the USDC synchronized with the
current Blender scene. Press Escape while the modal run is active to stop it.

For a dynamic hinge, place the Blender object origin at the axle center, select
the world X/Y/Z axis and set inertia, damping and maximum angular speed. The
export button writes the corresponding `grasph:*` attributes directly into the
USDC. No scene-specific build script is involved.

The first version supports axis-aligned world-space hinges. Fluid generators
and arbitrary/standard USD Physics joint orientations are planned extensions.

## Granular volume and flow sources (v0.5.1)

Assign **Granular Source** to a box-shaped mesh and choose **Granular Volume**
for an initial packed volume or **Granular Flow Source** for timed continuous
flow. A flow source exposes requested grains/second, world-space initial
velocity and start/end times. Its box is the outlet: grains are introduced on
the face pointed to by the largest component of Initial Velocity. A blocked
outlet throttles the actual flow rather than creating overlapping grains.

Both source types use Particle Radius, Material Density, Friction and
Restitution. Running the scene selects the external granular DEM/PBD path
automatically. It provides gravity, sphere/sphere and mesh contacts, sinks,
kinematic colliders and collision impulses on Dynamic Hinges. **Import Last
Result** creates a Geometry Nodes sphere display; its radius remains editable
under **GraSPH Granular Display**.

Wall friction uses an impulse-limited Coulomb response. Grains carry angular
velocity and solid-sphere inertia, so surface friction converts sliding into
rolling. Grain and collider **Rolling Resistance** controls the decay of spin.
Every Static Collider, Kinematic Collider, Dynamic Hinge and Dynamic Body has
its own Surface Friction, Surface Restitution and Rolling Resistance.

**Dynamic Body** adds freely integrated translation and rotation with mass,
local inertia diagonal and damping. Set its origin to the center of mass. The
current MVP uses the displayed **Collision Radius** as a spherical envelope for
body-versus-environment contact; grains still collide against its actual mesh
distance field. Result transforms are animated in the USD overlay.

This milestone supports box volumes and box-shaped flow outlets. Arbitrary
mesh-volume filling and two-way fluid/granular coupling are subsequent steps.
A scene must currently contain fluid sources or granular sources, not both.

## Fluid surface preview

Open `output/sharp_TankAnim_dp20_surface.blend` for the first 20 mm mesh test.
It references the original result USD, so keep that USD and its source scene
in place. The fluid animation spans frames 0–72 (3 seconds at 24 fps).

Select `GraSPH_Result_Surface` and open its Geometry Nodes modifier:

- Show Surface only: continuous surface preview.
- Show Particles only: debug spheres at the original particle positions.
- Both enabled: surface plus particle overlay.

Mesh Radius, Voxel Size and Threshold affect only visualization. Particle
Display Radius is independent. Initial values are radius=1.3*dp,
voxel size=0.4*dp, density=1 and threshold=0.6. These are preview defaults,
not a calibrated SPH density isosurface or a volume-conserving reconstruction.
Smaller voxels increase detail and evaluation cost. Sparse spray can remain
separate droplets. Meshing does not clip against colliders and may extend into
walls; it does not change particle positions or solver forces.

New result USD files include standard per-particle `velocities`. With
**Splashy Droplets** enabled, fast sparse particles receive three short
velocity-aligned trailing samples before Points to Volume. **Splash Speed** is
the activation threshold in m/s, **Velocity Stretch** is a trail-time scale in
seconds, and **Maximum Stretch** caps the visual trail length. This only changes
surface reconstruction; **Fast Droplet Radius** narrows the fast kernels as a
fraction of the normal mesh radius. It does not alter simulated positions or forces. Older
result files without velocities automatically retain the round fallback.

The Geometry Nodes graph uses a hidden USD-cached point-cloud copy as input,
then Points to Volume, Volume to Mesh, Shade Smooth and a water material.
The original imported cloud remains available. A separate mesh object hosts
the result so it can be evaluated/exported as a mesh. Surface reconstruction
runs when Blender evaluates the current frame; it is not a baked mesh cache.

For another result, run `grasph_surface.py` in Blender's Text Editor. Open the
3D View sidebar (`N`) > GraSPH, enter the dp used in the simulation and import
the result USD. For an already imported Point Cloud, select it and use
Create / Reset Surface. Reset restores the preview defaults. The script can
also be installed as a legacy Blender add-on. Loading the generated .blend
does not automatically register the sidebar; modifier controls work without it.

To create a separate .blend in the background:

```powershell
& 'C:\Program Files\Blender Foundation\Blender 5.0\blender.exe' `
  --background --factory-startup --python-exit-code 1 `
  --python blender/build_surface_scene.py -- `
  output/sharp_TankAnim_dp20_3s.usdc --dp 0.02 `
  --output output/another_surface.blend --frame 24 --end 72
```

The output must be a new filename. dp is explicit because visual USD point
widths may be changed independently of simulation spacing. Multiple imported
point clouds receive separate surfaces with the supplied common dp.

Tested in Blender 5.0.1 and 4.5.3 LTS: frames 0, 24, 48 and 72 produced nonempty meshes;
surface/particle/both modes produced the expected mesh and 1539 instances.

Blender documentation: [Points to Volume](https://docs.blender.org/manual/en/5.0/modeling/geometry_nodes/point/points_to_volume.html).
