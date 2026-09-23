# Paddle wheel channel

`PaddleWheel.blend` is a separate copy of the tank test, built with
`blender/build_paddle_wheel_scene.py` through the live MTEC Blender bridge.
`PaddleWheel.usdc` is the simulation source. The gate animation is retained.

- The channel is open at Y = -0.55 m; its upstream end is extended to Y = 0.80 m.
- The initial fluid extends from the gate at Y = 0.291 m to Y = 0.77 m.
- The eight-blade wheel is centered at (0, -0.30, 0.125) m and has a free X axle.
  Its blade tips reach about 0.079 m from the axle, leaving side passages.
  `grasph:hingeEnabled` locks its translation and permits rotation about X.
  Fluid pressure and collision impulses both contribute torque; the resulting
  angle is animated on `/root/PaddleWheel` in the output USD and recorded in
  `hinge_angles` / `hinge_angular_velocities` in the NPZ file.
- `SinkVolume` spans Y = -0.82 .. -0.555 m. Particle centers entering this
  volume are removed after collision response. The sink itself has no wall
  boundary or collision force.

Quick coarse run:

```powershell
py grasph.py examples/PaddleWheel/PaddleWheel.usdc --time 3.0 --dp 0.05 --name paddle_passage_dp50_3s
```

For scenes with sinks, `active_counts` and `active_particle_ids` in the NPZ
identify live particles at each snapshot. Particle arrays retain the initial
capacity, with removed rows padded with NaN (masks with zero and IDs with -1).
The USD Points samples and widths contain only live particles.

The 50 mm, 3 s check completed 10,000 steps without a solver instability.
It removed 75 of 189 particles; the first removal was recorded at 1.567 s.
All 91 USD point and width samples matched `active_counts`. This coarse
flow-through check predates the dynamic axle and is not a validated prediction
of wheel torque.
