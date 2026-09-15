# Benchmarks

Benchmarks are now **clients of the generic GraSPH solver**. They must not own or fork the timestep physics.

- `ritter.py` runs `examples/DamBreakRitter.usda` and adds the Ritter front report after the generic run.
- `slosh.py` runs `examples/SloshTest.usda` with the historical v0.37 half-sine +X body-acceleration pulse.
- Acoustic and moving-gate regressions remain in the legacy v15-v36 packages for the moment. They should be migrated as hooks/tests, not copied back into `grasph.py`.

This separation is intentional: a benchmark may measure GraSPH, but it does not define GraSPH.
