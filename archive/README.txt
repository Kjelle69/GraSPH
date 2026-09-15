GraSPH v0.52 - ANGLE-GATED ANCHOR-NORMAL REPLAY
================================================

Diagnostic only. No physics is advanced.

v0.51 showed that local PCA normals improve dp=10 mm strongly but fail badly
for dp=20 mm, with some raw->PCA rotations near 90 degrees.

v0.52 changes ONE thing only: sweep a maximum accepted PCA rotation.
If the PCA normal differs from the raw AIR anchor normal by more than the gate,
that seed falls back exactly to the raw normal. Shell, band, anchor identity,
interface point, J(s), wall completion, SOLID projection and EOS are unchanged.

Run:
  py grasph.py DamBreakRitter.usda output\ritter_surface_moment_v49_dp20.npz
  py grasph.py DamBreakRitter.usda output\ritter_surface_moment_v49_dp10.npz

Decision:
If one bounded gate improves BOTH resolutions, pursue a principled regularized
normal estimator. If no common gate works, do not tune the gate as production
physics; move to conservative surface quadrature/reconstruction.
