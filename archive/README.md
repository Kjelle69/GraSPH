# Legacy diagnostic line

Versions v0.38-v0.52 were benchmark/diagnostic branches used to isolate the free-surface atmospheric traction problem.

Important frozen conclusions:

- WCSPH continuity / pairwise reduced-pressure core passed the fluid-only and wall-coupled acoustic tests (v0.15-v0.17).
- finite-face AIR-vs-SOLID topology passed cross-geometry slosh tests (v0.35-v0.37).
- anchored-band atmospheric traction is **experimental**, not resolution-independent:
  - dp=20 mm loses projected free-surface depth moment dynamically;
  - dp=10 mm gains it dynamically;
  - changing only band depth, half-space kernel magnitude, PCA normals, or angle-gating did not provide a common resolution-independent fix.

Do not turn these diagnostics back into the main launcher. Future free-surface work belongs behind the `grasph_free_surface.py` interface and must pass both dp20/dp10 benchmark regressions before becoming the default.
