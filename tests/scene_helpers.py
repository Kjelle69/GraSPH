"""Small USD fixtures generated in temporary directories by the test suite."""

from pxr import Gf, Sdf, Usd, UsdGeom


def box(stage, path, lo, hi, role):
    points = [(lo[0], lo[1], lo[2]), (hi[0], lo[1], lo[2]),
              (hi[0], hi[1], lo[2]), (lo[0], hi[1], lo[2]),
              (lo[0], lo[1], hi[2]), (hi[0], lo[1], hi[2]),
              (hi[0], hi[1], hi[2]), (lo[0], hi[1], hi[2])]
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
             (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
    mesh.CreateFaceVertexCountsAttr([4] * 6)
    mesh.CreateFaceVertexIndicesAttr([i for face in faces for i in face])
    prim = mesh.GetPrim()
    prim.CreateAttribute('grasph:role', Sdf.ValueTypeNames.String).Set(role)
    return prim


def make_scene(path, mass):
    """Basin with a partly immersed body and a cavity in the initial fluid."""
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    sources = [((-0.2, -0.2, 0.04), (0.2, 0.2, 0.22)),
               ((-0.2, -0.2, 0.22), (-0.08, 0.2, 0.32)),
               ((0.08, -0.2, 0.22), (0.2, 0.2, 0.32)),
               ((-0.08, -0.2, 0.22), (0.08, -0.08, 0.32)),
               ((-0.08, 0.08, 0.22), (0.08, 0.2, 0.32))]
    for i, (lo, hi) in enumerate(sources):
        box(stage, f'/Fluid{i}', lo, hi, 'fluid')
    walls = [((-0.24, -0.24, -0.04), (0.24, 0.24, 0.0)),
             ((-0.24, -0.24, 0.0), (-0.2, 0.24, 0.5)),
             ((0.2, -0.24, 0.0), (0.24, 0.24, 0.5)),
             ((-0.2, -0.24, 0.0), (0.2, -0.2, 0.5)),
             ((-0.2, 0.2, 0.0), (0.2, 0.24, 0.5))]
    for i, (lo, hi) in enumerate(walls):
        box(stage, f'/Wall{i}', lo, hi, 'collider')
    driver = UsdGeom.Xform.Define(stage, '/Body')
    driver.AddTranslateOp().Set(Gf.Vec3d(0.0))
    driver.AddRotateXYZOp().Set(Gf.Vec3f(0.0))
    body = box(stage, '/Body/Geom', (-0.08, -0.08, 0.22),
               (0.08, 0.08, 0.38), 'collider')
    for name, kind, value in (
        ('dynamicBodyEnabled', Sdf.ValueTypeNames.Bool, True),
        ('bodyCenter', Sdf.ValueTypeNames.Double3, Gf.Vec3d(0, 0, 0.3)),
        ('mass', Sdf.ValueTypeNames.Double, mass),
        ('inertiaDiagonal', Sdf.ValueTypeNames.Double3, Gf.Vec3d(mass * 0.16**2 / 6)),
        ('collisionShape', Sdf.ValueTypeNames.String, 'convex')):
        body.CreateAttribute('grasph:' + name, kind).Set(value)
    stage.GetRootLayer().Save()
