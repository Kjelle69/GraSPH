from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Gf


@dataclass
class MeshData:
    path: str
    vertices: np.ndarray
    triangles: np.ndarray
    bounds_min: np.ndarray
    bounds_max: np.ndarray


@dataclass
class HingeData:
    path: str
    driver_path: str
    axis: np.ndarray
    center: np.ndarray
    moment_of_inertia: float
    angular_damping: float
    max_angular_speed: float


@dataclass
class GranularSourceData:
    mesh: MeshData
    source_mode: str
    particle_radius: float
    density: float
    friction: float
    restitution: float
    rolling_friction: float
    flow_rate: float
    initial_velocity: np.ndarray
    start_time: float
    end_time: float


@dataclass
class DynamicBodyData:
    path: str
    driver_path: str
    center: np.ndarray
    mass: float
    inertia: np.ndarray
    linear_damping: float
    angular_damping: float
    collision_shape: str
    collision_radius: float
    collision_margin: float
    collision_samples: np.ndarray
    joint_type: str = "free"
    joint_axis: np.ndarray | None = None
    joint_anchor: np.ndarray | None = None


@dataclass
class SceneData:
    stage: object
    fluid_sources: list
    granular_sources: list
    colliders: list
    sinks: list
    source_file: str
    start_time_code: float
    end_time_code: float
    time_codes_per_second: float
    frames_per_second: float
    reference_linear: dict
    reference_translation: dict
    motion_time_codes: dict
    motion_translations: dict
    motion_driver_paths: dict
    hinges: dict
    collider_materials: dict
    dynamic_bodies: dict

    def sim_seconds_to_time_code(self, t_seconds):
        return float(self.start_time_code + float(t_seconds) * self.time_codes_per_second)

    def collider_motion(self, t_seconds, linear_tol=1.0e-6):
        """Return translation deltas from the reference pose.

        FIX3 uses a motion table sampled from the composed USD stage at load
        time.  This deliberately decouples the solver loop from Blender's
        authored xform-op layout (parent Xform, matrix op, translate op, etc.)
        and makes the exact motion used by physics inspectable before a long run.
        """
        tc = self.sim_seconds_to_time_code(t_seconds)
        out = {}
        for mesh in self.colliders:
            times = self.motion_time_codes.get(mesh.path)
            vals = self.motion_translations.get(mesh.path)
            if times is None or vals is None or len(times) == 0:
                out[mesh.path] = np.zeros(3, dtype=np.float64)
                continue
            if tc <= times[0]:
                out[mesh.path] = vals[0].copy()
            elif tc >= times[-1]:
                out[mesh.path] = vals[-1].copy()
            else:
                k = int(np.searchsorted(times, tc, side="right"))
                t0, t1 = float(times[k-1]), float(times[k])
                a = 0.0 if abs(t1-t0) < 1e-15 else (tc-t0)/(t1-t0)
                out[mesh.path] = (1.0-a)*vals[k-1] + a*vals[k]
        return out

    def animated_collider_paths(self, tol=1.0e-10):
        paths = []
        for mesh in self.colliders:
            vals = self.motion_translations.get(mesh.path)
            if vals is not None and len(vals) and float(np.max(np.linalg.norm(vals, axis=1))) > tol:
                paths.append(mesh.path)
        return paths

    def print_motion_preflight(self):
        print()
        print("========================================")
        print("USD KINEMATIC PREFLIGHT (FIX3)")
        print("========================================")
        animated = self.animated_collider_paths()
        if not animated:
            print("Animated colliders: NONE")
            if self.hinges or self.dynamic_bodies:
                print(
                    f"Authored kinematic motion: none; {len(self.hinges)} dynamic "
                    f"hinge(s) and {len(self.dynamic_bodies)} dynamic body/bodies "
                    "will be integrated by the solver."
                )
            else:
                print("Physics will treat every collider as static.")
            return
        span = max(0.0, (self.end_time_code-self.start_time_code)/self.time_codes_per_second)
        probes = [0.0, .25*span, .5*span, .75*span, span]
        print(f"Animated colliders: {len(animated)}")
        print(f"Source span       : {span:.6f} s")
        for path in animated:
            print(f"  {path}")
            driver = self.motion_driver_paths.get(path, path)
            print(f"    xform driver : {driver}")
            for tt in probes:
                d = self.collider_motion(tt)[path]
                print(f"    t={tt:8.4f}s  d=({d[0]:+.6f}, {d[1]:+.6f}, {d[2]:+.6f}) m")



def _world_points(mesh_prim, time_code):
    mesh = UsdGeom.Mesh(mesh_prim)
    points = mesh.GetPointsAttr().Get(time_code) or []
    xcache = UsdGeom.XformCache(time_code)
    xf = xcache.GetLocalToWorldTransform(mesh_prim)
    out = []
    for p in points:
        wp = xf.Transform(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))
        out.append((float(wp[0]), float(wp[1]), float(wp[2])))
    return np.asarray(out, dtype=np.float64)


def _triangulate(mesh_prim, vertices, time_code):
    mesh = UsdGeom.Mesh(mesh_prim)
    counts = mesh.GetFaceVertexCountsAttr().Get(time_code) or []
    indices = mesh.GetFaceVertexIndicesAttr().Get(time_code) or []
    triangles = []
    cursor = 0
    for face_index, count in enumerate(counts):
        count = int(count)
        face = [int(v) for v in indices[cursor:cursor + count]]
        cursor += count
        if count < 3:
            continue
        points = np.asarray(vertices[face], dtype=np.float64)
        # USD permits concave n-gons. A fan from vertex zero fills concave
        # cut-outs (e.g. an open tank rim), creating invisible collision lids.
        # Ear clipping in the dominant plane preserves the actual polygon.
        normal = np.zeros(3, dtype=np.float64)
        for a, b in zip(points, np.roll(points, -1, axis=0)):
            normal += np.array((
                (a[1] - b[1]) * (a[2] + b[2]),
                (a[2] - b[2]) * (a[0] + b[0]),
                (a[0] - b[0]) * (a[1] + b[1]),
            ))
        drop_axis = int(np.argmax(np.abs(normal)))
        projected = np.delete(points, drop_axis, axis=1)
        span = max(float(np.ptp(projected, axis=0).max()), 1.0)
        eps = span * span * 1.0e-12

        def cross2(a, b, c):
            ab = b - a
            ac = c - a
            return ab[0] * ac[1] - ab[1] * ac[0]

        # Remove consecutive duplicate and collinear corners before clipping.
        ring = list(range(count))
        changed = True
        while changed and len(ring) > 3:
            changed = False
            for j, current in enumerate(ring):
                previous = ring[j - 1]
                following = ring[(j + 1) % len(ring)]
                if (np.linalg.norm(projected[current] - projected[previous]) <= eps
                        or abs(cross2(projected[previous], projected[current],
                                      projected[following])) <= eps):
                    ring.pop(j)
                    changed = True
                    break

        area2 = sum(
            projected[ring[j], 0] * projected[ring[(j + 1) % len(ring)], 1]
            - projected[ring[(j + 1) % len(ring)], 0] * projected[ring[j], 1]
            for j in range(len(ring))
        )
        if abs(area2) <= eps:
            raise ValueError(
                f"USD mesh {mesh_prim.GetPath()} face {face_index} is degenerate"
            )
        orientation = 1.0 if area2 > 0.0 else -1.0
        face_triangles = []
        while len(ring) > 3:
            ear_found = False
            for j, current in enumerate(ring):
                previous = ring[j - 1]
                following = ring[(j + 1) % len(ring)]
                a, b, c = (projected[index] for index in (previous, current, following))
                if orientation * cross2(a, b, c) <= eps:
                    continue
                contains_corner = False
                for other in ring:
                    if other in (previous, current, following):
                        continue
                    p = projected[other]
                    side0 = orientation * cross2(a, b, p)
                    side1 = orientation * cross2(b, c, p)
                    side2 = orientation * cross2(c, a, p)
                    if min(side0, side1, side2) >= -eps:
                        contains_corner = True
                        break
                if contains_corner:
                    continue
                face_triangles.append((face[previous], face[current], face[following]))
                ring.pop(j)
                ear_found = True
                break
            if not ear_found:
                raise ValueError(
                    f"Could not triangulate USD mesh {mesh_prim.GetPath()} "
                    f"face {face_index}; check for self-intersections"
                )
        face_triangles.append(tuple(face[index] for index in ring))
        triangles.extend(
            (vertices[a], vertices[b], vertices[c])
            for a, b, c in face_triangles
        )
    if not triangles:
        return np.empty((0, 3, 3), dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.float64)
    # A reflected Xform (for example Blender objects exported with a negative
    # scale) reverses geometric winding. Preserve the authored USD face
    # orientation in world space so collision normals keep their intended
    # direction. USD also supports explicitly left-handed meshes.
    orientation = mesh.GetOrientationAttr().Get(time_code) or UsdGeom.Tokens.rightHanded
    world_xform = UsdGeom.XformCache(time_code).GetLocalToWorldTransform(mesh_prim)
    reflected = world_xform.GetDeterminant3() < 0.0
    if (orientation == UsdGeom.Tokens.leftHanded) != reflected:
        triangles = triangles[:, (0, 2, 1), :]
    return triangles


def _role_for_prim(prim):
    attr = prim.GetAttribute("grasph:role")
    if attr and attr.HasAuthoredValueOpinion():
        value = attr.Get()
        if value is not None:
            return str(value).strip().lower()

    parts = [p.lower() for p in str(prim.GetPath()).split("/") if p]
    if any(p.startswith("sinkvolume") for p in parts):
        return "sink"
    if any(p == "fluidvolume" or p.startswith("fluidvolume") for p in parts):
        return "fluid"
    return "collider"


def _inherited_attribute(prim, name):
    """Return the nearest authored custom attribute and the prim owning it."""
    current = prim
    while current and current.IsValid():
        attr = current.GetAttribute(name)
        if attr and attr.HasAuthoredValueOpinion():
            value = attr.Get()
            if value is not None:
                return value, current
        current = current.GetParent()
    return None, None


def _nearest_api_ancestor(prim, schema):
    current = prim
    while current and current.IsValid():
        if current.HasAPI(schema):
            return current
        current = current.GetParent()
    return None


def _float_attribute(prim, name, default):
    value, _ = _inherited_attribute(prim, name)
    return float(default if value is None else value)


def _string_attribute(prim, name, default):
    value, _ = _inherited_attribute(prim, name)
    return str(default if value is None else value).strip().lower()


def _vec3_attribute(prim, name, default):
    value, _ = _inherited_attribute(prim, name)
    return np.asarray(default if value is None else value, dtype=np.float64)


def _physics_material_values(prim):
    try:
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
        if not material or not material.GetPrim().IsValid():
            return None
        api = UsdPhysics.MaterialAPI(material.GetPrim())
        friction = api.GetDynamicFrictionAttr().Get()
        if friction is None:
            friction = api.GetStaticFrictionAttr().Get()
        restitution = api.GetRestitutionAttr().Get()
        density = api.GetDensityAttr().Get()
        return {
            "friction": friction, "restitution": restitution, "density": density,
        }
    except Exception:
        return None


def _hinge_for_prim(prim):
    enabled, owner = _inherited_attribute(prim, "grasph:hingeEnabled")
    if not bool(enabled):
        return None

    axis_value, _ = _inherited_attribute(prim, "grasph:axis")
    axis_name = str(axis_value or "X").strip().upper()
    axes = {
        "X": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "Y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "Z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    }
    if axis_name not in axes:
        raise RuntimeError(f"Unsupported hinge axis {axis_name!r} on {prim.GetPath()}")

    center_value, _ = _inherited_attribute(prim, "grasph:axleCenter")
    if center_value is None:
        center_value, _ = _inherited_attribute(prim, "grasph:axle_center")
    if center_value is None:
        raise RuntimeError(f"Hinged collider {prim.GetPath()} has no grasph:axleCenter")

    inertia, _ = _inherited_attribute(prim, "grasph:momentOfInertia")
    damping, _ = _inherited_attribute(prim, "grasph:angularDamping")
    max_speed, _ = _inherited_attribute(prim, "grasph:maxAngularSpeed")
    inertia = float(inertia if inertia is not None else 0.002)
    if not np.isfinite(inertia) or inertia <= 0.0:
        raise RuntimeError(f"Hinged collider {prim.GetPath()} needs positive moment of inertia")

    driver = prim
    while driver and driver.IsValid():
        rotate = driver.GetAttribute("xformOp:rotateXYZ")
        if rotate and rotate.HasAuthoredValueOpinion():
            break
        driver = driver.GetParent()
    if not driver or not driver.IsValid():
        driver = owner

    return HingeData(
        path=str(prim.GetPath()),
        driver_path=str(driver.GetPath()),
        axis=axes[axis_name],
        center=np.asarray(center_value, dtype=np.float64),
        moment_of_inertia=inertia,
        angular_damping=max(0.0, float(damping if damping is not None else 0.002)),
        max_angular_speed=max(0.0, float(max_speed if max_speed is not None else 40.0)),
    )


def _physics_hinge_owners(stage):
    """Collect world-attached standard revolute joints by rigid-body path."""
    result = {}
    axes = {
        "X": np.array((1.0, 0.0, 0.0)),
        "Y": np.array((0.0, 1.0, 0.0)),
        "Z": np.array((0.0, 0.0, 1.0)),
    }
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        joint = UsdPhysics.RevoluteJoint(prim)
        body0 = joint.GetBody0Rel().GetTargets()
        body1 = joint.GetBody1Rel().GetTargets()
        # GraSPH's current hinge model is attached to the world. Two-body
        # joints remain valid USD but require a future compound-body solver.
        if bool(body0) == bool(body1):
            continue
        body_path = body1[0] if body1 else body0[0]
        position_attr = joint.GetLocalPos0Attr() if body1 else joint.GetLocalPos1Attr()
        center = position_attr.Get() or Gf.Vec3f(0.0)
        axis = axes.get(str(joint.GetAxisAttr().Get() or "X").upper())
        if axis is None:
            continue
        owner = stage.GetPrimAtPath(body_path)
        mass_api = UsdPhysics.MassAPI(owner)
        diagonal = mass_api.GetDiagonalInertiaAttr().Get()
        axis_index = int(np.argmax(np.abs(axis)))
        inertia = float(diagonal[axis_index]) if diagonal is not None else 0.002
        result[str(body_path)] = {
            "axis": axis,
            "center": np.asarray(center, dtype=np.float64),
            "inertia": max(inertia, 1.0e-9),
        }
    return result


def _physics_hinge_for_prim(prim, owners):
    current = prim
    while current and current.IsValid():
        data = owners.get(str(current.GetPath()))
        if data is not None:
            return HingeData(
                path=str(prim.GetPath()), driver_path=str(current.GetPath()),
                axis=data["axis"].copy(), center=data["center"].copy(),
                moment_of_inertia=data["inertia"], angular_damping=0.002,
                max_angular_speed=40.0,
            )
        current = current.GetParent()
    return None


def _dynamic_body_for_prim(prim, mesh_data):
    enabled, owner = _inherited_attribute(prim, "grasph:dynamicBodyEnabled")
    physics_owner = None
    if not bool(enabled):
        physics_owner = _nearest_api_ancestor(prim, UsdPhysics.RigidBodyAPI)
        if physics_owner is None:
            return None
        rigid = UsdPhysics.RigidBodyAPI(physics_owner)
        if bool(rigid.GetKinematicEnabledAttr().Get() or False):
            return None
        owner = physics_owner
    center, _ = _inherited_attribute(prim, "grasph:bodyCenter")
    mass, _ = _inherited_attribute(prim, "grasph:mass")
    inertia, _ = _inherited_attribute(prim, "grasph:inertiaDiagonal")
    linear_damping, _ = _inherited_attribute(prim, "grasph:linearDamping")
    angular_damping, _ = _inherited_attribute(prim, "grasph:bodyAngularDamping")
    collision_radius, _ = _inherited_attribute(prim, "grasph:collisionRadius")
    collision_shape, _ = _inherited_attribute(prim, "grasph:collisionShape")
    collision_margin, _ = _inherited_attribute(prim, "grasph:collisionMargin")
    joint_type, _ = _inherited_attribute(prim, "grasph:jointType")
    joint_axis, _ = _inherited_attribute(prim, "grasph:jointAxis")
    joint_anchor, _ = _inherited_attribute(prim, "grasph:jointAnchor")
    if physics_owner is not None:
        mass_api = UsdPhysics.MassAPI(physics_owner)
        physics_mass = mass_api.GetMassAttr().Get()
        physics_inertia = mass_api.GetDiagonalInertiaAttr().Get()
        physics_center = mass_api.GetCenterOfMassAttr().Get()
        size = np.maximum(mesh_data.bounds_max - mesh_data.bounds_min, 1.0e-6)
        if mass is None:
            mass = physics_mass
        if mass is None or float(mass) <= 0.0:
            mass = float(np.prod(size) * 1000.0)
        if inertia is None:
            inertia = physics_inertia
        if inertia is None or np.any(np.asarray(inertia) <= 0.0):
            inertia = (float(mass) / 12.0) * np.array((
                size[1] ** 2 + size[2] ** 2,
                size[0] ** 2 + size[2] ** 2,
                size[0] ** 2 + size[1] ** 2,
            ))
        if center is None and physics_center is not None:
            transform = UsdGeom.XformCache().GetLocalToWorldTransform(physics_owner)
            world_center = transform.Transform(Gf.Vec3d(*map(float, physics_center)))
            center = tuple(world_center)
        if center is None:
            center = 0.5 * (mesh_data.bounds_min + mesh_data.bounds_max)
        mesh_collision = UsdPhysics.MeshCollisionAPI(prim)
        approximation = mesh_collision.GetApproximationAttr().Get()
        if collision_shape is None:
            collision_shape = "sphere" if approximation == UsdPhysics.Tokens.boundingSphere else "convex"
        if collision_radius is None:
            collision_radius = 0.5 * float(np.max(size))
    mass = float(1.0 if mass is None else mass)
    inertia = np.asarray((0.1, 0.1, 0.1) if inertia is None else inertia, dtype=np.float64)
    if mass <= 0.0 or np.any(inertia <= 0.0):
        raise RuntimeError(f"Dynamic body {prim.GetPath()} needs positive mass and inertia")
    # Blender authors object transforms on the parent Xform and mesh geometry
    # on a child prim. Metadata is copied to both, so the attribute owner is not
    # necessarily the transform driver. Walk upward to the first prim carrying
    # an authored xform op, just as the hinge path does.
    driver = prim
    while driver and driver.IsValid():
        xformable = UsdGeom.Xformable(driver)
        if xformable and xformable.GetOrderedXformOps():
            break
        driver = driver.GetParent()
    if not driver or not driver.IsValid():
        driver = owner

    # Convex contact samples cover vertices, edge midpoints and face centroids.
    # The solver requires a convex authored mesh; concave decomposition remains
    # a separate compound-body feature.
    vertices = np.asarray(mesh_data.vertices, dtype=np.float64)
    triangles = np.asarray(mesh_data.triangles, dtype=np.float64)
    samples = [vertices]
    if len(triangles):
        samples.append(np.mean(triangles, axis=1))
        samples.append(np.vstack((
            0.5 * (triangles[:, 0] + triangles[:, 1]),
            0.5 * (triangles[:, 1] + triangles[:, 2]),
            0.5 * (triangles[:, 2] + triangles[:, 0]),
        )))
    samples = np.vstack(samples)
    scale = 1.0e-9
    _, unique = np.unique(np.round(samples / scale).astype(np.int64), axis=0, return_index=True)
    samples = samples[np.sort(unique)]

    return DynamicBodyData(
        path=str(prim.GetPath()), driver_path=str(driver.GetPath()),
        center=np.asarray(center if center is not None else (0.0, 0.0, 0.0), dtype=np.float64),
        mass=mass, inertia=inertia,
        linear_damping=max(0.0, float(0.05 if linear_damping is None else linear_damping)),
        angular_damping=max(0.0, float(0.05 if angular_damping is None else angular_damping)),
        collision_shape=str(collision_shape or "convex").strip().lower(),
        collision_radius=max(1.0e-6, float(0.1 if collision_radius is None else collision_radius)),
        collision_margin=max(1.0e-6, float(0.002 if collision_margin is None else collision_margin)),
        collision_samples=samples,
        joint_type=str(joint_type or "free").strip().lower(),
        joint_axis=(None if joint_axis is None else np.asarray(joint_axis, dtype=np.float64)),
        joint_anchor=(None if joint_anchor is None else np.asarray(joint_anchor, dtype=np.float64)),
    )


def _matrix_parts(xf):
    linear = np.array([[float(xf[i][j]) for j in range(3)] for i in range(3)], dtype=np.float64)
    tr = xf.ExtractTranslation()
    translation = np.array([float(tr[0]), float(tr[1]), float(tr[2])], dtype=np.float64)
    return linear, translation


def _xform_has_time_samples(prim):
    try:
        xf = UsdGeom.Xformable(prim)
        if not xf:
            return False
        for op in xf.GetOrderedXformOps():
            attr = op.GetAttr()
            try:
                if attr.GetNumTimeSamples() > 0:
                    return True
            except Exception:
                pass
            try:
                if attr.ValueMightBeTimeVarying():
                    return True
            except Exception:
                pass
    except Exception:
        return False
    return False

def _motion_driver_path(mesh_prim):
    """Nearest ancestor (including self) carrying animated xform samples."""
    p = mesh_prim
    while p and p.IsValid():
        if _xform_has_time_samples(p):
            return str(p.GetPath())
        p = p.GetParent()
    return str(mesh_prim.GetPath())

def _sample_motion_tables(stage, colliders, ref_linear, ref_translation,
                          start_tc, end_tc, tps, fps, hinges=None,
                          dynamic_bodies=None,
                          linear_tol=1e-6):
    # Sample at source-frame cadence. Blender's USD exporter normally bakes
    # object animation at this cadence, including parent-Xform animation.
    frame_step = float(tps/fps) if fps > 0 else 1.0
    if not np.isfinite(frame_step) or frame_step <= 0:
        frame_step = 1.0
    if end_tc <= start_tc + 1e-12:
        sample_tcs = np.asarray([start_tc], dtype=np.float64)
    else:
        n = int(np.floor((end_tc-start_tc)/frame_step + 1e-9)) + 1
        sample_tcs = start_tc + np.arange(n, dtype=np.float64)*frame_step
        if sample_tcs[-1] < end_tc - 1e-9:
            sample_tcs = np.append(sample_tcs, end_tc)
        else:
            sample_tcs[-1] = min(sample_tcs[-1], end_tc)

    motion_times = {}
    motion_vals = {}
    driver_paths = {}
    hinges = hinges or {}
    dynamic_bodies = dynamic_bodies or {}
    for mesh in colliders:
        prim = stage.GetPrimAtPath(mesh.path)
        driver_paths[mesh.path] = _motion_driver_path(prim)
        if mesh.path in hinges or mesh.path in dynamic_bodies:
            # A dynamic hinge owns its rotation from simulation time zero.
            # Imported result USDs may contain old wheel animation; treating
            # that as kinematic motion would both reject the scene and apply
            # two competing rotations. Keep the authored start pose as the
            # reference geometry and discard every authored motion delta.
            motion_times[mesh.path] = sample_tcs.copy()
            motion_vals[mesh.path] = np.zeros((len(sample_tcs), 3), dtype=np.float64)
            continue
        vals=[]
        for tc in sample_tcs:
            cache=UsdGeom.XformCache(Usd.TimeCode(float(tc)))
            xf=cache.GetLocalToWorldTransform(prim)
            linear, tr = _matrix_parts(xf)
            if np.max(np.abs(linear-ref_linear[mesh.path])) > linear_tol:
                raise RuntimeError(
                    f"Collider {mesh.path} changes rotation/scale in USD. "
                    "FIX3 supports translation-only kinematics."
                )
            vals.append(tr-ref_translation[mesh.path])
        motion_times[mesh.path]=sample_tcs.copy()
        motion_vals[mesh.path]=np.asarray(vals,dtype=np.float64)
    return motion_times, motion_vals, driver_paths


def load_scene(usd_file):
    usd_file = str(usd_file)
    print()
    print("========================================")
    print("GraSPH SCENE LOADER")
    print("========================================")
    print()
    print(f"Opening: {usd_file}")

    stage = Usd.Stage.Open(usd_file)
    if stage is None:
        raise RuntimeError(f"Could not open USD stage: {usd_file}")

    start_tc = float(stage.GetStartTimeCode())
    end_tc = float(stage.GetEndTimeCode())
    tps = float(stage.GetTimeCodesPerSecond())
    fps = float(stage.GetFramesPerSecond())
    if not np.isfinite(tps) or tps <= 0.0:
        tps = 24.0
    if not np.isfinite(fps) or fps <= 0.0:
        fps = tps
    ref_time = Usd.TimeCode(start_tc)

    print(f"USD time range : {start_tc:g} .. {end_tc:g}")
    print(f"USD fps/tps    : {fps:g} / {tps:g}")

    fluid_sources = []
    granular_sources = []
    colliders = []
    sinks = []
    hinges = {}
    dynamic_bodies = {}
    collider_materials = {}
    reference_linear = {}
    reference_translation = {}

    print()
    print("Scanning scene...")
    xcache = UsdGeom.XformCache(ref_time)
    physics_hinge_owners = _physics_hinge_owners(stage)
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        vertices = _world_points(prim, ref_time)
        if len(vertices) == 0:
            continue
        triangles = _triangulate(prim, vertices, ref_time)
        bmin = np.min(vertices, axis=0)
        bmax = np.max(vertices, axis=0)
        data = MeshData(
            path=str(prim.GetPath()),
            vertices=vertices,
            triangles=triangles,
            bounds_min=bmin,
            bounds_max=bmax,
        )
        role = _role_for_prim(prim)
        if role in ("fluid", "source", "fluid_source"):
            fluid_sources.append(data)
            label = "FLUID SOURCE"
        elif role in ("granular", "granular_source"):
            granular_sources.append(GranularSourceData(
                mesh=data,
                source_mode=_string_attribute(prim, "grasph:sourceMode", "volume"),
                particle_radius=_float_attribute(prim, "grasph:particleRadius", 0.025),
                density=_float_attribute(prim, "grasph:density", 1600.0),
                friction=_float_attribute(prim, "grasph:friction", 0.45),
                restitution=_float_attribute(prim, "grasph:restitution", 0.10),
                rolling_friction=_float_attribute(prim, "grasph:rollingFriction", 0.02),
                flow_rate=_float_attribute(prim, "grasph:flowRate", 100.0),
                initial_velocity=_vec3_attribute(
                    prim, "grasph:initialVelocity", (0.0, 0.0, -1.0)
                ),
                start_time=_float_attribute(prim, "grasph:startTime", 0.0),
                end_time=_float_attribute(prim, "grasph:endTime", 1.0),
            ))
            label = "GRANULAR SRC"
        elif role == "sink":
            sinks.append(data)
            label = "SINK"
        elif role in ("ignore", "visual", "disabled"):
            continue
        else:
            colliders.append(data)
            label = "COLLIDER"
            physics_material = _physics_material_values(prim) or {}
            custom_friction, _ = _inherited_attribute(prim, "grasph:friction")
            custom_restitution, _ = _inherited_attribute(prim, "grasph:restitution")
            collider_materials[data.path] = {
                "friction": max(0.0, float(
                    custom_friction if custom_friction is not None
                    else physics_material.get("friction")
                    if physics_material.get("friction") is not None else 0.45
                )),
                "restitution": float(np.clip(
                    custom_restitution if custom_restitution is not None
                    else physics_material.get("restitution")
                    if physics_material.get("restitution") is not None else 0.10,
                    0.0, 1.0
                )),
                "rolling_friction": max(
                    0.0, _float_attribute(prim, "grasph:rollingFriction", 0.02)
                ),
            }
            xf = xcache.GetLocalToWorldTransform(prim)
            linear, translation = _matrix_parts(xf)
            reference_linear[data.path] = linear
            reference_translation[data.path] = translation
            hinge = _hinge_for_prim(prim)
            if hinge is None:
                hinge = _physics_hinge_for_prim(prim, physics_hinge_owners)
            if hinge is not None:
                hinges[data.path] = hinge
            body = None if hinge is not None else _dynamic_body_for_prim(prim, data)
            if body is not None:
                dynamic_bodies[data.path] = body

        print(f"  {label:<13}: {data.path}" + (f" ({len(triangles)} triangles)" if label == "COLLIDER" else ""))
        if label == "FLUID SOURCE":
            size = bmax - bmin
            print(f"      min  : {bmin}")
            print(f"      max  : {bmax}")
            print(f"      size : {size}")

    print()
    print(f"Fluid sources : {len(fluid_sources)}")
    print(f"Granular sources: {len(granular_sources)}")
    print(f"Sink volumes : {len(sinks)}")
    print(f"Collider meshes: {len(colliders)}")
    print(f"Free axle hinges: {len(hinges)}")
    print(f"Dynamic bodies : {len(dynamic_bodies)}")
    for hinge in hinges.values():
        print(
            f"  {hinge.path}: axis={hinge.axis.astype(int)}, "
            f"center={hinge.center}, I={hinge.moment_of_inertia:g} kg m2"
        )

    scene = SceneData(
        stage=stage,
        fluid_sources=fluid_sources,
        granular_sources=granular_sources,
        colliders=colliders,
        sinks=sinks,
        source_file=str(Path(usd_file)),
        start_time_code=start_tc,
        end_time_code=end_tc,
        time_codes_per_second=tps,
        frames_per_second=fps,
        reference_linear=reference_linear,
        reference_translation=reference_translation,
        motion_time_codes={},
        motion_translations={},
        motion_driver_paths={},
        hinges=hinges,
        collider_materials=collider_materials,
        dynamic_bodies=dynamic_bodies,
    )

    motion_t, motion_v, motion_drivers = _sample_motion_tables(
        stage, colliders, reference_linear, reference_translation,
        start_tc, end_tc, tps, fps, hinges=hinges,
        dynamic_bodies=dynamic_bodies,
    )
    scene.motion_time_codes = motion_t
    scene.motion_translations = motion_v
    scene.motion_driver_paths = motion_drivers
    scene.print_motion_preflight()

    # Table-based animation summary used by the solver itself.
    animated = scene.animated_collider_paths()
    if animated:
        print("Animated/kinematic colliders used by PHYSICS:")
        for pth in animated:
            travel=float(np.max(np.linalg.norm(scene.motion_translations[pth],axis=1)))
            print(f"  {pth}  max travel={travel:.6f} m")
    else:
        print("WARNING: no collider motion was found in the composed USD stage.")

    return scene


def _axis_centers(lo, hi, dp):
    start = float(lo) + 0.5 * float(dp)
    stop = float(hi) - 0.5 * float(dp)
    if stop < start - 1.0e-12:
        return np.empty(0, dtype=np.float64)
    n = int(np.floor((stop - start) / float(dp) + 1.0e-9)) + 1
    return start + np.arange(n, dtype=np.float64) * float(dp)


def generate_fluid_particles(scene, dp, particle_radius=0.0):
    """Spawn regular particles from FluidVolume world-space AABBs at USD start time."""
    dp = float(dp)
    parts = []

    print()
    print("Generating fluid particles...")
    for source in scene.fluid_sources:
        xs = _axis_centers(source.bounds_min[0], source.bounds_max[0], dp)
        ys = _axis_centers(source.bounds_min[1], source.bounds_max[1], dp)
        zs = _axis_centers(source.bounds_min[2], source.bounds_max[2], dp)
        if len(xs) == 0 or len(ys) == 0 or len(zs) == 0:
            pos = np.empty((0, 3), dtype=np.float64)
        else:
            X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
            pos = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
        parts.append(pos)
        print()
        print(f"  {source.path}")
        print(f"      particles : {len(pos)}")
        print(f"      grid      : {len(xs)} x {len(ys)} x {len(zs)}")

    if not parts:
        out = np.empty((0, 3), dtype=np.float64)
    elif len(parts) == 1:
        out = parts[0]
    else:
        out = np.vstack(parts)
        scale = max(dp * 1.0e-8, 1.0e-12)
        keys = np.round(out / scale).astype(np.int64)
        _, keep = np.unique(keys, axis=0, return_index=True)
        out = out[np.sort(keep)]

    out = exclude_solid_fluid_particles(out, scene.colliders, particle_radius)
    print()
    print(f"Total fluid particles: {len(out)}")
    return out


def exclude_solid_fluid_particles(positions, meshes, particle_radius=0.0):
    """Keep fluid outside solid volumes and clear its collision radius.

    An AABB only culls queries; tank cavities remain fluid. Open sheets are
    collision surfaces, not volumes; only their finite radius clearance is cut.
    """
    from grasph_collision import points_inside_closed_mesh, is_closed_triangle_mesh, closest_point_triangle_batch
    radius = max(0.0, float(particle_radius))
    keep = np.ones(len(positions), dtype=bool)
    for mesh in meshes:
        candidates = np.flatnonzero(keep & np.all(
            (positions >= mesh.bounds_min - radius) & (positions <= mesh.bounds_max + radius), axis=1))
        closed = is_closed_triangle_mesh(mesh.triangles)
        if not len(candidates) or (not closed and radius == 0.0):
            continue
        removed = 0
        for start in range(0, len(candidates), 8192):
            ids = candidates[start:start + 8192]
            query = positions[ids]
            inside = points_inside_closed_mesh(query, mesh.triangles) if closed else np.zeros(len(ids), dtype=bool)
            if radius > 0.0:
                distance2 = np.full(len(ids), np.inf)
                for a, b, c in mesh.triangles:
                    delta = query - closest_point_triangle_batch(query, a, b, c)
                    distance2 = np.minimum(distance2, np.einsum('ij,ij->i', delta, delta))
                inside |= distance2 < radius**2
            keep[ids[inside]] = False
            removed += int(np.count_nonzero(inside))
        if removed:
            print(f"  Solid exclusion: {mesh.path}: {removed} initial overlapping fluid particles removed")
    return positions[keep]


def generate_granular_particles(scene):
    """Fill granular source AABBs with non-overlapping sphere centers."""
    positions = []
    source_ids = []
    print()
    print("Generating granular particles...")
    for source_id, source in enumerate(scene.granular_sources):
        if source.source_mode == "flow":
            continue
        radius = float(source.particle_radius)
        if not np.isfinite(radius) or radius <= 0.0:
            raise RuntimeError(
                f"Granular source {source.mesh.path} needs a positive particle radius"
            )
        diameter = 2.0 * radius
        xs = _axis_centers(source.mesh.bounds_min[0], source.mesh.bounds_max[0], diameter)
        ys = _axis_centers(source.mesh.bounds_min[1], source.mesh.bounds_max[1], diameter)
        zs = _axis_centers(source.mesh.bounds_min[2], source.mesh.bounds_max[2], diameter)
        if len(xs) and len(ys) and len(zs):
            X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
            pos = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
        else:
            pos = np.empty((0, 3), dtype=np.float64)
        positions.append(pos)
        source_ids.append(np.full(len(pos), source_id, dtype=np.int32))
        print(f"  {source.mesh.path}: {len(pos)} particles, radius={radius:g} m")

    if not positions:
        return np.empty((0, 3), dtype=np.float64), np.empty(0, dtype=np.int32)
    return np.vstack(positions), np.concatenate(source_ids)
