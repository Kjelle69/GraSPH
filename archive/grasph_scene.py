import sys
import math
from dataclasses import dataclass

import numpy as np

from pxr import Usd, UsdGeom, Gf


# ============================================================
# GraSPH Scene Loader
#
# OpenUSD -> neutral simulation data
#
# Responsibilities:
#
#   - Read OpenUSD scene
#   - Find FluidVolume objects
#   - Generate initial fluid particles
#   - Find collider meshes
#   - Convert collider geometry to world-space triangles
#
# NO SPH physics here.
# NO collision solving here.
# ============================================================


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class FluidSource:
    path: str
    minimum: np.ndarray
    maximum: np.ndarray


@dataclass
class ColliderMesh:
    path: str
    triangles: np.ndarray


@dataclass
class GraSPHScene:
    usd_file: str
    stage: object
    fluid_sources: list
    colliders: list


# ============================================================
# HELPERS
# ============================================================

def _path_parts(prim):
    return str(prim.GetPath()).split("/")


def is_fluid_volume(prim):
    """
    Temporary convention:

        Any mesh below an object named FluidVolume
        is treated as a fluid source.

    Later we replace this with USD/custom metadata.
    """

    return "FluidVolume" in _path_parts(prim)


# ============================================================
# WORLD TRANSFORM
# ============================================================

def get_world_matrix(stage, prim):

    cache = UsdGeom.XformCache(
        Usd.TimeCode.Default()
    )

    return cache.GetLocalToWorldTransform(
        prim
    )


# ============================================================
# MESH -> WORLD TRIANGLES
# ============================================================

def mesh_to_world_triangles(stage, prim):

    mesh = UsdGeom.Mesh(prim)

    points = mesh.GetPointsAttr().Get()
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()

    if points is None:
        return np.empty(
            (0, 3, 3),
            dtype=np.float64
        )

    matrix = get_world_matrix(
        stage,
        prim
    )

    # --------------------------------------------------------
    # Vertices -> world coordinates
    # --------------------------------------------------------

    world_points = []

    for p in points:

        wp = matrix.Transform(
            Gf.Vec3d(
                float(p[0]),
                float(p[1]),
                float(p[2])
            )
        )

        world_points.append(
            (
                wp[0],
                wp[1],
                wp[2]
            )
        )

    world_points = np.asarray(
        world_points,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Polygon -> triangles
    # --------------------------------------------------------

    triangles = []

    offset = 0

    for count_raw in counts:

        count = int(count_raw)

        face = indices[
            offset:offset + count
        ]

        offset += count

        if count < 3:
            continue

        # Fan triangulation
        i0 = int(face[0])

        for k in range(
            1,
            count - 1
        ):

            i1 = int(face[k])
            i2 = int(face[k + 1])

            triangles.append(
                (
                    world_points[i0],
                    world_points[i1],
                    world_points[i2]
                )
            )

    if not triangles:

        return np.empty(
            (0, 3, 3),
            dtype=np.float64
        )

    return np.asarray(
        triangles,
        dtype=np.float64
    )


# ============================================================
# WORLD BOUNDS
# ============================================================

def get_world_bounds(stage, prim):

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_]
    )

    bbox = bbox_cache.ComputeWorldBound(
        prim
    )

    aligned = bbox.ComputeAlignedBox()

    minimum = aligned.GetMin()
    maximum = aligned.GetMax()

    return (
        np.array(
            [
                minimum[0],
                minimum[1],
                minimum[2]
            ],
            dtype=np.float64
        ),
        np.array(
            [
                maximum[0],
                maximum[1],
                maximum[2]
            ],
            dtype=np.float64
        )
    )


# ============================================================
# SCENE LOADER
# ============================================================

def load_scene(usd_file):

    print()
    print("========================================")
    print("GraSPH SCENE LOADER")
    print("========================================")
    print()

    print(f"Opening: {usd_file}")
    print()

    stage = Usd.Stage.Open(
        usd_file
    )

    if not stage:

        raise RuntimeError(
            f"Kunde inte öppna USD: {usd_file}"
        )

    fluid_sources = []
    colliders = []

    # Avoid discovering the same FluidVolume multiple times
    discovered_fluid_paths = set()

    print("Scanning scene...")

    for prim in stage.Traverse():

        if not prim.IsA(
            UsdGeom.Mesh
        ):
            continue

        path = str(
            prim.GetPath()
        )

        # ----------------------------------------------------
        # Fluid source
        # ----------------------------------------------------

        if is_fluid_volume(prim):

            parent = prim.GetParent()

            parent_path = str(
                parent.GetPath()
            )

            if (
                parent_path
                not in discovered_fluid_paths
            ):

                minimum, maximum = (
                    get_world_bounds(
                        stage,
                        parent
                    )
                )

                fluid_sources.append(
                    FluidSource(
                        path=parent_path,
                        minimum=minimum,
                        maximum=maximum
                    )
                )

                discovered_fluid_paths.add(
                    parent_path
                )

                print(
                    f"  FLUID SOURCE : "
                    f"{parent_path}"
                )

                print(
                    f"      min  : "
                    f"{minimum}"
                )

                print(
                    f"      max  : "
                    f"{maximum}"
                )

                print(
                    f"      size : "
                    f"{maximum - minimum}"
                )

            continue

        # ----------------------------------------------------
        # Collider
        # ----------------------------------------------------

        triangles = (
            mesh_to_world_triangles(
                stage,
                prim
            )
        )

        if len(triangles) == 0:
            continue

        colliders.append(
            ColliderMesh(
                path=path,
                triangles=triangles
            )
        )

        print(
            f"  COLLIDER     : "
            f"{path} "
            f"({len(triangles)} triangles)"
        )

    print()
    print(
        f"Fluid sources : "
        f"{len(fluid_sources)}"
    )

    print(
        f"Collider meshes: "
        f"{len(colliders)}"
    )

    print()

    return GraSPHScene(
        usd_file=usd_file,
        stage=stage,
        fluid_sources=fluid_sources,
        colliders=colliders
    )


# ============================================================
# FLUID PARTICLE GENERATION
# ============================================================

def generate_fluid_particles(
    scene,
    dp,
    margin=None
):

    if margin is None:

        # Particle centres one radius away
        # from authoring volume boundary
        margin = 0.5 * dp

    all_particles = []

    print("Generating fluid particles...")
    print()

    for source in scene.fluid_sources:

        minimum = (
            source.minimum + margin
        )

        maximum = (
            source.maximum - margin
        )

        if np.any(
            maximum <= minimum
        ):

            print(
                f"WARNING: Fluid source "
                f"{source.path} "
                f"is too small."
            )

            continue

        xs = np.arange(
            minimum[0],
            maximum[0] + 1e-10,
            dp
        )

        ys = np.arange(
            minimum[1],
            maximum[1] + 1e-10,
            dp
        )

        zs = np.arange(
            minimum[2],
            maximum[2] + 1e-10,
            dp
        )

        # Vectorized cartesian grid
        X, Y, Z = np.meshgrid(
            xs,
            ys,
            zs,
            indexing="ij"
        )

        particles = np.column_stack(
            (
                X.ravel(),
                Y.ravel(),
                Z.ravel()
            )
        )

        print(
            f"  {source.path}"
        )

        print(
            f"      particles : "
            f"{len(particles)}"
        )

        print(
            f"      grid      : "
            f"{len(xs)} x "
            f"{len(ys)} x "
            f"{len(zs)}"
        )

        all_particles.append(
            particles
        )

    if not all_particles:

        return np.empty(
            (0, 3),
            dtype=np.float64
        )

    result = np.vstack(
        all_particles
    )

    print()
    print(
        f"Total fluid particles: "
        f"{len(result)}"
    )

    return result


# ============================================================
# COMMAND LINE SMOKE TEST
# ============================================================

def main():

    if len(sys.argv) > 1:

        usd_file = sys.argv[1]

    else:

        usd_file = "TankTest.usda"

    DP = 0.020

    scene = load_scene(
        usd_file
    )

    particles = (
        generate_fluid_particles(
            scene,
            dp=DP
        )
    )

    print()
    print("========================================")
    print("SCENE READY")
    print("========================================")
    print()

    print(
        f"USD              : "
        f"{scene.usd_file}"
    )

    print(
        f"Fluid particles  : "
        f"{len(particles)}"
    )

    total_triangles = sum(
        len(c.triangles)
        for c in scene.colliders
    )

    print(
        f"Collider meshes  : "
        f"{len(scene.colliders)}"
    )

    print(
        f"Collider triangles: "
        f"{total_triangles}"
    )

    print()

    if len(particles) > 0:

        print(
            f"Particle min: "
            f"{particles.min(axis=0)}"
        )

        print(
            f"Particle max: "
            f"{particles.max(axis=0)}"
        )


if __name__ == "__main__":
    main()