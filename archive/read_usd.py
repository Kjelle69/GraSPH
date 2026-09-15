from pxr import Usd, UsdGeom

USD_FILE = "TankTest.usda"


def print_bounds(stage, path):
    prim = stage.GetPrimAtPath(path)

    if not prim:
        print(f"{path}: hittades inte")
        return

    print(f"\n{path}")
    print(f"  Type: {prim.GetTypeName()}")

    # World transform
    xformable = UsdGeom.Xformable(prim)
    world_matrix = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    print("  World transform:")
    print(world_matrix)

    # Bounding box i world coordinates
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_]
    )

    bbox = bbox_cache.ComputeWorldBound(prim)
    aligned = bbox.ComputeAlignedBox()

    minimum = aligned.GetMin()
    maximum = aligned.GetMax()

    print(f"  Min: {minimum}")
    print(f"  Max: {maximum}")

    size = maximum - minimum
    print(f"  Size: {size}")


def main():
    stage = Usd.Stage.Open(USD_FILE)

    if not stage:
        print(f"Kunde inte öppna {USD_FILE}")
        return

    print("========================================")
    print("OpenUSD test")
    print("========================================")

    print("\nAlla prims i scenen:")

    for prim in stage.Traverse():
        print(f"  {prim.GetPath()}    [{prim.GetTypeName()}]")

    print("\n========================================")
    print("Objekt")
    print("========================================")

    # Blender-exporten verkar lägga objekten under /root
    print_bounds(stage, "/root/Tank")
    print_bounds(stage, "/root/Gate")
    print_bounds(stage, "/root/FluidVolume")


if __name__ == "__main__":
    main()