"""Linear midpoint refinement: preserve the piecewise planar rest surface."""
import numpy as np
from pxr import Sdf, UsdGeom, Vt


def midpoint_refine(points, triangles):
    points = np.asarray(points, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    pairs = np.stack((triangles[:, [0, 1]], triangles[:, [1, 2]],
                      triangles[:, [2, 0]]), axis=1)
    edges, inverse = np.unique(np.sort(pairs.reshape(-1, 2), axis=1),
                               axis=0, return_inverse=True)
    ab, bc, ca = (inverse.reshape(-1, 3) + len(points)).T
    a, b, c = triangles.T
    refined = np.stack((np.stack((a, ab, ca), axis=1),
                        np.stack((ab, b, bc), axis=1),
                        np.stack((ca, bc, c), axis=1),
                        np.stack((ab, bc, ca), axis=1)), axis=1).reshape(-1, 3)
    return np.concatenate((points, points[edges].mean(axis=1))), refined, edges


def refine_surface_mesh(mesh, levels):
    if levels not in (0, 1):
        raise ValueError('STRETCH4_MESH_REFINEMENT must be 0 (original) or 1 (4x faces)')
    prim = mesh.GetPrim()
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    triangles = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    source_vertices, source_faces = len(points), len(triangles)
    if levels:
        # This project's prepared garments use flat colors and no UVs. Refuse
        # unsupported data rather than silently misindexing a future asset.
        for pv in UsdGeom.PrimvarsAPI(mesh).GetPrimvars():
            if pv.HasValue() and pv.GetInterpolation() != UsdGeom.Tokens.constant:
                raise ValueError(f'refinement needs explicit primvar remapping: {pv.GetName()}')
        points, triangles, _ = midpoint_refine(points, triangles)
        mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
        mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(triangles.astype(np.int32).ravel()))
        mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(len(triangles), 3, np.int32)))
        for subset in UsdGeom.Subset.GetAllGeomSubsets(mesh):
            if subset.GetElementTypeAttr().Get() != UsdGeom.Tokens.face:
                raise ValueError(f'unsupported garment subset: {subset.GetPath()}')
            old = np.asarray(subset.GetIndicesAttr().Get(), dtype=np.int32)
            subset.GetIndicesAttr().Set(Vt.IntArray.FromNumpy((old[:, None] * 4 + np.arange(4)).astype(np.int32).ravel()))
        mesh.GetNormalsAttr().Block()
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateExtentAttr(UsdGeom.PointBased.ComputeExtent(mesh.GetPointsAttr().Get()))
    for name, value in (('sourceVertices', source_vertices), ('sourceFaces', source_faces),
                        ('refinement', levels)):
        prim.CreateAttribute('phyrc:' + name, Sdf.ValueTypeNames.Int).Set(value)
    print(f'[Mesh refinement] {source_vertices}/{source_faces} -> '
          f'{len(points)}/{len(triangles)} vertices/triangles; linear midpoint, material unchanged', flush=True)
