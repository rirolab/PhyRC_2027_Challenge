#!/usr/bin/env python3
"""Conservative USD triangle-mesh inter-robot collision checker.

Requires python-fcl + trimesh. Fail closed on missing/unsupported geometry.
Does not check self-collision, floor, cloth, or swept motion.
"""
import numpy as np
from .forward_kinematics import inverse

ROOT='/stretch/base_footprint'

def triangulate(counts,indices):
    tris=[];off=0
    for n in counts:
        face=indices[off:off+n];off+=n
        if n<3:raise ValueError('Invalid polygon')
        for i in range(1,n-1):tris.append([face[0],face[i],face[i+1]])
    return np.asarray(tris,dtype=np.int64)

def gf_matrix(g):
    return np.asarray([[float(g[i][j]) for j in range(4)] for i in range(4)],float).T

def load_colliders(usd_path, joints, allow_unmatched=False):
    """Resolve USD library collision meshes through /stretch/<link>/collisions references.

    Library meshes under /colliders are not physical robot-link instances.  A
    link's composed /collisions prim references a library subtree; geometry is
    measured relative to that library root, then attached to the instance root.
    """
    from pxr import Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        raise RuntimeError(f'Cannot open USD: {usd_path}')
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    bodies = {ROOT} | {j.parent for j in joints} | {j.child for j in joints}
    colliders, unmatched, seen = [], [], 0
    claimed = set()

    # Only the authored collision library is scanned.  An instance's prototype
    # should not be traversed and counted separately as another collider.
    library_meshes = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith('/colliders/'):
            continue
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            continue
        seen += 1
        library_meshes.append(prim)

    # Each link-attached reference is the authoritative library -> link mapping.
    for link in sorted(bodies):
        linkprim = stage.GetPrimAtPath(link)
        if not linkprim.IsValid():
            raise RuntimeError(f'FK body link not present in USD: {link}')
        refroot = stage.GetPrimAtPath(link + '/collisions')
        if not refroot.IsValid():
            continue
        references = refroot.GetMetadata('references')
        items = []
        if references is not None:
            # Sdf.ReferenceListOp exposes these properties in OpenUSD Python.
            # Their names differ from the list-editor API on prim specs.
            for attr in ('explicitItems','prependedItems','addedItems','appendedItems'):
                items.extend(list(getattr(references,attr,[]) or []))
        # Empty collision Xforms are permitted, but a non-empty unrecognised
        # reference must never be silently treated as collision-free.
        if not items:
            if refroot.IsInstance():
                unmatched.append({'path':str(refroot.GetPath()), 'reason':'instance has no readable USD reference'})
            continue
        if len(items) != 1:
            unmatched.append({'path':str(refroot.GetPath()), 'reason':f'expected 1 reference, got {len(items)}'})
            continue
        reference = items[0]
        source_root_path = str(reference.primPath)
        if reference.assetPath or not source_root_path.startswith('/colliders/'):
            unmatched.append({'path':str(refroot.GetPath()), 'reason':f'unsupported external or non-collider reference: {reference}'})
            continue
        source_root = stage.GetPrimAtPath(source_root_path)
        if not source_root.IsValid():
            unmatched.append({'path':str(refroot.GetPath()), 'reason':f'missing library root {source_root_path}'})
            continue
        # USD/Gf matrices multiply row vectors; gf_matrix transposes to our
        # column-vector convention.  Geometry stays in source-root coordinates.
        T_link = gf_matrix(cache.GetLocalToWorldTransform(linkprim))
        T_inst = gf_matrix(cache.GetLocalToWorldTransform(refroot))
        T_src = gf_matrix(cache.GetLocalToWorldTransform(source_root))
        T_link_inst = inverse(T_link) @ T_inst
        for prim in library_meshes:
            meshpath = str(prim.GetPath())
            if not meshpath.startswith(source_root_path + '/'):
                continue
            claimed.add(meshpath)
            if not prim.IsA(UsdGeom.Mesh):
                unmatched.append({'path':meshpath,'reason':'enabled collider is not a Mesh'})
                continue
            mesh = UsdGeom.Mesh(prim)
            points = mesh.GetPointsAttr().Get()
            counts = mesh.GetFaceVertexCountsAttr().Get()
            indices = mesh.GetFaceVertexIndicesAttr().Get()
            if points is None or counts is None or indices is None:
                unmatched.append({'path':meshpath,'reason':'missing mesh topology'})
                continue
            if mesh.GetSubdivisionSchemeAttr().Get() not in (None, 'none'):
                unmatched.append({'path':meshpath,'reason':'subdivision mesh requires tessellation'})
                continue
            vertices = np.asarray(points,dtype=np.float64)
            faces = triangulate(counts,indices)
            if len(vertices)==0 or len(faces)==0:
                unmatched.append({'path':meshpath,'reason':'empty mesh'})
                continue
            T_src_mesh = inverse(T_src) @ gf_matrix(cache.GetLocalToWorldTransform(prim))
            local = T_link_inst @ T_src_mesh
            instance_path = str(refroot.GetPath()) + meshpath[len(source_root_path):]
            colliders.append((instance_path, link, vertices, faces, local))

    for prim in library_meshes:
        meshpath = str(prim.GetPath())
        if meshpath not in claimed:
            unmatched.append({'path':meshpath,'reason':'enabled library collider has no mapped robot reference'})
    if seen == 0:
        raise RuntimeError('No enabled CollisionAPI prims found in /colliders')
    if unmatched and not allow_unmatched:
        raise RuntimeError('Incomplete collision coverage:\n' + json.dumps(unmatched[:30],indent=2))
    if not colliders:
        raise RuntimeError('No link-attached collision meshes resolved')
    if len(colliders) != seen:
        # A library mesh can be instanced multiple times.  For this particular
        # robot asset we expect one instance per enabled library mesh.
        raise RuntimeError(f'Collision mesh coverage unexpected: mapped={len(colliders)}, enabled={seen}')
    print(f'Collision coverage: {len(colliders)}/{seen} enabled library meshes resolved', flush=True)
    return colliders, unmatched, seen

