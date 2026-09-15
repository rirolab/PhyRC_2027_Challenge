"""Resize the shirt, then shorten its body without deforming the collar."""
import numpy as np


def surface_area(points, counts, indices):
    area, offset = 0.0, 0
    for count in counts:
        face = np.asarray(indices[offset:offset + int(count)], dtype=int)
        offset += int(count)
        cross = np.cross(points[face[1:-1]] - points[face[0]],
                         points[face[2:]] - points[face[0]])
        area += float(np.linalg.norm(cross, axis=1).sum()) * 0.5
    return area


def opening_area(points):
    centred = points - points.mean(axis=0)
    return float(np.linalg.norm(np.cross(centred, np.roll(centred, -1, axis=0)).sum(axis=0))) * 0.5


def resize_shirt(points, counts, indices, scale=1.2, neck_area_scale=1.2,
                 length_scale=10.0 / 12.0):
    if not np.isfinite([scale, neck_area_scale, length_scale]).all() or min(scale, neck_area_scale, length_scale) <= 0:
        raise ValueError('Shirt size and collar area scales must be positive and finite')
    p = np.asarray(points, dtype=float).copy()
    reference_area = surface_area(p, counts, indices)
    edges, neighbors, offset = {}, [set() for _ in p], 0
    for count in counts:
        face = [int(v) for v in indices[offset:offset + int(count)]]
        offset += int(count)
        for a, b in zip(face, face[1:] + face[:1]):
            edge = tuple(sorted((a, b)))
            edges[edge] = edges.get(edge, 0) + 1
            neighbors[a].add(b)
            neighbors[b].add(a)
    boundary = {}
    for (a, b), count in edges.items():
        if count == 1:
            boundary.setdefault(a, []).append(b)
            boundary.setdefault(b, []).append(a)
    if any(len(adjacent) != 2 for adjacent in boundary.values()):
        raise ValueError('Shirt openings must be closed manifold loops')
    seen, loops = set(), []
    for start in boundary:
        if start in seen:
            continue
        previous, current, loop = None, start, []
        while current not in seen:
            seen.add(current)
            loop.append(current)
            nxt = next(vertex for vertex in boundary[current] if vertex != previous)
            previous, current = current, nxt
        if current != start:
            raise ValueError('Shirt boundary loop is not closed')
        loops.append(np.asarray(loop, dtype=int))
    if len(loops) != 4:
        raise ValueError('Expected collar, hem and two cuff openings')
    hem = max(range(4), key=lambda i: np.ptp(p[loops[i], 0]))
    collar = max((i for i in range(4) if i != hem),
                 key=lambda i: abs(p[loops[i], 1].mean() - p[loops[hem], 1].mean()))
    neck = loops[collar]
    neck_before = opening_area(p[neck])
    centre = p[neck].mean(axis=0)
    # The requested area is relative to the OLD final shirt, not multiplied
    # by the enlargement's extra scale**2 a second time.
    local_scale = np.sqrt(neck_area_scale) / scale
    ring = sorted(set().union(*(neighbors[vertex] for vertex in neck)) - set(neck))
    for vertices, factor in ((neck, local_scale), (ring, (1.0 + local_scale) * 0.5)):
        p[vertices] = centre + (p[vertices] - centre) * factor
    p *= scale
    # Preserve every collar and adjacent-ring coordinate, not just its area.
    # Raw Y is the neck-to-hem axis. Compress only below the protected band,
    # keeping the neck end fixed and achieving the requested overall height.
    neck_area_before_height = opening_area(p[neck])
    direction = 1.0 if p[neck, 1].mean() > p[loops[hem], 1].mean() else -1.0
    axial = direction * p[:, 1]
    bottom, top = float(axial.min()), float(axial.max())
    height_before = top - bottom
    protected = np.concatenate((neck, np.asarray(ring, dtype=int)))
    cutoff = float(axial[protected].min())
    new_bottom = top - height_before * length_scale
    if height_before <= 0 or cutoff <= bottom or new_bottom >= cutoff:
        raise ValueError('Requested shirt height cannot preserve the complete collar band')
    below = axial < cutoff
    axial[below] = cutoff + (axial[below] - cutoff) * (cutoff - new_bottom) / (cutoff - bottom)
    p[below, 1] = direction * axial[below]
    area = surface_area(p, counts, indices)
    if min(reference_area, area, neck_before) <= 0:
        raise ValueError('Shirt and collar areas must be nonzero')
    return p, {
        'resizeScale': float(scale),
        'lengthScale': float(length_scale),
        'heightBefore': height_before,
        'heightAfter': float(np.ptp(p[:, 1])),
        'neckAreaBeforeHeight': neck_area_before_height,
        'massAreaRatio': reference_area / area,
        'referenceSurfaceArea': reference_area,
        'neckAreaBefore': neck_before,
        'neckAreaAfter': opening_area(p[neck]),
    }
