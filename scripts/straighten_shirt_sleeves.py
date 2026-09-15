"""Align sleeve centre lines with X while keeping the torso and topology fixed."""
import numpy as np


def straighten_sleeves(points, counts, indices):
    p = np.asarray(points, dtype=float).copy()
    edge_counts, adjacency = {}, [set() for _ in p]
    offset = 0
    for count in counts:
        face = [int(v) for v in indices[offset:offset + int(count)]]
        offset += int(count)
        for a, b in zip(face, face[1:] + face[:1]):
            edge = tuple(sorted((a, b)))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
            adjacency[a].add(b)
            adjacency[b].add(a)
    boundary = {}
    for (a, b), count in edge_counts.items():
        if count == 1:
            boundary.setdefault(a, []).append(b)
            boundary.setdefault(b, []).append(a)
    seen, loops = set(), []
    for start in boundary:
        if start in seen:
            continue
        stack, loop = [start], []
        while stack:
            vertex = stack.pop()
            if vertex in seen:
                continue
            seen.add(vertex)
            loop.append(vertex)
            stack.extend(boundary[vertex])
        loops.append(np.asarray(loop, dtype=int))
    if len(loops) != 4:
        raise ValueError(f'T-shirt needs four openings, found {len(loops)}')
    hem_index = max(range(4), key=lambda i: np.ptp(p[loops[i], 0]))
    hem = loops[hem_index]
    other = [i for i in range(4) if i != hem_index]
    collar_index = max(other, key=lambda i: abs(p[loops[i], 1].mean() - p[hem, 1].mean()))
    neck_sign = np.sign(p[loops[collar_index], 1].mean() - p[hem, 1].mean())
    edges = np.asarray(list(edge_counts), dtype=int)
    reports = []
    for cuff_index in other:
        if cuff_index == collar_index:
            continue
        cuff = loops[cuff_index]
        centre = p[cuff].mean(axis=0)
        side = float(np.sign(centre[0]))
        x = side * p[:, 0]
        lower = float(x[hem].max()) + 0.001
        upper = float(x[cuff].min()) - 0.002
        if upper <= lower:
            raise ValueError('Sleeve cuff overlaps the torso reference width')
        # The armpit is the highest lower silhouette between torso and cuff.
        # Point averages in this region include the long torso side seam and
        # incorrectly report a horizontal sleeve even when its tip droops.
        candidates = []
        for cut in np.linspace(lower, upper, 96):
            a, b = edges[:, 0], edges[:, 1]
            crossing = ((x[a] < cut) & (x[b] >= cut)) | ((x[b] < cut) & (x[a] >= cut))
            a, b = a[crossing], b[crossing]
            if not len(a):
                continue
            t = (cut - x[a]) / (x[b] - x[a])
            section = p[a] + t[:, None] * (p[b] - p[a])
            root = (section.min(axis=0) + section.max(axis=0)) * 0.5
            candidates.append((float((neck_sign * section[:, 1]).min()), cut, root))
        if not candidates:
            raise ValueError('No sleeve root cross-section found')
        _, cut, root = max(candidates, key=lambda row: row[0])
        # Keep both endpoints of every edge crossing the root plane fixed.
        # Consequently the measured root section stays exactly unchanged.
        outside = x > cut
        mobile = outside.copy()
        for vertex in np.flatnonzero(outside):
            if any(not outside[neighbor] for neighbor in adjacency[vertex]):
                mobile[vertex] = False
        if not mobile[cuff].all():
            raise ValueError('Sleeve too short to separate cuff from fixed root')
        displacement = np.zeros_like(p)
        displacement[cuff, 0] = centre[0] - p[cuff, 0]
        displacement[cuff, 1:] = root[1:] - centre[1:]
        free_mask = mobile.copy()
        free_mask[cuff] = False
        free = np.flatnonzero(free_mask)
        lookup = {int(vertex): i for i, vertex in enumerate(free)}
        matrix = np.zeros((len(free), len(free)))
        rhs = np.zeros((len(free), 3))
        for row, vertex in enumerate(free):
            matrix[row, row] = len(adjacency[vertex])
            for neighbor in adjacency[vertex]:
                column = lookup.get(neighbor)
                if column is None:
                    rhs[row] += displacement[neighbor]
                else:
                    matrix[row, column] -= 1.0
        if len(free):
            displacement[free] = np.linalg.solve(matrix, rhs)
        before_axis = centre - root
        p += displacement
        after_axis = p[cuff].mean(axis=0) - root
        reports.append({
            'side': 'right' if side > 0 else 'left',
            'root': root.tolist(),
            'cuff_before': centre.tolist(),
            'cuff_after': p[cuff].mean(axis=0).tolist(),
            'reach_m': float(side * after_axis[0]),
            'lean_before_deg': float(np.degrees(np.arctan2(before_axis[1], side * before_axis[0]))),
            'lean_after_deg': float(np.degrees(np.arctan2(after_axis[1], side * after_axis[0]))),
        })
    return p, reports
