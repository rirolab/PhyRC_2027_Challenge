"""Swept node guard for the static mannequin during interactive cloth dragging.

PhysX owns normal contact and friction. This guard rejects residual triangle
crossings after the legacy position-writing controller and FEM have run. It
does not infer inside/outside from an open body mesh or close garment openings.
"""
import os
import numpy as np
import torch
import warp as wp


@wp.kernel
def _project_contact(mesh: wp.uint64, start: wp.array(dtype=wp.vec3),
                     target: wp.array(dtype=wp.vec3), result: wp.array(dtype=wp.vec3),
                     clearance: float, slide: int):
    i = wp.tid()
    p = start[i]
    delta = target[i] - p
    query = wp.mesh_query_point_no_sign(mesh, p, clearance + wp.length(delta))
    if query.result:
        near = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
        away = p - near
        distance = wp.length(away)
        if distance > 1.e-7:
            normal = away / distance
            inward = wp.dot(delta, normal)
            gap = wp.max(distance - clearance, 0.0)
            if inward < -gap:
                if slide != 0:
                    delta = delta - normal * (inward + gap)
                else:
                    delta = delta * (gap / wp.max(-inward, 1.e-9))
    result[i] = p + delta


@wp.kernel
def _sweep_nodes(mesh: wp.uint64, start: wp.array(dtype=wp.vec3),
                 target: wp.array(dtype=wp.vec3), velocity: wp.array(dtype=wp.vec3),
                 result: wp.array(dtype=wp.vec3), out_velocity: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    p = start[i]
    wanted = target[i]
    v = velocity[i]
    # Sliding may encounter a second triangle at a crease. Sweep that segment
    # too; never append an unchecked tangential displacement after a contact.
    for attempt in range(3):
        delta = wanted - p
        length = wp.length(delta)
        if length < 1.e-7:
            break
        direction = delta / length
        hit = wp.mesh_query_ray(mesh, p, direction, length)
        if not hit.result:
            p = wanted
            break
        normal = hit.normal
        if wp.dot(normal, direction) > 0.0:
            normal = -normal
        # Stay on the starting side of the triangle. Winding and watertightness
        # are irrelevant: this is a two-sided trajectory/triangle test.
        p = p + direction * wp.max(hit.t - 0.001, 0.0)
        rest = wanted - p
        wanted = p + rest - normal * wp.min(wp.dot(rest, normal), 0.0)
        v = v - normal * wp.min(wp.dot(v, normal), 0.0)
    result[i] = p
    out_velocity[i] = v


@wp.kernel
def _mark_cut_edges(mesh: wp.uint64, points: wp.array(dtype=wp.vec3),
                    edges: wp.array(dtype=wp.vec2i), blocked: wp.array(dtype=wp.int32),
                    count: wp.array(dtype=wp.int32)):
    edge = edges[wp.tid()]
    delta = points[edge[1]] - points[edge[0]]
    length = wp.length(delta)
    if length > 1.e-6:
        hit = wp.mesh_query_ray(mesh, points[edge[0]], delta / length, length)
        if hit.result and hit.t > 1.e-6 and hit.t < length - 1.e-6:
            wp.atomic_max(blocked, edge[0], 1)
            wp.atomic_max(blocked, edge[1], 1)
            wp.atomic_add(count, 0, 1)


@wp.kernel
def _reject_cut_nodes(start: wp.array(dtype=wp.vec3), blocked: wp.array(dtype=wp.int32),
                      points: wp.array(dtype=wp.vec3), velocity: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    if blocked[i] != 0:
        points[i] = start[i]
        velocity[i] = wp.vec3(0.0)


@wp.kernel
def _mark_cut_faces(cloth_mesh: wp.uint64, body_points: wp.array(dtype=wp.vec3),
                    body_edges: wp.array(dtype=wp.vec2i), triangles: wp.array(dtype=wp.vec3i),
                    blocked: wp.array(dtype=wp.int32), count: wp.array(dtype=wp.int32)):
    edge = body_edges[wp.tid()]
    p = body_points[edge[0]]
    delta = body_points[edge[1]] - p
    length = wp.length(delta)
    if length > 1.e-6:
        hit = wp.mesh_query_ray(cloth_mesh, p, delta / length, length)
        if hit.result and hit.t > 1.e-6 and hit.t < length - 1.e-6:
            face = triangles[hit.face]
            for j in range(3):
                wp.atomic_max(blocked, face[j], 1)
            wp.atomic_add(count, 0, 1)


@wp.kernel
def _contact_corrections(cloth_mesh: wp.uint64, body_points: wp.array(dtype=wp.vec3),
                         start: wp.array(dtype=wp.vec3), triangles: wp.array(dtype=wp.vec3i),
                         blocked: wp.array(dtype=wp.int32), correction: wp.array(dtype=wp.vec3),
                         weight: wp.array(dtype=float)):
    body = body_points[wp.tid()]
    query = wp.mesh_query_point_no_sign(cloth_mesh, body, 0.15)
    if query.result:
        face = triangles[query.face]
        if blocked[face[0]] != 0 or blocked[face[1]] != 0 or blocked[face[2]] != 0:
            bary = wp.vec3(query.u, query.v, 1.0 - query.u - query.v)
            old = start[face[0]] * bary[0] + start[face[1]] * bary[1] + start[face[2]] * bary[2]
            away = old - body
            distance = wp.length(away)
            if distance > 1.e-6:
                normal = away / distance
                near = wp.mesh_eval_position(cloth_mesh, query.face, query.u, query.v)
                depth = 0.001 - wp.dot(near - body, normal)
                if depth > 0.0:
                    push = normal * depth / wp.max(wp.dot(bary, bary), 1.e-6)
                    for j in range(3):
                        wp.atomic_add(correction, face[j], push * bary[j])
                        wp.atomic_add(weight, face[j], 1.0)


@wp.kernel
def _apply_contact_corrections(points: wp.array(dtype=wp.vec3), velocity: wp.array(dtype=wp.vec3),
                               correction: wp.array(dtype=wp.vec3), weight: wp.array(dtype=float)):
    i = wp.tid()
    if weight[i] > 0.0:
        delta = correction[i] / weight[i]
        points[i] = points[i] + delta
        normal = wp.normalize(delta)
        velocity[i] = velocity[i] - normal * wp.min(wp.dot(velocity[i], normal), 0.0)


@wp.kernel
def _swept_bounds(start: wp.array(dtype=wp.vec3), end: wp.array(dtype=wp.vec3),
                  lower: wp.array(dtype=float), upper: wp.array(dtype=float)):
    i = wp.tid()
    for axis in range(3):
        wp.atomic_min(lower, axis, wp.min(start[i][axis], end[i][axis]))
        wp.atomic_max(upper, axis, wp.max(start[i][axis], end[i][axis]))


@wp.kernel
def _overlap_body(lower: wp.array(dtype=float), upper: wp.array(dtype=float),
                   body_lower: wp.vec3, body_upper: wp.vec3, active: wp.array(dtype=int)):
    overlap = int(1)
    for axis in range(3):
        if upper[axis] < body_lower[axis] - 1.e-6 or lower[axis] > body_upper[axis] + 1.e-6:
            overlap = 0
    active[0] = overlap


@wp.kernel
def _repair_mode(attempt: wp.array(dtype=int), resolve: wp.array(dtype=int),
                 reject: wp.array(dtype=int), max_iterations: int):
    # The final pass checks the last repair; it must not modify the surface.
    resolve[0] = int(attempt[0] < 8 and attempt[0] < max_iterations)
    reject[0] = int(attempt[0] >= 8 and attempt[0] < max_iterations)


@wp.kernel
def _advance_iteration(count: wp.array(dtype=int), attempt: wp.array(dtype=int),
                        active: wp.array(dtype=int), fallback: wp.array(dtype=int),
                        max_iterations: int):
    if count[0] == 0:
        active[0] = 0
    elif attempt[0] >= max_iterations:
        # This count describes the surface AFTER the final permitted repair.
        active[0] = 0
        fallback[0] = 1
    else:
        attempt[0] = attempt[0] + 1


class _GpuSweep:
    """Same bounded repair loop, with device-side decisions and reusable storage."""

    def __init__(self, owner, start, edges, triangles):
        self.owner = owner
        self.device = owner.device
        self.calls = 0
        # Keep topology owners alive: keys alone do not protect against a
        # deallocated tensor's address being reused after a reset.
        self.topology = (edges, triangles)
        self.edges = wp.from_torch(edges, dtype=wp.vec2i)
        self.triangles = wp.from_torch(triangles, dtype=wp.vec3i)
        self.start = wp.clone(wp.from_torch(start, dtype=wp.vec3))
        self.target = wp.empty_like(self.start)
        self.velocity = wp.empty_like(self.start)
        self.result = wp.empty_like(self.start)
        self.out_velocity = wp.empty_like(self.start)
        self.candidate = wp.empty_like(self.start)
        self.correction = wp.empty_like(self.start)
        self.weight = wp.zeros(len(start), dtype=float, device=self.device)
        self.blocked = wp.zeros(len(start), dtype=int, device=self.device)
        self.count = wp.zeros(1, dtype=int, device=self.device)
        self.attempt = wp.zeros(1, dtype=int, device=self.device)
        self.active = wp.zeros(1, dtype=int, device=self.device)
        self.fallback = wp.zeros(1, dtype=int, device=self.device)
        self.resolve = wp.zeros(1, dtype=int, device=self.device)
        self.reject = wp.zeros(1, dtype=int, device=self.device)
        self.lower = wp.empty(3, dtype=float, device=self.device)
        self.upper = wp.empty(3, dtype=float, device=self.device)
        # Build from real geometry. Refit cannot repair a BVH built at zeros.
        self.mesh = wp.Mesh(points=wp.clone(self.start), indices=wp.from_torch(triangles.reshape(-1)))
        self.result_tensor = wp.to_torch(self.result)
        self.velocity_tensor = wp.to_torch(self.out_velocity)
        wp.load_module(module=__name__, device=self.device)
        with wp.ScopedCapture(device=self.device, force_module_load=False) as capture:
            self.blocked.zero_()
            self.attempt.zero_()
            self.fallback.zero_()
            wp.launch(_sweep_nodes, len(start), inputs=[owner.mesh.id, self.start, self.target,
                      self.velocity, self.result, self.out_velocity], device=self.device)
            self.lower.fill_(1.e30)
            self.upper.fill_(-1.e30)
            wp.launch(_swept_bounds, len(start), inputs=[self.start, self.result, self.lower, self.upper], device=self.device)
            wp.launch(_overlap_body, 1, inputs=[self.lower, self.upper, owner.lower, owner.upper, self.active], device=self.device)
            # Run the first full intersection test before deciding whether
            # another repair iteration is needed. Mesh.refit cannot be inside
            # CUDA conditional graphs (Warp 1.13 allocates scan scratch).
            wp.copy(self.mesh.points, self.result)
            self.mesh.refit()
            wp.capture_if(self.active, self._iteration)
        self.graph = capture.graph
        with wp.ScopedCapture(device=self.device, force_module_load=False) as capture:
            # First check is in graph. Allow max_iterations repairs, followed
            # by one final check, instead of rejecting on a stale pre-repair count.
            for _ in range(owner.max_iterations):
                wp.copy(self.mesh.points, self.result)
                self.mesh.refit()
                wp.capture_if(self.active, self._iteration)
            wp.capture_if(self.fallback, self._fallback)
        self.tail_graph = capture.graph

    def _iteration(self):
        self.count.zero_()
        wp.launch(_mark_cut_edges, len(self.edges), inputs=[self.owner.mesh.id, self.result,
                  self.edges, self.blocked, self.count], device=self.device)
        wp.launch(_mark_cut_faces, len(self.owner.body_edges), inputs=[self.mesh.id,
                  self.owner.mesh.points, self.owner.body_edges, self.triangles,
                  self.blocked, self.count], device=self.device)
        wp.capture_if(self.count, self._repair)
        wp.launch(_advance_iteration, 1, inputs=[self.count, self.attempt, self.active,
                  self.fallback, self.owner.max_iterations], device=self.device)

    def _repair(self):
        wp.launch(_repair_mode, 1, inputs=[self.attempt, self.resolve, self.reject,
                  self.owner.max_iterations], device=self.device)
        wp.capture_if(self.resolve, self._resolve)
        wp.capture_if(self.reject, self._reject)

    def _resolve(self):
        self.correction.zero_()
        self.weight.zero_()
        wp.launch(_contact_corrections, len(self.owner.mesh.points), inputs=[self.mesh.id,
                  self.owner.mesh.points, self.start, self.triangles, self.blocked,
                  self.correction, self.weight], device=self.device)
        wp.launch(_apply_contact_corrections, len(self.start), inputs=[self.result,
                  self.out_velocity, self.correction, self.weight], device=self.device)
        wp.copy(self.candidate, self.result)
        wp.launch(_sweep_nodes, len(self.start), inputs=[self.owner.mesh.id, self.start,
                  self.candidate, self.out_velocity, self.result, self.out_velocity], device=self.device)

    def _reject(self):
        wp.launch(_reject_cut_nodes, len(self.start), inputs=[self.start, self.blocked,
                  self.result, self.out_velocity], device=self.device)

    def _fallback(self):
        wp.copy(self.result, self.start)
        self.out_velocity.zero_()

    def run(self, start, target, velocity):
        wp.copy(self.start, wp.from_torch(start, dtype=wp.vec3))
        wp.copy(self.target, wp.from_torch(target, dtype=wp.vec3))
        wp.copy(self.velocity, wp.from_torch(velocity, dtype=wp.vec3))
        wp.capture_launch(self.graph)
        # Skip the tail when the first check is already clear. The final
        # budget check is included in the tail and never performs a repair.
        if int(self.active.numpy()[0]):
            wp.capture_launch(self.tail_graph)
        self.calls += 1
        # Preserve independent result ownership for all existing callers.
        return self.result_tensor, self.velocity_tensor


class SurfaceContactGuard:
    def __init__(self, points, triangles, device, use_graph=True, max_iterations=None):
        self.device = device
        # This custom geometric repair budget is independent of FEM iterations.
        # There is always one additional check after the last allowed repair.
        if max_iterations is None:
            max_iterations = os.environ.get('STRETCH4_CONTACT_GUARD_ITERATIONS', '24')
        self.max_iterations = int(max_iterations)
        if not 1 <= self.max_iterations <= 128:
            raise ValueError('STRETCH4_CONTACT_GUARD_ITERATIONS must be in [1, 128]')
        self.lower = wp.vec3(*np.min(points, axis=0))
        self.upper = wp.vec3(*np.max(points, axis=0))
        self.lower_tensor = torch.as_tensor(np.min(points, axis=0), dtype=torch.float32, device=device)
        self.upper_tensor = torch.as_tensor(np.max(points, axis=0), dtype=torch.float32, device=device)
        self.use_graph = use_graph and wp.get_device(device).is_cuda and wp.is_conditional_graph_supported()
        # CUDA cannot capture the legacy default stream. A dedicated stream
        # uses GPU events for ordering with Torch, without a host synchronize.
        self.stream = wp.Stream(device) if self.use_graph else None
        self.torch_stream = wp.stream_to_torch(self.stream) if self.use_graph else None
        self.mesh = wp.Mesh(points=wp.array(points, dtype=wp.vec3, device=device),
                            indices=wp.array(np.asarray(triangles).reshape(-1),
                                             dtype=wp.int32, device=device))
        t = np.asarray(triangles)
        edges = np.unique(np.sort(np.concatenate((t[:, [0, 1]], t[:, [1, 2]],
                                                  t[:, [2, 0]])), axis=1), axis=0)
        self.body_edges = wp.array(edges, dtype=wp.vec2i, device=device)
        self.cloth_meshes = {}
        self.gpu_sweeps = {}

    def clear_cache(self):
        """A checkpoint/reset may replace the entire spatial configuration."""
        if self.gpu_sweeps:
            wp.synchronize_stream(self.stream)
        self.cloth_meshes.clear()
        self.gpu_sweeps.clear()

    def project(self, start, target, clearance, allow_slide=True, coherent=False):
        """Smooth nearest-triangle contact; no corrugated vertex-sphere layer."""
        start, target = start.contiguous(), target.contiguous()
        result = torch.empty_like(target)
        wp.launch(_project_contact, len(start), inputs=[self.mesh.id,
                  wp.from_torch(start, dtype=wp.vec3), wp.from_torch(target, dtype=wp.vec3),
                  wp.from_torch(result, dtype=wp.vec3), float(clearance), int(allow_slide)],
                  device=self.device)
        if coherent and allow_slide:
            correction = result - target
            size = torch.linalg.norm(correction, dim=1)
            blocked = size > 1.e-7
            if bool(blocked.any()):
                delta = target - start
                distance = torch.linalg.norm(delta, dim=1).clamp(min=1.e-9)
                inward = (delta * correction).sum(dim=1) / (distance * size.clamp(min=1.e-9))
                if bool((inward[blocked] < -0.35).float().mean() >= 0.5):
                    # A synthetic grip is one rigid patch. A direct inward
                    # command must stop coherently, not let each anchor walk
                    # independently around the sides of the obstacle.
                    wp.launch(_project_contact, len(start), inputs=[self.mesh.id,
                              wp.from_torch(start, dtype=wp.vec3), wp.from_torch(target, dtype=wp.vec3),
                              wp.from_torch(result, dtype=wp.vec3), float(clearance), 0], device=self.device)
                    progress = torch.where(distance > 1.e-7,
                        torch.linalg.norm(result-start, dim=1)/distance, torch.ones_like(distance)).min().clamp(0, 1)
                    result = start + delta * progress
        # The nearest plane is a local sliding approximation. Sweep the actual
        # resulting segment against all triangles before accepting it.
        return self.sweep(start, result, torch.zeros_like(start))[0]

    def count_intersections(self, points, edges, triangles):
        """Validate a checkpoint before accepting its starting topology."""
        points, edges, triangles = (x.contiguous() for x in (points, edges, triangles))
        blocked = wp.zeros(len(points), dtype=wp.int32, device=self.device)
        count = wp.zeros(1, dtype=wp.int32, device=self.device)
        mesh = wp.Mesh(points=wp.from_torch(points, dtype=wp.vec3),
                       indices=wp.from_torch(triangles.reshape(-1)))
        wp.launch(_mark_cut_edges, len(edges), inputs=[self.mesh.id,
                  wp.from_torch(points, dtype=wp.vec3), wp.from_torch(edges, dtype=wp.vec2i),
                  blocked, count], device=self.device)
        wp.launch(_mark_cut_faces, len(self.body_edges), inputs=[mesh.id, self.mesh.points,
                  self.body_edges, wp.from_torch(triangles, dtype=wp.vec3i), blocked, count],
                  device=self.device)
        return int(count.numpy()[0])

    def sweep(self, start, target, velocity, edges=None, triangles=None):
        start, target, velocity = (x.contiguous() for x in (start, target, velocity))
        if self.use_graph and edges is not None and triangles is not None:
            key = (edges.data_ptr(), triangles.data_ptr())
            current = torch.cuda.current_stream(start.device)
            self.torch_stream.wait_stream(current)
            with wp.ScopedStream(self.stream):
                cached = self.gpu_sweeps.get(key)
                # Rebuild occasionally as folds change the optimal spatial
                # partition. This changes search cost, never collision rules.
                if cached is None or cached.calls >= 240:
                    cached = _GpuSweep(self, start, edges, triangles)
                    self.gpu_sweeps[key] = cached
                result, out_velocity = cached.run(start, target, velocity)
            current.wait_stream(self.torch_stream)
            return result.clone(), out_velocity.clone()
        result = torch.empty_like(target)
        out_velocity = torch.empty_like(velocity)
        wp.launch(_sweep_nodes, len(start), inputs=[self.mesh.id,
                  wp.from_torch(start, dtype=wp.vec3), wp.from_torch(target, dtype=wp.vec3),
                  wp.from_torch(velocity, dtype=wp.vec3), wp.from_torch(result, dtype=wp.vec3),
                  wp.from_torch(out_velocity, dtype=wp.vec3)], device=self.device)
        if edges is not None:
            blocked = wp.zeros(len(start), dtype=wp.int32, device=self.device)
            cloth_mesh = None
            if triangles is not None:
                key = triangles.data_ptr()
                if key not in self.cloth_meshes:
                    self.cloth_meshes[key] = wp.Mesh(
                        # BVH refit updates bounds, not the spatial topology.
                        # Building at all-zero points makes every subsequent
                        # ray/nearest query traverse a degenerate hierarchy.
                        points=wp.clone(wp.from_torch(start, dtype=wp.vec3)),
                        indices=wp.from_torch(triangles.reshape(-1)))
                cloth_mesh = self.cloth_meshes[key]
            # A vertex can move around a thin feature while its incident edge
            # cuts through it. Reject those local solver/controller movements.
            # The mask grows monotonically, so neighbouring repairs cannot
            # oscillate or sum oversized push-out corrections onto shared nodes.
            for attempt in range(self.max_iterations + 1):
                count = wp.zeros(1, dtype=wp.int32, device=self.device)
                wp.launch(_mark_cut_edges, len(edges), inputs=[self.mesh.id,
                          wp.from_torch(result, dtype=wp.vec3), wp.from_torch(edges, dtype=wp.vec2i),
                          blocked, count], device=self.device)
                if cloth_mesh is not None:
                    # Vertex/edge checks on the cloth alone miss a thin arm
                    # entering the INTERIOR of a stretched cloth triangle.
                    # Test the reverse edge/face direction as well.
                    wp.copy(cloth_mesh.points, wp.from_torch(result, dtype=wp.vec3))
                    cloth_mesh.refit()
                    wp.launch(_mark_cut_faces, len(self.body_edges), inputs=[
                              cloth_mesh.id, self.mesh.points, self.body_edges,
                              wp.from_torch(triangles, dtype=wp.vec3i), blocked, count],
                              device=self.device)
                if int(count.numpy()[0]) == 0:
                    break
                if attempt == self.max_iterations:
                    # Reject only after checking the final repaired geometry.
                    # This is not a checkpoint reload or a robot-state rollback.
                    result = start.clone()
                    out_velocity = torch.zeros_like(velocity)
                    break
                if cloth_mesh is not None and attempt < 8:
                    # Resolve against the contacted surface before rejecting
                    # motion. Pure rollback lets FEM stretch against an old
                    # pinned pose every step, producing persistent chatter.
                    correction = wp.zeros(len(start), dtype=wp.vec3, device=self.device)
                    weight = wp.zeros(len(start), dtype=float, device=self.device)
                    wp.launch(_contact_corrections, len(self.mesh.points), inputs=[
                              cloth_mesh.id, self.mesh.points, wp.from_torch(start, dtype=wp.vec3),
                              wp.from_torch(triangles, dtype=wp.vec3i), blocked, correction, weight],
                              device=self.device)
                    wp.launch(_apply_contact_corrections, len(start), inputs=[
                              wp.from_torch(result, dtype=wp.vec3), wp.from_torch(out_velocity, dtype=wp.vec3),
                              correction, weight], device=self.device)
                    candidate = result.clone()
                    wp.launch(_sweep_nodes, len(start), inputs=[self.mesh.id,
                              wp.from_torch(start, dtype=wp.vec3), wp.from_torch(candidate, dtype=wp.vec3),
                              wp.from_torch(out_velocity, dtype=wp.vec3), wp.from_torch(result, dtype=wp.vec3),
                              wp.from_torch(out_velocity, dtype=wp.vec3)], device=self.device)
                    continue
                wp.launch(_reject_cut_nodes, len(start), inputs=[
                          wp.from_torch(start, dtype=wp.vec3), blocked,
                          wp.from_torch(result, dtype=wp.vec3),
                          wp.from_torch(out_velocity, dtype=wp.vec3)], device=self.device)
        return result, out_velocity
