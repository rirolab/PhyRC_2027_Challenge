"""Repackage RCareWorld's t_shirt.usd into the layout Particle_Garment expects.

Particle_Garment hardcodes the mesh path as `<garment prim>/mesh`:

    self.garment_mesh_prim_path = self.garment_prim_path + "/mesh"

The project's own garments oblige (their mesh sits at /World/mesh), but the
RCareWorld shirt keeps its geometry at /World/t_shirt/Mid_poly, so referencing it
gives "RuntimeError: Accessed invalid null prim" the moment SingleClothPrim looks
for the mesh.

Rather than patch that assumption in the library, republish the geometry under
the expected path. Copies the attributes the cloth pipeline actually reads and
leaves behind the source's render-settings prims, which are irrelevant here.

Usage: make_wearable_shirt.py <source.usd> <dest.usd>
"""
import os
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

# Geometry conversion also works with usd-core, without starting a simulator.
_app = None
try:
    from pxr import Usd
except ImportError:
    from isaacsim import SimulationApp
    _app = SimulationApp({"headless": True})

from pxr import Gf, Sdf, Usd, UsdGeom, Vt  # noqa: E402

src_path, dst_path = sys.argv[1], sys.argv[2]
# Optional third argument: keep only this fraction of the garment's length,
# measured from the neck down. Faces below the cut are dropped and their vertices
# with them, so the shirt gets SHORTER without its particles getting closer
# together -- squashing it instead would halve the spacing along that axis and
# leave the collision offsets, which are absolute distances, far too wide for it.
# Same spacing means the margins and the handling carry over untouched.
KEEP_LENGTH = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
_HEM_FACES = []
_HEM_FACES = []
src = Usd.Stage.Open(src_path)
if src is None:
    print(f"SHIRT-CONV: cannot open {src_path}", flush=True)
    if _app is not None:
        _app.close()
    raise SystemExit(1)

source_mesh = None
for prim in src.Traverse():
    if prim.IsA(UsdGeom.Mesh):
        source_mesh = UsdGeom.Mesh(prim)
        break
if source_mesh is None:
    print("SHIRT-CONV: no mesh in source", flush=True)
    if _app is not None:
        _app.close()
    raise SystemExit(1)

print(f"SHIRT-CONV: source mesh {source_mesh.GetPath()}", flush=True)

dst = Usd.Stage.CreateNew(dst_path) if not os.path.exists(dst_path) \
    else Usd.Stage.Open(dst_path)
dst.RemovePrim("/World")
world = UsdGeom.Xform.Define(dst, "/World")
dst.SetDefaultPrim(world.GetPrim())

# The source declares metersPerUnit 0.01 while its points are already in metres
# (0.79m across is a real t-shirt, 0.79cm is not). Publish honest metadata so the
# staging normalizer has nothing left to correct.
UsdGeom.SetStageMetersPerUnit(dst, 1.0)
UsdGeom.SetStageUpAxis(dst, UsdGeom.Tokens.z)

mesh = UsdGeom.Mesh.Define(dst, "/World/mesh")
pts = source_mesh.GetPointsAttr().Get()
counts = source_mesh.GetFaceVertexCountsAttr().Get()
idxs = source_mesh.GetFaceVertexIndicesAttr().Get()
# Normalise the garment's own frame before publishing.
#
# The env drops every garment at one spawn pose that assumes a shirt LYING FLAT
# and centred on its own origin. Assets do not agree on that. RCareWorld's is
# already laid flat (thin axis Z, centre within 1mm of the origin); Modelink's is
# authored standing upright as if worn -- thin axis X, and the whole mesh sitting
# 0.42m ABOVE its origin. Referenced as-is it spawns half a metre low and clips
# through the floor, and the solver then fights that overlap forever: measured
# 7-10 m/s of permanent thrash, never settling, versus 0.16 m/s for the flat one.
#
# So: order the axes by extent (largest to X, smallest to Z) to lay it flat, and
# centre it on its origin. The permutation is a rotation, not a reflection -- an
# odd one is corrected, otherwise the mesh mirrors and every face winds backwards.
# This is a no-op for an asset already in that convention, which is why the
# RCareWorld path is unaffected.
import numpy as _np  # noqa: E402

_p = _np.asarray(pts, dtype=float)
_size = _p.max(axis=0) - _p.min(axis=0)
_order = list(_np.argsort(-_size))            # descending extent
_basis = _np.zeros((3, 3))
for _row, _src in enumerate(_order):
    _basis[_row, _src] = 1.0
if _np.linalg.det(_basis) < 0:
    _basis[2, :] *= -1.0
_p = _p @ _basis.T
_p -= (_p.max(axis=0) + _p.min(axis=0)) / 2.0
print(f"SHIRT-CONV: axes {_size.round(3)} -> {(_p.max(axis=0) - _p.min(axis=0)).round(3)}"
      f" (laid flat, centred on origin)", flush=True)
pts = Vt.Vec3fArray([Gf.Vec3f(*map(float, _q)) for _q in _p])

# Reshape the garment: narrower across, longer down.
#
# The axes were sorted by extent just above and are X sleeve span, Y neck-to-hem,
# Z thickness, and the points are already centred on the origin, so this is a
# scale about the garment's own centre. Thickness is left alone -- the asset is a
# flat shell and squashing Z would only close the two layers onto each other.
#
# Done HERE, before the crop, so everything downstream is computed on the final
# proportions: KEEP_LENGTH still keeps a half of the shirt (a longer half now),
# the hem band is still HEM_STRIPE_WIDTH of the length, and the collar loop is
# resized after the body has already been narrowed.
#
# WHAT THIS COSTS, because it is not free and the file is otherwise careful to
# say so: this is the one operation that moves particle SPACING, and every
# collision offset on the particle system is an absolute distance that stays
# where it is. Across the shirt the spacing shrinks with the width and the sheet
# gets denser; down it the spacing grows and the sheet gets coarser. It is the
# reason the short shirt is CROPPED rather than scaled (see the note at the top
# of this file). Measured with scripts/probe_gap.py, at rest, before and after:
# median edge, p90, and the share of edges already wider than the 19.4mm a
# particle can block. Check those numbers again after changing these.
#
# SHIRT_FLATTEN is the third axis, and it is a different kind of change. This
# asset is a garment as WORN -- a closed shell 0.32m from chest to back -- not a
# flat pattern. Dropped on a table it collapses under its own weight into a
# rounded heap with the sleeve tubes folded under it, which is what "the sleeves are
# drooping downward ... make it a real T shape" is about.
#
# The sleeves are NOT the problem. scripts/dump_sleeveaxis.py measures them in
# this asset at 8.2 and 9.3 degrees off straight out -- already square to the
# body, already a T. What is not flat is the shell, so pressing it flat here,
# before it is ever simulated, makes it spawn as a T instead of arriving at a
# heap: 0.12 took it to 39mm thick and scripts/probe_sleevedir.py then read the
# settled sleeves at 5.0 and 3.3 degrees.
#
# The 6.x surface FEM preserves its authored rest shape much more strongly than
# the old particle cloth did. Leaving this at 1.0 therefore preserves a 325 mm
# chest cavity even while the shirt lies on a table. The 6.x port defaults to
# 0.03: roughly 10 mm front-to-back after scene scale, enough to retain two
# distinct layers and the collar/cuff/hem openings without looking inflated.
# The old warning against going below 0.06 was specific to the 19.4 mm PBD
# particle spheres. It does not apply to the port's continuous 1 mm surface.
_WIDTH_SCALE = float(os.environ.get("SHIRT_WIDTH_SCALE", "1.0"))
_LENGTH_SCALE = float(os.environ.get("SHIRT_LENGTH_SCALE", "1.0"))
_FLATTEN = float(os.environ.get("SHIRT_FLATTEN", "0.03"))
if (_WIDTH_SCALE, _LENGTH_SCALE, _FLATTEN) != (1.0, 1.0, 1.0):
    _was = _p.max(axis=0) - _p.min(axis=0)
    _p[:, 0] *= _WIDTH_SCALE
    _p[:, 1] *= _LENGTH_SCALE
    _p[:, 2] *= _FLATTEN
    _now = _p.max(axis=0) - _p.min(axis=0)
    pts = Vt.Vec3fArray([Gf.Vec3f(*map(float, _q)) for _q in _p])
    print(f"SHIRT-CONV: reshaped x{_WIDTH_SCALE} wide x{_LENGTH_SCALE} long "
          f"x{_FLATTEN} thick: span {_was[0]:.3f} -> {_now[0]:.3f}m, "
          f"length {_was[1]:.3f} -> {_now[1]:.3f}m, "
          f"thickness {_was[2]:.3f} -> {_now[2]:.3f}m", flush=True)

# Shorten how far the sleeves reach from the torso, per "shorten the sleeves
# just a little". The torso itself (excluding sleeves) runs a roughly constant X-span
# for most of the garment's length -- only the band near the neck end widens
# out to where the sleeves attach (same width-profile fact SHIRT_CUFF_SCALE's
# loop-picking above relies on). Reference that torso width from the FAR half
# of the garment, where no sleeve can reach, then pull anything wider than it
# toward the centre line, by an amount proportional to how far out it already
# sat -- the pull is zero right at the torso boundary and largest at the
# sleeve tip, so there is no seam at the transition, and the torso itself
# (and the cuff hole's own SIZE, SHIRT_CUFF_SCALE) are untouched.
# Half the previous 0.75 sleeve reach; body length and cuff scale stay separate.
_SLEEVE_LEN_SCALE = float(os.environ.get("SHIRT_SLEEVE_LENGTH_SCALE", "0.375"))
if _SLEEVE_LEN_SCALE != 1.0:
    _yv_sl = _p[:, 1]
    _ylo_sl, _yhi_sl = _yv_sl.min(), _yv_sl.max()
    _bw_sl = 0.15 * (_yhi_sl - _ylo_sl)
    _nhi_sl = (float(_np.ptp(_p[_yv_sl > _yhi_sl - _bw_sl][:, 0]))
              > float(_np.ptp(_p[_yv_sl < _ylo_sl + _bw_sl][:, 0])))
    _far_sl = (_yv_sl < _ylo_sl + 0.5 * (_yhi_sl - _ylo_sl)) if _nhi_sl \
        else (_yv_sl > _ylo_sl + 0.5 * (_yhi_sl - _ylo_sl))
    _torso_hw = float(_np.percentile(_np.abs(_p[_far_sl, 0]), 95))
    _absx_sl = _np.abs(_p[:, 0])
    _sleeve_v = _absx_sl > _torso_hw
    if bool(_sleeve_v.any()):
        _excess = _absx_sl[_sleeve_v] - _torso_hw
        _sign_sl = _np.sign(_p[_sleeve_v, 0])
        _p[_sleeve_v, 0] = _sign_sl * (_torso_hw + _excess * _SLEEVE_LEN_SCALE)
        pts = Vt.Vec3fArray([Gf.Vec3f(*map(float, _q)) for _q in _p])
        print(f"SHIRT-CONV: sleeve length x{_SLEEVE_LEN_SCALE}: "
              f"{int(_sleeve_v.sum())} verts beyond {_torso_hw * 1000:.0f}mm "
              f"torso half-width pulled in", flush=True)
    else:
        print("SHIRT-CONV: WARNING SHIRT_SLEEVE_LENGTH_SCALE set but no "
              "vertices found beyond the torso half-width", flush=True)

if KEEP_LENGTH < 1.0:
    # Axes are ordered by extent above: X sleeve span, Y neck-to-hem, Z thickness.
    _y = _p[:, 1]
    _lo, _hi = _y.min(), _y.max()
    # Which end is the neck? On a shirt laid flat the SLEEVES are the widest part
    # of the whole garment and they sit at the shoulders, next to the neck -- so
    # the wider end is the neck end, not the hem. Getting this backwards keeps the
    # wrong half: the first attempt cut the neck and both sleeves off and left a
    # tube, which showed up as the sleeve span shrinking 0.887m -> 0.537m when only
    # the length was supposed to change.
    _band = 0.15 * (_hi - _lo)
    _w_lo = float(_np.ptp(_p[_y < _lo + _band][:, 0]))
    _w_hi = float(_np.ptp(_p[_y > _hi - _band][:, 0]))
    _neck_at_hi = _w_hi > _w_lo
    if _neck_at_hi:
        _cut = _hi - KEEP_LENGTH * (_hi - _lo)
        _keep_v = _y >= _cut
    else:
        _cut = _lo + KEEP_LENGTH * (_hi - _lo)
        _keep_v = _y <= _cut

    _off, _keep_faces = 0, []
    for _c in counts:
        _f = idxs[_off:_off + _c]
        _off += _c
        if all(_keep_v[int(_v)] for _v in _f):
            _keep_faces.append(_f)
    _used = sorted({int(v) for f in _keep_faces for v in f})
    _remap = {old_i: new_i for new_i, old_i in enumerate(_used)}
    _p = _p[_used]
    counts = _np.array([len(f) for f in _keep_faces])
    idxs = _np.array([_remap[int(v)] for f in _keep_faces for v in f])
    _p[:, 1] -= (_p[:, 1].max() + _p[:, 1].min()) / 2.0
    pts = Vt.Vec3fArray([Gf.Vec3f(*map(float, _q)) for _q in _p])
    print(f"SHIRT-CONV: cropped to {KEEP_LENGTH:.2f} of its length "
          f"(neck at {'high' if _neck_at_hi else 'low'} end); "
          f"{len(_p)} points, {len(counts)} faces remain", flush=True)

# Mark the band of faces nearest the hem as a GeomSubset, so the env can bind a
# second material there. Without it the shirt is one flat colour and which end
# is the hem is anyone's guess on screen.
#
# UNCONDITIONAL now -- this used to live inside "if KEEP_LENGTH < 1.0", because
# the hem edge only existed as a fresh CUT when a crop ran. With GREEN_SHIRT_
# LENGTH defaulted to 1.0 (crop off, see run.sh) that block stopped running
# at all and the stripe silently vanished ("the hem stripe disappeared"). The
# shirt's own natural hem is just as real an edge as a cut one, so find it the
# same way regardless of whether a crop happened -- same widest-end-is-the-
# neck heuristic the NECK_SCALE block below already falls back to.
try:
    _nhi = _neck_at_hi                  # set by the crop block above, if it ran
except NameError:                       # KEEP_LENGTH >= 1.0, no crop ran
    _yv = _p[:, 1]
    _ylo, _yhi = _yv.min(), _yv.max()
    _bw = 0.15 * (_yhi - _ylo)
    _nhi = (float(_np.ptp(_p[_yv > _yhi - _bw][:, 0]))
            > float(_np.ptp(_p[_yv < _ylo + _bw][:, 0])))
    _lo, _hi = _ylo, _yhi
_hem_edge = _p[:, 1].min() if _nhi else _p[:, 1].max()
_band_w = float(os.environ.get("HEM_STRIPE_WIDTH", "0.10")) * abs(_hi - _lo)
# Select by the face's CLOSEST vertex to the edge, not its mean Y. The hem
# ring includes corner/seam faces where the hem meets the side seam, and
# those tend to be larger/more irregular than the plain interior quads --
# their mean Y gets pulled inward past a tight band_w even though part of
# the face genuinely touches the hem edge, which was cutting the two side
# corners out of the band and left the stripe looking centred ("the stripe is only
# in the middle"). Any-vertex-touches-the-band is inclusive enough to keep the full
# width, corners included.
_face_off, _hem_faces = 0, []
for _fi, _c in enumerate(counts):
    _f = idxs[_face_off:_face_off + _c]
    _face_off += _c
    _fy = _p[_f, 1]
    _closest = _fy.min() if _nhi else _fy.max()
    if abs(_closest - _hem_edge) <= _band_w:
        _hem_faces.append(_fi)
_HEM_FACES = _hem_faces
print(f"SHIRT-CONV: hem band = {len(_hem_faces)} faces", flush=True)

# Subdivide, if asked. This is the fix for the mannequin coming through the
# shirt, and it is the only one that actually closes the hole rather than
# widening the particle that has to plug it.
#
# PhysX particle cloth collides as SPHERES, one per vertex, so the fabric is
# only solid where neighbouring spheres overlap -- past that, the gap between
# them is empty space to the solver and a limb goes straight through. Measured
# on this garment with scripts/probe_gap.py: particles block 19.4mm, mesh edges
# run 16.6mm median / 22.3mm at p90 at rest, so 25% of the sheet was already
# open BEFORE anything stretched it, and 46% once it was pulled.
#
# One round of midpoint subdivision halves every edge and quadruples the face
# count: each triangle becomes four, sharing three new vertices at the edge
# midpoints. Edges at ~8mm sit comfortably inside the 19.4mm a particle blocks,
# with room for the local stretch that pulling adds.
# Shrink (or widen) the collar opening.
#
# Runs BEFORE subdivision on purpose, so the extra vertices are generated from
# the resized collar rather than the original one.
#
# The collar is TRACED, not assumed. A boundary edge -- one used by exactly one
# face -- belongs to an opening, and those edges form closed loops: on a cropped
# shirt there are four (collar, two cuffs, and the cut across the hem). Picking
# by geometry rather than by index keeps this working after the crop changes the
# numbering: the cuffs sit at the sleeve extremes in X, the hem cut spans the
# whole width, and the collar is the one near the neck end whose centre is on
# the garment's centre line.
_NECK_SCALE = float(os.environ.get("SHIRT_NECK_SCALE", "1.0"))
# Same boundary-loop technique as the collar, applied to the two cuffs
# instead, per "make the neck opening a bit bigger and the sleeve openings smaller". No cuff
# knob existed before this -- 1.0 is unchanged/original size.
_CUFF_SCALE = float(os.environ.get("SHIRT_CUFF_SCALE", "1.0"))
if _NECK_SCALE != 1.0 or _CUFF_SCALE != 1.0:
    try:
        _nhi = _neck_at_hi              # set by the crop above
    except NameError:                   # KEEP_LENGTH == 1.0, no crop ran
        _yv = _p[:, 1]
        _ylo, _yhi = _yv.min(), _yv.max()
        _bw = 0.15 * (_yhi - _ylo)
        _nhi = (float(_np.ptp(_p[_yv > _yhi - _bw][:, 0]))
                > float(_np.ptp(_p[_yv < _ylo + _bw][:, 0])))
    _ecount, _adj, _off = {}, {}, 0
    for _c in counts:
        _f = [int(v) for v in idxs[_off:_off + _c]]
        _off += _c
        for _i in range(_c):
            _a, _b = _f[_i], _f[(_i + 1) % _c]
            _k = (_a, _b) if _a < _b else (_b, _a)
            _ecount[_k] = _ecount.get(_k, 0) + 1
            _adj.setdefault(_a, set()).add(_b)
            _adj.setdefault(_b, set()).add(_a)
    _badj = {}
    for (_a, _b), _n in _ecount.items():
        if _n == 1:
            _badj.setdefault(_a, []).append(_b)
            _badj.setdefault(_b, []).append(_a)
    _seen, _loops = set(), []
    for _s in _badj:
        if _s in _seen:
            continue
        _stack, _comp = [_s], []
        while _stack:
            _v = _stack.pop()
            if _v in _seen:
                continue
            _seen.add(_v)
            _comp.append(_v)
            _stack.extend(_q for _q in _badj[_v] if _q not in _seen)
        _loops.append(_comp)
    if not _loops:
        print("SHIRT-CONV: WARNING no boundary loops found; collar/cuffs not resized",
              flush=True)
    else:
        def _scale_loop(_verts, _scale, _label):
            """Scale one boundary loop (plus a half-scaled ring around it)
            radially about its own centroid -- the collar/cuff-resize idiom,
            shared so collar and cuffs behave identically."""
            _verts = sorted(_verts)
            _c0 = _p[_verts].mean(axis=0)
            _rad = lambda ii: float(_np.mean(_np.linalg.norm(
                _p[ii][:, :2] - _c0[:2], axis=1)))
            _r0 = _rad(_verts)
            # Blend a cuff change through the sleeve, not just one triangle
            # ring. An abrupt resized lip creates a narrow throat immediately
            # behind a large-looking opening and a stiff crease under FEM.
            if _label.startswith("cuff"):
                import heapq
                _radius = float(os.environ.get("SHIRT_CUFF_BLEND", "0.20"))
                _dist = _np.full(len(_p), _np.inf)
                _queue = []
                for _v in _verts:
                    _dist[_v] = 0.0
                    heapq.heappush(_queue, (0.0, _v))
                while _queue:
                    _d, _v = heapq.heappop(_queue)
                    if _d > _dist[_v] or _d >= _radius:
                        continue
                    for _q in _adj.get(_v, ()):
                        _nd = _d + float(_np.linalg.norm(_p[_q] - _p[_v]))
                        if _nd < _radius and _nd < _dist[_q]:
                            _dist[_q] = _nd
                            heapq.heappush(_queue, (_nd, _q))
                _u = _np.clip(1.0 - _dist / max(_radius, 1.e-6), 0.0, 1.0)
                _weight = _u * _u * (3.0 - 2.0 * _u)
                _p[:, :2] += (_p[:, :2] - _c0[:2]) * ((_scale - 1.0) * _weight[:, None])
            else:
                _ring = set()
                for _v in _verts:
                    _ring |= _adj.get(_v, set())
                _ring = sorted(_ring - set(_verts))
                for _sel, _s in ((_verts, _scale), (_ring, 1.0 + (_scale - 1.0) * 0.5)):
                    if _sel:
                        _p[_sel, :2] = _c0[:2] + (_p[_sel, :2] - _c0[:2]) * _s
            print(f"SHIRT-CONV: {_label} x{_scale}: {len(_verts)} loop verts, "
                  f"mean radius {_r0 * 1000:.1f}mm -> {_rad(_verts) * 1000:.1f}mm", flush=True)

        _neck_y = _p[:, 1].max() if _nhi else _p[:, 1].min()
        _best, _bi = None, -1
        for _i, _lp in enumerate(_loops):
            _cen = _p[_lp].mean(axis=0)
            _score = abs(float(_cen[0])) + abs(float(_cen[1]) - float(_neck_y))
            if _best is None or _score < _best:
                _best, _bi = _score, _i
        if _NECK_SCALE != 1.0:
            _scale_loop(_loops[_bi], _NECK_SCALE, "collar")

        if _CUFF_SCALE != 1.0:
            # The cuffs are the two loops sitting furthest out in X -- the
            # sleeve extremes -- among whatever is left once the collar (near
            # X=0) is excluded. The hem cut is wide (spans the whole garment
            # width) rather than off to one side, so it sorts last and is
            # left alone.
            _rest = [_i for _i in range(len(_loops)) if _i != _bi]
            _rest.sort(key=lambda _i: abs(float(_p[_loops[_i]].mean(axis=0)[0])),
                      reverse=True)
            for _ci in _rest[:2]:
                _side = "right" if _p[_loops[_ci]].mean(axis=0)[0] > 0 else "left"
                _scale_loop(_loops[_ci], _CUFF_SCALE, f"cuff({_side})")

        pts = Vt.Vec3fArray([Gf.Vec3f(*map(float, _q)) for _q in _p])
        print(f"SHIRT-CONV: {len(_loops)} boundary loops seen", flush=True)

# Rectify the finished pattern, after cuff shaping, before FEM refinement.
from straighten_shirt_sleeves import straighten_sleeves
_sleeve_reports = []
if str(source_mesh.GetPath()) == "/t_shirt/t_shirt_001/t_shirt_001":
    _p, _sleeve_reports = straighten_sleeves(_p, counts, idxs)
    from resize_shirt import resize_shirt
    _p, _resize_metadata = resize_shirt(
        _p, counts, idxs,
        float(os.environ.get("SHIRT_UNIFORM_SCALE", "1.2")),
        float(os.environ.get("SHIRT_NECK_AREA_SCALE", "1.2")),
        float(os.environ.get("SHIRT_HEIGHT_SCALE", "0.8333333333333334")))
    for _name, _value in _resize_metadata.items():
        mesh.GetPrim().CreateAttribute("phyrc:" + _name, Sdf.ValueTypeNames.Double, custom=True).Set(_value)
    print(f"SHIRT-CONV: uniform resize {_resize_metadata}", flush=True)
pts = Vt.Vec3fArray([Gf.Vec3f(*map(float, _q)) for _q in _p])
for _report in _sleeve_reports:
    print(f"SHIRT-CONV: straight sleeve {_report}", flush=True)

# Cut ceiling-facing front collar into a large, sharp symmetrical V-neck with prominent yellow stripe band.
_VNECK_FACES = []
if str(source_mesh.GetPath()) == "/t_shirt/t_shirt_001/t_shirt_001":
    _is_short = (len(_p) < 4500)
    _full_to_shirt = {
        5077: 3982 if _is_short else 5077,
        5076: 3981 if _is_short else 5076,
        5075: 3980 if _is_short else 5075,
        5074: 3979 if _is_short else 5074,
        5073: 3978 if _is_short else 5073,
        5072: 3977 if _is_short else 5072,
        5071: 3976 if _is_short else 5071,
        5069: 3974 if _is_short else 5069,
        5068: 3973 if _is_short else 5068,
        5067: 3972 if _is_short else 5067,
        5066: 3971 if _is_short else 5066,
        5065: 3970 if _is_short else 5065,
        5064: 3969 if _is_short else 5064,
        5063: 3968 if _is_short else 5063,
        5061: 3966 if _is_short else 5061,
        5062: 3967 if _is_short else 5062,
        5114: 4019 if _is_short else 5114,
    }
    for _v in [1166, 1165, 1167, 1169, 1181, 1193, 1145, 1147, 1148, 1150, 1151,
               1153, 1154, 1156, 1157, 1159, 1160, 1161, 1162, 1163, 1164, 1168, 1180]:
        _full_to_shirt[_v] = _v

    _faces_31_full = [
        [1151, 1153, 1154], [1154, 1153, 1156], [1154, 1156, 1157], [1157, 1156, 1159],
        [1157, 1159, 1160], [1160, 1159, 1162], [1156, 1161, 1159], [1159, 1161, 1164],
        [1160, 1162, 1163], [1163, 1162, 1165], [1159, 1164, 1162], [1164, 1165, 1162],
        [1163, 1165, 1166], [1164, 1167, 1165], [1167, 1164, 1168], [1167, 1168, 1169],
        [5063, 1167, 1169], [5063, 1165, 1167], [5063, 5065, 1165], [5063, 5064, 5065],
        [5063, 5061, 5064], [5061, 5066, 5064], [5065, 1166, 1165], [5065, 5071, 1166],
        [5064, 5071, 5065], [5064, 5072, 5071], [5066, 5072, 5064], [5066, 5074, 5072],
        [5066, 5073, 5074], [5066, 5067, 5073], [5067, 5075, 5073]
    ]

    _extra_rem_faces = [
        [5068, 5076, 5075], [5067, 5068, 5075], [5069, 5077, 5076], [5068, 5069, 5076],
        [1148, 1150, 1151], [1151, 1150, 1153], [1145, 1147, 1148], [1148, 1147, 1150]
    ]

    _all_rem_full = _faces_31_full + _extra_rem_faces
    _all_rem_shirt_set = {tuple(sorted([_full_to_shirt[_v] for _v in _f])) for _f in _all_rem_full}

    _face_off = 0
    _orig_faces = []
    _rem_indices = set()
    for _fi, _c in enumerate(counts):
        _f = [int(_v) for _v in idxs[_face_off:_face_off + _c]]
        _orig_faces.append(_f)
        _face_off += _c
        if tuple(sorted(_f)) in _all_rem_shirt_set:
            _rem_indices.add(_fi)

    if len(_rem_indices) == 39:
        _repl_full = [
            [5077, 5068, 5069], [5077, 5067, 5068], [5077, 5066, 5067],
            [1145, 1147, 1150], [1145, 1150, 1153], [1145, 1153, 1156],
            [5066, 5063, 5061], [1156, 1161, 1164], [1164, 1168, 1169]
        ]
        _repl_shirt = [[_full_to_shirt[_v] for _v in _f] for _f in _repl_full]

        _new_faces = []
        _old_to_new_fi = {}
        for _fi, _f in enumerate(_orig_faces):
            if _fi not in _rem_indices:
                _old_to_new_fi[_fi] = len(_new_faces)
                _new_faces.append(_f)

        for _f in _repl_shirt:
            _new_faces.append(_f)

        if _HEM_FACES:
            _HEM_FACES = [_old_to_new_fi[_fi] for _fi in _HEM_FACES if _fi in _old_to_new_fi]

        # Smooth, deep apex pull-down (v1169 down 30 mm) preserving triangle orientation
        _p[_full_to_shirt[1169], 1] -= 0.030
        _p[_full_to_shirt[1181], 1] -= 0.020
        _p[_full_to_shirt[1168], 1] -= 0.015
        _p[_full_to_shirt[5062], 1] -= 0.015
        _p[_full_to_shirt[1180], 1] -= 0.010
        _p[_full_to_shirt[5114], 1] -= 0.010
        _p[_full_to_shirt[1193], 1] -= 0.010

        # Define V-neck boundary polyline for thick stripe band
        _ch_full = [5077, 5066, 5063, 1169, 1164, 1156, 1145]
        _ch_shirt = [_full_to_shirt[_v] for _v in _ch_full]
        _seg_a = _p[_ch_shirt[:-1], :2]
        _seg_b = _p[_ch_shirt[1:], :2]

        def _dist_to_v(_pt):
            _d_min = float("inf")
            for _a, _b in zip(_seg_a, _seg_b):
                _ab = _b - _a
                _t = _np.clip(_np.dot(_pt - _a, _ab) / _np.dot(_ab, _ab), 0.0, 1.0)
                _proj = _a + _t * _ab
                _d = float(_np.linalg.norm(_pt - _proj))
                if _d < _d_min: _d_min = _d
            return _d_min

        _vneck_faces = []
        for _nfi, _f in enumerate(_new_faces):
            _f_pts = _p[_f]
            if _f_pts[:, 2].mean() < -0.0005:
                _cen = _f_pts[:, :2].mean(axis=0)
                if _dist_to_v(_cen) <= 0.041 and _cen[1] <= 0.446 and _cen[1] >= _p[_full_to_shirt[1169], 1] - 0.015:
                    if _cen[1] < 0.435 or abs(_cen[0]) < 0.065:
                        _vneck_faces.append(_nfi)

        _VNECK_FACES = _vneck_faces

        # Prune unused interior collar vertices and compact indexing
        _used_v = set(_v for _f in _new_faces for _v in _f)
        _sorted_used = sorted(_used_v)
        _v_map = {_old: _new for _new, _old in enumerate(_sorted_used)}

        _p = _p[_sorted_used]
        _faces_compact = [[_v_map[_v] for _v in _f] for _f in _new_faces]
        counts = _np.array([len(_f) for _f in _faces_compact])
        idxs = _np.array([_v for _f in _faces_compact for _v in _f])
        pts = Vt.Vec3fArray([Gf.Vec3f(*map(float, _q)) for _q in _p])
        print(f"SHIRT-CONV: large V-neck cutout applied ({len(_rem_indices)} faces removed, "
              f"{len(_VNECK_FACES)} trim band faces, {len(_p)} points remain)", flush=True)

_SUBDIV = int(os.environ.get("SHIRT_SUBDIV", "0"))
for _round in range(_SUBDIV):
    _tris, _off = [], 0
    for _c in counts:
        _f = [int(v) for v in idxs[_off:_off + _c]]
        _off += _c
        for _k in range(1, _c - 1):
            _tris.append((_f[0], _f[_k], _f[_k + 1]))
    _pl = [list(map(float, q)) for q in _p]
    _mid = {}

    def _midpoint(a, b):
        _key = (a, b) if a < b else (b, a)
        if _key not in _mid:
            _mid[_key] = len(_pl)
            _pl.append([(_pl[a][i] + _pl[b][i]) / 2.0 for i in range(3)])
        return _mid[_key]

    _newf = []
    _face_of = []          # which ORIGINAL face each new face came from
    for _fi, (_a, _b, _c3) in enumerate(_tris):
        _ab, _bc, _ca = _midpoint(_a, _b), _midpoint(_b, _c3), _midpoint(_c3, _a)
        for _q in ((_a, _ab, _ca), (_ab, _b, _bc), (_ca, _bc, _c3), (_ab, _bc, _ca)):
            _newf.append(_q)
            _face_of.append(_fi)
    _p = _np.array(_pl, dtype=float)
    counts = _np.array([3] * len(_newf))
    idxs = _np.array([v for f in _newf for v in f])
    # The hem GeomSubset is face indices, so it has to be remapped onto the new
    # faces or the red band lands on arbitrary triangles. Unconditional now --
    # see the note above the hem-band block for why this no longer gates on
    # KEEP_LENGTH.
    if _HEM_FACES:
        _old = set(int(i) for i in _HEM_FACES)
        _HEM_FACES = [i for i, src in enumerate(_face_of) if src in _old]
    if _VNECK_FACES:
        _old_vn = set(int(i) for i in _VNECK_FACES)
        _VNECK_FACES = [i for i, src in enumerate(_face_of) if src in _old_vn]
    pts = Vt.Vec3fArray([Gf.Vec3f(*map(float, _q)) for _q in _p])
    _elen = _np.linalg.norm(_p[idxs.reshape(-1, 3)[:, 0]]
                            - _p[idxs.reshape(-1, 3)[:, 1]], axis=1)
    print(f"SHIRT-CONV: subdivision {_round + 1}: {len(_p)} points, "
          f"{len(counts)} faces, median edge {_np.median(_elen) * 1000:.2f}mm",
          flush=True)

mesh.CreatePointsAttr(pts)
mesh.CreateFaceVertexCountsAttr(Vt.IntArray([int(c) for c in counts]))
mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([int(i) for i in idxs]))

# Normals are deliberately NOT copied: the cloth solver rewrites the points every
# step, and stale normals bake in the original pose's shading -- the same trap
# the mannequin's skinning bake hit. Leaving them unauthored lets the renderer
# compute smooth ones from the live geometry.
#
# But dropping them makes sidedness matter. USD meshes default to single-sided,
# and the source asset got away with that only because it shipped normals; with
# none, the renderer derives facing from winding alone and backface-culls the
# front of the shirt -- on screen the garment's front face simply is not there.
# Cloth is a thin sheet that has to draw from both sides regardless of winding,
# so say so explicitly.
if _HEM_FACES:
    from pxr import UsdGeom as _UG
    _subset = _UG.Subset.CreateGeomSubset(
        mesh, "hem", _UG.Tokens.face, Vt.IntArray([int(i) for i in _HEM_FACES]))
    _subset.CreateFamilyNameAttr("materialBind")

if _VNECK_FACES:
    from pxr import UsdGeom as _UG
    _v_subset = _UG.Subset.CreateGeomSubset(
        mesh, "vneck", _UG.Tokens.face, Vt.IntArray([int(i) for i in _VNECK_FACES]))
    _v_subset.CreateFamilyNameAttr("materialBind")

mesh.CreateDoubleSidedAttr(True)
# faceVarying UVs are indexed per face-corner, so a crop invalidates them --
# copying the original array onto a mesh with fewer faces is a length mismatch.
# The garments are flat-coloured here, so dropping them costs nothing.
uv = source_mesh.GetPrim().GetAttribute("primvars:st")
if KEEP_LENGTH >= 1.0 and uv and uv.Get():
    pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    pv.Set(uv.Get())

ext = UsdGeom.Boundable(mesh).ComputeExtent(Usd.TimeCode.Default())
if ext:
    mesh.CreateExtentAttr(ext)

dst.GetRootLayer().Save()
print(f"SHIRT-CONV: wrote {dst_path} with {len(pts)} points, "
      f"{len(counts)} faces at /World/mesh", flush=True)
if _app is not None:
    _app.close()
