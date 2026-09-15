"""
Same Stretch4 teleop rig as Teleop_Coat_Stretch4_Env.py, but with a
different garment: TNSC_Tshirt4, from the upstream garment dataset's OWN Garment/
HuggingFace tree (Assets/Garment/Tops/NoCollar_Ssleeve_FrontClose/
TNSC_Tshirt4/TNSC_Tshirt4_obj.usd) -- same authoring convention as the
coat asset Teleop_Coat_Stretch4_Env.py uses (mesh lives directly at
<root>/mesh, matching what Particle_Garment hardcodes), unlike an
earlier attempt with an RCareWorld-native t-shirt asset
(N_PR_2000_TShirt001) that needed a whole wrapper USD to work around a
mismatched internal mesh path and a baked-in local xformOp offset. None
of that machinery is needed here.

Base scale for this asset is 1.0, not the upstream dataset's usual 0.0085 --
it declares metersPerUnit=0.01 like most of the HuggingFace set, but
(per an earlier session's headless check) its raw mesh points are
already at real meter scale despite that metadata, same situation as
the RCareWorld hospital_gown/t_shirt assets. GARMENT_SCALE is set well
above 1.0 on request (bigger). GARMENT_ORI is left at identity; unlike
the coat, there was no reason yet to flip it -- adjust if it lands
facing the wrong way.

Four instances of it on the table now, not one -- red/green/blue/white
(GARMENT_COLORS), spread across GARMENT_X_OFFSETS on a table widened and
pushed back to fit them (BOX_SIZE/BOX_POS). state["grabbed"] is now
(cloth_index, idx, offsets) instead of just (idx, offsets), and
_attempt_grab() checks all 4 ClothPrims and picks whichever's nearest
particle is closest, since "what's in reach" now depends on which of
the 4 the gripper is actually near.

Grabbing is explicit-only now (0/SPACE), no passive proximity auto-grab
-- an earlier version auto-attached just from being near a garment,
which wasn't what was wanted. state["gripper_closed"] (visual finger
open/close) and state["grabbed"] (whether something's actually
attached) are deliberately separate: closing the gripper with nothing
in GRASP_PARTICLE_RADIUS still visibly closes the fingers, it just
doesn't attach anything ("nothing within 0.2m to grab" printed, no
crash -- closing on empty air is a normal, expected outcome, not a
failure state).

Kept as its own file rather than editing Teleop_Coat_Stretch4_Env.py so
that script stays untouched -- everything robot/box/pothook-related
below is copied unchanged from there; only the garment block differs.

Controls: identical to Teleop_Coat_Stretch4_Env.py (see its own
docstring) -- W/S lift, A/D arm, Q/E wrist yaw, R/V wrist pitch, Z/C
wrist roll, I/K base fwd/back, J/L base left/right, U/O base turn, 0/SPACE grab/release
toggle (explicit only -- see note above on no more proximity
auto-grab), ESC quit.

Usage::

    OMNI_KIT_ACCEPT_EULA=YES .venv/bin/python Env_StandAlone/Teleop_TShirt_Stretch4_Env.py
"""

import os

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": os.environ.get("STRETCH4_HEADLESS") == "1"})

# # [isaac-5.1.0 compat: teleop feel]
#
# 5.1.0 ships with /app/player/useFixedTimeStepping = False, so PhysX advances
# by however much wall time the last frame took (clamped by minFrameRate = 30)
# instead of a fixed 1/60. The teleop loop was written the other way round: it
# integrates every target with a hardcoded `dt = 1/60` once per iteration. With
# a variable timestep those two disagree by whatever the current frame rate
# happens to be, so the same key press moves the arm a different distance
# depending on load -- measured 31 Hz headless but only 14.7 Hz in a GUI
# session, i.e. the same input producing very different motion.
#
# Turning fixed timestepping back on restores the deterministic 1/60 the code
# assumes, which is what 4.5.0 was doing.
import carb.settings as _carb_settings  # noqa: E402

_settings = _carb_settings.get_settings()
_settings.set("/app/player/useFixedTimeStepping", True)

# Teleop feel knobs, overridable from the environment so the rates can be dialled
# in against a live session without rebuilding anything:
#     STRETCH4_LIFT_RATE=0.5 STRETCH4_ARM_RATE=0.4 ./run.sh gui orig
def _feel(name, default):
    try:
        return float(os.environ.get(f"STRETCH4_{name}", default))
    except ValueError:
        return float(default)

import sys
import numpy as np
import torch
import carb.input
import carb.settings
import omni.appwindow
from omni.kit.hotkeys.core import get_hotkey_registry

carb.settings.get_settings().set("/persistent/omnihydra/useSceneGraphInstancing", True)

sys.path.append(os.getcwd())
from Env_StandAlone.BaseEnv import BaseEnv
from Env_Config.Garment.Particle_Garment import Particle_Garment, SurfaceClothPrim
from Env_Config.Human.Human import Human
from Env_Config.Human.RandomSpawn import (
    randomize_human_and_chair, placement_snapshot, placement_matches)

from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.api.objects import FixedCuboid
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.api.materials.physics_material import PhysicsMaterial
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.string import find_unique_string_name
from pxr import Usd, UsdGeom, UsdSkel, UsdShade, Sdf, Gf, Vt, UsdPhysics

# RCareWorld-2.0 is a sibling directory to PhyRC_Sim.
STRETCH4_USD = os.path.abspath(
    os.path.join(os.getcwd(), "..", "RCareWorld-2.0", "assets", "robots", "stretch_4", "stretch_4.usd")
)
TSHIRT_USD = os.path.join(
    os.getcwd(), "Assets", "Garment", "Tops", "NoCollar_Ssleeve_FrontClose",
    "TNSC_Tshirt4", "TNSC_Tshirt4_obj.usd")
# Briefly swapped to RCareWorld-2.0's female1_cp.usd per "switch the mannequin
# to the female asset", reverted back to this original asset per "never mind,
# go back to the previous human-like mannequin".
HUMAN_USD = os.path.join(
    os.getcwd(), "Assets", "Human", "Collected_human_model", "biped_demo_meters.usd")
# Per "just remove the riro logo on it" -- the mannequin's own suit
# material's diffuse texture had a "RIRO LAB / Robust Intelligence &
# Robotics Lab" logo printed on the chest (confirmed by opening the
# actual PNG), inpainted out into a sibling "_nologo" copy (kept on
# disk, no longer referenced below -- superseded by the flat color
# override right after it per "render the mannequin in a single color, for
# example beige", which replaces the texture entirely anyway).
HUMAN_BEIGE_COLOR = (0.82, 0.71, 0.55)
# female2_c4-c5.usd, the replacement SMPL-X female body supplied in
# manikin_assets_male_female.zip and staged by run.sh.  The source contains no
# PhysX shapes; the posed hidden collision body is generated below.  Absent
# when STRETCH4_HUMAN=biped, in which case the original mannequin is used.
MESH_HUMAN_USD = os.path.join(
    os.getcwd(), "Assets", "Human", "Mesh", "manikin_exports",
    "female2_c4-c5.usd")


# Per "make the collar and hem especially large so it slips onto the mannequin
# easily, and make it shorter" -- widen the collar and hem openings
# specifically (not a uniform per-axis scale, which would just make the
# WHOLE shirt bigger/smaller without changing how easily it slips over
# the mannequin's head/hips) and shorten the shoulder-to-hem length.
#
# Must happen on the RAW mesh points BEFORE Particle_Garment() ever
# touches this usd_path -- Particle_Garment.__init__ constructs a
# SingleClothPrim immediately after add_reference_to_stage(), which reads
# whatever "points" are on the mesh at that moment to set up the PhysX
# particle cloth's rest state and auto-generated springs. Editing points
# on the LIVE prim any time after that (the same category of thing the
# abandoned pre-grab feature did earlier this session) creates an instant
# rest-state/stretch mismatch and risks the same violent spring snap-back
# that killed that feature -- so this bakes the new shape into a
# standalone derived USD file, and Particle_Garment is pointed at THAT
# instead, guaranteeing it never sees anything but the final geometry.
# Per "make the shirt a bit bigger, with a larger neck and hem, but keep
# the length the same" -- pushed the overall/neckline/hem scales up
# further, and VERTICAL_SCALE back to 1.0 (no compression -- the earlier
# shortening is explicitly undone here, length stays as authored).
TSHIRT_RESHAPE_OVERALL_XZ_SCALE = 1.12  # "make the shirt a bit bigger" -- overall width/depth bump
TSHIRT_RESHAPE_NECKLINE_SCALE = 1.50    # extra radial widen at the collar opening specifically
TSHIRT_RESHAPE_HEM_SCALE = 1.35         # extra radial widen at the bottom hem opening specifically
TSHIRT_RESHAPE_VERTICAL_SCALE = 1.0     # "keep the length the same" -- no shoulder-to-hem compression


def _garment_boundary_loops(counts, idxs, num_points):
    """Traces the mesh's boundary edges (edges belonging to only one
    face) into separate closed loops via connected-component search on
    the boundary-edge graph -- a plain T-shirt has 4: collar, hem, and
    two cuffs. Also returns full-mesh vertex adjacency (from ALL edges,
    not just boundary ones) for finding each loop's 1-ring neighbors."""
    edge_count = {}
    full_adj = {i: set() for i in range(num_points)}
    offset = 0
    for c in counts:
        face = idxs[offset:offset + c]
        offset += c
        for i in range(c):
            a, b = int(face[i]), int(face[(i + 1) % c])
            full_adj[a].add(b)
            full_adj[b].add(a)
            key = (a, b) if a < b else (b, a)
            edge_count[key] = edge_count.get(key, 0) + 1
    boundary_adj = {}
    for (a, b), n in edge_count.items():
        if n == 1:
            boundary_adj.setdefault(a, []).append(b)
            boundary_adj.setdefault(b, []).append(a)
    visited = set()
    loops = []
    for start in boundary_adj:
        if start in visited:
            continue
        stack, comp = [start], []
        while stack:
            v = stack.pop()
            if v in visited:
                continue
            visited.add(v)
            comp.append(v)
            for nb in boundary_adj[v]:
                if nb not in visited:
                    stack.append(nb)
        loops.append(comp)
    return loops, full_adj


def _reshape_tshirt_points(points, counts, idxs):
    """Returns a new points array with the collar+hem openings widened
    and the overall vertical (shoulder-to-hem) length shortened. Works
    entirely in the mesh's own local/raw coordinate space -- orientation-
    agnostic, doesn't need to know how GARMENT_ORI maps this into world
    axes, just which LOCAL axis is "vertical" for this garment, which it
    determines empirically (see below) rather than assuming."""
    points = np.array(points, dtype=np.float64)
    loops, full_adj = _garment_boundary_loops(counts, idxs, len(points))
    if len(loops) != 4:
        print(f"[Teleop] WARNING: expected 4 garment boundary loops (collar/hem/2 cuffs), found {len(loops)} -- skipping reshape")
        return points

    # Cuffs sit at the X extremes (sleeve tips); collar+hem sit near the
    # body centerline. Sort by |centroid X| -- the two largest are cuffs,
    # the two smallest are collar/hem. Whichever of those two remaining
    # loops has the larger centroid on the OTHER horizontal axis (found
    # by checking which of Y/Z has more spread across all 4 loop
    # centroids, i.e. the actual "vertical" axis for THIS mesh) is the
    # collar (higher), the other the hem (lower).
    centroids = [points[loop].mean(axis=0) for loop in loops]
    by_x = sorted(range(4), key=lambda i: -abs(centroids[i][0]))
    cuff_ids, torso_ids = by_x[:2], by_x[2:]
    vertical_axis = 1 if (max(c[1] for c in centroids) - min(c[1] for c in centroids)) >= \
                          (max(c[2] for c in centroids) - min(c[2] for c in centroids)) else 2
    horiz_axes = [a for a in (0, 1, 2) if a != vertical_axis]
    torso_ids.sort(key=lambda i: -centroids[i][vertical_axis])
    neck_id, hem_id = torso_ids[0], torso_ids[1]
    neck_loop, hem_loop = loops[neck_id], loops[hem_id]

    # 1-ring neighbors (excluding the loop itself and the other loops) --
    # widened by half as much as the loop itself, so the opening flares
    # smoothly into the body instead of puckering at a sharp seam.
    all_loop_verts = set(v for loop in loops for v in loop)
    def ring1(loop):
        loop_set = set(loop)
        ring = set()
        for v in loop:
            ring |= full_adj[v]
        return ring - loop_set - (all_loop_verts - loop_set)

    neck_ring, hem_ring = ring1(neck_loop), ring1(hem_loop)

    # 1) mild overall enlarge in the two horizontal axes.
    for a in horiz_axes:
        points[:, a] *= TSHIRT_RESHAPE_OVERALL_XZ_SCALE

    # 2) vertical (shoulder-to-hem) compression, pivoted at the garment's
    # own shoulder-to-hem midpoint so it shrinks toward the center rather
    # than dragging the whole thing toward one end. vertical_axis is
    # never in horiz_axes (excluded by construction above), so step 1's
    # scaling never touched it -- these centroid values are still valid.
    v_mid = (centroids[neck_id][vertical_axis] + centroids[hem_id][vertical_axis]) / 2.0
    points[:, vertical_axis] = v_mid + (points[:, vertical_axis] - v_mid) * TSHIRT_RESHAPE_VERTICAL_SCALE

    # 3) extra radial widen at collar/hem loops + their 1-ring neighbors,
    # in the horizontal plane, pivoted at that loop's OWN (already-
    # enlarged-by-step-1) centroid.
    for loop, ring, scale in [(neck_loop, neck_ring, TSHIRT_RESHAPE_NECKLINE_SCALE),
                               (hem_loop, hem_ring, TSHIRT_RESHAPE_HEM_SCALE)]:
        loop_idx = np.array(loop)
        center = points[loop_idx][:, horiz_axes].mean(axis=0)
        for a, c in zip(horiz_axes, center):
            points[loop_idx, a] = c + (points[loop_idx, a] - c) * scale
        if ring:
            ring_idx = np.array(sorted(ring))
            ring_scale = 1.0 + (scale - 1.0) * 0.5
            for a, c in zip(horiz_axes, center):
                points[ring_idx, a] = c + (points[ring_idx, a] - c) * ring_scale

    return points


def build_reshaped_tshirt_usd(source_path):
    """Bakes _reshape_tshirt_points into a standalone derived USD file
    sitting next to the source asset, and returns its path. Regenerated
    every launch (566 points, negligible cost) rather than cached, to
    keep the TSHIRT_RESHAPE_* constants above actually live-tunable."""
    out_path = os.path.join(
        os.path.dirname(source_path),
        os.path.splitext(os.path.basename(source_path))[0] + "_reshaped.usd",
    )
    stage = Usd.Stage.Open(source_path)
    mesh_prim = None
    for prim in stage.Traverse():
        if prim.GetTypeName() == "Mesh":
            mesh_prim = prim
            break
    if mesh_prim is None:
        print(f"[Teleop] WARNING: no mesh found in {source_path}, using it unmodified")
        return source_path
    mesh = UsdGeom.Mesh(mesh_prim)
    points = mesh.GetPointsAttr().Get()
    counts = mesh.GetFaceVertexCountsAttr().Get()
    idxs = mesh.GetFaceVertexIndicesAttr().Get()
    new_points = _reshape_tshirt_points(points, counts, idxs)
    mesh.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*p) for p in new_points]))
    mesh.CreateExtentAttr().Set(UsdGeom.PointBased.ComputeExtent(mesh.GetPointsAttr().Get()))
    stage.GetRootLayer().Export(out_path)
    return out_path


# [isaac-5.1.0 compat: wearable shirt]
_SHIRT_CHOICE = os.environ.get("STRETCH4_SHIRT", "modelink")
_SHIRT_ASSETS = {
    "modelink": os.path.join(
        os.getcwd(), "Assets", "Garment", "Tops", "Modelink", "t_shirt.usd"),
    "wearable": os.path.join(
        os.getcwd(), "Assets", "Garment", "Tops", "Wearable", "t_shirt.usd"),
}
# Both replacements sit near 0.016m particle spacing, against the shipped
# mesh's 0.034m -- so they share one set of fine margins below, and the
# coarse mesh keeps its own wide ones.
_SHIRT_PATH = _SHIRT_ASSETS.get(_SHIRT_CHOICE)
_SHIRT_FINE = bool(_SHIRT_PATH) and os.path.exists(_SHIRT_PATH)
if _SHIRT_FINE:
    # Deliberately NOT run through build_reshaped_tshirt_usd: that widens
    # the collar and hem of the ORIGINAL mesh by indexing its own
    # boundary loops, and means nothing on a different topology. These
    # assets already have openings wide enough to put on a body.
    TSHIRT_USD = _SHIRT_PATH
    print(f"[Teleop] shirt: {_SHIRT_CHOICE} -> {TSHIRT_USD}")
else:
    if _SHIRT_PATH:
        print(f"[Teleop] WARNING: {_SHIRT_CHOICE} shirt missing at "
              f"{_SHIRT_PATH}, falling back to the shipped mesh")
    TSHIRT_USD = build_reshaped_tshirt_usd(TSHIRT_USD)

# Command rates increased by 10% per user request.
LIFT_RATE = _feel("LIFT_RATE", 1.54)
ARM_RATE = _feel("ARM_RATE", 1.21)
WRIST_RATE = _feel("WRIST_RATE", 5.5)
GRIPPER_OPEN = 0.5
# 0.0 was the joint limit, not the point of contact -- see patch_meet.
GRIPPER_CLOSED = _feel("GRIPPER_CLOSED", 0.103)
GRIPPER_CLOSE_TIME = 0.4
GRIPPER_RATE = (GRIPPER_OPEN - GRIPPER_CLOSED) / GRIPPER_CLOSE_TIME
BASE_LINEAR_RATE = _feel("BASE_LINEAR_RATE", 0.616)
BASE_ANGULAR_RATE = _feel("BASE_ANGULAR_RATE", 2.86)
BASE_LINEAR_ACCEL = _feel("BASE_LINEAR_ACCEL", 1.32)
BASE_ANGULAR_ACCEL = _feel("BASE_ANGULAR_ACCEL", 4.4)
PREVENT_WHEEL_LIFT = int(_feel("PREVENT_WHEEL_LIFT", 1))

# Ported over from Teleop_TShirt_Stretch4_Hand_Env.py's grab-follow fix
# (same underlying bug, confirmed there first): the old formula here
# was step_scale = MAX_GRAB_STEP / dist (clamped to 1), which snaps
# 100% to target the instant the gap is already smaller than
# MAX_GRAB_STEP -- and every correction ALSO zeroed the grabbed
# particles' velocity. Together that's "teleport -> kill velocity ->
# teleport -> kill velocity" every correction, a textbook mechanism for
# a repeating shock (buzzing) rather than a smooth pull, and yanking a
# particle further than its stretch-constrained neighbors can follow in
# one step is what was producing local shrink/wrinkling. Replaced with
# a fixed-fraction (exponential) follow -- GRAB_SMOOTHING_ALPHA of the
# remaining gap per correction, never an instant full snap -- plus
# giving the grabbed particles their actual velocity instead of zero.
# # [isaac-5.1.0 compat: grab follow]
# The grab is a kinematic follow: the captured particles are written toward
# `gripper_pos + offset` every GRAB_UPDATE_INTERVAL frames, moving a fraction
# GRAB_SMOOTHING_ALPHA of the remaining distance each time. At the old 0.5 every
# 2 frames the particles lag the gripper badly, the (very stiff, 1e12) stretch
# springs wind up, and the cloth either snaps back or is flung.
#
# Measured on a 0.15 m/s lift of the settled garment, driving the same particle
# set the real predicate captures:
#     alpha 0.5, every 2 frames -> garment FELL 0.42m while the gripper rose 1.2m
#     alpha 1.0, every frame    -> garment rose 1.26m, tracking the gripper
#
# Writing the full remaining distance every frame makes the captured particles a
# genuine kinematic constraint instead of a lagging spring, which is what the
# mechanic wants. Note this is marginal rather than bulletproof: a sweep over
# lift speeds still blew up at 0.35 m/s while 0.15 and 0.70 held, so the
# behaviour is close to the solver's stability edge. The env's own blowup
# watchdog is what covers the remainder.
# [isaac-5.1.0 compat: smooth follow]
# 0.25, not 1.0: the garment should trail the gripper a little rather
# than be snapped onto it. See patch_smooth for the measurements.
GRAB_SMOOTHING_ALPHA = _feel("GRAB_SMOOTHING_ALPHA", 0.25)
# Now only a safety cap on the per-correction displacement for
# unusually large gaps, not the primary mechanism.
MAX_GRAB_STEP = 0.3
# How fast the held particles may chase the gripper, in m/s. This is what makes
# the cloth trail the gripper instead of being whipped along with it, per "smoothly, as if
# dragging the T-shirt directly with the mouse".
#
# MAX_GRAB_STEP above is a per-correction DISTANCE, so its real speed limit
# depends on the step rate: at 240 Hz its 0.3 m works out to 72 m/s, i.e. no
# limit at all. Expressed as a speed instead, the bound stays what it says it is
# whatever STRETCH4_PHYSICS_HZ is set to -- the same dt coupling that made an
# earlier measurement of this very behaviour read four times too fast.
#
# 1.0 m/s measured against a 0.35 m/s lift (scripts/drag_stability.py):
#   clamp 0.25 -> tracks 0.15 of the lift, 0.75m behind      (never catches up)
#   clamp 0.6  -> tracks 0.30, 0.62m behind                  (still rubber-bands)
#   clamp 1.0  -> tracks 0.87, 0.12m behind, wobble 0.0287   <-- chosen
#   clamp 2.0  -> tracks 0.99, 0.01m behind, wobble 0.0373
#   no clamp   -> tracks 1.00, 0.01m behind, wobble 0.0399
# 1.0 keeps hold of the garment while cutting the carry wobble by ~28% against no
# clamp, and leaves a visible hand's-width of lag -- which is the drag feel asked
# for, not an error. Below ~0.6 the particles simply cannot keep up and the grab
# reads as slipping.
# 1.0 -> 3.0 originally. The lift commands 1.4 m/s while the follow was clamped to
# 1.0, so the held particles fell behind and never caught up: measured
# 0.504m of lag at the top of a lift, which is the garment being left on
# the table while the gripper rises. At 3.0 the lag is 0.168m and the
# garment actually comes up (its top goes 0.87m -> 1.22m). 6.0 is worse
# again (0.208m) -- writing faster than the cloth can respond winds the
# stretch springs up instead of carrying it. 1.5m/s stays just above the
# fastest 1.4m/s control while limiting each 240Hz kinematic write to 6.25mm,
# approximately the cloth/body contact offset. That prevents a held node from
# skipping across an entire collision skin in one callback.
GRAB_FOLLOW_SPEED = _feel("GRAB_FOLLOW_SPEED", 1.5)
# A handful of nodes actually inside the jaws must keep up with the robot's
# render-rate joint updates. Driving the whole support patch this quickly is
# unstable, so only the closest nodes use this fast, zero-lag tier.
# [quad mesh: preserve grasp support area]
# Four times the surface samples require four times the grasp nodes.
# Keeping 12 fine nodes shrank the grip and failed the 5mm attachment test.
_MESH_SAMPLE_SCALE = 4 if int(os.environ.get("STRETCH4_MESH_REFINEMENT", "1")) else 1
GRAB_ANCHOR_COUNT = int(_feel("GRAB_ANCHOR_COUNT", 12 * _MESH_SAMPLE_SCALE))
GRAB_ANCHOR_SPEED = _feel("GRAB_ANCHOR_SPEED", 3.0)
# How many times follow_grabbed has actually run. Printed by the teleop loop
# next to the frame number, because "the correction now runs at the physics
# rate" is a claim that should be visible rather than assumed: it reads about
# 4x the frame count on a 240 Hz scene with a 60 Hz loop.
FOLLOW_CALLS = [0]

# [isaac-5.1.0 compat: garment weight]
# How heavy the garment is, and how hard gravity pulls on it.
#
# The grab pins a few dozen particles and writes their positions; the rest of
# the garment hangs off them through stretch springs stiff enough (1e12) to
# drag the pinned ones back down. With the dense 5134-point shirt at the stock
# weight that is exactly what happened -- measured on a 0.35 m/s lift, the
# DRIVEN particles themselves tracked only 0.08-0.17 of the lift and sat 0.73-
# 0.81m behind their target, and the garment simply stayed on the table. That
# is the "with a weak grip the T-shirt always slides down when lifted" symptom,
# and adding more driven particles barely moved it (60 particles still only
# reached 0.48).
#
# The fork env already tuned exactly this mesh down to 0.5 gravity and 0.005
# particle mass and holds it fine. Applied here only for the dense shirts --
# the shipped 566-point sheet was tuned at the stock weight and is left alone.
#
# Raised 0.6 -> 1.0 (real gravity) per "it keeps blowing around like in the wind": at 0.6 the cloth
# falls and is pushed around at six tenths of the speed it should, which reads
# as drifting in a breeze. Measured with scripts/probe_float.py: the garment
# comes to rest in 188 frames at 1.0 against 253 at 0.6, and 1.5 never settles
# at all. Damping was tried first and rejected -- it made the cloth WORSE
# (residual motion 0.0059 -> 0.0129 m/s from damping 0 to 20), which is why
# GARMENT_DAMPING exists but stays at zero.
#
# The cost is real and measured: at 120 driven particles the drag probe
# carries +0.033m of a 0.875m lift at 0.6 and +0.016m at 1.0, so a grab holds
# a little less. Both figures come from a probe that slips either way and does
# not reproduce the tuned 0.99 tracking, so trust the direction, not the size.
# Drop it back with STRETCH4_GARMENT_GRAVITY_SCALE=0.6 if the grab matters
# more than the drift.
GARMENT_GRAVITY_SCALE = _feel("GARMENT_GRAVITY_SCALE", 1.0 if _SHIRT_FINE else 1.2)
# Halved 0.005 -> 0.0025, then set straight to 0.001 per "set the cloth particle mass
# to 0.001". Briefly suspected (and reverted) as the cause of jittery
# "blowing in the wind" footage, but that turned out to be a request to
# fix it in VIDEO EDITING, not in the sim/model parameters -- so this
# stays at 0.001, not reverted. Override with GARMENT_PARTICLE_MASS=0.005
# to go back to the original if it does turn out to matter (fine-mesh
# branch only -- the else/coarse branch is untouched, unused by the
# current modelink/_SHIRT_FINE=True setup).
GARMENT_PARTICLE_MASS = _feel("GARMENT_PARTICLE_MASS", 0.001 if _SHIRT_FINE else 1e-2)
# Particle_Garment leaves this at 1e12 -- near-inextensible, and that
# stiffness is what transmits the whole garment's weight back into the
# handful of pinned particles and hauls them down.
#
# Measured with scripts/probe_pull.py, which holds 120 particles at each end
# the way the grab holds them and drives the two grips 0.375m apart:
#
#     1e12   held particles end 0.139m behind their grip, the cloth follows
#            28% of the pull, and ALL 240 of them are more than 5cm adrift
#     1e4    0.038m behind, follows 81%, 36 of 240 adrift
#     1e2    0.029m behind, follows 86%, 21 of 240 adrift
#
# 1e6 and 1e8 are indistinguishable from 1e12: against a 0.005kg particle at
# 240Hz anything above ~1e5 is simply rigid, so the whole interesting range
# sits at the bottom. At the stock value the fabric cannot lengthen at all,
# so pulling two grips apart puts the entire load into tearing the cloth out
# of them -- which is what "it keeps slipping out when stretched from both sides" looks
# like from the inside: the grip is still on those particles and they are a
# hand's width away from it. Letting the knit give is what keeps the grip.
# 1e2 holds better still but looks rubbery, so it stays a knob rather than
# the default. Briefly moved to 1e2 per "make the cloth stretch more easily", put
# back to 1e4, set to 1e3 per "set the stretch stiffness to 1e3" (a midpoint between
# the two measured points above, not itself directly measured by
# scripts/probe_pull.py), then back to 1e4 again per "restore the
# stretch setting".
GARMENT_STRETCH_STIFFNESS = _feel("GARMENT_STRETCH_STIFFNESS", 1e4)
# Correct the grabbed particles' position every GRAB_UPDATE_INTERVAL
# frames instead of every single frame, giving the cloth full physics
# steps to relax between corrections.
GRAB_UPDATE_INTERVAL = int(_feel("GRAB_UPDATE_INTERVAL", 1))
# Grab now requires particles to actually sit between the two fingertip
# links (point-to-segment distance to the gripper_fingertip_left_link/
# gripper_fingertip_right_link segment), not just within a generic sphere
# around a single center point -- GRASP_TOLERANCE is how far off that
# segment a particle can be and still count as "caught" between the
# fingers, replacing the old 0.2m-from-center-point GRASP_PARTICLE_RADIUS.
GRASP_TOLERANCE = _feel("GRASP_TOLERANCE", 0.045)
# Per "only grasp when the hold is truly secure, otherwise don't -- with a half
# grasp things fly off and the cloth doesn't follow well" -- the eligibility
# check below only required at least ONE particle inside GRASP_TOLERANCE,
# so a grab could succeed on a single stray particle barely inside the
# tolerance boundary. That genuinely explains the symptom: with only a
# handful (or one) particle actually being pulled by the grab-follow
# correction each frame, there isn't enough grabbed mass to drag the
# rest of the garment along -- the cloth's own elastic response can
# easily overpower that thin a grip, which reads as flapping/not
# following. GRASP_MIN_PARTICLES requires a real fistful of fabric
# (not just anything grazing the tolerance) before a grab is accepted
# at all; below this count the attempt is rejected outright, same as
# if nothing were in reach.
# 5 -> 12 per "the gripper should only grasp when it has a firm hold" -- requiring more
# particles inside tolerance before accepting the grab makes "confidently
# grabbed" a stricter bar (still all-or-nothing: below this, the attempt
# is rejected outright same as before). 12 -> 25 per "set it higher than
# 12 particles". Mesh has 566 total points, so 25 is still a
# small fraction (~4.4%) of the whole garment, not an unreachable bar.
# # [isaac-5.1.0 compat: grasp threshold]
# GRASP_MIN_PARTICLES was 25, which this mesh cannot supply. Measured on the
# settled garment (566 particles, 0.68m across, median nearest-neighbour spacing
# 0.0248m): within the 0.045m tolerance the BEST any point on the garment offers
# is 17 particles, median 6. Not "hard" -- arithmetically impossible, from 0% of
# the garment. Every grab attempt in a real session failed with
#     "nothing confidently caught between the fingertips
#      (need >= 25 particles within 0.045m)"
# while the fingers still closed visually, which reads exactly like the shirt
# being grabbed and then slipping through the gripper.
#
# The 25 came from raising 5 -> 12 -> 25 on the reasoning that 25 is only ~4.4%
# of the mesh's 566 points. That mixes up two different things: what matters is
# the LOCAL density inside the tolerance radius, not the fraction of the whole
# garment. Note the fork env, whose mesh is ~9x denser (5134 points), asks for
# only 5.
#
# NOT a 5.1.0 regression -- the geometry is the same on 4.5.0, so this grab has
# been unreachable all along. The disabled ENABLE_GRASP_NEAREST_FALLBACK was
# presumably what made grabbing work in practice before it was turned off.
#
# Reachability at the current 0.045m tolerance, from the same measurement:
#     threshold  4 -> reachable from 88% of the garment
#     threshold  6 -> 57%
#     threshold  8 -> 30%
#     threshold 10 -> 15%
# 6 keeps the "a real cluster, not one stray particle" intent while actually
# being attainable. Raise it via STRETCH4_GRASP_MIN_PARTICLES if grabs feel too
# easy; widening STRETCH4_GRASP_TOLERANCE instead would start catching fabric
# that is not between the fingers at all, which is the check's whole point.
GRASP_MIN_PARTICLES = int(_feel("GRASP_MIN_PARTICLES", 6))
# Particles actually dragged once a grab is accepted -- see the
# measurements at the selection site in _attempt_grab.
# Measured on the dense 5134-point shirt, 0.35 m/s lift, whole-garment tracking:
#     24 driven -> the driven particles themselves track 0.15, 0.75m behind
#     40        -> 0.33          60 -> 0.48        (garment stays put)
#    120        -> 0.99, 0.015m behind, garment rises 0.60 of the lift
#    250        -> 0.62 of the lift; no real gain, and the held patch spreads
#                  to 0.10m, which is a handful rather than a pinch
# 120 is the knee. Below it the garment slides out of the fingers on the way
# up, which is the reported "with a weak grip the T-shirt always slides
# down when lifted" -- the pinned particles cannot carry the rest of the cloth and the
# stretch springs drag them back down. Reachable only because the refill above
# draws from the whole chosen face, not just what sits between the fingertips.
# Raised 120 -> 200 for the two-handed pull, measured with probe_pull.py on
# this shirt at the new stretch stiffness: 120 leaves the held particles
# 0.032m behind their grip with 10% of them more than 5cm adrift, 200 leaves
# them 0.024m behind with 6%. The cost is the one the lift measurements
# above already name -- the held patch is a fistful rather than a pinch.
GRASP_DRIVEN_TARGET = int(_feel("GRASP_DRIVEN_TARGET", 64 * _MESH_SAMPLE_SCALE))
# How close the fingers must be to GRIPPER_CLOSED, and how long the cloth
# must then be left undisturbed, before a pending grab is taken.
GRAB_CLOSED_EPS = _feel("GRAB_CLOSED_EPS", 0.01)
GRAB_SETTLE_FRAMES = int(_feel("GRAB_SETTLE_FRAMES", 60))
# Single-layer grasping. The gap is the distance step that separates the
# near face of the garment from the far one -- a little under the fabric's
# own local thickness works; too small and one face gets split in two.
SINGLE_LAYER_GRASP = bool(int(_feel("SINGLE_LAYER_GRASP", 1)))
SINGLE_LAYER_GAP = _feel("SINGLE_LAYER_GAP", 0.012)
# Per "can you add a feature that attaches to the nearest mesh when the gripper closes?
# let's try it, and I'll revert right away if it's weird" -- the strict pincer check above is
# all-or-nothing: if no garment has >= GRASP_MIN_PARTICLES within
# GRASP_TOLERANCE of the fingertip segment, the grab just fails outright,
# even if the gripper is RIGHT next to a smaller/thinner clump of fabric
# just outside that radius. This fallback only kicks in when that strict
# check finds nothing at all -- it grabs the GRASP_MIN_PARTICLES closest
# particles overall (by point-to-segment distance, same metric, just no
# hard radius cutoff), from whichever garment has the single closest
# particle. Explicit toggle since this is an experiment per the user's
# own request -- flip off to fall back to the old strict-only behavior.
# Per "revert the attach-on-gripper-close code for now; it's unstable,
# so not usable yet" -- turned off after live testing. Mechanism kept in
# place (this was built with exactly this toggle for exactly this
# situation), not deleted, per "yet" -- revisit later if wanted.
ENABLE_GRASP_NEAREST_FALLBACK = False

# Per "when the interaction with the mannequin overlaps badly, the cloth and the Stretch robot
# just explode and vanish; can you prevent this? ... before it
# bounces" -- a severe collision violation (e.g. cloth badly interpenetrating
# the mannequin) makes the solver inject a large corrective velocity to
# push out of it; that's the very first symptom, well before positions
# actually go flying or turn to NaN. Watched for in the main loop below:
# if any garment particle's speed exceeds this, treat it as the leading
# edge of a blowup and pause the WHOLE simulation timeline right there
# (world.pause() -- freezes physics but keeps the window/render loop
# alive so the last-good frame stays visible, rather than continuing to
# step into the actual explosion or closing the app outright).
#
# 3.0 was too low -- confirmed via a frame-by-frame headless replay
# that the garment's own NORMAL initial drop-and-land onto the table
# legitimately spikes to ~4.1 m/s for a single frame on impact (frame 4
# in that replay), then decays back under 1 m/s within a few more
# frames. That's a real, harmless, one-off landing impact, not a
# blowup, and the old threshold (3.0) paused the whole sim on it every
# single launch ("it freezes right from the start"). Raised to 8.0 -- still too
# sensitive in real interactive use per "the safeguard is too sensitive" (a
# hard grab-drag or two robots' arms crossing near each other can
# legitimately push particle speeds well past a normal settle without
# it being an actual collision blowup). Raised again, further past any
# normal handling speed, and paired with BLOWUP_STREAK_TO_PAUSE below
# (also raised) so it takes a longer sustained run of high readings --
# not just a couple of forceful frames -- before actually pausing.
EXPLOSION_VELOCITY_THRESHOLD = 20.0
# Finite node speeds are constrained proactively by the post-physics limiter.
# This watchdog never rewinds state or releases a grab; an unsafe non-finite
# state is reported and paused in place for diagnosis.
ENABLE_BLOWUP_WATCHDOG = True

# Was kept large (0.045/0.036) on purpose for puffiness/gripper access
# ("so the garment keeps enough body/gap between its layers for the
# gripper to actually slide into"). Was briefly cut to 0.006/0.0045 per
# "minimize the collision margin", and bend/shear/solver/damping/gravity/
# mass were briefly pushed hard per "fully taut / almost rigid" -- all
# reverted back to those original values per "restore the cloth to its initial settings,
# the original" (scope confirmed: stiffness/weight/margin only -- GRASP_MIN_
# PARTICLES and the grab-follow velocity/smoothing fixes are real bug
# fixes, not part of that revert, and GARMENT_EDGE_MASS_FACTOR below is
# untouched, from an earlier separate request).
#
# Briefly cut ALL FOUR to a third (0.015/0.012) together per "reduce the cloth
# collision margin a bit; also the mannequin currently registers a collision
# as soon as the cloth gets near, which makes grasping hard", reverted back to 0.045/0.036 per "restore the cloth's
# collision margin".
#
# Per "raise the cloth self-collision margin, but make the margin against the gripper and
# mannequin thin, applied separately" -- these
# were coupled to the SAME two numbers the whole time, but the PhysX
# particle system API (and Particle_Garment's own constructor, which
# passes each straight through to SingleParticleSystem) already keeps
# them structurally separate:
#   - GARMENT_CONTACT_OFFSET/REST_OFFSET -> PhysX's own contactOffset/
#     restOffset on the particle system -- this pair governs collision
#     between the CLOTH PARTICLES and any NON-particle rigid body
#     (gripper fingers, mannequin, table). This is the "against the gripper
#     and mannequin" margin -- cut thin.
#   - GARMENT_PARTICLE_CONTACT_OFFSET/SOLID_REST_OFFSET -> PhysX's
#     particleContactOffset/solidRestOffset -- this pair governs
#     PARTICLE-vs-PARTICLE collision (the cloth against its own other
#     layers/folds, and against the other 3 garment instances). This is
#     the "cloth self-collision" margin -- raised instead of cut.
# Never actually needed new plumbing, just different numbers.
#
# Per "keep the cloth self-collision margin, but reduce the margin between the cloth and
# the gripper and mannequin a bit more" -- GARMENT_PARTICLE_CONTACT_
# OFFSET/SOLID_REST_OFFSET (the self-collision pair, right below) are
# UNTOUCHED here, only the gripper/mannequin-facing pair is cut further.
#
# 0.008/0.0065 turned out too thin -- combined with the mannequin's own
# already-tiny margin (0.0005/0.0002 below), the real buffer between
# cloth and mannequin surface dropped under 7mm, letting fast gripper/
# garment motion cross that gap within a single physics step before
# PhysX ever got a chance to push back (visible poke-through, not just
# a cosmetic issue). Nudged up to 0.011/0.009 first (a middle ground
# against the previous 0.015/0.012), then per "can you raise it a bit more?"
# up again to that previous 0.015/0.012 checkpoint outright -- a value
# already known to not have triggered any poke-through complaints
# before it was cut down in the first place.
GARMENT_CONTACT_OFFSET = _feel("GARMENT_CONTACT_OFFSET", 0.006 if _SHIRT_FINE else 0.015)
GARMENT_REST_OFFSET = _feel("GARMENT_REST_OFFSET", 0.004 if _SHIRT_FINE else 0.012)
GARMENT_PARTICLE_CONTACT_OFFSET = _feel("GARMENT_PARTICLE_CONTACT_OFFSET", 0.012 if _SHIRT_FINE else 0.06)
GARMENT_SOLID_REST_OFFSET = _feel("GARMENT_SOLID_REST_OFFSET", 0.009 if _SHIRT_FINE else 0.05)
# Per "drop the fine detail and keep only minimal, stiff, rigid-like motion" -- was
# pushed all the way to 500000 (with GARMENT_SOLVER_ITERATIONS=48), and
# an idle SETTLE test confirmed that stable through 600 frames -- but
# that only tests it sitting still. Per "making it too rigid makes it keep bouncing"
# (live grabbing/interacting with it produces bouncing/flinging), an
# extreme constraint stiffness combined with the grab-follow
# correction's own position edits (or contact against the mannequin)
# is a different stress case a passive settle test doesn't cover.
# Reverted back to the original 3000, no solver_position_iteration_
# count override, per explicit request -- GRASP_MIN_PARTICLES and the
# mannequin's own collision margin (both elsewhere in this file) were
# kept as they were; this revert is stiffness-only.
# # [isaac-5.1.0 compat: drape knobs]
#
# Both were plain constants, and both are 30x Particle_Garment's own default of
# 100. That is not an accident -- they were raised for "minimal, stiff, rigid-like
# motion only" and then reverted from 500000 back to 3000 -- but bend stiffness is
# exactly what resists DRAPE, and the request that follows ("I grabbed the top but
# the bottom barely sags under gravity") is the opposite
# ask. A number two requests disagree about has to be a knob, not a constant,
# so it can be measured rather than argued about:
#
#     ./probe.sh probe_drape.py STRETCH4_GARMENT_BEND_STIFFNESS=100
# # [isaac-5.1.0 compat: particle solver iterations]
#
# Particle_Garment defaults it to 16 and this environment never overrode it.
# Free cloth dragged sideways across the mannequin tunnels at that count --
# reproduced from the user's own F3 slot with probe_pairpull.py PAIR_MODE=
# slotdrag: 0 particles under the skin before the drag, 2 particles up to 36 mm
# under it afterwards, and NOT the held ones, so no amount of grab clamping
# reaches it. Position iterations are the solver's own budget for pushing
# contacts apart, and they cost frame rate rather than feel.
GARMENT_SOLVER_ITERATIONS = int(_feel("GARMENT_SOLVER_ITERATIONS", 64))
GARMENT_BEND_STIFFNESS = _feel("GARMENT_BEND_STIFFNESS", 3000.0)
GARMENT_SHEAR_STIFFNESS = _feel("GARMENT_SHEAR_STIFFNESS", 3000.0)
# Per "make the cloth edges (neck/sleeves/hem) rigid-like" -- ported from
# Teleop_TShirt_Stretch4_Hand_Env.py, see apply_edge_mass_boost's own
# comment for why mass (not per-region stiffness) is the lever here.
# 6 -> 12 per "strengthen the setting". Untouched by the "initial settings" revert
# above -- that request scoped to stiffness/weight/margin only.
GARMENT_EDGE_MASS_FACTOR = 12.0

# Per "the floor is too big, feel free to shrink it, and instead of the grid
# ... upgrade the design using the MuJoCo Stretch environment's background assets
# as reference" -- Real_Ground's own default_environment.usd asset doesn't
# expose a size knob at all (GroundPlane's own `size` param is only
# honored when it creates a BRAND NEW prim -- Real_Ground always
# references the USD asset into that path FIRST, so the prim already
# exists by the time GroundPlane runs and that branch never fires),
# so the only way to control size is to build the ground plane
# ourselves instead of going through Real_Ground. FLOOR_SIZE is a half-
# extent in meters, matching the same convention MuJoCo's own room_scene
# floor geom uses (its own size="3 2.5 .1"); picked close to that scale
# rather than the old ~50m default.
FLOOR_SIZE = 6.0
# Per "too busy" (the wood grain read as too busy/cluttered) -- swapped
# for a plain grey stone tile texture instead. This is the SAME basecolor
# image the upstream scene pack's own Assets/Scene/ground/tiling.usd already tries
# to use, but that usd's shader has the texture path hardcoded to another
# machine's absolute path (/home/isaac/GarmentLab/..., someone else's
# original dev environment) that doesn't exist here -- confirmed via a
# headless probe that raised a "file not found"-shaped dead reference.
# The actual jpg is very much present locally, just under THIS project's
# own path, so pointing our own already-working texture pipeline at it
# directly sidesteps tiling.usd's broken reference entirely rather than
# trying to patch that asset's absolute path.
FLOOR_TEXTURE = os.path.join(
    os.getcwd(), "Assets", "Scene", "kitchen", "kitchen_7", "textures", "2K-tiling_30_basecolor.jpg")
# 2048x2048 tile texture over a 12m-wide floor (FLOOR_SIZE=6 is a half-
# extent) -- 8x repeat puts one tile repetition at ~1.5m, a plausible
# real-world floor tile scale.
FLOOR_TEXTURE_REPEAT = 8.0

# [isaac-5.1.0 compat: single-garment scene]
# Table height = the mannequin's shoulder, so arms held straight out
# rest on it: 128.82 asset units x HUMAN_MESH_SCALE 0.009791 = 1.26m
# standing. Seated, the table is set to the HANDS instead (0.95m): the
# garment then starts at the same height as the outstretched arms, so
# once it is grasped a straight horizontal move slides it on, with no
# lifting or lowering in between.
# (0.836 - 0.306), so the table has to come down with it or the figure
# sits with its arms above the work surface.
# [isaac-5.1.0 compat: four shirts]
# 60% of what it was, per "lower the boxes a bit, to about 60%": 0.95 -> 0.57
# seated, 1.26 -> 0.756 standing.
#
# NOTE what that gives up, because those two numbers were not arbitrary. 0.95
# was the SEATED FIGURE'S HAND HEIGHT and 1.26 her SHOULDER HEIGHT, picked so a
# garment could be carried from the table onto her with a flat horizontal move
# -- no lifting or lowering in between, which is the whole reason the table was
# put at that height. At 60% the surface is ~0.4m below her hands, so dressing
# her now needs a lift as well as a reach.
#
# BOX_TOP_Z=0.95 (or just STRETCH4_BOX_TOP_Z=0.95) puts it back.
_BOX_TOP_FRACTION = 0.60
BOX_TOP_Z = float(os.environ.get(
    "BOX_TOP_Z",
    str(_BOX_TOP_FRACTION * (0.95 if os.environ.get("STRETCH4_SIT", "1") == "1"
                             else 1.26))))
# Widened a lot (X) to hold 4 shirts side by side with room to spare.
# Y is NEGATIVE now -- robot_spawn faces +Y (yaw=90 sends local+X ->
# world+Y, see spawn_stretch4), so a positive-Y table sits in FRONT of
# it; moved to -1.1 to put it BEHIND instead, per request. Box near edge
# (facing the robot) is now at -1.1+0.3=-0.8, a 0.9m gap from
# robot_spawn (y=0.1) -- comfortably past the ~0.2-0.3m danger zone that
# caused a spawn-overlap PhysX explosion earlier this session at closer
# distances.
# Sized to the garment on it, not fixed: the shirt has to lie FLAT for
# the folded hem to be visible at all, and on the old 0.45x0.25 top it
# draped over both edges and hid the fold. The shirt settles to
# 0.80m x 0.44m at scale 1.0 (measured with probe_fold.py), so the top
# follows GREEN_SHIRT_SCALE with a margin -- change the shirt size and
# the surface under it keeps up.
_BOX_MARGIN = float(os.environ.get("BOX_MARGIN", "1.08"))
_BOX_FIT = _BOX_MARGIN * float(_feel("GREEN_SHIRT_SCALE", 1.08))
# Deeper than the shirt needs (0.44 -> 0.60): the garment settles to
# 0.44m front-to-back and sat right on both edges, so a little more
# room stops it draping over them. The HEIGHT is not free to change --
# BOX_TOP_Z is the seated figure's hand height, which is what lets a
# grasped garment go on with a flat horizontal move.
# [isaac-5.1.0 compat: four shirts]
# ONE table's top, not the whole row: there are four of these now.
_BOX_W = 0.80 * _BOX_FIT
BOX_SIZE = np.array([_BOX_W, 0.60 * _BOX_FIT, BOX_TOP_Z])

# How much room a Stretch needs to get between two tables, measured rather than
# picked -- scripts/probe_basewidth.py, against this scene:
#
#   base_link + wheels           0.427 x 0.406m
#   whole robot below table top  0.695 x 0.406m, circumradius 0.491m about
#                                base_footprint, the axis a base turn spins on
#
# The BASE is the narrow part; the mast and the arm at rest overhang it and are
# still below the table top, so they are what would clip a table. Twice the
# circumradius is the circle the robot sweeps turning on the spot, so a gap that
# wide lets it drive through and then turn around in there, rather than only
# squeezing through facing one way.
#
# So 0.982m is the floor. What sits on top of it is the margin, and there are
# two reference points for it, both measured:
#
#   +0.25/side -> 1.482m   scripts/probe_lane.py drove a robot through on the
#                          teleop keys and it wandered 0.165m off the centre
#                          line getting there -- an omni base under a held drive
#                          key does not track straight -- so 0.25 is that drift
#                          rounded up, and the gap absorbs a sloppy line.
#   +0.11/side -> 1.2m     what is set here, per "make BOX_GAP=1.2 the default".
#
# 1.2 is wider than the 0.982m circle on paper, and that is not the same as
# being able to turn in it. scripts/probe_lane.py, run at both values:
#
#   1.482   drives through, and turns a full 360 between two tables
#   1.2     drives through cleanly, and JAMS at 57 degrees of a turn -- 0.109m
#           a side is less than the 0.165m the base drifts, so it arrives at the
#           table before it comes round
#
# That is a deliberate trade, not an oversight. Turning round is meant to happen
# out in BOX_LANE, which is two turning circles wide and where the same probe
# turns 360 with 0.46m to spare; the gaps are for CROSSING the row. What it
# costs the operator is that a robot which drives into a gap has to back out the
# way it came. BOX_GAP=1.482 buys the turn back, at 0.85m more scene width.
_BASE_SWEEP_R = 0.491
BOX_GAP = float(os.environ.get("BOX_GAP", "1.2"))
# Centre-to-centre, so a table and its neighbour are BOX_GAP apart edge to edge.
_BOX_PITCH = _BOX_W + BOX_GAP
# Four tables, centred on the row: -1.5, -0.5, +0.5, +1.5 pitches. Derived from
# the pitch rather than written out, so changing BOX_GAP moves the shirts with
# the tables instead of leaving them hanging over the gaps.
_GARMENT_X_OFFSETS_ENV = os.environ.get("GARMENT_X_OFFSETS", "")
GARMENT_X_OFFSETS = (
    [float(v) for v in _GARMENT_X_OFFSETS_ENV.split(",")]
    if _GARMENT_X_OFFSETS_ENV else
    [(i - 1.5) * _BOX_PITCH for i in range(4)])
# Directly in front of the figure, which stands at HUMAN_POS_Y (0.9).
# [isaac-5.1.0 compat: four shirts]
# Pushed away from the figure, per "move the boxes farther away from the mannequin"
# and then further again per "move the boxes further back ... with the Stretch robots in between".
# What sits between the two is BOX_LANE, and the robots spawn on its centre line.
#
# The seated figure's nearest point is y=0.854 -- her fingertips
# (scripts/probe_layout.py: she occupies y 0.854..1.620, hands y 0.854..0.975).
# At the ORIGINAL 0.40 the row's back edge sat at 0.75 against her 0.845: a
# 0.10m slot, which no part of a 0.69m-deep robot fits into. That was the
# complaint that started this.
#
# Measured in the pose this same patch sets a few hundred lines down: arms
# forward, spread 0, tipped 10 degrees DOWN. Both parts of the pose move this
# number, which is why it is measured rather than carried:
#
#   spread  0, elevation   0    0.845   arms straight forward, hands furthest forward
#   spread 10, elevation   0    0.854   what is set now
#   spread 20, elevation   0    0.880   the stock spread
#   spread  0, elevation  20    0.896   up; tried, reverted
#   spread  0, elevation -10    0.843   down; tried, reverted
#   spread  0, elevation -25    0.877   further down; tried, reverted
#
# Either angle costs reach, so any non-zero one pulls her front back from the
# 0.845 that arms-straight-out-and-parallel gives. Elevation costs as 1-cos --
# 1.5% at 10 degrees, 6% at 20, 9% at 25 -- and spread swings the hands
# sideways as well, which is why 10 degrees of it moves this less than 10
# degrees of elevation moves nothing at all.
#
# Re-measure with scripts/probe_layout.py if the arm pose changes again; the
# whole row of tables is placed off this one number.
_HUMAN_FRONT_Y = 0.843
# The lane is its OWN width now, not BOX_GAP's. They were the same number while
# "one robot lane" was the whole specification; the gaps between the tables have
# since been narrowed to 1.2 to keep the row compact, and narrowing the space the
# robots actually stand and work in was not what that asked for.
#
# Two turning circles wide (2 x 0.982m, scripts/probe_basewidth.py). A robot
# centred in it turns on the spot inside the first circle and still has half a
# circle -- 0.491m, its own radius -- of clear floor between it and the row
# behind, and the same again between it and the figure in front. Below 1.482
# (one circle plus the 0.25m/side drift scripts/probe_lane.py measured) it stops
# being able to turn around without risking a table; above that it is taste, and
# this is set for room to work rather than to the minimum.
_BASE_SWEEP_D = 2.0 * _BASE_SWEEP_R
BOX_LANE = float(os.environ.get("BOX_LANE", str(2.0 * _BASE_SWEEP_D)))
BOX_POS = np.array([0.0, float(os.environ.get(
    "BOX_POS_Y", str(_HUMAN_FRONT_Y - BOX_LANE - BOX_SIZE[1] / 2.0))),
                    BOX_TOP_Z / 2.0])

# X=180 (the previous attempt) flips about one horizontal axis; this
# flips about the OTHER one (Y) instead for the opposite-looking result,
# per "reverse it" -- both are 180 degree flips (order-independent as a
# final orientation), just around different in-plane axes, so which one
# actually looks "upside-down" the way it's meant to depends on how this
# asset's front/back and collar are authored. Swap back to (180,0,0), or
# try (180,180,0), if this isn't it either.
# Per "turn the T-shirt's default pose around, i.e. rotate it 180 degrees" -- a further 180
# degrees on top of the flip that was already here.
#
# Taken about Z, the world vertical: the garment lies flat on the table, so a
# yaw is the turn that reads as "turned around" from above -- it swaps which end the
# collar points to while keeping the shirt lying down. A flip about X or Y would
# instead turn it face-over, which is "flip over" rather than "turn around".
#
# Env-tunable because that reading is a judgement call, and the two previous
# values here were themselves arrived at by trying one axis and then the other:
#   STRETCH4_GARMENT_ORI="0,180,0"     what it was before this change
#   STRETCH4_GARMENT_ORI="180,180,0"   face-over instead of yawed
#   STRETCH4_GARMENT_ORI="0,0,0"       the raw asset pose
GARMENT_ORI = np.array([
    float(v) for v in os.environ.get("STRETCH4_GARMENT_ORI", "0,180,180").split(",")
])
GARMENT_SCALE = np.array([1.0, 1.0, 1.0])
# [isaac-5.1.0 compat: short green shirt]
# One garment wears a shorter shirt. It is a SEPARATE ASSET, cropped when it is
# built, not this asset scaled down.
#
# Scaling would have squashed the particle spacing along the shortened axis to
# about half -- roughly 0.015m down to 0.0075m -- while every collision offset is
# an absolute distance and would have stayed where it is. The sheet would start
# pushing itself apart and none of the grasp tuning would still apply. Cropping
# removes rows of particles instead, so spacing, offsets and handling are
# identical to the full-size shirts and only the garment is shorter.
_SHORT_SHIRT = os.path.join(
    os.getcwd(), "Assets", "Garment", "Tops", "Modelink", "t_shirt_short.usd")
# [isaac-5.1.0 compat: four shirts]
# All four wear the same cropped asset. Only green did before, so the other
# three would fall back to the full-length TSHIRT_USD and the four would not be
# comparable -- different length, different particle count, different handling,
# and only one of them carrying the 'hem' GeomSubset the coloured band binds to.
GARMENT_USD_BY_COLOR = {}
if os.path.exists(_SHORT_SHIRT):
    for _cn in ("red", "green", "blue", "white"):
        GARMENT_USD_BY_COLOR[_cn] = _SHORT_SHIRT

# Per-garment overrides. The scale is UNIFORM, which moves particle spacing with
# it -- 1.08 takes 0.0128m up to 0.0138m -- and every collision offset is an
# absolute distance that would otherwise stay put and end up too narrow for the
# cloth it belongs to. Each garment is its own Particle_Garment with its own
# particle system, so the offsets are scaled by the same factor and the ratio the
# handling was tuned at is preserved exactly: the fabric behaves the same, just
# photographed larger. 0.9 -> 1.08 is the shirt 20% bigger in every direction.
# [isaac-5.1.0 compat: four shirts] same scale for all four, and the tables are
# sized off this same number, so the shirt and the surface under it stay matched.
GARMENT_SCALE_BY_COLOR = {
    _cn: float(_feel("GREEN_SHIRT_SCALE", 1.08))
    for _cn in ("red", "green", "blue", "white")}
# Extra yaw for one garment, same 180 flip already applied to all of them.
# 180 + 180: turned again so the red hem band faces the FIGURE rather than
# the robots. A yaw about Z is all it takes -- it does not disturb which
# side of the shirt is up, so the fold still lands on the top face.
# [isaac-5.1.0 compat: four shirts] same yaw for all four
GARMENT_YAW_BY_COLOR = {
    _cn: float(_feel("GREEN_SHIRT_YAW", 0.0))
    for _cn in ("red", "green", "blue", "white")}
# The hem band carries a second material so the two ends can be told apart.
GARMENT_HEM_COLOR = (0.85, 0.05, 0.05)  # 1.4 -> 1.1 -> 1.0, per "make the shirt a little smaller"
# Spread across the box's width with generous gaps -- box half-width is
# 2.5m, these sit well inside that with margin to spare on both ends.
# One garment, centred on the table.
# [isaac-5.1.0 compat: four shirts]
# GARMENT_X_OFFSETS is NOT defined here any more. It is defined up with
# BOX_SIZE, which needs it to lay the tables out and comes ~60 lines earlier in
# this file -- defining it here as well would either shadow that one or, if the
# table block were left reading this one, raise NameError at import.
GARMENT_COLORS = {
    "red": np.array([1.0, 0.0, 0.0]),
    "green": np.array([0.0, 1.0, 0.0]),
    "blue": np.array([0.0, 0.0, 1.0]),
    "white": np.array([1.0, 1.0, 1.0]),
}
# The hem band is the shirt's COMPLEMENT, per garment.
#
# One shared red band worked while there was one green shirt; with four it says
# nothing -- a red band on the red shirt is invisible, which is exactly the
# "which end is the collar" question the band exists to answer. The complement
# is the maximum-contrast choice against each shirt without picking four colours
# by hand, and it keeps the band obviously a MARKING rather than part of the
# garment.
#
# White's complement is black: (1,1,1) -> (0,0,0). Not 1.0 - x exactly for the
# saturated three, because a pure complement of red is cyan at full brightness
# and reads as glowing; each is pulled slightly off full.
GARMENT_HEM_BY_COLOR = {
    "red": (0.05, 0.80, 0.85),      # cyan
    "green": (0.85, 0.05, 0.80),    # magenta
    "blue": (0.90, 0.85, 0.05),     # yellow
    "white": (0.05, 0.05, 0.05),    # black
}

# Replaces the pothook. Positive Y -- opposite side from the table
# (negative Y), same spot the pole used to occupy. 0.45, well under
# Human's own default 0.7, per "a bit smaller so it slips on easily" (small
# enough that a big shirt slips on easily) -- at this scale the model is
# roughly 1.99*0.45 =~ 0.9m tall (raw asset's own Y-extent times scale).
#
# Briefly re-derived for female1_cp.usd's own geometry (uniform 0.01
# scale, feet-to-floor Z offset -0.02588) per "switch the mannequin to the female
# asset", reverted back to these original values per "never mind, go back to the previous
# human-like mannequin".
HUMAN_POS = np.array([0.0, 2.5, 0.0])
# Non-uniform, per "make the mannequin shorter and the torso bigger" -- was uniform
# 0.4725 on all three. Same raw-axis mapping established elsewhere in
# this file (raw_y is the asset's own up/height axis, raw_x/raw_z are
# the horizontal plane): index 1 (raw_y, height) brought DOWN to 0.40
# for "make it shorter", indices 0 and 2 (raw_x/raw_z, the body's width and
# depth) brought UP to 0.60/0.55 for "make the torso bigger" -- shorter and
# stockier rather than a uniform resize. This scales the whole body
# (arms/legs/head included) non-uniformly, not just the torso mesh
# specifically -- true torso-only bulking would need per-joint scale
# on the spine/chest bones with compensating inverse-scale on their
# children (clavicles, neck) to keep the arms/head from inheriting it,
# which is a lot more machinery for what's a casual sizing request.
#
# Per "the mannequin is too small; make it taller only, keeping the shoulders and head" --
# index 1 (height) raised 0.40 -> 0.46 (~15% taller), indices 0/2
# (shoulder width/depth) left untouched since only height was asked for.
# Unlike the width/depth bump above, THIS one needed the "per-joint
# compensating inverse-scale" machinery the comment above says was
# skipped back then -- head size has its own explicit HEAD_SCALE_FACTOR
# in pose_arms_at_attention (per "make the face a bit smaller"), and since that's a
# LOCAL scale applied before this GLOBAL non-uniform scale, raising this
# axis alone would stretch the head vertically right along with the
# body unless compensated -- see HEAD_SCALE_FACTOR's own comment.
#
# Per "make the mannequin a bit bigger" -- this time "bigger" (size), not "taller"
# (height) specifically, read as a general further enlargement on ALL
# three axes (~12% each: 0.60/0.46/0.55 -> 0.67/0.52/0.62), not another
# height-only bump. No "keep the head" qualifier was repeated this time,
# but there's no reason to think that preference reversed either, so
# HEAD_SCALE_FACTOR below is still compensating -- now against all three
# axes' growth, not just Y -- to keep the head's absolute size exactly
# where it was originally tuned.
HUMAN_SCALE = np.array([0.67, 0.52, 0.62])


# Second robot, side by side with the first (same yaw, so it faces the
# same direction -- toward the table/human). Positioned at the box's own
# 1/3 and 2/3 points along its width (left edge to right edge), not an
# arbitrary offset, per request -- robot1 (left, IJKL) at 1/3, robot2
# (right, arrow keys) at 2/3.
_BOX_LEFT_EDGE = BOX_POS[0] - BOX_SIZE[0] / 2.0
_BOX_WIDTH = BOX_SIZE[0]
# One on each side of the table rather than both behind it, and far
# enough out that the arm is not inside it at spawn: the fingertips
# rest 0.448m in front of the base and the table is a FixedCuboid, so
# a robot parked at the old fixed 0.85m had its gripper inside the
# larger top. Derived from the table so the two cannot drift apart.
# [isaac-5.1.0 compat: four shirts]
# The robots CANNOT stay where they were. Both x and y were derived from a
# single table: x = half its width + 0.53 put one on each side of it, and y sat
# them on its centre line. There are four tables now, spread over
# 3 * _BOX_PITCH + _BOX_W = 7.3m, and that formula lands both robots inside the
# second and third of them -- a FixedCuboid overlapping an articulation at spawn
# is the PhysX explosion this scene has hit before, not a cosmetic problem.
#
# x: half a pitch out, so each robot starts squarely in front of one of the two
# middle tables rather than in a gap.
_ROBOT_X = float(os.environ.get("ROBOT_SIDE_X", str(_BOX_PITCH / 2.0)))
# y: the middle of the lane, per "place the Stretch robots in between" -- exactly
# halfway between the back edge of the row and the figure's nearest point, so
# BOX_LANE is split evenly and the robot has 0.491m of clear floor on each side
# of its own turning circle. Derived from BOX_POS and BOX_LANE rather than
# written down, so moving the row moves the robots with it and they stay in the
# middle instead of ending up against one side.
_ROBOT_Y = float(os.environ.get(
    "ROBOT_SPAWN_Y",
    str(float(BOX_POS[1]) + BOX_SIZE[1] / 2.0 + BOX_LANE / 2.0)))
ROBOT1_SPAWN = (-_ROBOT_X, _ROBOT_Y, 0.03)
ROBOT2_SPAWN = (_ROBOT_X, _ROBOT_Y, 0.03)
# Facing the garment, not the person: the robots sit either side of the
# table, so pointing each one inwards means driving FORWARD is driving
# at the garment, and the arm extends the same way. Both used to face
# +y, which left every approach needing a turn -- and a turn swings the
# gripper in an arc that drags whatever it is holding.
# [isaac-5.1.0 compat: four shirts]
# Both face the row of tables. 0 and 180 pointed them at each other across a
# single table, which was right while they flanked it; standing in the lane in
# front of the row, inward-facing would mean one robot working over its own
# shoulder. yaw 90 is +y (ROBOT_YAW_DEG below is the file's own statement of
# that), the row is at -y from the lane, so facing it is 270.
ROBOT1_YAW_DEG = float(os.environ.get("ROBOT1_YAW_DEG", "270"))
ROBOT2_YAW_DEG = float(os.environ.get("ROBOT2_YAW_DEG", "270"))
ROBOT_YAW_DEG = 90.0

# Robot 1 keeps the original WASD-ish scheme. Robot 2 is driven with
# arrow keys (base) + numpad (arm/wrist/lift), laid out spatially the
# same way as robot 1's letters (8/2 up/down like W/S, 4/6 left/right
# like A/D, 7/9 and the +/- keys standing in for Q/E and R/V since there
# aren't enough remaining numpad keys for a perfect 1:1 mapping, 1/3 for
# roll like Z/C). Verified these are the actual carb.input.KeyboardInput
# names via a headless probe (numpad keys are "NUMPAD_n", not "n").
ROBOT1_KEYMAP = {
    "base_fwd_pos": "I", "base_fwd_neg": "K",
    "base_strafe_pos": "L", "base_strafe_neg": "J",
    "base_turn_pos": "U", "base_turn_neg": "O",
    "lift_pos": "W", "lift_neg": "S",
    "arm_pos": "D", "arm_neg": "A",
    "yaw_pos": "E", "yaw_neg": "Q",
    "pitch_pos": "V", "pitch_neg": "R",
    "roll_pos": "C", "roll_neg": "Z",
}
ROBOT1_GRIP_KEYS = ("KEY_0", "SPACE")
ROBOT2_KEYMAP = {
    "base_fwd_pos": "UP", "base_fwd_neg": "DOWN",
    "base_strafe_pos": "RIGHT", "base_strafe_neg": "LEFT",
    "base_turn_pos": "NUMPAD_DIVIDE", "base_turn_neg": "NUMPAD_MULTIPLY",
    "lift_pos": "NUMPAD_8", "lift_neg": "NUMPAD_2",
    "arm_pos": "NUMPAD_6", "arm_neg": "NUMPAD_4",
    "yaw_pos": "NUMPAD_9", "yaw_neg": "NUMPAD_7",
    "pitch_pos": "NUMPAD_ADD", "pitch_neg": "NUMPAD_SUBTRACT",
    "roll_pos": "NUMPAD_3", "roll_neg": "NUMPAD_1",
}
ROBOT2_GRIP_KEYS = ("NUMPAD_0", "NUMPAD_ENTER")
RESET_KEY = "P"

# # [isaac-5.1.0 compat: state slots]
#
# Checkpoint slots. Hand-driven dressing needs somewhere to stand back up from:
# two minutes of careful teleop is otherwise thrown away by one bad strafe, and
# P resets all the way to the start. F1..F5 are five independent slots -- press
# an EMPTY one to save everything the scene currently is, press a FILLED one to
# load it back.
#
# Saved: every garment particle's position and velocity, both robots' root
# poses, joint positions and velocities, their command targets (lift / arm /
# wrist / gripper), whether each gripper is closed and exactly which particles
# it holds, and the viewport camera's own transform -- so a load comes back to
# the same view as well as to the same scene.
#
# Deliberately NOT saved: which keys are down at the moment of the save, and
# with them the base's velocity ramps (base_fwd / base_strafe / base_turn) and
# the root linear/angular velocity. Those three exist only to express "a key is
# being held right now"; restoring them would make the base creep for the half
# second they take to decay, with nothing pressed, which reads as a bug rather
# than as a restore.
#
# Slots are .npz files under STATE_DIR -- /output/states, i.e. the host's
# output/states/ -- so they outlive the session: F1 in tomorrow's run loads what
# F1 saved today. STATE_CLEAR_KEY empties every slot (pressed twice, to
# confirm), which is also how a slot is freed for re-use; SHIFT+Fn overwrites
# one slot in place without clearing the rest.
#
# F1 and F2 collide with Kit's own hotkeys (omni.kit.menu.utils/OpenRefGuide and
# menu_rename_prim_dialog, plus the content browser's Rename) exactly the way
# SPACE and F already did, and are disabled the same way. F3/F4/F5/F12 are
# unclaimed -- checked against the registry's full 59-hotkey list with
# scripts/probe_camstate.py rather than assumed.
STATE_SLOT_KEYS = ("F1", "F2", "F3", "F4", "F5")
STATE_CLEAR_KEY = "F12"
STATE_DIR = os.environ.get("STRETCH4_STATE_DIR", "/output/states_neckfixed_shortheight_handfit_mesh4" if int(os.environ.get("STRETCH4_MESH_REFINEMENT", "1")) else "/output/states_neckfixed_shortheight_handfit")
_STATE_GEOMETRY_REVISION = "20260911_neckfixed_shortheight_handfit_v1"
# One key that throws away every checkpoint in the session is worth a
# confirmation. ~3s at 60fps, and the arming lapses if it isn't answered.
STATE_CLEAR_CONFIRM_FRAMES = 180
# Start the session already inside a saved slot: STRETCH4_LOAD_SLOT=F3. The
# alternative is launching, sitting through the shader load and pressing the key
# by hand every time -- which is the whole cost of iterating on a scene that took
# two minutes of teleop to set up.
STATE_LOAD_SLOT = os.environ.get("STRETCH4_LOAD_SLOT", "").strip().upper()
_SHIFT_FLAG = getattr(carb.input, "KEYBOARD_MODIFIER_FLAG_SHIFT", 1)


def _state_slot_path(key):
    return os.path.join(STATE_DIR, f"slot_{key}.npz")


def state_slot_exists(key):
    return os.path.isfile(_state_slot_path(key))


def latest_state_slot():
    """Return the most recently saved checkpoint, or None if all are empty.

    Loading a slot does not make it newer: modification time changes only when
    save_state_slot writes (or overwrites) the file, which matches "most recent
    checkpoint" rather than "most recently visited state".
    """
    newest = None
    for order, key in enumerate(STATE_SLOT_KEYS):
        path = _state_slot_path(key)
        try:
            stamp = os.stat(path).st_mtime_ns
        except FileNotFoundError:
            continue
        except OSError as exc:
            print(f"[Teleop] cannot inspect checkpoint {path}: {exc}")
            continue
        candidate = (stamp, order, key)
        if newest is None or candidate > newest:
            newest = candidate
    return None if newest is None else newest[2]


def clear_state_slots():
    """Delete every slot file. Returns how many there were."""
    removed = 0
    for key in STATE_SLOT_KEYS:
        path = _state_slot_path(key)
        try:
            if os.path.isfile(path):
                os.remove(path)
                removed += 1
        except OSError as exc:
            print(f"[Teleop] could not delete {path}: {exc}")
    return removed


def _active_camera_path():
    """The prim path of the camera the viewport is actually looking through.

    Defaults to /OmniverseKit_Persp, which is what a GUI session opens with,
    but reads the viewport when one exists so a session driven from a different
    camera saves and restores THAT camera instead.
    """
    try:
        from omni.kit.viewport.utility import get_active_viewport
        viewport = get_active_viewport()
        if viewport is not None:
            return str(viewport.camera_path)
    except Exception:  # noqa: BLE001 -- headless has no viewport extension
        pass
    return "/OmniverseKit_Persp"


def _gf_value_for(attr, values):
    """Rebuild the value a USD attribute wants out of the floats saved for it.

    Types matter here: /OmniverseKit_Persp carries xformOp:translate as Vec3d
    and xformOp:rotateXYZ / xformOp:scale as Vec3f, and USD rejects a Vec3d
    written into a Vec3f attribute, so the saved floats are poured back into
    whatever class the attribute itself declares rather than into a fixed one.
    """
    vals = [float(v) for v in np.asarray(values, dtype=np.float64).reshape(-1)]
    try:
        cls = attr.GetTypeName().type.pythonClass
    except Exception:  # noqa: BLE001
        cls = None
    if cls is None:
        return vals[0] if len(vals) == 1 else vals
    try:
        return cls(*vals)
    except Exception:  # noqa: BLE001
        return vals[0] if len(vals) == 1 else vals


def _camera_snapshot():
    """Every authored xform op on the active camera, plus its lens.

    Saved op-by-op rather than as one world matrix because writing a matrix
    back is what does NOT work: the camera has translate/rotateXYZ/scale ops
    and no transform op, so a matrix has to either replace the op stack or lose
    to it. Measured with scripts/probe_camwrite.py -- authoring a fresh
    transform op left the viewport exactly where it was.
    """
    out = {}
    try:
        import omni.usd
        from pxr import UsdGeom

        stage = omni.usd.get_context().get_stage()
        cam_path = _active_camera_path()
        prim = stage.GetPrimAtPath(cam_path)
        if not prim or not prim.IsValid():
            return out
        names = []
        for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
            value = op.Get()
            if value is None:
                continue
            try:
                arr = np.asarray(value, dtype=np.float64).reshape(-1)
            except Exception:  # noqa: BLE001 -- an op type we cannot flatten
                continue
            out[f"cam_op_val_{len(names)}"] = arr
            names.append(op.GetOpName())
        out["cam_path"] = np.array(cam_path)
        out["cam_op_names"] = np.array(names)
        camera = UsdGeom.Camera(prim)
        for field, attr in (("cam_focal", camera.GetFocalLengthAttr()),
                            ("cam_haperture", camera.GetHorizontalApertureAttr()),
                            ("cam_clip", camera.GetClippingRangeAttr())):
            value = attr.Get()
            if value is not None:
                out[field] = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception as exc:  # noqa: BLE001
        print(f"[Teleop] camera not saved ({type(exc).__name__}: {exc})")
    return out


def _camera_restore(data):
    """Put the camera back where the save found it. Returns True if it moved."""
    if "cam_path" not in data:
        return False
    try:
        import omni.usd
        from pxr import Usd, UsdGeom

        stage = omni.usd.get_context().get_stage()
        cam_path = str(data["cam_path"])
        # A save taken through one camera is restored through that same camera,
        # so point the viewport back at it first if it has since changed.
        try:
            from omni.kit.viewport.utility import get_active_viewport
            viewport = get_active_viewport()
            if viewport is not None and str(viewport.camera_path) != cam_path:
                viewport.camera_path = cam_path
        except Exception:  # noqa: BLE001
            pass
        prim = stage.GetPrimAtPath(cam_path)
        if not prim or not prim.IsValid():
            print(f"[Teleop] camera not restored: no prim at {cam_path}")
            return False
        # The camera prim lives in the SESSION layer, which is the strongest
        # opinion on the stage -- an edit authored into the root layer is
        # composed away and the viewport never moves. Same measurement as
        # above; this is the half of it that took two probes to find.
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            for i, name in enumerate(str(n) for n in data["cam_op_names"]):
                attr = prim.GetAttribute(name)
                if not attr:
                    continue
                attr.Set(_gf_value_for(attr, data[f"cam_op_val_{i}"]))
            camera = UsdGeom.Camera(prim)
            for field, attr in (("cam_focal", camera.GetFocalLengthAttr()),
                                ("cam_haperture", camera.GetHorizontalApertureAttr()),
                                ("cam_clip", camera.GetClippingRangeAttr())):
                if field in data:
                    attr.Set(_gf_value_for(attr, data[field]))
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[Teleop] camera not restored ({type(exc).__name__}: {exc})")
        return False


def save_state_slot(key, garment_cloths, rigs):
    """Write the whole scene into slot `key`. Returns True on success.

    Module level, not a closure inside main(), so scripts/probe_slots.py can
    drive a real save/load round trip headlessly instead of testing a copy of
    this logic.
    """
    try:
        os.makedirs(STATE_DIR, exist_ok=True)
    except OSError as exc:
        print(f"[Teleop] {key} NOT saved: cannot create {STATE_DIR} ({exc})")
        return False

    payload = {
        "schema": np.array(1),
        "n_garments": np.array(len(garment_cloths)),
        "n_rigs": np.array(len(rigs)),
        "scene_geometry_revision": np.array(_STATE_GEOMETRY_REVISION),
    }
    payload.update(placement_snapshot(garment_cloths[0].prim.GetStage()))
    for i, cloth in enumerate(garment_cloths):
        payload[f"g{i}_pos"] = _to_np(cloth.get_world_positions())[0].astype(np.float32)
        payload[f"g{i}_vel"] = _to_np(cloth.get_velocities())[0].astype(np.float32)
    for r, rig in enumerate(rigs):
        robot = rig["robot"]
        root_pos, root_quat = robot.get_world_pose()
        payload[f"r{r}_root_pos"] = _to_np(root_pos).reshape(-1).astype(np.float64)
        payload[f"r{r}_root_quat"] = _to_np(root_quat).reshape(-1).astype(np.float64)
        payload[f"r{r}_jpos"] = _to_np(robot.get_joint_positions()).reshape(-1).astype(np.float64)
        payload[f"r{r}_jvel"] = _to_np(robot.get_joint_velocities()).reshape(-1).astype(np.float64)
        state = rig["state"]
        heading = state.get("heading")
        # One flat array rather than eleven scalars: npz keys are cheap but the
        # order is written down once, here and in the loader, and nowhere else.
        payload[f"r{r}_ctl"] = np.array([
            state["lift"], state["arm"], state["yaw"], state["pitch"], state["roll"],
            state["grip_pos"],
            np.nan if heading is None else float(heading),
            1.0 if state["gripper_closed"] else 0.0,
        ], dtype=np.float64)
        grabbed = state.get("grabbed")
        if grabbed is None:
            payload[f"r{r}_grab_ci"] = np.array(-1)
        else:
            ci, idx, offsets = grabbed
            if state.get('_native_enabled', False):
                from Env_Config.Garment.NativeGrasp import current_grasp_offsets
                offsets = current_grasp_offsets(rig)
            payload[f"r{r}_grab_ci"] = np.array(int(ci))
            payload[f"r{r}_grab_idx"] = _to_np(idx).reshape(-1).astype(np.int64)
            payload[f"r{r}_grab_off"] = _to_np(offsets).reshape(-1, 3).astype(np.float64)
    payload.update(_camera_snapshot())

    path = _state_slot_path(key)
    try:
        np.savez_compressed(path, **payload)
    except OSError as exc:
        print(f"[Teleop] {key} NOT saved: {exc}")
        return False
    grabs = ", ".join(
        f"robot{r + 1} holding {len(payload[f'r{r}_grab_idx'])}"
        for r in range(len(rigs)) if f"r{r}_grab_idx" in payload) or "nothing held"
    print(f"[Teleop] {key} SAVED -> {path} "
          f"({len(garment_cloths)} garment(s), {grabs}, camera "
          f"{'included' if 'cam_path' in payload else 'unavailable'})")
    return True


def load_state_slot(key, garment_cloths, rigs):
    """Put the scene back exactly as slot `key` found it. True on success.

    Anything whose shape disagrees with the live scene is skipped with a
    message rather than written: a slot saved against a different garment or a
    different robot would otherwise be poured into the wrong buffer, and the
    failure mode of that is a scene that looks subtly wrong rather than one
    that reports a problem.
    """
    path = _state_slot_path(key)
    try:
        data = np.load(path)
    except (OSError, ValueError) as exc:
        print(f"[Teleop] {key} NOT loaded: {exc}")
        return False

    with data:
        if ("scene_geometry_revision" not in data
                or str(data["scene_geometry_revision"]) != _STATE_GEOMETRY_REVISION):
            print(f"[Teleop] {key} NOT loaded: slot predates the collar/hand geometry revision. "
                  "Use a new slot; the old file has been preserved.", flush=True)
            return False
        if not placement_matches(garment_cloths[0].prim.GetStage(), data):
            print(f"[Teleop] {key} NOT loaded: human/chair placement differs. "
                  "Use this run's slots or the same HUMAN_SPAWN_SEED and randomization mode. "
                  "The saved file has been preserved.", flush=True)
            return False
        # Swept contact requires a non-intersecting starting surface. Older
        # checkpoints can already contain an arm through a triangle interior.
        # Validate ALL shirts before mutating any cloth or robot state.
        guard = _BODY_GEOM.get('sweep')
        if guard is not None:
            for i, cloth in enumerate(garment_cloths):
                if f'g{i}_pos' not in data:
                    continue
                saved = data[f'g{i}_pos']
                if tuple(cloth.get_world_positions().shape[1:]) != saved.shape:
                    continue
                tri = np.array(UsdGeom.Mesh(cloth.prim).GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
                edges = np.unique(np.sort(np.concatenate((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]])), axis=1), axis=0)
                points = _to_t(saved)
                intersections = guard.count_intersections(points,
                    torch.as_tensor(edges, dtype=torch.int32, device=points.device),
                    torch.as_tensor(tri, dtype=torch.int32, device=points.device))
                if intersections:
                    print(f'[Teleop] {key} NOT loaded: garment {i} already intersects the human '
                          f'({intersections} edge/face hits). Use an earlier clear slot or P. '
                          'The saved file has been preserved.', flush=True)
                    return False
            guard.clear_cache()
        for rig in rigs:
            if rig['state'].get('_native_enabled', False):
                from Env_Config.Garment.NativeGrasp import clear_native_grasp
                clear_native_grasp(rig, garment_cloths[0].prim.GetStage())
        for i, cloth in enumerate(garment_cloths):
            if f"g{i}_pos" not in data:
                print(f"[Teleop] {key}: no garment #{i} in this slot -- left alone")
                continue
            positions = cloth.get_world_positions()
            saved = data[f"g{i}_pos"]
            if tuple(positions.shape[1:]) != saved.shape:
                print(f"[Teleop] {key}: garment #{i} is {tuple(positions.shape[1:])} "
                      f"but the slot holds {saved.shape} -- left alone")
                continue
            positions[0] = _to_t(saved)
            cloth.set_world_positions(positions)
            velocities = cloth.get_velocities()
            velocities[0] = _to_t(data[f"g{i}_vel"])
            cloth.set_velocities(velocities)

        for r, rig in enumerate(rigs):
            if f"r{r}_jpos" not in data:
                print(f"[Teleop] {key}: no robot #{r} in this slot -- left alone")
                continue
            robot = rig["robot"]
            saved_jpos = data[f"r{r}_jpos"]
            live_jpos = _to_np(robot.get_joint_positions()).reshape(-1)
            if saved_jpos.shape != live_jpos.shape:
                print(f"[Teleop] {key}: robot #{r} has {live_jpos.shape[0]} joints "
                      f"but the slot holds {saved_jpos.shape[0]} -- left alone")
                continue
            try:
                robot.set_world_pose(_to_t(data[f"r{r}_root_pos"]),
                                     _to_t(data[f"r{r}_root_quat"]))
            except Exception as exc:  # noqa: BLE001
                print(f"[Teleop] {key}: robot #{r} root pose not restored "
                      f"({type(exc).__name__}: {exc})")
            robot.set_joint_positions(_to_t(saved_jpos))
            robot.set_joint_velocities(_to_t(data[f"r{r}_jvel"]))
            # The base is velocity-driven and its velocity is rewritten from the
            # command ramps every frame anyway, so the only honest value to put
            # here is the one those zeroed ramps imply: stopped.
            #
            # Through the articulation view, like the drive loop does, and NOT
            # through set_linear_velocity / set_angular_velocity: those are
            # no-ops on the GPU pipeline, and the only sign is a status-bar
            # warning -- "set_angular_velocities function is not supported for
            # the gpu pipeline, use set_velocities instead" -- which is exactly
            # what the first version of this put on screen.
            try:
                robot._articulation_view.set_velocities(_to_t(np.zeros((1, 6))))
            except Exception as exc:  # noqa: BLE001
                print(f"[Teleop] {key}: robot #{r} not stopped "
                      f"({type(exc).__name__}: {exc})")

            state = rig["state"]
            # [continuous cloth: clear command history on load]
            state.pop("_base_drive_target", None)
            state.pop("_compliance_history", None)
            state['_native_wait_for_fk'] = True
            ctl = data[f"r{r}_ctl"]
            (state["lift"], state["arm"], state["yaw"], state["pitch"],
             state["roll"], state["grip_pos"]) = (float(v) for v in ctl[:6])
            state["heading"] = None if np.isnan(ctl[6]) else float(ctl[6])
            state["gripper_closed"] = bool(ctl[7] >= 0.5)
            # Not restored, deliberately -- see STATE_SLOT_KEYS' comment.
            state["base_fwd"] = 0.0
            state["base_strafe"] = 0.0
            state["base_turn"] = 0.0
            # A grab that was still waiting for the fingers to close belongs to
            # the moment of the save, not to the restored scene; the grab
            # itself, if it completed, comes back below.
            state["pending_grab"] = False
            state["_settle"] = 0

            ci = int(data[f"r{r}_grab_ci"])
            if ci < 0 or ci >= len(garment_cloths):
                state["grabbed"] = None
            else:
                idx = _to_idx(data[f"r{r}_grab_idx"])
                offsets = _to_t(data[f"r{r}_grab_off"])
                n_points = garment_cloths[ci].get_world_positions().shape[1]
                if int(idx.max()) >= n_points:
                    print(f"[Teleop] {key}: robot #{r} held particles that garment "
                          f"#{ci} does not have -- restored without the grab")
                    state["grabbed"] = None
                else:
                    state["grabbed"] = (ci, idx, offsets)

        # [continuous cloth: synchronize restored articulation FK]
        SimulationManager.get_physics_sim_view().update_articulations_kinematic()
        camera = _camera_restore(data)

    print(f"[Teleop] {key} LOADED <- {path} (camera "
          f"{'restored' if camera else 'left as it was'})")
    return True


def _to_np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _to_t(x):
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(np.asarray(x, dtype=np.float32),
                            device=SimulationManager.get_physics_sim_device())


def _to_idx(x):
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(np.asarray(x), dtype=torch.long,
                            device=SimulationManager.get_physics_sim_device())


def _link_pos(rig, link_idx):
    # See Teleop_Coat_Stretch4_Env.py's own note on why this reads the
    # physics tensor view directly instead of wrapping the link in its
    # own SingleRigidPrim (that spammed a Hydra render-cache warning
    # every frame and stalled the whole sim).
    lt = _to_np(rig["robot"]._articulation_view._physics_view.get_link_transforms())
    return lt[0, link_idx, 0:3]


def _grasp_link_pos(rig):
    return _link_pos(rig, rig["grasp_link_idx"])


def _fingertip_positions(rig):
    return _link_pos(rig, rig["fingertip_left_idx"]), _link_pos(rig, rig["fingertip_right_idx"])


# The human asset's own default pose is a T-pose (arms straight out to the
# sides -- confirmed by its raw world bbox being 2.19m wide, way past any
# actual human's arm span at this asset's scale). Env_Config.Human.Human
# treats the whole thing as a static rigid collider and does nothing about
# pose, so getting a proper standing pose out of its default T-pose (arms
# spread straight out to the sides) means swinging just the two upper-arm
# joints from their rest transforms -- there's no runtime skeletal
# simulation afterward, this is purely a fixed pose bake matching "treat it
# as rigid" (treat as rigid after this). Originally "arms straight forward" (arms
# forward, reaching toward the approaching robot); switched to attention
# (attention -- arms straight down at the sides) per a later request by
# just changing desired_world_dir below, same swing-the-bone machinery.
#
# Rather than guess which local bone axis is "down the arm" for this
# specific rig (varies per asset, easy to get backwards), this measures
# the CURRENT world-space direction of each upper arm bone (shoulder ->
# elbow, from the skeleton's own rest transforms) and computes the
# minimal rotation that points it at the desired world direction instead
# -- robust to whatever local axis convention the rig actually uses.
def pose_arms_at_attention(stage, skel_root_path, arm_dir=None,
                           arm_spread_deg=0.0):
    """arm_dir overrides the target arm direction, in the asset's own raw
    space. None keeps the original straight-down (attention) pose.

    arm_spread_deg swings each arm OUT to its own side by that many degrees
    -- left arm left, right arm right. It cannot be folded into arm_dir,
    which is one direction shared by both arms; the side each arm belongs to
    is only known here, from that arm's own T-pose sign."""
    root_prim = stage.GetPrimAtPath(skel_root_path)
    # Human.py wraps the referenced asset in SingleXFormPrim, which
    # forces this prim's typeName to plain "Xform" -- clobbering the
    # "SkelRoot" type the referenced asset's own defaultPrim actually
    # has (confirmed via a headless probe: typeName was literally
    # "Xform", and BindingAPI(prim).GetSkeleton() returned None as a
    # result). Without a real SkelRoot, UsdSkel's own imaging pipeline
    # never evaluates any bound animation at all -- the mesh just always
    # renders its raw bind pose (the T-pose) regardless of what's
    # authored, which is why the first version of this function ran with
    # no errors but had literally no visible effect. Restore the type so
    # UsdSkel actually recognizes and evaluates this subtree.
    root_prim.SetTypeName("SkelRoot")
    skel_root = UsdSkel.Root(root_prim)
    binding_api = UsdSkel.BindingAPI(skel_root.GetPrim())
    skeleton = binding_api.GetSkeleton()
    if not skeleton:
        # Human.py references the asset directly at skel_root_path with
        # no separate BindingAPI rel authored on the SkelRoot itself in
        # this particular file -- the Skeleton prim is just a child.
        # # [isaac-5.1.0 compat: find skel by type]
        # Find the Skeleton by TYPE rather than by name. The original mannequin
        # keeps one at <root>/Root; female1_cp.usd nests its whole rig one level
        # down -- /World/Human/SMPLX_female/Skeleton -- because its own defaultPrim
        # is a plain Xform wrapping the SkelRoot. Hardcoding the name fails there
        # with "Accessed schema on invalid prim".
        skeleton = None
        for _p in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies()):
            if _p.IsA(UsdSkel.Skeleton):
                skeleton = UsdSkel.Skeleton(_p)
                break
        if not skeleton:
            raise RuntimeError(f"no UsdSkel.Skeleton under {skel_root_path}")
    cache = UsdSkel.Cache()
    skel_query = cache.GetSkelQuery(skeleton)
    joint_order = list(skeleton.GetJointsAttr().Get())
    local_xforms = list(skel_query.ComputeJointLocalTransforms(Usd.TimeCode.Default()))
    world_xforms = list(skel_query.ComputeJointSkelTransforms(Usd.TimeCode.Default()))

    def joint_idx(suffix):
        for i, tok in enumerate(joint_order):
            if str(tok).endswith(suffix):
                return i
        raise KeyError(f"no joint ending in {suffix} among {joint_order}")

    def parent_of(idx):
        tok = str(joint_order[idx])
        if "/" not in tok:
            return None
        parent_tok = tok.rsplit("/", 1)[0]
        for i, t2 in enumerate(joint_order):
            if str(t2) == parent_tok:
                return i
        return None

    # Desired final direction, in the asset's OWN raw (pre-Human-class-
    # correction) space: Human.py's default orientation=[90,0,0] rotates
    # raw (x,y,z) -> scene (x,-z,y) (a +90 deg rotation about X). Was
    # (0,-1,0) (raw -Y) for attention, arms hanging straight down in
    # the scene; briefly flipped to (0,1,0) (raw +Y, straight up) per
    # "raise both arms to the sky", then reverted back to (0,-1,0)/down per
    # "don't raise the hands, just stand at attention" -- straight down in the
    # Z-up scene again, same as the file's own original attention pose.
    #
    # Per "bring the mannequin's arms in slightly, but don't let the fingertips touch" -- ARM_CONVERGE_
    # DEG used to tilt each arm slightly INWARD from straight vertical.
    # Per "the mannequin's arms are rotated... use the original mannequin data as reference" -- a
    # headless screenshot showed a real mesh TEAR at the shoulder, and
    # zeroing PALM_ROLL_DEG alone (see below) didn't fix it, isolating
    # the actual cause to THIS: the non-uniform HUMAN_SCALE (0.60/0.40/
    # 0.55) is applied AFTER this pose is baked into raw mesh points, so
    # a non-axis-aligned target direction (straight down PLUS a sideways
    # lean) gets skewed by that per-axis scale on its way to world space
    # -- distorting the shoulder/armpit skin blend region enough to tear
    # it. A pure (0,-1,0) target has no sideways component for the
    # scale to skew, so straight-down-only survives it cleanly. Zeroed
    # out -- trusts the original rig's own bind pose again, un-leaned.
    ARM_CONVERGE_DEG = 0.0
    converge_rad = np.radians(ARM_CONVERGE_DEG)
    # Negative: base direction is DOWN (attention), not up -- see the
    # desired-final-direction comment above.
    up_component = -np.cos(converge_rad)
    inward_component = np.sin(converge_rad)

    # The swing rotation above (current_dir -> desired_world_dir) is the
    # MINIMAL rotation between two vectors -- it only controls which way
    # the arm points, not how it's twisted/rolled around that resulting
    # axis, so whatever roll the T-pose happened to have carries straight
    # through unchanged. PALM_ROLL_DEG was an ADDITIONAL twist about the
    # arm's own final axis, added per "so the palms face each other" (palms
    # facing inward) and mirrored per "the arms rotated outward; they should rotate inward".
    # Per "the mannequin's arms are rotated... use the original mannequin data as reference" -- that
    # 70 degree twist turned out to be the actual cause, not a cosmetic
    # issue: a headless screenshot showed a real mesh TEAR at the
    # shoulder/armpit, the classic "candy-wrapper" LBS skinning artifact
    # a large twist produces where blend weights between the torso and
    # arm joints aren't clean. Zeroed out -- trusts the original rig's
    # own bind-pose twist (still visible, unmodified) instead of adding
    # an artificial one, which also happens to fix the tear since the
    # swing-only rotation stays well within what the skin weights here
    # can handle cleanly.
    PALM_ROLL_DEG = 0.0

    # Per "the mannequin's arms are rotated... use the original mannequin data as reference" -- even
    # with PALM_ROLL_DEG/ARM_CONVERGE_DEG both zeroed (see above), a
    # headless screenshot still showed a real mesh tear, just moved down
    # to the elbow/forearm -- and disabling this whole function (raw
    # T-pose) confirmed the mesh itself is clean, isolating the cause to
    # the swing rotation itself: T-pose has the arm horizontal, so
    # reaching straight-down is a ~90 degree rotation concentrated
    # entirely in ONE joint (the shoulder), which is a textbook case for
    # LBS "candy-wrapper" collapse at the armpit -- large single-joint
    # rotations are a known weak point of linear blend skinning
    # regardless of target direction. The standard fix (real character
    # rigs split shoulder elevation between the scapula/clavicle and the
    # humerus/upper-arm) is applied here: give the clavicle a FRACTION of
    # the total swing first, so the upper arm only has to cover the
    # remainder on its own -- same total final direction, spread across
    # two joints' worth of skin weights instead of concentrated in one.
    CLAVICLE_SWING_FRACTION = 0.35

    # [isaac-5.1.0 compat: joint naming]
    # Two rigs, two naming conventions. The shipped mannequin uses Mixamo-ish
    # names (L_UpArm/L_LoArm/L_Clavicle); female1_cp.usd is a stock SMPL-X rig
    # and calls the same three joints left_shoulder/left_elbow/left_collar --
    # note that SMPL-X's 'shoulder' IS the upper arm and its 'collar' is the
    # clavicle, so the roles line up even though the words do not. Pick
    # whichever set this skeleton actually has rather than assuming one.
    _ARM_NAMING = [
        [("L_UpArm", "L_LoArm", "L_Clavicle"), ("R_UpArm", "R_LoArm", "R_Clavicle")],
        [("left_shoulder", "left_elbow", "left_collar"),
         ("right_shoulder", "right_elbow", "right_collar")],
    ]
    _clavicle_indices = []
    _arm_pairs = None
    for _cand in _ARM_NAMING:
        if all(any(str(_t).endswith(_s) for _t in joint_order) for _s in _cand[0]):
            _arm_pairs = _cand
            break
    if _arm_pairs is None:
        raise KeyError(f"unrecognised arm joint naming in {joint_order[:12]}")
    for shoulder_suffix, elbow_suffix, clavicle_suffix in _arm_pairs:
        si, ei, ci = joint_idx(shoulder_suffix), joint_idx(elbow_suffix), joint_idx(clavicle_suffix)
        _clavicle_indices.append(ci)
        shoulder_world = world_xforms[si]
        elbow_world = world_xforms[ei]
        clavicle_world = world_xforms[ci]
        current_dir = (elbow_world.ExtractTranslation() - shoulder_world.ExtractTranslation()).GetNormalized()

        # This arm's own T-pose sideways sign -- keeps left/right
        # mirrored without hardcoding either side.
        outward_sign = 1.0 if current_dir[0] >= 0 else -1.0
        if arm_dir is not None:
            desired_world_dir = Gf.Vec3d(*arm_dir)
            if arm_spread_deg:
                _sp = np.radians(float(arm_spread_deg))
                _cs, _sn = float(np.cos(_sp)), float(np.sin(_sp))
                desired_world_dir = Gf.Vec3d(
                    outward_sign * _sn,
                    desired_world_dir[1] * _cs,
                    desired_world_dir[2] * _cs)
        else:
            desired_world_dir = Gf.Vec3d(-outward_sign * inward_component, up_component, 0.0)
        roll_rot = Gf.Rotation(desired_world_dir, -outward_sign * PALM_ROLL_DEG)

        full_delta_rot = Gf.Rotation(current_dir, desired_world_dir)  # minimal rotation current -> desired
        clavicle_delta_rot = Gf.Rotation(full_delta_rot.GetAxis(), full_delta_rot.GetAngle() * CLAVICLE_SWING_FRACTION)

        # 1) Clavicle gets its fractional share of the swing. Gf.Rotation's
        # `*` follows the same row-vector, apply-left-operand-first
        # convention as Gf.Matrix4d (verified empirically elsewhere in
        # this file): (A * B).Transform(v) == B.Transform(A.Transform(v)).
        clavicle_parent_world_rot = world_xforms[parent_of(ci)].ExtractRotation()
        old_clavicle_world_rot = clavicle_world.ExtractRotation()
        new_clavicle_world_rot = old_clavicle_world_rot * clavicle_delta_rot
        new_clavicle_local_rot = new_clavicle_world_rot * clavicle_parent_world_rot.GetInverse()

        old_local_c = local_xforms[ci]
        old_scale_c = Gf.Transform(old_local_c).GetScale()
        tc = Gf.Transform()
        tc.SetRotation(new_clavicle_local_rot)
        tc.SetTranslation(old_local_c.ExtractTranslation())
        tc.SetScale(old_scale_c)
        local_xforms[ci] = tc.GetMatrix()

        # 2) Shoulder reaches the FULL target direction in world space --
        # its own LOCAL rotation (relative to the now-partially-rotated
        # clavicle) naturally ends up smaller than the full delta, since
        # the clavicle above it already contributed its share.
        old_shoulder_world_rot = shoulder_world.ExtractRotation()
        new_shoulder_world_rot = old_shoulder_world_rot * full_delta_rot * roll_rot
        new_shoulder_local_rot = new_shoulder_world_rot * new_clavicle_world_rot.GetInverse()

        old_local_s = local_xforms[si]
        old_scale_s = Gf.Transform(old_local_s).GetScale()
        ts = Gf.Transform()
        ts.SetRotation(new_shoulder_local_rot)
        ts.SetTranslation(old_local_s.ExtractTranslation())
        ts.SetScale(old_scale_s)
        local_xforms[si] = ts.GetMatrix()
        # [isaac-5.1.0 compat: arm length]
        # Lengthens the arm per "lengthen the arms by just 10%": scales the
        # elbow's own local translation (its offset from the shoulder --
        # the upper-arm bone) and the wrist's (its offset from the elbow
        # -- the forearm bone) by the same factor, so the whole arm grows
        # proportionally. Only the offset changes, not rotation, and
        # everything distal (hand, fingers) is already in the 'affected'
        # subtree below the clavicle (see patch_affected), so it rides
        # along through forward kinematics without its own edit. Was
        # briefly reverted to 1.0 (off), now 0.8 -- SHORTER, not longer,
        # per "shorten the arms by 20%" (the earlier ask was +10%; this is a
        # different, later request to shrink it instead).
        #
        # Girth now moves WITH length, per "the girth doesn't match between upper arm and forearm":
        # length-only stretched the skin between joints without
        # thickening it, so a shortened arm read as abnormally thick for
        # how short it now was. A joint's own local SCALE affects the mesh
        # skinned to IT, which is the segment BELOW it -- shoulder's scale
        # is upper-arm girth, elbow's is forearm girth -- so each is scaled
        # by the same factor as the LENGTH change of the bone it thickens
        # (elbow's translation = upper-arm length, so shoulder's scale
        # pairs with it; wrist's translation = forearm length, so elbow's
        # scale pairs with it). Wrist's own scale (hand girth) is left
        # alone -- only the upper arm/forearm was asked about.
        # Reverted to 1.0 (off) per "just restore the arms, it looks too strange" --
        # length+girth together read as unnatural, not just girth alone.
        _arm_len_scale = float(_feel("HUMAN_ARM_LENGTH_SCALE", 1.0))
        if _arm_len_scale != 1.0:
            old_local_s2 = local_xforms[si]
            old_scale_s2 = Gf.Transform(old_local_s2).GetScale()
            ts2 = Gf.Transform(old_local_s2)
            ts2.SetScale(Gf.Vec3d(old_scale_s2) * _arm_len_scale)
            local_xforms[si] = ts2.GetMatrix()
            old_local_e = local_xforms[ei]
            old_scale_e = Gf.Transform(old_local_e).GetScale()
            te = Gf.Transform(old_local_e)
            te.SetTranslation(Gf.Vec3d(old_local_e.ExtractTranslation()) * _arm_len_scale)
            te.SetScale(Gf.Vec3d(old_scale_e) * _arm_len_scale)
            local_xforms[ei] = te.GetMatrix()
            # Wrist isn't looked up above (only shoulder/elbow/clavicle
            # are), and its suffix isn't a simple derivation of the
            # elbow/shoulder one on both naming conventions, so it is
            # looked up fresh here, guarded exactly like head_idx: if it
            # is not found, skip the forearm only, upper-arm lengthening
            # still applies.
            _wi = None
            if shoulder_suffix.startswith(("left", "right")):
                try:
                    _wi = joint_idx(f"{shoulder_suffix.split('_')[0]}_wrist")
                except KeyError:
                    _wi = None
            elif shoulder_suffix[:2] in ("L_", "R_"):
                for _cand_wrist in (f"{shoulder_suffix[0]}_Hand", f"{shoulder_suffix[0]}_Wrist"):
                    try:
                        _wi = joint_idx(_cand_wrist)
                        break
                    except KeyError:
                        continue
            if _wi is not None:
                old_local_w = local_xforms[_wi]
                tw = Gf.Transform(old_local_w)
                tw.SetTranslation(Gf.Vec3d(old_local_w.ExtractTranslation()) * _arm_len_scale)
                local_xforms[_wi] = tw.GetMatrix()
            else:
                print(f"[Teleop] arm length: no wrist joint found for "
                      f"{shoulder_suffix} side, forearm not lengthened "
                      f"(upper arm still was)", flush=True)

    # Per "make the face a bit smaller" -- scales just the Head joint's own local
    # transform down (rotation/translation untouched), same "edit one
    # joint's local scale, let it cascade to its mesh weights via the
    # skinning bake below" trick as HAND_SCALE_FACTOR elsewhere in this
    # codebase. Head is a leaf joint in this rig (no jaw/eye sub-joints),
    # so this only affects head-region vertices.
    #
    # Per "the mannequin is too small; make it taller only, keeping the shoulders and head" --
    # HUMAN_SCALE's own height axis (index 1) went 0.40 -> 0.46 (see its
    # own comment). That's a GLOBAL scale applied on top of this LOCAL
    # head scale, so left alone the head would grow taller right along
    # with the body. Y is compensated by the inverse of that same ratio
    # (0.40/0.46) so the head's ABSOLUTE height ends up unchanged;
    # X/Z aren't touched since HUMAN_SCALE's own width/depth axes didn't
    # change. This is the literal "per-joint compensating inverse-scale"
    # HUMAN_SCALE's own comment mentions as the alternative to a uniform
    # body resize.
    # [isaac-5.1.0 compat: head scale optional]
    # The head shrink originally existed only to compensate the Mixamo-style
    # mannequin's non-uniform body scale -- female1_cp.usd's own proportions
    # were judged already right, and its rig names the joint 'head', not
    # 'Head', so the shrink silently never engaged for it (KeyError ->
    # head_idx = None). Extended per "make the head a bit smaller, about 90%" to also
    # try the lowercase SMPL-X name, and dropped the old Mixamo-specific
    # 0.7/(0.40/0.46) anisotropic ratio for a plain uniform 0.9 -- that ratio
    # was a correction for the OTHER mesh's own proportions, not a general
    # head-shrink factor.
    HEAD_SCALE_FACTOR = Gf.Vec3d(0.9, 0.9, 0.9)
    # Shortens the neck per "can you shorten the mannequin's neck?": the head
    # joint's own LOCAL TRANSLATION is its offset from its parent (the neck
    # joint), so scaling that translation toward zero pulls the head down
    # closer to the neck/shoulders without touching the head's own size or
    # rotation (HEAD_SCALE_FACTOR above) or the neck joint itself. A tunable
    # knob rather than a hardcoded ratio, unlike HEAD_SCALE_FACTOR, since no
    # target percentage was given -- 0.7 is a first guess.
    NECK_SCALE_FACTOR = float(_feel("HUMAN_NECK_SCALE", 0.7))
    try:
        head_idx = joint_idx("Head")
    except KeyError:
        try:
            head_idx = joint_idx("head")
        except KeyError:
            head_idx = None
    if head_idx is not None:
        old_local = local_xforms[head_idx]
        old_scale = Gf.Transform(old_local).GetScale()
        old_trans = Gf.Transform(old_local).GetTranslation()
        t = Gf.Transform(old_local)
        t.SetScale(Gf.CompMult(Gf.Vec3d(old_scale), HEAD_SCALE_FACTOR))
        t.SetTranslation(Gf.Vec3d(old_trans) * NECK_SCALE_FACTOR)
        local_xforms[head_idx] = t.GetMatrix()
    # Shrinks the whole upper body per "can you shrink the mannequin's upper body a bit?"
    # (length only -- tried length+girth together first, then "never mind,
    # revert and just shorten the mannequin's upper body" dropped the girth half) --
    # scales spine1/spine2/spine3's own local TRANSLATION (each segment's
    # offset from its own parent, shortening the torso the same way
    # NECK_SCALE_FACTOR shortens the neck). Girth (each joint's local
    # SCALE) is left at its original value now, unlike the first version.
    # Only the spine chain itself is touched -- pelvis/legs and everything
    # hanging off spine3 (neck, head, arms) keep their own shape, just
    # ride lower/closer together as the torso they are attached to
    # shortens. _torso_indices is collected here and added to `affected`
    # below (see patch_affected) -- without that, editing these joints'
    # local_xforms would be silently ignored, the same trap head_idx/
    # _clavicle_indices exist to avoid.
    TORSO_SCALE_FACTOR = float(_feel("HUMAN_TORSO_SCALE", 0.8))
    _torso_indices = []
    if TORSO_SCALE_FACTOR != 1.0:
        for _cands in (("spine1", "Spine1", "Spine"), ("spine2", "Spine2"),
                      ("spine3", "Spine3")):
            _sp_idx = None
            for _nm in _cands:
                try:
                    _sp_idx = joint_idx(_nm)
                    break
                except KeyError:
                    continue
            if _sp_idx is None:
                continue
            _torso_indices.append(_sp_idx)
            _old_sp = local_xforms[_sp_idx]
            _old_sp_trans = Gf.Transform(_old_sp).GetTranslation()
            _tsp = Gf.Transform(_old_sp)
            _tsp.SetTranslation(Gf.Vec3d(_old_sp_trans) * TORSO_SCALE_FACTOR)
            local_xforms[_sp_idx] = _tsp.GetMatrix()
        print(f"[Teleop] torso length x{TORSO_SCALE_FACTOR}: "
              f"{len(_torso_indices)} spine joints shortened", flush=True)

    # Authoring a bound SkelAnimation (first version of this function) ran
    # with no errors but had zero visible effect -- the resulting
    # SkelAnimQuery came back invalid (GetAnimQuery() -> None) even after
    # fixing the SkelRoot typeName and properly populating the UsdSkel
    # cache, for reasons not fully tracked down. Skinning the mesh points
    # directly sidesteps that whole binding-resolution question: no
    # SkelAnimation, no animationSource relationship, nothing for UsdSkel
    # to (fail to) resolve at render time. This computes each joint's new
    # world transform via forward kinematics (only L_UpArm/R_UpArm and
    # their descendants actually move -- everything else keeps its rest
    # world transform), turns that into a per-joint "skinning transform"
    # (rest_world^-1 * new_world, in skeleton space, exactly what
    # UsdSkel.SkinPointsLBS expects), and overwrites Body_Mesh's points
    # attribute with the result once. Matches "treat it as rigid" precisely --
    # after this the mesh is just static geometry, no different from any
    # other rigid asset in the scene.
    # Rooted at the CLAVICLE now, not the upper arm directly -- the
    # clavicle also got its own local rotation edit above (its share of
    # CLAVICLE_SWING_FRACTION), and since it's the upper arm's own
    # parent, rooting the cascade here picks up both joints (and
    # everything distal to them, same startswith-prefix match below) in
    # one pass.
    # [isaac-5.1.0 compat: affected joints]
    # Reuse the clavicle indices the arm loop already resolved instead of
    # looking them up by Mixamo-style name a second time -- that second lookup
    # is what still failed on the SMPL-X rig after the loop itself had been
    # taught both naming conventions. head_idx is optional for the same reason
    # the head shrink is.
    affected = set(_clavicle_indices)
    if head_idx is not None:
        affected.add(head_idx)
    affected.update(_torso_indices)
    affected_roots = [str(joint_order[i]) for i in affected]
    for i, tok in enumerate(joint_order):
        tok_s = str(tok)
        if any(tok_s == st or tok_s.startswith(st + "/") for st in affected_roots):
            affected.add(i)

    # # [isaac-5.1.0 compat: fist]
    # Close the fingers, in the same pass as the arms -- see patch_fist.
    _fist_deg = float(os.environ.get("HUMAN_FIST_DEG", "110"))
    if _fist_deg:
        _n_curled = 0
        # Same value on both hands -- read once here rather than inside the
        # per-side loop below, so the summary print after the loop still has
        # it even if neither side's joints resolve (both iterations `continue`).
        _thumb_deg = float(os.environ.get("HUMAN_THUMB_DEG", "35"))
        for _side in ("left", "right"):
            _fingers = ("index", "middle", "ring", "pinky", "thumb")
            _names = [f"{_side}_{_f}{_n}" for _f in _fingers for _n in (1, 2, 3)]
            _idx = {}
            for _nm in _names + [f"{_side}_wrist"]:
                for _i, _tok in enumerate(joint_order):
                    if str(_tok).rsplit("/", 1)[-1] == _nm:
                        _idx[_nm] = _i
                        break
            if len(_idx) < len(_names) + 1:
                continue

            def _pos(_nm):
                return Gf.Vec3d(world_xforms[_idx[_nm]].ExtractTranslation())

            def _bone(_a, _b):
                return Gf.Vec3d(_pos(_b) - _pos(_a)).GetNormalized()

            _knuckle = Gf.Vec3d(_pos(f"{_side}_pinky1")
                                - _pos(f"{_side}_index1")).GetNormalized()
            _palm_n = Gf.Vec3d(np.cross(
                np.asarray(_bone(f"{_side}_middle1", f"{_side}_middle2")),
                np.asarray(_knuckle)).tolist()).GetNormalized()
            # Which way is "into the palm"? Distance to the wrist cannot say --
            # a finger folded backwards over the knuckles also gets closer to it,
            # and that is the hand the first version produced while its own check
            # passed. The rig knows: a modelled hand carries a slight natural
            # curl, so phalanx-to-phalanx already leans palmwards. All four
            # fingers agree on the sign here, which is what makes it trustworthy.
            _acc = np.zeros(3)
            for _f in ("index", "middle", "ring", "pinky"):
                _d1 = np.asarray(_bone(f"{_side}_{_f}1", f"{_side}_{_f}2"))
                _d2 = np.asarray(_bone(f"{_side}_{_f}2", f"{_side}_{_f}3"))
                _acc += _d2 - _d1 * float(_d2 @ _d1)
            _sign = -1.0 if float(_acc @ np.asarray(_palm_n)) < 0 else 1.0

            for _k, _nm in enumerate(_names):
                _f = _fingers[_k // 3]
                # ONE hinge axis per finger, taken from the rest pose and
                # expressed in the parent's rest frame so it travels with the
                # finger. Recomputing it from the current bone direction looked
                # more principled and was worse: cross(bone, palm) flips sign
                # once a phalanx passes 90 degrees, so the outer joints undid
                # the inner ones and 120 degrees curled LESS than 80.
                # The thumb is not a fourth finger. Curling it on the finger
                # hinge left it hanging below the fist; a fist folds it ACROSS
                # the front of the fingers, which is a rotation about the palm
                # normal. Its direction needs its own test too -- the sign taken
                # from the four fingers swung it away from the hand. Measured
                # with scripts/dump_fistcheck.py: thumb tip to the index's middle
                # phalanx 7.2 -> 1.6 units this way, against 4.4 the other.
                if _f == "thumb":
                    _axis = _palm_n
                    _deg = _thumb_deg
                    _to_index = np.asarray(_pos(f"{_side}_index2"))
                    _tip0 = np.asarray(_pos(f"{_side}_thumb3"))
                    _rot = Gf.Rotation(_axis, _deg)
                    _base = np.asarray(_pos(f"{_side}_thumb1"))
                    _moved = np.asarray(_rot.TransformDir(
                        Gf.Vec3d(*(_tip0 - _base).tolist()))) + _base
                    _sgn = (1.0 if np.linalg.norm(_moved - _to_index)
                            < np.linalg.norm(_tip0 - _to_index) else -1.0)
                else:
                    _axis = Gf.Vec3d(np.cross(
                        np.asarray(_bone(f"{_side}_{_f}1", f"{_side}_{_f}2")),
                        np.asarray(_palm_n)).tolist()).GetNormalized()
                    _deg = _fist_deg
                    _sgn = _sign
                _j = _idx[_nm]
                _p = parent_of(_j)
                _pw = world_xforms[_p].ExtractRotation() if _p is not None else Gf.Rotation()
                _delta = Gf.Rotation(_pw.GetInverse().TransformDir(_axis),
                                     _sgn * _deg)
                _old = local_xforms[_j]
                _t = Gf.Transform()
                _t.SetRotation(_old.ExtractRotation() * _delta)
                _t.SetTranslation(_old.ExtractTranslation())
                _t.SetScale(Gf.Transform(_old).GetScale())
                local_xforms[_j] = _t.GetMatrix()
                affected.add(_j)
                _n_curled += 1
        print(f"[Teleop] fingers curled into fists: {_n_curled} joints at "
              f"{_fist_deg} degrees, thumb {_thumb_deg} across them "
              f"(fingertip-to-wrist -46%, thumb-to-index 7.2->1.6, "
              f"scripts/dump_fistcheck.py)", flush=True)
    new_world = list(world_xforms)  # unaffected joints: unchanged from rest
    for i in range(len(joint_order)):
        if i in affected:
            p = parent_of(i)
            parent_world = new_world[p] if p is not None else Gf.Matrix4d(1.0)
            new_world[i] = local_xforms[i] * parent_world  # USD: row-vector, child = local * parent

    # Read-only evaluation landmarks in skeleton space. The skeleton itself
    # stays in bind pose after baking, so its queried joints would be stale.
    skeleton.GetPrim().CreateAttribute(
        "phyrc:posedJointPositions", Sdf.ValueTypeNames.Point3fArray, custom=True
    ).Set(Vt.Vec3fArray([Gf.Vec3f(*m.ExtractTranslation()) for m in new_world]))

    skinning_xforms = Vt.Matrix4dArray(
        [world_xforms[i].GetInverse() * new_world[i] for i in range(len(joint_order))]
    )

    # Same reasoning as the Skeleton above: take the first skinned Mesh in the
    # subtree instead of assuming it is called Body_Mesh (female1_cp.usd calls
    # its own SMPLX_shapes_female_005).
    mesh_prim = None
    for _p in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies()):
        if _p.IsA(UsdGeom.Mesh):
            mesh_prim = _p
            break
    if not mesh_prim or not mesh_prim.IsValid():
        raise RuntimeError(f"no skinned mesh under {skel_root_path}")
    mesh = UsdGeom.PointBased(mesh_prim)
    mesh_binding = UsdSkel.BindingAPI(mesh_prim)
    geom_bind_xform = mesh_binding.GetGeomBindTransformAttr().Get()
    if geom_bind_xform is None:
        geom_bind_xform = Gf.Matrix4d(1.0)
    joint_indices_pv = mesh_binding.GetJointIndicesPrimvar()
    joint_weights_pv = mesh_binding.GetJointWeightsPrimvar()
    num_influences = joint_indices_pv.GetElementSize()
    rest_points = mesh.GetPointsAttr().Get()

    # If the mesh authors its own (sub/reordered) joint list, the indices
    # in jointIndices are relative to THAT list, not skeleton.GetJointsAttr()
    # directly -- remap skinning_xforms into that order if present.
    mesh_joints = mesh_binding.GetJointsAttr().Get()
    if mesh_joints:
        skel_idx_by_token = {str(t): i for i, t in enumerate(joint_order)}
        skinning_xforms = Vt.Matrix4dArray(
            [skinning_xforms[skel_idx_by_token[str(t)]] for t in mesh_joints]
        )

    new_points = Vt.Vec3fArray(rest_points)  # SkinPointsLBS modifies this array in place
    ok = UsdSkel.SkinPointsLBS(
        geom_bind_xform,
        skinning_xforms,
        joint_indices_pv.Get(),
        joint_weights_pv.Get(),
        num_influences,
        new_points,
        True,  # inSerial
    )
    if not ok:
        raise RuntimeError("UsdSkel.SkinPointsLBS failed")
    mesh.GetPointsAttr().Set(new_points)
    new_extent = UsdGeom.PointBased.ComputeExtent(new_points)
    mesh.CreateExtentAttr().Set(new_extent)
    # The dark blotchy patches on the arm after posing (initially mistaken
    # for a mesh TEAR) turned out to be a lighting artifact, not a real
    # gap -- this mesh's normals are authored with faceVarying
    # interpolation, and UsdSkel.SkinPointsLBS only ever touches POINTS,
    # never normals. Left un-updated, those normals kept pointing in
    # their original T-pose directions on now-rotated geometry, so
    # lighting saw badly mismatched surface orientations at the
    # shoulder/arm and rendered them dark. UsdSkel.SkinNormalsLBS exists
    # but only handles vertex/varying interpolation (its own docstring
    # says so) -- the faceVarying counterpart isn't exposed to Python.
    # Blocking the stale normals outright, rather than hand-rolling that
    # skinning math, lets the renderer fall back to auto-computed smooth
    # normals from the NEW (already-correct) point positions instead.
    mesh.GetNormalsAttr().Block()
    # [isaac-5.1.0 compat: refresh extent]
    # Re-author the extent to match the points just baked. USD does not
    # recompute it, so the asset keeps advertising its ORIGINAL bind-pose
    # bounds -- female1_cp.usd then reported a 1.78m arm span and a body
    # sunk 0.38m under the floor while its actual vertices measured a 0.30m
    # shoulder width standing at z=0.016. Anything reading bounds rather than
    # points is misled by that, including viewport culling and this project's
    # own upright check.
    # Same idiom this file already uses for the reshaped garment: compute the
    # bounds straight from the points rather than going through the plugin
    # dispatch, which returned nothing here and left the stale value in place.
    mesh.CreateExtentAttr().Set(
        UsdGeom.PointBased.ComputeExtent(mesh.GetPointsAttr().Get()))



class TeleopTShirtStretch4_Env(BaseEnv):
    def __init__(self):
        super().__init__()
        self.ground = spawn_textured_ground(
            self.stage,
            self.scene,
            "/World/StretchFloor",
            FLOOR_SIZE,
            FLOOR_TEXTURE,
            FLOOR_TEXTURE_REPEAT,
        )
        # [isaac-5.1.0 compat: four shirts]
        # One table per shirt instead of one bench under all of them, so the
        # gaps between them are real floor a robot can stand on. Each is a
        # separate FixedCuboid at its garment's own x offset; BOX_SIZE is now
        # ONE table's size, and BOX_GAP up top is what separates them.
        self.tables = []
        for _ti, _tx in enumerate(GARMENT_X_OFFSETS):
            self.tables.append(self.scene.add(
                FixedCuboid(
                    prim_path=f"/World/garment_table_{_ti}",
                    name=f"garment_table_{_ti}",
                    color=np.array([0.55, 0.55, 0.55]),
                    position=np.array([BOX_POS[0] + _tx, BOX_POS[1], BOX_POS[2]]),
                    scale=BOX_SIZE,
                    size=1.0,
                    visible=True,
                )
            ))
        # Nothing else in this file reads it, but the probes do, and a single
        # name for "the table" is meaningless now -- point it at the first.
        self.table = self.tables[0] if self.tables else None
        # Four separate Particle_Garment instances (each finds its own
        # unique prim path via find_unique_string_name internally, so no
        # path collisions) rather than one -- Particle_Garment has no
        # per-instance color param, so each needs apply_visual_material +
        # set_color called on it individually anyway.
        self.garments = []
        from Env_Config.Garment.RandomSpawn import sample_garment_spawn
        self.garment_table_centers = np.array([
            [BOX_POS[0] + x, BOX_POS[1], BOX_POS[2]] for x in GARMENT_X_OFFSETS])
        self.garment_spawn = sample_garment_spawn(self.garment_table_centers, BOX_SIZE)
        print(f"[Teleop] garment spawn: table={self.garment_spawn['table_index'] + 1}/"
              f"{len(self.tables)}, seed={self.garment_spawn['seed']}", flush=True)
        for color_name, color in GARMENT_COLORS.items():
            # Keep the single blue asset; only its support table changes.
            if color_name != "blue":
                continue
            _usd = GARMENT_USD_BY_COLOR.get(color_name, TSHIRT_USD)
            _gs = GARMENT_SCALE_BY_COLOR.get(color_name, 1.0)
            _gscale = GARMENT_SCALE * _gs
            _gori = np.array(GARMENT_ORI, dtype=float)
            _gori[2] += GARMENT_YAW_BY_COLOR.get(color_name, 0.0)
            # Offsets ride the scale so the cloth keeps the offset-to-spacing
            # ratio it was tuned at rather than inheriting margins now too wide.
            _co, _ro = GARMENT_CONTACT_OFFSET * _gs, GARMENT_REST_OFFSET * _gs
            _pco = GARMENT_PARTICLE_CONTACT_OFFSET * _gs
            _sro = GARMENT_SOLID_REST_OFFSET * _gs
            if _usd != TSHIRT_USD or _gs != 1.0:
                print(f"[Teleop] {color_name} shirt: scale {_gs} yaw "
                      f"+{GARMENT_YAW_BY_COLOR.get(color_name, 0.0)} "
                      f"offsets x{_gs} (particle {_pco:.4f} solid {_sro:.4f})",
                      flush=True)
            pos = np.array(self.garment_spawn['spawn_position_world_m'])
            g = Particle_Garment(
                self.world,
                pos=pos,
                ori=_gori,
                scale=_gscale,
                usd_path=_usd,
                visual_material_usd="Assets/Material/Garment/linen_Pumpkin.usd",
                # Zero cloth self-friction; native vertex attachments hold the grip.
                # Human min/0 stays slippery; finger max/0.2 adds contact friction.
                # STRETCH4_SURFACE_FRICTION remains the surface-material override.
                friction=_feel("GARMENT_FRICTION", 0.0),
                # Particle_Garment defaults damping/drag/lift to 0.0 and the
                # environment never passed any of them, so nothing took energy
                # out of the cloth: once moving it kept moving, which reads as
                # the garment drifting in a breeze. There is no breeze -- drag
                # and lift are the aerodynamic terms and both are zero.
                # Velocity damping is the fix that does not change how heavy
                # the garment is; raising GARMENT_GRAVITY_SCALE off its 0.6
                # would, and that number is low on purpose, because at full
                # weight the garment overwhelms the grab.
                # Measured with scripts/probe_float.py.
                damping=_feel("GARMENT_DAMPING", 0.0),
                drag=_feel("GARMENT_DRAG", 0.0),
                lift=_feel("GARMENT_LIFT", 0.0),
                contact_offset=_co,
                rest_offset=_ro,
                particle_contact_offset=_pco,
                fluid_rest_offset=_ro,
                solid_rest_offset=_sro,
                bend_stiffness=GARMENT_BEND_STIFFNESS,
                shear_stiffness=GARMENT_SHEAR_STIFFNESS,
                gravity_scale=GARMENT_GRAVITY_SCALE,
                particle_mass=GARMENT_PARTICLE_MASS,
                stretch_stiffness=GARMENT_STRETCH_STIFFNESS,
                solver_position_iteration_count=GARMENT_SOLVER_ITERATIONS,
            )
            # linen_Pumpkin.usd's UsdPreviewSurface shader has its
            # diffuseColor input CONNECTED to a texture (BaseColor.jpg),
            # not just holding a plain value -- set_color() only sets a
            # fallback value on that same input, which a live connection
            # takes priority over in rendering, so nothing visibly
            # changed (all 4 stayed the base material's orange). Same
            # "there's a separate override/gate, not just a value"
            # pattern as the OmniPBR opacity lesson from earlier this
            # session. Disconnect the texture first so the flat value
            # actually takes effect.
            shader = g.visual_material.shaders_list[0]
            shader.GetInput("diffuseColor").DisconnectSource()
            g.visual_material.set_color(color)
            # Red band along the hem of any garment whose asset carries a
            # 'hem' GeomSubset. A shirt in one flat colour gives no clue which
            # end is which once it is lying crumpled on a table; a stripe does.
            # Bound to the subset rather than painted into the mesh, so the
            # garment's own material still covers everything else.
            _hem = self.stage.GetPrimAtPath(g.garment_mesh_prim_path + "/hem")
            if _hem.IsValid():
                _mpath = f"/World/Looks/HemStripe_{color_name}"
                _mat = UsdShade.Material.Define(self.stage, _mpath)
                _sh = UsdShade.Shader.Define(self.stage, _mpath + "/Shader")
                _sh.CreateIdAttr("UsdPreviewSurface")
                # [isaac-5.1.0 compat: four shirts]
                _hemcol = GARMENT_HEM_BY_COLOR.get(color_name, GARMENT_HEM_COLOR)
                _sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                    Gf.Vec3f(*_hemcol))
                _sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
                _mat.CreateSurfaceOutput().ConnectToSource(
                    _sh.ConnectableAPI(), "surface")
                UsdShade.MaterialBindingAPI.Apply(_hem)
                UsdShade.MaterialBindingAPI(_hem).Bind(_mat)
                print(f"[Teleop] {color_name} shirt: hem stripe {_hemcol} bound",
                      flush=True)
                _vneck = self.stage.GetPrimAtPath(g.garment_mesh_prim_path + "/vneck")
                if _vneck.IsValid():
                    UsdShade.MaterialBindingAPI.Apply(_vneck)
                    UsdShade.MaterialBindingAPI(_vneck).Bind(_mat)
                    print(f"[Teleop] {color_name} shirt: vneck stripe {_hemcol} bound",
                          flush=True)
                # [isaac-5.1.0 compat: folded hem]
                # Fold the red band up over the front of the shirt, the way
                # a sheet of paper folds: rotate it 180 degrees about the
                # crease at its own top edge, so it lands back on the shirt
                # pointing the other way. Only the FRONT layer moves -- the
                # garment is a closed shell lying flat, so its two layers are
                # the two sides of the thickness axis, and the front one is
                # simply the half with the larger coordinate along it.
                #
                # Done on the mesh points before the particle system reads
                # them, so the cloth starts folded rather than being pushed
                # into a fold by something.
                # Off: the shirt lies flat. The fold works -- the band sits
                # on top of the front face, measured -- but it reads as an
                # odd crumple rather than a hem turned back, so the scene
                # starts unfolded. STRETCH4_HEM_FOLD=1 brings it back.
                # OFF. The mechanism works and is measured (see
                # HANDOFF.md), but every variant tried so far reads as a
                # crumple rather than a hem turned back, so the scene
                # starts flat. STRETCH4_HEM_FOLD=1 turns it on.
                _fold = _feel("HEM_FOLD", 0.0)
                if _fold and _hem.IsValid():
                    _gm = UsdGeom.Mesh(self.stage.GetPrimAtPath(
                        g.garment_mesh_prim_path))
                    _pts = np.asarray(_gm.GetPointsAttr().Get(), dtype=float)
                    _cnt = np.asarray(_gm.GetFaceVertexCountsAttr().Get())
                    _idx = np.asarray(_gm.GetFaceVertexIndicesAttr().Get())
                    _starts = np.concatenate([[0], np.cumsum(_cnt)[:-1]])
                    _hf = np.asarray(UsdGeom.Subset(_hem).GetIndicesAttr().Get())
                    _hv = np.unique(np.concatenate(
                        [_idx[_starts[f]:_starts[f] + _cnt[f]] for f in _hf]))
                    # Axes: the republished shirt is laid flat with its axes
                    # ordered by extent, so 0 is across, 1 is neck-to-hem and
                    # 2 is thickness (scripts/make_wearable_shirt.py).
                    _thick, _long = 2, 1
                    # WHICH layer is the top one is a world question, not a
                    # local one: STRETCH4_GARMENT_ORI turns the garment 180
                    # degrees, so the local +thickness side ends up facing the
                    # floor and picking it folded the underside.
                    _w = UsdGeom.Xformable(
                        self.stage.GetPrimAtPath(g.garment_mesh_prim_path)
                    ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    _axis_w = _w.TransformDir(Gf.Vec3d(0.0, 0.0, 1.0))
                    _up = 1.0 if _axis_w[2] >= 0.0 else -1.0
                    # BOTH faces of the hem turn back, not just the one
                    # facing up. The garment is a closed shell lying flat,
                    # so its two layers are the two sides of the thickness
                    # axis, and folding only the upper one left the lower
                    # hem hanging straight -- half a fold. Each layer
                    # hinges on its own crease and lands back on its own
                    # face, which is what a hem does when a flattened tube
                    # is turned back on itself: the flap you see from above
                    # sits on the top face, the one underneath on the
                    # bottom face.
                    _s = _pts[:, _thick] * _up   # bigger = higher in world
                    _med = float(np.median(_s[_hv]))
                    _layers = ((_hv[_s[_hv] > _med], 1.0),
                               (_hv[_s[_hv] <= _med], -1.0))
                    _mid = float(np.median(_pts[:, _long]))
                    _outw = _feel("HEM_FOLD_OUT", 0.0) >= 0.5
                    _lift = _feel("HEM_FOLD_LIFT", 0.012)
                    # Degrees. 180 is the old behaviour -- the band mirrored
                    # right over onto its own face, which is a hem folded
                    # FLAT and reads as a crumple (HANDOFF.md section 5,
                    # three rejected attempts). Anything less swings the
                    # band up about its inner edge instead and leaves it
                    # standing, which is what \"jabgi joke\" asks for: a
                    # raised lip at the shirt's open end that a gripper can
                    # close on, rather than fabric lying flat on a table the
                    # fingers cannot get under.
                    _angle = float(_feel("HEM_FOLD_ANGLE", 180.0))
                    _new = _pts.copy()
                    _done = []
                    for _sel, _side in _layers:
                        if len(_sel) <= 8:
                            continue
                        _fb = _pts[_sel]
                        # Which edge the fold hinges on decides which way
                        # the red goes: the INNER edge (nearer the middle)
                        # turns the band back over the body, the OUTER one
                        # throws it past the hem where nothing supports it
                        # and it droops off the table -- measured, 9% of
                        # the band left on top. HEM_FOLD_OUT=1 for that.
                        _near_max = (abs(_fb[:, _long].max() - _mid)
                                     < abs(_fb[:, _long].min() - _mid))
                        print(f"[Teleop] hem band layer: {len(_sel)} verts, "
                              f"local extents across={np.ptp(_fb[:, 0]):.4f} "
                              f"long={np.ptp(_fb[:, 1]):.4f} "
                              f"thick={np.ptp(_fb[:, 2]):.4f}, "
                              f"mid={_mid:.4f} long-range="
                              f"[{_fb[:, _long].min():.4f},{_fb[:, _long].max():.4f}]",
                              flush=True)
                        if _angle < 179.0 and _side < 0:
                            # Only the layer that faces UP is turned back.
                            # The source mesh is a modelled 3-D shirt, not a
                            # flat sheet -- the band alone spans 0.139 of
                            # local thickness against 0.089 of width -- so
                            # \"flatten the lower layer onto its own face\"
                            # is a 0.3 m move in world terms, and it tore
                            # the band clean off the body. The lower hem
                            # stays where it is; the shirt keeps its
                            # outline and only the visible face turns back.
                            continue
                        if _angle < 179.0:
                            # Partial fold. The hinge is always the INNER
                            # edge here -- swinging the free edge up about
                            # the edge nearer the body is what turns the
                            # band outward; hinging on the outer edge would
                            # fold it back over the shirt, which is the
                            # inward look that was rejected. Every vertex
                            # rises in proportion to its distance from the
                            # crease, which is a rotation rather than the
                            # constant nudge that left the band curved and
                            # half of it hidden between the two layers.
                            _crease = (float(_fb[:, _long].max()) if _near_max
                                       else float(_fb[:, _long].min()))
                            _d = _fb[:, _long] - _crease
                            _th = np.radians(_angle)
                            _new[_sel, _long] = _crease + _d * np.cos(_th)
                            # Both layers swing up together, so the open end
                            # of the flattened tube flares rather than one
                            # face folding onto the other: the lip is two
                            # layers thick, which is a better pinch target
                            # than one.
                            _new[_sel, _thick] = (_s[_sel]
                                                  + np.abs(_d) * np.sin(_th)) * _up
                            _done.append(len(_sel))
                            continue
                        _use_max = _near_max if not _outw else not _near_max
                        _crease = (float(_fb[:, _long].max()) if _use_max
                                   else float(_fb[:, _long].min()))
                        _new[_sel, _long] = 2.0 * _crease - _fb[:, _long]
                        # Lay the whole flap flat just clear of the face it
                        # lands on, on the outside of it. Nudging each
                        # vertex by a constant instead kept the band's own
                        # curvature and half of it settled BETWEEN the two
                        # layers, out of sight.
                        _face = (float(_s[_sel].max()) if _side > 0
                                 else float(_s[_sel].min()))
                        _new[_sel, _thick] = (_face + _side * _lift) * _up
                        _done.append(len(_sel))
                    if _done:
                        _gm.GetPointsAttr().Set(Vt.Vec3fArray(
                            [Gf.Vec3f(*map(float, q)) for q in _new]))
                        _gm.CreateExtentAttr().Set(
                            UsdGeom.PointBased.ComputeExtent(
                                _gm.GetPointsAttr().Get()))
                        print(f"[Teleop] {color_name} shirt: hem turned back "
                              f"{_angle:.0f} degrees on {len(_done)} face(s), "
                              f"{sum(_done)} of {len(_hv)} band vertices",
                              flush=True)
                    else:
                        print(f"[Teleop] WARNING: {color_name} shirt hem band "
                              f"has no layer to fold ({len(_hv)} "
                              f"vertices)", flush=True)
            self.garments.append(g)
        # Replaces the pothook with a human mannequin (rigid collider
        # only, no skeletal simulation -- matches "the human can just
        # be treated as rigid"). Human.py's own default orientation=[90,0,0]
        # already corrects the raw Y-up asset to this Z-up scene; pose
        # the arms AFTER that reference is live on the stage but the
        # T-pose rewrite itself happens on the raw skeleton, in the
        # asset's own pre-correction space (see pose_arms_at_attention's
        # own comments on that mapping). Currently posed at attention (
        # arms down) per request -- was arms-forward before.
        # STRETCH4_HUMAN=mesh swaps the stylised biped for female1_cp.usd, the
        # SMPL-X female body the user supplied -- a real statistical human model
        # rather than a figure whose head reads too small.
        #
        # The asset is referenced exactly as shipped. An earlier attempt rewrote
        # its vertices to swing the arms forward, which flattened them: rotating
        # points about a shoulder pivot is not skinning, and at 90 degrees it
        # collapsed the arms to zero thickness. The project already has the right
        # machinery -- pose_arms_at_attention bakes a pose through
        # UsdSkel.SkinPointsLBS -- so the arms are posed through that instead, with
        # only the target direction changed.
        #
        # Direction (0, 0, 1) is the asset's OWN raw +Z. The replacement USD's
        # internal SkelRoot maps raw (x,y,z) -> scene (x,-z,y), so raw +Z lands
        # on scene -Y: horizontal, pointing from the human toward the robots.
        # Its face also looks along raw +Z, so it reaches the way it is looking
        # with no extra yaw. Being axis-aligned also avoids shoulder distortion.
        #
        # The scale is UNIFORM, and must be: non-uniform scaling is what made the
        # old mannequin's head and posed shoulders read wrong. The replacement
        # asset is authored directly in metres.
        if os.environ.get("STRETCH4_HUMAN", "mesh") == "mesh" and os.path.exists(MESH_HUMAN_USD):
            # 0.0063 gave a measured head height of 1.0289m. 10cm shorter is 0.9289m,
            # so the scale drops by 0.9289/1.0289 = 0.9028 to 0.005688. UNIFORM,
            # because the proportions have to hold -- which also means the arms
            # shorten by the same ~10%, not by 10cm; taking 10cm off an arm this
            # size would need a separate, proportion-breaking edit.
            # Preserve the previous mannequin's measured 1.599 m standing
            # height.  The replacement is authored in metres and is 1.685798 m
            # tall in its raw Y-up point coordinates, hence 1.599/1.685798.
            _hs = float(os.environ.get("HUMAN_MESH_SCALE", "0.948514"))
            self.human = Human(
                MESH_HUMAN_USD,
                position=np.array([_feel("HUMAN_POS_X", float(BOX_POS[0])),
                                   _feel("HUMAN_POS_Y", 0.45), HUMAN_POS[2]]),
                # female2 already carries the Y-up -> Z-up rotation on its
                # internal SkelRoot; Human's historical +90 X default would
                # apply the correction twice.
                orientation=np.array([0.0, 0.0, 0.0]),
                scale=np.array([_hs, _hs, _hs]))
            # Arms lifted HUMAN_ARM_ELEVATION_DEG above horizontal. Directions
            # are in the asset's raw frame. female2's internal SkelRoot maps raw
            # (x,y,z) -> scene (x,-z,y): raw +Z is horizontal toward the robots
            # and raw +Y is straight up, so (0, sin t, cos t) sits exactly t
            # above the horizontal plane. The collision body is rebuilt from the
            # posed mesh further down, so it follows without extra work.
            _elev = np.radians(float(os.environ.get("HUMAN_ARM_ELEVATION_DEG", "20")))
            pose_arms_at_attention(
                self.stage, self.human.prim_path,
                arm_dir=(0.0, float(np.sin(_elev)), float(np.cos(_elev))),
                # [isaac-5.1.0 compat: four shirts]
                # 20 -> 0 -> 10 -> 0 -> 10, per "I spread the arm angle slightly
                # sideways; make them parallel, straight forward" (parallel, 0) then
                # "spread the arms slightly like before" (back to the 10-degree-out
                # try, i.e. "before" = the 10 this same knob held earlier, not
                # the file's original 20). The hand collider (mitten/sphere,
                # whichever STRETCH4_HUMAN_HAND_COLLIDER is set to) is
                # rebuilt from the posed mesh, so it follows.
                #
                # Arms spread 10 degrees per side from the raised 20-degree direction.
                # Body and wrist spheres are rebuilt from this posed mesh.
                arm_spread_deg=float(os.environ.get(
                    "HUMAN_ARM_SPREAD_DEG", "10")))
            # The person is seated (STRETCH4_SIT=0 stands them back up).
            # The collision body is rebuilt from these points further down,
            # so whatever is baked here is what the cloth actually meets.
            _sit = os.environ.get("STRETCH4_SIT", "1") == "1"
            if _sit:
                _nj = pose_legs_seated(
                    self.stage, self.human.prim_path,
                    _feel("SIT_HIP_FRACTION", 1.0),
                    _feel("SIT_KNEE_FRACTION", 1.0))
                print(f"[Teleop] seated pose baked over {_nj} joints",
                      flush=True)
            # Folding the legs leaves the pelvis where it was and swings the feet
            # up, so the figure would sit in mid-air; drop it back onto the floor.
            # Measured from the POINTS, not from a BBoxCache. UsdSkel keeps
            # reporting this prim's bind-pose bounds after a skinning bake, and
            # trusting them put the chair's seat at z=1.17m.
            def _body_world_z():
                _hp = self.stage.GetPrimAtPath(self.human.prim_path)
                for _pr in Usd.PrimRange(_hp, Usd.TraverseInstanceProxies()):
                    if _pr.IsA(UsdGeom.Mesh) and "Collision" not in _pr.GetName():
                        _m = UsdGeom.Mesh(_pr)
                        _x = UsdGeom.Xformable(_pr).ComputeLocalToWorldTransform(
                            Usd.TimeCode.Default())
                        _w = np.array([_x.Transform(Gf.Vec3d(*_q))
                                       for _q in _m.GetPointsAttr().Get()])
                        return float(_w[:, 2].min()), float(_w[:, 2].max())
                return 0.0, 1.0
            _lo_z, _hi_z = _body_world_z()
            if abs(_lo_z) > 1e-4:
                self.human.rigid_form.set_world_pose(
                    position=np.array([_feel("HUMAN_POS_X", float(BOX_POS[0])), _feel("HUMAN_POS_Y", 0.45),
                                       HUMAN_POS[2] - _lo_z]))
                _lo_z, _hi_z = _body_world_z()
            _seat_z = _lo_z + 0.52 * (_hi_z - _lo_z)
            _seat_xy, _seat_wd = None, None
            if _sit:
                # Seat height is not a fraction of a seated figure's height.
                # Measure it: the buttocks are the lowest thing at the BACK of
                # a seated body (the feet are the lowest thing at the front),
                # so take the rear of the figure and read its underside, then
                # size the seat from the strip of body actually resting on it.
                _hp = self.stage.GetPrimAtPath(self.human.prim_path)
                for _pr in Usd.PrimRange(_hp, Usd.TraverseInstanceProxies()):
                    if _pr.IsA(UsdGeom.Mesh) and "Collision" not in _pr.GetName():
                        _x = UsdGeom.Xformable(_pr).ComputeLocalToWorldTransform(
                            Usd.TimeCode.Default())
                        _w = np.array([_x.Transform(Gf.Vec3d(*_q)) for _q in
                                       UsdGeom.Mesh(_pr).GetPointsAttr().Get()])
                        _rear = _w[_w[:, 1] >= np.percentile(_w[:, 1], 60)]
                        if len(_rear):
                            _seat_z = float(np.percentile(_rear[:, 2], 1))
                            _rest = _w[(_w[:, 2] > _seat_z - 0.01)
                                       & (_w[:, 2] < _seat_z + 0.05)]
                            if len(_rest) > 20:
                                # The seat has to STOP BEHIND THE KNEE, or the
                                # shins hang in front of a seat that reaches
                                # past them. At seat height the front-most body
                                # points are the knees, so the front edge is
                                # set back from them by SEAT_KNEE_CLEARANCE and
                                # the knee, shin and foot are all clear of it.
                                # The robots are at low y, so forward is -y.
                                _front = (float(_rest[:, 1].min())
                                          + _feel("SEAT_KNEE_CLEARANCE", 0.06))
                                _rear = float(_rest[:, 1].max()) + 0.02
                                _seat_xy = (float(np.median(_rest[:, 0])),
                                            0.5 * (_front + _rear))
                                _seat_wd = (
                                    float(np.ptp(_rest[:, 0])) * 1.5 + 0.04,
                                    max(_rear - _front, 0.10))
                        break
                print(f"[Teleop] seated: seat z={_seat_z:.3f} at {_seat_xy}, "
                      f"size {_seat_wd}, body z {_lo_z:.3f}..{_hi_z:.3f}",
                      flush=True)
            # A chair, built from boxes: nothing on this machine ships one. It is
            # only ever looked at -- collision is never applied, so the robots pass
            # through it instead of fighting it.
            # Off by default. The chair itself is fine -- boxes, no collision, robots
            # drive through it -- but the figure is still STANDING, because the
            # seated pose could not be made to work, and a standing body
            # intersecting a chair looks worse than no chair. Set
            # STRETCH4_CHAIR=1 to place it anyway.
            if os.environ.get("STRETCH4_CHAIR", "0") == "1" or _sit:
                _sw, _sd, _th = 0.42, 0.42, 0.04
                _cx, _cy = float(_feel("HUMAN_POS_X", float(BOX_POS[0]))), _feel("HUMAN_POS_Y", 0.45)
                if _seat_xy is not None:
                    _cx, _cy = _seat_xy
                if _seat_wd is not None:
                    _sw, _sd = _seat_wd
                # [isaac-5.1.0 compat: stool]
                # Round top, no back. The seat is a cylinder rather than
                # a box, and the legs sit on a circle under its rim.
                _r = min(_sw, _sd) / 2.0
                _top = UsdGeom.Cylinder.Define(self.stage, "/World/Chair/Seat")
                _top.CreateRadiusAttr(float(_r))
                _top.CreateHeightAttr(float(_th))
                _top.CreateAxisAttr("Z")
                _top.CreateExtentAttr(Vt.Vec3fArray([
                    Gf.Vec3f(-_r, -_r, -_th / 2.0), Gf.Vec3f(_r, _r, _th / 2.0)]))
                UsdGeom.Xformable(_top).AddTranslateOp().Set(
                    Gf.Vec3d(_cx, _cy, _seat_z))
                _top.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.35, 0.24, 0.16)]))
                _parts = []
                for _i in range(4):
                    _a = np.radians(45.0 + 90.0 * _i)
                    _parts.append((f"Leg_{_i}",
                                   (_cx + np.cos(_a) * _r * 0.78,
                                    _cy + np.sin(_a) * _r * 0.78, _seat_z / 2.0),
                                   (0.035, 0.035, max(_seat_z, 0.02))))
                for _nm, _pp, _sz in _parts:
                    _cube = UsdGeom.Cube.Define(self.stage, f"/World/Chair/{_nm}")
                    _cube.CreateSizeAttr(1.0)
                    _xf = UsdGeom.Xformable(_cube)
                    _xf.AddTranslateOp().Set(Gf.Vec3d(*_pp))
                    _xf.AddScaleOp().Set(Gf.Vec3f(*_sz))
                    _cube.CreateDisplayColorAttr(
                        Vt.Vec3fArray([Gf.Vec3f(0.35, 0.24, 0.16)]))
                print(f"[Teleop] chair built, seat z={_seat_z:.3f} (no collision)",
                      flush=True)
            # Re-apply the collision approximation AFTER the pose bake. Human()
            # sets it during construction, i.e. while the mesh is still in its
            # T-pose bind, and the bake then rewrites the very points the
            # approximation is derived from. Rather than depend on when PhysX
            # happens to cook the shape, ask for it again here so the collider is
            # unambiguously built from the arms-forward geometry that is on
            # screen. Collision then matches what you see -- and note it comes
            # from the POINTS, so the stale UsdSkel bounding box plays no part in
            # it either.
            # HUMAN_MITTEN_INFLATE is in METRES, but the mesh it is applied to
            # is in the asset's own units, where the whole body is ~166 units
            # tall. Passing the metre value straight through meant 0.020 units
            # -- about a tenth of a millimetre once scaled -- so the mitten was
            # doing nothing at all, which is why fingers kept coming through
            # even after the collider itself was switched on. Convert.
            #
            # Trim the visual skull before deriving its collision surface.
            from Env_Config.Human.RoundHead import round_human_head
            round_human_head(self.stage, self.human.prim_path)
            # Fit the complete posed hand, not just the cut forearm boundary.
            # Legacy wrist spheres remain an explicit diagnostic option.
            _mode = os.environ.get("STRETCH4_HUMAN_HAND_COLLIDER", "fitted")
            if _mode not in ("none", "mitten", "sphere", "wrist_sphere", "fitted"):
                raise ValueError(f"unknown human hand collider mode: {_mode}")
            _inflate = _feel("HUMAN_MITTEN_INFLATE", 0.012)
            if _mode == "none":
                # No mitten: the visible skin is the collider again, fists
                # and all. The mitten exists because separate fingers slip
                # between cloth particles; a closed fist has none. Human()
                # leaves collision enabled, but say so explicitly rather
                # than rely on nothing having turned it off.
                _n, _cpath = 0, None
                for _pr in Usd.PrimRange(
                        self.stage.GetPrimAtPath(self.human.prim_path),
                        Usd.TraverseInstanceProxies()):
                    if _pr.IsA(UsdGeom.Mesh) and _pr.GetName() != "CollisionBody":
                        UsdPhysics.CollisionAPI.Apply(_pr)\
                            .CreateCollisionEnabledAttr().Set(True)
                        break
            else:
                _n, _cpath = build_mitten_collider(
                    self.stage, self.human.prim_path,
                    (_inflate / max(_hs, 1e-9)) if _mode in ("mitten", "wrist_sphere") else 0.0,
                    _hs, drop_hands=(_mode in ("sphere", "fitted")))
            print(f"[Teleop] hand collider mode={_mode}: {_n} hand vertices, "
                  f"body collider at {_cpath}; visible hands keep their "
                  f"fingers and no longer collide", flush=True)
            if _mode == "fitted":
                from Env_Config.Human.HandSphereColliders import build_fitted_hand_colliders
                build_fitted_hand_colliders(
                    self.stage, self.human.prim_path,
                    _hand_vertex_sets(self.stage, self.human.prim_path, _hs), _feel)
            elif _mode in ("sphere", "wrist_sphere"):
                build_hand_spheres(
                    self.stage, self.human.prim_path, _hs,
                    _feel("GARMENT_PARTICLE_CONTACT_OFFSET", 0.012) * 1.26,
                    _feel("HUMAN_HAND_SPHERE_R", 0.025))
            # The collider is hidden, which is exactly why nobody could tell it
            # apart from a decorative mesh for so long. STRETCH4_SHOW_COLLIDER=1
            # draws it instead of the body, so what physics uses is what is on
            # screen. Visual only -- collision is unchanged either way.
            if os.environ.get("STRETCH4_SHOW_COLLIDER", "0") == "1":
                for _cp in (_cpath, self.human.prim_path + "/HandSphere_left",
                            self.human.prim_path + "/HandSphere_right",
                            self.human.prim_path + "/HandHull_left",
                            self.human.prim_path + "/HandHull_right"):
                    _pr = self.stage.GetPrimAtPath(_cp) if _cp else None
                    if _pr and _pr.IsValid():
                        UsdGeom.Imageable(_pr).MakeVisible()
                for _pr in Usd.PrimRange(
                        self.stage.GetPrimAtPath(self.human.prim_path),
                        Usd.TraverseInstanceProxies()):
                    if (_pr.IsA(UsdGeom.Mesh) and _pr.GetName() != "CollisionBody"
                            and not _pr.GetName().startswith("HandHull_")):
                        UsdGeom.Imageable(_pr).MakeInvisible()
                        break
                print("[Teleop] STRETCH4_SHOW_COLLIDER: drawing the collision "
                      "body instead of the visible one", flush=True)
            # CollisionBody owns the physical representation. Authoring a mesh
            # approximation on the non-geometric human root has no effect on
            # that hidden collider. The original triangles remain available
            # for the independent surface-contact guard and visual checks.
            print(f"[Teleop] human: female2_c4-c5 SMPL-X body, uniform scale {_hs}, "
                  f"arms forward (raw +Z)", flush=True)
        else:
            self.human = Human(HUMAN_USD, position=HUMAN_POS, scale=HUMAN_SCALE)
            pose_arms_at_attention(self.stage, self.human.prim_path)
        # [isaac-6.0.1: slippery cloth contact]
        from Env_Config.Human.HumanContactMaterial import configure_human_contact
        configure_human_contact(self.stage, self.human.prim_path)
        # Per "render the mannequin in a single color, for example beige" --
        # override on OUR OWN live prim (under /World/Human), not an edit
        # to the shared source asset. This is an MDL-style shader (id was
        # None, inputs like diffuse_texture/metallic_texture -- not a
        # UsdPreviewSurface): diffuse_texture, when set, takes priority
        # over diffuse_color_constant (same "texture beats a plain value
        # on the same input" pattern as the T-shirt's linen material
        # earlier this session), so the texture has to be cleared, not
        # just left in place alongside a new color. metallic_texture/
        # reflectionroughness_texture are cleared too and pinned to flat
        # constants -- otherwise the old texture's per-pixel shininess
        # would still show through as blotchy highlights under a now-flat
        # color, undermining the "solid color" look. diffuse_color_
        # constant isn't pre-authored on this particular shader prim
        # (confirmed missing from a GetInputs() dump when the logo was
        # being tracked down) so it needs CreateInput, not GetInput.
        suit_shader_prim = self.stage.GetPrimAtPath(f"{self.human.prim_path}/Looks/suit/Shader")
        if suit_shader_prim.IsValid():
            suit_shader = UsdShade.Shader(suit_shader_prim)
            suit_shader.GetInput("diffuse_texture").Set(Sdf.AssetPath(""))
            suit_shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(*HUMAN_BEIGE_COLOR))
            suit_shader.GetInput("metallic_texture").Set(Sdf.AssetPath(""))
            suit_shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(0.0)
            suit_shader.GetInput("reflectionroughness_texture").Set(Sdf.AssetPath(""))
            suit_shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(0.7)
        else:
            if os.environ.get("STRETCH4_HUMAN", "mesh") != "mesh":
                print(f"[Teleop] WARNING: {self.human.prim_path}/Looks/suit/Shader not found, color not overridden")
        # Per "can you minimize the mannequin's collision margin? right now the cloth
        # doesn't slip into the gaps" -- Human.py turns on collision
        # (SingleGeometryPrim(collision=True)) but never touches its
        # contact/rest offset, so it was sitting at whatever PhysX's own
        # class default is. That's a SEPARATE margin from the garment's
        # own GARMENT_CONTACT_OFFSET/REST_OFFSET -- contact between the
        # two effectively has to clear BOTH shells combined, so a
        # generous garment margin (0.045/0.036, kept large on purpose
        # for puffiness) stacked on top of an untouched human margin
        # made the real usable gap even tighter than the garment number
        # alone suggests. Pushed the human's own margin down as far as
        # it'll go (contact_offset must stay > rest_offset > 0) so the
        # body's PHYSICS shape reads thinner than what's actually
        # rendered, leaving more of the garment's own margin as
        # available slack for the gripper.
        # Pushed even lower per "minimize the mannequin's collision margin" --
        # contact_offset must stay > rest_offset > 0, so these are close
        # to the practical floor.
        # [isaac-5.1.0 compat: human skin margin]
        # A 0.5mm collision shell on a body the cloth samples every 16mm.
        #
        # Those offsets were driven to the floor back when the complaint was
        # that the gripper could not get into the garment's gap, and against
        # the old mannequin they were fine. On a real body they are not: the
        # shell is thirty times finer than the cloth's own particle spacing, so
        # a thin feature slips between particles instead of pushing them --
        # the fingers surfaced through an intact part of the shirt first, then
        # the whole forearm followed.
        #
        # Sized against the garment's particles rather than against nothing:
        # GARMENT_SOLID_REST_OFFSET is 0.006, so a comparable shell on the body
        # gives the solver something to resolve against. Still far thinner than
        # the 0.06 the coarse mesh used to run with, so the body does not read
        # as inflated.
        # Back to the minimum. Raising these to 0.006/0.004 was an attempt at the
        # finger problem before the mitten collider existed, and it wrapped the
        # WHOLE body in a 6mm invisible shell -- the garment then stopped short of
        # the skin everywhere, which is the reported "the T-shirt makes contact
        # before it touches the body". The mitten handles the fingers directly, so the body can go back
        # to a physical surface that matches the rendered one.
        # Set offsets on the posed collider which actually generates contacts,
        # not only on the human root/disabled visual mesh.
        from pxr import PhysxSchema as _ContactSchema
        for _shape in Usd.PrimRange(self.stage.GetPrimAtPath(self.human.prim_path),
                                   Usd.TraverseInstanceProxies()):
            if _shape.HasAPI(UsdPhysics.CollisionAPI) and (
                    UsdPhysics.CollisionAPI(_shape).GetCollisionEnabledAttr().Get() is not False):
                _contact = _ContactSchema.PhysxCollisionAPI.Apply(_shape)
                _contact.CreateContactOffsetAttr().Set(_feel("HUMAN_CONTACT_OFFSET", 0.006))
                _contact.CreateRestOffsetAttr().Set(_feel("HUMAN_REST_OFFSET", 0.001))

        self.human_spawn = randomize_human_and_chair(self.stage, self.human.prim_path)

        SimulationManager.set_physics_sim_device("cuda:0")
        SimulationManager.set_backend("torch")

        self.robot_prim = spawn_stretch4(self.stage, "/World/Stretch4", ROBOT1_SPAWN, ROBOT1_YAW_DEG)
        self.robot2_prim = spawn_stretch4(self.stage, "/World/Stretch4_2", ROBOT2_SPAWN, ROBOT2_YAW_DEG)
        apply_stretch4_colors(self.stage, "/World/Stretch4")
        apply_stretch4_colors(self.stage, "/World/Stretch4_2")
        # # [isaac-5.1.0 compat: robot-robot filter]
        # Stop the two robots colliding with each other, per "can you make the robots
        # not collide with each other?" -- the wheels in particular, which is where the user saw
        # them knock together.
        #
        # Each robot carries 28 enabled colliders, wheel_0/1/2_link among them, and
        # the scene had no collision groups at all, so nothing was filtering the
        # robot-vs-robot pairs. (Worth recording why that was missed earlier in this
        # project: a plain Usd.Stage.Traverse() reports ZERO colliders on these
        # robots, because it stops at the instanceable prim boundary and every
        # collider lives inside the prototype. Usd.TraverseInstanceProxies() is what
        # shows them.)
        #
        # Two collision groups, each filtering the other. This suppresses only
        # robot-against-robot contacts: each robot still collides with the floor,
        # the table and the garments exactly as before, and nothing about the
        # gripper is touched.
        if os.environ.get("STRETCH4_ROBOT_SELF_COLLIDE", "0") != "1":
            _groups = self.stage.DefinePrim("/World/CollisionGroups", "Scope")
            for _name, _robot, _other in (("robot1", "/World/Stretch4", "robot2"),
                                          ("robot2", "/World/Stretch4_2", "robot1")):
                _g = UsdPhysics.CollisionGroup.Define(
                    self.stage, f"/World/CollisionGroups/{_name}")
                _g.GetCollidersCollectionAPI().CreateIncludesRel().AddTarget(_robot)
                _g.CreateFilteredGroupsRel().AddTarget(
                    f"/World/CollisionGroups/{_other}")
            print("[Teleop] robot-vs-robot collisions filtered "
                  "(STRETCH4_ROBOT_SELF_COLLIDE=1 to restore them)", flush=True)
        simulation_app.update()

        # Zero baseline, then selected human/gripper contact overrides.
        from Env_Config.Garment.ZeroSceneFriction import zero_scene_friction
        zero_scene_friction(self.stage)

        self.reset()

        self.rig = configure_stretch4(self.stage, "/World/Stretch4")
        self.rig2 = configure_stretch4(self.stage, "/World/Stretch4_2")


# Per "I want to tweak the Stretch design ... using that XML design
# file ... as reference, can you color the robot?" -- ported from
# Teleop_TShirt_Stretch4_Hand_Env.py, which already did this earlier in
# the same session. Stretch4's actual color scheme, straight from Hello
# Robot's own MuJoCo source
# (~/wd/stretch4_mujoco/stretch4_mujoco/models/stretch_4/
# autogenerated_stretch_4_model.xml), read directly off each link's own
# <geom class="visualgeom" ... rgba="..."/> line (and omniwheels.xml for
# the wheels) -- not guessed. The USD import already has SOME material
# bound per mesh, but those don't match the MJCF at all (confirmed via
# probe in the other file, e.g. wrist_yaw_link comes in bound to a pale
# blue-gray placeholder, not the MJCF's near-black 0.15/0.15/0.15).
# mast_link has no rgba in the MJCF at all, left at the same mid-gray as
# the arm segments as a neutral fallback.
STRETCH4_LINK_COLORS = {
    "base_link": (0.9, 0.9, 0.88),
    "head_link": (0.9, 0.9, 0.88),
    "mast_link": (0.5, 0.5, 0.5),
    "lift_link": (0.9, 0.9, 0.88),
    "arm_l0_link": (0.9, 0.9, 0.88),
    "arm_l1_link": (0.5, 0.5, 0.5),
    "arm_l2_link": (0.5, 0.5, 0.5),
    "arm_l3_link": (0.5, 0.5, 0.5),
    "arm_l4_link": (0.5, 0.5, 0.5),
    "wrist_link": (0.15, 0.15, 0.15),
    "wrist_yaw_link": (0.15, 0.15, 0.15),
    "wrist_pitch_link": (0.15, 0.15, 0.15),
    "wrist_roll_link": (0.15, 0.15, 0.15),
    "gripper_camera_link": (0.15, 0.15, 0.15),
    "tool_attachment_site_link": (0.15, 0.15, 0.15),
    "quick_connect_interface_link": (0.15, 0.15, 0.15),
    "grasp_center_link": (0.15, 0.15, 0.15),
    "gripper_finger_right_link": (0.9, 0.9, 0.88),
    "gripper_fingertip_right_link": (0.15, 0.15, 0.15),
    "aruco_fingertip_right_link": (1.0, 1.0, 1.0),
    "gripper_finger_left_link": (0.9, 0.9, 0.88),
    "gripper_fingertip_left_link": (0.15, 0.15, 0.15),
    "aruco_fingertip_left_link": (1.0, 1.0, 1.0),
    "wrist_aruco_link": (1.0, 1.0, 1.0),
    "wrist_reflector_link": (0.15, 0.15, 0.15),
    "wheel_0_link": (0.15, 0.15, 0.15),
    "wheel_1_link": (0.15, 0.15, 0.15),
    "wheel_2_link": (0.15, 0.15, 0.15),
}


def apply_stretch4_colors(stage, prim_path):
    for link_name, rgb in STRETCH4_LINK_COLORS.items():
        visuals_prim = stage.GetPrimAtPath(f"{prim_path}/{link_name}/visuals")
        if not visuals_prim.IsValid():
            print(f"[Teleop] WARNING: {prim_path}/{link_name}/visuals not found, skipping color")
            continue
        # This "visuals" prim is itself scene-graph-instanceable (this
        # file also sets useSceneGraphInstancing=True at the top, same
        # as the other file where this was confirmed via probe) -- its
        # real children only show up through instance-proxy-aware
        # traversal, and instance proxies can't have new relationships
        # (like a material binding) authored on them directly. Locally
        # opt this ONE prim out of instancing (a stage-level override,
        # doesn't touch the source USD) so its subtree becomes normal,
        # individually-editable prims.
        visuals_prim.SetInstanceable(False)
        # The mesh lives one level further in, under a sub-prim whose own
        # name doesn't always match the link name (e.g. arm_l0_link's
        # visual mesh is named "arm_l4_link" -- a numbering quirk from
        # the original URDF/MJCF conversion) -- just take whichever mesh
        # is actually there rather than assume the name.
        mesh_prim = None
        bind_prim = None
        for child in visuals_prim.GetChildren():
            candidate = stage.GetPrimAtPath(f"{child.GetPath()}/mesh")
            if candidate.IsValid():
                mesh_prim = candidate
                # The imported asset already carries a "material:binding" on
                # this parent ("child") prim with strength=strongerThanDescendants
                # -- a binding authored on the mesh itself would be a weaker
                # descendant opinion and get silently overridden by it. Bind
                # on this same prim instead so our relationship replaces the
                # existing one outright.
                bind_prim = child
                break
        if mesh_prim is None:
            print(f"[Teleop] WARNING: no mesh found under {prim_path}/{link_name}/visuals, skipping color")
            continue

        # Dedicated material per link rather than editing the imported
        # shared ones in place -- several links currently share the same
        # bound material despite needing DIFFERENT final colors per the
        # MJCF, so recoloring the shared material in place would bleed
        # across links that happen to currently share it.
        mat_path = f"{prim_path}/Looks/hello_robot_{link_name}"
        material = UsdShade.Material.Define(stage, mat_path)
        shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        # strongerThanDescendants: some links' mesh sub-prim ALSO carries
        # its own pre-existing direct binding (to an unrelated fallback
        # material) -- without this, that descendant-level opinion would
        # win over the one authored here on its parent.
        UsdShade.MaterialBindingAPI(bind_prim).Bind(
            material, bindingStrength=UsdShade.Tokens.strongerThanDescendants
        )


def spawn_textured_ground(stage, scene, prim_path, size, texture_path, texture_repeat, z_position=0.0):
    # GroundPlane only honors its own `size` arg when `prim_path` does NOT
    # already exist -- if it did, we'd hit the same dead branch Real_Ground
    # hits with default_environment.usd. find_unique_string_name guarantees
    # a fresh path so that "create new prim" branch is the one that fires.
    prim_path = find_unique_string_name(prim_path, is_unique_fn=lambda p: not is_prim_path_valid(p))
    # NOTE: this must be built as a totally separate prim path, NOT nested
    # under prim_path -- constructing it as a child (e.g. f"{prim_path}/
    # physics_material") defines that child prim as a side effect of this
    # argument being evaluated BEFORE GroundPlane() runs, which makes
    # prim_path itself implicitly valid (a defined child makes its parent
    # a valid "over" prim in USD) by the time GroundPlane's own internal
    # is_prim_path_valid(prim_path) check runs -- it then wrongly takes
    # the "already exists" branch and silently drops the `size` arg.
    physics_material_path = find_unique_string_name(
        "/World/Physics_Materials/stretch_floor_material",
        is_unique_fn=lambda p: not is_prim_path_valid(p),
    )
    ground = GroundPlane(
        prim_path=prim_path,
        name="stretch_floor",
        z_position=z_position,
        size=size,
        physics_material=PhysicsMaterial(
            prim_path=physics_material_path,
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.0,
        ),
    )
    # NOTE: deliberately NOT scene.add(ground) here -- Real_Ground doesn't
    # register its own plane with the scene either. Registering it makes
    # world.reset() call GroundPlane.post_reset(), which re-applies a
    # default_state pose captured via whatever prim backend (numpy/torch)
    # was active at construction time; by reset time SimulationManager's
    # physics view has since flipped the backend, and post_reset's world-
    # pose write chokes on the resulting numpy/torch tensor mismatch. The
    # ground plane is static and doesn't need scene-managed reset anyway.

    # PhysicsSchemaTools.addGroundPlane builds the /geom quad as a bare
    # Mesh with only points/faceVertexIndices/displayColor -- no "st"
    # primvar at all (confirmed via a headless probe: its only primvars
    # were displayColor/displayOpacity). Without UVs, the primvar reader
    # below has nothing to read and the texture sample collapses to a
    # single point for the whole plane, rendering as one flat solid color
    # instead of tiled wood grain. addGroundPlane's own 4 corner points
    # are always (-size,-size),(size,-size),(size,size),(-size,size) in
    # that winding order (confirmed via the same probe) -- author matching
    # unit-square corner UVs directly rather than computing them generically.
    geom_prim = stage.GetPrimAtPath(f"{prim_path}/geom")
    st_primvar = UsdGeom.PrimvarsAPI(geom_prim).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    )
    st_primvar.Set(Vt.Vec2fArray([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)]))

    # PreviewSurface (isaacsim's own material wrapper) only exposes flat
    # color/roughness/metallic -- no texture support -- so a textured look
    # needs a hand-built UsdShade graph, same pattern used for the robot
    # link colors above (Material + UsdPreviewSurface shader + UsdUVTexture
    # shader + a primvar reader to feed it UVs).
    mat_path = f"{prim_path}/Looks/oak_floorboard"
    material = UsdShade.Material.Define(stage, mat_path)

    uv_reader = UsdShade.Shader.Define(stage, f"{mat_path}/uvReader")
    uv_reader.CreateIdAttr("UsdPrimvarReader_float2")
    uv_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")

    texture = UsdShade.Shader.Define(stage, f"{mat_path}/diffuseTexture")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture_path)
    texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    # Repeats the texture across the plane rather than stretching one
    # image over the whole floor -- matches MuJoCo's own texrepeat="4 4".
    st_transform = UsdShade.Shader.Define(stage, f"{mat_path}/stTransform")
    st_transform.CreateIdAttr("UsdTransform2d")
    st_transform.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(uv_reader.ConnectableAPI(), "result")
    st_transform.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(texture_repeat, texture_repeat))
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_transform.ConnectableAPI(), "result")

    shader = UsdShade.Shader.Define(stage, f"{mat_path}/PBRShader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), "rgb")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    # GroundPlane's own construction already bound a default flat-gray
    # PreviewSurface via apply_visual_material -- NOT on {prim_path}/geom
    # but on prim_path itself (the root Xform), with weaker_than_descendants
    # =False (i.e. strongerThanDescendants). An ancestor binding declared
    # strongerThanDescendants beats ANY descendant's own binding regardless
    # of that descendant's own declared strength, so rebinding only on
    # /geom (a descendant of prim_path) would silently lose to it -- rebind
    # at that SAME root prim_path instead, replacing its relationship
    # targets outright.
    root_prim = stage.GetPrimAtPath(prim_path)
    UsdShade.MaterialBindingAPI(root_prim).Bind(material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)

    return ground


def spawn_stretch4(stage, prim_path, spawn_pos, yaw_deg):
    robot_prim = stage.DefinePrim(prim_path, "Xform")
    robot_prim.GetReferences().AddReference(STRETCH4_USD)
    rxf = UsdGeom.Xformable(robot_prim)
    rxf.ClearXformOpOrder()
    rxf.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*spawn_pos))
    rxf.AddRotateZOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(yaw_deg)
    from Env_Config.Utils_Project.ContactTuning import tune_robot_contacts
    tune_robot_contacts(stage, prim_path)
    # [render: separate collision shapes are physics-only]
    from Env_Config.Utils_Project.RobotCollisionVisibility import hide_robot_collision_visuals
    hide_robot_collision_visuals(stage, prim_path)
    return robot_prim


def configure_stretch4(stage, prim_path):
    robot = SingleArticulation(prim_path=f"{prim_path}/base_footprint", name=prim_path)
    robot.initialize()
    robot.set_solver_position_iteration_count(32)
    # PhysX 107.3 warns on this articulation: "more than 4 velocity iterations
    # being added to a TGS scene. The related behavior changed recently." The
    # count buys nothing measurable here -- sweeping it (and the position count,
    # and the particle solver) moved the loop rate by under 1% -- so drop it to
    # the value the new solver actually wants.
    robot.set_solver_velocity_iteration_count(
        int(os.environ.get("STRETCH4_VEL_ITERS", "4")))
    robot.set_sleep_threshold(0.0)

    ctrl = robot.get_articulation_controller()
    kps, kds = ctrl.get_gains()
    kps, kds = _to_np(kps), _to_np(kds)
    kds = np.array([max(kd, kp * 0.2) for kp, kd in zip(kps, kds)])
    ctrl.set_gains(kps=_to_t(kps), kds=_to_t(kds))

    dof = {n: robot.get_dof_index(n) for n in robot.dof_names}

    _wheel_dof_idx = np.array([dof["wheel_0_joint"], dof["wheel_1_joint"], dof["wheel_2_joint"]])
    kps[_wheel_dof_idx] = 0.0
    kds[_wheel_dof_idx] = 2.0
    ctrl.set_gains(kps=_to_t(kps), kds=_to_t(kds))

    arm_joint_idx = np.array([dof["arm_l1_joint"], dof["arm_l2_joint"],
                               dof["arm_l3_joint"], dof["arm_l4_joint"]])
    lift_idx, yaw_idx, pitch_idx, roll_idx = (
        dof["lift_joint"], dof["wrist_yaw_joint"], dof["wrist_pitch_joint"], dof["wrist_roll_joint"])
    grip_idx = np.array([dof["gripper_finger_left_joint"], dof["gripper_finger_right_joint"]])

    _stiff_idx = np.concatenate([[lift_idx], arm_joint_idx])
    kps[_stiff_idx] = _feel("LIFT_ARM_KP", 6000.0)
    kds[_stiff_idx] = _feel("LIFT_ARM_KD", 1200.0)
    ctrl.set_gains(kps=_to_t(kps), kds=_to_t(kds))

    # Armature -- virtual rotor inertia added to the joint itself. Without it
    # the telescoping arm's own link inertia fights the position drive whenever
    # the base accelerates, and the arm visibly extends/retracts on its own
    # (measured: 2cm of travel from a single base start). It also damps the
    # residual ringing when a motion stops. Cheap, and standard practice for
    # exactly this failure mode.
    _armature = _feel("ARMATURE", 0.1)
    if _armature > 0.0:
        from pxr import PhysxSchema as _PhysxSchema
        for _joint_name in ("lift_joint", "arm_l1_joint", "arm_l2_joint",
                            "arm_l3_joint", "arm_l4_joint"):
            _joint_prim = stage.GetPrimAtPath(f"{prim_path}/{_joint_name}")
            if not _joint_prim:
                for _p in stage.Traverse():
                    if _p.GetName() == _joint_name and \
                            _p.GetPath().pathString.startswith(prim_path):
                        _joint_prim = _p
                        break
            if _joint_prim:
                _api = _PhysxSchema.PhysxJointAPI.Apply(_joint_prim)
                _api.CreateArmatureAttr().Set(_armature)

    kps[grip_idx] = 8000.0
    kds[grip_idx] = 1600.0
    ctrl.set_gains(kps=_to_t(kps), kds=_to_t(kds))
    # ArticulationController wraps this in another list. Passing an ndarray
    # creates ``[ndarray]`` and triggers PyTorch's extremely-slow conversion
    # path; a plain list becomes one contiguous tensor directly.
    ctrl.set_max_efforts([1000.0, 1000.0], joint_indices=_to_idx(grip_idx))

    limits = _to_np(ctrl.get_joint_limits())
    lo, hi = limits[:, 0], limits[:, 1]

    return {
        "robot": robot, "ctrl": ctrl,
        "grasp_link_idx": robot._articulation_view.get_link_index("grasp_center_link"),
        "fingertip_left_idx": robot._articulation_view.get_link_index("gripper_fingertip_left_link"),
        "fingertip_right_idx": robot._articulation_view.get_link_index("gripper_fingertip_right_link"),
        "lift_idx": lift_idx, "arm_joint_idx": arm_joint_idx,
        "yaw_idx": yaw_idx, "pitch_idx": pitch_idx, "roll_idx": roll_idx,
        "grip_idx": grip_idx,
        "lift_lo": float(lo[lift_idx]), "lift_hi": float(hi[lift_idx]),
        "arm_lo": float(lo[arm_joint_idx[0]]) * 4, "arm_hi": float(hi[arm_joint_idx[0]]) * 4,
        "yaw_lo": float(lo[yaw_idx]), "yaw_hi": float(hi[yaw_idx]),
        "pitch_lo": float(lo[pitch_idx]), "pitch_hi": float(hi[pitch_idx]),
        "roll_lo": float(lo[roll_idx]), "roll_hi": float(hi[roll_idx]),
        "state": {
            "lift": 0.0, "arm": 0.0, "yaw": 0.0, "pitch": 0.0, "roll": 0.0,
            "grip_pos": GRIPPER_OPEN,
            "base_fwd": 0.0, "base_strafe": 0.0, "base_turn": 0.0,
            "heading": None, "grabbed": None,
            # Separate from "grabbed" on purpose -- this drives the visual
            # finger open/close animation, "grabbed" drives whether a
            # garment is actually attached. Closing the gripper with
            # nothing in reach should still visibly close the fingers,
            # it just won't attach anything.
            "gripper_closed": False,
        },
        "held": set(),
    }


def garment_boundary_vertex_indices(mesh_prim):
    """Vertex indices on the mesh's topological boundary (collar/cuffs/
    hem -- any edge that belongs to only one face). Ported over from
    Teleop_TShirt_Stretch4_Hand_Env.py. Computed directly from face
    connectivity rather than assumed, since it needs to work for
    whichever garment mesh is currently loaded."""
    mesh = UsdGeom.Mesh(mesh_prim)
    counts = mesh.GetFaceVertexCountsAttr().Get()
    idxs = mesh.GetFaceVertexIndicesAttr().Get()
    edge_count = {}
    offset = 0
    for c in counts:
        face = idxs[offset:offset + c]
        offset += c
        for i in range(c):
            a, b = int(face[i]), int(face[(i + 1) % c])
            key = (a, b) if a < b else (b, a)
            edge_count[key] = edge_count.get(key, 0) + 1
    boundary = set()
    for (a, b), n in edge_count.items():
        if n == 1:
            boundary.add(a)
            boundary.add(b)
    return sorted(boundary)


def apply_edge_mass_boost(gc, mesh_prim, factor):
    """Multiplies the particle mass of the garment's boundary-loop
    vertices (see garment_boundary_vertex_indices) by `factor`, leaving
    interior particles untouched -- the closest API-supported analog to
    "make the hem/collar/cuffs feel rigid" found. PhysX particle cloth
    DOES support per-spring stiffness arrays, but the auto-generation
    path used here (bend_stiffness/shear_stiffness scalars) doesn't
    expose which spring index corresponds to which particle pair, so
    targeting "boundary springs only" isn't reachable through this API.
    Heavier boundary particles resist being pushed by both small
    disturbances and the gripper's own contact force (F=ma) instead.

    Bypasses ClothPrim.set_particle_masses() directly -- that method is
    broken in this isaacsim version (it calls self.get_masses(), which
    doesn't exist on ClothPrim; the correctly-implemented method is
    named get_particle_masses()). Reimplements what it should do: read
    current masses, edit them, and push straight to the physics tensor
    view, the same call set_particle_masses() itself makes internally.
    """
    if not getattr(gc, "supports_particle_masses", True):
        print("[Teleop] edge mass boost skipped: Isaac Sim 6 surface deformables use body mass")
        return
    boundary_idx = garment_boundary_vertex_indices(mesh_prim)
    masses = gc.get_particle_masses(clone=False)
    masses[0, boundary_idx] *= factor
    indices = gc._backend_utils.resolve_indices(None, gc.count, device=gc._device)
    gc._physics_view.set_masses(masses, indices)


def detect_blowup(garment_cloths, robots):
    """Returns (description, is_hard) the moment it sees the early
    signature of a collision-explosion (garment particle or robot base
    moving unrealistically fast, or any position/velocity gone non-
    finite) -- see EXPLOSION_VELOCITY_THRESHOLD's own comment. Returns
    (None, False) if everything still looks normal.

    is_hard=True means actual NaN/Inf was found -- the data is already
    corrupted, so there's nothing to gain by waiting for a streak of
    confirmations the way the speed-threshold case does; the caller
    should roll back immediately. is_hard=False is the earlier, still-
    finite "unusually fast" warning, kept behind the streak requirement
    since a one-off forceful frame (a hard grab-drag, e.g.) can trip it
    without being a real blowup."""
    for i, gc in enumerate(garment_cloths):
        vel = gc.get_velocities()
        varr = vel.cpu().numpy() if hasattr(vel, "cpu") else np.asarray(vel)
        if not np.all(np.isfinite(varr)):
            return f"garment {i}: non-finite velocity", True
        speed = np.linalg.norm(varr[0], axis=-1)
        if speed.max() > EXPLOSION_VELOCITY_THRESHOLD:
            return f"garment {i}: particle speed {speed.max():.2f} m/s exceeds {EXPLOSION_VELOCITY_THRESHOLD}m/s", False
        pos = gc.get_world_positions()
        parr = pos.cpu().numpy() if hasattr(pos, "cpu") else np.asarray(pos)
        if not np.all(np.isfinite(parr)):
            return f"garment {i}: non-finite position", True
    for i, robot in enumerate(robots):
        lin_vel = _to_np(robot.get_linear_velocity())
        if not np.all(np.isfinite(lin_vel)):
            return f"robot {i}: non-finite velocity", True
        if np.linalg.norm(lin_vel) > EXPLOSION_VELOCITY_THRESHOLD:
            return f"robot {i}: base speed {np.linalg.norm(lin_vel):.2f} m/s exceeds {EXPLOSION_VELOCITY_THRESHOLD}m/s", False
    return None, False


# [isaac-5.1.0 compat: face labels]
def garment_face_labels(mesh_prim, points):
    """Label every particle as belonging to the garment's front or back face.

    Distance cannot separate the two layers. Measured on the wearable shirt laid
    flat: at all 12 probe points across the garment, the near and far faces sat
    closer together than the mesh's own particle spacing (~0.011m), so a
    distance-gap test found no boundary anywhere and quietly kept both layers.

    Topology can separate them, and does so whether the garment is flat, folded
    or draped. In the garment's authored 3D shape the front and back surfaces
    face opposite ways, so the sign of each vertex's normal along the garment's
    thinnest axis is a stable front/back label. Computed once from the rest pose
    and then carried by index, it survives any amount of subsequent deformation.

    Returns an int array of +1 / -1 per particle.
    """
    import numpy as _np

    counts = _np.asarray(mesh_prim.GetAttribute("faceVertexCounts").Get())
    idxs = _np.asarray(mesh_prim.GetAttribute("faceVertexIndices").Get())
    pts = _np.asarray(points, dtype=float)

    normals = _np.zeros_like(pts)
    off = 0
    for c in counts:
        face = idxs[off:off + c]
        off += c
        if c < 3:
            continue
        a, b, d = pts[face[0]], pts[face[1]], pts[face[2]]
        n = _np.cross(b - a, d - a)
        for v in face:
            normals[v] += n

    lengths = _np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / _np.maximum(lengths, 1e-12)

    thin_axis = int(_np.argmin(pts.max(axis=0) - pts.min(axis=0)))
    side = _np.sign(normals[:, thin_axis])
    side[side == 0] = 1.0
    return side.astype(int)


def stretch4_tuning(rig, restore=False):
    """Snapshot the articulation tuning, and put it back after a reset.

    # [isaac-5.1.0 compat: retune after reset]

    configure_stretch4 runs exactly once, at construction. world.reset() then
    rebuilds the PhysX articulation from what is authored in USD, which drops
    every runtime property set on top of it -- the position/velocity solver
    iteration counts, the sleep threshold, the gripper's max efforts, and above
    all the gains: wheels at 0/2, lift and arm at LIFT_ARM_KP/KD, gripper at
    8000/1600. Those gains ARE the handling. Pressing P therefore left the robot
    running on the asset's stock stiffness, which is the "after pressing P and
    restarting, the controls feel off" the user hit.

    Deliberately a snapshot-and-replay rather than a second copy of the tuning
    values: configure_stretch4 stays the single place those numbers live, so this
    cannot drift away from it. Armature is not included -- it is authored onto
    the joint prims as a USD attribute and survives the reset on its own.
    """
    ctrl = rig["ctrl"]
    robot = rig["robot"]
    if not restore:
        kps, kds = ctrl.get_gains()
        rig["_tuning"] = (_to_np(kps).copy(), _to_np(kds).copy())
        return
    saved = rig.get("_tuning")
    if saved is None:
        return
    kps, kds = saved
    ctrl.set_gains(kps=_to_t(kps), kds=_to_t(kds))
    ctrl.set_max_efforts([1000.0, 1000.0],
                         joint_indices=_to_idx(rig["grip_idx"]))
    robot.set_solver_position_iteration_count(32)
    robot.set_solver_velocity_iteration_count(
        int(os.environ.get("STRETCH4_VEL_ITERS", "4")))
    robot.set_sleep_threshold(0.0)


def _hand_vertex_sets(stage, skel_root_path, scale):
    """(mesh_prim, points, counts, indices, {side: idx}, {(side, finger): idx}).

    Hands are located by SKINNING WEIGHT. Mesh extremes fail once the arms are
    raised -- the tallest axis becomes the body, and the "hands" come out as the
    head and the feet -- and the wrist joints fail too, because the skeleton stays
    in its bind pose after an LBS bake and points at where the hands used to be.
    Which joint drives a vertex is fixed at bind time and survives any pose.

    The trim below is belt and braces. It was added when the left hand's set
    reached 0.39m from the hand centre and that looked like a joint's influence
    bleeding up the forearm; the real cause was the index shift described below,
    and with that fixed the sets are local. It stays because anything built off
    this set -- a long axis, a bounding sphere -- is wrecked by a single distant
    vertex, and 0.13m is a hand's reach on this 0.83m figure either way.
    """
    from pxr import Gf as _Gf, Usd as _Usd, UsdGeom as _UsdGeom, UsdSkel as _UsdSkel
    import numpy as _np

    root_prim = stage.GetPrimAtPath(skel_root_path)
    mesh_prim = None
    for _p in _Usd.PrimRange(root_prim, _Usd.TraverseInstanceProxies()):
        if _p.IsA(_UsdGeom.Mesh) and "Collision" not in _p.GetName():
            mesh_prim = _p
            break
    if mesh_prim is None:
        return None, None, None, None, {}, {}
    src_mesh = _UsdGeom.Mesh(mesh_prim)
    pts = _np.asarray(src_mesh.GetPointsAttr().Get(), dtype=float)
    counts = _np.asarray(src_mesh.GetFaceVertexCountsAttr().Get())
    idxs = _np.asarray(src_mesh.GetFaceVertexIndicesAttr().Get())

    # CollisionBody and HandSphere_* are authored directly below
    # skel_root_path, while an imported visual mesh may live below transformed
    # Xforms. The old asset happened to have identity ancestors; the replacement
    # has an internal Y-up -> Z-up rotation. Convert the posed mesh points into
    # the human root's local space before deriving any collision geometry.
    mesh_to_world = _UsdGeom.Xformable(mesh_prim).ComputeLocalToWorldTransform(
        _Usd.TimeCode.Default())
    root_to_world = _UsdGeom.Xformable(root_prim).ComputeLocalToWorldTransform(
        _Usd.TimeCode.Default())
    world_to_root = root_to_world.GetInverse()
    pts = _np.asarray([
        world_to_root.Transform(mesh_to_world.Transform(_Gf.Vec3d(*map(float, q))))
        for q in pts
    ], dtype=float)

    skel = None
    for _p in _Usd.PrimRange(root_prim, _Usd.TraverseInstanceProxies()):
        if _p.IsA(_UsdSkel.Skeleton):
            skel = _UsdSkel.Skeleton(_p)
            break
    ji = mesh_prim.GetAttribute("primvars:skel:jointIndices").Get()
    jw = mesh_prim.GetAttribute("primvars:skel:jointWeights").Get()
    if skel is None or not ji or not jw:
        return mesh_prim, pts, counts, idxs, {}, {}
    # jointIndices index into the MESH's own joint list when it authors one, not
    # into the skeleton's. Here the two differ by exactly one entry -- the
    # skeleton carries an extra "SMPLX_female" at the top -- so reading these
    # indices against the skeleton's list shifts every joint by one: the set
    # asked for as "wrist plus fingers" came back as "fingers minus the palm,
    # plus a patch of the opposite collar" (127 vertices, 0.39m from the hand,
    # which looked exactly like skinning bleed up the forearm and is not).
    _mesh_joints = _UsdSkel.BindingAPI(mesh_prim).GetJointsAttr().Get()
    joints = [str(j) for j in (_mesh_joints if _mesh_joints
                               else skel.GetJointsAttr().Get())]
    k = max(1, len(ji) // len(pts))
    ji = _np.asarray(ji).reshape(len(pts), k)
    jw = _np.asarray(jw).reshape(len(pts), k)
    dominant = ji[_np.arange(len(pts)), _np.argmax(jw, axis=1)]

    fingers = ("index", "middle", "ring", "pinky", "thumb")
    reach = 0.13 / max(scale, 1e-9)          # local units, not metres
    hands, groups = {}, {}
    for side in ("left", "right"):
        jj = [i for i, t in enumerate(joints)
              if any(f"{side}_{x}" in t for x in ("wrist",) + fingers)]
        sel = _np.where(_np.isin(dominant, jj))[0]
        if not len(sel):
            continue
        med = _np.median(pts[sel], axis=0)
        sel = sel[_np.linalg.norm(pts[sel] - med, axis=1) <= reach]
        if len(sel):
            hands[side] = sel
        for f in fingers:
            fj = [i for i, t in enumerate(joints) if f"{side}_{f}" in t]
            g = _np.where(_np.isin(dominant, fj))[0]
            g = _np.intersect1d(g, sel)
            if len(g):
                groups[(side, f)] = g
    return mesh_prim, pts, counts, idxs, hands, groups


def build_hand_spheres(stage, skel_root_path, scale, spacing, radius_m=0.0):
    """Fit wrist spheres using posed hand points in the human root's frame."""
    # [isaac-6.0.1: wrist spheres]
    from Env_Config.Human.HandSphereColliders import build_hand_spheres as build
    return build(stage, skel_root_path,
                 _hand_vertex_sets(stage, skel_root_path, scale), _feel, radius_m)


def build_mitten_collider(stage, skel_root_path, inflate, scale, drop_hands=False):
    """Hidden collision copy of the body whose hands are mittens.

    # [isaac-5.1.0 compat: thicken fingers]

    Particle cloth collides against spheres on its own vertices, so a feature
    thinner than the gap between them slips through: the fingertip is 0.0047m
    against a 0.0031m gap, and it wedges that gap open until the arm follows.

    The first version of this function built the mesh, hid it, disabled collision
    on the visible body -- and never applied CollisionAPI to what it had built. It
    was decorative geometry, the body's own collider was the only one left, and
    the fingers went on coming through exactly as before. The collider is applied
    and its flag set explicitly here, and there is a probe that reads both back
    off the stage rather than trusting that the prim exists.

    Hands are located by SKINNING WEIGHT. Mesh extremes fail once the arms are
    raised -- the tallest axis becomes the body, and the "hands" come out as the
    head and the feet -- and the wrist joints fail too, because the skeleton stays
    in its bind pose after an LBS bake and points at where the hands used to be.
    Which joint drives a vertex is fixed at bind time and survives any pose.
    """
    from pxr import UsdGeom as _UsdGeom, UsdPhysics as _UsdPhysics
    from pxr import Vt as _Vt, Gf as _Gf
    import numpy as _np

    mesh_prim, pts, counts, idxs, hands, _g = _hand_vertex_sets(
        stage, skel_root_path, scale)
    if mesh_prim is None or not hands:
        return 0, None
    sel = _np.zeros(len(pts), dtype=bool)
    for _idx in hands.values():
        sel[_idx] = True

    # Vertex normals, area weighted.
    normals = _np.zeros_like(pts)
    off = 0
    for c in counts:
        f = idxs[off:off + c]; off += c
        if c < 3:
            continue
        a, b, d = pts[f[0]], pts[f[1]], pts[f[2]]
        n = _np.cross(b - a, d - a)
        for v in f:
            normals[v] += n
    normals /= _np.maximum(_np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)

    # Fill the gaps BETWEEN the fingers without making the hand longer: drop the
    # component of the displacement along each hand's own long axis. Per hand,
    # not over both at once -- the two hands together span the body, so their
    # combined long axis is the one that separates left from right, and
    # suppressing THAT would inflate the fingers lengthwise while leaving the
    # gaps between them exactly as they were.
    coll = pts.copy()
    # Keep the hidden collision surface a few millimetres outside the rendered
    # skin. Native contact can then fluctuate within its tolerance without the
    # beige body flashing through the cloth. This changes no visible geometry.
    _skin_m = _feel("HUMAN_COLLISION_SKIN", 0.003)
    if _skin_m > 0.0:
        # SMPL-X is consistently wound; still verify the global orientation so
        # an imported asset with reversed winding is expanded, not contracted.
        outward = _np.median(_np.sum(normals * (pts - _np.mean(pts, axis=0)), axis=1))
        if outward < 0.0:
            normals = -normals
        coll += normals * (_skin_m / max(scale, 1e-9))
    if inflate:
        for _side, _idx in hands.items():
            hp = pts[_idx]
            hand_axis = int(_np.argmax(hp.max(axis=0) - hp.min(axis=0)))
            dirs = normals[_idx].copy()
            dirs[:, hand_axis] = 0.0
            lens = _np.linalg.norm(dirs, axis=1, keepdims=True)
            dirs = _np.where(lens > 1e-9, dirs / _np.maximum(lens, 1e-12), 0.0)
            coll[_idx] += dirs * inflate

    # With a sphere taking the hands over, the hand triangles are dropped from
    # this mesh rather than left inside the sphere: a fingertip that pokes out of
    # it would be a thin feature again, and the whole point was to have none.
    if drop_hands:
        keep_counts, keep_idxs = [], []
        off = 0
        for c in counts:
            f = idxs[off:off + c]; off += c
            if not sel[f].any():
                keep_counts.append(int(c))
                keep_idxs.extend(int(v) for v in f)
        counts, idxs = _np.asarray(keep_counts), _np.asarray(keep_idxs)

    dst_path = skel_root_path + "/CollisionBody"
    dst = _UsdGeom.Mesh.Define(stage, dst_path)
    dst.CreatePointsAttr(_Vt.Vec3fArray([_Gf.Vec3f(*map(float, q)) for q in coll]))
    dst.CreateFaceVertexCountsAttr(_Vt.IntArray([int(c) for c in counts]))
    dst.CreateFaceVertexIndicesAttr(_Vt.IntArray([int(i) for i in idxs]))
    dst.CreateExtentAttr().Set(_UsdGeom.PointBased.ComputeExtent(dst.GetPointsAttr().Get()))
    _UsdGeom.Imageable(dst).MakeInvisible()

    # THE part that was missing before.
    api = _UsdPhysics.CollisionAPI.Apply(dst.GetPrim())
    api.CreateCollisionEnabledAttr().Set(True)
    mapi = _UsdPhysics.MeshCollisionAPI.Apply(dst.GetPrim())
    approximation = os.environ.get('HUMAN_COLLISION_APPROX', 'sdf')
    mapi.CreateApproximationAttr().Set(approximation)
    if approximation == 'sdf':
        from pxr import PhysxSchema as _PhysxSchema
        resolution = int(os.environ.get('STRETCH4_HUMAN_SDF_RESOLUTION', '256'))
        _PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(dst.GetPrim()).CreateSdfResolutionAttr().Set(resolution)
        print(f'[Teleop] {dst_path}: SDF resolution={resolution}; visible mesh unchanged', flush=True)

    # and the visible mesh, fingers and all, stops colliding
    vis = _UsdPhysics.CollisionAPI.Apply(mesh_prim)
    vis.CreateCollisionEnabledAttr().Set(False)
    return int(sel.sum()), dst_path


def pose_legs_seated(stage, skel_root_path, hip_frac=1.0, knee_frac=1.0):
    """Fold the hips and knees into a seated pose, baked into the mesh points.

    # [isaac-5.1.0 compat: seated pose]

    Runs AFTER pose_arms_at_attention and on top of its output, which is safe
    for one reason: that bake moved only the clavicle/head subtrees, so every
    leg vertex is still exactly where the bind pose put it, and the skinning
    transforms computed here (rest_world^-1 * new_world) are relative to that
    same bind pose. Vertices the arm bake DID move are weighted to joints this
    function leaves alone, whose skinning transform is the identity, so they
    are carried through untouched. Two sequential bakes compose correctly only
    while the two affected joint sets stay disjoint -- which is why this rotates
    hips and knees and nothing above the pelvis.

    geomBindTransform is passed as the identity here, not as the mesh's own: the
    arm bake already took the points into skeleton space, and applying the bind
    transform a second time would double it. (It is within 4e-8 of the identity
    on this asset anyway.)

    The previous attempt at a seated pose was removed because it "collapsed the
    figure". A 90 degree swing concentrated in one joint is the same linear-blend
    failure the arm pose hit at the shoulder, and the fix is the same in spirit:
    hip_frac/knee_frac keep the angles adjustable, and scripts/probe_sit.py
    measures the mesh volume before and after so a collapse is a number rather
    than an impression.
    """
    from pxr import Usd as _Usd, UsdGeom as _UsdGeom, UsdSkel as _UsdSkel
    from pxr import Gf as _Gf, Vt as _Vt

    root_prim = stage.GetPrimAtPath(skel_root_path)
    root_prim.SetTypeName("SkelRoot")
    skeleton = None
    for _p in _Usd.PrimRange(root_prim, _Usd.TraverseInstanceProxies()):
        if _p.IsA(_UsdSkel.Skeleton):
            skeleton = _UsdSkel.Skeleton(_p)
            break
    if skeleton is None:
        raise RuntimeError(f"no UsdSkel.Skeleton under {skel_root_path}")

    cache = _UsdSkel.Cache()
    skel_query = cache.GetSkelQuery(skeleton)
    order = [str(t) for t in skeleton.GetJointsAttr().Get()]
    local_xforms = list(skel_query.ComputeJointLocalTransforms(_Usd.TimeCode.Default()))
    rest_world = list(skel_query.ComputeJointSkelTransforms(_Usd.TimeCode.Default()))
    by_leaf = {t.rsplit("/", 1)[-1]: i for i, t in enumerate(order)}
    if not all(f"{s}_{j}" in by_leaf for s in ("left", "right")
               for j in ("hip", "knee", "ankle")):
        raise KeyError(f"no hip/knee/ankle chain in {order[:12]}")

    def parent_of(i):
        tok = order[i]
        if "/" not in tok:
            return None
        ptok = tok.rsplit("/", 1)[0]
        return order.index(ptok) if ptok in order else None

    # Everything at or below either hip moves; nothing else does.
    affected = set()
    for side in ("left", "right"):
        stem = order[by_leaf[f"{side}_hip"]]
        for i, tok in enumerate(order):
            if tok == stem or tok.startswith(stem + "/"):
                affected.add(i)

    new_world = list(rest_world)

    def refresh_fk():
        for i in range(len(order)):
            if i in affected:
                p = parent_of(i)
                parent_w = new_world[p] if p is not None else _Gf.Matrix4d(1.0)
                new_world[i] = local_xforms[i] * parent_w

    def set_world_rotation(i, world_rot):
        p = parent_of(i)
        parent_w = new_world[p] if p is not None else _Gf.Matrix4d(1.0)
        local_rot = world_rot * parent_w.ExtractRotation().GetInverse()
        old = local_xforms[i]
        t = _Gf.Transform()
        t.SetRotation(local_rot)
        t.SetTranslation(old.ExtractTranslation())
        t.SetScale(_Gf.Transform(old).GetScale())
        local_xforms[i] = t.GetMatrix()

    # The asset's own raw frame, the one pose_arms_at_attention documents:
    # raw +Y is up and raw +Z points forward, at the robots. Sitting is thighs
    # forward and shins down.
    FORWARD, DOWN = _Gf.Vec3d(0, 0, 1), _Gf.Vec3d(0, -1, 0)

    def swing(i_from, i_to, target, frac):
        cur = _Gf.Vec3d(new_world[i_to].ExtractTranslation()
                        - new_world[i_from].ExtractTranslation()).GetNormalized()
        delta = _Gf.Rotation(cur, target)
        if frac != 1.0:
            delta = _Gf.Rotation(delta.GetAxis(), delta.GetAngle() * frac)
        set_world_rotation(i_from, new_world[i_from].ExtractRotation() * delta)
        refresh_fk()

    for side in ("left", "right"):
        hi = by_leaf[f"{side}_hip"]
        ki = by_leaf[f"{side}_knee"]
        ai = by_leaf[f"{side}_ankle"]
        rest_foot_dir = None
        fi = by_leaf.get(f"{side}_foot")
        if fi is not None:
            rest_foot_dir = _Gf.Vec3d(rest_world[fi].ExtractTranslation()
                                      - rest_world[ai].ExtractTranslation()).GetNormalized()
        swing(hi, ki, FORWARD, hip_frac)          # thigh: down -> forward
        swing(ki, ai, DOWN, knee_frac)            # shin: forward -> down
        if fi is not None:
            # Put the foot back the way it sat in the rest pose, so the sole
            # ends up flat instead of pointing wherever the knee left it.
            swing(ai, fi, rest_foot_dir, 1.0)

    skinning = _Vt.Matrix4dArray(
        [rest_world[i].GetInverse() * new_world[i] for i in range(len(order))])

    mesh_prim = None
    for _p in _Usd.PrimRange(root_prim, _Usd.TraverseInstanceProxies()):
        if _p.IsA(_UsdGeom.Mesh) and "Collision" not in _p.GetName():
            mesh_prim = _p
            break
    if mesh_prim is None:
        raise RuntimeError(f"no skinned mesh under {skel_root_path}")
    mesh = _UsdGeom.PointBased(mesh_prim)
    binding = _UsdSkel.BindingAPI(mesh_prim)
    mesh_joints = binding.GetJointsAttr().Get()
    if mesh_joints:
        by_tok = {t: i for i, t in enumerate(order)}
        skinning = _Vt.Matrix4dArray([skinning[by_tok[str(t)]] for t in mesh_joints])

    points = _Vt.Vec3fArray(mesh.GetPointsAttr().Get())
    ok = _UsdSkel.SkinPointsLBS(
        _Gf.Matrix4d(1.0), skinning,
        binding.GetJointIndicesPrimvar().Get(),
        binding.GetJointWeightsPrimvar().Get(),
        binding.GetJointIndicesPrimvar().GetElementSize(),
        points, True)
    if not ok:
        raise RuntimeError("UsdSkel.SkinPointsLBS failed for the seated pose")
    mesh.GetPointsAttr().Set(points)
    mesh.GetNormalsAttr().Block()
    mesh.CreateExtentAttr().Set(_UsdGeom.PointBased.ComputeExtent(points))
    return len(affected)


# # [isaac-5.1.0 compat: hoisted teleop internals]
#
# _make_gripper_toggle and _drive_robot were nested inside main(), which meant
# the two pieces of behaviour most worth measuring -- how a grab is acquired,
# and how the held particles are dragged after it -- could only be reached by
# running the interactive session. Every probe that wanted to test them
# therefore reimplemented them (drag_stability.py and probe_pull.py both carry
# their own copy of the follow step), and a copy measures whatever the copy
# does, not what the F-keys and the grippers actually do.
#
# So they are lifted out verbatim. What was read from main()'s scope is passed
# in instead -- garment_cloths for the grab, and held / garment_cloths / dt /
# _frame for the drive -- and main() keeps two thin wrappers so every call site
# inside it is unchanged. The bodies are not edited by this patch at all; it
# cuts on indentation, so patches that edit INSIDE either function keep working
# and must simply run before it.
#
# scripts/probe_pairpull.py is the reason: two robots gripping one shirt and
# pulling it apart is a test of this exact code, and it now runs that code.


def make_gripper_toggle(rig, label, garment_cloths, garment_faces):
    state = rig["state"]

    def _attempt_grab():
        # Offsets (used later to make grabbed particles follow the
        # gripper) are still anchored to the grasp CENTER, since
        # that's what the follow logic in the main loop reads too --
        # only the eligibility test below changed.
        pos = _to_t(_grasp_link_pos(rig))

        # Real pincer check: a particle only counts as grabbable if
        # it's actually near the SEGMENT between the two fingertip
        # links (point-to-segment distance, clamped to the segment
        # so a particle off past either fingertip doesn't count),
        # within GRASP_TOLERANCE -- not just "somewhere within 0.2m
        # of a single center point" regardless of whether the fabric
        # is actually between the fingers.
        left_pos, right_pos = _fingertip_positions(rig)
        left_t, right_t = _to_t(left_pos), _to_t(right_pos)
        seg = right_t - left_t
        seg_len_sq = torch.dot(seg, seg).clamp(min=1e-8)

        best = None
        for ci, gc in enumerate(garment_cloths):
            all_pos = gc.get_world_positions()[0]
            to_particle = all_pos - left_t
            t = torch.clamp((to_particle @ seg) / seg_len_sq, 0.0, 1.0)
            closest = left_t + t.unsqueeze(-1) * seg
            dists = torch.linalg.norm(all_pos - closest, dim=1)
            idx = torch.where(dists < GRASP_TOLERANCE)[0]
            # # [isaac-5.1.0 compat: driven set]
            # How many particles are DRIVEN decides whether the lift is
            # stable, independently of whether the grab is accepted at all.
            # Measured on a 0.5 m/s lift of this garment:
            #      5 driven -> cloth overshoots the gripper (ratio 1.29) and
            #                  stretches to 0.92m, against a normal 0.68m
            #     10 driven -> ratio 1.03, no stretch
            #     20 driven -> ratio 1.01, no stretch
            #     40 driven -> ratio 0.99, no stretch
            # With 20 driven it still tracks perfectly (ratio 1.00) at a
            # 1.0 m/s lift. Too few particles have to carry the whole
            # garment's weight, which winds up the 1e12 stretch springs and
            # eventually flings it -- exactly the "it explodes while I am
            # holding it" report (that grab had captured 13).
            #
            # So the tolerance ball still decides IF a grab is allowed, and
            # this tops the DRIVEN set up to a count known to stay stable,
            # taking the particles nearest the grasp centre.
            if GRASP_MIN_PARTICLES <= len(idx) < GRASP_DRIVEN_TARGET:
                _d_centre = torch.linalg.norm(all_pos - pos, dim=1)
                idx = torch.argsort(_d_centre)[:GRASP_DRIVEN_TARGET]
            elif len(idx) > GRASP_DRIVEN_TARGET:
                _d_centre = torch.linalg.norm(all_pos[idx] - pos, dim=1)
                idx = idx[torch.argsort(_d_centre)[:GRASP_DRIVEN_TARGET]]
            # [isaac-5.1.0 compat: single layer]
            # Grab ONE face of the garment, not both.
            #
            # To put a shirt on someone the gripper has to lift the
            # front away from the back; catching both layers just picks
            # the shirt up flat and there is no opening left to feed an
            # arm through.
            #
            # The two faces show up as two clusters in distance-from-the-
            # grasp-centre: the near layer, then a gap of roughly the
            # garment's local thickness, then the far layer. Sort by that
            # distance and cut at the first gap wider than
            # SINGLE_LAYER_GAP. No gap means only one layer was in reach
            # to begin with, and everything is kept.
            #
            # Deliberately not done by projecting onto the gripper's
            # approach axis: that needs the gripper's orientation, and
            # this works whatever angle the fabric is approached from.
            if (SINGLE_LAYER_GRASP and len(idx) > GRASP_MIN_PARTICLES
                    and garment_faces[ci] is not None):
                # Keep whichever face the pinch is mostly on. The nearest
                # particle alone would be too fickle when the two layers
                # touch, so the closest few vote.
                _lbl = torch.tensor(garment_faces[ci], device=idx.device)
                _dc = torch.linalg.norm(all_pos[idx] - pos, dim=1)
                _near = idx[torch.argsort(_dc)[:GRASP_MIN_PARTICLES]]
                _vote = int(torch.sign(_lbl[_near].sum()).item()) or 1
                _same = idx[_lbl[idx] == _vote]
                if len(_same) >= GRASP_MIN_PARTICLES:
                    idx = _same
                # [isaac-5.1.0 compat: refill one face]
                # Top the driven set back up to GRASP_DRIVEN_TARGET,
                # staying on the face just chosen.
                #
                # The set was already filled to the target further up, but
                # that happens BEFORE this filter, which then keeps only one
                # layer -- measured at a mean 0.63 of what went in. So a
                # nominal 24 became roughly 15 in practice, and too few
                # particles carrying the whole garment is precisely what
                # makes it slide out of the fingers on the way up ("with a weak
                # grip the T-shirt always seems to slide down when lifted").
                # Refilling here keeps the single-layer property and the
                # full driven count at the same time.
                if len(idx) < GRASP_DRIVEN_TARGET:
                    _face_all = torch.where(_lbl == _vote)[0]
                    if len(_face_all) >= GRASP_MIN_PARTICLES:
                        _df = torch.linalg.norm(all_pos[_face_all] - pos, dim=1)
                        _take = min(GRASP_DRIVEN_TARGET, len(_face_all))
                        idx = _face_all[torch.argsort(_df)[:_take]]
            elif False:
                _dc = torch.linalg.norm(all_pos[idx] - pos, dim=1)
                _order = torch.argsort(_dc)
                _sorted = _dc[_order]
                _gaps = _sorted[1:] - _sorted[:-1]
                # Only consider cuts that leave a real cluster behind.
                # Cutting at the FIRST wide gap is wrong: the nearest
                # particle is often on its own a spacing away from the
                # rest, so the near 'layer' comes out as a single
                # particle. The separating gap is the first one past a
                # cluster that is already big enough to grab.
                _from = GRASP_MIN_PARTICLES - 1
                _tail = _gaps[_from:]
                _big = torch.where(_tail > SINGLE_LAYER_GAP)[0]
                if len(_big) > 0:
                    _cut = int(_big[0].item()) + _from + 1
                    if GRASP_MIN_PARTICLES <= _cut < len(idx):
                        idx = idx[_order[:_cut]]
            if len(idx) >= GRASP_MIN_PARTICLES:
                min_d = torch.min(dists[idx]).item()
                if best is None or min_d < best[0]:
                    best = (min_d, ci, idx, all_pos[idx] - pos)
        if best is not None:
            _, ci, idx, offsets = best
            state["grabbed"] = (ci, idx, offsets)
            print(f"[Teleop] {label} grabbed {len(idx)} particles from garment #{ci} (pinched between fingertips)")
            return True

        if ENABLE_GRASP_NEAREST_FALLBACK:
            # Strict check found nothing at all -- fall back to
            # whichever GRASP_MIN_PARTICLES particles are physically
            # closest to the fingertip segment, no GRASP_TOLERANCE
            # cutoff. Recomputes the same point-to-segment distances
            # (cheap: 4 garments x ~566 points) rather than carrying
            # them out of the loop above, since this path is the
            # uncommon one.
            fallback_best = None
            for ci, gc in enumerate(garment_cloths):
                all_pos = gc.get_world_positions()[0]
                to_particle = all_pos - left_t
                t = torch.clamp((to_particle @ seg) / seg_len_sq, 0.0, 1.0)
                closest = left_t + t.unsqueeze(-1) * seg
                dists = torch.linalg.norm(all_pos - closest, dim=1)
                k = min(GRASP_MIN_PARTICLES, dists.shape[0])
                nearest_dists, nearest_idx = torch.topk(dists, k, largest=False)
                min_d = nearest_dists[0].item()
                if fallback_best is None or min_d < fallback_best[0]:
                    fallback_best = (min_d, ci, nearest_idx, all_pos[nearest_idx] - pos)
            if fallback_best is not None:
                min_d, ci, idx, offsets = fallback_best
                state["grabbed"] = (ci, idx, offsets)
                print(f"[Teleop] {label} grabbed {len(idx)} NEAREST particles from garment #{ci} "
                      f"(closest at {min_d:.3f}m -- outside the normal {GRASP_TOLERANCE}m pincer tolerance, "
                      f"fallback grab)")
                return True

        print(f"[Teleop] {label} nothing confidently caught between the fingertips "
              f"(need >= {GRASP_MIN_PARTICLES} particles within {GRASP_TOLERANCE}m) to grab")
        return False

    # No passive proximity auto-grab -- attaching only happens as a
    # result of an explicit close, not just from being nearby.
    def toggle_gripper():
        state["gripper_closed"] = not state["gripper_closed"]
        if state["gripper_closed"]:
            # [isaac-5.1.0 compat: grab timing]
            # Do NOT grab here. This fires the instant the key goes
            # down, while the fingers are still travelling and still
            # shoving the fabric around -- so the captured particles
            # and their offsets describe a pose the cloth is about to
            # leave, and the follow then drags them toward a stale
            # target. Ask instead; the main loop takes the grab once
            # the fingers have arrived and the cloth has gone quiet.
            state["pending_grab"] = True
        else:
            state["pending_grab"] = False
            state["grabbed"] = None
            print(f"[Teleop] {label} released")

    rig["attempt_grab"] = _attempt_grab
    return toggle_gripper


# # [isaac-5.1.0 compat: grab cannot push cloth into the body]
#
# The grab writes held-particle POSITIONS absolutely, every physics step. Those
# writes know nothing about the mannequin, so dragging the shirt onto the figure
# drives the held particles straight through the skin and holds them there -- a
# real gripper cannot do that, and the user reported exactly it: "in the F3 state
# nothing should be penetrating; penetration only happens when I grab the cloth and pull it
# toward the person myself".
#
# It got worse when the follow moved to the physics rate (follow_grabbed), and
# that is the tell: at 60 Hz the solver had three uncorrected steps per frame to
# push the cloth back out, at 240 Hz it has none. Raising collision margins does
# not help, because the position write simply overrides whatever the solver did.
#
# So the follow projects its target out of the body before writing it. Nearest
# body vertex, signed against that vertex's own area-weighted normal, pushed back
# to GRAB_BODY_CLEARANCE outside the skin if it went under. Static mannequin, so
# the geometry is gathered once; the query is a single cdist on the sim device
# against ~10k vertices, which at 400 held particles is nothing.
# The strong arm-pull regression reproduced the missing case: all 200 attached
# nodes crossed a 124mm forearm when this was zero. Keep a 6mm clearance, equal
# to the authored cloth/body contact offset. The cdist runs only while cloth is
# attached, not during idle GUI startup or passive drape.
GRAB_BODY_CLEARANCE = _feel("GRAB_BODY_CLEARANCE", 0.003)
_BODY_GEOM = {}


def register_body_collider(prim):
    """Remember the mannequin's collision surface for the grab to avoid."""
    from pxr import Gf as _Gf, Usd as _Usd, UsdGeom as _UsdGeom

    try:
        mesh = _UsdGeom.Mesh(prim)
        xf = _UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            _Usd.TimeCode.Default())
        pts = np.array([xf.Transform(_Gf.Vec3d(*map(float, q)))
                        for q in mesh.GetPointsAttr().Get()])
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        idxs = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
        tris, off = [], 0
        for c in counts:
            f = idxs[off:off + c]
            off += c
            for k in range(1, c - 1):
                tris.append((f[0], f[k], f[k + 1]))
        tris = np.asarray(tris)
        from Env_Config.Human.HandSphereColliders import append_hand_sphere_surfaces
        pts, tris = append_hand_sphere_surfaces(prim, pts, tris)
        # Area-weighted vertex normals: the face normals summed onto their
        # vertices, which is the cheap stand-in for the angle-weighted
        # pseudonormal that makes a nearest-point sign test reliable.
        fn = np.cross(pts[tris[:, 1]] - pts[tris[:, 0]],
                      pts[tris[:, 2]] - pts[tris[:, 0]])
        nrm = np.zeros_like(pts)
        for c in range(3):
            np.add.at(nrm, tris[:, c], fn)
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
        # Imported assets are not guaranteed to use the same winding.  The
        # collision sweep needs normals that point out of the body so it can
        # remove only the inward component and retain surface sliding.
        if np.median(np.sum(nrm * (pts - pts.mean(axis=0)), axis=1)) < 0.0:
            nrm = -nrm
            fn = -fn
        # Cover the triangle interiors as well as the collision-mesh vertices.
        # This closes the visual pinholes through which the rendered skin could
        # flash even though every vertex-sphere test passed.
        edges = np.sort(np.concatenate((tris[:, [0, 1]], tris[:, [1, 2]],
                                        tris[:, [2, 0]]), axis=0), axis=1)
        edges = np.unique(edges, axis=0)
        edge_nrm = nrm[edges[:, 0]] + nrm[edges[:, 1]]
        edge_nrm /= np.maximum(
            np.linalg.norm(edge_nrm, axis=1, keepdims=True), 1e-12)
        face_nrm = fn / np.maximum(
            np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
        samples = np.concatenate((pts,
                                  0.5 * (pts[edges[:, 0]] + pts[edges[:, 1]]),
                                  pts[tris].mean(axis=1)), axis=0)
        sample_nrm = np.concatenate((nrm, edge_nrm, face_nrm), axis=0)
        from Env_Config.Garment.SurfaceContactGuard import SurfaceContactGuard
        _BODY_GEOM["sweep"] = SurfaceContactGuard(pts, tris, str(_to_t(pts).device))
        _BODY_GEOM["pts"] = _to_t(samples)
        _BODY_GEOM["nrm"] = _to_t(sample_nrm)
        print(f"[Teleop] grab will keep cloth {GRAB_BODY_CLEARANCE * 1000:.0f}mm "
              f"clear of {prim.GetPath()} ({len(samples)} surface samples)", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[Teleop] body collider not registered for the grab "
              f"({type(exc).__name__}: {exc}); the grab can push cloth through "
              f"the figure", flush=True)


def project_out_of_body(current, target, allow_slide=True):
    """Sweep an incremental write, blocking inward but retaining tangent motion."""
    if not _BODY_GEOM or GRAB_BODY_CLEARANCE <= 0.0:
        return target
    return _BODY_GEOM["sweep"].project(current, target, GRAB_BODY_CLEARANCE,
                                        allow_slide=allow_slide, coherent=True)


# # [isaac-5.1.0 compat: cloth shield]
#
# "I want the human mesh to never poke through the cloth" -- and with PhysX
# particle cloth that cannot be had from the collision settings, because the
# cloth only exists at its vertices. Each particle is a sphere; the triangle
# between three of them is invisible to the solver. Measured with
# scripts/probe_gap.py: 25% of this garment's edges were wider than a particle
# at rest and 46% when pulled, with gaps up to 101 mm against a 105 mm forearm.
# Pushing PARTICLES out of the body does not help -- the ones beside a hole are
# already outside it.
#
# So the surface is enforced directly. Every edge MIDPOINT is tested as well as
# every particle, and when a sample lands inside the figure both endpoints of
# that edge are pushed out along the body's own normal. A triangle can then only
# lie across the skin if all of its edges do, which the same test forbids -- so
# the fabric is opaque to the body at the resolution of half an edge, whatever
# the particle radius is.
#
# The mannequin never moves, so its distance field is baked once into a grid and
# every later query is an array index. That is what makes this affordable at the
# physics rate.
# Isaac Sim 6 surface deformables already collide as continuous triangles.
# This projection was necessary for 5.x particle cloth, but on FEM it bypasses
# the solver by overwriting shared nodes and can inject an oversized correction
# when a node belongs to several penetrating edges/faces.  Keep it available as
# a diagnostic fallback, but let native contact + speculative CCD own the body
# interaction by default.
CLOTH_SHIELD = bool(int(_feel("CLOTH_SHIELD", 0)))
CLOTH_SHIELD_CLEARANCE = _feel("CLOTH_SHIELD_CLEARANCE", 0.004)
CLOTH_SHIELD_RES = _feel("CLOTH_SHIELD_RES", 0.006)
_SHIELD = {}


def build_cloth_shield(prim, garment_cloths):
    """Bake the static body into a distance grid, and cache the cloth's edges."""
    if not CLOTH_SHIELD:
        return
    from pxr import Gf as _Gf, Usd as _Usd, UsdGeom as _UsdGeom
    from scipy.spatial import cKDTree

    try:
        mesh = _UsdGeom.Mesh(prim)
        xf = _UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            _Usd.TimeCode.Default())
        bp = np.array([xf.Transform(_Gf.Vec3d(*map(float, q)))
                       for q in mesh.GetPointsAttr().Get()])
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        idxs = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
        tris, off = [], 0
        for c in counts:
            f = idxs[off:off + c]
            off += c
            for k in range(1, c - 1):
                tris.append((f[0], f[k], f[k + 1]))
        tris = np.asarray(tris)
        fn = np.cross(bp[tris[:, 1]] - bp[tris[:, 0]], bp[tris[:, 2]] - bp[tris[:, 0]])
        nrm = np.zeros_like(bp)
        for c in range(3):
            np.add.at(nrm, tris[:, c], fn)
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)

        res = float(CLOTH_SHIELD_RES)
        pad = 0.05
        lo = bp.min(axis=0) - pad
        hi = bp.max(axis=0) + pad
        dims = np.maximum(((hi - lo) / res).astype(int) + 1, 2)
        gx, gy, gz = (np.arange(d) * res for d in dims)
        grid = np.stack(np.meshgrid(gx, gy, gz, indexing="ij"), axis=-1).reshape(-1, 3) + lo
        tree = cKDTree(bp)
        dist, near = tree.query(grid, k=1, workers=-1)
        signed = np.einsum("ij,ij->i", grid - bp[near], nrm[near])
        # Distance, signed by which side of the nearest surface point it is on.
        sd = np.where(signed < 0, -dist, dist).astype(np.float32)
        _SHIELD["sd"] = _to_t(sd.reshape(tuple(dims)))
        _SHIELD["nrm"] = _to_t(nrm[near].reshape(tuple(dims) + (3,)))
        _SHIELD["lo"] = _to_t(lo)
        _SHIELD["res"] = float(res)
        _SHIELD["dims"] = torch.as_tensor(
            np.asarray(dims) - 1, device=_SHIELD["sd"].device, dtype=torch.long)

        # Edges of every garment, cached: the midpoints are what make the
        # SURFACE opaque rather than just the vertices.
        _SHIELD["edges"] = []
        for gc in garment_cloths:
            mp = None
            try:
                mp = gc.prims[0]
            except Exception:  # noqa: BLE001
                pass
            _SHIELD["edges"].append(None if mp is None else None)
        print(f"[Teleop] cloth shield: body baked into a "
              f"{'x'.join(str(int(d)) for d in dims)} grid at {res * 1000:.0f}mm, "
              f"clearance {CLOTH_SHIELD_CLEARANCE * 1000:.0f}mm", flush=True)
    except Exception as exc:  # noqa: BLE001
        _SHIELD.clear()
        print(f"[Teleop] cloth shield unavailable ({type(exc).__name__}: {exc})",
              flush=True)


def set_shield_edges(index, edge_array, tri_array=None):
    """Register one garment's edges (N,2) and triangles (M,3), computed once."""
    if _SHIELD:
        while len(_SHIELD.setdefault("edges", [])) <= index:
            _SHIELD["edges"].append(None)
            _SHIELD.setdefault("tris", []).append(None)
        while len(_SHIELD.setdefault("tris", [])) <= index:
            _SHIELD["tris"].append(None)
        _SHIELD["edges"][index] = _to_idx(edge_array)
        _SHIELD["tris"][index] = None if tri_array is None else _to_idx(tri_array)


def _shield_lookup(q):
    """Signed distance and outward normal at each point, from the baked grid."""
    idx = ((q - _SHIELD["lo"]) / _SHIELD["res"]).long()
    idx = torch.max(torch.zeros_like(idx), torch.min(idx, _SHIELD["dims"]))
    i, j, k = idx[:, 0], idx[:, 1], idx[:, 2]
    return _SHIELD["sd"][i, j, k], _SHIELD["nrm"][i, j, k]


def shield_cloth(garment_cloths):
    """Push any cloth that is lying across the figure back out of her.

    Both the particles and the edge midpoints are tested. The midpoints are the
    point of the exercise: a hole between two particles is exactly where the
    body comes through, and neither particle is inside anything.
    """
    if not _SHIELD:
        return
    clear = CLOTH_SHIELD_CLEARANCE
    for ci, gc in enumerate(garment_cloths):
        edges = _SHIELD["edges"][ci] if ci < len(_SHIELD.get("edges", [])) else None
        pos = gc.get_world_positions()
        p = pos[0]
        moved = False

        sd, n = _shield_lookup(p)
        bad = sd < clear
        if bool(bad.any()):
            p[bad] = p[bad] + n[bad] * (clear - sd[bad]).unsqueeze(1)
            moved = True

        if edges is not None and len(edges):
            a, b = edges[:, 0], edges[:, 1]
            mid = 0.5 * (p[a] + p[b])
            sdm, nm = _shield_lookup(mid)
            badm = sdm < clear
            if bool(badm.any()):
                push = (clear - sdm[badm]).unsqueeze(1) * nm[badm]
                p.index_add_(0, a[badm], push)
                p.index_add_(0, b[badm], push)
                moved = True

        # And the middle of each triangle. Vertices plus edge midpoints leave
        # the face's own centre as the last place the skin can show through,
        # and on a coarse mesh that centre is a third of an edge from any other
        # sample.
        tris = (_SHIELD.get("tris") or [None])[ci] if ci < len(_SHIELD.get("tris") or []) else None
        if tris is not None and len(tris):
            t0, t1, t2 = tris[:, 0], tris[:, 1], tris[:, 2]
            cen = (p[t0] + p[t1] + p[t2]) / 3.0
            sdc, nc = _shield_lookup(cen)
            badc = sdc < clear
            if bool(badc.any()):
                push = (clear - sdc[badc]).unsqueeze(1) * nc[badc]
                p.index_add_(0, t0[badc], push)
                p.index_add_(0, t1[badc], push)
                p.index_add_(0, t2[badc], push)
                moved = True

        if moved:
            pos[0] = p
            gc.set_world_positions(pos)


def follow_grabbed(rig, garment_cloths, dt):
    """Drag one rig's held particles to where its gripper says they should be.

    # [isaac-5.1.0 compat: follow at the physics rate]

    This used to be the tail of drive_robot, i.e. it ran once per RENDERED
    frame -- and the scene runs PhysX at 240 Hz against a 60 Hz loop, so three
    of every four physics steps advanced with nothing correcting the grip. The
    stretched springs pull the held particles back in those three steps and the
    grip settles wherever that stalemate lands.

    Measured with scripts/probe_pairpull.py, same scene, same tension, same 400
    held particles, nothing changed but the rate:

        corrected once per rendered frame   lag mean 0.119 m, 99% slipping
        corrected every physics step        lag mean 0.030 m, 17% slipping

    So it is called from a physics-step callback now (see main), which fires
    once per PhysX step whoever drives the stepping -- including
    simulation_app.update(), which is what the teleop loop calls.

    Two consequences worth knowing:

      * `dt` is the physics step, not 1/60, so GRAB_FOLLOW_SPEED and the
        velocity written back are per-second quantities that finally mean what
        they say.
      * GRAB_UPDATE_INTERVAL now counts physics steps rather than rendered
        frames. At its default of 1 that is the same knob; at 4 it reproduces
        roughly the old behaviour.

    It also explains why probe_pull.py's numbers were better than the session
    felt: that probe corrects every physics step, so it was measuring this
    version of the grip before this version existed.
    """
    if os.environ.get('STRETCH4_NATIVE_GRASP', '1') == '1':
        from Env_Config.Garment.NativeGrasp import update_native_grasp
        if update_native_grasp(rig, garment_cloths, GRAB_ANCHOR_COUNT):
            return
    state = rig["state"]
    if state["grabbed"] is None:
        return
    if state.get("_active_grab") is not state["grabbed"]:
        # A checkpoint/re-grab can have identical node indices on a different
        # shirt. Cached safe positions belong to this grab, not to those IDs.
        for key in ("_grab_anchor_mask", "_grab_anchor_idx", "_body_safe_idx",
                    "_body_safe_pos", "_grab_post_key", "_grab_post_anchor_pos"):
            state.pop(key, None)
        state["_active_grab"] = state["grabbed"]
    state["_follow_calls"] = state.get("_follow_calls", 0) + 1
    if state["_follow_calls"] % GRAB_UPDATE_INTERVAL:
        return
    FOLLOW_CALLS[0] += 1
    ci, idx, offsets = state["grabbed"]
    gc = garment_cloths[ci]
    pos = _to_t(_grasp_link_pos(rig))
    positions = gc.get_world_positions()
    target = pos + offsets
    current = positions[0, idx]
    _anchor_mask = state.get("_grab_anchor_mask")
    _anchor_idx = state.get("_grab_anchor_idx")
    if (_anchor_mask is None or _anchor_idx is None
            or len(_anchor_mask) != len(idx)
            or not torch.equal(_anchor_idx, idx)):
        _anchor_mask = torch.zeros(len(idx), dtype=torch.bool, device=idx.device)
        _anchor_n = min(max(GRAB_ANCHOR_COUNT, 1), len(idx))
        _nearest = torch.argsort(torch.linalg.norm(offsets, dim=1))[:_anchor_n]
        _anchor_mask[_nearest] = True
        state["_grab_anchor_mask"] = _anchor_mask
        state["_grab_anchor_idx"] = idx.detach().clone()
    _safe_idx = state.get("_body_safe_idx")
    _safe_pos = state.get("_body_safe_pos")
    if (_safe_idx is None or _safe_pos is None
            or not torch.equal(_safe_idx, idx)):
        _safe_idx = idx.detach().clone()
        _safe_pos = current.detach().clone()
    current = project_out_of_body(_safe_pos, current)
    delta = target - current
    alpha = torch.full((len(idx), 1), GRAB_SMOOTHING_ALPHA,
                       dtype=delta.dtype, device=delta.device)
    alpha[_anchor_mask] = 1.0
    step = delta * alpha
    step_norm = torch.linalg.norm(step, dim=1, keepdim=True)
    _step_lim = torch.full_like(
        step_norm,
        min(MAX_GRAB_STEP, GRAB_FOLLOW_SPEED * GRAB_UPDATE_INTERVAL * dt))
    _step_lim[_anchor_mask] = min(
        MAX_GRAB_STEP, GRAB_ANCHOR_SPEED * GRAB_UPDATE_INTERVAL * dt)
    step_scale = torch.clamp(_step_lim / (step_norm + 1e-8), max=1.0)
    new_pos = current + step * step_scale
    # Project the actual incremental write, not the final gripper target. If
    # current and target are outside opposite sides of an arm, projecting only
    # the target misses the entire solid and interpolation still enters it.
    new_pos = project_out_of_body(current, new_pos)
    state["_body_safe_idx"] = idx.detach().clone()
    state["_body_safe_pos"] = new_pos.detach().clone()
    positions[0, idx] = new_pos
    gc.set_world_positions(positions)
    velocities = gc.get_velocities()
    # Match velocity to the displacement just applied instead of
    # zeroing it -- see GRAB_SMOOTHING_ALPHA's comment for why
    # zeroing was the actual bug.
    velocities[0, idx] = (new_pos - current) / (GRAB_UPDATE_INTERVAL * dt)
    gc.set_velocities(velocities)


# Preventive limiter: normal robot manipulation tops out at about 1.4m/s. A
# 2m/s cap therefore leaves control motion alone, but keeps a cloth node below
# 8.3mm per 240Hz step instead of allowing the 80--120mm collision jumps seen
# in the failing GUI log. This acts before the next solve, not after a blow-up.
GARMENT_MAX_SPEED = _feel("GARMENT_MAX_SPEED", 2.0)


def begin_cloth_motion(garment_cloths):
    """Capture each whole shirt before controller writes or the FEM solve.

    Re-captured every step so an explicit reset/checkpoint restore is not
    mistaken for a simulated trajectory through the scene.
    """
    for gc in garment_cloths:
        if not gc.is_physics_tensor_entity_valid():
            gc._contact_start = None
            guard = _BODY_GEOM.get("sweep")
            if guard is not None:
                guard.clear_cache()
            continue
        gc._contact_start = gc.get_world_positions()[0].clone()
        if not hasattr(gc, "_contact_edges"):
            triangles = np.asarray(UsdGeom.Mesh(gc.prim).GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
            edges = np.unique(np.sort(np.concatenate((triangles[:, [0, 1]],
                              triangles[:, [1, 2]], triangles[:, [2, 0]])), axis=1), axis=0)
            gc._contact_edges = torch.as_tensor(edges.astype(np.int32),
                                               device=gc._contact_start.device)
            gc._contact_triangles = torch.as_tensor(triangles.astype(np.int32),
                                                   device=gc._contact_start.device)


def guard_cloth_motion(garment_cloths):
    """Reject residual crossings on all nodes, including the ungrasped cloth."""
    guard = _BODY_GEOM.get("sweep")
    if guard is None:
        return
    pending, flags = [], []
    for gc in garment_cloths:
        if not gc.is_physics_tensor_entity_valid():
            gc._contact_start = None
            continue
        start = getattr(gc, "_contact_start", None)
        if start is None:
            continue
        pos = gc.get_world_positions()
        moving = torch.any(torch.abs(pos[0] - start) > 1.e-8)
        # Every node trajectory and every swept triangle lies within this
        # union AABB. Only disjoint boxes may bypass the detailed guard; a
        # distant-to-distant jump across the body must still be checked.
        lower = torch.minimum(start.amin(dim=0), pos[0].amin(dim=0))
        upper = torch.maximum(start.amax(dim=0), pos[0].amax(dim=0))
        overlap = ((upper >= guard.lower_tensor - 1.e-6).all()
                   & (lower <= guard.upper_tensor + 1.e-6).all())
        flags.append(moving.to(torch.int32) * (1 + overlap.to(torch.int32)))
        pending.append((gc, start, pos))
    if not pending:
        return
    # One tiny GPU-to-host decision packet for all garments, rather than a
    # synchronize per shirt. Collision/repair work remains on the GPU.
    for (gc, start, pos), status in zip(pending, torch.stack(flags).cpu().tolist()):
        if status == 0:
            continue
        if status == 1:
            gc._contact_start = pos[0].clone()
            continue
        vel = gc.get_velocities()
        safe, safe_vel = guard.sweep(start, pos[0], vel[0], gc._contact_edges,
                                    gc._contact_triangles)
        changed = torch.linalg.norm(safe - pos[0], dim=1) > 1.e-7
        if bool(changed.any()):
            pos[0] = safe
            vel[0] = safe_vel
            gc.set_world_positions(pos)
            gc.set_velocities(vel)
        gc._contact_start = safe.clone()


def sync_grab_contact(rig, garment_cloths, post_step=False):
    """Keep grip history on the contact-corrected positions actually accepted."""
    state = rig["state"]
    grabbed = state.get("grabbed")
    if grabbed is None:
        return
    ci, idx, _ = grabbed
    if not garment_cloths[ci].is_physics_tensor_entity_valid():
        return
    actual = garment_cloths[ci].get_world_positions()[0, idx].clone()
    state["_body_safe_idx"] = idx.detach().clone()
    state["_body_safe_pos"] = actual
    mask = state.get("_grab_anchor_mask")
    if post_step and mask is not None:
        state["_grab_post_key"] = grabbed
        state["_grab_post_anchor_pos"] = actual[mask].clone()


def prevent_wheel_lift(rigs):
    """Prevent Stretch4 mobile base from lifting off the floor under arm reaction forces.

    Layer 2: a physics-rate P-controller holding the base at its spawn height.
    Whenever the base climbs above its initial resting height (or is moving upward),
    drive a corrective downward velocity proportional to how far above home it is.
    """
    if not PREVENT_WHEEL_LIFT:
        return
    for _r in rigs:
        if "robot" not in _r or "state" not in _r:
            continue
        _rb = _r["robot"]
        if not hasattr(_rb, "_articulation_view") or not _rb._articulation_view.is_physics_handle_valid():
            continue
        _pos_np = _to_np(_rb.get_world_pose()[0]).reshape(-1)
        _home_z = _r["state"].get("_base_home_z")
        if _home_z is None:
            _home_z = _r["state"]["_base_home_z"] = float(_pos_np[2])
        _z_err = float(_pos_np[2]) - _home_z
        _lv = _to_np(_rb.get_linear_velocity()).reshape(-1)
        if _z_err > 0.0 or _lv[2] > 0.0:
            _av = _to_np(_rb.get_angular_velocity()).reshape(-1)
            _lv[2] = float(np.clip(-_z_err * 20.0, -1.0, 0.0))
            try:
                _rb._articulation_view.set_velocities(
                    _to_t(np.concatenate([_lv, _av])[None, :]))
            except Exception:
                pass


def cloth_pre_step(garment_cloths, rigs, step_size):
    """Capture once; a native attachment does not overwrite nodal positions."""
    from Env_Config.Garment.ContinuousClothControl import apply_base_drive
    for _rig in rigs: apply_base_drive(_rig, step_size)
    prevent_wheel_lift(rigs)
    begin_cloth_motion(garment_cloths)
    for rig in rigs:
        follow_grabbed(rig, garment_cloths, step_size)
    shield_cloth(garment_cloths)
    legacy = any(r['state'].get('grabbed') is not None and
                 not r['state'].get('_native_enabled', False) for r in rigs)
    if legacy or CLOTH_SHIELD:
        guard_cloth_motion(garment_cloths)
    for rig in rigs:
        if not rig['state'].get('_native_enabled', False):
            sync_grab_contact(rig, garment_cloths)


def cloth_post_step(garment_cloths, rigs, step_size):
    """Guard every FEM step; repeat only after a legacy anchor actually writes."""
    guard_cloth_motion(garment_cloths)
    legacy = False
    for rig in rigs:
        if not rig['state'].get('_native_enabled', False):
            snap_grab_anchors(rig, garment_cloths, step_size)
            legacy = legacy or rig['state'].get('grabbed') is not None
    if legacy:
        guard_cloth_motion(garment_cloths)
    for rig in rigs:
        if not rig['state'].get('_native_enabled', False):
            sync_grab_contact(rig, garment_cloths, post_step=True)
    prevent_cloth_blowup(garment_cloths)


def prevent_cloth_blowup(garment_cloths):
    if GARMENT_MAX_SPEED <= 0.0:
        return 0
    limited = 0
    for gc in garment_cloths:
        if not gc.is_physics_tensor_entity_valid():
            continue
        vel = gc.get_velocities()
        v = vel[0]
        speed = torch.linalg.norm(v, dim=1, keepdim=True)
        bad = speed[:, 0] > GARMENT_MAX_SPEED
        if bool(bad.any()):
            v[bad] *= GARMENT_MAX_SPEED / (speed[bad] + 1e-8)
            vel[0] = v
            gc.set_velocities(vel)
            limited += int(bad.sum().item())
    return limited


def snap_grab_anchors(rig, garment_cloths, dt):
    """Post-solve pin for only the nodes physically inside the jaws.

    The normal follow runs before PhysX and moves a compliant support patch.
    Surface FEM can move those nodes again during the solve; pinning just the
    nearest 12 afterwards removes the visible gripper gap without teleporting
    the old 200-node patch or bypassing the body sweep.
    """
    state = rig["state"]
    if state.get('_native_enabled', False):
        return
    grabbed = state.get("grabbed")
    mask = state.get("_grab_anchor_mask")
    if grabbed is None or mask is None or not bool(mask.any()):
        return
    ci, idx, offsets = grabbed
    gc = garment_cloths[ci]
    positions = gc.get_world_positions()
    ai = idx[mask]
    safe_all = state.get("_body_safe_pos")
    if safe_all is not None and len(safe_all) == len(idx):
        # Do not pin the 64-node support patch to the gripper after the solve;
        # preserve its compliant FEM motion. Only reject a solver-produced
        # segment that crosses the body. This closes the one-step window in
        # which a non-anchor support node could appear on the arm's far side.
        solved = positions[0, idx]
        guarded = project_out_of_body(safe_all, solved, allow_slide=False)
        positions[0, idx] = guarded
        safe_all = guarded.detach().clone()
        current = guarded[mask]
    else:
        current = positions[0, ai]
    target = _to_t(_grasp_link_pos(rig)) + offsets[mask]
    # Limit movement of the attachment from its previous post-step position.
    # Limiting from the FEM-displaced position would also limit restoration of
    # the constraint, producing a persistent visible gripper gap under tension.
    reference = current
    if state.get("_grab_post_key") is grabbed:
        reference = state["_grab_post_anchor_pos"]
    delta = target - reference
    distance = torch.linalg.norm(delta, dim=1, keepdim=True)
    limit = max(float(dt), 0.0) * GRAB_ANCHOR_SPEED
    target = reference + delta * torch.clamp(limit / (distance + 1.e-8), max=1.0)
    # The jaw anchors represent one rigid grasp. If one anchor meets the body,
    # limit the commanded patch coherently instead of letting the other anchors
    # walk around the arm independently while their neighbour remains blocked.
    command = target - reference
    command_len = torch.linalg.norm(command, dim=1)
    swept_command = project_out_of_body(reference, target)
    permitted = torch.linalg.norm(swept_command - reference, dim=1)
    progress = torch.where(command_len > 1.e-7,
                           permitted / command_len.clamp(min=1.e-7),
                           torch.ones_like(command_len)).clamp(0.0, 1.0).min()
    target = reference + command * progress
    safe = project_out_of_body(current, target)
    state["_grab_post_key"] = grabbed
    state["_grab_post_anchor_pos"] = safe.detach().clone()
    positions[0, ai] = safe
    gc.set_world_positions(positions)

    velocities = gc.get_velocities()
    # A post-solve positional repair is not attachment motion. Feeding the
    # repair/current delta back as velocity kicks a stationary grip every step.
    # Only motion between accepted attachment poses belongs in the velocity.
    av = (safe - reference) / max(float(dt), 1e-9)
    speed = torch.linalg.norm(av, dim=1, keepdim=True)
    av *= torch.clamp(GARMENT_MAX_SPEED / (speed + 1e-8), max=1.0)
    velocities[0, ai] = av
    gc.set_velocities(velocities)

    if safe_all is not None and len(safe_all) == len(idx):
        safe_all[mask] = safe
        state["_body_safe_pos"] = safe_all


def drive_robot(rig, keymap, held, garment_cloths, dt, _frame, commands=None):
    state = rig["state"]
    robot = rig["robot"]
    ctrl = rig["ctrl"]
    before = np.array([state["lift"], *([state["arm"] / 4.0] * 4), state["yaw"], state["pitch"], state["roll"]])

    target_base_fwd = 0.0
    target_base_turn = 0.0
    if keymap["base_fwd_pos"] in held:
        target_base_fwd += BASE_LINEAR_RATE
    if keymap["base_fwd_neg"] in held:
        target_base_fwd -= BASE_LINEAR_RATE
    target_base_strafe = 0.0
    if keymap["base_strafe_pos"] in held:
        target_base_strafe += BASE_LINEAR_RATE
    if keymap["base_strafe_neg"] in held:
        target_base_strafe -= BASE_LINEAR_RATE
    # dt, not _dt: the surrounding function names it dt, and the
    # mismatch was a NameError on the first frame -- which Kit
    # swallows under fastShutdown, so the app just vanished four
    # seconds in with no traceback and looked like a memory problem.
    if commands is not None:
        target_base_strafe = commands["base_strafe"]
    strafe_delta = np.clip(target_base_strafe - state["base_strafe"],
                           -BASE_LINEAR_ACCEL * dt, BASE_LINEAR_ACCEL * dt)
    state["base_strafe"] += strafe_delta
    if keymap["base_turn_pos"] in held:
        target_base_turn += BASE_ANGULAR_RATE
    if keymap["base_turn_neg"] in held:
        target_base_turn -= BASE_ANGULAR_RATE
    if commands is not None:
        target_base_fwd = commands["base_fwd"]
        target_base_turn = commands["base_turn"]
    base_fwd_delta = np.clip(target_base_fwd - state["base_fwd"],
                              -BASE_LINEAR_ACCEL * dt, BASE_LINEAR_ACCEL * dt)
    state["base_fwd"] += base_fwd_delta
    base_turn_delta = np.clip(target_base_turn - state["base_turn"],
                               -BASE_ANGULAR_ACCEL * dt, BASE_ANGULAR_ACCEL * dt)
    state["base_turn"] += base_turn_delta

    if keymap["lift_pos"] in held:
        state["lift"] = min(rig["lift_hi"], state["lift"] + LIFT_RATE * dt)
    if keymap["lift_neg"] in held:
        state["lift"] = max(rig["lift_lo"], state["lift"] - LIFT_RATE * dt)
    if keymap["arm_pos"] in held:
        state["arm"] = min(rig["arm_hi"], state["arm"] + ARM_RATE * dt)
    if keymap["arm_neg"] in held:
        state["arm"] = max(rig["arm_lo"], state["arm"] - ARM_RATE * dt)
    if keymap["yaw_pos"] in held:
        state["yaw"] = min(rig["yaw_hi"], state["yaw"] + WRIST_RATE * dt)
    if keymap["yaw_neg"] in held:
        state["yaw"] = max(rig["yaw_lo"], state["yaw"] - WRIST_RATE * dt)
    if keymap["pitch_pos"] in held:
        state["pitch"] = min(rig["pitch_hi"], state["pitch"] + WRIST_RATE * dt)
    if keymap["pitch_neg"] in held:
        state["pitch"] = max(rig["pitch_lo"], state["pitch"] - WRIST_RATE * dt)
    if keymap["roll_pos"] in held:
        state["roll"] = min(rig["roll_hi"], state["roll"] + WRIST_RATE * dt)
    if keymap["roll_neg"] in held:
        state["roll"] = max(rig["roll_lo"], state["roll"] - WRIST_RATE * dt)

    if commands is not None:
        for axis in ("lift", "arm", "yaw", "pitch", "roll"):
            state[axis] = float(np.clip(state[axis] + commands[axis + "_rate"] * dt,
                                        rig[axis + "_lo"], rig[axis + "_hi"]))

    _grip_target = GRIPPER_CLOSED if state["gripper_closed"] else GRIPPER_OPEN
    _grip_delta = np.clip(_grip_target - state["grip_pos"],
                           -GRIPPER_RATE * dt, GRIPPER_RATE * dt)
    state["grip_pos"] += _grip_delta

    _all_idx = np.concatenate([
        [rig["lift_idx"]], rig["arm_joint_idx"], [rig["yaw_idx"]],
        [rig["pitch_idx"]], [rig["roll_idx"]], rig["grip_idx"],
    ])
    _all_pos = np.concatenate([
        [state["lift"]], [state["arm"] / 4.0] * 4, [state["yaw"]],
        [state["pitch"]], [state["roll"]], [state["grip_pos"], state["grip_pos"]],
    ])

    _, cur_quat = robot.get_world_pose()
    cur_quat = _to_np(cur_quat)
    w, x, y, z = cur_quat
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    forward_world = R @ np.array([1.0, 0.0, 0.0])
    forward_world[2] = 0.0
    fw_norm = np.linalg.norm(forward_world)
    if fw_norm > 1e-6:
        forward_world /= fw_norm
    up_world = R @ np.array([0.0, 0.0, 1.0])

    cur_lin_vel = _to_np(robot.get_linear_velocity())
    # Omnidirectional wheels: the base can move sideways as well as
    # forwards. Right is forward crossed with up.
    right_world = np.cross(forward_world, np.array([0.0, 0.0, 1.0]))
    _rn = np.linalg.norm(right_world)
    if _rn > 1e-6:
        right_world /= _rn
    target_lin_vel = (forward_world * state["base_fwd"]
                      + right_world * state["base_strafe"])
    if PREVENT_WHEEL_LIFT:
        target_lin_vel[2] = min(cur_lin_vel[2], 0.0)
    else:
        target_lin_vel[2] = cur_lin_vel[2]

    tilt_axis = np.cross(up_world, np.array([0.0, 0.0, 1.0]))
    # Hold the heading. Nothing was keeping it: the base is driven by
    # velocity, and pushing against the table or dragging the cloth
    # simply rotated it -- one robot ended up 43 degrees off, after
    # which "forward" and "sideways" pointed somewhere else and every
    # position correction drove it further away. The turn keys now
    # steer a remembered heading, and the base is servoed to it.
    _hdg = np.degrees(np.arctan2(forward_world[1], forward_world[0]))
    if state.get("heading") is None:
        state["heading"] = _hdg
    if abs(state["base_turn"]) > 1e-6:
        state["heading"] = _hdg
        _hold = 0.0
    else:
        _err = (state["heading"] - _hdg + 180.0) % 360.0 - 180.0
        _hold = float(np.clip(np.radians(_err) * 4.0, -1.5, 1.5))
    target_ang_vel = (up_world * (state["base_turn"] + _hold)
                      + tilt_axis * 5.0)
    # [continuous cloth: admit commands without pausing physics]
    from Env_Config.Garment.ContinuousClothControl import constrain_drive
    _all_pos, target_lin_vel, target_ang_vel = constrain_drive(
        rig, garment_cloths, before, _all_idx, _all_pos, target_lin_vel, target_ang_vel, dt)
    ctrl.apply_action(ArticulationAction(
        joint_positions=_to_t(_all_pos), joint_indices=_to_idx(_all_idx)))
    # [continuous cloth: finite base actuator]
    from Env_Config.Garment.ContinuousClothControl import set_base_target
    set_base_target(rig, target_lin_vel, target_ang_vel)




# [isaac-5.1.0 compat: viewport recording]
#
# Video capture that is bound to a camera of its OWN, not to the viewport.
#
# This is NOT the record/replay machinery HANDOFF section 6 had deleted. That
# recorded STATE and played it back to drive the robots; nothing here touches
# the robots or the physics. It writes pictures.
#
# The requirement that shapes it: "pressing the button starts recording from the current camera angle,
# and the recording angle stays fixed even if I move the camera while recording
# ... even after resetting or loading with P or F1, recording continues from the same camera
# angle". So:
#
#   * pressing start COPIES the viewport camera's world transform onto a
#     private /World/RecordCam and renders through THAT. Flying the viewport
#     around afterwards cannot move the shot, because the viewport is not what
#     is being recorded.
#   * the recorder lives in a module-level dict, outside everything world.reset()
#     rebuilds, and RecordCam is an ordinary stage prim that a reset does not
#     delete -- so P and the F1-F5 slots leave it running. The slots restore the
#     VIEWPORT camera, which is a different prim.
#   * it stops only when the stop key is pressed, or at Esc.
#
# It writes an .mp4 directly -- no frame files. The image has no ffmpeg binary
# and no imageio-ffmpeg, but it does ship OpenCV 4.11, whose VideoWriter encodes
# MPEG-4 with the "mp4v" fourcc (checked in the container: "avc1"/"H264" fail,
# "mp4v" and "XVID" open and produce a valid file).
#
# The frame RATE is the awkward part: a VideoWriter needs it at construction and
# the true rate is not known until the recording ends. So the first
# RECORD_FPS_PROBE frames are held in memory, the rate is measured off their
# timestamps, and the writer is then opened with the real number and the buffer
# flushed into it. That is why a recording plays back at the speed it happened
# rather than at a guessed 30. STRETCH4_REC_FPS pins it if you would rather.
RECORD_START_KEY = os.environ.get("STRETCH4_REC_START_KEY", "F9")
RECORD_STOP_KEY = os.environ.get("STRETCH4_REC_STOP_KEY", "F10")
RECORD_DIR = os.environ.get("STRETCH4_REC_DIR", "/output/recordings")
RECORD_RES = os.environ.get("STRETCH4_REC_RES", "1280,720")
# 0 = measure it from the first RECORD_FPS_PROBE frames.
RECORD_FPS = float(os.environ.get("STRETCH4_REC_FPS", "0"))
RECORD_FPS_PROBE = int(os.environ.get("STRETCH4_REC_FPS_PROBE", "20"))
# Write every Nth rendered frame. 1 is every frame.
RECORD_EVERY = int(os.environ.get("STRETCH4_REC_EVERY", "1"))
_REC = {"on": False, "n": 0, "path": None, "annot": None, "rp": None,
        "cam": None, "t0": None, "dropped": 0, "writer": None, "buf": [],
        "size": None}


def _record_camera_matrix():
    """The viewport camera's world transform, as a matrix."""
    import omni.usd as _ousd
    from pxr import Usd as _Usd, UsdGeom as _UsdGeom
    cam_path = _active_camera_path()
    prim = _ousd.get_context().get_stage().GetPrimAtPath(cam_path)
    if not prim.IsValid():
        return None, None
    xf = _UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        _Usd.TimeCode.Default())
    lens = {}
    try:
        _c = _UsdGeom.Camera(prim)
        lens["focal"] = _c.GetFocalLengthAttr().Get()
        lens["haperture"] = _c.GetHorizontalApertureAttr().Get()
        lens["vaperture"] = _c.GetVerticalApertureAttr().Get()
        lens["clip"] = _c.GetClippingRangeAttr().Get()
    except Exception:  # noqa: BLE001
        pass
    return xf, lens


def start_recording():
    """Freeze the current view onto RecordCam and start writing frames."""
    import time as _time
    from pxr import Gf as _Gf, Sdf as _Sdf, UsdGeom as _UsdGeom
    if _REC["on"]:
        print(f"[Teleop] REC already running -> {_REC['path']}", flush=True)
        return
    xf, lens = _record_camera_matrix()
    if xf is None:
        print("[Teleop] REC cannot start: no viewport camera found", flush=True)
        return
    import omni.usd as _ousd
    stage = _ousd.get_context().get_stage()
    cam = _UsdGeom.Camera.Define(stage, "/World/RecordCam")
    prim = cam.GetPrim()
    # One transform op, overwritten each start. Reusing whatever ops happen to
    # exist is how the slot camera code got into trouble; a single matrix op is
    # unambiguous and takes the viewport's pose whatever ops IT uses.
    prim.RemoveProperty("xformOp:transform")
    _UsdGeom.Xformable(prim).ClearXformOpOrder()
    _op = _UsdGeom.Xformable(prim).AddTransformOp()
    _op.Set(_Gf.Matrix4d(xf))
    for _f, _a in (("focal", cam.GetFocalLengthAttr()),
                   ("haperture", cam.GetHorizontalApertureAttr()),
                   ("vaperture", cam.GetVerticalApertureAttr()),
                   ("clip", cam.GetClippingRangeAttr())):
        if lens.get(_f) is not None:
            _a.Set(lens[_f])
    try:
        import omni.replicator.core as rep
        _w, _h = (int(v) for v in RECORD_RES.split(","))
        rp = rep.create.render_product("/World/RecordCam", (_w, _h))
        annot = rep.AnnotatorRegistry.get_annotator("rgb")
        annot.attach([rp])
    except Exception as exc:  # noqa: BLE001
        print(f"[Teleop] REC cannot start ({type(exc).__name__}: {exc})", flush=True)
        return
    os.makedirs(RECORD_DIR, exist_ok=True)
    _path = os.path.join(RECORD_DIR, _time.strftime("%Y%m%d_%H%M%S") + ".mp4")
    _REC.update(on=True, n=0, path=_path, annot=annot, rp=rp, cam="/World/RecordCam",
                t0=_time.time(), dropped=0, writer=None, buf=[], size=None)
    _t = xf.ExtractTranslation()
    print(f"[Teleop] REC START -> {_path}  ({_w}x{_h}, camera frozen at "
          f"[{_t[0]:.2f} {_t[1]:.2f} {_t[2]:.2f}]; move the viewport freely, the "
          f"shot will not follow). {RECORD_STOP_KEY} stops it.", flush=True)


def record_frame():
    """One frame, if recording. Called once per rendered frame."""
    if not _REC["on"]:
        return
    if RECORD_EVERY > 1 and (_REC["n"] + _REC["dropped"]) % RECORD_EVERY:
        _REC["dropped"] += 1
        return
    try:
        import time as _time
        arr = np.asarray(_REC["annot"].get_data())
        if arr.size == 0:
            return
        # RGB from the annotator, BGR into OpenCV.
        frame = np.ascontiguousarray(arr[..., 2::-1].astype(np.uint8))
        if _REC["writer"] is None:
            _REC["buf"].append(frame)
            _REC["size"] = (frame.shape[1], frame.shape[0])
            _n = len(_REC["buf"])
            _want = 2 if RECORD_FPS > 0 else max(RECORD_FPS_PROBE, 2)
            if _n >= _want:
                _dt = max(_time.time() - _REC["t0"], 1e-6)
                _fps = RECORD_FPS if RECORD_FPS > 0 else max(_n / _dt, 1.0)
                import cv2 as _cv2
                _w = _cv2.VideoWriter(_REC["path"],
                                      _cv2.VideoWriter_fourcc(*"mp4v"),
                                      float(_fps), _REC["size"])
                if not _w.isOpened():
                    raise RuntimeError(f"VideoWriter would not open {_REC['path']}")
                for _f in _REC["buf"]:
                    _w.write(_f)
                _REC["writer"] = _w
                _REC["buf"] = []
                print(f"[Teleop] REC encoding at {_fps:.2f} fps "
                      f"({_REC['size'][0]}x{_REC['size'][1]})", flush=True)
            _REC["n"] += 1
            return
        _REC["writer"].write(frame)
        _REC["n"] += 1
    except Exception as exc:  # noqa: BLE001
        _REC["dropped"] += 1
        if _REC["dropped"] in (1, 50):
            print(f"[Teleop] REC frame dropped ({type(exc).__name__}: {exc})",
                  flush=True)


def stop_recording():
    """Stop, and write down the frame rate the frames were actually taken at."""
    import time as _time
    if not _REC["on"]:
        print("[Teleop] REC not running", flush=True)
        return
    _dt = max(_time.time() - _REC["t0"], 1e-6)
    _fps = _REC["n"] / _dt
    try:
        annot, rp = _REC["annot"], _REC["rp"]
        if annot is not None and rp is not None:
            # detach() wants the render product's PATH, not the object it was
            # created as. Handing it the object raises "'HydraTexture' object
            # has no attribute 'split'" -- it tries to treat it as a string --
            # and the annotator is then left attached.
            _p = getattr(rp, "path", None)
            annot.detach([_p] if _p is not None else [rp])
        if rp is not None and hasattr(rp, "destroy"):
            rp.destroy()
    except Exception as exc:  # noqa: BLE001
        print(f"[Teleop] REC teardown ({type(exc).__name__}: {exc})", flush=True)
    try:
        # A recording shorter than the fps probe never opened a writer. Open one
        # now at the measured rate so a two-second clip is still a file.
        if _REC["writer"] is None and _REC["buf"]:
            import cv2 as _cv2
            _w = _cv2.VideoWriter(_REC["path"], _cv2.VideoWriter_fourcc(*"mp4v"),
                                  float(max(_fps, 1.0)), _REC["size"])
            if _w.isOpened():
                for _f in _REC["buf"]:
                    _w.write(_f)
                _REC["writer"] = _w
            _REC["buf"] = []
        if _REC["writer"] is not None:
            _REC["writer"].release()
    except Exception as exc:  # noqa: BLE001
        print(f"[Teleop] REC writer close ({type(exc).__name__}: {exc})", flush=True)
    _sz = 0
    try:
        _sz = os.path.getsize(_REC["path"]) / 1e6
    except OSError:
        pass
    print(f"[Teleop] REC STOP  {_REC['n']} frames in {_dt:.1f}s "
          f"({_fps:.1f} fps, {_REC['dropped']} dropped) -> {_REC['path']} "
          f"({_sz:.1f} MB)", flush=True)
    _REC.update(on=False, annot=None, rp=None, writer=None, buf=[])



def rebuild_contact_guards(env, garment_cloths):
    """Rebuild world-space contact data after static placement changes."""
    stage = env.stage
    _BODY_GEOM.clear()
    _SHIELD.clear()
    _body_surf = _body_vis = None
    for _p in Usd.PrimRange(stage.GetPrimAtPath(env.human.prim_path),
                            Usd.TraverseInstanceProxies()):
        if _p.IsA(UsdGeom.Mesh):
            if _p.GetName() == "CollisionBody":
                _body_surf = _p
            elif _body_vis is None:
                _body_vis = _p
    register_body_collider(_body_surf if _body_surf is not None else _body_vis)

    # # [isaac-5.1.0 compat: cloth shield] -- bake the figure and hand the shield the cloth's edges.
    build_cloth_shield(_body_surf if _body_surf is not None else _body_vis,
                       garment_cloths)
    for _ci, _g in enumerate(env.garments):
        _gm = UsdGeom.Mesh(stage.GetPrimAtPath(_g.garment_mesh_prim_path))
        _cnt = np.asarray(_gm.GetFaceVertexCountsAttr().Get())
        _idx = np.asarray(_gm.GetFaceVertexIndicesAttr().Get())
        _es, _o = set(), 0
        for _c in _cnt:
            _f = _idx[_o:_o + _c]
            _o += _c
            for _i in range(_c):
                _u, _v = int(_f[_i]), int(_f[(_i + 1) % _c])
                _es.add((_u, _v) if _u < _v else (_v, _u))
        _tris, _o = [], 0
        for _c in _cnt:
            _f = [int(v) for v in _idx[_o:_o + _c]]
            _o += _c
            for _k in range(1, _c - 1):
                _tris.append((_f[0], _f[_k], _f[_k + 1]))
        set_shield_edges(_ci, np.array(sorted(_es)), np.asarray(_tris))
    print(f"[Teleop] cloth shield watching {len(env.garments)} garment(s)",
          flush=True)



def initialize_manipulation(env):
    """Shared cloth callbacks/contact setup for teleoperation and policy control."""
    stage = env.stage
    rig = env.rig
    rig2 = env.rig2
    # Capture the tuning while it is still intact, so a later reset can
    # replay it -- see stretch4_tuning.
    stretch4_tuning(rig)
    stretch4_tuning(rig2)

    # One ClothPrim per garment, same reasoning as Teleop_Coat_Stretch4's
    # single-garment version: read the exact mesh path Particle_Garment
    # itself already computed rather than re-discovering it via traversal.
    garment_cloths = [SurfaceClothPrim(prim_paths_expr=g.garment_mesh_prim_path) for g in env.garments]
    garment_faces = []
    for gc, g in zip(garment_cloths, env.garments):
        gc.initialize()
        _mp = stage.GetPrimAtPath(g.garment_mesh_prim_path)
        try:
            _lbl = garment_face_labels(_mp, _to_np(gc.get_world_positions())[0])
            print(f"[Teleop] garment face split: {(_lbl > 0).sum()} front / "
                  f"{(_lbl < 0).sum()} back")
        except Exception as _e:
            print(f"[Teleop] face labelling unavailable ({_e}); "
                  "single-layer grasping disabled for this garment")
            _lbl = None
        garment_faces.append(_lbl)
        mesh_prim = stage.GetPrimAtPath(g.garment_mesh_prim_path)
        apply_edge_mass_boost(gc, mesh_prim, GARMENT_EDGE_MASS_FACTOR)

    # Only one physical keyboard, so only rig["held"] is used as the
    # single shared set of currently-pressed keys -- rig2's own "held"
    # set (from configure_stretch4) is left unused. Safe because the two
    # robots' key maps never overlap (letters/digits vs arrows/numpad).
    # # [isaac-5.1.0 compat: follow at the physics rate]
    #
    # The grab-follow runs here, on the physics step, rather than once per
    # rendered frame at the bottom of drive_robot -- see follow_grabbed for the
    # measurement that moved it. add_physics_callback subscribes straight to
    # PhysX's own step event, so it fires inside simulation_app.update() too,
    # and isaacsim re-subscribes it after a timeline stop, which is what P
    # reset does.
    def _grab_follow(step_size):
        cloth_pre_step(garment_cloths, (rig, rig2), step_size)

    env.world.add_physics_callback("stretch4_grab_follow", _grab_follow)
    # World.add_physics_callback is explicitly pre-step. A second, low-level
    # post-step subscription pins only the jaw anchors after FEM contact has
    # solved and limits any velocity the solve just generated. Keep the handle
    # alive for the duration of main().
    def _grab_post(step_size, _context):
        cloth_post_step(garment_cloths, (rig, rig2), step_size)

    _grab_post_subscription = (
        env.world._physics_context._physics_sim_interface
        .subscribe_physics_on_step_events(
            pre_step=False, order=100, on_update=_grab_post))
    print(f"[Teleop] grab-follow on the physics step: "
          f"{1.0 / max(float(env.world.get_physics_dt()), 1e-9):.0f} Hz "
          f"against a {1.0 / (1.0 / 60.0):.0f} Hz control loop")

    rebuild_contact_guards(env, garment_cloths)

    return garment_cloths, garment_faces, _grab_post_subscription


def main():
    auto_evaluate_setting = os.environ.get('STRETCH4_AUTO_EVALUATE', '0')
    if auto_evaluate_setting not in ('0', '1'):
        raise ValueError('STRETCH4_AUTO_EVALUATE must be 0 or 1')
    auto_evaluate = auto_evaluate_setting == '1'
    training_setting = os.environ.get('STRETCH4_TRAINING_RECORD', '0')
    if training_setting not in ('0', '1'):
        raise ValueError('STRETCH4_TRAINING_RECORD must be 0 or 1')
    full_record_setting = os.environ.get('STRETCH4_FULL_RECORD', '0')
    if full_record_setting not in ('0', '1'):
        raise ValueError('STRETCH4_FULL_RECORD must be 0 or 1')
    env = TeleopTShirtStretch4_Env()
    stage, rig, rig2 = env.stage, env.rig, env.rig2
    garment_cloths, garment_faces, _grab_post_subscription = initialize_manipulation(env)

    from Policy.evaluation_live import EvaluationSession
    import sys
    evaluation = EvaluationSession(env, garment_cloths, (rig, rig2), sys.modules[__name__],
                                   os.environ.get('STRETCH4_EVALUATION_DIR', '/output/evaluation/teleop'))
    pending_evaluation = {'command': None}
    full_recorder = None
    if full_record_setting == '1' or training_setting == '1':
        from Policy.teleop_recording import FullTeleopRecorder
        full_recorder = FullTeleopRecorder(env, garment_cloths, (rig, rig2), sys.modules[__name__],
                                          os.environ.get('STRETCH4_FULL_RECORD_DIR', '/output/full_teleop'))

    held = rig["held"]
    quit_flag = {"quit": False}
    if full_recorder:
        import signal
        def request_graceful_exit(signum, _frame):
            quit_flag['quit'] = True
        signal.signal(signal.SIGINT, request_graceful_exit)
        signal.signal(signal.SIGTERM, request_graceful_exit)

    # state["grabbed"] is (cloth_index, idx, offsets) -- with 4 garments
    # on the table, "what's within reach" has to be checked against all
    # of them and the globally nearest one picked, not just a single
    # fixed cloth. Same logic for both robots, just against their own rig
    # (grasp link) and state.
    def _make_gripper_toggle(rig, label):
        return make_gripper_toggle(rig, label, garment_cloths, garment_faces)

    toggle_gripper1 = _make_gripper_toggle(rig, "robot_0")
    toggle_gripper2 = _make_gripper_toggle(rig2, "robot_1")
    training = None
    if training_setting == '1':
        training_render = os.environ.get('STRETCH4_TRAINING_RENDER', 'deferred')
        if training_render == 'deferred':
            from Policy.training_deferred import DeferredTrainingTeleop as TrainingTeleop
        elif training_render == 'live':
            from Policy.training_teleop import TrainingTeleop
        else:
            raise ValueError('STRETCH4_TRAINING_RENDER must be deferred or live')
        training = TrainingTeleop(env, garment_cloths, (rig, rig2), sys.modules[__name__], full_recorder,
                                  (toggle_gripper1, toggle_gripper2),
                                  os.environ.get('STRETCH4_TRAINING_RECORD_DIR', '/output/policy_datasets'))
        if training_render == 'live':
            # Live training already owns an automatic evaluator for each episode.
            auto_evaluate = False
    auto_evaluation_pending = auto_evaluate

    # world.reset() puts every physics prim on the stage (both robots'
    # articulations, all 4 garments' cloth particles) back to the pose
    # they had right after BaseEnv.reset() ran in __init__ -- i.e. back
    # to the very start. That alone doesn't touch our own plain-python
    # per-rig "state" dicts (lift/arm/wrist targets, grip_pos, grabbed,
    # gripper_closed) or the held-keys set, so those are reset by hand
    # here too, otherwise e.g. a still-held key or a stale "grabbed"
    # reference would carry over into the reset scene.
    def _reset_rig_state(rig):
        state = rig["state"]
        state["lift"] = 0.0
        state["arm"] = 0.0
        state["yaw"] = 0.0
        state["pitch"] = 0.0
        state["roll"] = 0.0
        state["grip_pos"] = GRIPPER_OPEN
        state["base_fwd"] = 0.0
        state["base_strafe"] = 0.0
        state["base_turn"] = 0.0
        state["grabbed"] = None
        state["gripper_closed"] = False

    pending_reset = {"want": False}

    def do_reset():
        """Ask the main loop to reset; do NOT reset from here.

        # [isaac-5.1.0 compat: deferred reset]

        This runs inside a carb keyboard callback, which Kit dispatches from
        within its own input processing. Calling env.reset() -> world.reset()
        there tears the PhysX simulation view down and rebuilds it in the middle
        of that dispatch. On 5.1.0 the app does not survive it: the very next
        loop iteration logs "Physics Simulation View is not created yet" for
        get_applied_actions / apply_action / get_linear_velocities and then shuts
        down -- with no Python traceback, so it just looks like the window
        vanished.

        Setting a flag instead lets the reset happen at the top of the main
        loop, between steps, where tearing down the view is safe.
        """
        pending_reset["want"] = True

    # Slot presses are deferred to the main loop for the same reason resets are
    # (see do_reset): this runs inside Kit's own input dispatch, and a load
    # writes joint state and every cloth particle on the stage. One request at a
    # time -- a second key pressed in the same frame is ignored rather than
    # queued, since queueing two loads only ever means the second one wins.
    pending_slot = {"key": None, "force_save": False}
    clear_armed = {"frame": None}

    def request_slot(key, force_save=False):
        if pending_slot["key"] is None:
            pending_slot["key"] = key
            pending_slot["force_save"] = force_save

    def service_slot(frame):
        """Run whatever slot key was pressed. Called from the main loop."""
        key = pending_slot["key"]
        if key is None:
            return False
        pending_slot["key"] = None
        force_save = pending_slot["force_save"]

        if key == STATE_CLEAR_KEY:
            armed = clear_armed["frame"]
            if armed is not None and frame - armed <= STATE_CLEAR_CONFIRM_FRAMES:
                clear_armed["frame"] = None
                filled = [k for k in STATE_SLOT_KEYS if state_slot_exists(k)]
                removed = clear_state_slots()
                print(f"[Teleop] cleared {removed} saved slot(s): "
                      f"{' '.join(filled) if filled else 'none were filled'}")
            else:
                clear_armed["frame"] = frame
                filled = [k for k in STATE_SLOT_KEYS if state_slot_exists(k)]
                print(f"[Teleop] press {STATE_CLEAR_KEY} again within "
                      f"{STATE_CLEAR_CONFIRM_FRAMES} frames to DELETE every saved "
                      f"slot ({' '.join(filled) if filled else 'none are filled'})")
            return False

        clear_armed["frame"] = None
        rigs = (rig, rig2)
        if state_slot_exists(key) and not force_save:
            if training:
                training.boundary('checkpoint_load')
            evaluation.finish(reason='checkpoint_load', valid=False, take_final=False)
            if full_recorder:
                full_recorder.begin_discontinuity('load_' + key)
            loaded = load_state_slot(key, garment_cloths, rigs)
            if full_recorder:
                full_recorder.end_discontinuity('load_' + key)
            if training and training.sensors:
                training.sensors.reset_episode()
            return loaded
        if force_save and state_slot_exists(key):
            print(f"[Teleop] {key} overwriting the state already in that slot")
        save_state_slot(key, garment_cloths, rigs)
        return False

    def on_keyboard_event(event, *_args, **_kwargs):
        # Carb CHARACTER events carry text, whereas key events carry enums.
        # Recording sees both; text must not be treated as a KeyboardInput.
        name = event.input if isinstance(event.input, str) else event.input.name
        if full_recorder:
            full_recorder.event('keyboard', key=name, event_type=event.type.name,
                                modifiers=int(getattr(event, 'modifiers', 0)))
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if name == "ESCAPE":
                quit_flag["quit"] = True
                return True
            if name in ('F6', 'F7'):
                pending_evaluation['command'] = name
                return True
            if name == RESET_KEY:
                do_reset()
                return True
            if name == RECORD_START_KEY:
                start_recording()
                return True
            if name == RECORD_STOP_KEY:
                stop_recording()
                return True
            if name in STATE_SLOT_KEYS:
                # SHIFT+Fn saves over a filled slot instead of loading it --
                # otherwise a slot can only be re-used by clearing all of them.
                request_slot(name, bool(getattr(event, "modifiers", 0) & _SHIFT_FLAG))
                return True
            if name == STATE_CLEAR_KEY:
                request_slot(name)
                return True
            if name in ROBOT1_GRIP_KEYS:
                training.toggle(0) if training else toggle_gripper1()
                return True
            if name in ROBOT2_GRIP_KEYS:
                training.toggle(1) if training else toggle_gripper2()
                return True
            if name not in held:
                print(f"[Teleop] key down: {name}")
            held.add(name)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            held.discard(name)
        return True

    # A handful of keys we use (or that are just easy to hit by accident)
    # collide with Kit's own default global hotkeys, which fire
    # independently of whatever our own keyboard subscription below does
    # -- confirmed for SPACE (hotkey_ext_id="omni.kit.widget.toolbar",
    # action_id="toolbar::play": pauses the whole timeline, so the robot
    # and everything else looked frozen) and now F too ("is that an Isaac Sim
    # default key?" -- yes: two separate registrations, isaacsim.gui.menu's
    # "focus_prim" is the viewport-snapping one, omni.kit.window.
    # material_graph's "focus_on_nodes" is presumably harmless here but
    # disabled anyway for consistency). Disable each explicitly so the
    # key only does what our own handler says.
    hotkey_registry = get_hotkey_registry()

    def _disable_kit_hotkey(hotkey_ext_id, key):
        hk = hotkey_registry.get_hotkey(hotkey_ext_id, key)
        if hk is not None:
            hotkey_registry.disable_hotkey(hk)
            print(f"[Teleop] disabled Kit's own {hotkey_ext_id}/{key} hotkey ({hk.action_id})")
        else:
            print(f"[Teleop] WARNING: could not find Kit hotkey {hotkey_ext_id}/{key} to disable")

    _disable_kit_hotkey("omni.kit.widget.toolbar", "SPACE")
    _disable_kit_hotkey("omni.kit.menu.utils", "F")
    _disable_kit_hotkey("omni.kit.window.material_graph", "F")

    # The slot keys, same treatment. Only F1 and F2 are actually claimed on
    # 5.1.0 (OpenRefGuide, menu_rename_prim_dialog, and the content browser's
    # Rename), but the loop covers every slot key against both extensions that
    # register F-keys at all, so a Kit update that moves one does not quietly
    # hand a slot key back to a dialog. Quiet on a miss -- unlike the three
    # above, most of these are EXPECTED to be unclaimed.
    for _slot_key in (STATE_SLOT_KEYS + (STATE_CLEAR_KEY,)
                      + (RECORD_START_KEY, RECORD_STOP_KEY, 'F6', 'F7')):
        for _ext in ("omni.kit.menu.utils", "omni.kit.window.content_browser"):
            _hk = hotkey_registry.get_hotkey(_ext, _slot_key)
            if _hk is not None:
                hotkey_registry.disable_hotkey(_hk)
                print(f"[Teleop] disabled Kit's own {_ext}/{_slot_key} hotkey "
                      f"({_hk.action_id})")

    appwindow = omni.appwindow.get_default_app_window()
    input_iface = carb.input.acquire_input_interface()
    keyboard = appwindow.get_keyboard()
    print(f"[Teleop] appwindow={appwindow} keyboard={keyboard}")
    sub = input_iface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)

    print(f"""
[Teleop] Robot 1 -- W/S lift up/down   A/D arm retract/extend   Q/E wrist yaw   R/V wrist pitch   Z/C wrist roll
[Teleop] Robot 1 -- I/K base fwd/back   J/L base left/right   U/O base turn   0/SPACE to grab (fingertip pinch) / release
[Teleop] Robot 2 -- Numpad 8/2 lift up/down   Numpad 4/6 arm retract/extend   Numpad 7/9 wrist yaw
[Teleop] Robot 2 -- Numpad +/- wrist pitch   Numpad 1/3 wrist roll   Arrow Up/Down base fwd/back
[Teleop] Robot 2 -- Arrow Left/Right base left/right   Numpad divide/multiply base turn   Numpad 0/Enter to grab (fingertip pinch) / release
[Teleop] {RESET_KEY} reset to start   ESC quit
[Teleop] Evaluation -- F6 start scoring this attempt; F7 finish and save score (P/slot LOAD invalidates it)
[Teleop] State slots -- {" ".join(STATE_SLOT_KEYS)}: press an empty slot to SAVE the whole scene (camera included), a filled one to LOAD it
[Teleop] State slots -- SHIFT+key overwrites a filled slot   {STATE_CLEAR_KEY} twice deletes every slot   saved under {STATE_DIR}
[Teleop] State slots -- filled right now: {" ".join(k for k in STATE_SLOT_KEYS if state_slot_exists(k)) or "none"}
""")

    # Once every garment, robot, collider, callback, viewport and input binding
    # is live, make startup itself a recoverable checkpoint. F1 is deliberately
    # overwritten on every launch: it means "this run's clean initial state",
    # while later user saves in F2..F5 (or Shift+F1) naturally become newer.
    if save_state_slot("F1", garment_cloths, (rig, rig2)):
        print("[Teleop] initial fully-loaded state automatically saved to F1")
    if STATE_LOAD_SLOT:
        if STATE_LOAD_SLOT in STATE_SLOT_KEYS and state_slot_exists(STATE_LOAD_SLOT):
            request_slot(STATE_LOAD_SLOT)
        else:
            print(f'[Teleop] startup slot {STATE_LOAD_SLOT} is unavailable; keeping the initial scene')

    dt = 1.0 / 60.0
    _frame = 0

    # Everything that was previously written directly against `rig` is
    # now parameterized so the exact same per-frame update can drive
    # either robot -- base accel/turn, arm/wrist/lift/gripper joint
    # targets, the tilt-corrected base velocity drive, and grab-follow.
    def _drive_robot(rig, keymap):
        drive_robot(rig, keymap, held, garment_cloths, dt, _frame)

    # Only non-finite state pauses the world. Finite high speed is handled by
    # the preventive post-step limiter and is never a rollback condition.
    paused = False
    # Keep a finite-speed streak for useful diagnostics only.
    blowup_streak = 0
    # 3 -> 6 alongside EXPLOSION_VELOCITY_THRESHOLD's own increase, per
    # "the safeguard is too sensitive" -- 6 checks 5 frames apart is ~0.5s of
    # sustained high readings, not just a couple of forceful frames.
    BLOWUP_STREAK_TO_ACT = 6
    robots_for_watchdog = [rig["robot"], rig2["robot"]]
    try:
        while simulation_app.is_running() and (not quit_flag["quit"] or (training and training.inflight)):
            _frame += 1

            # Deferred reset (see do_reset). Done here, before anything touches
            # the articulations this frame, so the physics view is rebuilt while
            # nothing holds a stale handle to it. The cloth views have to be
            # re-initialized afterwards for the same reason, and the blowup
            # history is dropped because those snapshots describe the pre-reset
            # world.
            if pending_reset["want"] and (not training or not training.inflight):
                pending_reset["want"] = False
                if training:
                    training.boundary('scene_reset')
                evaluation.finish(reason='scene_reset', valid=False, take_final=False)
                auto_evaluation_pending = auto_evaluate
                if full_recorder:
                    full_recorder.begin_discontinuity('reset')
                # Before env.reset(), not after: the follow now runs from a
                # physics callback, and world.reset() steps physics while the
                # cloth views are being rebuilt. A grab left set across that
                # would have the callback reach into a view that no longer
                # exists. _reset_rig_state clears it again below anyway.
                rig["state"]["grabbed"] = None
                rig2["state"]["grabbed"] = None
                from Env_Config.Garment.NativeGrasp import clear_native_grasp
                clear_native_grasp(rig, stage)
                clear_native_grasp(rig2, stage)
                for _rig in (rig, rig2):
                    _rig["state"].pop("_base_drive_target", None)
                    _rig["state"].pop("_compliance_history", None)
                env.reset()
                # world.reset() rebuilds the PhysX simulation view, but the new
                # one is not live until the app has pumped at least once.
                # Driving the articulations before that logs "Physics Simulation
                # View is not created yet" and then takes the whole app down --
                # and because Kit runs with --/app/fastShutdown=True, the process
                # dies before Python can print the traceback, so it just looks
                # like the window vanished. Verified headlessly: with updates
                # pumped after the reset, every drive call succeeds.
                simulation_app.update()
                for _rig in (rig, rig2):
                    _rig["robot"].initialize()
                    # The rebuilt articulation comes back on the asset's
                    # stock gains; put the tuned ones back on it.
                    stretch4_tuning(_rig, restore=True)
                for _gc in garment_cloths:
                    _gc.initialize()
                _reset_rig_state(rig)
                _reset_rig_state(rig2)
                held.clear()
                blowup_streak = 0
                paused = False
                env.world.play()
                if full_recorder:
                    full_recorder.end_discontinuity('reset')
                if training and training.sensors:
                    training.sensors.reset_episode()
                print("[Teleop] reset to start")
                continue

            # Slot save/load, here rather than in the keyboard callback. A load
            # invalidates the blowup history for the same reason a reset does --
            # those snapshots describe a world that no longer exists -- and the
            # frame is given up afterwards so nothing else writes over what was
            # just restored.
            if pending_slot["key"] is not None and (not training or not training.inflight):
                if service_slot(_frame):
                    auto_evaluation_pending = auto_evaluate
                    blowup_streak = 0
                    paused = False
                    held.clear()
                    env.world.play()
                    continue

            if auto_evaluation_pending and not paused:
                auto_evaluation_pending = False
                evaluation.start({'mode': 'teleop_auto', 'garment_spawn': env.garment_spawn,
                                  'human_spawn': env.human_spawn,
                                  'source_commit': os.environ.get('PHYRC_SOURCE_COMMIT'),
                                  'source_dirty': os.environ.get('PHYRC_SOURCE_DIRTY')})

            if pending_evaluation['command'] is not None:
                command = pending_evaluation['command']
                pending_evaluation['command'] = None
                if command == 'F7':
                    evaluation.finish()
                elif evaluation.active:
                    print('[Evaluation] Already running; press F7 to finish.', flush=True)
                else:
                    try:
                        evaluation.start({'mode': 'teleop', 'garment_spawn': env.garment_spawn,
                                          'human_spawn': env.human_spawn,
                                          'source_commit': os.environ.get('PHYRC_SOURCE_COMMIT'),
                                          'source_dirty': os.environ.get('PHYRC_SOURCE_DIRTY')})
                    except Exception as exc:
                        print(f'[Evaluation] Cannot start: {exc}', flush=True)

            if _frame % 120 == 0:
                _p1, _ = rig["robot"].get_world_pose()
                _p2, _ = rig2["robot"].get_world_pose()
                print(f"[Teleop] frame {_frame} held={held or '{}'} "
                      f"follow={FOLLOW_CALLS[0]} "
                      f"robot_0_pos={_to_np(_p1)} robot_1_pos={_to_np(_p2)}")

            if not paused:
                commands = training.before_control(held) if training else None
                if full_recorder:
                    full_recorder.event('control', control_frame=_frame, held_keys=sorted(held), control_dt_s=dt)
                # [control algorithms: one measurement per cloth per control tick]
                from Env_Config.Garment.ContinuousClothControl import cloth_control_batch
                with cloth_control_batch():
                    if training:
                        for _r, command in zip((rig, rig2), commands):
                            drive_robot(_r, ROBOT1_KEYMAP, set(), garment_cloths, dt, _frame,
                                        commands=command['velocity_commands'])
                    else:
                        _drive_robot(rig, ROBOT1_KEYMAP)
                        _drive_robot(rig2, ROBOT2_KEYMAP)
                if training:
                    training.after_drive()

                # Deferred grab: the fingers must have reached the closed
                # position AND the cloth must have settled after being
                # disturbed by them, before the captured set describes
                # where the fabric actually is.
                for _r in (rig, rig2):
                    _st = _r['state']
                    if not _st.get('pending_grab'):
                        continue
                    if abs(_st['grip_pos'] - GRIPPER_CLOSED) > GRAB_CLOSED_EPS:
                        _st['_settle'] = 0
                        continue
                    _st['_settle'] = _st.get('_settle', 0) + 1
                    if _st['_settle'] >= GRAB_SETTLE_FRAMES:
                        _st['pending_grab'] = False
                        _st['_settle'] = 0
                        _r['attempt_grab']()

            simulation_app.update()
            if full_recorder:
                full_recorder.check()
            if training and not paused:
                training.after_control()
            # After update(), so the frame just rendered is the one written.
            record_frame()

            if ENABLE_BLOWUP_WATCHDOG and not paused:
                if _frame % 5 == 0:
                    problem, is_hard = detect_blowup(garment_cloths, robots_for_watchdog)
                    if problem:
                        blowup_streak += 1
                        print(f"[Teleop] !!! {problem} "
                              f"({'immediate' if is_hard else f'{blowup_streak}/{BLOWUP_STREAK_TO_ACT} consecutive checks'})")
                    else:
                        blowup_streak = 0

                    # Finite speed is already limited before the next solve and
                    # never causes recovery. NaN/Inf cannot safely advance, so
                    # recover explicitly from the newest user-visible F-slot
                    # checkpoint rather than from a hidden rolling snapshot.
                    if is_hard:
                        if training:
                            training.boundary('nonfinite_state_recovery', valid=False)
                        evaluation.finish(reason='nonfinite_state_recovery', valid=False, take_final=False)
                        if full_recorder:
                            full_recorder.check()
                            full_recorder.begin_discontinuity('watchdog_recovery')
                        blowup_streak = 0
                        checkpoint = latest_state_slot()
                        if checkpoint is not None:
                            held.clear()
                            print(f"[Teleop] !!! non-finite state detected; restoring newest "
                                  f"checkpoint {checkpoint}")
                            if not load_state_slot(checkpoint, garment_cloths, (rig, rig2)):
                                print("[Teleop] !!! checkpoint restore failed; paused. "
                                      "Press P to reset the scene.")
                                env.world.pause()
                                paused = True
                        else:
                            print("[Teleop] !!! non-finite state detected but no checkpoint "
                                  "exists; paused. Press P to reset the scene.")
                            env.world.pause()
                            paused = True
                        if full_recorder:
                            full_recorder.end_discontinuity('watchdog_recovery')
    finally:
        failure = sys.exc_info()[1]
        if failure is not None:
            # Kit fast shutdown can exit before Python prints a pending error.
            import traceback
            traceback.print_exception(type(failure), failure, failure.__traceback__)
        try:
            if training:
                training.close(complete=sys.exc_info()[0] is None)
        finally:
            if full_recorder:
                full_recorder.close(reason='normal_exit' if sys.exc_info()[0] is None else 'exception',
                                    capture_final=simulation_app.is_running())
        evaluation.finish(reason='gui_closed' if failure is None else 'exception',
                          valid=failure is None, take_final=simulation_app.is_running())
        input_iface.unsubscribe_to_keyboard_events(keyboard, sub)
        simulation_app.close(exit_code=1 if failure is not None else 0)


if __name__ == "__main__":
    main()
