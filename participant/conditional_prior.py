"""Scene-conditioned *diagnostic* strategy intent, NOT privileged scene truth.

The role rule is user-asserted for the two mirrored pickup configurations.
The 06-Nov replay provides partial evidence for left-side R0-yield, but did not
visually verify the original block. Never claim a user-supplied scene hint is
inferred from head/wrist RGB-D. Fail closed on missing scene-side evidence.
"""
from __future__ import annotations
import math
import json
from pathlib import Path
import numpy as np

def _load_rules():
    path=Path(__file__).with_name('conditional_strategy_prior.json')
    data=json.loads(path.read_text(encoding='utf-8'))
    if data.get('schema')!='phyrc_v2_conditional_strategy_prior_v1' or data.get('motion_authorized') is not False:
        raise ValueError('Unrecognized or unsafe conditional prior schema')
    rules=data.get('rules',{})
    for side,yielder in (('LEFT',0),('RIGHT',1)):
        r=rules.get(side,{})
        if r.get('yielding_robot')!=yielder or r.get('passing_robot')!=1-yielder:
            raise ValueError('Invalid scene-side role mapping in prior')
    return rules

RULES=_load_rules()

def choose_intent(side, *, origin="UNKNOWN", confidence=None):
    s = str(side or "UNKNOWN").upper()
    if s not in RULES:
        return dict(status="SCENE_UNKNOWN", garment_side="UNKNOWN",
                    role_selected=False, yielding_robot=None, passing_robot=None,
                    turn_preference="UNKNOWN", scene_evidence=origin,
                    explanation="No verified left/right scene observation; keep both role proposals; do not guess.")
    v=dict(RULES[s])
    v.update(status="CONDITIONAL_PRIOR_ONLY", garment_side=s, role_selected=True,
             scene_evidence=origin, scene_confidence=confidence,
             explanation=("LEFT: R0 yields corridor, R1 advances and can rotate CW" if s=="LEFT"
                          else "RIGHT: R1 yields corridor, R0 advances; turn direction unverified"),
             motion_authorized=False)
    return v


def tactical_candidates(start, intent, *, yield_offset_m=.12):
    """Produce explicit paired-base *candidate* phase anchors, never commands.

    Relative pair opening is only a hypothesis and may not correspond to the
    actual table corridor. The coupled 2D planners must reject unknown cells.
    """
    p=np.asarray(start,float).reshape(2,2)
    if not intent.get("role_selected"):
        return []
    y=int(intent['yielding_robot']); passer=int(intent['passing_robot'])
    dist=float(np.linalg.norm(p[1]-p[0]))
    if not math.isfinite(dist) or dist<1e-5: return []
    outward=(p[y]-p[passer])/dist
    via=p.copy();via[y]+=float(yield_offset_m)*outward
    swap=p[::-1].copy()
    return [dict(family="CONDITIONAL_YIELD_PASS_REVIEW",passer=passer,
                 yielding_robot=y,phase_goals=[via,swap],
                 strategy_id=intent['strategy_id'],
                 role_basis="CONDITIONAL_SCENE_PRIOR",
                 target_basis="pair_exchange_proxy_not_manikin_or_garment_goal",
                 full_body_authorized=False)]


def choose_proximity_intent(evidence):
    """Select paired roles from earlier *pre-contact* world-frame distances.

    This rule is intentionally independent of LEFT/RIGHT relative to mannequin:
    garment nearest R0 -> R1 yields; garment nearest R1 -> R0 yields.
    Do not infer table side or rotation direction from this observation.
    """
    if not evidence or evidence.get('status') != 'PROXIMITY_ROLE_HEURISTIC':
        return choose_intent(None, origin='INSUFFICIENT_PREGRASP_HEAD_RGBD')
    near=int(evidence['closest_robot'])
    if near not in (0,1):
        return choose_intent(None, origin='INVALID_CLOSEST_ROBOT')
    return dict(status='PROXIMITY_ROLE_HEURISTIC',
                garment_side='UNKNOWN_RELATIVE_TO_MANNEQUIN',
                role_selected=True, yielding_robot=1-near, passing_robot=near,
                turn_preference='UNVERIFIED',
                strategy_id=f'CLOTH_NEAR_R{near}_R{1-near}_YIELD_REVIEW',
                scene_evidence='PUBLIC_PREGRASP_HEAD_RGBD_AND_BASE_POSE_WORLD',
                scene_confidence='TEMPORAL_CORROBORATION_NOT_SEMANTIC_CERTIFICATION',
                basis=evidence,
                explanation=f'Cloth is nearer R{near} in world XY before contact; R{1-near} yields, R{near} passes. Motion and corridor geometry unverified.',
                full_body_authorized=False, motion_authorized=False)
