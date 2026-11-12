"""Fail-closed, observation-driven FSM for *internal* two-arm shirt reorientation.

Offline dry-run only: NO robot action is generated or dispatched. All positive
claims (visual two-arm grasp, safe plan, execution, target orientation) must
come from separate validated modules. Robot gripper CLOSE intent is NOT proof.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Optional


class Phase(str, Enum):
    OBSERVE = 'OBSERVE'
    CHECK_GRASP = 'CHECK_GRASP'
    PLAN_REORIENTATION = 'PLAN_REORIENTATION'
    EXECUTE = 'EXECUTE'
    MONITOR = 'MONITOR'
    VERIFY = 'VERIFY'
    RECOVER = 'RECOVER'
    SUCCESS = 'SUCCESS'
    ABORTED = 'ABORTED'


@dataclass(frozen=True)
class Evidence:
    """Externally generated permitted-observation evidence; None means UNKNOWN.

    valid_cameras: two camera-wise garment states valid now (not grasp proof)
    grasp_visually_supported: independently verified BOTH hands still hold shirt
    orientation_known: semantic orientation estimate in a consistent frame
    plan_safe: validated motion plan available AND passing safety checks
    execution_started/completed: controller/monitor status, not inferred from timestamps
    target_verified: orientation AND handoff requirements checked, not from centroid
    hazard: collision-risk, excessive tension, invalid controller feedback, etc.
    """
    time_s: float
    valid_cameras: bool
    grasp_visually_supported: Optional[bool] = None
    orientation_known: Optional[bool] = None
    plan_safe: Optional[bool] = None
    execution_started: Optional[bool] = None
    execution_completed: Optional[bool] = None
    target_verified: Optional[bool] = None
    hazard: bool = False
    # diagnostic only: never used to pass grasp check
    both_grippers_commanded_closed: Optional[bool] = None

    def __post_init__(self):
        if not isfinite(self.time_s) or self.time_s < 0:
            raise ValueError('time_s must be finite and nonnegative')
        for field in ('valid_cameras','hazard'):
            if type(getattr(self, field)) is not bool:
                raise ValueError(f'{field} must be bool')
        for field in ('grasp_visually_supported','orientation_known','plan_safe',
                      'execution_started','execution_completed','target_verified',
                      'both_grippers_commanded_closed'):
            val = getattr(self, field)
            if val is not None and type(val) is not bool:
                raise ValueError(f'{field} must be bool or None')


@dataclass(frozen=True)
class Settings:
    min_verified_duration_s: float = 0.2  # internal pilot value, NOT PhyRC benchmark
    phase_timeout_s: float = 15.0
    max_recoveries: int = 1

    def __post_init__(self):
        if not isfinite(self.min_verified_duration_s) or self.min_verified_duration_s < 0:
            raise ValueError('invalid verification duration')
        if not isfinite(self.phase_timeout_s) or self.phase_timeout_s <= 0:
            raise ValueError('invalid timeout')
        if self.max_recoveries < 0:
            raise ValueError('invalid recovery budget')


@dataclass(frozen=True)
class Transition:
    at_s: float
    before: str
    after: str
    reason: str


class ReorientationFSM:
    def __init__(self, settings: Settings = Settings()):
        self.cfg = settings
        self.phase = Phase.OBSERVE
        self.entered_s: Optional[float] = None
        self.last_s: Optional[float] = None
        self.verified_since_s: Optional[float] = None
        self.recoveries = 0
        self.transitions: list[Transition] = []

    def _move(self, next_phase: Phase, time_s: float, reason: str):
        if self.phase != next_phase:
            self.transitions.append(Transition(time_s,self.phase.value,next_phase.value,reason))
            self.phase = next_phase
            self.entered_s = time_s
            self.verified_since_s = None

    def _recover(self, t: float, reason: str):
        self._move(Phase.RECOVER,t,reason)

    def step(self, e: Evidence) -> Phase:
        t=e.time_s
        if self.last_s is not None and t <= self.last_s:
            raise ValueError('non-increasing simulation time; reset FSM on episode reset')
        self.last_s=t
        if self.entered_s is None: self.entered_s=t
        if self.phase in (Phase.SUCCESS,Phase.ABORTED): return self.phase
        if e.hazard:
            self._recover(t,'explicit_hazard')
            return self.phase
        if t - self.entered_s > self.cfg.phase_timeout_s:
            self._recover(t,'phase_timeout')
            return self.phase

        if self.phase == Phase.OBSERVE:
            if e.valid_cameras: self._move(Phase.CHECK_GRASP,t,'camera_observations_valid')
        elif self.phase == Phase.CHECK_GRASP:
            if not e.valid_cameras:
                self._recover(t,'lost_perception')
            elif e.grasp_visually_supported is True:
                self._move(Phase.PLAN_REORIENTATION,t,'both_grasps_visually_supported')
            elif e.grasp_visually_supported is False:
                self._recover(t,'visual_grasp_failed')
            # None -> remain here, even if both commanded closed
        elif self.phase == Phase.PLAN_REORIENTATION:
            if e.grasp_visually_supported is not True:
                self._recover(t,'grasp_not_supported_during_planning')
            elif e.orientation_known is True and e.plan_safe is True:
                self._move(Phase.EXECUTE,t,'orientation_and_safe_plan_available')
            elif e.plan_safe is False:
                self._recover(t,'plan_rejected_by_safety')
        elif self.phase == Phase.EXECUTE:
            if e.grasp_visually_supported is not True or not e.valid_cameras:
                self._recover(t,'execution_evidence_lost')
            elif e.plan_safe is not True:
                self._recover(t,'plan_safety_not_confirmed')
            elif e.execution_started is True:
                self._move(Phase.MONITOR,t,'execution_acknowledged')
        elif self.phase == Phase.MONITOR:
            if e.grasp_visually_supported is not True or not e.valid_cameras or e.plan_safe is not True:
                self._recover(t,'monitor_guard_failed')
            elif e.execution_completed is True:
                self._move(Phase.VERIFY,t,'execution_completed_reported')
        elif self.phase == Phase.VERIFY:
            if e.grasp_visually_supported is not True or not e.valid_cameras:
                self._recover(t,'verification_grasp_or_perception_lost')
            elif e.target_verified is False:
                self._recover(t,'target_not_verified')
            elif e.target_verified is True:
                if self.verified_since_s is None: self.verified_since_s=t
                if t-self.verified_since_s >= self.cfg.min_verified_duration_s:
                    self._move(Phase.SUCCESS,t,'target_stable_and_grasp_supported')
            else:
                self.verified_since_s=None
        elif self.phase == Phase.RECOVER:
            # Recovery *requires external acknowledgement* of a fresh validated hold;
            # no guessed motion or automatic regrasp is performed here.
            if e.valid_cameras and e.grasp_visually_supported is True and not e.hazard:
                if self.recoveries >= self.cfg.max_recoveries:
                    self._move(Phase.ABORTED,t,'recovery_budget_exhausted')
                else:
                    self.recoveries += 1
                    self._move(Phase.PLAN_REORIENTATION,t,'recovery_evidence_confirmed')
            elif t-self.entered_s >= self.cfg.phase_timeout_s:
                self._move(Phase.ABORTED,t,'recovery_timeout')
        return self.phase
