"""Physics-tick score clock, started once by garment/mannequin contact evidence.

This class does not infer contact. Its caller must supply the selected,
explicitly documented contact measurement for EVERY physics tick.
"""
import math


class ContactClock:
    def __init__(self, physics_dt_s):
        self.dt = float(physics_dt_s)
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError('physics_dt_s must be finite and positive')
        self.last_tick = None
        self.first_contact_tick = None

    def update(self, tick, contact):
        if type(tick) is not int or tick < 0:
            raise ValueError('Expected a nonnegative integer physics tick')
        if type(contact) is not bool:
            raise ValueError('Contact evidence must be an explicit boolean')
        if self.last_tick is None:
            if tick != 0:
                raise ValueError('Contact monitoring must start at tick 0')
        elif tick != self.last_tick + 1:
            raise ValueError('Contact monitoring requires every consecutive physics tick')
        if contact and self.first_contact_tick is None:
            self.first_contact_tick = tick
        self.last_tick = tick

    def result(self, raw_points):
        points = float(raw_points)
        if not math.isfinite(points) or points < 0:
            raise ValueError('raw_points must be finite and nonnegative')
        if self.last_tick is None:
            raise ValueError('No contact measurements')
        first = self.first_contact_tick
        elapsed_ticks = 0 if first is None else self.last_tick - first
        elapsed = elapsed_ticks * self.dt
        status = 'no_contact' if first is None else 'awaiting_elapsed_time' if elapsed_ticks == 0 else 'scored'
        return {'timing_basis': 'first_garment_mannequin_contact',
                'first_contact_tick': first,
                'first_contact_time_s': None if first is None else first * self.dt,
                'elapsed_episode_time_s': self.last_tick * self.dt,
                'task_time_s': elapsed,
                'score_status': status,
                'final_score': points / elapsed if elapsed_ticks > 0 else None,
                'score_unit': 'points/s'}
