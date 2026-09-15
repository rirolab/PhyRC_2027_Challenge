"""Phase 1 score accounting from timestamped simulator measurements.

Source: Phase 1 proposal plus the user's short-sleeve/pickup revision.
No Isaac imports are required.
This module scores measurements; it does not infer dressing from proximity or
gripper commands. See docs/PHASE1_EVALUATION.md for the measurement contract and
the implementation choices that the proposal leaves unspecified.
"""
from dataclasses import asdict, dataclass
import math
from statistics import fmean


ARMS = ('left', 'right')
STAGES = ('pickup', 'first_sleeve', 'opposite_shoulder', 'second_sleeve')
EPS = 1e-8
RATE_RULE = 'pickup-excluded-last-award-v9'
SLEEVE_FIELDS = {'upper_arm_sleeve_covered', 'hand_out_of_sleeve'}
QUALITY_FIELDS = {'upper_arm_coverage', 'neck_out', 'front_facing'}
UPPER_ARM_FULL_CREDIT_FRACTION = 0.35


def number(value, name, minimum=0.0):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a number')
    value = float(value)
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f'{name} must be finite and >= {minimum}')
    return value


def boolean(value, name):
    if not isinstance(value, bool):
        raise ValueError(f'{name} must be a JSON/Python boolean')
    return value


@dataclass(frozen=True)
class ScoringConfig:
    # 20 Hz matches DressingEnv. Physics-step recording can use 1/240 instead.
    max_sample_gap_s: float = 0.05
    time_limit_s: float | None = None
    simultaneous_first_arm: str = 'left'

    def __post_init__(self):
        if number(self.max_sample_gap_s, 'max_sample_gap_s') <= 0:
            raise ValueError('max_sample_gap_s must be positive')
        if self.time_limit_s is not None and number(self.time_limit_s, 'time_limit_s') <= 0:
            raise ValueError('time_limit_s must be positive')
        if self.simultaneous_first_arm not in ARMS:
            raise ValueError('simultaneous_first_arm must be left or right')


def validate_sample(sample):
    """Reject missing measurements instead of silently substituting zero."""
    if not isinstance(sample, dict):
        raise ValueError('Each sample must be an object')
    try:
        t = number(sample['time_s'], 'time_s')
        holding = sample['gripper_holding']
        if not isinstance(holding, (list, tuple)) or len(holding) != 2:
            raise ValueError('gripper_holding must contain exactly two booleans')
        holding = [boolean(v, 'gripper_holding') for v in holding]
        clear = boolean(sample['garment_lifted_clear'], 'garment_lifted_clear')
        lifted = sample.get('gripper_lifted', [clear, clear])
        if not isinstance(lifted, (list, tuple)) or len(lifted) != 2:
            raise ValueError('gripper_lifted must contain exactly two booleans')
        lifted = [boolean(v, 'gripper_lifted') for v in lifted]
        complete = boolean(sample.get('dressing_complete', False), 'dressing_complete')
        quality = {}
        if QUALITY_FIELDS.intersection(sample):
            if not QUALITY_FIELDS <= sample.keys():
                raise ValueError('v6 requires upper_arm_coverage, neck_out and front_facing together')
            upper = sample['upper_arm_coverage']
            if not isinstance(upper, dict) or set(upper) != set(ARMS):
                raise ValueError('upper_arm_coverage must contain left/right fractions')
            upper = {a: number(upper[a], 'upper_arm_coverage.' + a) for a in ARMS}
            if any(v > 1 for v in upper.values()):
                raise ValueError('upper_arm_coverage must be within [0,1]')
            quality = dict(upper_arm_coverage=upper, neck_out=boolean(sample['neck_out'], 'neck_out'),
                           front_facing=boolean(sample['front_facing'], 'front_facing'))
            complete &= quality['neck_out'] and quality['front_facing']
        if SLEEVE_FIELDS.intersection(sample):
            if not SLEEVE_FIELDS <= sample.keys() or not QUALITY_FIELDS <= sample.keys():
                raise ValueError('v7 requires sleeve coverage, hand exit, neck and front evidence together')
            for key in SLEEVE_FIELDS:
                if not isinstance(sample[key], dict) or set(sample[key]) != set(ARMS):
                    raise ValueError(f'{key} must contain exactly left/right booleans')
                quality[key] = {a: boolean(sample[key][a], key + '.' + a) for a in ARMS}
            complete &= all(quality['hand_out_of_sleeve'].values())
        if 'neck_passed' in sample:
            if not SLEEVE_FIELDS <= sample.keys() or not QUALITY_FIELDS <= sample.keys():
                raise ValueError('v8 neck passage requires all sleeve and quality measurements')
            quality['neck_passed'] = boolean(sample['neck_passed'], 'neck_passed')
            if quality['neck_passed'] != quality['neck_out']:
                raise ValueError('neck_passed must match the current geometric neck_out evidence')
        if 'rate_rule' in sample:
            if sample['rate_rule'] != RATE_RULE or 'neck_passed' not in quality or 'first_contact_time_s' not in sample:
                raise ValueError('v9 rate rule requires v8 neck and contact-timed measurements')
            quality['rate_rule'] = RATE_RULE
        mappings = {}
        for key in ('wrist_in_sleeve', 'garment_beyond_shoulder', 'arm_coverage'):
            if not isinstance(sample[key], dict) or set(sample[key]) != set(ARMS):
                raise ValueError(f'{key} must have exactly left and right keys')
            if key == 'arm_coverage':
                mappings[key] = {arm: number(sample[key][arm], f'{key}.{arm}') for arm in ARMS}
                if any(v > 1 for v in mappings[key].values()):
                    raise ValueError('arm_coverage uses fractions [0,1], not percentages [0,100]')
            else:
                mappings[key] = {arm: boolean(sample[key][arm], f'{key}.{arm}') for arm in ARMS}
        clock = {}
        if 'first_contact_time_s' in sample:
            first = sample['first_contact_time_s']
            if first is not None:
                first = number(first, 'first_contact_time_s')
                if first > t + EPS:
                    raise ValueError('First contact cannot be in the future')
            clock['first_contact_time_s'] = first
            if 'physics_tick' in sample:
                tick, first_tick = sample['physics_tick'], sample.get('first_contact_tick')
                dt = number(sample.get('physics_dt_s'), 'physics_dt_s')
                if type(tick) is not int or tick < 0 or dt <= 0 or abs(tick * dt - t) > EPS:
                    raise ValueError('Physics tick/dt does not match sample time')
                if first is None:
                    if first_tick is not None:
                        raise ValueError('Contact tick without contact time')
                elif (type(first_tick) is not int or not 0 <= first_tick <= tick
                      or abs(first_tick * dt - first) > EPS):
                    raise ValueError('Contact tick does not match contact time')
                clock.update(physics_tick=tick, first_contact_tick=first_tick, physics_dt_s=dt)
        return dict(time_s=t, gripper_holding=holding, garment_lifted_clear=clear,
                    gripper_lifted=lifted, dressing_complete=complete, **mappings, **clock, **quality)
    except KeyError as exc:
        raise ValueError(f'Missing measurement: {exc.args[0]}') from exc


class Phase1Scorer:
    """Accumulate milestones and best arm progress; latch complete dressing.

    At least two samples are required, starting at episode time zero. Pickup
    requires the SAME gripper to hold a lifted grasp region for >=3 s.
    Changing grippers does not splice two shorter holds into a qualifying hold.
    """
    def __init__(self, config=None):
        self.config = config or ScoringConfig()
        self.reset()

    def reset(self):
        self.last = None
        self.rate_rule = None
        self.last_rate_points = 0.0
        self.last_score_award_time_s = None
        self.last_score_award_tick = None
        self.sample_count = 0
        self.hold_started = [None, None]
        self.first_arm = None
        self.awarded_at = {stage: None for stage in STAGES}
        self.best_coverage = {arm: 0.0 for arm in ARMS}
        self.completion_started = None
        self.dressing_completed_at = None
        self.success_evaluated = True
        self.quality_mode = None
        self.best_upper = {a: 0.0 for a in ARMS}
        self.quality_started = {'neck': None, 'front': None}
        self.quality_awarded = {'neck': None, 'front': None}

    def update(self, sample):
        completion_measured = isinstance(sample, dict) and 'dressing_complete' in sample
        sample = validate_sample(sample)
        quality_mode = 8 if 'neck_passed' in sample else (7 if 'upper_arm_sleeve_covered' in sample else (6 if 'upper_arm_coverage' in sample else 0))
        if self.quality_mode is not None and self.quality_mode != quality_mode:
            raise ValueError('Cannot mix legacy coverage, v6 quality v7 sleeve and v8 neck passage measurements in an episode')
        if self.last is not None and self.rate_rule != sample.get('rate_rule'):
            raise ValueError('Cannot mix rate rules within an episode')
        self.rate_rule = sample.get('rate_rule')
        self.quality_mode = quality_mode
        t = sample['time_s']
        if self.last is None:
            if abs(t) > EPS:
                raise ValueError('The first measurement must be at episode time 0')
        else:
            gap = t - self.last['time_s']
            if ('first_contact_time_s' in sample) != ('first_contact_time_s' in self.last):
                raise ValueError('Cannot change timing basis during an episode')
            previous_contact = self.last.get('first_contact_time_s')
            if previous_contact is not None and sample['first_contact_time_s'] != previous_contact:
                raise ValueError('First contact time must remain latched')
            if previous_contact is None and sample.get('first_contact_time_s') is not None:
                if sample['first_contact_time_s'] <= self.last['time_s']:
                    raise ValueError('Contact was omitted from an earlier sample')
            if gap <= 0:
                raise ValueError('Sample times must strictly increase')
            if gap > self.config.max_sample_gap_s + EPS:
                raise ValueError('Missing samples: gap exceeds max_sample_gap_s')
        if self.config.time_limit_s is not None and t > self.config.time_limit_s + EPS:
            raise ValueError('Measurement exceeds time limit; stop sampling at the deadline')

        self.success_evaluated &= completion_measured
        for i, holding in enumerate(sample['gripper_holding']):
            if holding and sample['gripper_lifted'][i]:
                if self.hold_started[i] is None:
                    self.hold_started[i] = t
                if t - self.hold_started[i] >= 3.0 - EPS:
                    self._award('pickup', t)
            else:
                self.hold_started[i] = None

        wrists = sample['wrist_in_sleeve']
        exited = {a: wrists[a] and sample['hand_out_of_sleeve'][a] for a in ARMS} if self.quality_mode >= 7 else wrists
        if self.first_arm is None:
            entered = [arm for arm in ARMS if exited[arm]]
            if entered:
                self.first_arm = (entered[0] if len(entered) == 1
                                  else self.config.simultaneous_first_arm)
                self._award('first_sleeve', t)
        if self.first_arm is not None:
            other = 'right' if self.first_arm == 'left' else 'left'
            if sample['garment_beyond_shoulder'][other]:
                self._award('opposite_shoulder', t)
            if exited[other]:
                self._award('second_sleeve', t)
        if self.first_arm is not None:
            for arm in ARMS:
                self.best_coverage[arm] = max(self.best_coverage[arm], sample['arm_coverage'][arm])
        # Require a short continuous confirmation, rather than one noisy frame.
        if sample['dressing_complete'] and all(wrists.values()):
            if self.completion_started is None:
                self.completion_started = t
            if t - self.completion_started >= 0.5 - EPS and self.dressing_completed_at is None:
                self.dressing_completed_at = t
        else:
            self.completion_started = None
        if self.quality_mode:
            for arm in ARMS:
                if wrists[arm]:
                    evidence = sample['upper_arm_sleeve_covered'][arm] if self.quality_mode >= 7 else sample['upper_arm_coverage'][arm]
                    self.best_upper[arm] = max(self.best_upper[arm], evidence)
            for key, evidence in [('neck', sample['neck_out']),
                                  ('front', sample['neck_out'] and sample['front_facing'])]:
                if evidence:
                    if self.quality_started[key] is None:
                        self.quality_started[key] = t
                    required_hold = 0.0 if self.quality_mode == 8 and key == 'neck' else .5
                    if t - self.quality_started[key] >= required_hold - EPS and self.quality_awarded[key] is None:
                        self.quality_awarded[key] = t
                else:
                    self.quality_started[key] = None
        self.last = sample
        self.sample_count += 1
        points = self.points()
        rate_points = points['raw_points'] - points['score_items']['pickup']['points']
        if rate_points > self.last_rate_points + EPS:
            self.last_score_award_time_s = t
            self.last_score_award_tick = sample.get('physics_tick')
        self.last_rate_points = rate_points
        return points

    def _award(self, stage, t):
        if self.awarded_at[stage] is None:
            self.awarded_at[stage] = t

    def points(self):
        breakdown = {stage: 5.0 if t is not None else 0.0
                     for stage, t in self.awarded_at.items()}
        n = m = 0.0
        if self.first_arm is not None:
            other = 'right' if self.first_arm == 'left' else 'left'
            n = self.best_coverage[self.first_arm]
            m = self.best_coverage[other]
        complete = self.dressing_completed_at is not None
        breakdown['first_arm_coverage'] = 10.0 if complete else 10.0 * n
        breakdown['second_arm_coverage'] = 20.0 if complete else 20.0 * m
        overall = breakdown['first_arm_coverage'] + breakdown['second_arm_coverage']
        items = {stage: {'points': breakdown[stage], 'max_points': 5.0,
                         'achieved': self.awarded_at[stage] is not None,
                         'achieved_at_episode_s': self.awarded_at[stage]} for stage in STAGES}
        items['overall_dressing'] = {
            'points': overall, 'max_points': 30.0,
            'first_arm': self.first_arm,
            'second_arm': None if self.first_arm is None else ('right' if self.first_arm == 'left' else 'left'),
            'n': n, 'm': m, 'coverage_basis': 'best measured fraction per arm',
            'first_arm_points': breakdown['first_arm_coverage'], 'max_first_arm_points': 10.0,
            'second_arm_points': breakdown['second_arm_coverage'], 'max_second_arm_points': 20.0,
            'measured_coverage_points': 10.0 * n + 20.0 * m,
            'full_dressing_override': complete,
        }
        result = {'raw_points': sum(breakdown.values()), 'max_raw_points': 50.0,
                'score_items': items, 'score_breakdown_version': 'separate-dressing-items-v4',
                'overall_dressing_points': overall,
                'max_overall_dressing_points': 30.0,
                'dressing_complete': complete,
                'success_evaluated': self.success_evaluated,
                'success': self.success_evaluated and complete,
                'success_basis': 'confirmed sleeves and neck for 0.5s; achieved during episode',
                'max_score_achieved': sum(breakdown.values()) >= 50.0,
                'coverage_score_overridden': complete,
                'current_arm_coverage': dict(self.last['arm_coverage']) if self.last else dict(self.best_coverage),
                'dressing_completed_at_s': self.dressing_completed_at,
                'pickup_hold_seconds': [0.0 if start is None or self.last is None else
                                         self.last['time_s'] - start for start in self.hold_started],
                'breakdown': breakdown, 'first_arm': self.first_arm,
                'first_arm_coverage': n, 'second_arm_coverage': m,
                'milestone_times_s': dict(self.awarded_at)}
        if self.quality_mode:
            # Replace only the 30-point component. Sleeve-entry milestones stay separate.
            components = {a + '_upper_arm': 5 * (bool(self.best_upper[a]) if self.quality_mode >= 7 else min(self.best_upper[a] / UPPER_ARM_FULL_CREDIT_FRACTION, 1))
                          for a in ARMS}
            components.update(neck=10.0 if self.quality_awarded['neck'] is not None else 0.0,
                              front_orientation=10.0 if self.quality_awarded['front'] is not None else 0.0)
            breakdown = {key: breakdown[key] for key in STAGES}
            breakdown.update(components)
            overall = sum(components.values())
            items['overall_dressing'] = dict(points=overall, max_points=30.0, components=components,
                component_max_points=dict(left_upper_arm=5, right_upper_arm=5, neck=10, front_orientation=10),
                upper_arm_coverage=dict(self.best_upper), upper_arm_full_credit_fraction=UPPER_ARM_FULL_CREDIT_FRACTION,
                current_upper_arm_coverage=dict(self.last['upper_arm_coverage']) if self.last else dict(self.best_upper),
                coverage_basis='shoulder-to-elbow centreline only; forearm excluded',
                neck_confirmed_at_s=self.quality_awarded['neck'], front_confirmed_at_s=self.quality_awarded['front'],
                full_dressing_override=False)
            result.update(raw_points=sum(breakdown.values()), breakdown=breakdown,
                          score_breakdown_version='upper-arm-neck-front-v6', overall_dressing_points=overall,
                          coverage_score_overridden=False, max_score_achieved=sum(breakdown.values()) >= 50.0,
                          success_basis='both sleeves, exposed neck/head and V-neck front for 0.5s; achieved during episode')
            if self.quality_mode >= 7:
                item = items['overall_dressing']
                for key in ('upper_arm_coverage', 'upper_arm_full_credit_fraction', 'current_upper_arm_coverage'):
                    item.pop(key)
                item.update(upper_arm_sleeve_covered={a: bool(self.best_upper[a]) for a in ARMS},
                            current_upper_arm_sleeve_covered=dict(self.last['upper_arm_sleeve_covered']),
                            coverage_basis='binary: upper-arm segment inside its routed sleeve; torso and forearm excluded')
                result['score_breakdown_version'] = 'sleeve-cover-hand-exit-v7'
                result['success_basis'] = 'both hands out of distinct cuffs, exposed neck/head and V-neck front for 0.5s; achieved during episode'
                for key in ('first_sleeve', 'second_sleeve'):
                    items[key]['criterion'] = 'hand exits through its cuff; wrist and distal hand outside'
            if self.quality_mode == 8:
                items['overall_dressing'].update(
                    neck_criterion='neck/head exposed through the collar; binary passage, no dwell',
                    neck_required_hold_s=0.0, front_required_hold_s=0.5)
                result['score_breakdown_version'] = 'instant-neck-overall-v8'
            # Do not expose legacy arm coverage as if it still earned points.
            for key in ('first_arm_coverage', 'second_arm_coverage'):
                result.pop(key)
        return result

    def rate_result(self):
        """v9 cumulative dressing points per time to the latest dressing award.

        Retain full episode duration separately. Pickup cannot advance this clock.
        No dressing award is a zero score; zero-time positive points have no rate.
        """
        first = self.last['first_contact_time_s']
        end = self.last_score_award_time_s
        elapsed = 0.0 if first is None or end is None else max(0.0, end - first)
        if first is not None and self.last_score_award_tick is not None:
            elapsed = max(0, self.last_score_award_tick - self.last['first_contact_tick']) * self.last['physics_dt_s']
        points = self.points()
        numerator = points['raw_points'] - points['score_items']['pickup']['points']
        status = ('no_contact' if first is None else 'no_dressing_points' if numerator <= EPS
                  else 'awaiting_elapsed_time' if elapsed <= EPS else 'scored')
        rate = 0.0 if status == 'no_dressing_points' else numerator / elapsed if status == 'scored' else None
        return dict(rate_rule=RATE_RULE, scoring_revision=RATE_RULE,
                    rate_points=numerator, max_rate_points=45.0,
                    excluded_pickup_points=points['score_items']['pickup']['points'],
                    last_score_award_time_s=end, last_score_award_tick=self.last_score_award_tick,
                    rate_end_basis='last_non_pickup_point_increase',
                    task_time_s=elapsed, elapsed_episode_time_s=self.last['time_s'],
                    contact_elapsed_time_s=0.0 if first is None else self.last['time_s'] - first,
                    first_contact_time_s=first, timing_basis='first_garment_mannequin_contact',
                    score_status=status, final_score=rate, score_unit='points/s')

    def result(self):
        if self.sample_count < 2 or self.last['time_s'] <= 0:
            raise ValueError('Final scoring requires a positive duration and at least two samples')
        result = self.points()
        if self.rate_rule == RATE_RULE:
            return dict(result, **self.rate_result(), sample_count=self.sample_count)
        revision = 'instant-neck-overall-v8' if self.quality_mode == 8 else ('sleeve-cover-hand-exit-v7' if self.quality_mode >= 7 else ('upper-arm-neck-front-v6' if self.quality_mode else 'separate-dressing-items-v5'))
        duration = self.last['time_s']
        if 'first_contact_time_s' in self.last:
            first = self.last['first_contact_time_s']
            elapsed = 0.0 if first is None else max(0.0, duration - first)
            if first is not None and 'physics_tick' in self.last:
                elapsed = (self.last['physics_tick'] - self.last['first_contact_tick']) * self.last['physics_dt_s']
            status = 'no_contact' if first is None else 'awaiting_elapsed_time' if elapsed <= EPS else 'scored'
            return dict(result, task_time_s=elapsed, elapsed_episode_time_s=duration,
                        first_contact_time_s=first, timing_basis='first_garment_mannequin_contact',
                        scoring_revision=revision, score_status=status,
                        final_score=result['raw_points'] / elapsed if status == 'scored' else None,
                        score_unit='points/s', sample_count=self.sample_count)
        return dict(result, task_time_s=duration,
                    scoring_revision=revision, timing_basis='episode_start_legacy',
                    final_score=result['raw_points'] / duration,
                    score_unit='points/s', sample_count=self.sample_count)


def format_score_items(result):
    """One shared display for live teleop, replay and batch/policy evaluation."""
    items = result['score_items']
    labels = [('pickup', 'Pickup'), ('first_sleeve', 'First sleeve'),
              ('opposite_shoulder', 'Opposite shoulder'), ('second_sleeve', 'Second sleeve'),
              ('overall_dressing', 'Overall dressing')]
    parts = [f"{label} {items[key]['points']:.2f}/{items[key]['max_points']:.0f}" for key, label in labels]
    overall = items['overall_dressing']
    if 'components' in overall:
        c = overall['components']
        parts[-1] += f" (upper L {c['left_upper_arm']:.2f}/5, upper R {c['right_upper_arm']:.2f}/5"
        if 'neck' in c:
            parts[-1] += f", neck {c['neck']:.2f}/10, V-neck front {c['front_orientation']:.2f}/10"
        parts[-1] += ')'
    else:
        parts[-1] += (f" (first {overall['first_arm_points']:.2f}/10, "
                      f"second {overall['second_arm_points']:.2f}/20)")
    if overall['full_dressing_override']:
        parts[-1] += ' [complete-dressing override]'
    success = 'UNKNOWN' if not result.get('success_evaluated', False) else 'YES' if result['success'] else 'NO'
    parts.append('Dressing success: ' + success)
    return ' | '.join(parts)


def evaluate_episode(samples, config=None):
    """Evaluation function: final score = (5+5+5+5+10*n+20*m) / seconds.

    Each 5-point term is included only if its milestone was achieved.
    n and m retain best progress; full dressing overrides their score terms.
    """
    scorer = Phase1Scorer(config)
    for sample in samples:
        scorer.update(sample)
    return scorer.result()


def evaluate_submissions(data, config=None):
    """Mean points/s across shared unique seeds; best whole submission counts.

    Mean aggregation is an explicit implementation choice, NOT a formula stated
    in the proposal. Do not select the best seed or mix episodes across retries.
    """
    config = config or ScoringConfig()
    if not isinstance(data, dict) or data.get('schema_version') not in ('phase1-measurements-v1', 'phase1-measurements-v2', 'phase1-measurements-v3', 'phase1-measurements-v4', 'phase1-measurements-v5', 'phase1-measurements-v6', 'phase1-measurements-v7'):
        raise ValueError('Expected schema_version phase1-measurements-v1, v2, v3, v4, v5, v6 or v7')
    submissions = data.get('submissions')
    if not isinstance(submissions, list) or not submissions:
        raise ValueError('At least one submission is required')
    reference_seeds, ids, results = None, set(), []
    for submission in submissions:
        sid = submission.get('submission_id')
        if not isinstance(sid, str) or not sid.strip() or sid in ids:
            raise ValueError('submission_id must be a unique nonempty string')
        ids.add(sid)
        episodes = submission.get('episodes')
        if not isinstance(episodes, list) or len(episodes) < 2:
            raise ValueError('Each submission needs at least two randomized initial-state seeds')
        seeds, scores = set(), []
        for episode in episodes:
            seed = episode.get('seed')
            if type(seed) is not int or seed < 0 or seed in seeds:
                raise ValueError('Each episode needs a unique nonnegative integer seed')
            seeds.add(seed)
            if not isinstance(episode.get('samples'), list):
                raise ValueError('Each episode needs a samples list')
            try:
                if data['schema_version'] not in ('phase1-measurements-v4', 'phase1-measurements-v5', 'phase1-measurements-v6', 'phase1-measurements-v7') and any(
                        isinstance(sample, dict) and QUALITY_FIELDS.intersection(sample) for sample in episode['samples']):
                    raise ValueError('Quality measurements require phase1-measurements-v4')
                if data['schema_version'] not in ('phase1-measurements-v3', 'phase1-measurements-v4', 'phase1-measurements-v5', 'phase1-measurements-v6', 'phase1-measurements-v7') and any(
                        isinstance(sample, dict) and 'first_contact_time_s' in sample for sample in episode['samples']):
                    raise ValueError('Contact-timed traces require v3; do not mix timing bases in a ranking')
                if data['schema_version'] in ('phase1-measurements-v2', 'phase1-measurements-v3', 'phase1-measurements-v4', 'phase1-measurements-v5', 'phase1-measurements-v6', 'phase1-measurements-v7'):
                    for sample in episode['samples']:
                        if not isinstance(sample, dict) or not {'gripper_lifted', 'dressing_complete'} <= sample.keys():
                            raise ValueError('v2 requires gripper_lifted and dressing_complete measurements')
                        if data['schema_version'] in ('phase1-measurements-v3', 'phase1-measurements-v4', 'phase1-measurements-v5', 'phase1-measurements-v6', 'phase1-measurements-v7') and 'first_contact_time_s' not in sample:
                            raise ValueError('v3 requires first_contact_time_s (null until first contact)')
                        if data['schema_version'] in ('phase1-measurements-v4', 'phase1-measurements-v5', 'phase1-measurements-v6', 'phase1-measurements-v7') and not QUALITY_FIELDS <= sample.keys():
                            raise ValueError('v4 requires upper-arm, neck and front measurements; regenerate from raw state')
                for sample in episode['samples']:
                    if data['schema_version'] in ('phase1-measurements-v5', 'phase1-measurements-v6', 'phase1-measurements-v7'):
                        if not SLEEVE_FIELDS <= sample.keys():
                            raise ValueError('v5 requires actual sleeve coverage and hand exit; regenerate from raw states')
                    elif SLEEVE_FIELDS.intersection(sample):
                        raise ValueError('Sleeve/hand evidence requires phase1-measurements-v5')
                for sample in episode['samples']:
                    if (data['schema_version'] in ('phase1-measurements-v6', 'phase1-measurements-v7')) != ('neck_passed' in sample):
                        raise ValueError('v6 requires neck_passed; older schemas cannot contain v8 neck passage evidence')
                for sample in episode['samples']:
                    if (data['schema_version'] == 'phase1-measurements-v7') != ('rate_rule' in sample):
                        raise ValueError('v7 requires rate_rule; old schemas retain their historical rate formula')
                scores.append(dict(seed=seed, **evaluate_episode(episode['samples'], config)))
            except ValueError as exc:
                raise ValueError(f'Submission {sid}, seed {seed}: {exc}') from exc
        if reference_seeds is not None and seeds != reference_seeds:
            raise ValueError('All submissions must use the same seed set')
        reference_seeds = seeds
        for episode in scores:
            # Preserve an undefined measured rate as null; a completed attempt
            # without a denominator contributes zero to ranking, never vanishes.
            episode['ranking_score'] = 0.0 if episode['final_score'] is None else episode['final_score']
            episode['ranking_status'] = (episode['score_status'] if episode['final_score'] is None
                                         else 'scored')
        known = [e for e in scores if e['success_evaluated']]
        results.append({'submission_id': sid, 'episodes': scores,
                        'final_score': fmean(e['ranking_score'] for e in scores),
                        'zero_ranked_episode_count': sum(e['final_score'] is None for e in scores),
                        'success_evaluated_episode_count': len(known),
                        'success_count': sum(e['success'] for e in known),
                        'success_rate': fmean(e['success'] for e in known) if len(known) == len(scores) else None,
                        'mean_raw_points': fmean(e['raw_points'] for e in scores)})
    best = max(results, key=lambda result: result['final_score'])
    contact_timing = data['schema_version'] in ('phase1-measurements-v3', 'phase1-measurements-v4', 'phase1-measurements-v5', 'phase1-measurements-v6', 'phase1-measurements-v7')
    rate_mode = data['schema_version'] == 'phase1-measurements-v7'
    neck_mode = data['schema_version'] in ('phase1-measurements-v6', 'phase1-measurements-v7')
    sleeve_mode = data['schema_version'] in ('phase1-measurements-v5', 'phase1-measurements-v6', 'phase1-measurements-v7')
    quality_mode = data['schema_version'] in ('phase1-measurements-v4', 'phase1-measurements-v5', 'phase1-measurements-v6', 'phase1-measurements-v7')
    return {'schema_version': 'phase1-scores-v9' if rate_mode else 'phase1-scores-v8' if neck_mode else ('phase1-scores-v7' if sleeve_mode else ('phase1-scores-v6' if quality_mode else 'phase1-scores-v5')),
            'rules_source': ('User revision: upper arms 10, exposed neck 10, V-neck front 10; independent milestones 20'
                             if quality_mode else 'Phase 1 proposal with user revision: lifted grasp region, best progress, sleeves plus neck completion'),
            'scoring_revision': RATE_RULE if rate_mode else 'instant-neck-overall-v8' if neck_mode else ('sleeve-cover-hand-exit-v7' if sleeve_mode else ('upper-arm-neck-front-v6' if quality_mode else 'separate-dressing-items-v5')),
            'timing_basis': 'first_garment_mannequin_contact' if contact_timing else 'episode_start_legacy',
            'aggregation': 'arithmetic mean of episode ranking_score; all seeds included',
            'undefined_rate_policy': 'no_contact or zero post-contact duration contributes 0; measured final_score remains null',
            'config': asdict(config), 'submissions': results,
            'best_submission_id': best['submission_id'],
            'final_score': best['final_score'], 'score_unit': 'points/s'}


class Phase1TaskEvaluator:
    """DressingEnv evaluator hook with an explicit trusted measurement callback.

    measure(info) must return validate_sample()'s fields except time_s. It must
    read actual geometry/attachments, not action intentions. The environment
    clock supplies time_s. This adapter does not install geometric detectors.
    """
    def __init__(self, measure, config=None):
        if not callable(measure):
            raise ValueError('measure must be callable')
        self.measure = measure
        self.scorer = Phase1Scorer(config)
        self.started = False

    def _sample(self, info):
        sample = dict(self.measure(info))
        sample['time_s'] = info['episode_time_s']
        return sample

    def reset(self, initial_info):
        self.started = False
        self.scorer.reset()
        self.scorer.update(self._sample(initial_info))
        self.started = True

    def __call__(self, previous_obs, action, next_obs, info):
        if not self.started:
            raise RuntimeError('Call evaluator.reset() before stepping')
        before = self.scorer.points()['raw_points']
        measurement = self._sample(info)
        self.scorer.update(measurement)
        result = self.scorer.result()
        limit = self.scorer.config.time_limit_s
        terminated = limit is not None and info['episode_time_s'] >= limit - EPS
        # Training reward is a raw-point delta; final ranking uses points/s.
        # Success now includes the measured neck/sleeve completion condition.
        result['phase1_score_evaluated'] = True
        return result['raw_points'] - before, terminated, result
