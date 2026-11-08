# Evaluation and scoring

Scoring uses the actual state of the shirt, manikin, and grippers in the simulated scene, not how the images look.
The current scoring rule is `pickup-excluded-last-award-v9`.

## Live scoring during teleop

```bash
./run.sh gui --evaluate 1
# Also collect training data:
./run.sh gui --training-record 1 --evaluate 1
```

Evaluation starts automatically before your first control input; there is no need to press F6/F7.
Pressing ESC saves `output/evaluation/teleop/<eval-id>/result.json` and `samples.jsonl`.
Resetting the scene with `P` or restoring a save slot ends the current live attempt as invalid and starts scoring the next attempt.
For fair comparisons, use consecutive attempts with the same reset and stop conditions.

Training recordings also save per-episode evaluation automatically. In the default mode, scoring happens during export after you quit;
add `--evaluate 1` to also see the score while driving. `--training-render live` already includes its own live evaluation.
A live evaluation interrupted by an error does not count as a valid score.

## Re-scoring from a replay

```bash
# Re-score from the raw state log without a display (runs on CPU, no GPU needed)
./run.sh cpu /scripts/evaluate_replay.py \
  /output/full_teleop/<run-id> --output /output/evaluation/replay_check

# Watch the same raw state in Isaac Sim while scoring it
./run.sh replay output/full_teleop/<run-id> \
  --evaluate /output/evaluation/replay_visual
```

Replay scoring reads the recorded physics state, contact geometry, and grasp records. It does not score by playing back an MP4,
and it does not reproduce the exact solver history with fresh physics.
Replay speed and your computer's processing time do not affect scoring times.

## Score items

| Item | Max | Condition |
|---|---:|---|
| pickup | 5 | Lift the grasped region with the same gripper at least 5 cm above where it was grasped, and hold for 3 s of simulation time |
| first_sleeve | 5 | The first hand passes through its sleeve opening and comes out |
| opposite_shoulder | 5 | Part of the shirt passes over the opposite shoulder joint |
| second_sleeve | 5 | The other hand comes out through the other sleeve |
| overall_dressing: left_upper_arm | 5 | The left sleeve covers part of the upper arm |
| overall_dressing: right_upper_arm | 5 | The right sleeve covers part of the upper arm |
| overall_dressing: neck | 10 | Awarded as soon as a proper pass through the neck opening is confirmed |
| overall_dressing: front_orientation | 10 | With the head out, the V-neck front faces the front of the manikin for 0.5 s |
| **Total** | **50** | Includes 30 Overall points |

A sleeve pass requires evidence that both the wrist and the fingertips are out. Two arms through the same sleeve do not count.
Each upper arm is pass/fail for 5 or 0 points, not scaled by coverage length. Torso coverage and forearms are not scored.
The neck item checks that the chest is inside the shirt and the head/neck is properly out through the neck opening. It has no hold time.
V-neck orientation requires the front to be within 45° of forward with at least 2 cm of projected front direction, so the state is clearly distinguishable. Wearing it backwards scores 0 for orientation.

Points, once earned, are kept even if the state changes later.
**Dressing success** means both hands are out of different sleeves, the head/neck is out, and the front/back orientation is correct, all confirmed together for 0.5 s.
50 points and a successful demonstration are not the same thing, so check `success` in the results separately.

## points/s and failures

```text
numerator   = raw_points − pickup points actually earned      max 45
denominator = time of last dressing award − time of first contact   simulation seconds
points/s    = numerator / denominator
```

First contact is fixed at the 240 Hz physics tick where the shirt and manikin collision meshes first come into contact range.
It is not the PhysX solver's internal contact event itself.
Awards are checked on 20 Hz evaluation samples. Time spent after contact is lost still counts toward the next award.

- Waiting without earning more dressing points keeps your last award time and points/s unchanged.
- Earning pickup later does not change the numerator or denominator; the 5 pickup points still appear in the item total.
- Each new dressing award recomputes the rate from the cumulative points and time since first contact. Taking longer can lower the rate.
- Contact with 0 dressing points gives `no_dressing_points` and points/s = 0.
- No contact gives `no_contact` with a null rate. Positive points with a zero denominator give `awaiting_elapsed_time` with a null rate.
- These null-rate failures count as 0 when aggregating. Failed seeds are not dropped to raise the average.
- Results over multiple shared seeds are the arithmetic mean of all episode scores.

## Result JSON fields

| Field | Meaning |
|---|---|
| valid | Whether the evaluation record is valid. If false, do not use it as a score |
| result.raw_points | Cumulative total including pickup, max 50 |
| result.score_items | Values for pickup, first sleeve, shoulder, second sleeve, and Overall items |
| result.rate_points | Numerator excluding pickup, max 45 |
| result.task_time_s | The points/s denominator above |
| result.first_contact_time_s | First contact time from episode start |
| result.last_score_award_time_s | Time of the last dressing award |
| result.elapsed_episode_time_s | Total episode time including warm-up |
| result.contact_elapsed_time_s | Total time from contact until now or the end |
| result.final_score | points/s, or null |
| result.score_status | scored / no_contact / no_dressing_points / awaiting_elapsed_time |
| result.success | Whether the dressing success condition was reached |

The terminal prints whenever the score, success, or first-contact status changes, and the final result is always saved.
`samples.jsonl` includes samples that were not printed. For your final policy, use the multi-seed evaluation in the [policy guide](POLICY.md).
