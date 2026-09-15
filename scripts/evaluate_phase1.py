#!/usr/bin/env python3
"""Print Phase 1 final scores from Isaac measurement traces (or a synthetic demo)."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / 'src/PhyRC_Sim').is_dir() and Path('/project/src/PhyRC_Sim').is_dir():
    ROOT = Path('/project')
sys.path.insert(0, str(ROOT / 'src/PhyRC_Sim'))
from Policy.evaluation import ScoringConfig, evaluate_submissions, format_score_items


def demo_data():
    """Synthetic 50/10 + 20/10 points/s episodes, averaging 3.5 points/s."""
    episodes = []
    for seed, complete in [(42, True), (43, False)]:
        samples = []
        for tick in range(201):
            t = tick / 20
            samples.append({
                'time_s': t, 'first_contact_time_s': 0.0, 'gripper_holding': [True, False],
                'garment_lifted_clear': True,
                'gripper_lifted': [True, False],
                'dressing_complete': bool(complete and t >= 6),
                'wrist_in_sleeve': {'left': t >= 4, 'right': complete and t >= 6},
                'garment_beyond_shoulder': {'left': False, 'right': t >= 5},
                'arm_coverage': {'left': (1.0 if complete else 0.5) if t >= 4 else 0.0,
                                 'right': 1.0 if complete and t >= 6 else 0.0},
            })
        episodes.append({'seed': seed, 'samples': samples})
    return {'schema_version': 'phase1-measurements-v3',
            'submissions': [{'submission_id': 'synthetic-demo', 'episodes': episodes}]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--input', type=Path, help='Measurement JSON; see docs/competition/EVALUATION.md')
    source.add_argument('--demo', action='store_true', help='Score synthetic data, without Isaac Sim')
    parser.add_argument('--output', type=Path, help='Write detailed score JSON')
    parser.add_argument('--write-demo-input', type=Path, help='With --demo, save a complete input example')
    parser.add_argument('--max-sample-gap-s', type=float, default=0.05)
    parser.add_argument('--time-limit-s', type=float, help='Explicit episode deadline; no proposal default')
    parser.add_argument('--simultaneous-first-arm', choices=('left', 'right'), default='left')
    args = parser.parse_args(argv)
    if args.write_demo_input and not args.demo:
        parser.error('--write-demo-input requires --demo')
    paths = [p.resolve() for p in (args.input, args.output, args.write_demo_input) if p]
    if len(paths) != len(set(paths)):
        parser.error('Input and output paths must be different')
    try:
        config = ScoringConfig(args.max_sample_gap_s, args.time_limit_s, args.simultaneous_first_arm)
        data = demo_data() if args.demo else json.loads(args.input.read_text(encoding='utf-8'))
        report = evaluate_submissions(data, config)
        report['synthetic_demo'] = args.demo
        if args.input:
            import hashlib
            report['input_sha256'] = hashlib.sha256(args.input.read_bytes()).hexdigest()
        for path, content in ((args.output, report), (args.write_demo_input, data)):
            if path:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(content, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    except (ValueError, OSError, TypeError, AttributeError) as exc:
        parser.exit(2, f'Evaluation failed: {exc}\n')
    if args.demo:
        print('SYNTHETIC DEMO — not an Isaac Sim evaluation result')
    for submission in report['submissions']:
        print(f"Submission: {submission['submission_id']}")
        for episode in submission['episodes']:
            print(f"  {format_score_items(episode)}")
            rate = (f"{episode['final_score']:.6f} points/s" if episode['final_score'] is not None
                    else f"N/A ({episode['score_status']}); ranking contribution 0")
            rate_detail = (f" | rate numerator={episode['rate_points']:.3f}/45 (pickup excluded)"
                           if 'rate_points' in episode else '')
            print(f"  seed={episode['seed']}: {episode['raw_points']:.3f}/50 points, "
                  f"{episode['task_time_s']:.3f} s | {rate}{rate_detail}")
        print(f"  Mean score: {submission['final_score']:.6f} points/s")
    print(f"FINAL SCORE: {report['final_score']:.6f} points/s "
          f"(best submission: {report['best_submission_id']})")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
