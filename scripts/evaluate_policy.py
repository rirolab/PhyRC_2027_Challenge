#!/usr/bin/env python3
"""Run a saved policy in Isaac over multiple seeds using the teleop scorer."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path('/project') if Path('/project/src/PhyRC_Sim').is_dir() else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/PhyRC_Sim'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--policy', type=Path, help='Python adapter defining load_policy(checkpoint, observation_space, action_space)')
    source.add_argument('--zero-policy', action='store_true', help='Integration check with neutral actions; not a learned policy')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43])
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--profile', choices=['actor_rgbd', 'measured_state'], default='actor_rgbd')
    parser.add_argument('--output', type=Path, default=Path('/output/evaluation/policy'))
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(args.seeds) < 2 or min(args.seeds) < 0:
        parser.error('Provide at least two distinct nonnegative seeds')
    import math
    if not math.isfinite(args.seconds) or args.seconds < 0.05 or not math.isclose(args.seconds * 20, round(args.seconds * 20)):
        parser.error('--seconds must be a positive multiple of 0.05')
    if args.zero_policy and args.checkpoint:
        parser.error('--checkpoint requires --policy')
    if args.checkpoint and not args.checkpoint.is_file():
        parser.error('Checkpoint not found')
    if args.policy and not args.policy.is_file():
        parser.error('Policy adapter not found')
    from policy_cli import install_failure_handler
    install_failure_handler()
    from Policy.environment import DressingEnv
    from Policy.evaluation_live import EvaluationSession
    from Policy.evaluation import evaluate_submissions
    import numpy as np
    from Env_Config.Randomization import spawn_randomization_enabled
    if not spawn_randomization_enabled():
        parser.error('Multi-seed policy evaluation requires STRETCH4_RANDOMIZE=1')
    env = DressingEnv(profile=args.profile, max_episode_steps=round(args.seconds * 20))
    session = EvaluationSession.for_policy(env, args.output / 'episodes')
    # Imported AFTER SimulationApp starts so bundled Torch/Isaac are available.
    if args.zero_policy:
        policy = lambda obs: np.zeros(env.action_space.shape, np.float32)
    else:
        spec = importlib.util.spec_from_file_location('phyrc_submitted_policy', args.policy)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        policy = module.load_policy(checkpoint=args.checkpoint, observation_space=env.observation_space,
                                    action_space=env.action_space)
        if not callable(policy):
            raise ValueError('load_policy must return a callable policy(observation)')
    episodes = []
    metadata = {'mode': 'policy', 'zero_policy': args.zero_policy, 'profile': args.profile,
                'policy_sha256': hashlib.sha256(args.policy.read_bytes()).hexdigest() if args.policy else None,
                'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if args.checkpoint else None}
    try:
        for seed in args.seeds:
            obs, info = env.reset(seed=seed)
            if hasattr(policy, 'reset'):
                policy.reset()
            session.start(dict(metadata, seed=seed, human_spawn=info['human_spawn'], garment_spawn=info['garment_spawn'],
                               source_sha256=info['source_sha256'], source_commit=info['source_commit']))
            while True:
                action = np.asarray(policy(obs), dtype=np.float32)
                obs, _, terminated, truncated, info = env.step(action)
                if not session.active:
                    raise RuntimeError(f'Evaluation measurement failed: {session.last_report}')
                if terminated or truncated:
                    break
            report = session.finish(reason='policy_horizon')
            if not report['valid']:
                raise RuntimeError(f'Invalid evaluation: {report}')
            episodes.append({'seed': seed, 'samples': session.trace})
        data = {'schema_version': 'phase1-measurements-v7',
                'submissions': [{'submission_id': args.policy.stem if args.policy else 'zero-policy', 'episodes': episodes}]}
        results = evaluate_submissions(data)
        results['run_metadata'] = metadata
        # Unique run folder prevents a repeated evaluation overwriting evidence.
        path = args.output / ('summary_' + session.path.name)
        path.mkdir(parents=True, exist_ok=False)
        (path / 'measurements.json').write_text(json.dumps(data, allow_nan=False) + '\n')
        (path / 'scores.json').write_text(json.dumps(results, indent=2, allow_nan=False) + '\n')
        print(f"FINAL SCORE: {results['final_score']:.6f} points/s | {path / 'scores.json'}", flush=True)
    except BaseException:
        session.finish(reason='policy_exception', valid=False, take_final=False)
        raise
    env.close()


if __name__ == '__main__':
    main()
