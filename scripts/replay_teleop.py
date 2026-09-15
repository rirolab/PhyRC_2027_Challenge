#!/usr/bin/env python3
"""Replay recorded 3D states without re-simulating cloth or dropping ticks."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import tempfile

ROOT = Path('/project') if Path('/project/src/PhyRC_Sim').is_dir() else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/PhyRC_Sim'))
from Policy.teleop_recording import ArchiveReader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording', type=Path)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--verify', action='store_true', help='Check exact vertex/pose readback after every applied frame')
    parser.add_argument('--verify-only', action='store_true', help='USD CPU check; use ./run.sh cpu /scripts/replay_teleop.py ...')
    parser.add_argument('--speed', type=float, default=1, help='Playback speed; 0 runs without pacing. No frames are skipped.')
    parser.add_argument('--free-camera', action='store_true', help='Allow viewport camera navigation instead of recorded camera motion')
    parser.add_argument('--allow-incomplete', action='store_true')
    parser.add_argument('--report', type=Path, help='Optional verification report outside the immutable recording')
    parser.add_argument('--snapshot', type=Path, help='Optional PNG of the final replay state (requires rendering)')
    parser.add_argument('--evaluate', type=Path, help='Save replay-state evaluation under this output directory')
    parser.add_argument('--evaluation-start-tick', type=int, default=0)
    parser.add_argument('--evaluation-end-tick', type=int)
    parser.add_argument('--contact-context', type=Path, help='Exported collision context for older v2 recordings')
    parser.add_argument('--quality-context', type=Path, help='Exported V-neck identity for recordings before v6')
    args = parser.parse_args()
    if not math.isfinite(args.speed) or args.speed < 0:
        parser.error('--speed must be finite and nonnegative')
    if args.snapshot and args.verify_only:
        parser.error('--snapshot requires rendering')
    for output in (args.report, args.snapshot, args.evaluate):
        if output and (output.resolve() == args.recording.resolve() or args.recording.resolve() in output.resolve().parents):
            parser.error('Write reports/snapshots outside the source recording directory')
    reader = ArchiveReader(args.recording, args.allow_incomplete)
    evaluation = None
    if args.evaluate:
        if not reader.manifest['complete']:
            parser.error('Incomplete recordings cannot produce a valid evaluation')
        from Policy.evaluation_replay import ReplayEvaluation
        from Policy.evaluation_contact import recording_contact_context
        from Policy.evaluation_quality import recording_quality_context
        evaluation = ReplayEvaluation(reader.manifest['metadata'], args.evaluate,
                                      args.evaluation_start_tick, args.evaluation_end_tick,
                                      recording_contact_context(reader, args.contact_context),
                                      recording_quality_context(reader, args.quality_context))
    scene_path = args.recording / 'scene.usdc'
    if hashlib.sha256(scene_path.read_bytes()).hexdigest() != reader.manifest['scene_sha256']:
        parser.error('Recorded scene checksum mismatch')
    app, annotator = None, None
    if not args.verify_only:
        from isaacsim import SimulationApp
        app = SimulationApp({'headless': args.headless})
        import omni.usd
        import omni.timeline
        from pxr import Usd
        from Policy.teleop_playback import disable_physics
        omni.timeline.get_timeline_interface().stop()
        # Strip even the OmniPhysics APIs BEFORE opening the scene in Kit;
        # otherwise its background cooker can touch the archived deformable.
        source_stage = Usd.Stage.Open(str(scene_path))
        disable_physics(source_stage)
        temporary = tempfile.TemporaryDirectory(prefix='phyrc_replay_')
        playback_path = Path(temporary.name) / 'playback.usdc'
        source_stage.Flatten().Export(str(playback_path))
        if not omni.usd.get_context().open_stage(str(playback_path)):
            raise RuntimeError('Could not open recorded scene')
        for _ in range(12):
            app.update()
        stage = omni.usd.get_context().get_stage()
    else:
        from pxr import Usd
        stage = Usd.Stage.Open(str(scene_path))
    from Policy.teleop_playback import ReplayScene
    replay = ReplayScene(stage, reader.manifest['metadata'], follow_camera=not args.free_camera)
    if app:
        from omni.kit.viewport.utility import get_active_viewport
        viewport = get_active_viewport()
        if viewport and not args.free_camera:
            viewport.camera_path = '/World/FullReplayCamera'
        if args.snapshot:
            import omni.replicator.core as rep
            camera_path = str(viewport.camera_path) if args.free_camera and viewport else '/World/FullReplayCamera'
            product = rep.create.render_product(camera_path, (960, 540))
            annotator = rep.AnnotatorRegistry.get_annotator('rgb')
            annotator.attach([product])
    frames, tick, values = 0, 0, None
    started = time.monotonic()
    dt = reader.manifest['metadata']['physics_dt_s']
    try:
        for header, values in reader.frames():
            if app and not app.is_running():
                break
            replay.apply(values)
            if evaluation:
                evaluation.consume(header, values)
            if args.verify or args.verify_only:
                replay.verify(values)
            frames += 1
            tick = header['tick']
            if app:
                # Physics schemas are removed, and the timeline stays stopped.
                omni.timeline.get_timeline_interface().stop()
                app.update()
                if args.verify:
                    replay.verify(values)  # Rendering must not overwrite the state.
                if args.speed:
                    remaining = started + tick * dt / args.speed - time.monotonic()
                    while remaining > 0 and app.is_running():
                        time.sleep(min(remaining, 0.01))
                        app.update()
                        remaining = started + tick * dt / args.speed - time.monotonic()
        if annotator and values is not None and app.is_running():
            from PIL import Image
            for _ in range(8):
                app.update()
            rep.orchestrator.step(rt_subframes=4, delta_time=0.0,
                                  pause_timeline=True, wait_for_render=True)
            replay.verify(values)
            rgba = annotator.get_data()
            if rgba.size == 0:
                raise RuntimeError('Replay camera produced no image')
            args.snapshot.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgba[:, :, :3]).save(args.snapshot)
        report = {'schema': 'phyrc-replay-verification-v1', 'frames_applied': frames,
                  'frames_expected': reader.manifest['frames'], 'all_frames_applied': frames == reader.manifest['frames'],
                  'last_physics_tick': tick, 'recorded_simulation_duration_s': tick * dt,
                  'state_readback_verified': args.verify or args.verify_only,
                  'physics_resimulated': False, 'recording_complete': reader.manifest['complete']}
        if evaluation:
            if frames != reader.manifest['frames']:
                raise ValueError('Replay ended early; evaluation is incomplete')
            report['evaluation'] = evaluation.finish()
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + '\n')
        # EvaluationSession already prints changed scores and the final score.
        print('REPLAY ' + json.dumps({k: v for k, v in report.items() if k != 'evaluation'}), flush=True)
    except BaseException:
        if evaluation and evaluation.session and evaluation.session.active:
            evaluation.session.finish(reason='replay_interrupted_or_failed', valid=False, take_final=False)
        import traceback
        traceback.print_exc()
        if app:
            app.close(exit_code=1)
        raise
    if app:
        app.close()


if __name__ == '__main__':
    main()
