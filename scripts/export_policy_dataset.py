#!/usr/bin/env python3
"""Render deferred policy observations from immutable physics state; no new solve."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import uuid

import numpy as np
ROOT = Path('/project') if Path('/project/config').exists() else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/PhyRC_Sim'))
from Policy.teleop_recording import ArchiveReader, atomic_json
from Policy.training_dataset import DatasetWriter


class RecordedSensors:
    def __init__(self, stage, config, viewport_resolution):
        from pxr import UsdGeom, Gf
        import omni.replicator.core as rep
        self.config, self.items = config, []
        rep.orchestrator.set_capture_on_play(False)
        w, h = config['default_dimensions']['width'], config['default_dimensions']['height']
        for spec in config['cameras']:
            camera = UsdGeom.Camera.Define(stage, '/World/ExportSensors/' + spec['id'])
            aperture = 20.955
            focal = aperture / (2 * np.tan(np.deg2rad(spec['horizontal_fov_deg']) / 2))
            camera.CreateProjectionAttr('perspective')
            camera.CreateHorizontalApertureAttr(aperture)
            camera.CreateVerticalApertureAttr(aperture * h / w)
            camera.CreateFocalLengthAttr(float(focal))
            camera.CreateClippingRangeAttr(Gf.Vec2f(*spec['clip_m']))
            product = rep.create.render_product(str(camera.GetPath()), (w, h))
            rgb = rep.AnnotatorRegistry.get_annotator('rgb')
            depth = rep.AnnotatorRegistry.get_annotator('distance_to_image_plane')
            rgb.attach([product])
            depth.attach([product])
            self.items.append(dict(op=camera.AddTransformOp(), rgb=rgb, depth=depth, spec=spec))
        self.viewport_resolution = tuple(map(int, viewport_resolution))
        product = rep.create.render_product('/World/FullReplayCamera', self.viewport_resolution)
        self.viewport_rgb = rep.AnnotatorRegistry.get_annotator('rgb')
        self.viewport_rgb.attach([product])

    def capture(self, values):
        from pxr import Gf
        import omni.replicator.core as rep
        import omni.timeline
        for item, pose in zip(self.items, values['policy_camera_world']):
            item['op'].Set(Gf.Matrix4d(pose.T.tolist()))
        omni.timeline.get_timeline_interface().stop()
        rep.orchestrator.step(rt_subframes=4, delta_time=0.0, pause_timeline=True, wait_for_render=True)
        w, h = self.config['default_dimensions']['width'], self.config['default_dimensions']['height']
        rgb, depth, masks, intrinsics = [], [], [], []
        for item in self.items:
            image = np.asarray(item['rgb'].get_data())
            z = np.asarray(item['depth'].get_data(), np.float32)
            if image.shape != (h, w, 4) or z.shape != (h, w):
                raise RuntimeError('Incomplete rendered camera frame')
            near, far = item['spec']['clip_m']
            valid = np.isfinite(z) & (z >= near) & (z <= far)
            rgb.append(image[..., :3].copy())
            depth.append(np.where(valid, z, 0)[..., None])
            masks.append(valid[..., None])
            f = w / (2 * np.tan(np.deg2rad(item['spec']['horizontal_fov_deg']) / 2))
            intrinsics.append([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
        obs = {k[len('policy_obs/'):]: v.copy() for k, v in values.items() if k.startswith('policy_obs/')}
        obs.update(rgb=np.stack(rgb), depth=np.stack(depth), depth_valid=np.stack(masks))
        view = np.asarray(self.viewport_rgb.get_data())
        vw, vh = self.viewport_resolution
        if view.shape != (vh, vw, 4):
            raise RuntimeError('Incomplete viewport frame')
        timestamp = float(obs['simulation_time_s'])
        audit = dict(camera_timestamp_s=np.full(5, timestamp, np.float64),
                     camera_intrinsics=np.array(intrinsics),
                     camera_world_from_optical=values['policy_camera_world'] @ np.diag([1, -1, -1, 1]),
                     viewport_rgb=view[..., :3].copy(), viewport_world=values['camera_world'],
                     viewport_lens=values['camera_lens'], viewport_timestamp_s=np.float64(timestamp))
        return obs, audit


def export(recording, output=None):
    reader = ArchiveReader(recording)  # Explicitly refuse incomplete or corrupt sources.
    events = [json.loads(s) for s in (recording / 'events.jsonl').read_text().splitlines()]
    captures = [e['metadata'] for e in events if e['kind'] == 'training_capture']
    if len(captures) != 1 or captures[0]['schema'] != 'phyrc-deferred-capture-v1':
        raise ValueError('Recording lacks deferred policy observations; use --training-record 1')
    meta = captures[0]
    starts = {e['episode']: e for e in events if e['kind'] == 'policy_episode_start'}
    ends = {e['episode']: e for e in events if e['kind'] == 'policy_episode_end'}
    actions = [e for e in events if e['kind'] == 'policy_action']
    if not starts or set(starts) != set(ends) or any(not e['valid'] or not e['samples'] for e in ends.values()):
        raise ValueError('Capture has an incomplete/empty episode')
    output = Path(output or meta['dataset_path'])
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'policy.hdf5').exists() or (output / 'audit.hdf5').exists():
        raise ValueError('Output already contains HDF5 data; choose a new --output directory')
    scene_file = recording / 'scene.usdc'
    if hashlib.sha256(scene_file.read_bytes()).hexdigest() != reader.manifest['scene_sha256']:
        raise ValueError('Recorded scene checksum mismatch')
    # Start the GPU only after all readily checkable inputs pass.
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True})
    writer = None
    try:
        import omni.usd
        import omni.timeline
        from pxr import Usd
        from Policy.teleop_playback import disable_physics, ReplayScene
        from Policy.evaluation_replay import ReplayEvaluation
        from Policy.evaluation_quality import recording_quality_context
        quality = recording_quality_context(reader)
        omni.timeline.get_timeline_interface().stop()
        stage = Usd.Stage.Open(str(scene_file))
        disable_physics(stage)
        with tempfile.TemporaryDirectory(prefix='phyrc_dataset_render_') as temporary:
            playback_path = Path(temporary) / 'scene.usdc'
            stage.Flatten().Export(str(playback_path))
            if not omni.usd.get_context().open_stage(str(playback_path)):
                raise RuntimeError('Could not open recorded scene')
            for _ in range(12):
                app.update()
            stage = omni.usd.get_context().get_stage()
            replay = ReplayScene(stage, reader.manifest['metadata'])
            pending = output / ('.export_' + uuid.uuid4().hex[:10])
            writer = DatasetWriter(pending, meta['contract'], dict(meta, physics_resimulated=False,
                                   export_source_manifest_sha256=hashlib.sha256((recording/'manifest.json').read_bytes()).hexdigest()))
            sensors, evaluator, current, episode = None, None, None, None
            frames, transitions = 0, 0
            for header, values in reader.frames():
                if header['kind'] == 'policy_observation':
                    incoming = int(values['policy_episode'])
                    frame = int(values['policy_frame'])
                    if frame == 0:
                        if evaluator is not None:
                            raise ValueError('Previous episode did not finish')
                        episode = incoming
                        start, end = starts[episode], ends[episode]
                        if header['tick'] != start['tick']:
                            raise ValueError('Episode start does not match observation')
                        evaluator = ReplayEvaluation(reader.manifest['metadata'], pending/'evaluation', start['tick'], end['tick'],
                                                     quality_context=quality)
                        writer.start_episode(dict(start, start_tick=start['tick']))
                        current = None
                    if incoming != episode:
                        raise ValueError('Episode index changed without initial observation')
                if evaluator:
                    evaluator.consume(header, values)
                if header['kind'] != 'policy_observation':
                    continue
                replay.apply(values)
                replay.verify(values)
                if sensors is None:
                    sensors = RecordedSensors(stage, meta['contract'], values['policy_viewport_resolution'])
                obs, audit = sensors.capture(values)
                replay.verify(values)  # Rendering cannot overwrite the authoritative state.
                frames += 1
                if current is not None:
                    previous_obs, previous_audit, previous_tick, previous_frame = current
                    if frame != previous_frame + 1:
                        raise ValueError('Missing policy observation')
                    action = obs['previous_action']
                    recorded_action = actions[transitions]
                    if (recorded_action['sample_index'] != transitions or recorded_action['tick'] != previous_tick
                            or recorded_action['end_tick'] != header['tick']
                            or not np.array_equal(action, np.asarray(recorded_action['action'], np.float32))):
                        raise ValueError('Observation/action alignment differs from the input journal')
                    score = evaluator.session.scorer.result()
                    audit_pair = dict(previous_audit, **{'next_' + k: v for k, v in audit.items()})
                    audit_pair.update(controller_targets_applied=values['policy_applied_targets'],
                                      control_start_ticks=np.arange(previous_tick, previous_tick + 12, 4, dtype=np.int64),
                                      score_items=np.array([v['points'] for v in score['score_items'].values()], np.float32),
                                      raw_points=np.float32(score['raw_points']),
                                      first_contact_tick=np.int64(-1 if evaluator.session.clock.first_contact_tick is None else evaluator.session.clock.first_contact_tick),
                                      score_time_s=np.float64(evaluator.session.clock_result(score)['task_time_s']))
                    writer.append(previous_obs, action, obs, start_tick=previous_tick, end_tick=header['tick'], audit=audit_pair)
                    transitions += 1
                current = obs, audit, header['tick'], frame
                if header['tick'] == ends[episode]['tick']:
                    if frame != ends[episode]['samples']:
                        raise ValueError('Episode transition count differs from capture')
                    result = evaluator.finish()
                    writer.finish_episode(ends[episode]['reason'], result, valid=True)
                    evaluator, current = None, None
                if frames % 20 == 0:
                    print(f'[Dataset export] {frames} observations rendered / {transitions} transitions', flush=True)
            if evaluator is not None or transitions != sum(e['samples'] for e in ends.values()):
                raise ValueError('Export ended before all complete transitions were rendered')
            writer.close()
            for name in ('policy.hdf5', 'audit.hdf5', 'evaluation'):
                (pending / name).rename(output / name)
            pending.rmdir()
            status = output / 'capture.json'
            report = json.loads(status.read_text()) if status.exists() else dict(meta)
            report.update(status='ready', complete=True, exported_observation_frames=frames, exported_transitions=transitions,
                          physics_resimulated=False)
            atomic_json(status, report)
            print('[Dataset export] READY ' + str(output), flush=True)
    except BaseException:
        if writer is not None and not writer.closed:
            writer.close(complete=False)
        import traceback
        traceback.print_exc()
        app.close(exit_code=1)
        raise
    app.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording', type=Path, nargs='?')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--request', type=Path)
    args = parser.parse_args()
    if args.request:
        if args.recording or args.output:
            parser.error('Use --request alone or recording [--output]')
        request = json.loads(args.request.read_text())
        args.recording, args.output = Path(request['recording']), Path(request['output'])
    if not args.recording:
        parser.error('Provide a recording or --request')
    export(args.recording, args.output)


if __name__ == '__main__':
    main()
