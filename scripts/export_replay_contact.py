#!/usr/bin/env python3
"""One-time CPU/USD collision context export for older v2 full recordings."""
import argparse
import hashlib
from pathlib import Path
import sys

ROOT = Path('/project') if Path('/project/src/PhyRC_Sim').is_dir() else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/PhyRC_Sim'))
from Policy.teleop_recording import ArchiveReader, atomic_json
from Policy.evaluation_contact import contact_context_from_stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    reader = ArchiveReader(args.recording)
    if reader.manifest['metadata'].get('evaluation_state_version') != 1:
        parser.error('v1 recordings also lack grasp evidence; full scoring is unsupported')
    output = args.output or args.recording.parent.parent / 'evaluation/contact_contexts' / (args.recording.name + '.json')
    if output.resolve() == args.recording.resolve() or args.recording.resolve() in output.resolve().parents:
        parser.error('Do not write inside the original recording')
    scene = args.recording / 'scene.usdc'
    digest = hashlib.sha256(scene.read_bytes()).hexdigest()
    if digest != reader.manifest['scene_sha256']:
        parser.error('Recorded scene checksum mismatch')
    from pxr import Usd
    stage = Usd.Stage.Open(str(scene))
    contact = contact_context_from_stage(stage, reader.manifest['metadata']['garment_paths'][0])
    data = {'scene_sha256': digest,
            'manifest_sha256': hashlib.sha256((args.recording / 'manifest.json').read_bytes()).hexdigest(),
            'contact': contact}
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, data)
    print(f'Collision context saved: {output}', flush=True)


if __name__ == '__main__':
    main()
