"""Export rest-mesh sleeve regions and authored V-neck identity without changing an old recording."""
import argparse
import hashlib
from pathlib import Path
import sys
ROOT = Path('/project') if Path('/project/config').exists() else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/PhyRC_Sim'))
from Policy.teleop_recording import ArchiveReader, atomic_json
from Policy.evaluation_quality import extract_quality_context

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('recording', type=Path)
p.add_argument('--output', type=Path)
a = p.parse_args()
reader = ArchiveReader(a.recording)
output = a.output or a.recording.parent.parent/'evaluation/quality_contexts'/(a.recording.name+'.json')
if output.resolve() == a.recording.resolve() or a.recording.resolve() in output.resolve().parents:
    p.error('Write the sidecar outside the original recording')
quality = extract_quality_context(reader)
output.parent.mkdir(parents=True, exist_ok=True)
atomic_json(output, dict(scene_sha256=reader.manifest['scene_sha256'],
                        manifest_sha256=hashlib.sha256((a.recording/'manifest.json').read_bytes()).hexdigest(), quality=quality))
print('Sleeve/V-neck context saved: ' + str(output))
