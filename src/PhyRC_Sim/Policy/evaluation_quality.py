"""Read-only sleeve/front identity upgrade for archives recorded before scoring v7."""
import hashlib
import json
from pathlib import Path


def extract_quality_context(reader):
    from pxr import Usd, UsdGeom
    import numpy as np
    from .evaluation_geometry import vneck_front_vertices, GarmentGeometry
    scene = reader.path / 'scene.usdc'
    if hashlib.sha256(scene.read_bytes()).hexdigest() != reader.manifest['scene_sha256']:
        raise ValueError('Recorded scene checksum mismatch')
    stage = Usd.Stage.Open(str(scene))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(reader.manifest['metadata']['garment_paths'][0]))
    g = GarmentGeometry(np.asarray(mesh.GetPointsAttr().Get(), float),
                        np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), int).reshape(-1, 3),
                        front_ids=vneck_front_vertices(mesh))
    return dict(front_ids=g.front_ids.tolist(), faces=g.faces.tolist(),
                sleeve_regions=[ids.tolist() for ids in g.sleeve_regions])


def recording_quality_context(reader, context_path=None):
    _, values = next(reader.frames())
    context = json.loads(str(values.get('evaluation_context', '{}')))
    if context.get('front_ids') is not None and context.get('sleeve_regions') is not None:
        return None
    path = Path(context_path) if context_path else reader.path.parent.parent / 'evaluation/quality_contexts' / (reader.path.name + '.json')
    if path.exists():
        data = json.loads(path.read_text())
        if (data['manifest_sha256'] != hashlib.sha256((reader.path/'manifest.json').read_bytes()).hexdigest()
                or data['scene_sha256'] != reader.manifest['scene_sha256']
                or data['scene_sha256'] != hashlib.sha256((reader.path/'scene.usdc').read_bytes()).hexdigest()):
            raise ValueError('V-neck annotation does not match this recording')
        if data['quality'].get('sleeve_regions') is not None:
            return data['quality']
    if context_path and not path.exists():
        raise ValueError('Quality context file not found: ' + str(path))
    try:
        return extract_quality_context(reader)
    except ImportError as exc:
        raise ValueError('This old recording needs a one-time CPU/USD sleeve/V-neck annotation export: '
                         'PHYRC_ACCEPT_EULA=1 ./run.sh cpu /scripts/export_replay_quality.py '
                         + str(reader.path) + ' (use its container path under /output). Then retry headless evaluation.') from exc
