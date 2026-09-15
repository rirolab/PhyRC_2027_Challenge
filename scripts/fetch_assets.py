"""Fetch only locked runtime assets; refuse missing or modified input files."""
import hashlib
import json
from pathlib import Path
import sys
import urllib.request
from remote_zip import open_remote_zip

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    manifest = json.loads((ROOT / 'config/assets.lock.json').read_text())
    for asset in manifest['assets']:
        destination = ROOT / asset['path']
        if not destination.is_file():
            if '--check' in sys.argv or not any(k in asset for k in ('archive_url', 'url')):
                raise SystemExit(f'Missing required asset: {destination}')
            print(f"Downloading locked asset: {asset['path']}", flush=True)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + '.partial')
            if 'archive_url' in asset:
                with open_remote_zip(asset['archive_url']) as archive:
                    data = archive.read(asset['member'])
            else:
                with urllib.request.urlopen(asset['url'], timeout=120) as response:
                    data = response.read()
            if hashlib.sha256(data).hexdigest() != asset['sha256']:
                raise SystemExit(f'Download checksum mismatch: {destination}')
            temporary.write_bytes(data)
            temporary.replace(destination)
        if digest(destination) != asset['sha256']:
            raise SystemExit(f'Asset checksum mismatch: {destination}; restore the locked input.')
        print(f"OK {asset['path']}", flush=True)


if __name__ == '__main__':
    main()
