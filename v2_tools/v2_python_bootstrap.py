#!/usr/bin/env python3
import glob
import os
import sys
from pathlib import Path

if len(sys.argv) < 2:
    raise SystemExit("Usage: bootstrap.py SCRIPT [ARGS...]")

usd_dirs = [
    Path(p)
    for p in sorted(glob.glob(
        "/isaac-sim/extscache/omni.usd.libs-*"
    ))
    if Path(p).is_dir()
]

if not usd_dirs:
    raise SystemExit("ERROR: omni.usd.libs not found")

deps = Path("/project/participant/.deps")

if not deps.is_dir():
    raise SystemExit("ERROR: V2 .deps not found")

usd = usd_dirs[0]
env = os.environ.copy()

env["PYTHONPATH"] = ":".join(filter(None, [
    str(deps),
    str(usd),
    env.get("PYTHONPATH", "")
]))

env["LD_LIBRARY_PATH"] = ":".join(filter(None, [
    str(usd / "bin"),
    "/isaac-sim/kit/lib",
    env.get("LD_LIBRARY_PATH", "")
]))

print("[V2] USD:", usd, flush=True)
print("[V2] Dependencies:", deps, flush=True)

os.execve(
    "/isaac-sim/python.sh",
    ["/isaac-sim/python.sh", *sys.argv[1:]],
    env
)
