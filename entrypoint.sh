#!/bin/bash
# Set the working directory the env files require, then hand off to Isaac Sim's
# python.
#
# Both env files resolve every asset off the CURRENT WORKING DIRECTORY, and the
# Stretch4 robot USD off `cwd/../RCareWorld-2.0/...`, so the two repos have to
# stay siblings and cwd has to be the PhyRC_Sim root (export README
# section 1). run.sh mounts a staged copy of both at /workspace to satisfy that,
# so the host working tree is never a write target.
set -euo pipefail
umask 0002

cd /workspace/PhyRC_Sim

exec /isaac-sim/python.sh "$@"
