"""Shared startup/reset switch for scene placement randomization."""
import os


def spawn_randomization_enabled():
    value = os.environ.get('STRETCH4_RANDOMIZE', '1')
    if value not in ('0', '1'):
        raise ValueError('STRETCH4_RANDOMIZE must be 0 or 1')
    return value == '1'
