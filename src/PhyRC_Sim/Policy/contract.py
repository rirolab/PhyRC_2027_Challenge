"""Configurable schema validation and pure action decoding; no Isaac imports."""
import json
from pathlib import Path

import numpy as np


def load_contract(path):
    return json.loads(Path(path).read_text())


def observation_shapes(contract, *, robots, joints, profile='measured_state'):
    dimensions = dict(contract['default_dimensions'], robots=robots, joints=joints)
    dimensions['cameras'] = len(contract['cameras'])
    selected = contract['profiles'][profile]['include_availability']
    return {name: tuple(dimensions[d] if isinstance(d, str) else d for d in spec['shape'])
            for name, spec in contract['observations'].items() if spec['availability'] in selected}


def validate_observation(values, contract, *, robots, joints, profile='measured_state'):
    shapes = observation_shapes(contract, robots=robots, joints=joints, profile=profile)
    if set(values) != set(shapes):
        raise ValueError(f'Observation keys differ: missing={set(shapes)-set(values)}, extra={set(values)-set(shapes)}')
    for name, shape in shapes.items():
        value, spec = np.asarray(values[name]), contract['observations'][name]
        if value.shape != shape or value.dtype != np.dtype(spec['dtype']):
            raise ValueError(f'{name}: expected {shape}/{spec["dtype"]}, got {value.shape}/{value.dtype}')
        if not np.isfinite(value).all():
            raise ValueError(f'{name}: nonfinite observation')
        if 'range' in spec and (np.any(value < spec['range'][0]) or np.any(value > spec['range'][1])):
            raise ValueError(f'{name}: out of range')


def pack_measured_state(snapshot, contract):
    """Stack compatible robot manifests without hard-coding their DOF indices."""
    robots = snapshot['robots']
    names = [[j['name'] for j in robot['joints']] for robot in robots]
    if not robots or any(row != names[0] for row in names):
        raise ValueError('This profile needs matching named joint layouts; use per-robot records for heterogeneous robots')
    values = {}
    for name in ('joint_position', 'joint_velocity', 'base_pose_world', 'base_twist_world'):
        values[name] = np.array([robot[name] for robot in robots], dtype=np.float32)
    values['grasp_pose_base'] = np.array([r['links']['grasp_center']['pose_base'] for r in robots], dtype=np.float32)
    values['fingertip_position_base'] = np.array([
        [r['links'][name]['pose_base'][:3] for name in ('fingertip_left', 'fingertip_right')]
        for r in robots], dtype=np.float32)
    values['fingertip_origin_distance'] = np.array([[r['fingertip_origin_distance_m']] for r in robots], dtype=np.float32)
    values['controller_target'] = np.array([[r['command_targets'][k] for k in contract['controller_target_columns']]
                                            for r in robots], dtype=np.float32)
    values['gripper_close_command'] = np.array([[r['gripper_close_command']] for r in robots], dtype=bool)
    values['simulation_time_s'] = np.array(snapshot['simulation_time_s'], dtype=np.float64)
    validate_observation(values, contract, robots=len(robots), joints=len(names[0]))
    return values


def decode_action(action, contract, runtime_parameters, *, robots=2):
    """Decode normalized proposals to physical rates; never call a controller."""
    value = np.asarray(action)
    fields = contract['action']['fields']
    if value.shape != (robots, len(fields)) or value.dtype != np.float32:
        raise ValueError(f'Expected float32 action shape {(robots, len(fields))}')
    if not np.isfinite(value).all() or np.any(np.abs(value) > 1):
        raise ValueError('Action must be finite and inside [-1,1]')
    result = []
    for row in value:
        rates = {}
        for a, field in zip(row[:-1], fields[:-1]):
            scale = float(runtime_parameters[field['runtime_scale']])
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError('Runtime rate scales must be finite and positive')
            rates[field['backend_key']] = float(a) * scale * field['backend_sign']
        grip = fields[-1]
        intent = 'open' if row[-1] < grip['open_below'] else 'close' if row[-1] > grip['close_above'] else 'hold'
        result.append({'velocity_commands': rates, 'gripper_intent': intent})
    return result


def resolve_gripper_intent(previous_closed, intent):
    if intent not in ('open', 'close', 'hold'):
        raise ValueError('Unknown gripper intent')
    return bool(previous_closed) if intent == 'hold' else intent == 'close'
