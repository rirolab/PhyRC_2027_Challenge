"""Single-scene Gymnasium adapter around the maintained dressing simulator.

One instance per process; Isaac owns a process-wide application. Reward and
success belong to a replaceable task evaluator, not the physics backend.
"""
from copy import deepcopy
import hashlib
import os
import subprocess
from pathlib import Path

import gymnasium as gym
import numpy as np
from .contract import load_contract, observation_shapes, pack_measured_state, decode_action, resolve_gripper_intent, validate_observation
from .state import array, snapshot


class DressingEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 20}
    _claimed = False

    def __init__(self, *, profile='actor_rgbd', resolution=None, max_episode_steps=400,
                 evaluator=None, contract_path=None, render_mode=None, settle_ticks=90, initial_slot=None):
        if profile not in ('measured_state', 'actor_rgbd'):
            raise ValueError('Unknown observation profile')
        if render_mode not in (None, 'rgb_array') or (render_mode and profile != 'actor_rgbd'):
            raise ValueError('rgb_array requires actor_rgbd')
        if (not isinstance(max_episode_steps, int) or not isinstance(settle_ticks, int)
                or max_episode_steps < 1 or settle_ticks < 60):
            raise ValueError('Positive episode horizon and at least 60 settling ticks required')
        if DressingEnv._claimed:
            raise RuntimeError('Isaac supports one DressingEnv per process; launch a new process')
        contract_path = Path(contract_path or Path(__file__).resolve().parents[3] / 'config/policy_interface.json')
        self.contract = load_contract(contract_path)
        timing = self.contract['timing']
        if (timing['policy_hz'], timing['control_hz'], timing['control_ticks_per_action']) != (20, 60, 3):
            raise ValueError('This adapter currently requires 20 Hz policy and 60 Hz control')
        if resolution is not None:
            width, height = resolution
            if not all(isinstance(v, int) and 16 <= v <= 2048 for v in (width, height)):
                raise ValueError('Resolution is (width,height), integers in [16,2048]')
            self.contract['default_dimensions'].update(width=width, height=height)
        self.profile, self.render_mode = profile, render_mode
        self.max_episode_steps, self.settle_ticks = int(max_episode_steps), int(settle_ticks)
        self.evaluator = evaluator
        if evaluator is not None and not callable(evaluator):
            raise ValueError('evaluator must be callable')
        self._done, self._closed, self._episode = True, False, 0
        self.sensors = None
        os.environ['STRETCH4_HEADLESS'] = '1'
        os.environ['STRETCH4_SHOW_COLLIDER'] = '0'
        import Env_StandAlone.Teleop_TShirt_Stretch4_Env as M
        DressingEnv._claimed = True
        self.M = M
        self.initial_slot = Path(initial_slot).resolve() if initial_slot else None
        original_placement = M.randomize_human_and_chair
        if self.initial_slot:
            from .stages import validate_slot, restore_placement
            validate_slot(self.initial_slot, M._STATE_GEOMETRY_REVISION)
            M.randomize_human_and_chair = lambda stage, path: restore_placement(stage, self.initial_slot)
        try:
            self.backend = M.TeleopTShirtStretch4_Env()
        finally:
            M.randomize_human_and_chair = original_placement
        self.world, self.stage = self.backend.world, self.backend.stage
        self.rigs = (self.backend.rig, self.backend.rig2)
        self.cloths, self.faces, self._post = M.initialize_manipulation(self.backend)
        self._initial_states = [deepcopy(r['state']) for r in self.rigs]
        self._initial_robots = [(array(r['robot'].get_world_pose()[0]), array(r['robot'].get_world_pose()[1]),
                                 array(r['robot'].get_joint_positions())) for r in self.rigs]
        self._initial_cloths = [array(c.get_world_positions()) for c in self.cloths]
        self._initial_garment_spawn = deepcopy(self.backend.garment_spawn)
        self._toggles = [M.make_gripper_toggle(r, f'policy_robot_{i}', self.cloths, self.faces)
                         for i, r in enumerate(self.rigs)]
        self._runtime = {key: float(getattr(M, key)) for key in
                         ('BASE_LINEAR_RATE', 'BASE_ANGULAR_RATE', 'LIFT_RATE', 'ARM_RATE', 'WRIST_RATE')}
        self._physics_per_tick = round((1 / 60) / self.world.get_physics_dt())
        if not np.isclose(self._physics_per_tick * self.world.get_physics_dt(), 1 / 60):
            raise ValueError('Physics frequency must be an integer multiple of 60 Hz')
        self.action_space = gym.spaces.Box(-1, 1, (len(self.rigs), 9), dtype=np.float32)
        shapes = observation_shapes(self.contract, robots=len(self.rigs), joints=len(self.rigs[0]['robot'].dof_names), profile=profile)
        spaces = {}
        for name, shape in shapes.items():
            spec = self.contract['observations'][name]
            dtype = np.dtype(spec['dtype'])
            low, high = spec.get('range', (0, 1) if dtype == bool else (-np.inf, np.inf))
            spaces[name] = gym.spaces.Box(low, high, shape, dtype=dtype)
        self.observation_space = gym.spaces.Dict(spaces)
        root = Path(__file__).resolve().parent
        self._hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob('*.py'))}
        repo = Path(__file__).resolve().parents[3]
        try:
            self._source_commit = subprocess.check_output(
                ['git', '-c', f'safe.directory={repo}', '-C', str(repo), 'rev-parse', 'HEAD'],
                text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            self._source_commit = os.environ.get('PHYRC_SOURCE_COMMIT')
        self._source_dirty = os.environ.get('PHYRC_SOURCE_DIRTY')
        self._source_dirty = None if self._source_dirty is None else self._source_dirty == '1'
        garment_sampler = Path(M.__file__).resolve().parents[1] / 'Env_Config/Garment/RandomSpawn.py'
        self._hashes['garment_random_spawn.py'] = hashlib.sha256(garment_sampler.read_bytes()).hexdigest()
        self._hashes['contract'] = hashlib.sha256(contract_path.read_bytes()).hexdigest()
        self._hashes['teleop'] = hashlib.sha256(Path(M.__file__).read_bytes()).hexdigest()
        if profile == 'actor_rgbd':
            from .sensors import RGBDSensors
            self.sensors = RGBDSensors(self.stage, self.world, self.rigs, self.contract)

    def _attach_callbacks(self):
        self.world.add_physics_callback('stretch4_grab_follow',
            lambda dt: self.M.cloth_pre_step(self.cloths, self.rigs, dt))
        self._post = self.world._physics_context._physics_sim_interface.subscribe_physics_on_step_events(
            pre_step=False, order=100, on_update=lambda dt, context: self.M.cloth_post_step(self.cloths, self.rigs, dt))

    def _detach_callbacks(self):
        self.world.remove_physics_callback('stretch4_grab_follow')
        self._post = None

    def _tick(self, decoded):
        from Env_Config.Garment.ContinuousClothControl import cloth_control_batch
        with cloth_control_batch():
            for rig, command in zip(self.rigs, decoded):
                self.M.drive_robot(rig, self.M.ROBOT1_KEYMAP, set(), self.cloths, 1 / 60, 0,
                                   commands=command['velocity_commands'])
        for rig in self.rigs:
            state = rig['state']
            if state.get('pending_grab'):
                if abs(state['grip_pos'] - self.M.GRIPPER_CLOSED) > self.M.GRAB_CLOSED_EPS:
                    state['_settle'] = 0
                else:
                    state['_settle'] = state.get('_settle', 0) + 1
                    if state['_settle'] >= self.M.GRAB_SETTLE_FRAMES:
                        state['pending_grab'], state['_settle'] = False, 0
                        rig['attempt_grab']()
        for _ in range(self._physics_per_tick):
            self.world.step(render=False)

    def reset(self, *, seed=None, options=None):
        if self._closed:
            raise RuntimeError('Environment is closed')
        if options:
            raise ValueError('No reset options currently supported; use constructor configuration')
        super().reset(seed=seed)
        if self.initial_slot:
            return self._reset_slot(seed)
        # Explicit seed describes the placement directly. None advances Gym's RNG.
        self.episode_seed = int(seed) if seed is not None else int(self.np_random.integers(0, 2**63))
        self._done = True
        from pxr import UsdGeom
        from Env_Config.Garment.NativeGrasp import clear_native_grasp
        self._detach_callbacks()
        for rig, initial in zip(self.rigs, self._initial_states):
            clear_native_grasp(rig, self.stage)
            rig['state'].clear()
            rig['state'].update(deepcopy(initial))
            rig['held'].clear()
        self.world.stop()
        # Remove the previous outer op before sampling about the ORIGINAL pose.
        for name in ('Human', 'Chair'):
            prim = self.stage.GetPrimAtPath('/World/' + name)
            xf = UsdGeom.Xformable(prim)
            xf.SetXformOpOrder([op for op in xf.GetOrderedXformOps() if op.GetOpName() != 'xformOp:transform:randomSpawn'], xf.GetResetXformStack())
            prim.RemoveProperty('xformOp:transform:randomSpawn')
        old_seed = os.environ.get('HUMAN_SPAWN_SEED')
        try:
            os.environ['HUMAN_SPAWN_SEED'] = str(self.episode_seed)
            self.spawn = self.M.randomize_human_and_chair(self.stage, self.backend.human.prim_path)
        finally:
            if old_seed is None:
                os.environ.pop('HUMAN_SPAWN_SEED', None)
            else:
                os.environ['HUMAN_SPAWN_SEED'] = old_seed
        self.backend.reset()
        self.M.simulation_app.update()
        for rig, (pos, quat, q) in zip(self.rigs, self._initial_robots):
            robot = rig['robot']
            robot.initialize()
            self.M.stretch4_tuning(rig, restore=True)
            robot.set_world_pose(self.M._to_t(pos), self.M._to_t(quat))
            robot.set_joint_positions(self.M._to_t(q))
            robot.set_joint_velocities(self.M._to_t(np.zeros_like(q)))
            robot._articulation_view.set_velocities(self.M._to_t(np.zeros((1, 6))))
        from Env_Config.Garment.RandomSpawn import sample_garment_spawn
        self.garment_spawn = sample_garment_spawn(self.backend.garment_table_centers, self.M.BOX_SIZE, seed=self.episode_seed)
        offset = (np.array(self.garment_spawn['spawn_position_world_m'])
                  - np.array(self._initial_garment_spawn['spawn_position_world_m']))
        reset_positions = [(p + offset).astype(np.float32) for p in self._initial_cloths]
        self.backend.garment_spawn = deepcopy(self.garment_spawn)
        for cloth, positions in zip(self.cloths, reset_positions):
            cloth.initialize()
            cloth.set_world_positions(positions.copy())
            cloth.set_velocities(np.zeros_like(positions))
        restore_error = max(float(np.max(np.abs(array(c.get_world_positions()) - p)))
                            for c, p in zip(self.cloths, reset_positions))
        if restore_error > 1e-7:
            raise RuntimeError(f'Cloth reset restore error: {restore_error} m')
        self.M.rebuild_contact_guards(self.backend, self.cloths)
        self._attach_callbacks()
        neutral = decode_action(np.zeros(self.action_space.shape, np.float32), self.contract, self._runtime)
        for _ in range(self.settle_ticks):
            self._tick(neutral)
        self._previous_action = np.zeros(self.action_space.shape, np.float32)
        self._step_id = 0
        self._episode += 1
        self._start_time = float(self.world.current_time)
        if self.sensors:
            self.sensors.reset_episode()
        observation, info = self._observe()
        speeds = np.linalg.norm(observation['base_twist_world'][:, :3], axis=1)
        finger_error = max(float(np.max(np.abs(array(r['robot'].get_joint_positions())[r['grip_idx']] - self.M.GRIPPER_OPEN))) for r in self.rigs)
        self._reset_metrics = {'control_ticks': self.settle_ticks, 'cloth_restore_max_abs_m': restore_error, 'max_base_linear_speed_m_s': float(speeds.max()),
                               'max_finger_open_error_rad': finger_error}
        if speeds.max() > .05 or finger_error > .03:
            raise RuntimeError('Reset did not settle; increase settle_ticks or inspect physics: ' + str(self._reset_metrics))
        info['reset_settling'] = dict(self._reset_metrics)
        self._last_observation = {key: value.copy() for key, value in observation.items()}
        if self.evaluator is not None and hasattr(self.evaluator, 'reset'):
            self.evaluator.reset(info)
        self._done = False
        return observation, info

    def _reset_slot(self, seed):
        """Use production checkpoint loading; never save into the input directory."""
        from .stages import load_slot
        self._done = True
        self.episode_seed = int(seed) if seed is not None else int(self.np_random.integers(0, 2**63))
        load_slot(self, self.initial_slot)
        self.spawn = {'restored_from': str(self.initial_slot), 'randomized': False}
        self.garment_spawn = {'restored_from': str(self.initial_slot), 'randomized': False}
        neutral = decode_action(np.zeros(self.action_space.shape, np.float32), self.contract, self._runtime)
        for _ in range(self.settle_ticks):
            self._tick(neutral)
        self._previous_action = np.zeros(self.action_space.shape, np.float32)
        self._step_id = 0
        self._episode += 1
        self._start_time = float(self.world.current_time)
        if self.sensors:
            self.sensors.reset_episode()
        observation, info = self._observe()
        info['reset_settling'] = {'control_ticks': self.settle_ticks, 'mode': 'checkpoint; FEM history not bitwise restored'}
        self._last_observation = {key: value.copy() for key, value in observation.items()}
        if self.evaluator is not None and hasattr(self.evaluator, 'reset'):
            self.evaluator.reset(info)
        self._done = False
        return observation, info

    def _observe(self):
        meta, _ = snapshot(self.M, self.cloths, self.rigs)
        if not meta['all_arrays_finite']:
            raise RuntimeError('Nonfinite physics state; episode invalid, call reset')
        observation = pack_measured_state(meta, self.contract)
        cameras = []
        if self.sensors:
            sensor_values, cameras = self.sensors.capture()
            observation.update(sensor_values, previous_action=self._previous_action.copy())
        validate_observation(observation, self.contract, robots=len(self.rigs), joints=len(meta['robots'][0]['joints']), profile=self.profile)
        info = {'schema_version': self.contract['schema_version'], 'episode_seed': self.episode_seed,
                'episode_id': self._episode, 'step_id': self._step_id, 'human_spawn': deepcopy(self.spawn),
                'garment_spawn': deepcopy(self.garment_spawn),
                'episode_time_s': float(self.world.current_time) - self._start_time,
                'resolved_config': deepcopy(self.contract), 'source_sha256': dict(self._hashes), 'source_commit': self._source_commit, 'source_is_dirty': self._source_dirty,
                'environment_config': {'profile': self.profile, 'max_episode_steps': self.max_episode_steps,
                    'settle_ticks': self.settle_ticks, 'physics_hz': 1 / self.world.get_physics_dt(),
                    'control_hz': 60, 'policy_hz': 20},
                'state': meta, 'cameras': cameras, 'success_evaluated': False,
                'termination_reason': None, 'task_evaluator_configured': self.evaluator is not None}
        return observation, info

    def step(self, action):
        if self._closed or self._done:
            raise RuntimeError('Call reset before step (also after termination/truncation)')
        decoded = decode_action(action, self.contract, self._runtime, robots=len(self.rigs))
        before = float(self.world.current_time)
        try:
            for rig, command, toggle in zip(self.rigs, decoded, self._toggles):
                closed = bool(rig['state']['gripper_closed'])
                if resolve_gripper_intent(closed, command['gripper_intent']) != closed:
                    toggle()
            for _ in range(3):
                self._tick(decoded)
            if not np.isclose(float(self.world.current_time) - before, 0.05, atol=1e-7, rtol=0):
                raise RuntimeError('Policy step did not advance exactly 0.05 simulated seconds')
            self._previous_action = np.array(action, copy=True)
            self._step_id += 1
            observation, info = self._observe()
            reward, terminated = 0.0, False
            if self.evaluator is not None:
                reward, terminated, task_info = self.evaluator(self._last_observation, action.copy(), observation, info)
                reward, terminated = float(reward), bool(terminated)
                if not np.isfinite(reward):
                    raise ValueError('Evaluator returned nonfinite reward')
                info['task'] = task_info
                info['success_evaluated'] = bool(task_info.get('success_evaluated', False))
            truncated = self._step_id >= self.max_episode_steps and not terminated
            self._done = terminated or truncated
            info['termination_reason'] = 'task_evaluator' if terminated else 'time_limit' if truncated else None
            self._last_observation = {key: value.copy() for key, value in observation.items()}
            return observation, reward, terminated, truncated, info
        except Exception:
            self._done = True
            raise

    def render(self):
        if self._closed or self.sensors is None or not hasattr(self, '_last_observation'):
            raise RuntimeError('Reset an actor_rgbd environment first')
        return self._last_observation['rgb'][0].copy()

    def close(self):
        if not self._closed:
            self._closed = True
            self._done = True
            self._detach_callbacks()
            if self.sensors:
                self.sensors.close()
            # This can fast-exit: callers must save artifacts BEFORE close().
            self.M.simulation_app.close()
