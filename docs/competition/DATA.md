# Training data guide

This guide explains how to turn the files you collect into training data, and what the numbers mean.
If the terms are unfamiliar, start with the [glossary](GLOSSARY.md).
[Real CSV example](../examples/teleop/README.md) · [Column dictionary for the example](../examples/teleop/columns.csv)

## From collection to a finished dataset

```bash
./run.sh gui --training-record 1
# To also see the score while driving:
./run.sh gui --training-record 1 --evaluate 1
```

Recording starts immediately. Pressing ESC starts the image export, and `[Dataset export] READY` means everything is done.

> [!IMPORTANT]
> **A closed Isaac Sim window does not mean your data is saved.** Do not close the terminal or press `Ctrl+C` until `[Dataset export] READY ...` appears.
> If you interrupted it, see [retrying the export](#retrying-the-export-and-checking-files) below.

Check that `capture.json` shows `status=ready` and `complete=true`. Other states: `recording` (still collecting),
`awaiting_export` (waiting for export), and `incomplete` (the run ended before a valid step finished).
Export re-renders the saved raw state. It does not re-simulate the demonstration with new physics;
scene state, camera poses, and simulation time at each observation are exactly those of the original run.

Use **`--training-record 1`** for training data.
The default mode, which renders after you finish driving, is recommended. `--training-render live` renders RGB-D while you drive and can be slow.

To try again while driving, click the 3D view and press **`P`**.
The scene resets, and the attempts before and after the reset are saved as separate episodes (`demo_0`, `demo_1`, ...).
When you are done, quit with ESC and wait for READY.

### Raw log for replay and verification

A training recording also creates `full_teleop/<run-id>/`. It holds per-physics-step robot poses and joint states,
cloth vertex positions and velocities, control inputs, and grasp state.
You can replay or re-score the scene from it. For policy training, use the allowed observations and actions in `policy.hdf5`.

`./run.sh gui --full-record 1` records only this raw state log.
It does not produce RGB/depth images or training HDF5.
You never need it together with `--training-record 1`, which already saves the raw log.

## Timing: what one row means

```text
obs[t] ── action[t] applied for 0.05 s ── next_obs[t] (= obs[t+1] in the same episode)
```

**Hz is steps per second of simulation time**, so 20 Hz is once every 0.05 s.
Each policy command is held for 3 controller steps at 60 Hz, or 12 physics steps at 240 Hz.
One physics step is called a tick, so `end_tick - start_tick = 12` means one action step completed.
Key changes mid-step take effect at the next policy step. Do not build labels by skipping frames or averaging actions.
0.05 s is simulation time: even if a slow PC takes 0.5 s of real time, the data spacing is still 0.05 s.
Depth, RGB, and robot state are aligned to the same observation time. Because of a few warm-up ticks, the first `simulation_time_s` can be slightly above 0.

## HDF5 layout

```text
policy.hdf5
  data/demo_0/
    obs/<public observation key>       (N, ...)
    next_obs/<public observation key>  (N, ...)
    actions                            (N,18), float32
    rewards                            (N,), 0 by default
    dones                              (N,), marks the last step
    terminated, truncated              (N,)
  data/demo_1/ ...                     next attempt after pressing P
```

**`(N,18)` means N rows × 18 columns.** Each row is the 18 commands sent to both robots at one moment.
`N` is the number of completed action steps. `obs` and `next_obs` each have N entries, so a continuous episode has N+1 distinct observation times.
The loader rejects incomplete files and episodes. `dones=1` does not mean the dressing succeeded,
and the reward of 0 is only a file-format placeholder, unrelated to the evaluation score.

The layout follows the [robomimic dataset format](https://robomimic.github.io/docs/datasets/overview.html).
You still need model-specific image preprocessing and a rollout adapter to run your policy in this environment. Split train/validation by episode rather than mixing individual frames.

## Public observation fields and frames

HDF5 flattens non-image arrays. For example, `(2,13)` joint positions become `(26,)`.
The order is **all of robot_0's values, then all of robot_1's values**, and quaternions are **w, x, y, z**.
The world frame is Z-up, lengths are in meters. The base frame is X forward / Y left / Z up; camera optical frames are X right / Y down / Z forward.

| Field | Shape at run time | Flattened columns / units |
|---|---|---|
| joint_position | `(2,13)` | The 13 joint positions per robot listed below, m or rad |
| joint_velocity | `(2,13)` | Same order, m/s or rad/s |
| base_pose_world | `(2,7)` | Per robot x,y,z,qw,qx,qy,qz |
| base_twist_world | `(2,6)` | Per robot vx,vy,vz,wx,wy,wz (m/s, rad/s) |
| grasp_pose_base | `(2,7)` | Gripper center in the robot base frame, x,y,z,qw,qx,qy,qz |
| fingertip_position_base | `(2,2,3)` | Per robot, xyz of fingertip link 0 then link 1 (m) |
| fingertip_origin_distance | `(2,1)` | Distance between the two fingertip link origins (m), not the gap between contact surfaces |
| controller_target | `(2,9)` | Target columns listed below; not the measured joint positions |
| gripper_close_command | `(2,1)` | bool, intent to close; not a grasp-success sensor |
| previous_action | `(2,9)` | Normalized action from the previous step; 0 at the start |
| simulation_time_s | scalar | Simulation time (s); a single element in HDF5/CSV |

Joint order per robot in the current example is below. Always interpret joints using the names and units of the version you submit with.

| Index within robot | Joint | Position unit |
|---:|---|---|
| 0,1,2 | wheel_0_joint, wheel_1_joint, wheel_2_joint | rad |
| 3 | lift_joint | m |
| 4,5,6,7 | arm_l1_joint, arm_l2_joint, arm_l3_joint, arm_l4_joint | m |
| 8,9,10 | wrist_yaw_joint, wrist_pitch_joint, wrist_roll_joint | rad |
| 11,12 | gripper_finger_left_joint, gripper_finger_right_joint | rad |

For example, `joint_position[3]` is robot_0's lift and `[16]` is robot_1's lift.
Base actions do not directly set the wheel velocities in the joint array.

`controller_target` columns per robot:
`lift (m), arm (total extension, m), yaw (rad), pitch (rad), roll (rad), grip_pos (rad), base_fwd (m/s), base_strafe (m/s), base_turn (rad/s)`.
Note that `base_strafe` here is positive to the right, while the action `base_left_velocity` is positive to the left.
Do not compare their signs directly.

### Cameras

| Camera ID | Location | Valid depth range |
|---|---|---|
| overview | External camera facing the manikin; placed for the initial spawn, then fixed | 0.05–10 m |
| robot_0_wrist | robot_0 wrist | 0.03–3 m |
| robot_1_wrist | robot_1 wrist | 0.03–3 m |
| robot_0_head | robot_0 head | 0.05–5 m |
| robot_1_head | robot_1 head | 0.05–5 m |

Robot cameras move with their robot. Every camera provides RGB, depth, and depth_valid.
HDF5 keys look like `overview_rgb`, `robot_0_wrist_depth`, `robot_1_head_depth_valid`.
RGB is `(256,256,3)` uint8; depth and mask are `(256,256,1)` float32 and bool.
Depth is optical Z distance, not Euclidean ray length. Invalid depth is 0 with mask=false.
Camera matrices and the free viewport are audit information, separate from the allowed image observations.

## Actions: why you see −1, 0, 1

Keyboard input is either pressed or released, so movement and rotation actions are −1, 0, or 1.
Pressing both directions at once cancels out to 0. A policy may use any continuous value in `[-1,1]`.
In the CSV, the 9 `robot_0.*` columns followed by the 9 `robot_1.*` columns give the HDF5 `(18,)` order.

| Column (per robot) | Key | Physical meaning |
|---:|---|---|
| 0 | base_forward_velocity | value × BASE_LINEAR_RATE, forward m/s |
| 1 | base_left_velocity | value × BASE_LINEAR_RATE, left m/s |
| 2 | base_yaw_velocity | value × BASE_ANGULAR_RATE, counter-clockwise rad/s |
| 3 | lift_velocity | value × LIFT_RATE, up m/s |
| 4 | arm_extension_velocity | value × ARM_RATE, total extension of the 4 telescoping joints, m/s |
| 5 | wrist_yaw_velocity | value × WRIST_RATE, rad/s |
| 6 | wrist_pitch_velocity | value × WRIST_RATE, rad/s |
| 7 | wrist_roll_velocity | value × WRIST_RATE, rad/s |
| 8 | gripper_command | >0.5 close, <−0.5 open, otherwise keep previous intent |

The scales recorded in the example are BASE_LINEAR_RATE=0.616, BASE_ANGULAR_RATE=2.86,
LIFT_RATE=1.54, ARM_RATE=1.21, WRIST_RATE=5.5.
These convert the requested command. Measured velocities can differ after joint limits, acceleration, and physics.
Each recording's `resolved_runtime_rates` takes precedence. Do not change the competition scales.

In the example actions CSV, sample 247 has robot_1.gripper_command=1. That is the moment a close was requested;
the 0s after it keep the close intent rather than opening. Wrist yaw and roll were not used in this demonstration, so they are all 0 for both robots.
This one example is not diverse enough to learn every motion from.

## CSV viewer files

| File | One row is |
|---|---|
| demo_0_actions.csv | One action step: sample, obs/next times, start/end tick, 18 normalized commands |
| demo_0_obs.csv | All non-image public observations just before the step |
| demo_0_next_obs.csv | All non-image public observations just after the step |
| demo_0_applied_targets.csv | One controller step; each sample has control_substep 0, 1, 2. Audit only |
| demo_0_evaluation.csv | Score, denominator, and status at the end of the step. For selecting and analyzing demonstrations |
| video_frame_index.csv | Maps preview MP4 frames to the original observation and tick |
| columns.csv | Definition, unit, and example value for every column above |

The CSVs are UTF-8 with BOM so they open in Excel or LibreOffice. `-0.0` is numerically 0.
`sample` is the row number within an episode, starting at 0; `physics_tick` is the physics tick in the raw log.
`first_contact_tick` counts from the start of the evaluation episode, and −1 means no contact yet.
`score_time_s` runs from first contact to **the last dressing award**, which is not the same as video time or total episode time.

## Training data vs. audit data

Train on the allowed observations and actions in `policy.hdf5`.
Ground-truth object state, camera poses, and applied targets in `audit.hdf5`, `evaluation/`, and the raw archive are for verification and reproducibility, not for policy input or generating action labels.
To pick successful demonstrations, check that the audit metadata has `success_known=true` and `success=true`.
This records that the success condition was reached under the current evaluation rules; it does not guarantee the shirt is still on in the last frame.

## Retrying the export and checking files

Replace `<run-id>` with the name of your recording folder.

```bash
# Recording ended normally but image export did not finish
./run.sh python /scripts/export_policy_dataset.py /output/full_teleop/<run-id>

# Check the public observation/action spec of the HDF5
./run.sh cpu /scripts/inspect_policy_dataset.py \
  /output/policy_datasets/<run-id>/policy.hdf5

# Check that observations, actions, and cameras match the raw log
./run.sh cpu /scripts/check_deferred_dataset.py \
  /output/full_teleop/<run-id> /output/policy_datasets/<run-id>
```

`passed=true` means storage and synchronization checks passed; it says nothing about dressing success.
The exporter never overwrites a finished HDF5. Existing files are kept; use `--output /output/<another-folder>` if needed.
Incomplete raw logs are not exported automatically. The normal fix is to collect again.
