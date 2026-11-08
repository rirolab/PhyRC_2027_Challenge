# Glossary: terms and numbers for first-time teams

## What does `(2,9)` mean?

The numbers in parentheses are the **shape of an array**. `(2,9)` is **2 rows × 9 columns**.
For actions in this environment, the 2 rows are the 2 robots and the 9 columns are one robot's 9 commands.
That is 18 commands in total, grouped by robot.

```text
row 0 = robot_0: [forward, left, base_yaw, lift, arm_extension, wrist_yaw, wrist_pitch, wrist_roll, gripper]
row 1 = robot_1: [forward, left, base_yaw, lift, arm_extension, wrist_yaw, wrist_pitch, wrist_roll, gripper]
```

Row and column indices in Python start at 0.

```python
import numpy as np

action = np.zeros((2, 9), dtype=np.float32)
action[0, 3] = 1.0   # raise robot_0's lift
action[1, 8] = -1.0  # ask robot_1's gripper to open
```

All other values are 0. For movement and rotation, 0 means no drive command; it does not mean the measured velocity instantly becomes 0.
For the gripper, 0 keeps the previous open/close intent.

## Are −1, 0, 1 velocities?

They are **normalized values** that express command size on a common scale, not measured velocities or joint angles.

| Value | Movement / rotation | Gripper |
|---|---|---|
| +1 | 100% of the configured command scale, positive direction | Request close |
| 0 | No drive command on that axis | Keep previous intent |
| −1 | 100% of the configured command scale, negative direction | Request open |
| +0.3 | 30% of the configured command scale, positive direction | Keep intent (the gripper uses its own thresholds) |

Keyboard keys are either pressed or released, so teleop data is mostly −1, 0, 1. A trained policy can also output values in between.
For example, with a lift scale of 1.54 m/s, `action[0,3] = 0.1` requests 0.154 m/s.
The robot's actual velocity, after acceleration, joint limits, and physics, shows up in the **observations**.
The gripper closes above 0.5, opens below −0.5, and keeps its previous intent in between.

## Reading other array shapes

| Notation | How to read it | Example here |
|---|---|---|
| `(2,9)` | 2 rows, 9 values each | The action sent to both robots |
| `(18,)` | A flat list of 18 values | The two action rows joined for storage. The trailing comma marks 1-D |
| `(N,18)` | N time steps, 18 values each | `actions` in HDF5. N=1,142 in the example |
| `(2,13)` | 2 robots, 13 joint values each | joint_position / joint_velocity |
| `(2,7)` | 2 robots, 7 pose values each | 3 for position + 4 for orientation |
| `(5,256,256,3)` | 5 cameras, height 256, width 256, 3 color channels | RGB images |
| `(5,256,256,1)` | The same 5 images, 1 value per pixel | depth or depth_valid |
| scalar | A single number | simulation_time_s |

The 5 cameras are 1 overview + robot_0 wrist/head (2) + robot_1 wrist/head (2).
HDF5 stores each camera under its own key, so a single `overview_rgb` frame is `(256,256,3)`.
When non-image arrays are flattened into CSV/HDF5, all of robot_0's values come first, then robot_1's.
`joint_position[13]` is the 14th value of the flattened array, which is robot_1's first joint.

### Why 13 joints but only 9 commands?

Joint state measures each joint in the robot individually. Commands are the functions a human or policy controls.
For example, one arm-extension command moves several arm joints together, and one gripper command drives both fingers.
So the number of joints in the observations does not have to match the number of action commands.

`controller_target` is also `(2,9)`, but its column order and units differ from the action.
It holds the internal controller's current position and velocity targets, while the action is the normalized command for the next step.
See the [data guide](DATA.md#public-observation-fields-and-frames) for each column.

## Observations, actions, policies, episodes

| Term | Meaning |
|---|---|
| GUI / headless | Running with a simulation window / without one. Headless runs still simulate physics and render cameras |
| viewport | The area of the Isaac Sim window that shows the 3D scene. Click it before driving so it receives keyboard input |
| teleop | A human driving the robots directly with the keyboard |
| demonstration | A recording of a human performing the task. It can succeed or fail |
| policy | A model or function that looks at the current allowed observations and decides the next robot command |
| observation (obs) | What the policy receives as input, such as camera images and robot positions and velocities |
| action | The command a policy or human requests for the robots in the next step |
| next_obs | The observation after that action has been applied |
| proprioception | Observations of the robot's own joints, base, and so on, as opposed to cameras |
| episode | One attempt, from start until it ends or is reset |
| transition | One obs → action → next_obs group; 0.05 s of simulation time here |
| randomization | Varying the starting layout within set ranges on each attempt |
| seed | A number used for randomization. It selects the same starting layout again under the same version and settings, but does not guarantee identical physics on another PC |
| inference | Computing the next action with a trained policy |
| rollout | Running a policy in the environment for one attempt |
| checkpoint | Saved model weights. Not the same as a save slot for the robot scene |

Within one episode, `next_obs[t]` equals the next row's `obs[t+1]`.
So N action steps give N+1 distinct observation times.
When a reset starts a new episode, do not join the last observation of one episode with the first observation of the next.

## Time: 20 Hz, 60 Hz, 240 Hz, and ticks

- **20 Hz policy**: 20 commands per second of simulation time. Each command is held for 0.05 s.
- **60 Hz control**: the internal controller runs 3 times during that 0.05 s.
- **240 Hz physics**: physics runs 12 times during that 0.05 s. One physics step is called a tick.

On a slow PC, computing 0.05 s of simulation might take 0.5 s of real time.
Data timestamps and scoring always use simulation time.
An MP4's fps (frames shown per second of playback), the 20 Hz recording rate, and the 240 Hz physics rate are three different numbers.
The 5 fps preview videos show only some of the images; they do not change the 20 Hz rate of the actual training data.

## Position, orientation, and image values

| Term | Meaning |
|---|---|
| world frame | A reference fixed to the whole scene. Z is up |
| base frame | Values relative to that robot's base. The frame moves with the robot |
| pose | Position and orientation together. Here, 7 values: x,y,z,qw,qx,qy,qz |
| quaternion | 4 values that represent orientation, in w,x,y,z order. Do not read them as angles |
| yaw / pitch / roll | Rotations about different joint axes. Keys and positive directions follow the action spec |
| m / rad | Meters / radians. About 3.14159 rad is 180° |
| twist | 3 linear velocities and 3 angular velocities. Not the position or angle itself |
| egocentric | The view from a camera mounted on the robot (wrist or head) |
| overview | The view from the allowed external camera looking at the scene |
| RGB-D | An observation with both color (RGB) and depth |
| RGB | 3 values per pixel: red, green, blue |
| depth | The optical Z depth of each pixel in meters. Not a brightness value |
| depth_valid / mask | true/false for whether a pixel's depth can be used |
| uint8 | An integer from 0 to 255. Used for RGB |
| float32 / float64 | 32/64-bit numbers with decimals. The type itself does not define a physical unit like m/s |
| bool | true or false. May appear as 1/0 in CSV |
| dtype | How an array's numbers are stored, e.g. uint8, float32, bool |
| HWC / CHW | Image array order: height, width, channels / channels, height, width |

Invalid depth is 0 with mask=false. Use the mask so a depth of 0 is not treated as an object right in front of the camera.
`gripper_close_command=true` means the gripper was told to close, not that it actually grasped the cloth.

## Files and scores

In `--training-record 1`, `1` turns training recording on and `0` turns it off.
`--evaluate 1` turns automatic evaluation on in the same way. The value 1 is not a number of demonstrations or a playback speed.
`<run-id>` in the docs is a placeholder for the actual folder name that was created.

| Term | Meaning |
|---|---|
| HDF5 | A training data file that bundles many arrays like folders. policy.hdf5 is the public training data |
| CSV | A text table. Open it in Excel or LibreOffice to see the numbers |
| audit data | Data for checking reproducibility, synchronization, and scores. Not a policy input |
| raw archive | The per-tick raw state and input log. For replay and verification; must not be replayed as a policy |
| deferred export | The default collection mode: state is saved while you drive, and images are rendered after ESC |
| READY | The terminal message that image and training file generation is complete |
| reward | The reward field for learning. 0 by default in collected files and unrelated to the evaluation score |
| done | Marks the end of an episode. Not a success flag |
| raw_points | Cumulative total including pickup, max 50 |
| rate_points | The points/s numerator excluding pickup, max 45 |
| points/s | Points excluding pickup, divided by the time from first contact to the last dressing award |
| success | Whether the evaluator's dressing success condition was reached |

**ESC ends teleop; READY means the files are finished.** After ESC, keep the terminal open until READY.
[Data details](DATA.md) · [Scoring](EVALUATION.md) · [Allowed and prohibited information](RULES.md)
