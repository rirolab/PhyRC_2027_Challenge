# PhyRC 2027 Challenge · Phase 1 Participant Guide

**Challenge website: [emprise.cs.cornell.edu/phyrc/challenge-2027](https://emprise.cs.cornell.edu/phyrc/challenge-2027/)** (schedule, registration, and announcements)

This is an Isaac Sim 6.0.1 environment where two **Stretch4 robots** work together to put a T-shirt on a manikin.
You collect demonstrations by driving the robots with the keyboard (**teleop**).
You then use those demonstrations to train a **policy**: a model that looks at camera images and robot state and decides what to do next.
At run time, the policy sends **9 control commands per robot, 18 in total** for the two robots.
This repository contains the simulation environment and guides for setup, teleop, data collection, running a policy, and evaluation.

[Glossary](docs/competition/GLOSSARY.md) · [Setup](docs/competition/SETUP.md) · [Data & CSV columns](docs/competition/DATA.md) · [Running a policy](docs/competition/POLICY.md) · [Evaluation](docs/competition/EVALUATION.md) · [Rules](docs/competition/RULES.md) · [Example data](docs/examples/teleop/README.md)

## Teleop demo

![Two robots dressing the manikin in a T-shirt, successful episode, 32x speed](docs/videos/Back_Front_32x.gif)

A **successful episode** in which the two robots dress the manikin together, shown at **32x speed**.

## 1. Install and run

You need **Linux x86-64, an NVIDIA RTX GPU, Docker, and the NVIDIA Container Toolkit**.
You do not need to install Isaac Sim, ROS, Conda, or the CUDA Toolkit on the host.
The [setup guide](docs/competition/SETUP.md) walks through every step.

Once Docker can see your GPU:

```bash
git clone \
  https://github.com/rirolab/PhyRC_2027_Challenge.git
cd PhyRC_2027_Challenge
./run.sh doctor
./run.sh build
./run.sh prepare
./run.sh smoke
./run.sh gui
```

Before you start, read [NVIDIA's container terms and install notes](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/installation/install_container.html) and the [third-party asset notes](THIRD_PARTY.md).

## 2. Teleop controls

Click inside the 3D view (the viewport) of the Isaac Sim window so it receives keyboard input. The robots are named **robot_0 / robot_1**, the same names used in the data.
You can drive both robots at the same time. robot_1 needs a numeric keypad.

| Action | robot_0 | robot_1 |
|---|---|---|
| Forward / back | `I` / `K` | `↑` / `↓` |
| Move left / right | `J` / `L` | `←` / `→` |
| Rotate base + / − | `U` / `O` | Keypad `/` / `*` |
| Lift up / down | `W` / `S` | Keypad `8` / `2` |
| Extend / retract arm | `D` / `A` | Keypad `6` / `4` |
| Wrist yaw + / − | `E` / `Q` | Keypad `9` / `7` |
| Wrist pitch + / − | `V` / `R` | Keypad `+` / `-` |
| Wrist roll + / − | `C` / `Z` | Keypad `3` / `1` |
| Toggle gripper | `Space` or top-row `0` | Keypad `0` or `Enter` |
| Reset scene and start a new attempt | `P` | (shared) |
| Quit and save | `ESC` | (shared) |

On a slow PC, simulation time can run slower than real time.
Timestamps in the data and all scoring use **simulation time**.

## 3. Collect training data

```bash
./run.sh gui --training-record 1
```

1. Recording starts automatically. There is no record button.
2. Perform your demonstration, then press **`ESC` in the Isaac Sim window**.
3. After the window closes, RGB-D images and HDF5 files are generated. **Keep the terminal open until you see `[Dataset export] READY ...`.**

> [!IMPORTANT]
> **A closed Isaac Sim window does not mean your data is saved.** Do not close the terminal or press `Ctrl+C` until `[Dataset export] READY ...` appears.
> Export can take a while depending on the recording length and your PC. If you stop it early, the training files will be incomplete.

| Goal | Command |
|---|---|
| Practice driving | `./run.sh gui` |
| Collect training data | `./run.sh gui --training-record 1` |
| Watch live scoring from the start | `./run.sh gui --evaluate 1` |
| Collect training data with live scoring | `./run.sh gui --training-record 1 --evaluate 1` |
| Practice with a fixed starting layout | `./run.sh gui --no-randomization` |

`--training-record 1` saves camera images, robot state, and actions for training.
It also saves the raw state log needed for replay and verification. By default, images are rendered after you finish driving, which keeps teleop responsive.
`--training-render live` renders the HDF5 data while you drive instead, at a higher rendering cost.
Training recordings always save evaluation results. Add `--evaluate 1` to also see the score change while you drive;
the live score and the post-export score are then saved separately.

`./run.sh gui --training-record 1 --evaluate 1` writes the following:

```text
output/
  full_teleop/<run-id>/          raw state and input log, for replay and verification
  policy_datasets/<run-id>/
    policy.hdf5                observations and actions for policy training
    audit.hdf5                 state, images, and scores for verification (never a policy input)
    capture.json               collection and export status
    evaluation/                per-episode evaluation results
  evaluation/teleop/<eval-id>/   live evaluation from --evaluate 1
```

**To try again, click the 3D view and press `P`.**
The scene resets and a new attempt begins. Each attempt is saved as a separate episode in the training file:
the first attempt is `demo_0`, the attempt after pressing `P` is `demo_1`, and so on.

If you force-quit the window or process, the file is marked `complete=false` and the training loader rejects it. If collection finished normally and only the export was interrupted, see [retrying the export](docs/competition/DATA.md#retrying-the-export-and-checking-files).

## 4. Observations

The policy runs at **20 Hz**, the internal controller at **60 Hz**, and physics at **240 Hz**.
Hz means steps per second of simulation time, so 20 Hz is one command every 0.05 s.
Each action is held for 0.05 s: 3 controller steps and 12 physics steps.
Data is stored as `obs[t] → action[t] → next_obs[t]`, with cameras and robot state captured at the same simulation time.

| Data | Shape at run time | Meaning |
|---|---|---|
| RGB | `(5,256,256,3)` | 1 overview camera + 2 wrist and 2 head cameras, uint8 RGB |
| Depth / valid mask | `(5,256,256,1)` each | Optical Z distance in m (float32) / bool. Invalid depth is 0 |
| Joint position / velocity | `(2,13)` each | Per joint, m or rad / m/s or rad/s |
| Robot base pose | `(2,7)` | World xyz (m) + quaternion wxyz |
| Robot base velocity | `(2,6)` | World linear velocity xyz + angular velocity xyz |
| Gripper pose | `(2,7)` | xyz + quaternion wxyz in the robot base frame |
| Fingertip positions / distance | `(2,2,3)`, `(2,1)` | Link origins in the base frame, m |
| Controller target | `(2,9)` | Current controller setpoints, not measured positions |
| Gripper close command | `(2,1)` | bool. The intent to close, not whether anything was grasped |
| Previous action | `(2,9)` | Normalized command from the previous step |
| Simulation time | scalar | Seconds |

Exact keys, column order, and units are in the [data guide](docs/competition/DATA.md) and the [interface spec](config/policy_interface.json).
**Use only these observations and the actions below for training and inference.**

### RGB-D examples

#### RGB

![RGB from the 5 policy cameras](docs/videos/sample_cameras_rgb.gif)

From left to right: **overview → robot_0 wrist → robot_1 wrist → robot_0 head → robot_1 head**.

#### Depth

![Depth previews from the same moment](docs/videos/sample_cameras_depth.gif)

Brighter is closer, using each camera's display range.
Black means invalid or near the maximum range, so use `depth_valid` alongside depth when training.
The GIFs show a short excerpt. The actual training data is recorded at 20 Hz with no compression or dropped frames.

The example data is a real recording of **1,142 transitions (57.1 s of simulation time)**.
It is a partial demonstration that only earned the 5-point pickup; it is not a successful demonstration or policy.
See the [CSV files and viewer page](docs/examples/teleop/README.md) for details.

## 5. Actions

**`(2,9)` is the action shape: 2 robots × 9 commands per robot.**
Row 0 is robot_0 and row 1 is robot_1. The 9 columns in each row map to the commands in the table below.

```text
             fwd  left  base_yaw  lift  arm  wrist_yaw  pitch  roll  gripper
robot_0:     [ 0,    0,       0,     1,    0,         0,     0,    0,       0 ]
robot_1:     [ 0,    0,       0,     0,    0,         0,     0,    0,       0 ]
```

This example raises robot_0's lift and sends no drive command on any other axis.
A gripper value of 0 keeps the previous open/close intent. In Python this is `action[0,3] = 1` (indices start at 0).

Commands are `float32` in the range **−1 to +1**.
When saved, the two rows are concatenated into a flat `(18,)` vector.
A stack of commands over time has shape `(N,18)`, where `N` is the number of action steps; N is 1,142 in the example.

| Column (per robot) | Command | Positive direction / meaning |
|---|---|---|
| 0 | base_forward_velocity | Forward |
| 1 | base_left_velocity | Robot's left |
| 2 | base_yaw_velocity | Counter-clockwise about base +Z |
| 3 | lift_velocity | Lift up |
| 4 | arm_extension_velocity | Extend arm |
| 5–7 | wrist_yaw / pitch / roll_velocity | Positive rotation of that joint |
| 8 | gripper_command | +1 close, −1 open, 0 keep previous intent |

For movement and rotation, keyboard demonstrations use **+1 = drive in the positive direction, −1 = negative direction, 0 = no drive command**.
Because keys are either pressed or not, you will only see `−1, 0, 1` in teleop data. A policy can also output values in between, such as `0.3`.
`lift_velocity=1` does **not** mean 1 m/s; it requests 100% of the configured lift speed.
Actual positions and velocities are in the observations and can differ from the request because of physics and joint limits.
A gripper value of 0 is **not** "open"; it keeps the previous intent. A close request is not repeated every frame.

[actions CSV](docs/examples/teleop/demo_0_actions.csv) · [obs CSV](docs/examples/teleop/demo_0_obs.csv) · [next_obs CSV](docs/examples/teleop/demo_0_next_obs.csv) · [all CSV columns explained](docs/examples/teleop/columns.csv)

Each CSV row is one 0.05 s action step. `obs` is taken just before the step and `next_obs` just after.
`[i]` in a CSV column name is the index into the array flattened in robot order. For example, `joint_position[13]` is robot_1's first joint.
The [data guide](docs/competition/DATA.md) explains every field, index, and unit with real example values.

## 6. Train and run a policy

`policy.hdf5` uses the robomimic-style layout `data/demo_N/{obs,next_obs,actions,rewards,dones}`.
The provided imitation-learning loader returns only the allowed observations and actions. RGB is uint8 (0–255) in HWC order (height, width, channels), so preprocess it for your model.
Keep depth in meters or normalize it explicitly. Split train/validation by episode, and keep track of which demonstrations succeeded.
`reward=0` is just a placeholder for the file format, and `done=1` marks the end of an episode, not a successful dressing.

After training, write a [policy adapter](docs/competition/POLICY.md) and evaluate it:

```bash
./run.sh python /scripts/evaluate_policy.py \
  --policy /project/participant/policy.py \
  --checkpoint /output/checkpoints/policy.pt \
  --seeds 42 43 --seconds 60 --output /output/evaluation/my_policy
```

Replace the policy and checkpoint paths with your own files.
Local scores on the public practice seeds are not final scores. For submission, prepare your code, checkpoint, dependencies, data sources, and run instructions so the results can be reproduced.
Schedules, final seeds, where to submit, and time limits will be announced by the organizers on the [challenge website](https://emprise.cs.cornell.edu/phyrc/challenge-2027/).

## 7. Scoring and points/s

```bash
# Score from the start with no button press; ESC ends and saves the result
./run.sh gui --evaluate 1
```

| Item | Points | Condition |
|---|---:|---|
| Pick up | 5 | Lift the grasped part with the same gripper and hold it for 3 s |
| First sleeve | 5 | The first hand comes out through a sleeve opening |
| Opposite shoulder | 5 | Part of the shirt passes over the opposite shoulder joint |
| Second sleeve | 5 | The other hand comes out through the other sleeve |
| Overall: left upper arm | 5 | That sleeve covers part of the upper arm |
| Overall: right upper arm | 5 | That sleeve covers part of the upper arm |
| Overall: neck | 10 | Awarded as soon as the head passes through the neck opening |
| Overall: V-neck facing front | 10 | With the head out, the front faces the right way for 0.5 s |
| **Total** | **50** | Overall is the last 4 items, 30 points |

Forearms are not scored. Points, once earned, are kept. A successful demonstration means both hands, the neck, and the V-neck direction are all confirmed at the same time; this is separate from reaching 50 points.

```text
points/s = (total points − pickup points actually earned)
           / (time of last dressing award − time of first shirt–manikin contact)
```

Time is measured in simulation ticks. The numerator excludes pickup, so its maximum is 45, and waiting after your last award does not grow the denominator.
Contact is detected using the collision meshes of the shirt and manikin. Attempts with no contact, or with a zero denominator, count as 0 when averaging over episodes.
The [evaluation guide](docs/competition/EVALUATION.md) covers failure handling, success criteria, result fields, and replay scoring.

## 8. Allowed information and fair play

Train on the allowed **RGB, depth, mask, robot state, and previous action**, and control the robots only through the **`(2,9)` action** in the spec.
Observing the robot's own base pose and joint state is allowed. So is estimating object positions from images.

The following are not allowed:

- Reading cloth vertices/particles, exact shirt/manikin/chair poses, collision meshes, or internal grasp or evaluation ground truth for control.
- Using that ground truth to generate demonstrations, action labels, or teacher policies.
- Replaying or directly setting recorded robot or cloth state trajectories and submitting that as a policy.
- Using `audit.hdf5`, raw replays, or the freely moved teleop viewport as policy input.
- Changing the physics or actuation limits of the robots, shirt, or manikin in the evaluation environment, or tampering with evaluation results.

**Submissions found to break these rules during the organizers' verification will be disqualified.**
The organizers may review training code, data sources and how they were generated, checkpoints, and run logs, and may re-evaluate from different initial states.
Please read the full [rules](docs/competition/RULES.md).

Shapes, Hz, ticks, obs/next_obs, quaternions, and other terms are explained with examples in the [glossary](docs/competition/GLOSSARY.md).

## Getting help

Open an issue in this repository's [Issues tab](https://github.com/rirolab/PhyRC_2027_Challenge/issues) or email phyrc.challenge@gmail.com.
For setup problems, include your OS, GPU, driver version, the command you ran, and the last error message.
For data problems, include the status from `capture.json` and the run ID. Never share passwords or tokens.
Full HDF5 recordings and raw archives are large, so do not commit them to Git.

## License

The code in this repository is licensed under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE).
Robot, manikin, and garment assets and downloaded third-party files are not covered by this license and keep their own terms; see [THIRD_PARTY.md](THIRD_PARTY.md).
