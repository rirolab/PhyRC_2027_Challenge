# Rules: allowed information and actions

Policy training and execution may use only the public observations in the [data guide](DATA.md) and the `(2,9)` action in
[`config/policy_interface.json`](../../config/policy_interface.json).
These rules apply to generating training data, training, and inference.

## Allowed

- The designated overview camera and the wrist and head cameras of both robots (5 in total): RGB, depth, depth_valid.
- Joint positions and velocities, **each robot's own** base pose and velocity, gripper and fingertip state in the base frame, public controller targets and gripper intent, the previous action, and simulation_time_s.
- Static camera calibration, joint names and units, and public action scales needed to interpret the spec.
- Models and controllers that **estimate** object or garment positions and features from allowed observations.
- Training on demonstrations of public observations and actions recorded by humans through teleop.
- Using scores and success labels to select demonstrations and analyze results. These labels must not be used as policy inputs or to create action targets.

The robot base's world pose counts as allowed localization in this track; using it is not a violation.
That is different from reading ground-truth poses of the manikin, shirt, or chair, or overwriting recorded state to reproduce behavior.

## Not allowed

1. Giving the policy cloth vertex/particle positions or velocities, exact shirt/manikin/chair positions, ground-truth segmentation,
   collision meshes, internal grasp anchors or success state, or the evaluator's geometric checks.
2. Using that internal information to create target points, action labels, or automatic demonstrations, or to train and distill a teacher policy.
3. Replaying robot or cloth state trajectories from the raw archive as if they were answers, or directly setting USD or simulator state.
   The policy must receive allowed observations every 20 Hz step and output an action that follows the spec.
4. Using audit files, evaluation JSON, the freely moved teleop viewport, episode spawn seeds,
   or dynamic camera world matrices that reveal object positions as training or inference inputs.
5. Changing the physics, geometry, actuation limits, or reset rules of the robots, grippers, shirt, or manikin, or the evaluation functions, to improve results.

Do not manipulate the scene through any path other than the adapter that returns actions.
Research and debugging scripts in this repository that read internal ground truth are not allowed as training baselines.

## Verification and submission

**Submissions found to use prohibited information or bypass the environment during the organizers' own verification process will be disqualified.**
You must be able to provide your training code, data sources and how they were generated, checkpoints, dependencies, run instructions, and logs.
The organizers may review these materials and re-evaluate from different initial states. A high score alone is never treated as evidence of a violation.

Passing the local runner's schema checks does not certify compliance.
Final decisions follow the organizers' verification. Local practice seeds and recordings are not the final judging data.
Where to submit, deadlines, and final run limits will be announced with the competition.
