# Training and running a policy

What you may use for training inputs and actions is defined in the [rules](RULES.md).
The environment provides public observations every 0.05 s of simulation time (20 Hz), and your policy outputs commands for both robots at once.
The output shape `(2,9)` means 9 commands for each of the 2 robots.
The [glossary](GLOSSARY.md) explains observations, actions, transitions, and file formats.

## 1. Loading data

`PolicyDataset` returns only the public observations and actions from a finished `policy.hdf5`.
You can run the example below from the repository root. Install NumPy, h5py, and whatever training framework you prefer.

```python
import sys
sys.path.insert(0, 'src/PhyRC_Sim')
from Policy.dataset_reader import PolicyDataset

path = 'output/policy_datasets/<run-id>/policy.hdf5'
dataset = PolicyDataset(path, observation_keys=[
    'overview_rgb', 'robot_0_wrist_rgb', 'robot_1_wrist_rgb',
    'robot_0_head_rgb', 'robot_1_head_rgb',
    'joint_position', 'joint_velocity', 'previous_action',
])
obs, action = dataset[0]
# obs['overview_rgb']: uint8 (256,256,3)
# obs['joint_position']: float32 (26,)
# action: float32 (18,), robot_0's 9 values followed by robot_1's 9
print(len(dataset), action.shape)
dataset.close()
```

You can add other allowed observation keys or depth. Do not add audit files or raw archives as inputs.
In imitation learning, the model predicts from `obs` the action the human applied for the next 0.05 s.
Model architecture, preprocessing, loss, and framework are up to you.

- RGB: convert uint8 HWC to CHW/float as your model expects, usually dividing by 255.
- Depth: handle float32 meters together with the valid mask, so invalid zeros are not mistaken for nearby objects.
- Robot state: compute normalization statistics on training data and reuse the same statistics at inference.
- Action: keep the original `[-1,1]` spec. Do not reorder robots or joints.
- Consecutive frames are highly correlated, so do not split neighboring frames of the same episode across train and validation.
- Collect many starting layouts and attempts. The single partial demonstration we provide is not proof that full dressing can be learned from it.
- Models that use past observations or actions are fine for long tasks. Reset their memory at the start of each episode.

The default `rewards=0` is not a dressing reward designed for RL.
Use scores and success labels to select and analyze demonstrations, not to create action labels from internal ground truth.

## 2. Inference adapter: connecting your model to the simulator

The adapter is glue code that turns simulator observations into model inputs and model outputs into robot commands.
Write a Python file whose `load_policy(checkpoint, observation_space, action_space)` returns a callable policy object.
`checkpoint` is your saved model weights, and `observation_space` and `action_space` describe input and output shapes and ranges.
At run time the evaluator passes observations in their runtime shapes, which differ from CSV/HDF5:
joints `(2,13)`, action `(2,9)`, RGB `(5,256,256,3)`, depth `(5,256,256,1)`.
Camera order is overview, robot_0 wrist, robot_1 wrist, robot_0 head, robot_1 head.
HDF5 stores each camera under its own key, so apply matching preprocessing in both places.

```python
# participant/policy.py
import numpy as np

def load_policy(checkpoint, observation_space, action_space):
    # Load your model, checkpoint, and the preprocessing you used in training.
    model = load_your_model(checkpoint)

    class Policy:
        def reset(self):
            # Clear per-episode memory such as recurrent state or action history
            reset_your_model_state(model)

        def __call__(self, obs):
            inputs = preprocess_public_observations(obs)
            prediction = predict_your_action(model, inputs)
            return np.asarray(prediction, dtype=np.float32).reshape(2, 9)

    return Policy()
```

`load_your_model` and the other placeholders are functions you implement.
The evaluator loads the adapter after Isaac Sim starts, so you can import Torch or other model dependencies inside it.
Outputs must pass shape, range, and finiteness checks (no NaN or infinity). Clip your model's outputs to the valid range in your adapter.
Inference time counts against the episode time limit: each step costs 0.05 s of simulation time plus the time your policy call took, and the episode ends once the total passes `--seconds`.
The policy only receives public observations. Do not bypass this by reading other simulator APIs in the same process.

## 3. Local evaluation

To check that the interface runs at all, use the zero-action example:

```bash
./run.sh python /scripts/evaluate_policy.py \
  --policy /project/examples/zero_policy.py \
  --seeds 42 43 --seconds 1 --output /output/evaluation/adapter_check
```

This is not a trained dressing policy, so a score of 0 is expected. Evaluate your trained policy like this:

```bash
./run.sh python /scripts/evaluate_policy.py \
  --policy /project/participant/policy.py \
  --checkpoint /output/checkpoints/policy.pt \
  --seeds 42 43 --seconds 60 --output /output/evaluation/my_policy
```

`--seconds` is the episode time limit, counting simulation time plus your policy's own inference time. `--seeds` picks random starting layouts; give one or more different seeds.
Observations default to `actor_rgbd`. For experiments that use only the allowed non-image state, run with `--profile measured_state`.
Multi-seed evaluation requires randomization, so `--no-randomization` and `STRETCH4_RANDOMIZE=0` cannot be used here.

Results are saved per run in `scores.json`, along with measurement traces and per-episode evaluation JSON.
Local practice scores are not the final judging. Running multiple environments in one process, or running the GUI and GPU evaluation at the same time, is not supported.

## Preparing a submission

Submit a `.zip` as often as you like before the deadline; only your last accepted submission is evaluated. Unzipped, it looks like this:

```text
submission.zip
├── policy.py          required, at the top level; defines load_policy(...)
├── checkpoint.pt      optional; passed to load_policy as checkpoint
├── Dockerfile         optional; only if you need extra packages
└── ...                any other code or files your policy needs
```

- `checkpoint` is the file whose name starts with `checkpoint.`, or `None` if there is none. Load other files relative to `policy.py`, for example `Path(__file__).parent / 'encoder.pt'`.
- To import your own modules, add the folder to the import path at the top of `policy.py`: `sys.path.insert(0, str(Path(__file__).parent))`.
- If you need system or Python packages that are not in the official image, include a `Dockerfile` that extends it. We build it with your submission folder as the build context and evaluate in that image; without one, we use the official image.
  The official image is `phyrc-2027:isaac-6.0.1`. It is not downloaded from anywhere: `./run.sh build` creates it on your machine from this repository's `Dockerfile` ([setup step 5](SETUP.md#5-build-the-environment)).
  Run `./run.sh build` first; `docker image ls phyrc-2027` should then list it. Write your own `Dockerfile` in your submission folder; do not edit the repository's `Dockerfile`. Yours must start with `FROM phyrc-2027:isaac-6.0.1` and only add packages.
  Do not reinstall `torch` or `torchvision`, or change the versions of packages already in the image.

```dockerfile
FROM phyrc-2027:isaac-6.0.1
USER root
RUN apt-get update && apt-get install -y --no-install-recommends <system packages>
RUN /isaac-sim/python.sh -m pip install --no-cache-dir <package>==<version>
USER isaac-sim
```

- Put all weights in the zip. Nothing is downloaded during evaluation.

Each submission runs on one RTX 3090 (24 GB). The simulator uses about 10 GB of GPU memory, so keep your policy within 12 GB.
The final evaluation runs every team's last submission on the same three hidden seeds. Leaderboard checkpoints on Nov 17, Dec 1, Dec 4, Dec 8 and Dec 12 (23:59 AoE) score each team's latest submission on one hidden seed; they show progress and do not count toward the ranking.
The episode limit is 900 s, counting simulation time plus your policy's inference time. A submission whose image fails to build, or whose run crashes or produces no valid score, scores 0.

Before submitting, unzip your submission into `participant/` in a fresh clone and run the same command we run.
If you included a `Dockerfile`, build it first and add `PHYRC_IMAGE=my-phyrc` in front of `./run.sh`:

```bash
docker build -t my-phyrc participant
./run.sh python /scripts/evaluate_policy.py \
  --policy /project/participant/policy.py \
  --checkpoint /project/participant/checkpoint.pt \
  --seeds 42 --seconds 900 --output /output/evaluation/submission_check
```

Keep track of your code version, checkpoint, preprocessing and observation keys, library versions, data sources, and reproducible run commands like the ones above.
Also record how you split train/validation episodes and how you selected successful demonstrations. Round deadlines are announced on Codabench.
