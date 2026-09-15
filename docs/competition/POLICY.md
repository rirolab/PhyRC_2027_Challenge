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

`--seconds` is the episode time limit in simulation time. `--seeds` picks random starting layouts; give at least two different seeds.
Observations default to `actor_rgbd`. For experiments that use only the allowed non-image state, run with `--profile measured_state`.
Multi-seed evaluation requires randomization, so `--no-randomization` and `STRETCH4_RANDOMIZE=0` cannot be used here.

Results are saved per run in `scores.json`, along with measurement traces and per-episode evaluation JSON.
Local practice scores are not the final judging. Running multiple environments in one process, or running the GUI and GPU evaluation at the same time, is not supported.

## Preparing a submission

Keep track of your code version, checkpoint, preprocessing and observation keys, library versions, data sources, and reproducible run commands like the ones above.
Also record how you split train/validation episodes and how you selected successful demonstrations. Submission details and deadlines will be announced by the organizers.
