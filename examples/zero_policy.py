"""Participant adapter example: neutral actions for interface checks only."""
import numpy as np


def load_policy(checkpoint, observation_space, action_space):
    class NeutralPolicy:
        def reset(self):
            pass

        def __call__(self, observation):
            return np.zeros(action_space.shape, dtype=np.float32)

    return NeutralPolicy()
