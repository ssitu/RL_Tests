from typing import Tuple

import gym
import numpy as np

from envs.env import Env


def preprocess(obs: np.array) -> np.array:
    return np.expand_dims(np.sum(obs.transpose([2, 0, 1]), axis=0, dtype=float), axis=0)


class Pong(Env):
    """
    Wrapper for the CartPole gym environment
    """

    def __init__(self):
        super().__init__()
        self.env = gym.make('Pong-v0')

    def transition(self, action: int) -> Tuple[np.array, float, bool]:
        next_state, reward, done, _ = self.env.step(action)
        return preprocess(next_state), reward, done

    def reset(self) -> np.array:
        return preprocess(self.env.reset())

    def get_observation_space(self) -> tuple:
        height, width, channels = self.env.observation_space.shape
        return 1, 1, height, width

    def get_action_space(self) -> int:
        return self.env.action_space.n

    def render(self):
        self.env.render()

    def __del__(self):
        self.env.close()
