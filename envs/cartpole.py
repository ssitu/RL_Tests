from typing import Tuple

import gym
import numpy

from envs.env import Env


class CartPole(Env):
    """
    Wrapper for the CartPole gym environment
    """
    def __init__(self):
        super().__init__()
        self.env = gym.make('CartPole-v1')

    def step(self, action: int) -> Tuple[numpy.array, float, bool]:
        next_state, reward, done, _ = self.env.step(action)
        return next_state, reward, done

    def reset(self) -> numpy.ndarray:
        return self.env.reset()

    def get_observation_space(self) -> tuple:
        return self.env.observation_space.shape

    def get_action_space(self) -> int:
        return self.env.action_space.n

    def render(self):
        self.env.render()

    def seed(self, seed):
        self.env.seed(seed)

    def __del__(self):
        self.env.close()

