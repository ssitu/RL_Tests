from typing import Tuple

import gym
import numpy as np

from envs.env import Env


class CartPole(Env):
    """
    Wrapper for the CartPole gym environment
    """

    def __init__(self, human_render=False):
        super().__init__()
        self.env = gym.make('CartPole-v1', new_step_api=True, render_mode="human" if human_render else None)

    def step(self, action: int) -> Tuple[np.array, float, bool, np.ndarray | None]:
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        return next_state, reward, truncated or terminated, None

    def reset(self) -> Tuple[np.ndarray, np.ndarray | None]:
        return self.env.reset(), None

    def get_observation_space(self) -> tuple:
        return self.env.observation_space.shape

    def get_action_space(self) -> int:
        return self.env.action_space.n

    def seed(self, seed):
        self.env.reset(seed=seed)

    def __del__(self):
        self.env.close()
