import numpy as np
from typing import Optional


class Agent:
    def __init__(self, name: str):
        self.name = name

    def reset(self):
        """
        Resets the agent.
        :return: None
        """
        pass

    def get_action(self, obs: np.ndarray, action_mask: Optional[np.ndarray | None], training=True) -> int: 
        """
        :param obs: The observation from the environment
        :param action_mask: The action mask from the environment. 
            This is a int array of 0s and 1s, where a 1 in the ith means the ith action is valid, and i being in the range [0, action_space)
        :param training: Whether the agent is training or not
        :return: The action to take, actions are non-negative integers, without gaps (i.e. in order)
        """
        pass

    def give_reward(self, reward: float):
        """
        The Agent is given the reward from the environment.
        :param reward: The reward from the environment
        :return: None
        """

    def train(self):
        """
        Trains the agent.
        :return: None
        """
        pass

    def save(self, filename: str = None):
        """
        :param filename: Used as the filename if given
        """
        pass

    def load(self, filename: str = None):
        """
        :param filename: Used as the filename if given
        """
        pass
