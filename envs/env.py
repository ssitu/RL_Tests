from typing import Tuple

import numpy as np


class Env:
    """
    A general environment
    """

    def __init__(self, human_render=False):
        self.human_render = human_render

    @classmethod
    def get_observation_space(cls) -> tuple:
        """
        Get the observation space (shape of the numpy array representing an observation) for this environment
        :return: None
        """
        raise NotImplementedError
    
    @classmethod
    def get_action_space(cls) -> int:
        """
        Get the action space (total number of actions) for this environment
        :return: None
        """
        raise NotImplementedError

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, np.ndarray | None]:
        """
        Perform a step in the environment
        :param action: The action to perform
        :return: A tuple of the following:
            the next state, 
            the reward, 
            a bool indicating whether the episode has terminated, 
            and a numpy array for the action mask for the valid actions in the next state. Is None if the env doesn't support action masks.
        """
        raise NotImplementedError

    def reset(self) -> Tuple[np.ndarray, np.ndarray | None]:
        """
        Reset the environment and return the initial state
        :return: A tuple of the following:
            The initial state,
            A int array for the action mask for the valid actions in the initial state. Is None if the env doesn't support action masks.
        """
        raise NotImplementedError

    def render(self):
        """
        Render this environment
        :return: None
        """
        raise NotImplementedError

    def seed(self, seed):
        """
        Set a seed for the environment
        :return: None
        """
        raise NotImplementedError
