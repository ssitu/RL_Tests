from typing import Tuple

import numpy


class Env:
    """
    A general environment
    """

    def __init__(self, human_render=False):
        self.human_render = human_render

    def step(self, action: int) -> Tuple[numpy.ndarray, float, bool]:
        """
        Perform a step in the environment
        :param action: The action to perform
        :return: The next state, the reward, and whether the episode has terminated
        """
        raise NotImplementedError

    def reset(self) -> numpy.ndarray:
        """
        Reset the environment and return the initial state
        :return: The initial state
        """
        raise NotImplementedError

    def get_observation_space(self) -> tuple:
        """
        Obtain the observation space for this environment
        :return: None
        """
        raise NotImplementedError

    def get_action_space(self) -> int:
        """
        Obtain the action space for this environment
        :return: None
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
