from typing import Tuple

import numpy


class Env:
    """
    A general environment
    """

    def __init__(self):
        pass

    def step(self, action: int) -> Tuple[numpy.ndarray, float, bool]:
        raise NotImplementedError

    def reset(self) -> numpy.ndarray:
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
