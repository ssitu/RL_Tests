import numpy


class Agent:
    def __init__(self, name: str):
        self.name = name

    def reset(self):
        pass

    def get_action(self, obs: numpy.ndarray, training=True):
        pass

    def give_reward(self, reward: float):
        pass

    def train(self):
        pass

    def save(self, filename: str = None):
        """
        :param filename: Used as the filename if given
        """
        raise NotImplementedError

    def load(self, filename: str = None):
        """
        :param filename: Used as the filename if given
        """
        raise NotImplementedError
