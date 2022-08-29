from abc import ABC


class EpisodicReturn(ABC):
    """
    The type of estimate to use for the return
    """

    def __init__(self):
        pass

    def calculate_return(self, rewards: list, state_values: list) -> list:
        """
        Calculate the return for an episode, for each time step
        :param rewards: The rewards collected over the episode
        :param state_values: The state values for the episode
        :return: The return for the episode for each time step
        """
        pass
