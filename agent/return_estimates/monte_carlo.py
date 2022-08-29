import utils
from agent.return_estimates.return_estimate import EpisodicReturn


class MonteCarlo(EpisodicReturn):
    """
    The state value estimate using the Monte-Carlo method
    """

    def __init__(self, discount: float):
        super().__init__()
        self.discount = discount

    def calculate_return(self, rewards: list, state_values: list) -> list:
        """
        Calculate the return for an episode, for each time step
        :param rewards: The rewards collected over the episode
        :param state_values: The state values for the episode. It is not used in Monte Carlo estimate
        :return: The return for the episode for each time step
        """
        return utils.discounted_rewards(rewards, discount=self.discount)
