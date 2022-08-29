import utils
from agent.return_estimates.return_estimate import EpisodicReturn


class TDLambda(EpisodicReturn):

    def __init__(self, discount: float, lam: float):
        super().__init__()
        self.discount = discount
        self.lam = lam

    def calculate_return(self, rewards: list, state_values: list) -> list:
        """
        Calculate the return for an episode, for each time step
        :param rewards: The rewards collected over the episode
        :param state_values: The state values for the episode. It is not used in Monte Carlo estimate
        :return: The return for the episode for each time step
        """
        return utils.td_lambda_return(rewards, state_values, discount=self.discount, lam=self.lam)
