def discounted_rewards(rewards, discount: float) -> list:
    """
    Calculates the discounted rewards for an episode
    :param rewards: The rewards for an episode
    :param discount: The discount factor
    :return: The discounted rewards for an episode
    """
    pass


def td_lambda_return(rewards, state_values, discount: float, lam: float) -> list:
    """
    Calculates the return for an episode, for each time step
    :param rewards: The rewards collected over the episode
    :param state_values: The state values for the episode. It is not used in Monte Carlo estimate
    :param discount: The discount factor
    :param lam: The lambda factor
    :return: The return for the episode for each time step
    """
    pass


def sample_distribution(distribution: list, rand_f: float) -> int:
    """
    Sample a random index from the given distribution
    :param distribution: The discrete distribution to sample from, should be a list of floats summing to 1
    :param rand_f: The random number to sample from the distribution (between 0 and 1)
    :return: The random index according to the distribution
    """
    pass
