import random

import numpy as np
import torch
from torch import tensor

import rust_utils

SMALL_VAL = 1e-8


def action_probs(probs: tensor, actions: tensor):
    """
    Creates a tensor made up of the probabilities of the taken actions
    :param probs: tensor of shape (batch_size, action_space), should be akin to a list of probabilities distributions
    :param actions: tensor of shape (batch_size,), should be akin to a list of the actions taken over the episode
    :return: tensor of shape (batch_size,), the probabilities of the taken actions
    """
    # probs: batch_size x n_actions
    # actions: batch_size x 1
    # return: batch_size x 1
    return probs.gather(1, actions.unsqueeze(1))


def discounted_rewards(rewards: list, discount: float):
    """
    Calculates the discounted rewards for an episode
    """
    return rust_utils.discounted_rewards(rewards, discount)


def entropy(probs: tensor):
    return -(probs * torch.log(probs + SMALL_VAL)).sum(dim=1)


def kl_div(probs1: tensor, probs2: tensor):
    if len(probs1.size()) > 1:
        return (probs1 * torch.log(probs1 / (probs2 + SMALL_VAL) + SMALL_VAL)).sum(dim=1).unsqueeze(1)
    else:
        return (probs1 * torch.log(probs1 / (probs2 + SMALL_VAL) + SMALL_VAL)).sum()


def n_step(rewards: tensor, values: tensor, discount, t, n):
    pass


if __name__ == "__main__":
    # Testing the discounted_rewards function
    rewards = [random.randint(-100, 100) for x in range(1, 1000)]
    discount = 0.9
    # Calculate the discounted rewards
    discounted_reward = 0.
    ans = [0.] * len(rewards)
    for time_step, reward in zip(reversed(range(len(rewards))), reversed(rewards)):
        discounted_reward = discount * discounted_reward + reward
        ans[time_step] = discounted_reward
    # Test the discounted_rewards function
    result = discounted_rewards(rewards, discount)
    assert np.allclose(ans, result, atol=1e-3)
