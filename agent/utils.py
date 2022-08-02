import torch
from torch import tensor

SMALL_VAL = 1e-8


def discounted_rewards(rewards, discount: float):
    discounted_reward = 0.
    result = [0.] * len(rewards)
    for time_step, reward in zip(reversed(range(len(rewards))), reversed(rewards)):
        discounted_reward = discount * discounted_reward + reward
        result[time_step] = discounted_reward
    return result


def entropy(probs: tensor):
    return -(probs * torch.log(probs + SMALL_VAL)).sum(dim=1)


def kl_div(probs1: tensor, probs2: tensor):
    if len(probs1.size()) > 1:
        return (probs1 * torch.log(probs1 / (probs2 + SMALL_VAL) + SMALL_VAL)).sum(dim=1).unsqueeze(1)
    else:
        return (probs1 * torch.log(probs1 / (probs2 + SMALL_VAL) + SMALL_VAL)).sum()


def n_step(rewards: tensor, values: tensor, discount, t, n):
    pass
