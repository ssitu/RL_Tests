import os
import random

import numpy as np
import torch
from torch import Tensor

import rust_utils

SMALL_VAL = 1e-8
SAVE_MODEL_LABEL = "model"
SAVE_OPTIMIZER_LABEL = "optimizer"
SAVE_FILE_EXTENSION = ".pt"
AGENTS_FOLDER = "agent/"
SAVED_MODELS_FOLDER = "saved_models/"


def action_probs(probs: Tensor, actions: Tensor):
    """
    Creates a Tensor made up of the probabilities of the taken actions
    :param probs: Tensor of shape (batch_size, action_space), should be akin to a list of probabilities distributions
    :param actions: Tensor of shape (batch_size,), should be akin to a list of the actions taken over the episode
    :return: Tensor of shape (batch_size,), the probabilities of the taken actions
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


def td_lambda_return(rewards: list, state_values: list, discount: float, lam: float):
    """
    Calculates the TD-lambda return for an episode
    :param rewards: list of rewards for an episode
    :param state_values: list of state values for an episode
    :param discount: discount factor
    :param lam: lambda factor, 0 <= lambda <= 1, where 0 is the same as one step TD, and 1 is the Monte Carlo method
    :return: list of returns for an episode
    """
    return rust_utils.td_lambda_return(rewards, state_values, discount, lam)


def entropy(probs: Tensor):
    return -(probs * torch.log(probs + SMALL_VAL)).sum(dim=1)


def kl_div(probs1: Tensor, probs2: Tensor):
    if len(probs1.size()) > 1:
        return (probs1 * torch.log(probs1 / (probs2 + SMALL_VAL) + SMALL_VAL)).sum(dim=1).unsqueeze(1)
    else:
        return (probs1 * torch.log(probs1 / (probs2 + SMALL_VAL) + SMALL_VAL)).sum()


def get_current_directory():
    return os.path.dirname(os.path.abspath(__file__)) + "/"


def generate_filepath(filename: str):
    current_dir = get_current_directory()
    filepath = f"{current_dir}{AGENTS_FOLDER}{SAVED_MODELS_FOLDER}{filename}{SAVE_FILE_EXTENSION}"
    return filepath


def save(model: torch.nn.Module, optimizer: torch.optim.Optimizer, name: str):
    """
    Save a model and its optimizer to a file
    :param model: The model to save
    :param optimizer: The optimizer to save
    :param name: The name for the file
    :return: None
    """
    to_save = {
        SAVE_MODEL_LABEL: model.state_dict(),
        SAVE_OPTIMIZER_LABEL: optimizer.state_dict(),
    }
    filepath = generate_filepath(name)
    if not os.path.exists(filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(to_save, filepath)


def load(model: torch.nn.Module, optimizer: torch.optim.Optimizer, name: str, device):
    """
    Load a model and its optimizer from a file
    :param model: The model to load into
    :param optimizer: The optimizer to load into
    :param name: The name of the file
    :return: None
    """
    filepath = generate_filepath(name)
    try:
        loaded = torch.load(filepath, map_location=device)
        model.load_state_dict(loaded[SAVE_MODEL_LABEL])
        optimizer.load_state_dict(loaded[SAVE_OPTIMIZER_LABEL])
    except FileNotFoundError as e:
        print(f"Could not load model, file not found: \n{filepath}")


def seed_global_rng(seed=0):
    # Seeding the random number generators
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def use_cuda(on: bool) -> torch.device:
    # Assuming that one cuda device is to be used
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if cuda_available and on:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        cuda_id = torch.cuda.current_device()
        print(f"CUDA current device id: {cuda_id}")
        print(f"CUDA device name: {torch.cuda.get_device_name(cuda_id)}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    else:
        print(f"Cores Available: {torch.get_num_threads()}")
        return torch.device("cpu")


if __name__ == "__main__":
    # Testing the discounted_rewards function
    rewards = [random.randint(-100, 100) for x in range(1, 10000)]
    discount = 0.99
    # Calculate the discounted rewards
    discounted_reward = 0.
    ans = [0.] * len(rewards)
    for time_step, reward in zip(reversed(range(len(rewards))), reversed(rewards)):
        discounted_reward = discount * discounted_reward + reward
        ans[time_step] = discounted_reward
    # Test the discounted_rewards function
    result = discounted_rewards(rewards, discount)
    assert np.allclose(ans, result, atol=1e-3)
