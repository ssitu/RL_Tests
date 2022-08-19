import numpy
import torch

from agent.agent import Agent

DEFAULT_DEVICE = torch.device("cpu")


class AgentTorch(Agent):

    def __init__(self, name: str,  device=DEFAULT_DEVICE):
        super().__init__(name)
        self.device = device

    def reset(self):
        super().reset()

    def get_action(self, obs: numpy.ndarray):
        super().get_action(obs)

    def give_reward(self, reward: float):
        super().give_reward(reward)

    def train(self):
        super().train()
