import torch
from torch.optim import Optimizer

import utils
from agent.agent import Agent
from agent.architectures.architecture import Architecture


class AgentTorch(Agent):

    def __init__(self, model: Architecture, optimizer: Optimizer, device: torch.device):
        super().__init__(model.name)
        self.model = model
        self.optimizer = optimizer
        self.device = device

    def save(self, filename: str = None):
        utils.save(self.model, self.optimizer, filename if filename else self.model.name)

    def load(self, filename: str = None):
        utils.load(self.model, self.optimizer, filename if filename else self.model.name, self.device)
