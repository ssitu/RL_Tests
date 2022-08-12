from typing import Tuple

import torch
from torch import tensor
from torch.nn import Sequential
from torchinfo import torchinfo

from agent.agent_torch import DEFAULT_DEVICE


class AGACNet(torch.nn.Module):
    def __init__(self, name: str, device=DEFAULT_DEVICE):
        super().__init__()
        self.name = name
        self.device = device

    def forward(self, obs: tensor) -> Tuple[tensor, float, tensor]:
        pass

    def initialize(self, input_shape):
        torchinfo.summary(self, input_shape, device=self.device)


class Separate(AGACNet):
    def __init__(self, actor: Sequential, critic: Sequential, adversary: Sequential, name: str, device=DEFAULT_DEVICE):
        super().__init__(name, device)
        self.actor = actor
        self.critic = critic
        self.adversary = adversary

    def forward(self, obs: tensor) -> Tuple[tensor, float, tensor]:
        return self.actor(obs), self.critic(obs), self.adversary(obs)


class MultiHead(AGACNet):
    def __init__(self, body: Sequential,
                 actor: Sequential, critic: Sequential, adversary: Sequential, name: str, device=DEFAULT_DEVICE):
        super().__init__(name, device)
        self.body = body
        self.actor = actor
        self.critic = critic
        self.adversary = adversary

    def forward(self, obs: tensor) -> Tuple[tensor, float, tensor]:
        x = self.body(obs)
        return self.actor(x), self.critic(x), self.adversary(x)
