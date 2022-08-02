from typing import Tuple

import torch
from torch import tensor
from torch.nn import Sequential
from torchinfo import torchinfo


class AGACNet(torch.nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def forward(self, obs: tensor) -> Tuple[tensor, float, tensor]:
        pass

    def initialize(self, input_shape):
        torchinfo.summary(self, input_shape)


class Separate(AGACNet):
    def __init__(self, actor: Sequential, critic: Sequential, adversary: Sequential, name: str):
        super().__init__(name)
        self.actor = actor
        self.critic = critic
        self.adversary = adversary

    def forward(self, obs: tensor) -> Tuple[tensor, float, tensor]:
        return self.actor(obs), self.critic(obs), self.adversary(obs)


class MultiHead(AGACNet):
    def __init__(self, body: Sequential, actor: Sequential, critic: Sequential, adversary: Sequential, name: str):
        super().__init__(name)
        self.body = body
        self.actor = actor
        self.critic = critic
        self.adversary = adversary

    def forward(self, obs: tensor) -> Tuple[tensor, float, tensor]:
        x = self.body(obs)
        return self.actor(x), self.critic(x), self.adversary(x)
