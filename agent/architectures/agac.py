from typing import Tuple

import torch
from torch import Tensor
from torch.nn import Sequential

from agent.architectures.architecture import Architecture


class AGACNet(Architecture):
    def __init__(self, name: str, device: torch.device):
        super().__init__(name, device=device)

    def forward(self, obs: Tensor) -> Tuple[Tensor, float, Tensor]:
        pass


class Separate(AGACNet):
    def __init__(self, actor: Sequential, critic: Sequential, adversary: Sequential, name: str, device: torch.device):
        super().__init__(name, device)
        self.actor = actor
        self.critic = critic
        self.adversary = adversary

    def forward(self, obs: Tensor) -> Tuple[Tensor, float, Tensor]:
        return self.actor(obs), self.critic(obs), self.adversary(obs)


class MultiHead(AGACNet):
    def __init__(self, body: Sequential,
                 actor: Sequential, critic: Sequential, adversary: Sequential, name: str, device: torch.device):
        super().__init__(name, device)
        self.body = body
        self.actor = actor
        self.critic = critic
        self.adversary = adversary

    def forward(self, obs: Tensor) -> Tuple[Tensor, float, Tensor]:
        x = self.body(obs)
        return self.actor(x), self.critic(x), self.adversary(x)
