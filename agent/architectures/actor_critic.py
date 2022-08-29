from typing import Tuple

import torch
from torch import Tensor
from torch.nn import Sequential

from agent.architectures.architecture import Architecture


class ACNet(Architecture):
    def __init__(self, name: str, device: torch.device):
        super().__init__(name, device=device)

    def forward(self, obs: Tensor) -> Tuple[Tensor, float]:
        """
        Make a forward pass through the model
        :return: Outputs a tuple containing
        a probability distribution over the possible actions to take,
        and the critic's state-value estimate
        """
        pass


class Separate(ACNet):
    def __init__(self, actor: Sequential, critic: Sequential, name: str, device: torch.device):
        super().__init__(name, device=device)
        self.actor = actor
        self.critic = critic

    def forward(self, obs: Tensor) -> Tuple[Tensor, float]:
        return self.actor(obs), self.critic(obs)


class TwoHeaded(ACNet):
    def __init__(self, body: Sequential, actor: Sequential, critic: Sequential, name: str, device: torch.device):
        super().__init__(name, device=device)
        self.body = body
        self.actor = actor
        self.critic = critic

    def forward(self, obs: Tensor) -> Tuple[Tensor, float]:
        x = self.body(obs)
        state_value = self.critic(x)
        prob_dist = self.actor(x)
        return prob_dist, state_value
