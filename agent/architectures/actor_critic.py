from typing import Tuple

from torch import tensor
from torch.nn import Sequential, Module
from torchinfo import torchinfo

from agent.agent_torch import DEFAULT_DEVICE


class ACNet(Module):
    def __init__(self, name: str, device=DEFAULT_DEVICE):
        super().__init__()
        self.name = name
        self.device = device

    def forward(self, obs: tensor) -> Tuple[tensor, float]:
        """
        Make a forward pass through the model
        :return: Outputs a tuple containing
        a probability distribution over the possible actions to take,
        and the critic's state-value estimate
        """
        pass

    def initialize(self, input_shape):
        torchinfo.summary(self, input_shape, device=self.device)


class Separate(ACNet):
    def __init__(self, actor: Sequential, critic: Sequential, name: str, device=DEFAULT_DEVICE):
        super().__init__(name, device=device)
        self.actor = actor
        self.critic = critic

    def forward(self, obs: tensor) -> Tuple[tensor, float]:
        return self.actor(obs), self.critic(obs)


class TwoHeaded(ACNet):
    def __init__(self, body: Sequential, actor: Sequential, critic: Sequential, name: str, device=DEFAULT_DEVICE):
        super().__init__(name, device=device)
        self.body = body
        self.actor = actor
        self.critic = critic

    def forward(self, obs: tensor) -> Tuple[tensor, float]:
        x = self.body(obs)
        state_value = self.critic(x)
        prob_dist = self.actor(x)
        return prob_dist, state_value
