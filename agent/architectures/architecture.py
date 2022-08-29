import torch
from torch.nn import Module
from torchinfo import torchinfo


class Architecture(Module):
    def __init__(self, name: str, device: torch.device):
        super().__init__()
        self.name = name
        self.device = device

    def initialize(self, input_shape: tuple):
        torchinfo.summary(self, input_shape, device=self.device)
