import numpy
import torch

from agent.agent import Agent

cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")
if cuda_available:
    print(f"CUDA version: {torch.version.cuda}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    cuda_id = torch.cuda.current_device()
    print(f"CUDA current device id: {cuda_id}")
    print(f"CUDA device name: {torch.cuda.get_device_name(cuda_id)}")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    print(f"Cores Available: {torch.get_num_threads()}")


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AgentTorch(Agent):

    def __init__(self):
        super().__init__()

    def reset(self):
        super().reset()

    def get_action(self, obs: numpy.ndarray):
        super().get_action(obs)

    def give_reward(self, reward: float):
        super().give_reward(reward)

    def train(self):
        super().train()

