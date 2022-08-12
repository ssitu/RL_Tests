import random

import numpy.random
import torch


def seed_rng(seed=0):
    # Seeding the random number generators
    random.seed(seed)
    torch.manual_seed(seed)
    numpy.random.seed(seed)


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
        return torch.device("cuda")
    else:
        print(f"Cores Available: {torch.get_num_threads()}")
        return torch.device("cpu")


if __name__ == '__main__':
    from agent.agent_factory import AgentFactory
    from envs.cartpole import CartPole
    from controller import Controller
    from envs.bandit import Bandit

    device = use_cuda(False)

    # Environment
    env = CartPole()

    seed = 34358734
    agent_factory = AgentFactory(env, device=device)

    seed_rng(seed)
    agent = agent_factory.normal_1d()
    control = Controller(env, agent)
    control.seed(seed)
    control.plot.start()
    control.play(500, training=True, render=type(env) == Bandit)
    control.plot.save("1")
    control.plot.stop()

    seed_rng(seed)
    agent2 = agent_factory.agac_1d()
    control2 = Controller(env, agent2)
    control2.seed(seed)
    control2.plot.start()
    control2.play(500, training=True, render=type(env) == Bandit)
    control2.plot.save("2")
    control2.plot.stop()
