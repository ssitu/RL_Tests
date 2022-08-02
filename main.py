import random

import numpy.random
import torch


def seed_rng():
    # Seeding the random number generators
    seed = 13593050
    env.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    numpy.random.seed(seed)


if __name__ == '__main__':
    from agent.agent_factory import AgentFactory
    from envs.cartpole import CartPole
    from controller import Controller
    from envs.pong import Pong
    from envs.bandit import Bandit

    # Environments
    cartpole = CartPole()
    pong = Pong()
    bandit = Bandit(15)
    env = cartpole

    seed_rng()

    agent_factory = AgentFactory(env)
    agent = agent_factory.normal_1d()
    controller = Controller(env, agent)
    controller.start_plot()
    controller.play(2000, training=True, render=env is bandit)
    # controller.play(5, training=False, render=True)
    controller.save_plot("1")
    controller.stop_plot()

    seed_rng()
    agent2 = agent_factory.agac_1d()
    controller2 = Controller(env, agent2)
    controller2.start_plot()
    controller2.play(2000, training=True, render=env is bandit)
    controller2.save_plot("2")
    controller2.stop_plot()
