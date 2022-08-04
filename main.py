import random

import numpy.random
import torch
import controller


def seed_rng(seed=0):
    # Seeding the random number generators
    random.seed(seed)
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    controller.seed(seed)


if __name__ == '__main__':
    from agent.agent_factory import AgentFactory
    from envs.cartpole import CartPole
    from controller import Controller
    from envs.pong import Pong
    from envs.bandit import Bandit

    # Environment
    env = CartPole()

    seed = 756283
    seed_rng(seed)

    agent_factory = AgentFactory(env)
    agent = agent_factory.normal_1d()
    control = Controller(env, agent)
    control.plot.start()
    control.play(1000, training=True, render=type(env) == Bandit)
    # controller.play(5, training=False, render=True)
    control.plot.save("1")
    control.plot.stop()

    seed_rng(seed)
    agent2 = agent_factory.agac_1d()
    control2 = Controller(env, agent2)
    control2.plot.start()
    control2.play(1000, training=True, render=type(env) == Bandit)
    control2.plot.save("2")
    control2.plot.stop()
