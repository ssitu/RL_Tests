import copy
from utils import use_cuda, seed_global_rng

def agac(self):
    import torch
    import agent.architectures.agac as agac
    from agent.agent_agac import AgentAGAC
    from agent.return_estimates.monte_carlo import MonteCarlo

    actor = torch.nn.Sequential(
        torch.nn.LazyLinear(10),
        torch.nn.LeakyReLU(),
        torch.nn.LazyLinear(10),
        torch.nn.LeakyReLU(),
        torch.nn.LazyLinear(10),
        torch.nn.LeakyReLU(),
        torch.nn.LazyLinear(10),
        torch.nn.LeakyReLU(),
        torch.nn.LazyLinear(10),
        torch.nn.LeakyReLU(),
        torch.nn.LazyLinear(self._action_space),
        torch.nn.Softmax(dim=-1)
    ).to(self.device)

    critic = torch.nn.Sequential(
        torch.nn.LazyLinear(10),
        torch.nn.LeakyReLU(),
        torch.nn.LazyLinear(10),
        torch.nn.LeakyReLU(),
        torch.nn.LazyLinear(1),
    ).to(self.device)

    clone = torch.nn.Sequential(*[copy.deepcopy(layer) for layer in actor])
    clone.load_state_dict(actor.state_dict())
    adversary = clone.to(self.device)

    model = agac.Separate(actor, critic, adversary, "cartpole", self.device)
    model.initialize(self._observation_space)
    optimizer = torch.optim.Adam([
        {"params": actor.parameters(), "lr": .0005},
        {"params": critic.parameters(), "lr": .0005},
        {"params": adversary.parameters(), "lr": .0001}
    ])
    return_estimation = MonteCarlo(discount=.99)
    return AgentAGAC(model, optimizer, return_estimation, device=self.device)


def ppo(self):
    import torch
    import agent.architectures.actor_critic as ac
    from agent.agent_ppo import AgentPPO
    from agent.return_estimates.monte_carlo import MonteCarlo

    actor = torch.nn.Sequential(
        torch.nn.LazyLinear(10),
        torch.nn.LeakyReLU(),
        torch.nn.LazyLinear(10),
        torch.nn.LeakyReLU(),
        torch.nn.LazyLinear(self._action_space),
        torch.nn.Softmax(dim=-1)
    ).to(self.device)
    critic = torch.nn.Sequential(
        torch.nn.LazyLinear(10),
        torch.nn.LeakyReLU(),
        torch.nn.LazyLinear(10),
        torch.nn.LeakyReLU(),
        torch.nn.LazyLinear(1),
    ).to(self.device)
    model = ac.Separate(actor, critic, "cartpole", device=self.device)
    model.initialize(self._observation_space)
    optimizer = torch.optim.Adam([
        {"params": actor.parameters(), "lr": .00005},
        {"params": critic.parameters(), 'lr': .0001}
    ])
    return_estimation = MonteCarlo(discount=.99)
    return AgentPPO(model, optimizer, return_estimation, device=self.device)

if __name__ == '__main__':
    from agent.agent_factory import AgentFactory
    from controller import Controller
    from envs.cartpole import CartPole

    # Environment
    env = CartPole(human_render=False)

    seed = 7777
    agent_factory = AgentFactory(env, device=use_cuda(False))

    iterations = 2000

    seed_global_rng(seed)
    agent = agac(agent_factory)
    control = Controller(env, [agent])
    control.seed(seed)
    control.plot.moving_avg_len = 100
    control.plot.start()
    control.play(iterations, training=True)
    # control.plot.save("cartpole")
    control.plot.stop()
