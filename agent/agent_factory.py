import torch.nn

import agent.architectures.actor_critic as ac
import agent.architectures.agac as agac
from agent.agent_agac import AgentAGAC
from agent.agent_ppo import AgentPPO
from agent.agent_torch import DEFAULT_DEVICE
from agent.return_estimates.monte_carlo import MonteCarlo
from agent.return_estimates.td_lambda import TDLambda
from envs.env import Env


class AgentFactory:

    def __init__(self, env: Env, device=DEFAULT_DEVICE):
        self._env = env
        self._action_space = self._env.get_action_space()
        self._observation_space = self._env.get_observation_space()
        self.device = device

    def ppo_separate_small_1d(self) -> AgentPPO:
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
        model = ac.Separate(actor, critic, "test", device=self.device)
        model.initialize(self._observation_space)
        optimizer = torch.optim.Adam([
            {"params": actor.parameters(), "lr": .0005},
            {"params": critic.parameters(), 'lr': .001}
        ])
        return_estimation = MonteCarlo(discount=.99)
        return AgentPPO(model, optimizer, return_estimation, device=self.device)

    def ppo_separate_small_1d_td_lambda(self) -> AgentPPO:
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
        model = ac.Separate(actor, critic, "test", device=self.device)
        model.initialize(self._observation_space)
        optimizer = torch.optim.Adam([
            {"params": actor.parameters(), "lr": .0005},
            {"params": critic.parameters(), 'lr': .001}
        ])
        return_estimation = TDLambda(discount=.99, lam=.95)
        return AgentPPO(model, optimizer, return_estimation, device=self.device)

    def ppo_separate_wide_1d(self, name) -> AgentPPO:
        actor = torch.nn.Sequential(
            torch.nn.LazyLinear(100),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(100),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(100),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(100),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(100),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(self._action_space),
            torch.nn.Softmax(dim=-1)
        ).to(self.device)
        critic = torch.nn.Sequential(
            torch.nn.LazyLinear(100),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(100),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(100),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(100),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(100),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(1),
        ).to(self.device)
        model = ac.Separate(actor, critic, name, device=self.device)
        model.initialize(self._observation_space)
        optimizer = torch.optim.Adam([
            {"params": actor.parameters(), "lr": .00005},
            {"params": critic.parameters(), 'lr': .0001}
        ])
        return_estimation = MonteCarlo(discount=.99)
        return AgentPPO(model, optimizer, return_estimation, device=self.device)

    def ppo_separate_critic_heavy_1d(self, name) -> AgentPPO:
        actor = torch.nn.Sequential(
            torch.nn.LazyLinear(50),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(50),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(self._action_space),
            torch.nn.Softmax(dim=-1)
        ).to(self.device)
        critic = torch.nn.Sequential(
            torch.nn.LazyLinear(50),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(50),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(50),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(50),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(1),
        ).to(self.device)
        model = ac.Separate(actor, critic, name, device=self.device)
        model.initialize(self._observation_space)
        optimizer = torch.optim.Adam([
            {"params": actor.parameters(), "lr": .0001},
            {"params": critic.parameters(), 'lr': .0005}
        ])
        return_estimation = MonteCarlo(discount=.99)
        return AgentPPO(model, optimizer, return_estimation, device=self.device)

    def ppo_separate_deep_1d(self) -> AgentPPO:
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
            torch.nn.LazyLinear(1),
        ).to(self.device)
        model = ac.Separate(actor, critic, "test", device=self.device)
        model.initialize(self._observation_space)
        optimizer = torch.optim.Adam([
            {"params": actor.parameters(), "lr": .0005},
            {"params": critic.parameters(), 'lr': .0005}
        ])
        return_estimation = MonteCarlo(discount=.99)
        return AgentPPO(model, optimizer, return_estimation, device=self.device)

    def ppo_twohead_wide_1d(self) -> AgentPPO:
        body = torch.nn.Sequential(
            torch.nn.LazyLinear(50),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(50),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(50),
            torch.nn.LeakyReLU(),
        ).to(self.device)
        actor = torch.nn.Sequential(
            torch.nn.LazyLinear(50),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(50),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(50),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(self._action_space),
            torch.nn.Softmax(dim=-1)
        ).to(self.device)
        critic = torch.nn.Sequential(
            torch.nn.LazyLinear(50),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(50),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(50),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(1)
        ).to(self.device)
        model = ac.TwoHeaded(body, actor, critic, "test", device=self.device)
        model.initialize(self._observation_space)
        optimizer = torch.optim.Adam([
            {"params": body.parameters(), "lr": .00005},
            {"params": actor.parameters(), "lr": .0001},
            {"params": critic.parameters(), 'lr': .0005}
        ])
        return_estimation = MonteCarlo(discount=.99)
        return AgentPPO(model, optimizer, return_estimation, device=self.device)

    def ppo_twohead_small_1d(self) -> AgentPPO:
        body = torch.nn.Sequential(
            torch.nn.LazyLinear(20),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(20),
            torch.nn.LeakyReLU(),
        ).to(self.device)
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
            torch.nn.LazyLinear(1)
        ).to(self.device)
        model = ac.TwoHeaded(body, actor, critic, "test", device=self.device)
        model.initialize(self._observation_space)
        optimizer = torch.optim.Adam([
            {"params": body.parameters(), "lr": .00005},
            {"params": actor.parameters(), "lr": .0001},
            {"params": critic.parameters(), 'lr': .0005}
        ])
        return_estimation = MonteCarlo(discount=.99)
        return AgentPPO(model, optimizer, return_estimation, device=self.device)

    def ppo_separate_small_2d(self) -> AgentPPO:
        actor = torch.nn.Sequential(
            torch.nn.LazyConv2d(5, 5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3),
            torch.nn.LazyConv2d(5, 5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3),
            torch.nn.LazyConv2d(5, 5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3),
            torch.nn.LazyConv2d(5, 3),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.LazyLinear(10),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(5),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(self._action_space),
            torch.nn.Softmax(dim=-1)
        ).to(self.device)
        critic = torch.nn.Sequential(
            torch.nn.LazyConv2d(5, 5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3),
            torch.nn.LazyConv2d(5, 5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3),
            torch.nn.LazyConv2d(5, 5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3),
            torch.nn.LazyConv2d(5, 3),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.LazyLinear(10),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(10),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(1),
        ).to(self.device)
        model = ac.Separate(actor, critic, "test", device=self.device)
        obs_space = self._observation_space
        model.initialize(obs_space)
        optimizer = torch.optim.Adam([
            {"params": actor.parameters(), "lr": .0001},
            {"params": critic.parameters(), 'lr': .001}
        ])
        return_estimation = MonteCarlo(discount=.99)
        return AgentPPO(model, optimizer, return_estimation, device=self.device)

    def ppo_twohead_small_2d(self) -> AgentPPO:
        body = torch.nn.Sequential(
            torch.nn.LazyConv2d(5, 5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3),
            torch.nn.LazyConv2d(5, 5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3),
            torch.nn.LazyConv2d(5, 5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3),
            torch.nn.LazyConv2d(5, 3),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        ).to(self.device)
        actor = torch.nn.Sequential(
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
            torch.nn.LazyLinear(10),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(10),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(1)
        ).to(self.device)
        model = ac.TwoHeaded(body, actor, critic, "test", self.device)
        model.initialize(self._observation_space)
        optimizer = torch.optim.Adam([
            {"params": body.parameters(), "lr": .0001},
            {"params": actor.parameters(), "lr": .0005},
            {"params": critic.parameters(), 'lr': .001}
        ])
        return_estimation = MonteCarlo(discount=.99)
        return AgentPPO(model, optimizer, return_estimation, device=self.device)

    def agac_1d(self) -> AgentAGAC:
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

        adversary = torch.nn.Sequential(
            torch.nn.LazyLinear(10),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(10),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(self._action_space),
            torch.nn.Softmax(dim=-1)
        ).to(self.device)

        model = agac.Separate(actor, critic, adversary, "test", self.device)
        model.initialize(self._observation_space)
        optimizer = torch.optim.Adam([
            {"params": actor.parameters(), "lr": .0005},
            {"params": critic.parameters(), "lr": .0005},
            {"params": adversary.parameters(), "lr": .0005}
        ])
        return_estimation = MonteCarlo(discount=.99)
        return AgentAGAC(model, optimizer, return_estimation, device=self.device)
