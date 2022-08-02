import torch.nn
import torch_optimizer as optimizers

import agent.architectures.actor_critic as ac
import agent.architectures.agac as agac
from agent.agent import Agent
from agent.agent_agac import AgentAGAC
from agent.agent_ppo import AgentPPO
from agent.architectures.agac import AGACNet
from envs.env import Env


class AgentFactory:

    def __init__(self, env: Env):
        self.env = env
        self.action_space = self.env.get_action_space()
        self.observation_space = self.env.get_observation_space()

    def normal_1d(self) -> AgentPPO:
        actor = torch.nn.Sequential(
            torch.nn.LazyLinear(10),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(10),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(self.env.get_action_space()),
            torch.nn.Softmax(dim=-1)
        )

        critic = torch.nn.Sequential(
            torch.nn.LazyLinear(10),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(10),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(1),
        )
        model = ac.Separate(actor, critic, "Separate")
        model.initialize(self.observation_space)
        optimizer = torch.optim.Adam([
            {"params": actor.parameters(), "lr": .0005},
            {"params": critic.parameters(), 'lr': .0005}
        ])
        return AgentPPO(model, optimizer)

    def normal_1d_wide(self) -> AgentPPO:
        actor = torch.nn.Sequential(
            torch.nn.LazyLinear(50),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(50),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(50),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(50),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(self.env.get_action_space()),
            torch.nn.Softmax(dim=-1)
        )

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
        )
        model = ac.Separate(actor, critic, "Separate")
        model.initialize(self.observation_space)
        optimizer = torch.optim.Adam([
            {"params": actor.parameters(), "lr": .0005},
            {"params": critic.parameters(), 'lr': .0005}
        ])
        return AgentPPO(model, optimizer)

    def normal_1d_deep(self) -> AgentPPO:
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
            torch.nn.LazyLinear(self.env.get_action_space()),
            torch.nn.Softmax(dim=-1)
        )

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
        )
        model = ac.Separate(actor, critic, "Separate")
        model.initialize(self.observation_space)
        optimizer = torch.optim.Adam([
            {"params": actor.parameters(), "lr": .0005},
            {"params": critic.parameters(), 'lr': .0005}
        ])
        return AgentPPO(model, optimizer)

    def twohead_1d_deep_wide(self) -> AgentPPO:
        body = torch.nn.Sequential(
            torch.nn.LazyLinear(50),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(50),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(50),
            torch.nn.LeakyReLU(),
        )
        actor = torch.nn.Sequential(
            torch.nn.LazyLinear(50),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(50),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(50),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(self.env.get_action_space()),
            torch.nn.Softmax(dim=-1)
        )
        critic = torch.nn.Sequential(
            torch.nn.LazyLinear(50),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(50),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(50),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(1)
        )
        model = ac.TwoHeaded(body, actor, critic, "DeepTwoHead")
        model.initialize(self.observation_space)
        optimizer = optimizers.Yogi([
            {"params": body.parameters(), "lr": .00005},
            {"params": actor.parameters(), "lr": .0001},
            {"params": critic.parameters(), 'lr': .0005}
        ])
        agent = AgentPPO(model, optimizer)
        return agent

    def twohead_1d(self) -> AgentPPO:
        body = torch.nn.Sequential(
            torch.nn.LazyLinear(20),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(20),
            torch.nn.LeakyReLU(),
        )
        actor = torch.nn.Sequential(
            torch.nn.LazyLinear(10),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(10),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(self.env.get_action_space()),
            torch.nn.Softmax(dim=-1)
        )
        critic = torch.nn.Sequential(
            torch.nn.LazyLinear(10),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(10),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(1)
        )
        model = ac.TwoHeaded(body, actor, critic, "DeepTwoHead")
        model.initialize(self.observation_space)
        optimizer = torch.optim.Adam([
            {"params": body.parameters(), "lr": .00005},
            {"params": actor.parameters(), "lr": .0001},
            {"params": critic.parameters(), 'lr': .0005}
        ])
        agent = AgentPPO(model, optimizer)
        return agent

    def normal_2d(self) -> AgentPPO:
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
            torch.nn.LazyLinear(self.env.get_action_space()),
            torch.nn.Softmax(dim=-1)
        )
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
        )
        model = ac.Separate(actor, critic, "normal_2d")
        obs_space = self.observation_space
        model.initialize(obs_space)
        optimizer = torch.optim.Adam([
            {"params": actor.parameters(), "lr": .0001},
            {"params": critic.parameters(), 'lr': .001}
        ])
        return AgentPPO(model, optimizer)

    def twohead_2d(self) -> AgentPPO:
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
        )
        actor = torch.nn.Sequential(
            torch.nn.LazyLinear(10),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(10),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(10),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(10),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(self.env.get_action_space()),
            torch.nn.Softmax(dim=-1)
        )
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
        )
        model = ac.TwoHeaded(body, actor, critic, "DeepTwoHead")
        model.initialize(self.observation_space)
        optimizer = torch.optim.Adam([
            {"params": body.parameters(), "lr": .0001},
            {"params": actor.parameters(), "lr": .0005},
            {"params": critic.parameters(), 'lr': .001}
        ])
        agent = AgentPPO(model, optimizer)
        return agent

    def agac_1d(self) -> AgentAGAC:
        actor = torch.nn.Sequential(
            torch.nn.LazyLinear(10),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(10),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(self.env.get_action_space()),
            torch.nn.Softmax(dim=-1)
        )

        critic = torch.nn.Sequential(
            torch.nn.LazyLinear(10),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(10),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(1),
        )

        adversary = torch.nn.Sequential(
            torch.nn.LazyLinear(10),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(10),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(self.env.get_action_space()),
            torch.nn.Softmax(dim=-1)
        )

        model = agac.Separate(actor, critic, adversary, "AGAC")
        model.initialize(self.observation_space)
        optimizer = torch.optim.Adam([
            {"params": actor.parameters(), "lr": .0005},
            {"params": critic.parameters(), "lr": .0005},
            {"params": adversary.parameters(), "lr": .0005}
        ])
        return AgentAGAC(model, optimizer)
