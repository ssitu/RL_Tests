import numpy
import torch

# Constants
from torch.optim import Optimizer

from agent import utils
from agent.architectures.actor_critic import ACNet
from agent.agent_torch import AgentTorch

DISCOUNT_FACTOR = .99
EPSILON = .2
# Times to train on the same experience
EXTRA_TRAININGS_PER_EPISODE = 5
ENTROPY_WEIGHT = .01


class AgentPPO(AgentTorch):

    def __init__(self, model: ACNet, optimizer: Optimizer):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.critic_loss_fn = torch.nn.HuberLoss()
        self.states = []
        self.actions = []
        self.collected_probs = []
        # self.collected_state_values = []
        self.rewards = []

    def get_action(self, obs: numpy.ndarray, training=True):
        state = torch.tensor(obs, dtype=torch.float).unsqueeze(0)
        # Sample action
        probabilities, state_value = self.model(state)
        dist = torch.distributions.Categorical(probs=probabilities)
        action = dist.sample().item()
        if training:
            self.states.append(state)
            self.collected_probs.append(probabilities)
            self.actions.append(action)
        return action

    def give_reward(self, reward: float):
        self.rewards.append(reward)

    def discounted_rewards(self):
        discounted_reward = 0.
        discounted_rewards = [0.] * len(self.rewards)
        for time_step, reward in zip(reversed(range(len(self.rewards))), reversed(self.rewards)):
            discounted_reward = DISCOUNT_FACTOR * discounted_reward + reward
            discounted_rewards[time_step] = discounted_reward
        return discounted_rewards

    def train(self):
        states_batch = torch.cat(self.states).detach()
        # old_probabilities = torch.cat(self.collected_probs).detach()
        old_probabilities, _ = self.model(states_batch)
        old_probabilities = old_probabilities.detach()
        # Isolate the collected_probs of the performed actions under the new policy
        old_action_probabilities = old_probabilities.gather(1, torch.tensor(self.actions).unsqueeze(1)).detach()

        # Do one iteration with the already calculated collected_probs during the episode
        # In this case, ratio = 0
        for i in range(EXTRA_TRAININGS_PER_EPISODE):
            # L_clip = min(r(th)A, clip(r(th), 1 - ep, 1 + ep)A)

            #
            # Calculate the ratio of the probability of the action under the new policy over the old
            #

            # Probabilities of the current policy
            new_probabilities, state_values = self.model(states_batch)
            new_action_probabilities = new_probabilities.gather(1, torch.tensor(self.actions).unsqueeze(1))
            # Calculate ratios
            ratios = new_action_probabilities / old_action_probabilities

            # Clipped
            clipped_ratios = torch.clip(ratios, 1 - EPSILON, 1 + EPSILON)
            # Advantages
            discounted_rewards = torch.tensor(utils.discounted_rewards(self.rewards, DISCOUNT_FACTOR)).unsqueeze(1)
            advantages = discounted_rewards - state_values
            advantages = advantages.detach()  # Prevent the loss_clip from affecting the gradients of the critic

            # Entropy
            entropy = ENTROPY_WEIGHT * utils.entropy(new_probabilities)

            # Loss
            objective_clip = torch.min(ratios * advantages, clipped_ratios * advantages).sum()
            loss_critic = (self.critic_loss_fn(state_values, discounted_rewards)).sum()
            objective_entropy = entropy.sum()
            loss = -objective_clip + loss_critic - objective_entropy
            # Training
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

    def save(self):
        pass

    def load(self):
        pass

    def reset(self):
        self.states.clear()
        self.actions.clear()
        self.collected_probs.clear()
        self.rewards.clear()
