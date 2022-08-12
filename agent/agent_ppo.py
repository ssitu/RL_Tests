import random

import numpy
import torch

# Constants
from torch.optim import Optimizer

import rust_utils
from agent import utils
from agent.architectures.actor_critic import ACNet
from agent.agent_torch import AgentTorch, DEFAULT_DEVICE

DISCOUNT_FACTOR = .99
EPSILON = .2
# Times to train on the same experience
EXTRA_TRAININGS_PER_EPISODE = 5
ENTROPY_WEIGHT = .01


class AgentPPO(AgentTorch):

    def __init__(self, model: ACNet, optimizer: Optimizer, device=DEFAULT_DEVICE):
        super().__init__(device=device)
        self.model = model
        self.optimizer = optimizer
        self.critic_loss_fn = torch.nn.HuberLoss()
        self.states = []
        self.actions = []
        self.collected_probs = []
        # self.collected_state_values = []
        self.rewards = []

    def get_action(self, obs: numpy.ndarray, training=True):
        state = torch.tensor(obs, dtype=torch.float, device=self.device).unsqueeze(0)
        # Sample action
        probabilities, state_value = self.model(state)
        action = rust_utils.sample_distribution(probabilities.tolist()[0], random.random())
        if training:
            self.states.append(state)
            self.collected_probs.append(probabilities)
            self.actions.append(action)
        return action

    def give_reward(self, reward: float):
        self.rewards.append(reward)

    def train(self):
        states_batch = torch.cat(self.states).detach()
        old_probabilities = torch.cat(self.collected_probs).detach()
        # old_probabilities, _ = self.model(states_batch)
        old_probabilities = old_probabilities.detach()
        # Isolate the collected_probs of the performed actions under the new policy
        # Convert actions to tensor
        actions_batch = torch.tensor(self.actions, device=self.device).detach()
        old_action_probabilities = utils.action_probs(old_probabilities, actions_batch).detach()
        # Discounted rewards does not change during training
        discounted_rewards = torch.tensor(
            utils.discounted_rewards(self.rewards, DISCOUNT_FACTOR),
            device=self.device
        ).unsqueeze(1).detach()

        for i in range(EXTRA_TRAININGS_PER_EPISODE):
            # L_clip = min(r(th)A, clip(r(th), 1 - ep, 1 + ep)A)

            #
            # Calculate the ratio of the probability of the action under the new policy over the old
            #

            # Probabilities of the current policy
            new_probabilities, state_values = self.model(states_batch)
            # new_action_probabilities = new_probabilities.gather(1, torch.tensor(self.actions).unsqueeze(1))
            new_action_probabilities = utils.action_probs(new_probabilities, actions_batch)
            # Calculate ratios
            ratios = new_action_probabilities / old_action_probabilities

            # Clipped
            clipped_ratios = torch.clip(ratios, 1 - EPSILON, 1 + EPSILON)
            # Advantages
            advantages = discounted_rewards - state_values
            advantages = advantages.detach()  # Prevent the loss_clip from affecting the gradients of the critic

            # Entropy
            entropy = ENTROPY_WEIGHT * utils.entropy(new_probabilities)

            # Loss
            objective_clip = torch.min(ratios * advantages, clipped_ratios * advantages).sum()
            loss_critic = self.critic_loss_fn(state_values, discounted_rewards)
            objective_entropy = entropy.sum()
            loss = -objective_clip + loss_critic - objective_entropy
            # Training
            self.optimizer.zero_grad(set_to_none=True)
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
