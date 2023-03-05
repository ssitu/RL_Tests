import random
import numpy as np
import torch
from typing import Optional

# Constants
from torch.optim import Optimizer

import rust_utils
import utils
from agent.architectures.actor_critic import ACNet
from agent.agent_torch import AgentTorch
from agent.return_estimates.return_estimate import EpisodicReturn

DISCOUNT_FACTOR = .99
EPSILON = .2
# Times to train on the same experience
EXTRA_TRAININGS_PER_EPISODE = 5
ENTROPY_WEIGHT = .01


class AgentPPO(AgentTorch):

    def __init__(self, model: ACNet, optimizer: Optimizer, return_estimation: EpisodicReturn, device: torch.device):
        super().__init__(model, optimizer, device=device)
        self.critic_loss_fn = torch.nn.HuberLoss()
        self.states = []
        self.actions = []
        self.collected_probs = []
        self.collected_state_values = []
        self.rewards = []
        self.greedy = False
        self.return_estimation = return_estimation

    def get_action(self, obs: np.ndarray, action_mask: Optional[np.ndarray | None], training=True):
        state = torch.as_tensor(obs, dtype=torch.float, device=self.device).unsqueeze(0)
        # Sample action
        probabilities, state_value = self.model(state)
        # Multiply the probabilities with the action mask and then renormalize
        probabilities = probabilities * torch.as_tensor(action_mask, dtype=torch.float, device=self.device)
        probabilities = probabilities / torch.sum(probabilities)
        if self.greedy:
            max = 0
            action = 0
            for i, x in enumerate(probabilities.tolist()[0]):
                if max < x:
                    max = x
                    action = i
        else:
            action = rust_utils.sample_distribution(probabilities.tolist()[0], random.random())
        if training:
            self.states.append(state)
            self.collected_probs.append(probabilities)
            self.collected_state_values.append(state_value)
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
        actions_batch = torch.as_tensor(self.actions, device=self.device).detach()
        old_action_probabilities = utils.action_probs(old_probabilities, actions_batch).detach()
        # Calculate the return estimate for each time step
        # The return does not change during training
        returns = self.return_estimation.calculate_return(self.rewards, self.collected_state_values)
        returns = torch.as_tensor(returns, device=self.device).unsqueeze(1).detach()

        for i in range(EXTRA_TRAININGS_PER_EPISODE):
            # L_clip = min(r(th)A, clip(r(th), 1 - ep, 1 + ep)A)

            #
            # Calculate the ratio of the probability of the action under the new policy over the old
            #

            # Probabilities of the current policy
            new_probabilities, state_values = self.model(states_batch)
            new_action_probabilities = utils.action_probs(new_probabilities, actions_batch)
            # Calculate ratios
            ratios = new_action_probabilities / old_action_probabilities

            # Clipped
            clipped_ratios = torch.clip(ratios, 1 - EPSILON, 1 + EPSILON)
            # Advantages
            # advantages = discounted_rewards - state_values
            advantages = returns - state_values
            advantages = advantages.detach()  # Prevent the loss_clip from affecting the gradients of the critic

            # Entropy
            entropy = ENTROPY_WEIGHT * utils.entropy(new_probabilities)

            # Loss
            objective_clip = torch.min(ratios * advantages, clipped_ratios * advantages).sum()
            loss_critic = self.critic_loss_fn(state_values, returns)
            objective_entropy = entropy.sum()
            loss = -objective_clip + loss_critic - objective_entropy
            # Training
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward(retain_graph=True)
            self.optimizer.step()

    def reset(self):
        self.states.clear()
        self.actions.clear()
        self.collected_probs.clear()
        self.collected_state_values.clear()
        self.rewards.clear()

    def set_greedy(self, greedy: bool):
        # Instead of sampling from the distribution given by the policy network, action with the highest value is chosen
        self.greedy = greedy
