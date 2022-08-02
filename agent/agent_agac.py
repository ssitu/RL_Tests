import numpy
import torch

# Constants
from torch.optim import Optimizer

from agent import utils
from agent.architectures.agac import AGACNet
from agent.agent_torch import AgentTorch

DISCOUNT_FACTOR = .99
EPSILON = .2
# Times to train on the same experience
EXTRA_TRAININGS_PER_EPISODE = 5
ENTROPY_WEIGHT = .01
SMALL_VAL = 1e-7
# AGAC
AGAC_WEIGHT = 1
c = .001


class AgentAGAC(AgentTorch):

    def __init__(self, model: AGACNet, optimizer: Optimizer):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.critic_loss_fn = torch.nn.HuberLoss()
        self.states = []
        self.actions = []
        self.collected_probabilities = []
        self.rewards = []
        self.follow_adversary = False

    def get_action(self, obs: numpy.ndarray, training=True):
        state = torch.tensor(obs, dtype=torch.float).unsqueeze(0)
        # Sample action
        probabilities, _, adversary_probs = self.model(state)
        if not self.follow_adversary:
            dist = torch.distributions.Categorical(probs=probabilities)
        else:
            dist = torch.distributions.Categorical(probs=adversary_probs)
        action = dist.sample().item()
        if training:
            self.states.append(state)
            self.collected_probabilities.append(probabilities)
            self.actions.append(action)
        return action

    def give_reward(self, reward: float):
        self.rewards.append(reward)

    def train(self):
        states_batch = torch.cat(self.states).detach()
        old_probabilities = torch.cat(self.collected_probabilities).detach()
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
            new_probabilities, state_values, adv_probs = self.model(states_batch)
            new_action_probs = new_probabilities.gather(1, torch.tensor(self.actions).unsqueeze(1))
            # Calculate ratios
            ratios = new_action_probs / old_action_probabilities

            # Clipped
            clipped_ratios = torch.clip(ratios, 1 - EPSILON, 1 + EPSILON)
            # Advantages
            discounted_rewards = torch.tensor(utils.discounted_rewards(self.rewards, DISCOUNT_FACTOR)).unsqueeze(1)
            state_values += c * utils.kl_div(new_probabilities.detach(), adv_probs.detach())  # AGAC
            advantages = discounted_rewards - state_values
            advantages = advantages.detach()  # Prevent the loss_clip from affecting the gradients of the critic
            adv_action_probs = adv_probs.gather(1, torch.tensor(self.actions).unsqueeze(1))
            advantages += c * (torch.log(new_action_probs) - torch.log(adv_action_probs.detach()))  # AGAC

            # Entropy
            entropy = ENTROPY_WEIGHT * utils.entropy(new_probabilities)

            # KL divergence of adversary and actor
            kl_div_adversary = utils.kl_div(new_probabilities.detach(), adv_probs)  # AGAC

            # Loss
            objective_clip = torch.min(ratios * advantages, clipped_ratios * advantages).sum()
            loss_critic = (self.critic_loss_fn(state_values, discounted_rewards)).sum()
            objective_entropy = entropy.sum()
            loss_adv = AGAC_WEIGHT * kl_div_adversary.sum()
            loss = -objective_clip + loss_critic - objective_entropy + loss_adv
            # Training
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

    def use_adversary(self):
        self.follow_adversary = True

    def save(self):
        pass

    def load(self):
        pass

    def reset(self):
        self.states.clear()
        self.actions.clear()
        self.collected_probabilities.clear()
        self.rewards.clear()
