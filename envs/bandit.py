import random
from typing import Tuple

import numpy

from envs.env import Env
import matplotlib.pyplot as plt


class Bandit(Env):
    def __init__(self, arms: int, human_render=False):
        super().__init__(human_render)
        self.fig = None
        self.arms = arms
        self.probabilities = [random.random() for i in range(arms)]
        self.probabilities[-1] = 1
        self.action_count = [0] * arms
        self.highest_action = 0
        self.highest_action_count = 0
        self.action_count_history = [[] for i in range(arms)]
        self.time_steps = 0
        self.last_action_taken = 0
        self.random_generator = random.Random()

    def get_obs(self) -> numpy.ndarray:
        return numpy.array([0])

    def step(self, action: int) -> Tuple[numpy.ndarray, float, bool]:
        # self.action_count[action] += 1
        # self.time_steps += 1
        # for i in range(self.arms):
        #     self.action_count_history[i].append(self.action_count[i]/self.time_steps)
        #
        # if self.highest_action < self.action_count[action]:
        #     self.highest_action = action
        #     self.highest_action_count = self.action_count[action]
        if self.human_render:
            self.render()
        self.last_action_taken = action
        reward = 1 if self.random_generator.random() < self.probabilities[action] else 0
        return self.get_obs(), reward, True

    def reset(self) -> numpy.ndarray:
        return self.get_obs()

    def get_observation_space(self) -> tuple:
        return 1,

    def get_action_space(self) -> int:
        return self.arms

    def render(self):
        # print(f"Most picked action: {self.highest_action}, with probability {self.collected_probs[self.highest_action]}")
        if self.fig is None:
            # Initialize figure
            self.fig = plt.figure()
            self.fig.suptitle("Percentage of actions taken over time")
        plt.cla()

        # Calculating the percentage that the action has been taken for each action over the start of the environment
        self.action_count[self.last_action_taken] += 1
        self.time_steps += 1
        for i in range(self.arms):
            self.action_count_history[i].append(self.action_count[i] / self.time_steps)
        # Track the highest action and count
        if self.highest_action < self.action_count[self.last_action_taken]:
            self.highest_action = self.last_action_taken
            self.highest_action_count = self.action_count[self.last_action_taken]
        # Plotting a line for each action
        x = self.time_steps
        for i in range(self.arms):
            plt.plot(self.action_count_history[i])
            y = self.action_count_history[i][-1]
            plt.text(x, y, f"{i}, {self.probabilities[i]}")
        plt.pause(.001)

    def seed(self, seed):
        self.random_generator.seed(seed)