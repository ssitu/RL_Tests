from typing import Union, List

from agent.agent import Agent
from envs.env import Env
from plot import Plot
import numpy as np

class ActionMaskWarning(UserWarning):
    warning_displayed = False

    @classmethod
    def warn(cls):
        if not cls.warning_displayed:
            import warnings
            warnings.warn("Action mask returned by environment is all zeros")
            cls.warning_displayed = True

class Controller:
    """
    Handles interactions between the environment and agents
    """

    def __init__(self, env: Env, agents: List[Agent], verbose: bool = False, saving_interval: Union[None, int] = None,
                 load_at_reset: bool = False,
                 save_plot_interval: Union[None, int] = None):
        self.env = env
        self.agents = agents
        self.plot = Plot("Performance", "Episodes", "Rewards")
        self.plot.update_interval = .1
        self._seed = None
        self.verbose = verbose
        self.saving_interval = saving_interval
        self.load_at_reset = load_at_reset
        self.save_plot_interval = save_plot_interval

    def play(self, num_episodes=1, training=True):
        episodes_played = 0
        episode_rewards_sum = 0
        # Episode loop
        if self.seed is not None:
            self.env.seed(self._seed)
        while episodes_played < num_episodes:
            # Start of an episode
            for agent in self.agents:
                agent.reset()
            if self.load_at_reset:
                for agent in self.agents:
                    agent.load()
                    agent.save(filename=agent.name + "InUse")
            obs, action_mask = self.env.reset()
            done = False
            while not done:
                for agent in self.agents:
                    # Create a default action mask if the environment does not provide one
                    if action_mask is None:
                        action_mask = np.ones(self.env.get_action_space(), dtype=int)
                    # Get the action from the agent
                    action = agent.get_action(obs, action_mask, training=training)
                    # Perform the action and get the next state, reward, and done
                    obs, reward, done, action_mask = self.env.step(action)
                    # Make a one-time warning if the action mask is all zeros
                    if action_mask is not None and not np.any(action_mask) and not done:
                        ActionMaskWarning.warn()
                    episode_rewards_sum += reward
                    agent.give_reward(reward)
            # End of an episode
            episodes_played += 1
            if self.verbose:
                print(f"Rewards collected: {episode_rewards_sum}")
            if self.saving_interval is not None and episodes_played % self.saving_interval == 0:
                for agent in self.agents:
                    agent.save()
            if self.save_plot_interval and episodes_played % self.save_plot_interval == 0:
                self.plot.save("LatestPerformancePlot")
            # TODO: Separate the plot for each agent into subplots or different colored lines
            self.plot.add_data(episode_rewards_sum)
            episode_rewards_sum = 0
            # Train agent
            if training:
                for agent in self.agents:
                    agent.train()

    def set_agents(self, agents: List[Agent]):
        """
        Set the agents for the environment
        :param agent: The new agents
        :return: None
        """
        self.agents = agents

    def get_agents(self) -> Agent:
        """
        Obtain the environment's agents
        :return: None
        """
        return self.agents

    def seed(self, seed):
        """
        Set a seed for the environment
        :param seed: The seed to set
        :return: None
        """
        self._seed = seed
