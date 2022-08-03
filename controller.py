from agent.agent import Agent
from envs.env import Env
from plot import Plot


class Controller:
    """
    Handles interactions between the environment and agent
    """

    def __init__(self, env: Env, agent: Agent):
        self.env = env
        self.agent = agent
        self.plot = Plot("Performance", "Episodes", "Rewards")

    def play(self, num_episodes=1, training=True, render=False):
        episodes_played = 0
        episode_rewards_sum = 0
        # Episode loop
        while episodes_played < num_episodes:
            # Start of an episode
            self.agent.reset()
            obs = self.env.reset()
            done = False
            while not done:
                action = self.agent.get_action(obs)
                obs, reward, done = self.env.step(action)
                if render:
                    self.env.render()
                episode_rewards_sum += reward
                self.agent.give_reward(reward)
            # End of an episode
            episodes_played += 1
            if render:
                print(f"Rewards collected: {episode_rewards_sum}")
            self.plot.plot_data(episode_rewards_sum)
            episode_rewards_sum = 0
            # Train agent
            if training:
                self.agent.train()

    def set_agent(self, agent: Agent):
        """
        Set the agent for the environment
        :param agent: The new agent
        :return: None
        """
        self.agent = agent

    def get_agent(self) -> Agent:
        """
        Obtain the environment's agent
        :return: None
        """
        return self.agent

    def start_plot(self):
        self.plot.start_updates()

    def stop_plot(self):
        self.plot.stop_updates()

    def __del__(self):
        self.stop_plot()

    def save_plot(self, name=""):
        self.plot.save(name=name)
