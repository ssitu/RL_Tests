from agent.agent import Agent
from envs.env import Env
from plot import Plot

CONTROLLER_SEED = 0


def seed(seed):
    CONTROLLER_SEED = seed


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
            self.env.seed(CONTROLLER_SEED)
            obs = self.env.reset()
            done = False
            while not done:
                # Get the action from the agent
                action = self.agent.get_action(obs)
                # Perform the action and get the next state, reward, and done
                obs, reward, done = self.env.step(action)
                if render:
                    self.env.render()
                episode_rewards_sum += reward
                self.agent.give_reward(reward)
            # End of an episode
            episodes_played += 1
            if render:
                print(f"Rewards collected: {episode_rewards_sum}")
            self.plot.add_data(episode_rewards_sum)
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
