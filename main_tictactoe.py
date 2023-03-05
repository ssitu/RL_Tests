import argparse

from utils import use_cuda, seed_global_rng
from agent.agent_factory import AgentFactory
from agent.agent_human import AgentHuman
from controller import Controller
from envs.tictactoe import TicTacToe

DEFAULT_AGENT_NAME = "TicTacToe"


def main_human():
    # Environment
    enemy_agent = AgentHuman(
        name="Human", key_to_action_mapping={}, real_time=False)
        
    factory = AgentFactory(TicTacToe, use_cuda(False))
    env = TicTacToe(enemy_agent=enemy_agent, human_render=True)

    agent = factory.ppo_separate_wide_1d(DEFAULT_AGENT_NAME)
    control = Controller(env, [agent], load_at_reset=True)
    control.play(50000000, training=False)


def main_train(agent_name):
    seed = 7777
    seed_global_rng(seed)
    agent_factory = AgentFactory(TicTacToe, device=use_cuda(True))
    enemy_agent = agent_factory.ppo_separate_wide_1d(agent_name)
    env = TicTacToe(enemy_agent=enemy_agent, human_render=False)
    agent = agent_factory.ppo_separate_wide_1d(agent_name)
    agent.load()
    control = Controller(
        env, [agent], saving_interval=200, save_plot_interval=200)
    control.seed(seed)
    control.plot.moving_avg_len = 10000
    control.plot.start()
    control.play(50000000, training=True)
    control.plot.stop()


def main(agent_name):
    agent_factory = AgentFactory(TicTacToe, device=use_cuda(True))
    enemy_agent = agent_factory.ppo_separate_wide_1d(agent_name)
    env = TicTacToe(enemy_agent=enemy_agent, human_render=True)
    agent = agent_factory.ppo_separate_wide_1d(agent_name)
    agent.load()
    # Loading at reset allows for running the training process in the background while this agent plays in real time
    control = Controller(env, [agent], load_at_reset=True)
    control.play(50000000, training=False)


if __name__ == '__main__':
    # Parse arguments
    # --human to play the game yourself
    # --train to train the agent
    # By default, let the agent play without training
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--human", help="Play the game yourself", action="store_true")
    parser.add_argument("--train", help="Train the agent", action="store_true")
    # Specify the .pt file to load for the agent, will check the default name if not specified
    parser.add_argument(
        "--load", help="Specify the .pt file to load for the agent", default=DEFAULT_AGENT_NAME)
    args = parser.parse_args()

    if args.human:
        main_human()
    elif args.train:
        main_train(args.load)
    else:
        main(args.load)
