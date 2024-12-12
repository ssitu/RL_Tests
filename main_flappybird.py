import argparse

from utils import use_cuda, seed_global_rng
from agent.agent_factory import AgentFactory
from agent.agent import Agent
from agent.agent_human import AgentHuman
from controller import Controller
from envs.flappybird import FlappyBird, KEY_ACTION_MAPPING

DEFAULT_AGENT_NAME = "FlappyBird"


def main_human():
    env = FlappyBird(human_render=True, truncate=False,
                     fastest_speed=False)
    agent = AgentHuman(name="human", key_to_action_mapping=KEY_ACTION_MAPPING, real_time=True)
    control = Controller(env, [agent])
    control.play(50000000, training=False)


def main_train(agent_name):
    env = FlappyBird(human_render=False, truncate=True)
    seed = 7777
    agent_factory = AgentFactory(env, device=use_cuda(True))
    seed_global_rng(seed)
    agent = agent_factory.ppo_separate_critic_heavy_1d(agent_name)
    agent.load()
    actor, critic = agent.optimizer.param_groups
    # actor['lr'] = 0.000001
    # critic['lr'] = 0.000001
    control = Controller(
        env, [agent], saving_interval=200, save_plot_interval=200)
    control.seed(seed)
    control.plot.moving_avg_len = 10000
    control.plot.start()
    control.play(50000000, training=True)
    control.plot.stop()


def main(agent_name):
    env = FlappyBird(human_render=True, truncate=False, fastest_speed=True)
    agent_factory = AgentFactory(env, device=use_cuda(True))
    agent = agent_factory.ppo_separate_critic_heavy_1d(agent_name)
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
