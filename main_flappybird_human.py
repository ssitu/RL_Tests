if __name__ == '__main__':
    from utils import use_cuda
    from agent.agent_factory import AgentFactory
    from agent.agent import Agent
    from controller import Controller
    from envs.flappybird import FlappyBird

    # Environment
    env = FlappyBird(human_render=True, truncate=False, fastest_speed=False, human_player=True)

    # Empty agent
    agent = Agent(name="human")
    control = Controller(env, agent, load_at_reset=True)
    control.play(50000000, training=False)
