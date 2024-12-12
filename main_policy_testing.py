if __name__ == '__main__':
    from utils import use_cuda
    from agent.agent_factory import AgentFactory
    from controller import Controller
    from envs.flappybird import FlappyBird

    # Environment
    env = FlappyBird(human_render=True, truncate=False, fastest_speed=True)

    device = use_cuda(False)
    agent_factory = AgentFactory(env, device)

    agent = agent_factory.ppo_separate_critic_heavy_1d("FlappyBirdFinal")
    agent.set_greedy(True)
    control = Controller(env, [agent], load_at_reset=True)
    control.play(50000000, training=False)
