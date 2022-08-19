from utils import use_cuda, seed_global_rng

if __name__ == '__main__':
    from agent.agent_factory import AgentFactory
    from controller import Controller
    from envs.bandit import Bandit
    from envs.cartpole import CartPole
    from envs.flappybird import FlappyBird

    device = use_cuda(True)

    # Environment
    env = FlappyBird(human_render=False, truncate=True)

    seed = 7777
    agent_factory = AgentFactory(env, device=device)

    seed_global_rng(seed)
    agent = agent_factory.ppo_separate_critic_heavy_1d("FlappyBird")
    agent.load()
    control = Controller(env, agent, saving_interval=200, save_plot_interval=200)
    control.seed(seed)
    control.plot.moving_avg_len = 10000
    # control.plot.start()
    control.play(50000000, training=True)
    # control.plot.save("1")
    # control.plot.stop()

    # env2 = FlappyBird(human_render=False)
    # seed_rng(seed)
    # agent2 = agent_factory.agac_1d()
    # control2 = Controller(env2, agent2)
    # control2.seed(seed)
    # control2.plot.start()
    # control2.play(500, training=True)
    # control2.plot.save("2")
    # control2.plot.stop()
