from utils import use_cuda, seed_global_rng

if __name__ == '__main__':
    from agent.agent_factory import AgentFactory
    from controller import Controller
    from envs.bandit import Bandit
    from envs.cartpole import CartPole
    from envs.flappybird import FlappyBird
    from envs.tictactoe import TicTacToe

    device = use_cuda(False)

    # Environment
    env = CartPole(human_render=False)

    seed = 7777
    agent_factory = AgentFactory(env, device=device)

    iterations = 2000

    seed_global_rng(seed)
    agent = agent_factory.ppo_separate_small_1d()
    control = Controller(env, [agent])
    control.seed(seed)
    control.plot.moving_avg_len = 100
    control.plot.start()
    control.play(iterations, training=True)
    control.plot.save("1")
    control.plot.stop()

    seed_global_rng(seed)
    agent2 = agent_factory.ppo_separate_small_1d_td_lambda()
    control2 = Controller(env, [agent2])
    control2.seed(seed)
    control2.plot.moving_avg_len = 100
    control2.plot.start()
    control2.play(iterations, training=True)
    control2.plot.save("2")
    control2.plot.stop()
