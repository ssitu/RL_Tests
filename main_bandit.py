from utils import use_cuda, seed_global_rng
from controller import Controller
from envs.bandit import Bandit
from agent.agent_factory import AgentFactory

if __name__ == '__main__':
    device = use_cuda(False)
    # Environment
    env = Bandit(30, human_render=True)
    seed = 777
    iterations = 2000

    agent_factory = AgentFactory(env, device)
    seed_global_rng(seed)
    agent = agent_factory.ppo_twohead_small_1d()
    control = Controller(env, [agent])
    control.seed(seed)
    control.plot.moving_avg_len = 100
    control.plot.start()
    control.play(iterations, training=True)
    control.plot.stop()