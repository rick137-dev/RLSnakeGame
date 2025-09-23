from Agents.TabularQAgent import TabularQAgent
from Visualize import *
from gameEnv import *


env = SnakeEnvironment(step_limit=4000,step_reward=-0.01)
agent = TabularQAgent(learning_rate = 0.05, discount_factor = 0.999, evaluation_episodes = 200,evaluation_episode_max_length = 2000)


episode = env.record_episode(agent)
anim = Visualizer.visualize_episode(episode,save_local=False)

