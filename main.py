from TabularReinforceAgent import TabularReinforceAgent
from Visualize import *
from gameEnv import *


env = SnakeEnvironment(step_limit=4000,step_reward=-0.01)
agent = TabularReinforceAgent(evaluation_episode_max_length = 4000, learning_rate=0.02, discount_factor=1)

episode = env.record_episode(agent)
Visualizer.visualize_episode(episode)