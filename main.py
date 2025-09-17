from Agent import RandomAgent, TabularReinforceAgent
from Visualize import *
from gameEnv import *


env = SnakeEnvironment(step_limit=2000)
agent = TabularReinforceAgent(evaluation_episode_max_length = 800, learning_rate=0.08)

agent.train(3000,100,env,True)


episode = env.record_episode(agent)
Visualizer.visualize_episode(episode)