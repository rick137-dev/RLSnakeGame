from Agent import RandomAgent, TabularReinforceAgent
from Visualize import *
from gameEnv import *


env = SnakeEnvironment()
agent = TabularReinforceAgent(evaluation_episode_max_length = 400)

#agent.train(3000,100,env,False)


episode = env.record_episode(agent)
Visualizer.visualize_episode(episode)