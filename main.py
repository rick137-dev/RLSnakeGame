from Agent import RandomAgent, TabularReinforceAgent
from Visualize import *
from gameEnv import *


env = SnakeEnvironment()
agent = TabularReinforceAgent(evaluation_episode_max_length = 200)

agent.train(500,100,env,True)


episode = env.record_episode(agent)
Visualizer.visualize_episode(episode)