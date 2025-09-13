from Agent import RandomAgent
from Visualize import *
from gameEnv import *


env = SnakeEnvironment(42)
agent = RandomAgent(50)

return_dict = env.record_episode(agent)





