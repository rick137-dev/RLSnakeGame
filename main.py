from Agent import RandomAgent
from Visualize import *
from gameEnv import *


env = SnakeEnvironment()
agent = RandomAgent()

return_dict = env.record_episode(agent)
boards = return_dict["board_history"]



