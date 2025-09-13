from abc import ABC, abstractmethod
import numpy as np

"""
The movement directions, and action mappings, are:
    - 0 is UP
    - 1 is RIGHT
    - 2 is DOWN
    - 3 is LEFT
"""

REINFORCE_CHECKPOINT = r"REINFORCE_CHECKPOINTS"


class Agent(ABC):

    @abstractmethod
    def observe(self, env):
        pass

    @abstractmethod
    def act(self,env):
        pass



class RandomAgent(Agent):

    def observe(self,env):
        return env

    def act(self,env):
        return np.random.randint(0,3)


class ReinforceAgent(Agent):

    def __init__(self,):
        pass


