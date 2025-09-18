
from abc import ABC, abstractmethod
import numpy as np


"""
The movement directions, and action mappings, are:
    - 0 is UP
    - 1 is RIGHT
    - 2 is DOWN
    - 3 is LEFT
"""



NUM_ACTIONS = 4


class Agent(ABC):

    @abstractmethod
    def observe(self, env):
        pass

    @abstractmethod
    def act(self,env):
        pass



class RandomAgent(Agent):
    def __init__(self, seed= None):
        self.seed = seed
        if self.seed:
            self.rng = np.random.default_rng(self.seed)
        else:
            self.rng = np.random.default_rng()

    def observe(self,env):
        return env

    def act(self,env):
        return self.rng.integers(NUM_ACTIONS)




