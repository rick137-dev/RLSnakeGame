from abc import ABC, abstractmethod
import numpy as np
from Observer import Observer
import os


"""
The movement directions, and action mappings, are:
    - 0 is UP
    - 1 is RIGHT
    - 2 is DOWN
    - 3 is LEFT
"""




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
        return self.rng.integers(4)


class ReinforceAgent(Agent):

    def __init__(self,learning_rate = 0.01, discount_factor = 0.9, REINFORCE_CHECKPOINT = r"REINFORCE_CHECKPOINTS" ):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.REINFORCE_CHECKPOINT = REINFORCE_CHECKPOINT
        self.training_flag = False

    def set_training_flag(self, flag):
        self.training_flag = flag


    def train(self, number_of_episodes):
        pass

    def observe(self, env):
        return Observer.encode_1024(env)

    def act(self,env):
        pass


