from abc import ABC, abstractmethod
import numpy as np

#The 4 actions, move LEFT, RIGHT, UP DOWN are classified using 0,1,2,3

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