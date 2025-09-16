import re
from abc import ABC, abstractmethod
import numpy as np
from Observer import Observer
import os
from joblib import dump, load

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
        return self.rng.integers(4)


class TabularReinforceAgent(Agent):

    def __init__(self,learning_rate = 0.01, discount_factor = 0.9,number_of_states = 1024 ,evaluation_episodes = 1000, REINFORCE_CHECKPOINT = r"REINFORCE_CHECKPOINTS"):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.REINFORCE_CHECKPOINT = REINFORCE_CHECKPOINT
        self.number_of_states = number_of_states
        self.training_flag = False
        self.H_Version = None
        self.evaluation_episodes = evaluation_episodes

        if os.path.isdir(os.path.join(os.getcwd(), self.REINFORCE_CHECKPOINT)):
            self.set_H_table()
        else:
            os.mkdir(self.REINFORCE_CHECKPOINT)
            self.H  = 1e-3 * np.random.randn(self.number_of_states,NUM_ACTIONS)
            self.H_Version = 1
            self.save_H_table()


    def set_training_flag(self, flag):
        self.training_flag = flag

    def save_H_table(self):
        files = os.listdir(self.REINFORCE_CHECKPOINT)
        if len(files)==1:
            file= files[0]
            filename =  os.path.join(self.REINFORCE_CHECKPOINT, file)
            os.remove(filename)
        new_filename = "H_" + str(self.H_Version) + ".joblib"
        new_filename = os.path.join(self.REINFORCE_CHECKPOINT,new_filename)
        dump(self.H,new_filename)

    def set_H_table(self):
        if os.path.isdir(os.path.join(os.getcwd(), self.REINFORCE_CHECKPOINT)):
            files = os.listdir(self.REINFORCE_CHECKPOINT)
            H_file = files[0]
            match = re.match(r"H_(\d+).joblib", H_file)
            self.H_Version = int(match.group(1))
            filename = os.path.join(self.REINFORCE_CHECKPOINT, H_file)
            self.H = load(filename)


    @staticmethod
    def softmax(H,state):
        Z = H[state] - H[state].max() #This is done for numerical stability, doesn't impact choice
        exp_Z = np.exp(Z)
        return exp_Z / exp_Z.sum()


    def train(self, number_of_episodes):
        pass

    def evaluate(self):
        pass

    def observe(self, env):
        return Observer.encode_1024(env)

    def act(self,env):
        if self.training_flag:
            pass
        else:
            pass


