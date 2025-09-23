from Agents.Agent import *
from Observer import Observer
import os
from gameEnv import SnakeEnvironment
import numpy as np

class DeepQAgent(Agent):

    def __init__(self, DEEP_Q_CHECKPOINT = r"DEEP_Q_CHECKPOINTS"):
        self.DEEP_Q_CHECKPOINT = DEEP_Q_CHECKPOINT

    def observe(self, env):
        pass

    def act(self,env):
        pass
