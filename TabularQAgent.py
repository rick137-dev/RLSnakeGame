from Agent import *
from Observer import Observer
import os
from joblib import dump, load
from gameEnv import SnakeEnvironment
import time
import re
import numpy as np

class TabularQLearning(Agent):
    def __init__(self,learning_rate = 0.05, discount_factor = 0.99,number_of_states = 1024 ,evaluation_episodes = 100,evaluation_episode_max_length = 2000, seed = None,training_seed = None,  TABULAR_Q_CHECKPOINT = r"TABULAR_Q_CHECKPOINT"):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.number_of_states = number_of_states
        self.evaluation_episodes = evaluation_episodes
        self.evaluation_episode_max_length = evaluation_episode_max_length
        self.seed = seed
        self.training_seed = training_seed
        self.TABULAR_Q_CHECKPOINT = TABULAR_Q_CHECKPOINT
        self.training_flag = False
        self.Q = None
        self.Q_Version = None
        self.epsilon = 0.1


        if self.seed is not None:
            self.act_rng = np.random.default_rng(self.seed)
        else:
            self.act_rng = np.random.default_rng()

        if self.training_seed is not None:
            self.train_rng = np.random.default_rng(self.training_seed)
        else:
            self.train_rng = np.random.default_rng()


        if os.path.isdir(os.path.join(os.getcwd(), self.TABULAR_Q_CHECKPOINT)):
            self.set_Q_table()
        else:
            os.mkdir(self.TABULAR_Q_CHECKPOINT)
            self.Q  = 1e-3 * np.random.randn(self.number_of_states,NUM_ACTIONS)
            self.Q_Version = 1
            self.save_Q_table()


    def set_training_flag(self,flag):
        self.training_flag = flag

    def save_Q_table(self):
        files = os.listdir(self.TABULAR_Q_CHECKPOINT)
        if len(files)==1:
            file= files[0]
            filename =  os.path.join(self.TABULAR_Q_CHECKPOINT, file)
            os.remove(filename)
        new_filename = "Q_" + str(self.Q_Version) + ".joblib"
        new_filename = os.path.join(self.TABULAR_Q_CHECKPOINT,new_filename)
        dump(self.Q,new_filename)


    def set_Q_table(self):
        if os.path.isdir(os.path.join(os.getcwd(), self.TABULAR_Q_CHECKPOINT)):
            files = os.listdir(self.TABULAR_Q_CHECKPOINT)
            Q_file = files[0]
            match = re.match(r"Q_(\d+).joblib", Q_file)
            self.Q_Version = int(match.group(1))
            filename = os.path.join(self.TABULAR_Q_CHECKPOINT, Q_file)
            self.Q = load(filename)

    def evaluate(self,seed = None):
        total_return = float(0)
        original_flag = self.training_flag
        self.set_training_flag(False)

        env = SnakeEnvironment() if seed is None else SnakeEnvironment(seed)

        env.set_max_step_limit(self.evaluation_episode_max_length)
        eval_returns = []

        for _ in range(self.evaluation_episodes):
            return_dict = env.record_episode(agent = self)
            total_return += float(return_dict["total_return"])
            eval_returns.append(float(return_dict["total_return"]))


        self.set_training_flag(original_flag)
        return total_return/self.evaluation_episodes , eval_returns


    def train(self,training_episodes, checkpoint_iteration, env:SnakeEnvironment, max_evaluation_episodes,print_statement = False):
        original_flag = self.training_flag
        self.set_training_flag(True)



        for iteration in range(training_episodes):
            return_dict = env.record_episode(agent = self)
            envs = return_dict["total_environment_history"]

            if (iteration+1)%checkpoint_iteration ==0:
                print("")
                current_step_limit = env.get_max_step_limit()
                env.set_max_step_limit(max_evaluation_episodes)


                env.set_max_step_limit(current_step_limit)


        self.set_training_flag(original_flag)

    def observe(self, env):
        return Observer.encode_1024(env)

    def act(self,env):
        state = self.observe(env)
        action_probabilities = self.Q[state]
        if self.training_flag:
            p = [self.epsilon, 1-self.epsilon]
            random_action_choice = self.train_rng.choice(len(p),p=p)
            if random_action_choice ==0:
                return self.act_rng.integers(NUM_ACTIONS)
            else:
                return np.argmax(action_probabilities)

        else:
            return np.argmax(action_probabilities)
