import re
from abc import ABC, abstractmethod
import numpy as np
from Observer import Observer
import os
from joblib import dump, load
from gameEnv import SnakeEnvironment
import time

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



"""
NOTE: For REINFORCE Agent, during training it's important to use an env without a seed, otherwise the agent will train on the same sequence of fruit positions
During evaluation you can input a specific seed to reproduce results.

The Tabular Reinforce always has 1 H table stored on the disk, which is the best performing policy currently found
"""

class TabularReinforceAgent(Agent):

    def __init__(self,learning_rate = 0.05, discount_factor = 0.99,number_of_states = 1024 ,evaluation_episodes = 100,evaluation_episode_max_length = 2000, seed = None,  REINFORCE_CHECKPOINT = r"REINFORCE_CHECKPOINTS"):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.REINFORCE_CHECKPOINT = REINFORCE_CHECKPOINT
        self.number_of_states = number_of_states
        self.training_flag = False
        self.H_Version = None
        self.evaluation_episodes = evaluation_episodes
        self.seed = seed
        self.evaluation_episode_max_length  = evaluation_episode_max_length

        if self.seed is not None:
            self.rng = np.random.default_rng(self.seed)
        else:
            self.rng = np.random.default_rng()


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


    def get_discounted_returns(self,rewards):
        G = 0.0
        discounted_return = [0.0] * len(rewards)
        for t in range(len(rewards) - 1, -1, -1):
            G = rewards[t] + self.discount_factor * G
            discounted_return[t] = G
        return discounted_return


    def train(self, number_of_episodes, checkpoint_iteration, env,print_statements = False):

        original_flag = self.training_flag
        self.set_training_flag(True)

        evaluation_rewards = []

        evaluation_durations = []
        training_steps_done =0
        best_evaluation_reward = self.evaluate()

        start_time = time.time()

        for current_iteration in range(number_of_episodes):
            return_dict = env.record_episode(agent = self)
            rewards = return_dict["rewards"]
            env_history = return_dict["total_environment_history"]
            actions = return_dict["actions_taken"]
            G = self.get_discounted_returns(rewards)

            for index, reward_t  in enumerate(rewards):
                training_steps_done+= 1

                G_t = G[index]
                current_state = Observer.encode_1024(env_history[index])

                probabilities = TabularReinforceAgent.softmax(self.H,current_state)
                for current_action in range(NUM_ACTIONS):
                    action_security = 1 if current_action == actions[index] else 0
                    self.H[current_state][current_action] = self.H[current_state][current_action] + self.learning_rate * G_t * (action_security - probabilities[current_action])

            if (current_iteration+1) % checkpoint_iteration ==0:
                eval_reward = self.evaluate()
                evaluation_rewards.append(eval_reward)
                evaluation_durations.append(return_dict["total_steps"])

                if print_statements:
                    print("Current Iteration is " + str(current_iteration+1)+" and bext reward value seen so far is " + str(best_evaluation_reward))
                if eval_reward > best_evaluation_reward:
                    best_evaluation_reward = eval_reward
                    self.H_Version += 1
                    self.save_H_table()


        end_time = time.time()
        self.set_training_flag(original_flag)
        time_took = end_time - start_time
        print("Training completed, current iteration of H is " + str(self.H_Version) + " and the program took "+str(time_took)+ " seconds to run. Best reward seen is "+str(best_evaluation_reward))
        return evaluation_rewards , evaluation_durations , self.discount_factor, self.learning_rate, training_steps_done, best_evaluation_reward


    def evaluate(self, seed = None):
        total_return= float(0)
        original_flag = self.training_flag
        env = SnakeEnvironment() if seed is None else SnakeEnvironment(seed)

        self.set_training_flag(False)

        env.step_limit = self.evaluation_episode_max_length

        for _ in range(self.evaluation_episodes):
            result = env.record_episode(agent = self)
            total_return += float(result["total_return"])

        self.set_training_flag(original_flag)
        return total_return/self.evaluation_episodes


    def observe(self, env):
        return Observer.encode_1024(env)

    def act(self,env):
        state = self.observe(env)
        action_probabilities = TabularReinforceAgent.softmax(self.H, state)

        if self.training_flag:
            return self.rng.choice(len(action_probabilities),p = action_probabilities)
        else:
            return np.argmax(action_probabilities)



