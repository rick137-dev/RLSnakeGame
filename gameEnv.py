import numpy as np
from collections import deque

""""
Environment is  15x15 grid with toroidal shape, which means as the snake moves across one side
it appears on the opposite side. 

It is represented by a numpy array:
    - 0 is empty
    - 1 is snake body (including head)
    - 2 is fruit
    
The movement directions, and action mappings, are:
    - 0 is UP
    - 1 is RIGHT
    - 2 is DOWN
    - 3 is LEFT
    
    
The reward is +1 for eating fruit, -2 for death, and +alpha for every step.
Initial position of the snake head is in the centre, and the initial direction of movement is nowhere, thus allowing 
the player or agent to make the first move. 
    
"""


direction_map = {
                0 : (-1,0),
                1: (0,1),
                2: (1,0),
                3 : (0,-1)

}

BOARD_SIZE = 15

class SnakeEnvironment:

    def __init__(self,fruit_spawn_seed=None, step_reward = 0.01, step_limit = 2000):
        self.fruit_spawn_seed = fruit_spawn_seed
        if self.fruit_spawn_seed is not None:
            self.rng =  np.random.default_rng(self.fruit_spawn_seed)
        else:
            self.rng = np.random.default_rng()

        self.board = None
        self.head_position = (None,None)
        self.movement_directions = (None,None)
        self.snake_body_deque = None
        self.fruit_position = (None,None)
        self.growing = None
        self.step_reward = step_reward
        self.current_steps = None
        self.step_limit = step_limit
        self.reset()



    def reset(self):


        self.board = np.zeros((BOARD_SIZE,BOARD_SIZE),dtype=np.int8)
        new_head_position = (BOARD_SIZE//2,BOARD_SIZE//2)
        self.growing = 0
        self.snake_body_deque = deque()
        self.set_new_snake_head(new_head_position)
        self.movement_directions = (0,0)
        self.current_steps =0
        self.set_fruit()



    def get_free_positions(self):
        positions = []
        for i in range(0,BOARD_SIZE):
            for j in range(0,BOARD_SIZE):
                if self.board[i][j] ==0:
                    positions.append((i,j))
        return positions


    def set_fruit(self):
        positions = self.get_free_positions()
        if positions is None or len(positions)==0:
            return False
        rand_index = self.rng.integers(len(positions) )
        new_fruit = positions[rand_index]
        position_i, position_j = new_fruit
        self.fruit_position = new_fruit
        self.board[position_i][position_j] = 2
        return True


    def remove_snake_end(self):
        position_i , position_j = self.snake_body_deque.pop()
        self.board[position_i][position_j] = 0

    def set_new_snake_head(self,new_head_position):
        self.head_position = new_head_position
        position_i , position_j = new_head_position
        self.snake_body_deque.appendleft(new_head_position)
        self.board[position_i][position_j] = 1

    def step(self, action):
        temp_direction = self.movement_directions

        self.movement_directions = direction_map[action]
        di, dj = self.movement_directions
        current_i , current_j = self.head_position
        new_i , new_j = (current_i + di)%BOARD_SIZE , (current_j + dj)%BOARD_SIZE


        current_reward =0
        ate_fruit = False
        self.current_steps += 1
        current_reward += self.step_reward
        self_intersect = False
        truncated = False

        trivial_intersect_point = None

        if len(self.snake_body_deque)>1:
            trivial_intersect_point = self.snake_body_deque[1]

        if trivial_intersect_point is not None and new_i == trivial_intersect_point[0] and new_j == trivial_intersect_point[1]:
            self.movement_directions = temp_direction
            di, dj = self.movement_directions
            new_i , new_j = (current_i + di)%BOARD_SIZE , (current_j + dj)%BOARD_SIZE


        if self.current_steps >=self.step_limit:
            truncated = True

        if self.board[new_i][new_j] == 1:
            #Check if intersecting tail that will not be there
            current_tail = self.snake_body_deque[-1]
            stepping_null_tail = new_i == current_tail[0] and new_j == current_tail[1] and self.growing==0
            if stepping_null_tail is False:
                current_reward -= 2
                self_intersect = True


        if self.board[new_i][new_j] == 2:
            ate_fruit = True

        if self_intersect == False and truncated == False:
            new_head_position = new_i , new_j
            self.set_new_snake_head(new_head_position)

            if ate_fruit is True:
                current_reward += 1
                self.growing += 1
                self.set_fruit()

            if self.growing ==0:
                self.remove_snake_end()
            else:
                self.growing -= 1



        return current_reward, self.current_steps, truncated, self_intersect, ate_fruit

    def record_episode(self,agent):
        truncated = False
        self_intersect = False
        self.reset()
        total_reward =0
        actions_taken = []
        rewards = []
        fruit_eaten = 0
        head_positions = []
        board_history = []
        fruit_eat_history = []



        head_positions.append(self.head_position)
        board_history.append(self.board.copy())


        #The head positions and board is duplicated at the end, since head is not changed if it's not a valid move
        while truncated is False and self_intersect is False:
            action = agent.act(self)
            actions_taken.append(action)
            temp_reward , _ , truncated , self_intersect, ate_fruit = self.step(action)
            rewards.append(temp_reward)
            total_reward += temp_reward
            board_history.append(self.board.copy())
            fruit_eat_history.append(ate_fruit)
            head_positions.append(self.head_position)
            if ate_fruit:
                fruit_eaten+=1


        return_dict = {
            "total_return" :total_reward,
            "self_intersected" : self_intersect,
            "truncated" : truncated,
            "rewards" : rewards,
            "total_fruit_eaten" : fruit_eaten,
            "actions_taken" : actions_taken,
            "total_steps" : self.current_steps,
            "max_steps" : self.step_limit,
            "board_history" : board_history,
            "fruit_eaten_history":fruit_eat_history,
            "head_positions":head_positions

        }


        return return_dict





