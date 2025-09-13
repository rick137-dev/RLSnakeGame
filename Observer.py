import gameEnv
from gameEnv import  inverse_direction_map

"""
The Observer class contains various encoding methods for the environment
"""


class Observer:

    """
    The encode_1024 method uses the following bits to encode the environment to a number in the range [0,1023]:
                - Fruit Quadrant Bit : Determines in which of the 4 quadrants the fruit is currently in, this is 1 number in the range [0,3]
                - Danger Bit : Determines if in the 3 possible directions the agent can go to (not counting trivial 180 degree turns) there is the snake body, these are 3 bits, each can be 0 or 1, 1 means danger
                - Head Position Bit : Determines in which of the 4 Quadrants the head is in, similar to fruit quadrant bit
                - Head Direction Bit : in which direction is the head currently moving in, this is 1 number in range [0,3]
                - Growth bit - simply determines if the snake is growing or not, could be 1 or 0

            Total space range is 1024.
    """

    @staticmethod
    def get_quadrant_bit(size, position):
        pos_i , pos_j = position

        if pos_i <= size//2 and pos_j <= size//2:
            return 0
        elif pos_i<=size//2 and pos_j > size//2:
            return 1
        elif pos_i>size//2 and pos_j > size//2:
            return 2
        elif pos_i>size//2 and pos_j <= size//2:
            return 3
        else:
            return -1 #Should never happen


    @staticmethod
    def encode_1024(env: gameEnv.SnakeEnvironment):

        head_direction_bit = inverse_direction_map[env.movement_directions]
        growth_bit = 0
        if env.growing >0:
            growth_bit = 1

        fruit_quadrant_bit = Observer.get_quadrant_bit(env.board_size, env.fruit_position)
        head_position_bit = Observer.get_quadrant_bit(env.board_size, env.head_position)

        if fruit_quadrant_bit == -1 or head_position_bit == -1:
            return -1

        danger_bit_left = 0
        danger_bit_right = 0
        danger_bit_up = 0

        head_i, head_j = env.head_position
        current_direction = inverse_direction_map[env.movement_directions]

        if current_direction == 0:
            if env.board[head_i][(head_j - 1) % env.board_size] == 1:
                danger_bit_left = 1
            if env.board[head_i][(head_j + 1) % env.board_size] == 1:
                danger_bit_right = 1
            if env.board[(head_i - 1) % env.board_size][head_j] == 1:
                danger_bit_up = 1

        elif current_direction == 1:
            if env.board[(head_i - 1) % env.board_size][head_j] == 1:
                danger_bit_left = 1
            if env.board[(head_i + 1) % env.board_size][head_j] == 1:
                danger_bit_right = 1
            if env.board[head_i][(head_j + 1) % env.board_size] == 1:
                danger_bit_up = 1

        elif current_direction == 2:
            if env.board[head_i][(head_j + 1) % env.board_size] == 1:
                danger_bit_left = 1
            if env.board[head_i][(head_j - 1) % env.board_size] == 1:
                danger_bit_right = 1
            if env.board[(head_i + 1) % env.board_size][head_j] == 1:
                danger_bit_up = 1

        else:
            if env.board[(head_i + 1) % env.board_size][head_j] == 1:
                danger_bit_left = 1
            if env.board[(head_i - 1) % env.board_size][head_j] == 1:
                danger_bit_right = 1
            if env.board[head_i][(head_j - 1) % env.board_size] == 1:
                danger_bit_up = 1


        #Here we pack all the information in 1 number using bit shifts

        danger_code = (danger_bit_up << 2) | (danger_bit_right << 1) | danger_bit_left

        code = (
                (fruit_quadrant_bit << 8) |
                (danger_code << 5) |
                (head_position_bit << 3) |
                (head_direction_bit << 1) |
                growth_bit
                )

        return code


