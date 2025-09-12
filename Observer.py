
"""
The Observer class contains various encoding methods for the environment
"""

class Observer:

    """
    This method uses the following bits to encode the environment to a number in the range [0,255]:
                - Fruit Quadrant Bit : Determines in which of the 4 quadrants the fruit is currently in, this is 1 number in the range [0,3]
                - Danger Bit : Determines if in the 3 possible directions the agent can go to (not counting trivial 180 degree turns) there is the snake body, these are 3 bits, each can be 0 or 1
                - Head Position Bit : Determines in which of the 4 Quadrants the head is in, similar to fruit quadrant bit
    """
    def encode_256(self,env):
        pass

