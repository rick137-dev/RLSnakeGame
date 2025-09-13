import matplotlib.pyplot as plt
import numpy as np
import time
import imageio

"""
Visualizer class stores all methods relating to visualizations
"""


COLOR_BOARD  = np.array([0.90, 0.90, 0.90])
COLOR_SNAKE= np.array([0.10, 0.70, 0.10])
COLOR_HEAD = np.array([0.00, 0.45, 0.00])
COLOR_FRUIT = np.array([1.00, 0.55, 0.00])



class Visualizer:

    @staticmethod
    def convert_board_to_image(board, head_position):
        board_size = board.shape[0]
        img = np.full((board_size, board_size, 3), COLOR_BOARD, dtype=np.float32)
        for i in range(board_size):
            for j in range(board_size):
                if board[i][j] ==1:
                    img[i,j,:] = COLOR_SNAKE
                if board[i][j] == 2:
                    img[i, j, :] = COLOR_FRUIT

        head_i , head_j = head_position
        img[head_i,head_j,:] = COLOR_HEAD
        return img


    @staticmethod
    def convert_to_image_list(episode):
        board_history = episode["board_history"]
        head_positions = episode["head_positions"]

        images = []
        for i in range(len(board_history)):
            img = Visualizer.convert_board_to_image(board_history[i],head_positions[i])
            images.append(img)

        return images



    @staticmethod
    def visualize_episode(episode, fps = 5, save_local = False):
        images = Visualizer.convert_to_image_list(episode)
        plt.ion()
        fig, ax = plt.subplots()
        im = ax.imshow(images[0], interpolation="nearest")
        ax.axis("off")
        for frame in images:
            im.set_data(frame)
            plt.draw()
            plt.pause(1.0 / fps)

        plt.ioff()
        plt.show()








