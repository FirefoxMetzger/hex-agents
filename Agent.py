from minihex.HexGame import HexGame, player
import trueskill
import numpy as np


class Agent(object):
    def __init__(self):
        self.rating = trueskill.Rating()

    def act(self, state, active_player, info=None):
        raise NotImplementedError()

    def __str__(self):
        return f"Abstract Agent"


class RandomAgent(Agent):
    def act(self, state, active_player):
        board = state
        coords = np.where(board[2, ...] == 1)
        idx = np.ravel_multi_index(coords, board.shape[1:])
        choice = np.random.randint(len(idx))
        return idx[choice]

    def __str__(self):
        return "Random"
