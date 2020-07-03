from nmcts.NeuralSearchNode import NeuralSearchNode
import numpy as np
from minihex.HexGame import HexGame
from minihex import player
from mcts.MCTSAgent import MCTSAgent
import random


class NMCTSAgent(MCTSAgent):
    def __init__(self,
                 agent=None,
                 model_file="best_model.h5",
                 env=None,
                 depth=1000,
                 board_size=9,
                 active_player=player.BLACK, player=player.BLACK):
        super(MCTSAgent, self).__init__()
        if env is None:
            board = player.EMPTY * np.ones((board_size, board_size))
            self.root_node = NeuralSearchNode(HexGame(
                active_player,
                board,
                player
            ), agent, model_file)
        else:
            self.root_node = NeuralSearchNode(env, agent, model_file)
        self.depth = depth
        self.agent = agent
        self.model_file = model_file

    def reset(self, env):
        self.root_node = NeuralSearchNode(env, self.agent, self.model_file)

    def __str__(self):
        return f"NMCTS({self.depth})"
