from nmcts.NeuralSearchNode import NeuralSearchNode
import numpy as np
from minihex.HexGame import HexGame
from minihex import player
from mcts.MCTSAgent import MCTSAgent
import random


class NMCTSAgent(MCTSAgent):
    def __init__(self,
                 env=None,
                 depth=1000,
                 board_size=9,
                 active_player=player.BLACK, player=player.BLACK,
                 **kwargs):
        super(MCTSAgent, self).__init__()
        if env is None:
            board = player.EMPTY * np.ones((board_size, board_size))
            env = HexGame(active_player, board, player)

        self.root_node = NeuralSearchNode(env, **kwargs)
        self.depth = depth
        self.agent = self.root_node.network_agent
        self.board_size = board_size
        self.active_player = active_player
        self.player = player

    def reset(self, env=None):
        if env is None:
            board = player.EMPTY * np.ones((self.board_size, self.board_size))
            env = HexGame(self.active_player, board, self.player)
        self.root_node = NeuralSearchNode(env, agent=self.agent)

    def __str__(self):
        return f"<NMCTS({self.depth}) agent at {hex(id(self))}>"
