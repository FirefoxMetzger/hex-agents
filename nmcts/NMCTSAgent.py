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

    def deferred_plan(self):
        for _ in range(self.depth):
            yield from self.root_node.add_leaf_deferred()

    def act_greedy(self, state, active_player, info):
        # creates a generator that yields states to be evaluated by the network
        self.update_root_state(state, info)
        action = self.policy()
        return action

    def reset(self, env):
        self.root_node = NeuralSearchNode(env, agent=self.agent)

    def __str__(self):
        return f"<NMCTS({self.depth}) agent at {hex(id(self))}>"
