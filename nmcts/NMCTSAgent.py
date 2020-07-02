from nmcts.NeuralSearchNode import NeuralSearchNode
import numpy as np
from minihex.HexGame import HexGame
from minihex import player
from Agent import Agent
from mcts.MCTSAgent import MCTSAgent
import random


class NMCTSAgent(MCTSAgent):
    def __init__(self, env, depth=1000):
        super(MCTSAgent, self).__init__()
        self.root_node = NeuralSearchNode(env)
        self.depth = depth

    def reset(self, env):
        self.root_node = NeuralSearchNode(env)

    def __str__(self):
        return f"NMCTS({self.depth})"
