from minihex import HexGame, player, random_policy
import trueskill
import numpy as np


class Agent(object):
    def __init__(self):
        self.rating = trueskill.Rating()

    def act(self, state, active_player, info=None):
        raise NotImplementedError()

    def reset(self, state):
        raise NotImplementedError()

    def __str__(self):
        return f"Abstract Agent"


class RandomAgent(Agent):
    def act(self, state, active_player, info):
        return random_policy(state, active_player, info)

    def reset(self, env):
        return

    def __str__(self):
        return "Random"
