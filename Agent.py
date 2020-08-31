from minihex import HexGame, player, random_policy
import trueskill
import numpy as np
import random as random


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
    def __init__(self, board_size=9):
        self.all_actions = np.arange(board_size ** 2)
        super(RandomAgent, self).__init__()

    def act(self, state, active_player, info):
        valid_actions = self.all_actions[(state == 2).flatten()]
        idx = int(random.random() * len(valid_actions))
        return valid_actions[idx]

    def act_greedy(self, state, active_player, info):
        return self.act(state, active_player, info)

    def update_root_state_deferred(self, info):
        return
        yield

    def deferred_plan(self):
        return
        yield

    def reset(self, env):
        return

    def __str__(self):
        return "Random"
