import numpy as np
from minihex import empty_tiles, player
from minihex.HexGame import HexGame
from copy import deepcopy

VALUE_CONSTANT = np.sqrt(2)


class SearchNode(object):
    def __init__(self, env):
        self.total_simulations = 0.0
        self.total_wins = 0
        self.children = dict()
        self.env = env
        self.available_actions = env.get_possible_actions()

    def action_value(self, action, is_greedy=False):
        if action not in self.children:
            return np.inf
        elif self.children[action].is_terminal:
            # assert self.total_simulations == 1
            child = self.children[action]
            return 1 - child.total_wins / self.total_simulations
        else:
            child = self.children[action]
            parent_sims = self.total_simulations
            child_sims = child.total_simulations
            total_wins = child.total_wins
            win_rate = 1 - total_wins / child_sims
            upper_bound_estimate = (VALUE_CONSTANT *
                                    np.sqrt(np.log(parent_sims) / child_sims))
            if is_greedy:
                return win_rate
            else:
                return win_rate + upper_bound_estimate

    @property
    def is_terminal(self):
        return self.env.done

    def select(self):
        child_values = list()
        for action in self.available_actions:
            child_value = self.action_value(action)
            child_values.append(child_value)

        child_values = np.array(child_values)
        # if len(child_values) == 0:
        #     import pdb; pdb.set_trace()
        value = max(child_values)
        child_idx = np.where(child_values == value)[0]
        chosen_idx = np.random.choice(child_idx)

        return self.available_actions[chosen_idx]

    def expand(self, action):
        if self.is_terminal:
            return self
        new_env = deepcopy(self.env)
        new_env.make_move(action)
        child = SearchNode(new_env)
        self.children[action] = child
        return child

    def simulate(self):
        sim_env = deepcopy(self.env)
        while not sim_env.done:
            valid_moves = sim_env.get_possible_actions()
            action = np.random.choice(valid_moves)
            winner = sim_env.make_move(action)

        return sim_env.winner

    def backup(self, winner):
        self.total_simulations += 1
        if self.env.active_player == winner:
            self.total_wins += 1
