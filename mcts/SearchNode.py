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

        board_size = env.board.shape[1]
        self.greedy_Q = np.inf * np.ones(board_size ** 2,
                                         dtype=np.float32)
        self.Q = np.inf * np.ones(board_size ** 2,
                                  dtype=np.float32)

    def add_leaf(self):
        if self.is_terminal:
            winner = self.env.winner
            self.backup(winner)
            return winner

        action = self.select()

        if action in self.children:
            winner = self.children[action].add_leaf()
        else:
            child = self.expand(action)
            winner = child.simulate()
            child.backup(winner)

        self.backup(winner, action)
        return winner

    def action_value(self, action, is_greedy=False):
        if is_greedy:
            return self.greedy_Q[action]
        else:
            return self.Q[action]

    @property
    def is_terminal(self):
        return self.env.done

    def select(self):
        action_scores = self.Q[self.available_actions]
        best_actions = np.where(action_scores == np.max(action_scores))[0]
        idx = np.random.randint(len(best_actions))
        best_action_idx = best_actions[idx]
        return self.available_actions[best_action_idx]

    def expand(self, action):
        new_env = deepcopy(self.env)
        new_env.make_move(action)
        child = SearchNode(new_env)
        self.children[action] = child
        return child

    def simulate(self):
        sim_env = deepcopy(self.env)
        while not sim_env.done:
            valid_moves = sim_env.get_possible_actions()
            idx = np.random.randint(len(valid_moves))
            action = valid_moves[idx]
            winner = sim_env.make_move(action)

        return sim_env.winner

    def backup(self, winner, action=None):
        self.total_simulations += 1
        if self.env.active_player == winner:
            self.total_wins += 1

        if action is not None:
            self.update_action_value(action)

    def update_action_value(self, action):
        child = self.children[action]

        parent_sims = self.total_simulations
        child_sims = child.total_simulations
        child_total_wins = child.total_wins
        win_rate = 1 - child_total_wins / child_sims
        upper_bound_estimate = (VALUE_CONSTANT *
                                np.sqrt(np.log(parent_sims) / child_sims))
        self.Q[action] = win_rate + upper_bound_estimate
        self.greedy_Q[action] = win_rate
