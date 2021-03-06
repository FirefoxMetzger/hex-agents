import numpy as np
from minihex import player
from minihex.HexGame import HexGame, HexEnv
import minihex
import gym
from copy import deepcopy
from Agent import RandomAgent
import random

from scheduler.tasks import Rollout

VALUE_CONSTANT = np.sqrt(2)


class SearchNode(object):
    def __init__(self, env):
        self.total_simulations = 0.0
        self.total_wins = 0
        self.children = dict()

        self.env = env

        self.available_actions = env.get_possible_actions()
        self.winner = env.winner
        self.is_terminal = env.done
        self.active_player = env.active_player

        board_size = env.board.shape[1]
        self.greedy_Q = np.inf * np.ones(board_size ** 2,
                                         dtype=np.float32)
        self.greedy_Q = self.greedy_Q.tolist()
        self.Q = np.inf * np.ones(board_size ** 2,
                                  dtype=np.float32)
        self.Q = self.Q.tolist()
        self.agent = RandomAgent(board_size=board_size)

    def add_leaf(self):
        if self.is_terminal:
            winner = self.winner
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

    def select(self):
        best_actions = list()
        best_q = self.Q[self.available_actions[0]]
        for action in self.available_actions:
            if best_q == self.Q[action]:
                best_actions.append(action)
            elif self.Q[action] > best_q:
                best_actions = [action]
                best_q = self.Q[action]

        if len(best_actions) > 1:
            best_action = best_actions[random.randint(0, len(best_actions)-1)]
        else:
            best_action = best_actions[0]

        return best_action

    def expand(self, action):
        new_env = deepcopy(self.env)
        new_env.make_move(action)
        child = SearchNode(new_env)
        self.children[action] = child

        return child

    def simulate(self):
        if self.is_terminal:
            return self.winner

        env = HexEnv(
            opponent_policy=self.agent.act,
            player_color=self.active_player,
            active_player=self.active_player,
            board=self.env.board.copy(),
            regions=self.env.regions.copy())

        state, info = env.reset()
        done = False
        while not done:
            action = self.agent.act(state[0], state[1], info)
            state, reward, done, info = env.step(action)

        return env.simulator.winner

    def backup(self, winner, action=None):
        self.total_simulations += 1
        if self.active_player == winner:
            self.total_wins += 1

        if action is not None:
            self.update_action_value(action)

    def backup_deferred(self, winner, action=None):
        self.total_simulations += 1
        if self.active_player == winner:
            self.total_wins += 1

        if action is not None:
            yield from self.update_action_value_deferred(action)

    def update_action_value_deferred(self, action):
        return self.update_action_value(action)
        yield

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

    def add_leaf_deferred(self):
        if self.is_terminal:
            winner = self.winner
            yield from self.backup_deferred(winner)
            return winner

        action = self.select()

        if action in self.children:
            gen = self.children[action].add_leaf_deferred()
            winner = yield from gen
        else:
            gen = self.batched_expand_and_simulate(action)
            winner = yield from gen

        yield from self.backup_deferred(winner, action)
        return winner

    def batched_expand_and_simulate(self, action):
        task = Rollout(
            sim=self.env,
            action_history=[action]
        )
        new_env, winner = yield task

        child = SearchNode(new_env)
        self.children[action] = child
        yield from child.backup_deferred(winner)

        return winner
