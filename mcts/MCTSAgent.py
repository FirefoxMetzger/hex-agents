from mcts.SearchNode import SearchNode
import numpy as np
from minihex.HexGame import HexGame
from minihex import player
from Agent import Agent
import random


class MCTSAgent(Agent):
    def __init__(self,
                 env=None,
                 depth=1000,
                 board_size=9,
                 active_player=player.BLACK, player=player.BLACK):
        super(MCTSAgent, self).__init__()
        if env is None:
            board = player.EMPTY * np.ones((board_size, board_size))
            self.root_node = SearchNode(HexGame(
                active_player,
                board,
                player
            ))
        else:
            self.root_node = SearchNode(env)
        self.depth = depth

    def act(self, state, active_player, info=None):
        self.update_root_state(state, info)
        for _ in range(self.depth):
            self.root_node.add_leaf()
        action = self.policy()
        return action

    def reset(self, env):
        self.root_node = SearchNode(env)

    def deferred_plan(self):
        if self.root_node.is_terminal:
            return

        for _ in range(self.depth):
            yield from self.root_node.add_leaf_deferred()

    def update_root_state_deferred(self, info):
        if info is None:
            return

        last_move = info["last_move_player"]
        last_opponent_move = info["last_move_opponent"]
        if last_move is not None:
            if last_move not in self.root_node.children:
                gen_fn = self.root_node.batched_expand_and_simulate
                yield from gen_fn(last_move)
            self.root_node = self.root_node.children[last_move]

        if last_opponent_move is not None:
            if last_opponent_move not in self.root_node.children:
                gen_fn = self.root_node.batched_expand_and_simulate
                yield from gen_fn(last_opponent_move)
            self.root_node = self.root_node.children[last_opponent_move]

    def act_greedy(self, state, active_player, info):
        action = self.policy()
        return action

    def policy(self):
        qualities = self.quality()
        max_value = np.max(qualities)
        action_fn = self.root_node.available_actions
        valid_actions = action_fn[qualities == max_value]
        idx = int(random.random() * len(valid_actions))
        return valid_actions[idx]

    def quality(self, is_greedy=True):
        qualities = list()
        for action in self.root_node.available_actions:
            qualities.append(self.root_node.action_value(action,
                                                         is_greedy=is_greedy))
        return np.array(qualities)

    def update_root_state(self, state, info):
        if info is None:
            return

        last_move = info["last_move_player"]
        last_opponent_move = info["last_move_opponent"]

        if last_move is not None:
            if last_move not in self.root_node.children:
                self.root_node.expand(last_move)
            self.root_node = self.root_node.children[last_move]

        if last_opponent_move is not None:
            if last_opponent_move not in self.root_node.children:
                self.root_node.expand(last_opponent_move)
            self.root_node = self.root_node.children[last_opponent_move]

    def belief_matrix(self):
        quality = self.quality()
        board_shape = self.root_node.env.board.shape
        positions = np.unravel_index(self.root_node.available_actions,
                                     self.root_node.env.board.shape)
        win_prop = -1 * np.ones(board_shape)
        win_prop[positions[0], positions[1]] = quality
        return win_prop[2:-2, 2:-2]

    def __str__(self):
        return f"MCTS({self.depth})"
