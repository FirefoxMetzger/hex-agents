from mcts.SearchNode import SearchNode
from copy import deepcopy
import numpy as np
from minihex.HexGame import HexGame
from minihex import player
from Agent import Agent


class MCTSAgent(Agent):
    def __init__(self, env, depth=1000):
        super(MCTSAgent, self).__init__()
        self.root_node = SearchNode(env)
        self.depth = depth

    def add_leaf(self):
        history = [self.root_node]

        # select
        while True:
            node = history[-1]
            action = node.select()
            if action is None:
                return
            if action in node.children:
                history.append(node.children[action])
            else:
                break

        node = node.expand(action)
        history.append(node)

        reward = node.simulate()

        for node in reversed(history):
            node.backup(reward)

    def act(self, state, active_player, info):
        self.update_root_state(state, info)
        self.plan(self.depth)
        action = self.policy()
        return action

    def plan(self, num_simulations=100):
        for _ in range(num_simulations):
            self.add_leaf()

        return self.policy()

    def policy(self):
        qualities = self.quality()
        max_value = np.max(qualities)
        valid_actions = self.root_node.available_actions[qualities == max_value]
        return np.random.choice(valid_actions)

    def quality(self, is_greedy=True):
        qualities = list()
        for action in self.root_node.available_actions:
            qualities.append(self.root_node.action_value(action,
                                                         is_greedy=is_greedy))

        return np.array(qualities)

    def update_root_state(self, state, info):
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
