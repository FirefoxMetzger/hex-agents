from mcts.SearchNode import SearchNode
from copy import deepcopy
import numpy as np
from minihex.HexGame import HexGame
from minihex import player


class MCTSAgent(object):
    def __init__(self, env):
        self.root_node = SearchNode(env)

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

    def update_root_state(self, state):
        active_player = self.root_node.env.active_player
        player = self.root_node.env.player
        env = HexGame(active_player, state, player)
        self.root_node = SearchNode(env)


if __name__ == "__main__":
    import gym
    import minihex

    board_size = 5
    board = np.zeros((3, board_size+4, board_size+4))
    board[player.BLACK, :2, :] = 1
    board[player.BLACK, -2:, :] = 1
    board[player.WHITE, :, :2] = 1
    board[player.WHITE, :, -2:] = 1
    board[2, 2:-2, 2:-2] = 1

    env = HexGame(player.BLACK, board, player.BLACK)
    agent = MCTSAgent(env)
    print(agent.plan())
    print(agent.quality())
    import pdb; pdb.set_trace()
