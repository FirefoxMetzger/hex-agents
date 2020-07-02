from mcts.SearchNode import SearchNode
from copy import deepcopy
from anthony_net.utils import convert_state
from anthony_net.NNAgent import NNAgent
import random

WEIGHT_a = 100


class NeuralSearchNode(SearchNode):
    def __init__(self, env, agent=None):
        super(NeuralSearchNode, self).__init__(env)
        if agent is None:
            self.agent = NNAgent("best_model.h5")
        else:
            self.agent = agent
        self.network_policy = self.agent.predict_env(self.env)

    def expand(self, action):
        new_env = deepcopy(self.env)
        new_env.make_move(action)
        child = NeuralSearchNode(new_env, agent=self.agent)
        self.children[action] = child

        return child

    def update_action_value(self, action):
        super(NeuralSearchNode, self).update_action_value(action)

        child = self.children[action]
        child_sims = child.total_simulations
        predicted_q = self.network_policy[action]
        self.Q[action] += (WEIGHT_a * predicted_q / (child_sims + 1))
        self.greedy_Q[action] += WEIGHT_a * predicted_q / (child_sims + 1)
