from mcts.SearchNode import SearchNode
from copy import deepcopy
from anthony_net.utils import convert_state
from anthony_net.NNAgent import NNAgent
from minihex.HexGame import HexEnv
import random

from scheduler.tasks import NNEval, Rollout

WEIGHT_a = 100


class NeuralSearchNode(SearchNode):
    def __init__(self, env, agent=None, model_file=None, network_policy=None):
        super(NeuralSearchNode, self).__init__(env)
        if agent is None and isinstance(model_file, str):
            self.network_agent = NNAgent(model_file)
        else:
            self.network_agent = agent

        self.network_policy = None
        if model_file is not None:
            self.network_policy = self.network_agent.predict_env(env)
        if network_policy is not None:
            self.network_policy = network_policy

    def expand(self, action):
        new_env = deepcopy(self.env)
        new_env.make_move(action)
        child = NeuralSearchNode(new_env, agent=self.network_agent)
        self.children[action] = child

        return child

    def update_action_value(self, action):
        super(NeuralSearchNode, self).update_action_value(action)

        child = self.children[action]
        child_sims = child.total_simulations
        predicted_q = self.network_policy[action]
        self.Q[action] += WEIGHT_a * predicted_q / (child_sims + 1)
        self.greedy_Q[action] += WEIGHT_a * predicted_q / (child_sims + 1)

    def update_action_value_deferred(self, action):
        if self.network_policy is None:
            policy = yield NNEval(sim=self.env)
            self.network_policy = policy.tolist()    

        self.update_action_value(action)

    def batched_expand_and_simulate(self, action):
        task = Rollout(
            sim=self.env,
            action_history=[action]
        )
        new_env, winner = yield task

        child = NeuralSearchNode(new_env)
        self.children[action] = child
        child.backup(winner)

        return winner
