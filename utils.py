from minihex.HexGame import HexEnv, HexGame, player
from Agent import RandomAgent
from nmcts.NMCTSAgent import NMCTSAgent
from copy import deepcopy
import numpy as np


def simulate(env, board_size=5):
    agent = RandomAgent(env.board_size)
    env = HexEnv(
        opponent_policy=agent.act,
        player_color=env.active_player,
        active_player=env.active_player,
        board=env.board.copy(),
        regions=env.regions.copy())

    state, info = env.reset()
    done = False
    while not done:
        action = agent.act(state[0], state[1], info)
        state, reward, done, info = env.step(action)

    return env.simulator.winner


def nmcts_builder(args):
    depth, env, initial_policy = args
    return NMCTSAgent(depth=depth, env=env,
                      network_policy=initial_policy)


def step_and_rollout(env, action_history):
    new_env = deepcopy(env)
    for action in action_history:
        new_env.make_move(action)

    winner = new_env.winner
    if winner is None:
        winner = simulate(new_env)

    return new_env, winner


class _save_array(object):
    def __init__(self, func, filename):
        self.func = func
        self.filename = filename
        self.idx = 0

    def __call__(self, *args, **kwargs):
        result = self.func(*args, **kwargs)
        # hax ahead
        if not isinstance(result, tuple):
            _result = [result]
        else:
            _result = result
        np.savez(self.filename.format(idx=self.idx), *_result)
        self.idx += 1
        return result


def save_array(filename):
    def decorator(func):
        return _save_array(func, filename)
    return decorator
