from minihex.HexGame import HexEnv, HexGame, player
from Agent import RandomAgent
from nmcts.NMCTSAgent import NMCTSAgent
from anthony_net.utils import convert_state


def simulate(env, board_size=5):
    agent = RandomAgent(board_size)
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


def dataset_converter(example):
    return convert_state(HexGame(player.BLACK, example, player.BLACK))
