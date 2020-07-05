
from minihex.HexGame import HexEnv
from Agent import RandomAgent


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
