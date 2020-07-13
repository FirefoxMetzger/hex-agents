
from minihex.HexGame import HexEnv
from Agent import RandomAgent
from copy import deepcopy


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


def step_and_rollout(env, action_history):
    new_env = deepcopy(env)
    for action in action_history:
        new_env.make_move(action)

    winner = new_env.winner
    if winner is None:
        winner = simulate(new_env)

    return new_env, winner
