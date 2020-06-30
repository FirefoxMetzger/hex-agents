import numpy as np
from mcts.MCTSAgent import MCTSAgent
import gym
from minihex import player, HexGame
import minihex
import tqdm
from enum import IntEnum
from copy import deepcopy
from multiprocessing import Pool, cpu_count


def convert_state(state):
    board = state.board
    board_size = board.shape[1]
    converted_state = np.zeros((6, board_size+4, board_size+4))

    converted_state[[0, 2], :2, :] = 1
    converted_state[[0, 3], -2:, :] = 1
    converted_state[[1, 4], :, :2] = 1
    converted_state[[1, 5], :, -2:] = 1

    converted_state[:2, 2:-2, 2:-2] = board[:2, ...]

    region_black = np.pad(state.regions[player.BLACK], 1)
    region_white = np.pad(state.regions[player.WHITE], 1)

    positions = np.where(region_black == region_black[1, 1])
    converted_state[2][positions] = 1

    positions = np.where(region_black == region_black[-2, -2])
    converted_state[3][positions] = 1

    positions = np.where(region_white == region_white[1, 1])
    converted_state[4][positions] = 1

    positions = np.where(region_white == region_white[-2, -2])
    converted_state[5][positions] = 1

    return np.moveaxis(converted_state, 0, -1)


def play_match(agent, opponent, sim=None):
    color = player.WHITE if np.random.rand() > 0.5 else player.BLACK
    env = gym.make("hex-v0",
                   opponent_policy=opponent.act,
                   board_size=5,
                   player_color=color)
    state, info = env.reset()

    history = list()
    done = False
    while not done:
        action = agent.act(state[0], state[1], info)
        history.append((deepcopy(env), action))
        state, reward, done, info = env.step(action)

    random_element = np.random.randint(len(history))
    state, action = history[random_element]
    if color == player.BLACK:
        return convert_state(state), (action, -1)
    else:
        return convert_state(state), (-1, action)


def generate_sample(idx):
    # generate a random board state
    num_white_stones = np.random.randint(5 ** 2 // 2)
    if np.random.rand() > 0.5:
        num_black_stones = num_white_stones + 1
        active_player = player.WHITE
    else:
        num_black_stones = num_white_stones
        active_player = player.BLACK
    positions = np.random.rand(5, 5)
    ny, nx = np.unravel_index(np.argsort(positions.flatten()), (5, 5))
    white_y = ny[:num_white_stones]
    white_x = nx[:num_white_stones]
    black_y = ny[num_white_stones:num_white_stones+num_black_stones]
    black_x = nx[num_white_stones:num_white_stones+num_black_stones]
    board = np.zeros((3, 5, 5))
    board[2, ...] = 1
    board[player.WHITE, white_y, white_x] = 1
    board[2, white_y, white_x] = 0
    board[player.BLACK, black_y, black_x] = 1
    board[2, black_y, black_x] = 0

    # instantiate expert at the generated position and query
    # expert action
    sim = HexGame(active_player, board, active_player)
    agent = MCTSAgent(sim, depth=1000)

    info = {
        'state': board,
        'last_move_opponent': None,
        'last_move_player': None
    }
    action = agent.act(board, active_player, info)
    if active_player == player.BLACK:
        return convert_state(sim), (action, -1)
    else:
        return convert_state(sim), (-1, action)


def generate_dataset(num_examples, prefix=None):
    if prefix is None:
        prefix = ""
    else:
        prefix += "_"

    env = gym.make("hex-v0", opponent_policy=None, board_size=5)
    state = env.reset()
    hexgame = env.simulator

    dataset = list()
    labels = list()
    # for _ in range(12):
    #     return_val = generate_sample(0)
    # import pdb; pdb.set_trace()
    with Pool(cpu_count() - 2) as workers:
        return_val = list(tqdm.tqdm(workers.imap(
                                        generate_sample,
                                        [hexgame for _ in range(num_examples)],
                                        chunksize=512),
                                    total=num_examples))
    for example, label in return_val:
        dataset.append(example)
        labels.append(label)

    with open(f"{prefix}data.npy", "wb") as out_file:
        np.save(out_file, np.stack(dataset, axis=0))
    with open(f"{prefix}labels.npy", "wb") as out_file:
        np.save(out_file, np.stack(labels, axis=0))


if __name__ == "__main__":
    generate_dataset(100000, prefix="training")
    generate_dataset(3000, prefix="validation")
    generate_dataset(3000, prefix="test")
