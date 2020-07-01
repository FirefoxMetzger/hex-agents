import numpy as np
from mcts.MCTSAgent import MCTSAgent
import gym
from minihex import player, HexGame
import minihex
import tqdm
from anthony_net.utils import convert_state
from anthony_net.NNAgent import NNAgent


def generate_sample(board_size=5):
    # generate a random board state
    num_white_stones = np.random.randint(board_size ** 2 // 2)
    if np.random.rand() > 0.5:
        num_black_stones = num_white_stones + 1
        active_player = player.WHITE
    else:
        num_black_stones = num_white_stones
        active_player = player.BLACK
    positions = np.random.rand(board_size, board_size)
    board_shape = (board_size, board_size)
    ny, nx = np.unravel_index(np.argsort(positions.flatten()), board_shape)
    white_y = ny[:num_white_stones]
    white_x = nx[:num_white_stones]
    black_y = ny[num_white_stones:num_white_stones+num_black_stones]
    black_x = nx[num_white_stones:num_white_stones+num_black_stones]
    board = np.zeros((3, board_size, board_size))
    board[2, ...] = 1
    board[player.WHITE, white_y, white_x] = 1
    board[2, white_y, white_x] = 0
    board[player.BLACK, black_y, black_x] = 1
    board[2, black_y, black_x] = 0

    # instantiate expert at the generated position and query
    # expert action
    sim = HexGame(active_player, board, active_player)
    return convert_state(sim), active_player


def generate_dataset(num_examples, prefix=None, batch_size=32):
    if prefix is None:
        prefix = ""
    else:
        prefix += "_"

    num_batches = num_examples // batch_size + 1
    agent = NNAgent('best_model.h5')

    dataset = list()
    active_player = list()
    labels = list()
    for batch in tqdm.tqdm(range(num_batches)):
        for sample_idx in range(batch_size):
            sample, player = generate_sample()
            dataset.append(sample)
            active_player.append(player)
    dataset = np.stack(dataset, axis=0)
    active_player = np.stack(active_player, axis=0)

    labels = agent.predict(dataset, active_player)

    with open(f"{prefix}data.npy", "wb") as out_file:
        np.save(out_file, dataset)
    with open(f"{prefix}labels.npy", "wb") as out_file:
        np.save(out_file, labels)


if __name__ == "__main__":
    generate_dataset(100000, prefix="training")
    # generate_dataset(3000, prefix="validation")
    # generate_dataset(3000, prefix="test")
