import numpy as np
from mcts.MCTSAgent import MCTSAgent
import gym
from minihex import player, HexGame
import minihex
import tqdm
from multiprocessing import Pool, cpu_count
from utils import convert_state
from utils import generate_sample as generate_board
import random


def generate_sample(board_size=5):
    board, active_player = generate_board(board_size)

    sim = HexGame(active_player, board, active_player)
    agent = MCTSAgent(sim, depth=1000)
    action = agent.act(board, active_player)

    if active_player == player.BLACK:
        return convert_state(sim), (action, -1)
    else:
        return convert_state(sim), (-1, action)


def generate_dataset(num_examples, prefix=None, board_size=5):
    if prefix is None:
        prefix = ""
    else:
        prefix += "_"

    dataset = list()
    labels = list()
    with Pool(cpu_count() - 2) as workers:
        return_val = list(tqdm.tqdm(workers.imap(
                                        generate_sample,
                                        [board_size for idx in range(num_examples)],
                                        chunksize=1),
                                    total=num_examples))
    for example, label in return_val:
        dataset.append(example)
        labels.append(label)

    with open(f"{prefix}data.npy", "wb") as out_file:
        np.save(out_file, np.stack(dataset, axis=0))
    with open(f"{prefix}labels.npy", "wb") as out_file:
        np.save(out_file, np.stack(labels, axis=0))


if __name__ == "__main__":
    generate_dataset(500000, prefix="training", board_size=9)
    generate_dataset(10000, prefix="validation", board_size=9)
    generate_dataset(10000, prefix="test", board_size=9)
